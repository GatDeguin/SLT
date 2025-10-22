#!/usr/bin/env python3
"""Command line entry-point to fine-tune the validated multi-stream SLT model.

The script wires together the reusable components from :mod:`slt` to
instantiate the dataset, encoder/decoder pair and the optimisation loop. Por
defecto intenta cargar los pesos validados del flujo ``single_signer`` cuando el
checkpoint descargado está disponible localmente; puedes desactivarlos mediante
``--pretrained none`` o indicar otra ruta con ``--pretrained-checkpoint``.
Tokenizer y decoder dependen de ``transformers``, por lo que el paquete debe
estar instalado junto con PyTorch.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from dataclasses import asdict, fields
from pathlib import Path
from collections.abc import Mapping
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PreTrainedTokenizerBase

from slt.data import LsaTMultiStream
from slt.training.configuration import (
    DataConfig,
    ModelConfig,
    OptimConfig,
    TrainingConfig,
    resolve_configs,
)
from slt.training.data import create_dataloader, normalise_mix_spec
from slt.training.loops import eval_epoch, multistream_loss, train_epoch
from slt.training.models import MultiStreamClassifier
from slt.training.optim import create_optimizer, create_scheduler
from slt.utils.cli import parse_range_pair, parse_translation_range
from slt.utils.general import set_seed
from slt.utils.text import create_tokenizer

try:  # pragma: no cover - numpy optional for RNG capture
    import numpy as np
except Exception:  # pragma: no cover - numpy optional
    np = None  # type: ignore

try:  # pragma: no cover - TensorBoard is an optional dependency.
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - if TensorBoard is not installed we noop.
    SummaryWriter = None  # type: ignore


def _resolve_device(flag: str | torch.device) -> torch.device:
    if isinstance(flag, torch.device):
        device = flag
    else:
        value = str(flag).strip().lower()
        if value == "auto":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            device = torch.device(flag)
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA solicitada pero no disponible. Se usará CPU.")
        return torch.device("cpu")
    return device
def _levenshtein_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, char_a in enumerate(a, start=1):
        current = [i]
        for j, char_b in enumerate(b, start=1):
            cost = 0 if char_a == char_b else 1
            current.append(
                min(
                    current[j - 1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def _build_perplexity_metric() -> Callable[[Any, torch.Tensor], float]:
    def _metric(outputs: Any, targets: torch.Tensor) -> float:
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        vocab = logits.size(-1)
        logits_flat = logits.view(-1, vocab)
        targets_flat = targets.view(-1)
        mask = targets_flat.ne(-100)
        token_count = int(mask.sum().item())
        if token_count == 0:
            return float("nan")
        log_probs = logits_flat.log_softmax(dim=-1)
        masked_log_probs = log_probs[mask]
        masked_targets = targets_flat[mask].long()
        nll = F.nll_loss(masked_log_probs, masked_targets, reduction="sum")
        perplexity = torch.exp(nll / token_count)
        return float(perplexity.detach().cpu().item())

    return _metric


def _build_cer_metric(tokenizer: PreTrainedTokenizerBase) -> Callable[[Any, torch.Tensor], float]:
    def _metric(outputs: Any, targets: torch.Tensor) -> float:
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        predictions = logits.argmax(dim=-1)
        targets_cpu = targets.detach().cpu()
        preds_cpu = predictions.detach().cpu()
        mask_cpu = targets_cpu.ne(-100)
        total_distance = 0
        total_chars = 0

        for pred_row, target_row, mask_row in zip(preds_cpu, targets_cpu, mask_cpu):
            target_tokens = target_row[mask_row]
            if target_tokens.numel() == 0:
                continue
            pred_tokens = pred_row[: target_tokens.numel()]
            target_tokens = target_tokens.tolist()
            pred_tokens = pred_tokens.tolist()
            target_text = tokenizer.decode(target_tokens, skip_special_tokens=True)
            pred_text = tokenizer.decode(pred_tokens, skip_special_tokens=True)
            if not target_text and not pred_text:
                continue
            distance = _levenshtein_distance(target_text, pred_text)
            total_distance += distance
            total_chars += max(len(target_text), 1)

        if total_chars == 0:
            return 0.0
        return total_distance / total_chars

    return _metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--config", type=Path, help="YAML or JSON configuration template")
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override configuration values using dotted keys, e.g. data.batch_size=16",
    )

    # Data arguments
    parser.add_argument("--face-dir", type=Path, help="Directory with cropped face frames")
    parser.add_argument("--hand-left-dir", type=Path, help="Directory with left hand frames")
    parser.add_argument("--hand-right-dir", type=Path, help="Directory with right hand frames")
    parser.add_argument("--pose-dir", type=Path, help="Directory with pose .npz files")
    parser.add_argument("--keypoints-dir", type=Path, help="Directory with MediaPipe keypoints")
    parser.add_argument("--metadata-csv", type=Path, help="CSV file with video_id/text pairs")
    parser.add_argument("--train-index", type=Path, help="CSV file listing training video IDs")
    parser.add_argument("--val-index", type=Path, help="CSV file listing validation video IDs")
    parser.add_argument("--work-dir", type=Path, help="Directory where checkpoints/logs will be saved")
    parser.add_argument(
        "--gloss-csv",
        type=Path,
        help="Optional CSV with columns video_id;gloss;ctc_labels",
    )

    parser.add_argument("--num-workers", type=int, help="Number of DataLoader worker processes")
    parser.add_argument("--batch-size", type=int, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, help="Validation batch size (defaults to training size)")
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--device", type=str, help="Torch device identifier (e.g. cuda, cuda:0, cpu)")
    parser.add_argument(
        "--precision",
        choices=["fp32", "amp"],
        help="Numerical precision. 'amp' enables automatic mixed precision on CUDA",
    )
    parser.add_argument(
        "--tensorboard",
        type=Path,
        help="Optional TensorBoard log directory. When omitted TensorBoard logging is disabled.",
    )
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pinned memory in the data loaders")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer identifier or local path")
    parser.add_argument("--max-target-length", type=int, help="Maximum length of the tokenised target sequences")
    parser.add_argument(
        "--mix-stream",
        dest="mix_streams",
        action="append",
        default=[],
        metavar="STREAM[:P]",
        help=(
            "Optionally permute individual streams across the batch with probability P."
            " STREAM can be face, hand-left, hand-right or pose."
        ),
    )
    parser.add_argument(
        "--keypoint-normalize-center",
        dest="keypoint_normalize_center",
        action="store_true",
        help="Normalise keypoints around the image centre before augmentations.",
    )
    parser.add_argument(
        "--no-keypoint-normalize-center",
        dest="keypoint_normalize_center",
        action="store_false",
        help="Disable centre normalisation prior to keypoint augmentations.",
    )
    parser.set_defaults(keypoint_normalize_center=None)
    parser.add_argument(
        "--keypoint-scale-range",
        type=str,
        help="Uniform scale factor range applied to keypoints (e.g. 0.9,1.1).",
    )
    parser.add_argument(
        "--keypoint-translate-range",
        type=str,
        help="Translation offsets (1, 2 or 4 floats) applied after scaling/rotation.",
    )
    parser.add_argument(
        "--keypoint-rotate-range",
        type=str,
        help="Rotation range in degrees around the frame centre (e.g. -10,10).",
    )
    parser.add_argument(
        "--keypoint-resample-range",
        type=str,
        help="Temporal resampling ratio range applied before frame selection.",
    )

    # Model arguments
    parser.add_argument("--image-size", type=int, help="Input image resolution expected by the ViT backbones")
    parser.add_argument("--projector-dim", type=int, help="Dimensionality of the per-stream projectors")
    parser.add_argument("--d-model", type=int, help="Temporal encoder embedding dimension")
    parser.add_argument("--pose-landmarks", type=int, help="Number of pose landmarks in the NPZ files")
    parser.add_argument("--sequence-length", type=int, help="Temporal sequence length used during sampling")
    parser.add_argument("--projector-dropout", type=float, help="Dropout applied inside the projectors")
    parser.add_argument("--fusion-dropout", type=float, help="Dropout applied before stream fusion")
    parser.add_argument("--temporal-nhead", type=int, help="Number of attention heads in the temporal encoder")
    parser.add_argument("--temporal-layers", type=int, help="Number of transformer layers in the temporal encoder")
    parser.add_argument("--temporal-dim-feedforward", type=int, help="Feed-forward dimension inside the temporal encoder")
    parser.add_argument("--temporal-dropout", type=float, help="Dropout used by the temporal encoder")
    parser.add_argument("--decoder-layers", type=int, help="Number of layers in the seq2seq decoder")
    parser.add_argument("--decoder-heads", type=int, help="Number of attention heads in the seq2seq decoder")
    parser.add_argument("--decoder-dropout", type=float, help="Dropout probability inside the seq2seq decoder")
    parser.add_argument("--face-backbone", type=str, help="Backbone specification for the face stream")
    parser.add_argument(
        "--hand-left-backbone",
        type=str,
        help="Backbone specification for the left hand stream",
    )
    parser.add_argument(
        "--hand-right-backbone",
        type=str,
        help="Backbone specification for the right hand stream",
    )
    parser.add_argument(
        "--freeze-face-backbone",
        action="store_true",
        help="Freeze the face backbone",
    )
    parser.add_argument(
        "--freeze-hand-left-backbone",
        action="store_true",
        help="Freeze the left hand backbone",
    )
    parser.add_argument(
        "--freeze-hand-right-backbone",
        action="store_true",
        help="Freeze the right hand backbone",
    )
    parser.add_argument(
        "--pretrained",
        type=str,
        help="Pretrained encoder/decoder weights to load (single_signer or none)",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        help=(
            "Path to the downloaded single_signer checkpoint used when --pretrained"
            " is enabled"
        ),
    )
    parser.add_argument("--decoder-model", type=str, help="Pretrained decoder model name or path")
    parser.add_argument("--decoder-config", type=str, help="Decoder configuration name or path")
    parser.add_argument(
        "--decoder-class",
        type=str,
        help="Python path to a custom decoder class (module:ClassName or module.ClassName)",
    )
    parser.add_argument(
        "--decoder-kwargs",
        type=str,
        help="JSON object with keyword arguments forwarded to the decoder constructor",
    )
    parser.add_argument(
        "--use-mska",
        dest="use_mska",
        action="store_true",
        help="Enable the MSKA keypoint encoder branch",
    )
    parser.add_argument(
        "--no-mska",
        dest="use_mska",
        action="store_false",
        help="Disable the MSKA keypoint encoder branch",
    )
    parser.set_defaults(use_mska=None)
    parser.add_argument("--mska-heads", type=int, help="Number of attention heads used by MSKA")
    parser.add_argument(
        "--mska-ff-multiplier",
        type=int,
        help="Feed-forward multiplier applied inside the MSKA transformer",
    )
    parser.add_argument(
        "--mska-dropout",
        type=float,
        help="Dropout probability applied by MSKA encoders and heads",
    )
    parser.add_argument(
        "--mska-stream-heads",
        type=int,
        help="Number of attention heads inside each keypoint stream encoder",
    )
    parser.add_argument(
        "--mska-temporal-blocks",
        type=int,
        help="Number of temporal convolutional blocks applied per stream",
    )
    parser.add_argument(
        "--mska-temporal-kernel",
        type=int,
        help="Temporal kernel size used by the stream convolutional blocks",
    )
    parser.add_argument(
        "--mska-temporal-dilation",
        type=int,
        help="Temporal dilation factor applied by the stream convolutional blocks",
    )
    parser.add_argument(
        "--mska-input-dim",
        type=int,
        help="Dimensionality of the keypoint vectors provided to MSKA",
    )
    parser.add_argument(
        "--mska-ctc-vocab",
        type=int,
        help="Vocabulary size used by the MSKA CTC classification heads",
    )
    parser.add_argument(
        "--mska-detach-teacher",
        dest="mska_detach_teacher",
        action="store_true",
        help="Detach fused logits before distillation (teacher without gradients)",
    )
    parser.add_argument(
        "--mska-attach-teacher",
        dest="mska_detach_teacher",
        action="store_false",
        help="Propagate gradients through the fused logits during distillation",
    )
    parser.set_defaults(mska_detach_teacher=None)
    parser.add_argument(
        "--mska-gloss-hidden-dim",
        dest="mska_gloss_hidden_dim",
        type=int,
        help="Hidden dimension of the gloss MLP applied to MSKA fused embeddings",
    )
    parser.add_argument(
        "--mska-gloss-activation",
        dest="mska_gloss_activation",
        choices=("relu", "gelu", "silu", "tanh"),
        help="Activation function inserted between the gloss MLP layers",
    )
    parser.add_argument(
        "--mska-gloss-dropout",
        dest="mska_gloss_dropout",
        type=float,
        help="Dropout probability applied inside the gloss MLP",
    )
    parser.add_argument(
        "--mska-gloss-fusion",
        dest="mska_gloss_fusion",
        choices=("add", "concat", "none"),
        help="Strategy to expose the gloss sequence to the decoder",
    )
    parser.add_argument(
        "--mska-translation-weight",
        type=float,
        help="Weight applied to the translation cross-entropy term",
    )
    parser.add_argument(
        "--mska-ctc-weight",
        type=float,
        help="Weight applied to the summed MSKA CTC losses",
    )
    parser.add_argument(
        "--mska-distillation-weight",
        type=float,
        help="Weight applied to the MSKA distillation loss",
    )
    parser.add_argument(
        "--mska-distillation-temperature",
        type=float,
        help="Temperature used to soften logits during distillation",
    )

    # Optimiser and loop arguments
    parser.add_argument("--epochs", type=int, help="Number of training epochs")
    parser.add_argument("--optimizer", type=str, help="Optimizer type (adamw, adam, sgd)")
    parser.add_argument("--lr", type=float, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, help="L2 weight decay")
    parser.add_argument(
        "--scheduler",
        choices=["none", "steplr", "cosine"],
        help="Optional learning rate scheduler",
    )
    parser.add_argument("--scheduler-step-size", type=int, help="Step size for StepLR scheduler")
    parser.add_argument("--scheduler-gamma", type=float, help="Gamma for StepLR scheduler")
    parser.add_argument("--scheduler-tmax", type=int, help="T_max parameter for CosineAnnealingLR")
    parser.add_argument("--label-smoothing", type=float, help="Label smoothing applied to the loss")
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        help="Optional maximum norm for gradient clipping before each optimisation step.",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--compile-mode",
        type=str,
        help="torch.compile mode (default, reduce-overhead, max-autotune)",
    )
    parser.add_argument(
        "--compile",
        dest="compile_flag",
        action="store_true",
        help="Enable torch.compile",
    )
    parser.add_argument(
        "--no-compile",
        dest="compile_flag",
        action="store_false",
        help="Disable torch.compile",
    )
    parser.set_defaults(compile_flag=None)

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help=(
            "Checkpoint path to resume training from (loads model, optimiser "
            "and AMP scaler state)."
        ),
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Initial checkpoint to warm-start the model weights before training",
    )

    args = parser.parse_args()
    if args.decoder_config and args.decoder_model:
        parser.error("--decoder-model and --decoder-config are mutually exclusive")
    if args.decoder_class and (args.decoder_model or args.decoder_config):
        parser.error("--decoder-class cannot be combined with --decoder-model/--decoder-config")
    if args.decoder_kwargs:
        try:
            parsed_kwargs = json.loads(args.decoder_kwargs)
        except json.JSONDecodeError as exc:
            parser.error(f"Unable to parse --decoder-kwargs as JSON: {exc}")
        if not isinstance(parsed_kwargs, dict):
            parser.error("--decoder-kwargs must encode a JSON object")
        args.decoder_kwargs = parsed_kwargs
    else:
        args.decoder_kwargs = None
    if args.resume and args.init_checkpoint:
        parser.error("--resume and --init-checkpoint are mutually exclusive")
    if args.mix_streams:
        mix_spec: dict[str, float] = {}
        for entry in args.mix_streams:
            name, _, prob_text = entry.partition(":")
            name = name.strip()
            prob = 1.0
            if prob_text:
                try:
                    prob = float(prob_text)
                except ValueError:
                    parser.error(f"Invalid probability for --mix-stream '{entry}'")
            mix_spec[name] = prob
        try:
            args.mix_streams = normalise_mix_spec(mix_spec)
        except ValueError as exc:
            parser.error(str(exc))
    else:
        args.mix_streams = None
    if args.precision == "float32":
        args.precision = "fp32"
    if getattr(args, "keypoint_scale_range", None) is not None:
        raw = args.keypoint_scale_range.strip()
        if raw.lower() in {"none", "off"}:
            args.keypoint_scale_range = None
        else:
            try:
                args.keypoint_scale_range = parse_range_pair(
                    raw,
                    positive=True,
                    symmetric_single=False,
                )
            except ValueError as exc:
                parser.error(f"--keypoint-scale-range: {exc}")
    if getattr(args, "keypoint_translate_range", None) is not None:
        raw = args.keypoint_translate_range.strip()
        if raw.lower() in {"none", "off"}:
            args.keypoint_translate_range = None
        else:
            try:
                args.keypoint_translate_range = parse_translation_range(raw)
            except ValueError as exc:
                parser.error(f"--keypoint-translate-range: {exc}")
    if getattr(args, "keypoint_rotate_range", None) is not None:
        raw = args.keypoint_rotate_range.strip()
        if raw.lower() in {"none", "off"}:
            args.keypoint_rotate_range = None
        else:
            try:
                args.keypoint_rotate_range = parse_range_pair(
                    raw,
                    positive=False,
                    symmetric_single=True,
                )
            except ValueError as exc:
                parser.error(f"--keypoint-rotate-range: {exc}")
    if getattr(args, "keypoint_resample_range", None) is not None:
        raw = args.keypoint_resample_range.strip()
        if raw.lower() in {"none", "off"}:
            args.keypoint_resample_range = None
        else:
            try:
                args.keypoint_resample_range = parse_range_pair(
                    raw,
                    positive=True,
                    symmetric_single=False,
                )
            except ValueError as exc:
                parser.error(f"--keypoint-resample-range: {exc}")
    if args.clip_grad_norm is not None and args.clip_grad_norm <= 0:
        logging.warning("Ignoring non-positive --clip-grad-norm value: %s", args.clip_grad_norm)
        args.clip_grad_norm = None
    explicit_bool_flags = set()
    for name in ("use_mska", "mska_detach_teacher", "keypoint_normalize_center"):
        if getattr(args, name, None) is not None:
            explicit_bool_flags.add(name)
    args._explicit_bool_flags = explicit_bool_flags
    return args


def _collect_cli_overrides(args: argparse.Namespace) -> dict[str, dict[str, Any]]:
    sections = {
        "data": DataConfig,
        "model": ModelConfig,
        "optim": OptimConfig,
        "training": TrainingConfig,
    }
    overrides: dict[str, dict[str, Any]] = {key: {} for key in sections}

    for section, cls in sections.items():
        for field_obj in fields(cls):
            attr_name = field_obj.name
            if not hasattr(args, attr_name):
                continue
            value = getattr(args, attr_name)
            if isinstance(value, bool):
                if value or attr_name in getattr(args, "_explicit_bool_flags", set()):
                    overrides[section][attr_name] = value
            elif value is not None:
                overrides[section][attr_name] = value

    if getattr(args, "clip_grad_norm", None) is not None:
        overrides["optim"]["grad_clip_norm"] = args.clip_grad_norm
    if getattr(args, "scheduler", None) is not None:
        overrides["optim"]["scheduler"] = args.scheduler
    if getattr(args, "compile_flag", None) is not None:
        overrides["training"]["compile"] = bool(args.compile_flag)
    if getattr(args, "compile_mode", None) is not None:
        overrides["training"]["compile_mode"] = args.compile_mode
    if getattr(args, "grad_accum_steps", None) is not None:
        overrides["training"]["grad_accum_steps"] = args.grad_accum_steps
    if getattr(args, "tensorboard", None) is not None:
        overrides["data"]["tensorboard"] = args.tensorboard
    if getattr(args, "no_pin_memory", False):
        overrides["data"]["pin_memory"] = False
    if getattr(args, "scheduler_tmax", None) is not None:
        overrides["optim"]["scheduler_tmax"] = args.scheduler_tmax

    return overrides


def build_configs(
    args: argparse.Namespace,
) -> tuple[DataConfig, ModelConfig, OptimConfig, TrainingConfig, dict[str, Any]]:
    cli_overrides = _collect_cli_overrides(args)
    base: dict[str, Any] = {}
    return resolve_configs(
        config_path=args.config,
        cli_overrides=cli_overrides,
        set_overrides=args.overrides,
        base=base,
    )


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _ensure_exists(path: Path, *, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def _validate_paths(config: DataConfig) -> None:
    _ensure_exists(config.face_dir, kind="Face directory")
    _ensure_exists(config.hand_left_dir, kind="Left hand directory")
    _ensure_exists(config.hand_right_dir, kind="Right hand directory")
    _ensure_exists(config.pose_dir, kind="Pose directory")
    if config.keypoints_dir:
        _ensure_exists(config.keypoints_dir, kind="Keypoints directory")
    _ensure_exists(config.metadata_csv, kind="Metadata CSV")
    _ensure_exists(config.train_index, kind="Train index CSV")
    _ensure_exists(config.val_index, kind="Validation index CSV")
    if config.gloss_csv:
        _ensure_exists(config.gloss_csv, kind="Gloss CSV")
    config.work_dir.mkdir(parents=True, exist_ok=True)
def _maybe_compile_model(model: nn.Module, training: TrainingConfig) -> nn.Module:
    if not training.compile:
        return model
    compile_fn = getattr(torch, "compile", None)
    if compile_fn is None:  # pragma: no cover - torch.compile may be unavailable
        logging.warning("torch.compile requested but not available in this PyTorch build.")
        return model
    compile_kwargs = {}
    if training.compile_mode:
        compile_kwargs["mode"] = training.compile_mode
    try:
        compiled = compile_fn(model, **compile_kwargs)
        logging.info(
            "Model compiled with torch.compile (mode=%s)",
            training.compile_mode or "default",
        )
        return compiled
    except Exception:
        logging.exception("torch.compile failed; continuing with eager execution.")
        return model


def _serialise_config(
    work_dir: Path,
    data: DataConfig,
    model: ModelConfig,
    optim: OptimConfig,
    training: TrainingConfig,
    merged: Mapping[str, Any],
) -> None:
    resolved = {
        "data": asdict(data),
        "model": asdict(model),
        "optim": asdict(optim),
        "training": asdict(training),
    }
    resolved_payload = json.loads(json.dumps(resolved, default=str))
    (work_dir / "config.json").write_text(
        json.dumps(resolved_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    merged_payload = json.loads(json.dumps(dict(merged), default=str))
    (work_dir / "config.merged.json").write_text(
        json.dumps(merged_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):  # pragma: no cover - defensive
        return None


def _convert_metrics(metrics: Mapping[str, Any] | None) -> dict[str, float | None]:
    converted: dict[str, float | None] = {}
    if not metrics:
        return converted
    for name, value in metrics.items():
        converted[name] = _safe_float(value)
    return converted


def _append_metrics(path: Path, record: Mapping[str, Any]) -> None:
    payload = json.loads(json.dumps(record, default=str))
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _capture_rng_state() -> dict[str, Any]:
    state: dict[str, Any] = {
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
    }
    if np is not None:
        try:  # pragma: no cover - numpy may be unavailable
            state["numpy"] = np.random.get_state()
        except Exception:
            logging.debug("Unable to capture NumPy RNG state", exc_info=True)
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    return state


def _save_rng_state(work_dir: Path) -> None:
    work_dir.mkdir(parents=True, exist_ok=True)
    state = _capture_rng_state()
    torch.save(state, work_dir / "rng_state.pt")


def _load_rng_state(work_dir: Path) -> None:
    path = work_dir / "rng_state.pt"
    if not path.exists():
        return
    state = torch.load(path, map_location="cpu", weights_only=False)
    python_state = state.get("python")
    if python_state is not None:
        random.setstate(python_state)
    torch_state = state.get("torch")
    if torch_state is not None:
        torch.set_rng_state(torch_state)
    numpy_state = state.get("numpy")
    if numpy_state is not None and np is not None:
        np.random.set_state(numpy_state)
    cuda_state = state.get("cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _load_initial_checkpoint(model: nn.Module, path: Path, *, device: torch.device) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Initial checkpoint not found: {path}")
    logging.info("Loading initial weights from %s", path)
    payload = torch.load(path, map_location=device, weights_only=False)
    if isinstance(payload, Mapping) and "model_state" in payload:
        state_dict = payload["model_state"]
    else:
        state_dict = payload
    if not isinstance(state_dict, Mapping):
        raise ValueError("Initial checkpoint must be a state dict or contain a 'model_state' key")
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        logging.warning("Missing keys when loading init checkpoint: %s", missing)
    if unexpected:
        logging.warning("Unexpected keys when loading init checkpoint: %s", unexpected)


def _unwrap_model(module: nn.Module) -> nn.Module:
    for attr in ("_orig_mod", "module", "_module"):
        inner = getattr(module, attr, None)
        if isinstance(inner, nn.Module):
            return inner
    return module


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    scaler: torch.cuda.amp.GradScaler | None = None,
    best_val: float | None = None,
    config: Mapping[str, Any] | None = None,
    scheduler: Any | None = None,
) -> None:
    base_model = _unwrap_model(model)

    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": float(val_loss),
    }
    if hasattr(base_model, "encoder") and isinstance(base_model.encoder, nn.Module):
        state["encoder_state"] = base_model.encoder.state_dict()
        mska_encoder = getattr(base_model.encoder, "mska_encoder", None)
        if isinstance(mska_encoder, nn.Module):
            state["mska_state"] = mska_encoder.state_dict()
    if hasattr(base_model, "decoder") and isinstance(base_model.decoder, nn.Module):
        state["decoder_state"] = base_model.decoder.state_dict()
    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()
    if best_val is not None:
        state["best_val"] = float(best_val)
    if config is not None:
        state["config"] = json.loads(json.dumps(dict(config), default=str))
    if scheduler is not None:
        state["scheduler_state"] = scheduler.state_dict()
    torch.save(state, path)


def main() -> None:
    args = parse_args()
    (
        data_config,
        model_config,
        optim_config,
        training_config,
        merged_config,
    ) = build_configs(args)

    if training_config.grad_accum_steps < 1:
        raise ValueError("grad_accum_steps must be a positive integer")

    setup_logging()
    _validate_paths(data_config)

    set_seed(data_config.seed)

    if args.resume is not None:
        _load_rng_state(data_config.work_dir)
    else:
        _save_rng_state(data_config.work_dir)

    if data_config.precision.lower() == "float32":
        data_config.precision = "fp32"

    device = _resolve_device(data_config.device)

    tokenizer_source = data_config.tokenizer or model_config.decoder_model
    if tokenizer_source is None:
        raise ValueError("Tokenizer source could not be resolved.")
    tokenizer = create_tokenizer(tokenizer_source)

    train_dataset = LsaTMultiStream(
        face_dir=str(data_config.face_dir),
        hand_l_dir=str(data_config.hand_left_dir),
        hand_r_dir=str(data_config.hand_right_dir),
        pose_dir=str(data_config.pose_dir),
        csv_path=str(data_config.metadata_csv),
        index_csv=str(data_config.train_index),
        keypoints_dir=str(data_config.keypoints_dir)
        if data_config.keypoints_dir
        else None,
        gloss_csv=str(data_config.gloss_csv) if data_config.gloss_csv else None,
        T=model_config.sequence_length,
        img_size=model_config.image_size,
        lkp_count=model_config.pose_landmarks,
        keypoint_normalize_center=data_config.keypoint_normalize_center,
        keypoint_scale_range=data_config.keypoint_scale_range,
        keypoint_translate_range=data_config.keypoint_translate_range,
        keypoint_rotate_range=data_config.keypoint_rotate_range,
        keypoint_resample_range=data_config.keypoint_resample_range,
    )
    val_dataset = LsaTMultiStream(
        face_dir=str(data_config.face_dir),
        hand_l_dir=str(data_config.hand_left_dir),
        hand_r_dir=str(data_config.hand_right_dir),
        pose_dir=str(data_config.pose_dir),
        csv_path=str(data_config.metadata_csv),
        index_csv=str(data_config.val_index),
        keypoints_dir=str(data_config.keypoints_dir)
        if data_config.keypoints_dir
        else None,
        gloss_csv=str(data_config.gloss_csv) if data_config.gloss_csv else None,
        T=model_config.sequence_length,
        img_size=model_config.image_size,
        lkp_count=model_config.pose_landmarks,
        keypoint_normalize_center=data_config.keypoint_normalize_center,
        keypoint_scale_range=data_config.keypoint_scale_range,
        keypoint_translate_range=data_config.keypoint_translate_range,
        keypoint_rotate_range=data_config.keypoint_rotate_range,
        keypoint_resample_range=data_config.keypoint_resample_range,
    )

    val_batch_size = data_config.val_batch_size or data_config.batch_size
    try:
        train_mix = normalise_mix_spec(data_config.mix_streams or {})
    except ValueError as exc:
        raise ValueError(f"Invalid mix_streams configuration: {exc}") from exc
    data_config.mix_streams = train_mix

    train_loader = create_dataloader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        tokenizer=tokenizer,
        max_length=data_config.max_target_length,
        mix_streams=train_mix,
        seed=data_config.seed,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        tokenizer=tokenizer,
        max_length=data_config.max_target_length,
        mix_streams=None,
        seed=data_config.seed,
    )

    model = MultiStreamClassifier(model_config, tokenizer).to(device)
    if training_config.init_checkpoint is not None:
        _load_initial_checkpoint(model, training_config.init_checkpoint, device=device)
    model = _maybe_compile_model(model, training_config)
    optimizer = create_optimizer(
        model.parameters(),
        {
            "type": optim_config.optimizer,
            "lr": optim_config.lr,
            "weight_decay": optim_config.weight_decay,
        },
    )

    scheduler = None
    if optim_config.scheduler:
        if optim_config.scheduler == "steplr":
            scheduler = create_scheduler(
                optimizer,
                {
                    "type": "steplr",
                    "step_size": optim_config.scheduler_step_size,
                    "gamma": optim_config.scheduler_gamma,
                },
            )
        elif optim_config.scheduler == "cosine":
            scheduler = create_scheduler(
                optimizer,
                {
                    "type": "cosine",
                    "t_max": optim_config.scheduler_step_size,
                },
            )

    def _loss_fn(
        outputs: Any,
        targets: Mapping[str, Any],
        _: Optional[Mapping[str, Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return multistream_loss(
            outputs,
            targets,
            label_smoothing=optim_config.label_smoothing,
            translation_weight=model_config.mska_translation_weight,
            ctc_weight=model_config.mska_ctc_weight,
            distillation_weight=model_config.mska_distillation_weight,
            distillation_temperature=model_config.mska_distillation_temperature,
        )

    use_amp = (
        data_config.precision == "amp"
        and device.type == "cuda"
        and torch.cuda.is_available()
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    perplexity_metric = _build_perplexity_metric()
    cer_metric = _build_cer_metric(tokenizer)
    train_metrics = {"perplexity": perplexity_metric}
    eval_metrics = {"perplexity": perplexity_metric, "cer": cer_metric}

    writer = None
    if data_config.tensorboard is not None:
        if SummaryWriter is None:
            logging.warning(
                "TensorBoard requested but not available. Install tensorboard "
                "to enable logging."
            )
        else:
            data_config.tensorboard.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(data_config.tensorboard))

    _serialise_config(
        data_config.work_dir,
        data_config,
        model_config,
        optim_config,
        training_config,
        merged_config,
    )

    metrics_path = data_config.work_dir / "metrics.jsonl"
    if args.resume is None and metrics_path.exists():
        metrics_path.unlink()

    best_val = float("inf")
    best_path = data_config.work_dir / "best.pt"
    last_path = data_config.work_dir / "last.pt"
    start_epoch = 0

    if args.resume is not None:
        checkpoint_path = args.resume
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if scaler is not None and "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        if scheduler is not None and "scheduler_state" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state"])
        best_val = float(checkpoint.get("best_val", checkpoint.get("val_loss", best_val)))
        start_epoch = int(checkpoint.get("epoch", 0))
        logging.info("Resumed training from %s at epoch %d", checkpoint_path, start_epoch)

    if training_config.epochs <= start_epoch:
        logging.info(
            "Checkpoint epoch (%d) is greater than or equal to requested "
            "epochs (%d). Nothing to do.",
            start_epoch,
            training_config.epochs,
        )
        if writer is not None:
            writer.close()
        return

    logging.info("Starting training for %d epochs", training_config.epochs)
    for epoch in range(start_epoch + 1, training_config.epochs + 1):
        train_result = train_epoch(
            model,
            train_loader,
            optimizer,
            _loss_fn,
            device=device,
            scaler=scaler,
            grad_clip_norm=optim_config.grad_clip_norm,
            grad_accum_steps=training_config.grad_accum_steps,
            metrics=train_metrics,
        )
        val_result = eval_epoch(
            model,
            val_loader,
            _loss_fn,
            device=device,
            metrics=eval_metrics,
        )
        train_loss = train_result.loss
        val_loss = val_result.loss
        train_ctc = train_result.metrics.get("loss_ctc_weighted")
        val_ctc = val_result.metrics.get("loss_ctc_weighted")
        train_dist = train_result.metrics.get("loss_distillation_weighted")
        val_dist = val_result.metrics.get("loss_distillation_weighted")

        logging.info(
            "Epoch %d/%d - train_loss=%.4f - val_loss=%.4f - ppl_train=%.4f - "
            "ppl_val=%.4f - cer_val=%.4f - ctc_train=%.4f - ctc_val=%.4f - "
            "dist_train=%.4f - dist_val=%.4f",
            epoch,
            training_config.epochs,
            train_loss,
            val_loss,
            train_result.metrics.get("perplexity", float("nan")),
            val_result.metrics.get("perplexity", float("nan")),
            val_result.metrics.get("cer", float("nan")),
            float("nan") if train_ctc is None else float(train_ctc),
            float("nan") if val_ctc is None else float(val_ctc),
            float("nan") if train_dist is None else float(train_dist),
            float("nan") if val_dist is None else float(val_dist),
        )

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)
            if train_result.metrics:
                for name, value in train_result.metrics.items():
                    writer.add_scalar(f"train/{name}", value, epoch)
            if val_result.metrics:
                for name, value in val_result.metrics.items():
                    writer.add_scalar(f"val/{name}", value, epoch)

        improved = val_loss < best_val
        _save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            scaler=scaler,
            best_val=best_val,
            config=merged_config,
            scheduler=scheduler,
        )
        _save_rng_state(data_config.work_dir)

        if improved:
            best_val = val_loss
            logging.info("New best validation loss: %.4f", best_val)
            _save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                scaler=scaler,
                best_val=best_val,
                config=merged_config,
                scheduler=scheduler,
            )

        current_lr = optimizer.param_groups[0].get("lr", optim_config.lr)
        record = {
            "epoch": epoch,
            "train": {
                "loss": _safe_float(train_loss),
                **_convert_metrics(train_result.metrics),
            },
            "val": {
                "loss": _safe_float(val_loss),
                **_convert_metrics(val_result.metrics),
            },
            "learning_rate": _safe_float(current_lr),
            "best_val": _safe_float(best_val),
            "improved": improved,
        }
        _append_metrics(metrics_path, record)

        if scheduler is not None:
            scheduler.step()

    if writer is not None:
        writer.close()

    logging.info("Training completed. Best validation loss: %.4f", best_val)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
