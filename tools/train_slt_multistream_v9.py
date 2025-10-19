#!/usr/bin/env python3
"""Command line entry-point to train the multi-stream SLT stub model.

The script wires together the reusable components from :mod:`slt` to
instantiate the dataset, encoder/decoder pair and the optimisation loop.
Tokenizer and decoder functionality rely on HuggingFace ``transformers`` so the
package must be installed alongside PyTorch.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple, Union, get_args, get_origin

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from slt.data import LsaTMultiStream, collate_fn
from slt.models import MultiStreamEncoder, TextSeq2SeqDecoder, ViTConfig, load_dinov2_backbone
from slt.training.loops import LoopResult, eval_epoch, train_epoch
from slt.training.optim import create_optimizer, create_scheduler
from slt.utils.general import set_seed
from slt.utils.text import create_tokenizer, encode_batch

from transformers import AutoConfig, PreTrainedTokenizerBase

try:
    import yaml
except Exception:  # pragma: no cover - yaml is optional
    yaml = None  # type: ignore

try:  # pragma: no cover - numpy optional for RNG capture
    import numpy as np
except Exception:  # pragma: no cover - numpy optional
    np = None  # type: ignore

try:  # pragma: no cover - TensorBoard is an optional dependency.
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - if TensorBoard is not installed we noop.
    SummaryWriter = None  # type: ignore


@dataclass
class ModelConfig:
    """Model hyper-parameters exposed in the CLI."""

    image_size: int = 224
    projector_dim: int = 256
    d_model: int = 512
    pose_landmarks: int = 13
    projector_dropout: float = 0.0
    fusion_dropout: float = 0.0
    temporal_nhead: int = 8
    temporal_layers: int = 6
    temporal_dim_feedforward: int = 2048
    temporal_dropout: float = 0.1
    sequence_length: int = 128
    decoder_layers: int = 2
    decoder_heads: int = 8
    decoder_dropout: float = 0.1
    decoder_model: Optional[str] = None
    decoder_config: Optional[str] = None
    face_backbone: Optional[str] = None
    hand_left_backbone: Optional[str] = None
    hand_right_backbone: Optional[str] = None
    freeze_face_backbone: bool = False
    freeze_hand_left_backbone: bool = False
    freeze_hand_right_backbone: bool = False


@dataclass
class OptimConfig:
    """Optimisation related hyper-parameters."""

    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    scheduler: Optional[str] = None
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.5
    label_smoothing: float = 0.1
    grad_clip_norm: Optional[float] = None


@dataclass
class TrainingConfig:
    """Configuration of the outer training loop."""

    epochs: int = 40
    grad_accum_steps: int = 1
    compile: bool = False
    compile_mode: Optional[str] = None


@dataclass
class DataConfig:
    """Paths and data loading configuration."""

    face_dir: Path
    hand_left_dir: Path
    hand_right_dir: Path
    pose_dir: Path
    metadata_csv: Path
    train_index: Path
    val_index: Path
    work_dir: Path
    num_workers: int = 0
    batch_size: int = 4
    val_batch_size: Optional[int] = None
    seed: int = 1234
    device: str = "cuda"
    precision: str = "amp"
    tensorboard: Optional[Path] = None
    pin_memory: bool = True
    tokenizer: Optional[str] = None
    max_target_length: int = 128


class MultiStreamClassifier(nn.Module):
    """Convenience wrapper around the encoder/decoder pair."""

    def __init__(self, config: ModelConfig, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()

        vit_config = ViTConfig(image_size=config.image_size)
        temporal_kwargs = {
            "nhead": config.temporal_nhead,
            "nlayers": config.temporal_layers,
            "dim_feedforward": config.temporal_dim_feedforward,
            "dropout": config.temporal_dropout,
        }

        backbone_specs = {
            "face": (config.face_backbone, config.freeze_face_backbone),
            "hand_left": (config.hand_left_backbone, config.freeze_hand_left_backbone),
            "hand_right": (config.hand_right_backbone, config.freeze_hand_right_backbone),
        }
        external_backbones: Dict[str, torch.nn.Module] = {}
        for stream, (spec, freeze_flag) in backbone_specs.items():
            if spec:
                external_backbones[stream] = load_dinov2_backbone(spec, freeze=freeze_flag)

        self.encoder = MultiStreamEncoder(
            backbone_config=vit_config,
            projector_dim=config.projector_dim,
            d_model=config.d_model,
            pose_dim=3 * config.pose_landmarks,
            positional_num_positions=config.sequence_length,
            projector_dropout=config.projector_dropout,
            fusion_dropout=config.fusion_dropout,
            temporal_kwargs=temporal_kwargs,
            backbones=external_backbones if external_backbones else None,
        )
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if not vocab_size:
            vocab_size = len(tokenizer)

        decoder_config = None
        if config.decoder_config:
            decoder_config = AutoConfig.from_pretrained(config.decoder_config)
            hidden_size = getattr(decoder_config, "d_model", None)
            if hidden_size is not None and hidden_size != config.d_model:
                raise ValueError(
                    "Decoder configuration hidden size does not match encoder dimensionality: "
                    f"expected {config.d_model}, got {hidden_size}."
                )

        decoder_model_name = None if decoder_config is not None else config.decoder_model

        self.decoder = TextSeq2SeqDecoder(
            d_model=config.d_model,
            vocab_size=int(vocab_size),
            num_layers=config.decoder_layers,
            num_heads=config.decoder_heads,
            dropout=config.decoder_dropout,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            pretrained_model_name_or_path=decoder_model_name,
            config=decoder_config,
        )

    def forward(
        self,
        *,
        face: torch.Tensor,
        hand_l: torch.Tensor,
        hand_r: torch.Tensor,
        pose: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        miss_mask_hl: Optional[torch.Tensor] = None,
        miss_mask_hr: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded = self.encoder(
            face,
            hand_l,
            hand_r,
            pose,
            pad_mask=pad_mask,
            miss_mask_hl=miss_mask_hl,
            miss_mask_hr=miss_mask_hr,
        )
        if encoder_attention_mask is None and pad_mask is not None:
            encoder_attention_mask = pad_mask.to(torch.long)
        return self.decoder(
            encoded,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

    def generate(
        self,
        *,
        face: torch.Tensor,
        hand_l: torch.Tensor,
        hand_r: torch.Tensor,
        pose: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        miss_mask_hl: Optional[torch.Tensor] = None,
        miss_mask_hr: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs: Any,
    ) -> torch.LongTensor:
        encoded = self.encoder(
            face,
            hand_l,
            hand_r,
            pose,
            pad_mask=pad_mask,
            miss_mask_hl=miss_mask_hl,
            miss_mask_hr=miss_mask_hr,
        )
        if encoder_attention_mask is None and pad_mask is not None:
            encoder_attention_mask = pad_mask.to(torch.long)
        return self.decoder.generate(
            encoded,
            encoder_attention_mask=encoder_attention_mask,
            **generation_kwargs,
        )


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
    parser.add_argument("--metadata-csv", type=Path, help="CSV file with video_id/text pairs")
    parser.add_argument("--train-index", type=Path, help="CSV file listing training video IDs")
    parser.add_argument("--val-index", type=Path, help="CSV file listing validation video IDs")
    parser.add_argument("--work-dir", type=Path, help="Directory where checkpoints/logs will be saved")

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
    parser.add_argument("--hand-left-backbone", type=str, help="Backbone specification for the left hand stream")
    parser.add_argument("--hand-right-backbone", type=str, help="Backbone specification for the right hand stream")
    parser.add_argument("--freeze-face-backbone", action="store_true", help="Freeze the face backbone")
    parser.add_argument("--freeze-hand-left-backbone", action="store_true", help="Freeze the left hand backbone")
    parser.add_argument("--freeze-hand-right-backbone", action="store_true", help="Freeze the right hand backbone")
    parser.add_argument("--decoder-model", type=str, help="Pretrained decoder model name or path")
    parser.add_argument("--decoder-config", type=str, help="Decoder configuration name or path")

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
    parser.add_argument("--grad-accum-steps", type=int, help="Number of gradient accumulation steps")
    parser.add_argument("--compile-mode", type=str, help="torch.compile mode (default, reduce-overhead, max-autotune)")
    parser.add_argument("--compile", dest="compile_flag", action="store_true", help="Enable torch.compile")
    parser.add_argument("--no-compile", dest="compile_flag", action="store_false", help="Disable torch.compile")
    parser.set_defaults(compile_flag=None)

    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint path to resume training from (loads model, optimiser and AMP scaler state).",
    )

    args = parser.parse_args()
    if args.decoder_config and args.decoder_model:
        parser.error("--decoder-model and --decoder-config are mutually exclusive")
    if args.clip_grad_norm is not None and args.clip_grad_norm <= 0:
        logging.warning("Ignoring non-positive --clip-grad-norm value: %s", args.clip_grad_norm)
        args.clip_grad_norm = None
    return args


def _dataclass_defaults(cls: type) -> Dict[str, Any]:
    defaults: Dict[str, Any] = {}
    for field in fields(cls):
        if field.default is not MISSING:
            defaults[field.name] = field.default
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            defaults[field.name] = field.default_factory()  # type: ignore[misc]
    return defaults


def _load_config_template(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        if yaml is None:
            raise RuntimeError("PyYAML is required to parse YAML configuration files")
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, Mapping):
        raise ValueError("Configuration root must be a mapping")
    return dict(data)


def _deep_update(target: Dict[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            target[key] = _deep_update(dict(target[key]), value)
        else:
            target[key] = value
    return target


def _coerce_override_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered == "null":
            return None
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value


def _apply_string_overrides(config: Dict[str, Any], overrides: Iterable[str]) -> None:
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override format '{item}'. Expected KEY=VALUE")
        key, raw_value = item.split("=", 1)
        value = _coerce_override_value(raw_value)
        parts = key.split(".")
        cursor: Dict[str, Any] = config
        for part in parts[:-1]:
            if part not in cursor or not isinstance(cursor[part], dict):
                cursor[part] = {}
            cursor = cursor[part]
        cursor[parts[-1]] = value


def _coerce_field_value(field_obj, value: Any) -> Any:
    if value is None:
        return None
    annotation = field_obj.type
    origin = get_origin(annotation)
    if origin is Union:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            annotation = args[0]
            origin = get_origin(annotation)
    if annotation is Path and not isinstance(value, Path):
        return Path(value)
    if annotation is int and isinstance(value, str):
        return int(value)
    if annotation is float and isinstance(value, str):
        return float(value)
    if annotation is bool and isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
    return value


def _instantiate_config(cls: type, values: Mapping[str, Any]) -> Any:
    kwargs: Dict[str, Any] = {}
    missing: list[str] = []
    for field_obj in fields(cls):
        if field_obj.name in values:
            val = values[field_obj.name]
            if isinstance(val, str) and field_obj.type is Path:
                val = Path(val)
            kwargs[field_obj.name] = _coerce_field_value(field_obj, val)
        elif field_obj.default is not MISSING or field_obj.default_factory is not MISSING:  # type: ignore[attr-defined]
            continue
        else:
            missing.append(field_obj.name)
    if missing:
        raise ValueError(
            f"Missing configuration values for {cls.__name__}: {', '.join(missing)}"
        )
    return cls(**kwargs)


def _collect_cli_overrides(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    sections = {
        "data": DataConfig,
        "model": ModelConfig,
        "optim": OptimConfig,
        "training": TrainingConfig,
    }
    overrides: Dict[str, Dict[str, Any]] = {key: {} for key in sections}

    for section, cls in sections.items():
        for field_obj in fields(cls):
            attr_name = field_obj.name
            if not hasattr(args, attr_name):
                continue
            value = getattr(args, attr_name)
            if isinstance(value, bool):
                if value:
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
) -> Tuple[DataConfig, ModelConfig, OptimConfig, TrainingConfig, Dict[str, Any]]:
    base: Dict[str, Any] = {
        "data": _dataclass_defaults(DataConfig),
        "model": _dataclass_defaults(ModelConfig),
        "optim": _dataclass_defaults(OptimConfig),
        "training": _dataclass_defaults(TrainingConfig),
    }

    if args.config is not None:
        loaded = _load_config_template(args.config)
        for key in ("data", "model", "optim", "training"):
            if key in loaded and isinstance(loaded[key], Mapping):
                base[key] = _deep_update(base.get(key, {}), loaded[key])
        for key, value in loaded.items():
            if key not in base:
                base[key] = value

    cli_overrides = _collect_cli_overrides(args)
    for section, values in cli_overrides.items():
        if not values:
            continue
        base.setdefault(section, {})
        base[section].update(values)

    _apply_string_overrides(base, args.overrides)

    data_config = _instantiate_config(DataConfig, base.get("data", {}))
    model_config = _instantiate_config(ModelConfig, base.get("model", {}))
    optim_config = _instantiate_config(OptimConfig, base.get("optim", {}))
    training_config = _instantiate_config(TrainingConfig, base.get("training", {}))

    scheduler_choice = optim_config.scheduler
    if scheduler_choice is not None:
        scheduler_choice = scheduler_choice.lower()
        if scheduler_choice == "none":
            optim_config.scheduler = None
        else:
            optim_config.scheduler = scheduler_choice

    if optim_config.scheduler == "cosine":
        tmax = base.get("optim", {}).get("scheduler_tmax")
        if tmax is None:
            tmax = optim_config.scheduler_step_size
        optim_config.scheduler_step_size = int(tmax)

    if data_config.val_batch_size is None:
        data_config.val_batch_size = data_config.batch_size

    return data_config, model_config, optim_config, training_config, base


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
    _ensure_exists(config.metadata_csv, kind="Metadata CSV")
    _ensure_exists(config.train_index, kind="Train index CSV")
    _ensure_exists(config.val_index, kind="Validation index CSV")
    config.work_dir.mkdir(parents=True, exist_ok=True)


def _build_collate(
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
) -> Callable[[Iterable[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    def _collate(batch: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        merged = collate_fn(batch)
        tokenized = encode_batch(
            tokenizer,
            merged["texts"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        inputs: Dict[str, torch.Tensor] = {
            "face": merged["face"],
            "hand_l": merged["hand_l"],
            "hand_r": merged["hand_r"],
            "pose": merged["pose"],
            "pad_mask": merged["pad_mask"],
            "lengths": merged["lengths"],
            "miss_mask_hl": merged["miss_mask_hl"],
            "miss_mask_hr": merged["miss_mask_hr"],
            "labels": labels,
            "decoder_attention_mask": attention_mask,
            "encoder_attention_mask": merged["pad_mask"].to(torch.long),
        }
        return {"inputs": inputs, "labels": labels, "video_ids": merged["video_ids"]}

    return _collate


def _create_dataloader(
    dataset: LsaTMultiStream,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> DataLoader:
    collate = _build_collate(tokenizer, max_length=max_length)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )


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
        logging.info("Model compiled with torch.compile (mode=%s)", training.compile_mode or "default")
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


def _capture_rng_state() -> Dict[str, Any]:
    state: Dict[str, Any] = {
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


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    best_val: Optional[float] = None,
    config: Optional[Mapping[str, Any]] = None,
) -> None:
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": float(val_loss),
    }
    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()
    if best_val is not None:
        state["best_val"] = float(best_val)
    if config is not None:
        state["config"] = json.loads(json.dumps(dict(config), default=str))
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

    device = torch.device(data_config.device)
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

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
        T=model_config.sequence_length,
        img_size=model_config.image_size,
        lkp_count=model_config.pose_landmarks,
    )
    val_dataset = LsaTMultiStream(
        face_dir=str(data_config.face_dir),
        hand_l_dir=str(data_config.hand_left_dir),
        hand_r_dir=str(data_config.hand_right_dir),
        pose_dir=str(data_config.pose_dir),
        csv_path=str(data_config.metadata_csv),
        index_csv=str(data_config.val_index),
        T=model_config.sequence_length,
        img_size=model_config.image_size,
        lkp_count=model_config.pose_landmarks,
    )

    val_batch_size = data_config.val_batch_size or data_config.batch_size
    train_loader = _create_dataloader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        tokenizer=tokenizer,
        max_length=data_config.max_target_length,
    )
    val_loader = _create_dataloader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        tokenizer=tokenizer,
        max_length=data_config.max_target_length,
    )

    model = MultiStreamClassifier(model_config, tokenizer).to(device)
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

    def _loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        vocab = logits.size(-1)
        try:
            return F.cross_entropy(
                logits.view(-1, vocab),
                targets.view(-1),
                ignore_index=-100,
                label_smoothing=optim_config.label_smoothing,
            )
        except TypeError:  # pragma: no cover - fallback when label smoothing unsupported
            logging.warning(
                "Label smoothing unsupported in this PyTorch version. Falling back to default loss.")
            return F.cross_entropy(
                logits.view(-1, vocab),
                targets.view(-1),
                ignore_index=-100,
            )

    use_amp = data_config.precision == "amp" and device.type == "cuda" and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    perplexity_metric = _build_perplexity_metric()
    cer_metric = _build_cer_metric(tokenizer)
    train_metrics = {"perplexity": perplexity_metric}
    eval_metrics = {"perplexity": perplexity_metric, "cer": cer_metric}

    writer = None
    if data_config.tensorboard is not None:
        if SummaryWriter is None:
            logging.warning("TensorBoard requested but not available. Install tensorboard to enable logging.")
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
        best_val = float(checkpoint.get("best_val", checkpoint.get("val_loss", best_val)))
        start_epoch = int(checkpoint.get("epoch", 0))
        logging.info("Resumed training from %s at epoch %d", checkpoint_path, start_epoch)

    if training_config.epochs <= start_epoch:
        logging.info(
            "Checkpoint epoch (%d) is greater than or equal to requested epochs (%d). Nothing to do.",
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

        logging.info(
            "Epoch %d/%d - train_loss=%.4f - val_loss=%.4f - ppl_train=%.4f - ppl_val=%.4f - cer_val=%.4f",
            epoch,
            training_config.epochs,
            train_loss,
            val_loss,
            train_result.metrics.get("perplexity", float("nan")),
            val_result.metrics.get("perplexity", float("nan")),
            val_result.metrics.get("cer", float("nan")),
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

        _save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            scaler=scaler,
            best_val=best_val,
            config=merged_config,
        )
        _save_rng_state(data_config.work_dir)

        if val_loss < best_val:
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
            )

        if scheduler is not None:
            scheduler.step()

    if writer is not None:
        writer.close()

    logging.info("Training completed. Best validation loss: %.4f", best_val)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
