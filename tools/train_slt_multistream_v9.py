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
import time
from collections.abc import Mapping
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import Subset
from transformers import PreTrainedTokenizerBase

from slt.data import LsaTMultiStream
from slt.training.configuration import (
    DataConfig,
    ModelConfig,
    OptimConfig,
    TrainingConfig,
    load_config_template,
    resolve_configs,
)
from slt.training.data import create_dataloader, normalise_mix_spec
from slt.training.loops import eval_epoch, multistream_loss, train_epoch
from slt.training.models import MultiStreamClassifier
from slt.training.optim import create_optimizer, create_scheduler
from slt.utils.cli import parse_range_pair, parse_translation_range
from slt.utils.general import set_seed
from slt.utils.text import create_tokenizer


_REPO_ROOT = Path(__file__).resolve().parents[1]
_DECODER_PRESET_DIR = _REPO_ROOT / "configs" / "presets"
_DECODER_PRESETS: dict[str, dict[str, Any]] = {
    "signmusketeers": {
        "path": _DECODER_PRESET_DIR / "decoder_signmusketeers_t5.yaml",
        "aliases": {"sign-musketeers", "sign_musketeers"},
        "summary": (
            "Preset inspirado en SignMusketeers que concatena rostro, manos y pose hacia"
            " un decoder T5 v1.1 Base."
        ),
    },
    "mska_paper_mbart": {
        "path": _DECODER_PRESET_DIR / "mska_paper_mbart.yaml",
        "aliases": {"mska-paper-mbart", "mska_slt_mbart", "mska_slt"},
        "summary": (
            "Replica los hiperparámetros MSKA-SLT (8 bloques, 6 cabezas) y permite"
            " alternar entre mBART-large CC25 y T5 v1.1 Base como decoder."
        ),
    },
}

_DECODER_MODEL_ALIASES: dict[str, str] = {
    "mbart": "facebook/mbart-large-cc25",
    "mbart-large": "facebook/mbart-large-cc25",
    "mbart-large-cc25": "facebook/mbart-large-cc25",
    "facebook/mbart-large-cc25": "facebook/mbart-large-cc25",
    "t5": "google/t5-v1_1-base",
    "t5-base": "google/t5-v1_1-base",
    "t5-v1_1-base": "google/t5-v1_1-base",
    "google/t5-v1_1-base": "google/t5-v1_1-base",
}


def _decoder_preset_names() -> list[str]:
    return sorted(_DECODER_PRESETS)


def _normalise_preset_name(raw: str) -> str:
    return raw.strip().lower().replace("-", "_")


def _resolve_decoder_preset(raw: str) -> tuple[str, dict[str, Any]]:
    normalised = _normalise_preset_name(raw)
    for name, spec in _DECODER_PRESETS.items():
        aliases = spec.get("aliases", set())
        if normalised == name or normalised in aliases:
            path = spec["path"]
            if not path.exists():
                raise FileNotFoundError(
                    f"Decoder preset '{name}' not found at {path}."
                )
            payload = dict(load_config_template(path))
            metadata = dict(payload.get("metadata", {}))
            metadata.setdefault("decoder_preset", name)
            payload["metadata"] = metadata
            return name, payload
    available = ", ".join(_decoder_preset_names())
    raise ValueError(
        f"Unknown decoder preset '{raw}'. Available presets: {available or 'none'}."
    )

try:  # pragma: no cover - numpy optional for RNG capture
    import numpy as np
except Exception:  # pragma: no cover - numpy optional
    np = None  # type: ignore

try:  # pragma: no cover - TensorBoard is an optional dependency.
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - if TensorBoard is not installed we noop.
    SummaryWriter = None  # type: ignore

try:  # pragma: no cover - sacrebleu is optional during training.
    from sacrebleu.metrics import BLEU, CHRF
except Exception:  # pragma: no cover - optional dependency may be missing.
    BLEU = None  # type: ignore[assignment]
    CHRF = None  # type: ignore[assignment]


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


class _TeacherForcingController:
    """Utility to manage teacher forcing and scheduled sampling."""

    def __init__(
        self,
        mode: str,
        ratio: float,
        min_ratio: float,
        decay: float,
    ) -> None:
        self.mode = mode
        self.initial_ratio = float(ratio)
        self.min_ratio = float(min_ratio)
        self.decay = float(decay)

    def ratio_for_epoch(self, epoch: int) -> float:
        if self.mode != "scheduled":
            return float(self.initial_ratio)
        steps = max(epoch - 1, 0)
        ratio = self.initial_ratio * (self.decay ** steps)
        ratio = max(self.min_ratio, ratio)
        return float(min(1.0, max(0.0, ratio)))

    def make_forward_fn(
        self, model: nn.Module, ratio: float
    ) -> Optional[Callable[..., Any]]:
        if self.mode != "scheduled" or ratio >= 1.0 - 1e-6:
            return None

        def _forward(**inputs: Any) -> Any:
            return self._scheduled_sampling_forward(model, inputs, ratio)

        return _forward

    def _scheduled_sampling_forward(
        self,
        model: nn.Module,
        inputs: Mapping[str, Any],
        ratio: float,
    ) -> Any:
        labels = inputs.get("labels")
        if not isinstance(labels, torch.Tensor):
            return model(**inputs)
        decoder = getattr(model, "decoder", None)
        if decoder is None or not hasattr(decoder, "prepare_decoder_input_ids"):
            return model(**inputs)

        # Avoid mutating the caller inputs
        scheduled_inputs = dict(inputs)

        with torch.no_grad():
            preview = model(**scheduled_inputs)
            logits = getattr(preview, "logits", None)
            if logits is None and hasattr(preview, "decoder"):
                logits = getattr(preview.decoder, "logits", None)
            if logits is None:
                return model(**inputs)
            predictions = logits.argmax(dim=-1)

        decoder_input_ids = decoder.prepare_decoder_input_ids(labels).clone()
        if decoder_input_ids.size(1) <= 1:
            scheduled_inputs["decoder_input_ids"] = decoder_input_ids
            return model(**scheduled_inputs)

        if not isinstance(predictions, torch.Tensor) or predictions.size() != labels.size():
            scheduled_inputs["decoder_input_ids"] = decoder_input_ids
            return model(**scheduled_inputs)

        teacher_tokens = labels[:, :-1]
        valid_mask = teacher_tokens.ne(-100)
        decoder_mask = scheduled_inputs.get("decoder_attention_mask")
        if isinstance(decoder_mask, torch.Tensor) and decoder_mask.size(1) > 1:
            valid_mask = valid_mask & decoder_mask[:, :-1].to(dtype=torch.bool)

        if valid_mask.any():
            bernoulli = torch.rand_like(valid_mask, dtype=torch.float, device=labels.device)
            use_teacher = bernoulli < ratio
            replacement_mask = valid_mask & ~use_teacher
            if replacement_mask.any():
                replacements = predictions[:, :-1].to(dtype=decoder_input_ids.dtype)
                decoder_input_ids[:, 1:][replacement_mask] = replacements[replacement_mask]

        scheduled_inputs["decoder_input_ids"] = decoder_input_ids
        return model(**scheduled_inputs)


def _move_to_device(data: Any, device: torch.device) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, Mapping):
        return type(data)({key: _move_to_device(value, device) for key, value in data.items()})
    if isinstance(data, (list, tuple)):
        return type(data)(_move_to_device(item, device) for item in data)
    return data


def _decode_label_texts(
    labels: torch.Tensor, tokenizer: PreTrainedTokenizerBase
) -> list[str]:
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
    labels_cpu = labels.detach().to(device="cpu")
    sanitised = labels_cpu.clone()
    sanitised[sanitised < 0] = pad_id
    return tokenizer.batch_decode(sanitised.tolist(), skip_special_tokens=True)


def _compute_validation_text_metrics(
    model: MultiStreamClassifier,
    loader: torch.utils.data.DataLoader,
    tokenizer: PreTrainedTokenizerBase,
    device: torch.device,
    *,
    max_length: int,
    num_beams: int = 1,
) -> dict[str, float]:
    if loader is None:
        return {}
    if BLEU is None or CHRF is None:
        logging.debug(
            "sacrebleu no está disponible; se omiten métricas BLEU/ChrF durante la validación."
        )
        return {}
    bleu_metric = BLEU(tokenize="13a")
    chrf_metric = CHRF()
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
    predictions: list[str] = []
    references: list[str] = []
    generation_kwargs = {
        "max_length": int(max_length),
        "num_beams": max(1, int(num_beams)),
        "early_stopping": num_beams > 1,
        "pad_token_id": pad_id,
        "eos_token_id": eos_id,
    }
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            batch_inputs = batch.get("inputs")
            if not isinstance(batch_inputs, Mapping):
                continue
            generation_inputs: dict[str, Any] = {}
            for key, value in batch_inputs.items():
                if key in {"labels", "decoder_attention_mask"}:
                    continue
                generation_inputs[key] = value
            prepared = _move_to_device(generation_inputs, device)
            sequences = model.generate(**prepared, **generation_kwargs)
            if hasattr(sequences, "sequences"):
                sequences_tensor = sequences.sequences  # type: ignore[attr-defined]
            else:
                sequences_tensor = sequences
            decoded_batch = tokenizer.batch_decode(
                sequences_tensor.detach().to(device="cpu").tolist(),
                skip_special_tokens=True,
            )
            predictions.extend(decoded_batch)
            label_tensor = batch.get("labels")
            if isinstance(label_tensor, torch.Tensor):
                references.extend(_decode_label_texts(label_tensor, tokenizer))
            else:
                references.extend([""] * len(decoded_batch))
    if not predictions or not references:
        return {}
    bleu_score = bleu_metric.corpus_score(predictions, [references]).score
    chrf_score = chrf_metric.corpus_score(predictions, [references]).score
    return {"bleu": float(bleu_score), "chrf": float(chrf_score)}


def _decoder_preset_help() -> str:
    if not _DECODER_PRESETS:
        return "Nombre del preset de decoder a aplicar."
    lines = [
        "Nombre del preset de decoder a aplicar. Disponibles:",
    ]
    for name in _decoder_preset_names():
        summary = _DECODER_PRESETS[name].get("summary")
        if summary:
            lines.append(f"  - {name}: {summary}")
        else:
            lines.append(f"  - {name}")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--config", type=Path, help="YAML or JSON configuration template")
    parser.add_argument(
        "--decoder-preset",
        type=str,
        help=_decoder_preset_help(),
    )
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
    parser.add_argument(
        "--tokenizer-local-files-only",
        action="store_true",
        help="Avoid network requests when loading the tokenizer.",
    )
    parser.add_argument(
        "--tokenizer-search-path",
        dest="tokenizer_search_paths",
        action="append",
        default=None,
        metavar="PATH",
        help="Additional directories checked before downloading the tokenizer.",
    )
    parser.add_argument(
        "--tokenizer-path-env",
        dest="tokenizer_path_env_vars",
        action="append",
        default=None,
        metavar="ENV",
        help="Environment variables with tokenizer paths (os.pathsep separated).",
    )
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
    parser.add_argument(
        "--leaky-relu-negative-slope",
        dest="leaky_relu_negative_slope",
        type=float,
        help="Negative slope used by all LeakyReLU activations in MSKA components",
    )
    parser.add_argument("--temporal-nhead", type=int, help="Number of attention heads in the temporal encoder")
    parser.add_argument("--temporal-layers", type=int, help="Number of transformer layers in the temporal encoder")
    parser.add_argument("--temporal-dim-feedforward", type=int, help="Feed-forward dimension inside the temporal encoder")
    parser.add_argument("--temporal-dropout", type=float, help="Dropout used by the temporal encoder")
    parser.add_argument("--decoder-layers", type=int, help="Number of layers in the seq2seq decoder")
    parser.add_argument("--decoder-heads", type=int, help="Number of attention heads in the seq2seq decoder")
    parser.add_argument("--decoder-dropout", type=float, help="Dropout probability inside the seq2seq decoder")
    parser.add_argument(
        "--decoder-local-files-only",
        action="store_true",
        help="Avoid network requests when loading decoder weights.",
    )
    parser.add_argument(
        "--decoder-search-path",
        dest="decoder_search_paths",
        action="append",
        default=None,
        metavar="PATH",
        help="Additional directories or files searched for decoder checkpoints.",
    )
    parser.add_argument(
        "--decoder-path-env",
        dest="decoder_path_env_vars",
        action="append",
        default=None,
        metavar="ENV",
        help="Environment variables pointing to decoder weight paths.",
    )
    parser.add_argument(
        "--decoder-hf-repo",
        dest="decoder_hf_repo",
        type=str,
        help="Hugging Face repository used to prefetch decoder weights offline.",
    )
    parser.add_argument(
        "--decoder-hf-filename",
        dest="decoder_hf_filename",
        type=str,
        help="Filename within the Hugging Face repo to download (defaults to weights).",
    )
    parser.add_argument(
        "--decoder-hf-revision",
        dest="decoder_hf_revision",
        type=str,
        help="Revision (branch, tag or commit) to use when downloading decoder weights.",
    )
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
    parser.add_argument(
        "--decoder-model",
        type=str,
        help=(
            "Pretrained decoder model name or path (alias mbart, mbart-large,"
            " mbart-large-cc25 → facebook/mbart-large-cc25)"
        ),
    )
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
        "--decoder-prompt-length",
        type=int,
        help="Número de embeddings de prompt aprendibles concatenados al codificador",
    )
    parser.add_argument(
        "--decoder-prompt-init",
        type=str,
        choices=["normal", "zero", "uniform", "tokens", "vocab"],
        help="Inicialización de las embeddings de prompt (normal, zero, uniform, tokens, vocab)",
    )
    parser.add_argument(
        "--decoder-prompt-std",
        type=float,
        help="Desviación estándar utilizada cuando prompt-init=normal",
    )
    parser.add_argument(
        "--decoder-prompt-range",
        type=float,
        help="Amplitud del intervalo uniforme cuando prompt-init=uniform",
    )
    parser.add_argument(
        "--decoder-prompt-text",
        type=str,
        help="Texto usado para inicializar las embeddings de prompt mediante el tokenizador",
    )
    parser.add_argument(
        "--decoder-prompt-tokens",
        type=str,
        help="Lista de IDs de tokens separados por comas para inicializar las embeddings de prompt",
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
        "--mska-use-sgr",
        dest="mska_use_sgr",
        action="store_true",
        help="Enable the shared global refinement (SGR) matrix inside MSKA streams",
    )
    parser.add_argument(
        "--mska-no-sgr",
        dest="mska_use_sgr",
        action="store_false",
        help="Disable the shared global refinement (SGR) matrix",
    )
    parser.set_defaults(mska_use_sgr=None)
    parser.add_argument(
        "--mska-sgr-shared",
        dest="mska_sgr_shared",
        action="store_true",
        help="Share the SGR matrix across all MSKA streams",
    )
    parser.add_argument(
        "--mska-sgr-per-stream",
        dest="mska_sgr_shared",
        action="store_false",
        help="Learn an independent SGR matrix per MSKA stream",
    )
    parser.set_defaults(mska_sgr_shared=None)
    parser.add_argument(
        "--mska-sgr-activation",
        type=str,
        help="Activation applied to the SGR matrix (softmax/sigmoid/tanh/relu/identity)",
    )
    parser.add_argument(
        "--mska-sgr-mix",
        type=float,
        help="Mixture factor between local attention and the SGR matrix (0-1)",
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
        "--mska-gloss-second-hidden-dim",
        dest="mska_gloss_second_hidden_dim",
        type=int,
        help="Second hidden size of the MSKA gloss MLP before the projection",
    )
    parser.add_argument(
        "--mska-gloss-activation",
        dest="mska_gloss_activation",
        choices=("leaky_relu",),
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
    parser.add_argument(
        "--lr-encoder",
        type=float,
        help="Learning rate for the encoder parameter group",
    )
    parser.add_argument(
        "--lr-decoder",
        type=float,
        help="Learning rate for the decoder parameter group",
    )
    parser.add_argument(
        "--lr-mska",
        type=float,
        help="Learning rate for MSKA-specific branches",
    )
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
        "--max-train-steps",
        type=int,
        help="Maximum number of training iterations across all epochs",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        help="Limit the training dataset to the first N samples",
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
    parser.add_argument(
        "--teacher-forcing-mode",
        choices=["standard", "scheduled"],
        help="Estrategia de teacher forcing a emplear durante el entrenamiento",
    )
    parser.add_argument(
        "--teacher-forcing-ratio",
        type=float,
        help="Probabilidad inicial de usar el token objetivo (0-1)",
    )
    parser.add_argument(
        "--teacher-forcing-min-ratio",
        type=float,
        help="Cota inferior de la probabilidad cuando se usa scheduled sampling",
    )
    parser.add_argument(
        "--teacher-forcing-decay",
        type=float,
        help="Factor multiplicativo aplicado cada época en scheduled sampling",
    )

    args = parser.parse_args()
    if args.decoder_preset:
        try:
            preset_name, preset_payload = _resolve_decoder_preset(args.decoder_preset)
        except (ValueError, FileNotFoundError) as exc:
            parser.error(str(exc))
        else:
            args.decoder_preset = preset_name
            args._decoder_preset_payload = preset_payload
    if args.decoder_config and args.decoder_model:
        parser.error("--decoder-model and --decoder-config are mutually exclusive")
    if args.decoder_class and (args.decoder_model or args.decoder_config):
        parser.error("--decoder-class cannot be combined with --decoder-model/--decoder-config")
    if args.decoder_model:
        canonical = _DECODER_MODEL_ALIASES.get(args.decoder_model.strip().lower())
        if canonical:
            args.decoder_model = canonical
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
    if args.decoder_prompt_tokens:
        raw_tokens = [item.strip() for item in args.decoder_prompt_tokens.split(",") if item.strip()]
        if not raw_tokens:
            args.decoder_prompt_tokens = None
        else:
            try:
                args.decoder_prompt_tokens = [int(value) for value in raw_tokens]
            except ValueError as exc:
                parser.error(f"--decoder-prompt-tokens must contain integers: {exc}")
    if args.teacher_forcing_mode:
        args.teacher_forcing_mode = args.teacher_forcing_mode.strip().lower()
    for name in ("teacher_forcing_ratio", "teacher_forcing_min_ratio"):
        value = getattr(args, name, None)
        if value is not None and not 0.0 <= value <= 1.0:
            parser.error(f"--{name.replace('_', '-')} must be between 0 and 1")
    if (
        args.teacher_forcing_ratio is not None
        and args.teacher_forcing_min_ratio is not None
        and args.teacher_forcing_min_ratio > args.teacher_forcing_ratio
    ):
        parser.error("--teacher-forcing-min-ratio cannot be greater than --teacher-forcing-ratio")
    if (
        args.teacher_forcing_mode == "scheduled"
        and args.teacher_forcing_decay is not None
        and args.teacher_forcing_decay <= 0
    ):
        parser.error("--teacher-forcing-decay must be positive when using scheduled sampling")
    for attr in ("lr_encoder", "lr_decoder", "lr_mska"):
        value = getattr(args, attr, None)
        if value is not None and value <= 0:
            parser.error(f"--{attr.replace('_', '-')} must be positive")
    for attr in ("max_train_steps", "subset_size"):
        value = getattr(args, attr, None)
        if value is not None and value <= 0:
            parser.error(f"--{attr.replace('_', '-')} must be greater than zero")
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
    if args.tokenizer_search_paths is not None:
        args.tokenizer_search_paths = [
            str(Path(path).expanduser()) for path in args.tokenizer_search_paths
        ]
    if args.tokenizer_path_env_vars is not None:
        cleaned = [str(var).strip() for var in args.tokenizer_path_env_vars if var]
        args.tokenizer_path_env_vars = cleaned or None
    if args.decoder_search_paths is not None:
        args.decoder_search_paths = [
            str(Path(path).expanduser()) for path in args.decoder_search_paths
        ]
    if args.decoder_path_env_vars is not None:
        cleaned = [str(var).strip() for var in args.decoder_path_env_vars if var]
        args.decoder_path_env_vars = cleaned or None
    explicit_bool_flags = set()
    for name in (
        "use_mska",
        "mska_detach_teacher",
        "keypoint_normalize_center",
        "mska_use_sgr",
        "mska_sgr_shared",
        "tokenizer_local_files_only",
        "decoder_local_files_only",
    ):
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


def _collect_explicit_override_fields(
    cli_overrides: Mapping[str, Mapping[str, Any]],
    set_overrides: Iterable[str],
) -> tuple[set[str], set[str]]:
    data_fields = {str(key) for key in cli_overrides.get("data", {})}
    model_fields = {str(key) for key in cli_overrides.get("model", {})}

    for item in set_overrides:
        if "=" not in item:
            continue
        dotted, _ = item.split("=", 1)
        dotted = dotted.strip()
        if dotted.startswith("data."):
            data_fields.add(dotted.split(".", 1)[1])
        elif dotted.startswith("model."):
            model_fields.add(dotted.split(".", 1)[1])

    return data_fields, model_fields


def _select_decoder_variant(
    metadata: Mapping[str, Any],
    identifier: Optional[str],
) -> tuple[Optional[str], Optional[Mapping[str, Any]]]:
    variants = metadata.get("decoder_variants") if metadata else None
    if not isinstance(variants, Mapping):
        return None, None
    if identifier:
        target = identifier.strip().lower()
        for name, spec in variants.items():
            if not isinstance(spec, Mapping):
                continue
            aliases: set[str] = {str(name).strip().lower()}
            model_name = spec.get("model")
            if isinstance(model_name, str):
                aliases.add(model_name.strip().lower())
            raw_aliases = spec.get("aliases", ())
            if isinstance(raw_aliases, Iterable) and not isinstance(raw_aliases, (str, bytes)):
                aliases.update(str(item).strip().lower() for item in raw_aliases)
            if target in aliases:
                return str(name), spec
    default_name = metadata.get("default_decoder")
    if isinstance(default_name, str) and default_name in variants:
        spec = variants[default_name]
        if isinstance(spec, Mapping):
            return default_name, spec
    return None, None


def _apply_decoder_variant_defaults(
    *,
    variant_name: str,
    variant_spec: Mapping[str, Any],
    metadata: dict[str, Any],
    data_config: DataConfig,
    model_config: ModelConfig,
    merged: dict[str, Any],
    explicit_data_fields: set[str],
    explicit_model_fields: set[str],
) -> None:
    metadata["decoder_variant"] = variant_name

    model_section = merged.setdefault("model", {})
    data_section = merged.setdefault("data", {})

    decoder_model = variant_spec.get("model")
    if isinstance(decoder_model, str) and "decoder_model" not in explicit_model_fields:
        model_config.decoder_model = decoder_model
        model_section["decoder_model"] = decoder_model

    for field_name in ("decoder_layers", "decoder_heads", "decoder_dropout"):
        if field_name in explicit_model_fields:
            continue
        if field_name in variant_spec:
            value = variant_spec[field_name]
            if value is not None:
                setattr(model_config, field_name, value)
                model_section[field_name] = value

    if "decoder_kwargs" not in explicit_model_fields:
        decoder_kwargs = variant_spec.get("decoder_kwargs")
        if isinstance(decoder_kwargs, Mapping):
            model_config.decoder_kwargs = dict(decoder_kwargs)
            model_section["decoder_kwargs"] = dict(decoder_kwargs)

    tokenizer_name = variant_spec.get("tokenizer")
    if isinstance(tokenizer_name, str) and "tokenizer" not in explicit_data_fields:
        data_config.tokenizer = tokenizer_name
        data_section["tokenizer"] = tokenizer_name


def build_configs(
    args: argparse.Namespace,
) -> tuple[DataConfig, ModelConfig, OptimConfig, TrainingConfig, dict[str, Any]]:
    cli_overrides = _collect_cli_overrides(args)
    explicit_data_fields, explicit_model_fields = _collect_explicit_override_fields(
        cli_overrides,
        args.overrides,
    )
    base: dict[str, Any] = {}
    decoder_preset = getattr(args, "decoder_preset", None)
    if decoder_preset:
        preset_payload = getattr(args, "_decoder_preset_payload", None)
        if preset_payload is None:
            _, preset_payload = _resolve_decoder_preset(decoder_preset)
        base = dict(preset_payload)
    (
        data_config,
        model_config,
        optim_config,
        training_config,
        merged_config,
    ) = resolve_configs(
        config_path=args.config,
        cli_overrides=cli_overrides,
        set_overrides=args.overrides,
        base=base,
    )

    metadata_obj = merged_config.get("metadata", {})
    if isinstance(metadata_obj, dict):
        metadata = metadata_obj
    else:
        metadata = dict(metadata_obj) if metadata_obj else {}
        merged_config["metadata"] = metadata
    variant_identifier = getattr(args, "decoder_model", None) or model_config.decoder_model
    variant_name, variant_spec = _select_decoder_variant(
        metadata,
        variant_identifier,
    )
    if variant_name and variant_spec:
        _apply_decoder_variant_defaults(
            variant_name=variant_name,
            variant_spec=variant_spec,
            metadata=metadata,
            data_config=data_config,
            model_config=model_config,
            merged=merged_config,
            explicit_data_fields=explicit_data_fields,
            explicit_model_fields=explicit_model_fields,
        )

    return data_config, model_config, optim_config, training_config, merged_config


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
    tokenizer = create_tokenizer(
        tokenizer_source,
        local_files_only=data_config.tokenizer_local_files_only,
        local_paths=(data_config.tokenizer_search_paths or None),
        env_var_paths=(data_config.tokenizer_path_env_vars or None),
    )

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

    subset_size = training_config.subset_size
    if subset_size is not None:
        try:
            dataset_length = len(train_dataset)
        except TypeError:
            dataset_length = None
        if dataset_length is None or subset_size < dataset_length:
            capped = subset_size if dataset_length is None else min(subset_size, dataset_length)
            indices = range(capped)
            train_dataset = Subset(train_dataset, indices)
            logging.info("Limiting training dataset to %d samples", capped)
        else:
            logging.info(
                "Subset size %d exceeds dataset length %d; training on full dataset.",
                subset_size,
                dataset_length,
            )

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
    encoder_params: list[nn.Parameter] = []
    decoder_params: list[nn.Parameter] = []
    mska_params: list[nn.Parameter] = []
    mska_param_ids: set[int] = set()

    if hasattr(model, "encoder"):
        encoder_module = model.encoder
        mska_encoder = getattr(encoder_module, "mska_encoder", None)
        if isinstance(mska_encoder, nn.Module):
            for param in mska_encoder.parameters():
                if param.requires_grad:
                    mska_params.append(param)
                    mska_param_ids.add(id(param))
        gloss_mlp = getattr(encoder_module, "_mska_gloss_mlp", None)
        if isinstance(gloss_mlp, nn.Module):
            for param in gloss_mlp.parameters():
                if param.requires_grad:
                    mska_params.append(param)
                    mska_param_ids.add(id(param))
        for param in encoder_module.parameters():
            if not param.requires_grad or id(param) in mska_param_ids:
                continue
            encoder_params.append(param)

    if hasattr(model, "decoder"):
        for param in model.decoder.parameters():
            if param.requires_grad:
                decoder_params.append(param)

    encoder_lr = optim_config.lr_encoder or optim_config.lr
    decoder_lr = optim_config.lr_decoder or optim_config.lr
    mska_lr = optim_config.lr_mska or encoder_lr

    param_groups: list[dict[str, Any]] = []
    if encoder_params:
        param_groups.append({"params": encoder_params, "lr": float(encoder_lr)})
    if mska_params:
        param_groups.append({"params": mska_params, "lr": float(mska_lr)})
    if decoder_params:
        param_groups.append({"params": decoder_params, "lr": float(decoder_lr)})

    if not param_groups:
        raise RuntimeError("No trainable parameters found for optimiser setup")

    optimizer = create_optimizer(
        param_groups,
        {
            "type": optim_config.optimizer,
            "lr": optim_config.lr,
            "weight_decay": optim_config.weight_decay,
        },
    )

    scheduler = None
    if optim_config.scheduler:
        if optim_config.scheduler == "steplr":
            step_size = optim_config.scheduler_step_size
            if training_config.max_train_steps is not None:
                step_size = max(1, min(step_size, training_config.max_train_steps))
            scheduler = create_scheduler(
                optimizer,
                {
                    "type": "steplr",
                    "step_size": step_size,
                    "gamma": optim_config.scheduler_gamma,
                },
            )
        elif optim_config.scheduler == "cosine":
            t_max = optim_config.scheduler_step_size
            if training_config.max_train_steps is not None:
                t_max = max(1, min(t_max, training_config.max_train_steps))
            scheduler = create_scheduler(
                optimizer,
                {
                    "type": "cosine",
                    "t_max": t_max,
                },
            )

    teacher_controller = _TeacherForcingController(
        training_config.teacher_forcing_mode,
        training_config.teacher_forcing_ratio,
        training_config.teacher_forcing_min_ratio,
        training_config.teacher_forcing_decay,
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

    max_train_steps = training_config.max_train_steps
    steps_completed = 0
    if max_train_steps is not None:
        logging.info("Limiting total training steps to %d", max_train_steps)

    logging.info("Starting training for %d epochs", training_config.epochs)
    last_logged_ratio: Optional[float] = None
    for epoch in range(start_epoch + 1, training_config.epochs + 1):
        epoch_start = time.perf_counter()
        peak_memory_bytes: Optional[int] = None
        if device.type == "cuda" and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats(device)
        remaining_steps: Optional[int] = None
        if max_train_steps is not None:
            remaining_steps = max_train_steps - steps_completed
            if remaining_steps <= 0:
                logging.info(
                    "Reached max_train_steps=%d before epoch %d; stopping training.",
                    max_train_steps,
                    epoch,
                )
                break
        current_ratio = teacher_controller.ratio_for_epoch(epoch)
        forward_fn = teacher_controller.make_forward_fn(model, current_ratio)
        if last_logged_ratio is None or abs(current_ratio - last_logged_ratio) > 1e-6:
            logging.info(
                "Epoch %d - teacher forcing ratio (mode=%s): %.4f",
                epoch,
                training_config.teacher_forcing_mode,
                current_ratio,
            )
            last_logged_ratio = current_ratio
        train_result = train_epoch(
            model,
            train_loader,
            optimizer,
            _loss_fn,
            device=device,
            scaler=scaler,
            grad_clip_norm=optim_config.grad_clip_norm,
            grad_accum_steps=training_config.grad_accum_steps,
            max_steps=remaining_steps,
            metrics=train_metrics,
            forward_fn=forward_fn,
        )
        steps_completed += train_result.steps
        val_result = eval_epoch(
            model,
            val_loader,
            _loss_fn,
            device=device,
            metrics=eval_metrics,
        )
        text_metrics = _compute_validation_text_metrics(
            model,
            val_loader,
            tokenizer,
            device,
            max_length=data_config.max_target_length,
        )
        if text_metrics:
            val_result.metrics.update(text_metrics)
        train_loss = train_result.loss
        val_loss = val_result.loss
        train_ctc = train_result.metrics.get("loss_ctc_weighted")
        val_ctc = val_result.metrics.get("loss_ctc_weighted")
        train_dist = train_result.metrics.get("loss_distillation_weighted")
        val_dist = val_result.metrics.get("loss_distillation_weighted")
        epoch_time = time.perf_counter() - epoch_start
        if device.type == "cuda" and torch.cuda.is_available():
            peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
        bleu_val = val_result.metrics.get("bleu", float("nan"))
        chrf_val = val_result.metrics.get("chrf", float("nan"))
        peak_memory_gb = (
            float(peak_memory_bytes) / (1024**3)
            if peak_memory_bytes is not None
            else float("nan")
        )

        logging.info(
            "Epoch %d/%d - train_loss=%.4f - val_loss=%.4f - ppl_train=%.4f - "
            "ppl_val=%.4f - cer_val=%.4f - bleu_val=%.2f - chrf_val=%.2f - "
            "ctc_train=%.4f - ctc_val=%.4f - dist_train=%.4f - dist_val=%.4f - "
            "epoch_time=%.2fs - peak_mem=%.2fGB",
            epoch,
            training_config.epochs,
            train_loss,
            val_loss,
            train_result.metrics.get("perplexity", float("nan")),
            val_result.metrics.get("perplexity", float("nan")),
            val_result.metrics.get("cer", float("nan")),
            bleu_val,
            chrf_val,
            float("nan") if train_ctc is None else float(train_ctc),
            float("nan") if val_ctc is None else float(val_ctc),
            float("nan") if train_dist is None else float(train_dist),
            float("nan") if val_dist is None else float(val_dist),
            epoch_time,
            peak_memory_gb,
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
            writer.add_scalar("system/epoch_time_s", epoch_time, epoch)
            if peak_memory_bytes is not None:
                writer.add_scalar(
                    "system/peak_memory_gb",
                    peak_memory_gb,
                    epoch,
                )
            writer.add_scalar(
                "train/teacher_forcing_ratio",
                current_ratio,
                epoch,
            )

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
                "steps": int(train_result.steps),
                **_convert_metrics(train_result.metrics),
            },
            "val": {
                "loss": _safe_float(val_loss),
                **_convert_metrics(val_result.metrics),
            },
            "learning_rate": _safe_float(current_lr),
            "best_val": _safe_float(best_val),
            "improved": improved,
            "epoch_time_s": _safe_float(epoch_time),
            "peak_memory_bytes": peak_memory_bytes,
            "teacher_forcing": {
                "mode": training_config.teacher_forcing_mode,
                "ratio": _safe_float(current_ratio),
            },
        }
        _append_metrics(metrics_path, record)

        if scheduler is not None and train_result.steps > 0:
            scheduler.step()

        if max_train_steps is not None and steps_completed >= max_train_steps:
            logging.info(
                "Reached max_train_steps=%d after epoch %d; stopping training loop.",
                max_train_steps,
                epoch,
            )
            break

    if writer is not None:
        writer.close()

    logging.info("Training completed. Best validation loss: %.4f", best_val)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
