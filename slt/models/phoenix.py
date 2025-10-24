"""Helpers to load the Phoenix 2014 MSKA checkpoint."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Tuple, Union

import torch

from .backbones import ViTConfig
from .multistream import MultiStreamEncoder
from .mska import MSKAEncoder
from .temporal import TextSeq2SeqDecoder

__all__ = [
    "CHECKPOINT_ENV_VAR",
    "CHECKPOINT_FILENAME",
    "PhoenixCheckpointNotFound",
    "PhoenixMetadata",
    "apply_phoenix_weights",
    "load_phoenix_checkpoint",
    "load_phoenix_encoder",
    "resolve_phoenix_checkpoint_path",
]

CHECKPOINT_FILENAME = "best.pth"
CHECKPOINT_ENV_VAR = "SLT_PHOENIX_CHECKPOINT"
_DEFAULT_DIR = Path("data") / "phoenix_2014"
_REPO_ROOT = Path(__file__).resolve().parents[2]


class PhoenixCheckpointNotFound(FileNotFoundError):
    """Raised when the Phoenix 2014 checkpoint cannot be located."""


@dataclass(frozen=True)
class PhoenixMetadata:
    """Metadata extracted from the Phoenix 2014 checkpoint."""

    config: Mapping[str, Any]
    info: Mapping[str, Any]


def resolve_phoenix_checkpoint_path(
    checkpoint_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> Path:
    """Resolve the Phoenix 2014 checkpoint path.

    The resolution order is:

    1. The explicit ``checkpoint_path`` argument when provided.
    2. The environment variable :envvar:`SLT_PHOENIX_CHECKPOINT`.
    3. ``data/phoenix_2014/best.pth`` relative to the current working directory.
    4. ``data/phoenix_2014/best.pth`` relative to the repository root.
    """

    candidates = []
    if checkpoint_path is not None:
        candidates.append(Path(checkpoint_path))
    env_path = os.getenv(CHECKPOINT_ENV_VAR)
    if env_path:
        candidates.append(Path(env_path))
    candidates.append(Path.cwd() / _DEFAULT_DIR / CHECKPOINT_FILENAME)
    candidates.append(_REPO_ROOT / _DEFAULT_DIR / CHECKPOINT_FILENAME)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise PhoenixCheckpointNotFound(
        "Unable to locate the Phoenix 2014 checkpoint. Provide a path explicitly, "
        f"set ${CHECKPOINT_ENV_VAR} or place {CHECKPOINT_FILENAME} under {_DEFAULT_DIR}."
    )


def _ensure_mapping(name: str, value: Any) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"Expected '{name}' to be a mapping, got {type(value)!r}")
    return value


def load_phoenix_checkpoint(
    *,
    checkpoint_path: Optional[Union[str, os.PathLike[str]]] = None,
    map_location: Optional[Union[str, torch.device]] = None,
) -> Tuple[PhoenixMetadata, MutableMapping[str, Any]]:
    """Load the Phoenix 2014 checkpoint returning metadata and tensors."""

    path = resolve_phoenix_checkpoint_path(checkpoint_path)
    payload: MutableMapping[str, Any] = torch.load(
        path, map_location=map_location, weights_only=False
    )

    config = _ensure_mapping("config", payload.get("config", {}))
    info: dict[str, Any] = {}
    for key in ("epoch", "val_loss", "best_val"):
        if key in payload:
            info[key] = payload[key]
    metadata = PhoenixMetadata(config=config, info=info)
    return metadata, payload


def _coerce_int(value: Any, *, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected integer-compatible value, received {value!r}") from exc


def _coerce_float(value: Any, *, default: float) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected float-compatible value, received {value!r}") from exc


def _build_mska_encoder(model_cfg: Mapping[str, Any], *, embed_dim: int) -> Optional[MSKAEncoder]:
    if not model_cfg.get("use_mska"):
        return None

    vocab_candidate = model_cfg.get("mska_ctc_vocab")
    if vocab_candidate in {None, 0, ""}:
        vocab_candidate = model_cfg.get("decoder_vocab_size") or model_cfg.get("vocab_size")
    vocab_size = max(2, _coerce_int(vocab_candidate, default=32_000))

    return MSKAEncoder(
        input_dim=_coerce_int(model_cfg.get("mska_input_dim"), default=3),
        embed_dim=embed_dim,
        stream_names=("face", "hand_left", "hand_right", "pose"),
        num_heads=_coerce_int(model_cfg.get("mska_heads"), default=4),
        ff_multiplier=_coerce_int(model_cfg.get("mska_ff_multiplier"), default=4),
        dropout=_coerce_float(model_cfg.get("mska_dropout"), default=0.1),
        ctc_vocab_size=vocab_size,
        detach_teacher=bool(model_cfg.get("mska_detach_teacher", True)),
        stream_attention_heads=_coerce_int(model_cfg.get("mska_stream_heads"), default=4),
        stream_temporal_blocks=_coerce_int(model_cfg.get("mska_temporal_blocks"), default=2),
        stream_temporal_kernel=_coerce_int(model_cfg.get("mska_temporal_kernel"), default=3),
        stream_temporal_dilation=_coerce_int(model_cfg.get("mska_temporal_dilation"), default=1),
        use_global_attention=bool(model_cfg.get("mska_use_sgr", False)),
        global_attention_activation=str(
            model_cfg.get("mska_sgr_activation", "softmax")
        ).lower(),
        global_attention_mix=_coerce_float(model_cfg.get("mska_sgr_mix"), default=0.5),
        global_attention_shared=bool(model_cfg.get("mska_sgr_shared", False)),
        leaky_relu_negative_slope=_coerce_float(
            model_cfg.get("leaky_relu_negative_slope"), default=0.01
        ),
    )


def load_phoenix_encoder(
    *,
    checkpoint_path: Optional[Union[str, os.PathLike[str]]] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> Tuple[MultiStreamEncoder, PhoenixMetadata, MutableMapping[str, Any]]:
    """Instantiate the encoder with Phoenix 2014 weights."""

    metadata, payload = load_phoenix_checkpoint(
        checkpoint_path=checkpoint_path, map_location=map_location
    )

    model_cfg = _ensure_mapping("config.model", metadata.config.get("model", {}))
    projector_dim = _coerce_int(model_cfg.get("projector_dim"), default=256)
    temporal_kwargs = {
        "nhead": _coerce_int(model_cfg.get("temporal_nhead"), default=4),
        "nlayers": _coerce_int(model_cfg.get("temporal_layers"), default=3),
        "dim_feedforward": _coerce_int(model_cfg.get("temporal_dim_feedforward"), default=384),
        "dropout": _coerce_float(model_cfg.get("temporal_dropout"), default=0.05),
    }
    encoder = MultiStreamEncoder(
        backbone_config=ViTConfig(image_size=_coerce_int(model_cfg.get("image_size"), default=224)),
        projector_dim=projector_dim,
        d_model=_coerce_int(model_cfg.get("d_model"), default=512),
        pose_dim=3 * _coerce_int(model_cfg.get("pose_landmarks"), default=13),
        positional_num_positions=_coerce_int(model_cfg.get("sequence_length"), default=128),
        projector_dropout=_coerce_float(model_cfg.get("projector_dropout"), default=0.05),
        fusion_dropout=_coerce_float(model_cfg.get("fusion_dropout"), default=0.05),
        leaky_relu_negative_slope=_coerce_float(
            model_cfg.get("leaky_relu_negative_slope"), default=0.01
        ),
        temporal_kwargs=temporal_kwargs,
        mska=_build_mska_encoder(model_cfg, embed_dim=projector_dim),
        mska_gloss_hidden_dim=model_cfg.get("mska_gloss_hidden_dim"),
        mska_gloss_second_hidden_dim=model_cfg.get("mska_gloss_second_hidden_dim"),
        mska_gloss_activation=str(model_cfg.get("mska_gloss_activation", "leaky_relu")),
        mska_gloss_dropout=_coerce_float(model_cfg.get("mska_gloss_dropout"), default=0.0),
    )

    encoder_state = payload.get("encoder_state")
    if not isinstance(encoder_state, Mapping):
        raise ValueError("Phoenix checkpoint missing 'encoder_state' mapping")
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(
            "Phoenix checkpoint is incompatible with the current encoder architecture: "
            f"missing keys={missing}, unexpected keys={unexpected}"
        )

    mska_state = payload.get("mska_state")
    if isinstance(mska_state, Mapping) and encoder.mska_encoder is not None:
        encoder.mska_encoder.load_state_dict(mska_state, strict=strict)

    return encoder, metadata, payload


def apply_phoenix_weights(
    encoder: MultiStreamEncoder,
    *,
    decoder: Optional[TextSeq2SeqDecoder] = None,
    checkpoint_path: Optional[Union[str, os.PathLike[str]]] = None,
    map_location: Optional[Union[str, torch.device]] = None,
    strict: bool = True,
) -> PhoenixMetadata:
    """Load Phoenix weights into the provided encoder/decoder pair."""

    metadata, payload = load_phoenix_checkpoint(
        checkpoint_path=checkpoint_path, map_location=map_location
    )

    encoder_state = payload.get("encoder_state")
    if not isinstance(encoder_state, Mapping):
        raise ValueError("Phoenix checkpoint missing 'encoder_state' mapping")
    missing, unexpected = encoder.load_state_dict(encoder_state, strict=strict)
    if strict and (missing or unexpected):
        raise RuntimeError(
            "Phoenix checkpoint is incompatible with the provided encoder: "
            f"missing keys={missing}, unexpected keys={unexpected}"
        )

    mska_state = payload.get("mska_state")
    if isinstance(mska_state, Mapping) and encoder.mska_encoder is not None:
        encoder.mska_encoder.load_state_dict(mska_state, strict=strict)

    if decoder is not None:
        decoder_state = payload.get("decoder_state")
        if isinstance(decoder_state, Mapping):
            missing_dec, unexpected_dec = decoder.load_state_dict(decoder_state, strict=strict)
            if strict and (missing_dec or unexpected_dec):
                raise RuntimeError(
                    "Phoenix checkpoint is incompatible with the provided decoder: "
                    f"missing keys={missing_dec}, unexpected keys={unexpected_dec}"
                )

    return metadata
