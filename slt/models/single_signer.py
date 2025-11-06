"""Helpers to load the validated single-signer multi-stream model.

The checkpoint is not bundled with the repository to avoid shipping large
binary assets. Download ``single_signer_multistream.pt`` separately, place it
under ``data/single_signer/`` or expose its path through the environment
variable :envvar:`SLT_SINGLE_SIGNER_CHECKPOINT`. The helpers in this module will
search those locations unless an explicit ``checkpoint_path`` is provided.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple, Union, cast

import torch
from torch import nn

from .multistream import MultiStreamEncoder
from .temporal import TextSeq2SeqDecoder

__all__ = [
    "CHECKPOINT_FILENAME",
    "CHECKPOINT_ENV_VAR",
    "SingleSignerCheckpointNotFound",
    "TinyConvBackbone",
    "resolve_single_signer_checkpoint_path",
    "build_single_signer_backbones",
    "load_single_signer_checkpoint",
    "load_single_signer_encoder",
    "load_single_signer_components",
]

CHECKPOINT_FILENAME = "single_signer_multistream.pt"
CHECKPOINT_ENV_VAR = "SLT_SINGLE_SIGNER_CHECKPOINT"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_KNOWN_CHECKPOINT_NAMES = (
    CHECKPOINT_FILENAME,
    "single_signer_multistream.pth",
    "best.pt",
    "best.pth",
)


class SingleSignerCheckpointNotFound(FileNotFoundError):
    """Raised when the single-signer checkpoint cannot be located."""


def resolve_single_signer_checkpoint_path(
    checkpoint_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> Path:
    """Resolve the checkpoint path searching common locations.

    The order of precedence is:

    1. The explicit ``checkpoint_path`` argument when provided.
    2. The environment variable :envvar:`SLT_SINGLE_SIGNER_CHECKPOINT`.
    3. ``data/single_signer/`` relative to the current working directory.
    4. ``data/single_signer/`` relative to the repository root (a fallback for
       callers executed from other paths).
    5. The current working directory.
    6. The repository root directory.

    Within each directory the function recognises the validated checkpoint name
    as well as common alternatives (``single_signer_multistream.pth``,
    ``best.pt`` and ``best.pth``).

    Parameters
    ----------
    checkpoint_path:
        Optional path provided by the caller.

    Returns
    -------
    pathlib.Path
        The first existing path that matches the search order.

    Raises
    ------
    SingleSignerCheckpointNotFound
        If no existing path is found.
    """

    candidates = []
    if checkpoint_path is not None:
        candidates.append(Path(checkpoint_path))
    env_path = os.getenv(CHECKPOINT_ENV_VAR)
    if env_path:
        candidates.append(Path(env_path))
    search_directories = [
        Path.cwd() / "data" / "single_signer",
        _REPO_ROOT / "data" / "single_signer",
        Path.cwd(),
        _REPO_ROOT,
    ]
    seen: set[Path] = {candidate for candidate in candidates}
    for directory in search_directories:
        for filename in _KNOWN_CHECKPOINT_NAMES:
            candidate = directory / filename
            if candidate not in seen:
                candidates.append(candidate)
                seen.add(candidate)

    for candidate in candidates:
        if candidate.exists():
            return candidate

    message = (
        "Unable to locate the single-signer checkpoint. Provide a path explicitly, "
        f"set ${CHECKPOINT_ENV_VAR} or place {CHECKPOINT_FILENAME} under "
        "data/single_signer/."
    )
    raise SingleSignerCheckpointNotFound(message)


@dataclass(frozen=True)
class SingleSignerMetadata:
    """Metadata stored alongside the checkpoint."""

    schema_version: str
    task: str
    encoder_kwargs: Mapping[str, Any]
    decoder_kwargs: Mapping[str, Any]
    backbone_kwargs: Mapping[str, Any]
    tokenizer_info: Mapping[str, Any]
    extra: Mapping[str, Any]


class TinyConvBackbone(nn.Module):
    """Lightweight convolutional encoder used in the single-signer weights."""

    def __init__(
        self,
        *,
        in_channels: int = 3,
        base_channels: int = 32,
        features: int = 192,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 2),
            nn.GELU(),
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels * 4),
            nn.GELU(),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Linear(base_channels * 4, features)
        self.num_features = features

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        if frames.dim() != 4:
            raise ValueError(
                "TinyConvBackbone expects a 4D tensor (batch, channels, height, width)."
            )
        encoded = self.stem(frames)
        pooled = self.pool(encoded).flatten(1)
        pooled = self.dropout(pooled)
        return self.projection(pooled)


def build_single_signer_backbones(**kwargs: Any) -> Dict[str, TinyConvBackbone]:
    """Return freshly initialised convolutional backbones for each stream."""

    common_kwargs = dict(kwargs)
    return {
        "face": TinyConvBackbone(**common_kwargs),
        "hand_left": TinyConvBackbone(**common_kwargs),
        "hand_right": TinyConvBackbone(**common_kwargs),
    }


def _coerce_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        if isinstance(value, bool):
            return int(value)
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _extract_model_config(checkpoint: Mapping[str, Any]) -> Mapping[str, Any]:
    config = checkpoint.get("config")
    if not isinstance(config, Mapping):
        return {}
    model_cfg = config.get("model")
    if isinstance(model_cfg, Mapping):
        return model_cfg
    return {}


def _encoder_kwargs_from_config(model_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    explicit = model_cfg.get("encoder_kwargs")
    if isinstance(explicit, Mapping):
        return dict(explicit)

    kwargs: Dict[str, Any] = {}
    projector_dim = _coerce_int(model_cfg.get("projector_dim"))
    if projector_dim is not None:
        kwargs["projector_dim"] = projector_dim
    d_model = _coerce_int(model_cfg.get("d_model"))
    if d_model is not None:
        kwargs["d_model"] = d_model
    pose_dim = _coerce_int(model_cfg.get("pose_dim"))
    if pose_dim is None:
        pose_landmarks = _coerce_int(model_cfg.get("pose_landmarks"))
        if pose_landmarks is not None:
            pose_dim = pose_landmarks * 3
    if pose_dim is not None:
        kwargs["pose_dim"] = pose_dim
    positions = _coerce_int(model_cfg.get("sequence_length"))
    if positions is not None:
        kwargs["positional_num_positions"] = positions
    projector_dropout = _coerce_float(model_cfg.get("projector_dropout"))
    if projector_dropout is not None:
        kwargs["projector_dropout"] = projector_dropout
    fusion_dropout = _coerce_float(model_cfg.get("fusion_dropout"))
    if fusion_dropout is not None:
        kwargs["fusion_dropout"] = fusion_dropout
    negative_slope = _coerce_float(model_cfg.get("leaky_relu_negative_slope"))
    if negative_slope is not None:
        kwargs["leaky_relu_negative_slope"] = negative_slope

    temporal_kwargs: Dict[str, Any] = {}
    nhead = _coerce_int(model_cfg.get("temporal_nhead"))
    if nhead is not None:
        temporal_kwargs["nhead"] = nhead
    nlayers = _coerce_int(model_cfg.get("temporal_layers"))
    if nlayers is not None:
        temporal_kwargs["nlayers"] = nlayers
    dim_feedforward = _coerce_int(model_cfg.get("temporal_dim_feedforward"))
    if dim_feedforward is not None:
        temporal_kwargs["dim_feedforward"] = dim_feedforward
    temporal_dropout = _coerce_float(model_cfg.get("temporal_dropout"))
    if temporal_dropout is not None:
        temporal_kwargs["dropout"] = temporal_dropout
    if temporal_kwargs:
        kwargs["temporal_kwargs"] = temporal_kwargs

    for key in (
        "mska_gloss_hidden_dim",
        "mska_gloss_second_hidden_dim",
        "mska_gloss_activation",
        "mska_gloss_dropout",
    ):
        value = model_cfg.get(key)
        if value is not None:
            kwargs[key] = value

    return kwargs


def _backbone_kwargs_from_config(
    model_cfg: Mapping[str, Any], encoder_kwargs: Mapping[str, Any]
) -> Dict[str, Any]:
    explicit = model_cfg.get("encoder_backbone_kwargs")
    if isinstance(explicit, Mapping):
        return dict(explicit)

    backbone_kwargs: Dict[str, Any] = {}
    in_channels = _coerce_int(model_cfg.get("backbone_in_channels"))
    if in_channels is not None:
        backbone_kwargs["in_channels"] = in_channels
    base_channels = _coerce_int(model_cfg.get("backbone_base_channels"))
    if base_channels is not None:
        backbone_kwargs["base_channels"] = base_channels
    features = _coerce_int(model_cfg.get("backbone_features"))
    if features is None:
        projector_dim = encoder_kwargs.get("projector_dim")
        if isinstance(projector_dim, int):
            features = projector_dim
    if features is not None:
        backbone_kwargs["features"] = features
    dropout = _coerce_float(model_cfg.get("backbone_dropout"))
    if dropout is not None:
        backbone_kwargs["dropout"] = dropout
    return backbone_kwargs


def _decoder_kwargs_from_config(model_cfg: Mapping[str, Any]) -> Dict[str, Any]:
    explicit = model_cfg.get("decoder_kwargs")
    base: Dict[str, Any] = dict(explicit) if isinstance(explicit, Mapping) else {}

    d_model = _coerce_int(model_cfg.get("d_model"))
    if d_model is not None:
        base.setdefault("d_model", d_model)
    num_layers = _coerce_int(model_cfg.get("decoder_layers"))
    if num_layers is not None:
        base.setdefault("num_layers", num_layers)
    num_heads = _coerce_int(model_cfg.get("decoder_heads"))
    if num_heads is not None:
        base.setdefault("num_heads", num_heads)
    dropout = _coerce_float(model_cfg.get("decoder_dropout"))
    if dropout is not None:
        base.setdefault("dropout", dropout)
    prompt_length = _coerce_int(model_cfg.get("decoder_prompt_length"))
    if prompt_length is not None:
        base.setdefault("prompt_length", prompt_length)
    for source, target in (
        ("decoder_prompt_init", "prompt_init"),
        ("decoder_prompt_std", "prompt_init_std"),
        ("decoder_prompt_range", "prompt_init_range"),
        ("decoder_prompt_tokens", "prompt_init_tokens"),
    ):
        value = model_cfg.get(source)
        if value is not None:
            base.setdefault(target, value)
    local_only = model_cfg.get("decoder_local_files_only")
    if isinstance(local_only, bool):
        base.setdefault("local_files_only", local_only)
    for source, target in (
        ("decoder_search_paths", "local_paths"),
        ("decoder_path_env_vars", "env_var_paths"),
    ):
        value = model_cfg.get(source)
        if value:
            base.setdefault(target, value)
    decoder_model = model_cfg.get("decoder_model")
    if decoder_model:
        base.setdefault("pretrained_model_name_or_path", decoder_model)
    return base


def load_single_signer_checkpoint(
    *,
    checkpoint_path: Optional[Union[str, os.PathLike[str]]] = None,
    map_location: Optional[torch.device] = None,
) -> Tuple[SingleSignerMetadata, MutableMapping[str, Any]]:
    """Load the checkpoint returning metadata and tensors."""

    path = resolve_single_signer_checkpoint_path(checkpoint_path)
    checkpoint: MutableMapping[str, Any] = torch.load(path, map_location=map_location)

    encoder_blob = checkpoint.get("encoder", {})
    decoder_blob = checkpoint.get("decoder", {})

    encoder_kwargs = {}
    backbone_kwargs = {}
    decoder_kwargs = {}
    if isinstance(encoder_blob, Mapping):
        encoder_kwargs = dict(encoder_blob.get("init_kwargs", {}))
        backbone_kwargs = dict(encoder_blob.get("backbone_kwargs", {}))
    encoder_kwargs = encoder_kwargs or checkpoint.get("encoder_kwargs", {})
    backbone_kwargs = backbone_kwargs or checkpoint.get("encoder_backbone_kwargs", {})
    if isinstance(decoder_blob, Mapping):
        decoder_kwargs = dict(decoder_blob.get("init_kwargs", {}))
    decoder_kwargs = decoder_kwargs or checkpoint.get("decoder_kwargs", {})

    model_cfg = _extract_model_config(checkpoint)
    if model_cfg:
        config_encoder_kwargs = _encoder_kwargs_from_config(model_cfg)
        if not encoder_kwargs:
            encoder_kwargs = config_encoder_kwargs
        config_backbone_kwargs = _backbone_kwargs_from_config(model_cfg, encoder_kwargs)
        if not backbone_kwargs:
            backbone_kwargs = config_backbone_kwargs
        config_decoder_kwargs = _decoder_kwargs_from_config(model_cfg)
        if not decoder_kwargs:
            decoder_kwargs = config_decoder_kwargs

    metadata = SingleSignerMetadata(
        schema_version=str(checkpoint.get("schema_version", "1.0")),
        task=str(checkpoint.get("task", "single_signer")),
        encoder_kwargs=encoder_kwargs,
        decoder_kwargs=decoder_kwargs,
        backbone_kwargs=backbone_kwargs,
        tokenizer_info=checkpoint.get("tokenizer", {}),
        extra=checkpoint.get("metadata", {}),
    )
    return metadata, checkpoint


def _resolve_state_dict(
    component: Any,
    *,
    name: str,
    checkpoint: Optional[Mapping[str, Any]] = None,
) -> Mapping[str, Any]:
    """Return a state_dict mapping from a checkpoint component."""

    state_dict = component.get("state_dict") if isinstance(component, Mapping) else None
    if isinstance(state_dict, Mapping):
        return state_dict

    if isinstance(component, Mapping):
        if state_dict is not None and not isinstance(state_dict, Mapping):
            raise TypeError(
                f"Checkpoint field '{name}' expected a mapping under 'state_dict' but "
                f"received {type(state_dict)!r}."
            )
        if component and all(isinstance(key, str) for key in component):
            sample_key = next(iter(component))
            if isinstance(sample_key, str) and "." in sample_key:
                warnings.warn(
                    (
                        f"Checkpoint field '{name}' does not define 'state_dict'. "
                        "Interpreting the mapping as a legacy raw state_dict."
                    ),
                    RuntimeWarning,
                    stacklevel=2,
                )
                return cast(Mapping[str, Any], component)

    if checkpoint is not None:
        fallback_keys = (
            f"{name}_state_dict",
            f"{name}_state",
            f"{name}_model_state_dict",
            f"{name}_model_state",
            "model_state_dict",
            "model_state",
        )
        for key in fallback_keys:
            payload = checkpoint.get(key)
            if not isinstance(payload, Mapping):
                continue
            if key.endswith("model_state_dict") or key.endswith("model_state"):
                extracted = _slice_model_state(payload, module=name)
                if extracted:
                    return extracted
            else:
                return cast(Mapping[str, Any], payload)

    return {}


def _slice_model_state(state: Mapping[str, Any], *, module: str) -> Mapping[str, Any]:
    prefixes = (f"{module}.", f"module.{module}.")
    for prefix in prefixes:
        sliced: Dict[str, Any] = {}
        for key, value in state.items():
            if isinstance(key, str) and key.startswith(prefix):
                sliced[key[len(prefix) :]] = value
        if sliced:
            return sliced

    token = f"{module}."
    sliced_fallback: Dict[str, Any] = {}
    for key, value in state.items():
        if not isinstance(key, str):
            continue
        index = key.find(token)
        if index != -1:
            sliced_fallback[key[index + len(token) :]] = value
    return sliced_fallback


def load_single_signer_encoder(
    *,
    checkpoint_path: Optional[Union[str, os.PathLike[str]]] = None,
    map_location: Optional[torch.device] = None,
    strict: bool = True,
) -> Tuple[MultiStreamEncoder, SingleSignerMetadata, MutableMapping[str, Any]]:
    """Instantiate the encoder with the validated single-signer weights."""

    metadata, checkpoint = load_single_signer_checkpoint(
        checkpoint_path=checkpoint_path, map_location=map_location
    )
    encoder_blob: Mapping[str, Any] = {}
    raw_encoder_blob = checkpoint.get("encoder", {})
    if isinstance(raw_encoder_blob, Mapping):
        encoder_blob = raw_encoder_blob
    encoder_kwargs = dict(metadata.encoder_kwargs)
    backbone_kwargs = dict(metadata.backbone_kwargs)
    backbones = build_single_signer_backbones(**backbone_kwargs)
    encoder = MultiStreamEncoder(backbones=backbones, **encoder_kwargs)
    encoder_state = _resolve_state_dict(encoder_blob, name="encoder", checkpoint=checkpoint)
    encoder.load_state_dict(encoder_state, strict=strict)
    return encoder, metadata, checkpoint


def _patch_decoder_kwargs(
    decoder_kwargs: Dict[str, Any],
    tokenizer: Optional[object],
    tokenizer_info: Mapping[str, Any],
) -> None:
    if tokenizer is None:
        decoder_kwargs.setdefault("pad_token_id", int(tokenizer_info.get("pad_token_id", 0)))
        decoder_kwargs.setdefault("eos_token_id", int(tokenizer_info.get("eos_token_id", 1)))
        return

    pad_id = getattr(tokenizer, "pad_token_id", None)
    eos_id = getattr(tokenizer, "eos_token_id", None)
    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is None and hasattr(tokenizer, "__len__"):
        try:
            vocab_size = len(tokenizer)  # type: ignore[arg-type]
        except TypeError:
            vocab_size = None

    if pad_id is None:
        pad_id = tokenizer_info.get("pad_token_id", 0)
    if eos_id is None:
        eos_id = tokenizer_info.get("eos_token_id", pad_id)
    decoder_kwargs["pad_token_id"] = int(pad_id)
    decoder_kwargs["eos_token_id"] = int(eos_id)

    if vocab_size is not None:
        decoder_kwargs["vocab_size"] = int(max(vocab_size, decoder_kwargs.get("vocab_size", 0)))


def load_single_signer_components(
    tokenizer: Optional[object] = None,
    *,
    checkpoint_path: Optional[Union[str, os.PathLike[str]]] = None,
    map_location: Optional[torch.device] = None,
    strict: bool = True,
) -> Tuple[MultiStreamEncoder, TextSeq2SeqDecoder, SingleSignerMetadata]:
    """Return encoder and decoder modules initialised with the validated weights."""

    encoder, metadata, checkpoint = load_single_signer_encoder(
        checkpoint_path=checkpoint_path, map_location=map_location, strict=strict
    )
    decoder_blob: Mapping[str, Any] = {}
    raw_decoder_blob = checkpoint.get("decoder", {})
    if isinstance(raw_decoder_blob, Mapping):
        decoder_blob = raw_decoder_blob

    decoder_kwargs = dict(metadata.decoder_kwargs)
    _patch_decoder_kwargs(decoder_kwargs, tokenizer, metadata.tokenizer_info)
    decoder = TextSeq2SeqDecoder(**decoder_kwargs)
    decoder_state = _resolve_state_dict(decoder_blob, name="decoder", checkpoint=checkpoint)
    decoder.load_state_dict(decoder_state, strict=strict)

    return encoder, decoder, metadata
