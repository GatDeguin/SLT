"""Helpers to load the validated single-signer multi-stream model.

The checkpoint is not bundled with the repository to avoid shipping large
binary assets. Download ``single_signer_multistream.pt`` separately, place it
under ``data/single_signer/`` or expose its path through the environment
variable :envvar:`SLT_SINGLE_SIGNER_CHECKPOINT`. The helpers in this module will
search those locations unless an explicit ``checkpoint_path`` is provided.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional, Tuple, Union

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


class SingleSignerCheckpointNotFound(FileNotFoundError):
    """Raised when the single-signer checkpoint cannot be located."""


def resolve_single_signer_checkpoint_path(
    checkpoint_path: Optional[Union[str, os.PathLike[str]]] = None,
) -> Path:
    """Resolve the checkpoint path searching common locations.

    The order of precedence is:

    1. The explicit ``checkpoint_path`` argument when provided.
    2. The environment variable :envvar:`SLT_SINGLE_SIGNER_CHECKPOINT`.
    3. ``data/single_signer/single_signer_multistream.pt`` relative to the
       current working directory.
    4. ``data/single_signer/single_signer_multistream.pt`` relative to the
       repository root (a fallback for callers executed from other paths).

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
    candidates.append(Path.cwd() / "data" / "single_signer" / CHECKPOINT_FILENAME)
    candidates.append(_REPO_ROOT / "data" / "single_signer" / CHECKPOINT_FILENAME)

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


def load_single_signer_checkpoint(
    *,
    checkpoint_path: Optional[Union[str, os.PathLike[str]]] = None,
    map_location: Optional[torch.device] = None,
) -> Tuple[SingleSignerMetadata, MutableMapping[str, Any]]:
    """Load the checkpoint returning metadata and tensors."""

    path = resolve_single_signer_checkpoint_path(checkpoint_path)
    checkpoint: MutableMapping[str, Any] = torch.load(path, map_location=map_location)

    metadata = SingleSignerMetadata(
        schema_version=str(checkpoint.get("schema_version", "1.0")),
        task=str(checkpoint.get("task", "single_signer")),
        encoder_kwargs=checkpoint.get("encoder", {}).get("init_kwargs", {}),
        decoder_kwargs=checkpoint.get("decoder", {}).get("init_kwargs", {}),
        backbone_kwargs=checkpoint.get("encoder", {}).get("backbone_kwargs", {}),
        tokenizer_info=checkpoint.get("tokenizer", {}),
        extra=checkpoint.get("metadata", {}),
    )
    return metadata, checkpoint


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
    encoder_blob = checkpoint.get("encoder", {})
    encoder_kwargs = dict(metadata.encoder_kwargs)
    backbone_kwargs = dict(metadata.backbone_kwargs)
    backbones = build_single_signer_backbones(**backbone_kwargs)
    encoder = MultiStreamEncoder(backbones=backbones, **encoder_kwargs)
    encoder_state = encoder_blob.get("state_dict", {})
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
    decoder_blob = checkpoint.get("decoder", {})

    decoder_kwargs = dict(metadata.decoder_kwargs)
    _patch_decoder_kwargs(decoder_kwargs, tokenizer, metadata.tokenizer_info)
    decoder = TextSeq2SeqDecoder(**decoder_kwargs)
    decoder_state = decoder_blob.get("state_dict", {})
    decoder.load_state_dict(decoder_state, strict=strict)

    return encoder, decoder, metadata
