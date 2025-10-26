"""Utility helpers for model initialisation and weight loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional

import torch

from .mska import MSKAEncoder

__all__ = ["load_mska_encoder_state"]


def _load_payload(source: Any) -> tuple[dict[str, Any], Optional[Path]]:
    if source is None:
        raise ValueError("source must not be None")
    if isinstance(source, (str, Path)):
        path = Path(source).expanduser()
        obj = torch.load(path, map_location="cpu")
        if isinstance(obj, Mapping):
            return dict(obj), path
        raise TypeError(
            f"Checkpoint at {path} does not contain a mapping payload"
        )
    if isinstance(source, Mapping):
        return dict(source), None
    raise TypeError("Unsupported checkpoint source type")


def _extract_state_dict(payload: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key in ("state_dict", "model", "encoder"):
        value = payload.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    if all(isinstance(key, str) for key in payload):
        return dict(payload)
    raise ValueError("Unable to locate a state_dict within the checkpoint payload")


def load_mska_encoder_state(
    encoder: MSKAEncoder,
    *,
    face_checkpoint: Any | None = None,
    hand_checkpoint: Any | None = None,
    strict: bool = True,
) -> dict[str, torch.nn.modules.module._IncompatibleKeys]:
    """Load pretrained weights for ``encoder`` streams.

    The helper expects checkpoints produced by the DINO wrappers with
    ``--export-checkpoint`` enabled. Face checkpoints are applied to the
    ``"face"`` stream. Hand checkpoints are replicated to both
    ``"hand_left"`` and ``"hand_right"``.
    """

    results: dict[str, torch.nn.modules.module._IncompatibleKeys] = {}

    if face_checkpoint is not None and "face" in encoder.encoders:
        face_payload, _ = _load_payload(face_checkpoint)
        face_state = _extract_state_dict(face_payload)
        results["face"] = encoder.encoders["face"].load_state_dict(face_state, strict=strict)

    if hand_checkpoint is not None:
        hand_payload, _ = _load_payload(hand_checkpoint)
        streams = hand_payload.get("streams")
        if isinstance(streams, Mapping):
            left_state = streams.get("hand_left") or streams.get("left")
            right_state = streams.get("hand_right") or streams.get("right")
            base_state = streams.get("hand")
        else:
            left_state = right_state = base_state = None
        if left_state is None or not isinstance(left_state, Mapping):
            left_state = base_state
        if right_state is None or not isinstance(right_state, Mapping):
            right_state = base_state
        hand_state = _extract_state_dict(hand_payload) if base_state is None else dict(base_state)
        for stream_name, stream_state in {
            "hand_left": left_state,
            "hand_right": right_state,
        }.items():
            if stream_state is None:
                stream_state = hand_state
            if stream_name in encoder.encoders and isinstance(stream_state, Mapping):
                results[stream_name] = encoder.encoders[stream_name].load_state_dict(
                    dict(stream_state), strict=strict
                )

    return results
