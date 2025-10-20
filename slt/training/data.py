"""Data pipeline helpers shared across CLIs."""
from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional

import torch
from torch.utils.data import DataLoader, get_worker_info
from transformers import PreTrainedTokenizerBase

from slt.data import LsaTMultiStream, collate_fn
from slt.utils.text import encode_batch


_CANONICAL_STREAMS = {
    "face": ("face",),
    "hand_left": ("hand_l", "miss_mask_hl"),
    "hand_right": ("hand_r", "miss_mask_hr"),
    "pose": ("pose", "pose_conf_mask"),
}

_STREAM_ALIASES = {
    "hand-left": "hand_left",
    "hand-right": "hand_right",
    "hand_l": "hand_left",
    "hand_r": "hand_right",
    "hl": "hand_left",
    "hr": "hand_right",
}


def _canonical_stream_name(stream: str) -> Optional[str]:
    stream = stream.strip().lower()
    if stream in _CANONICAL_STREAMS:
        return stream
    return _STREAM_ALIASES.get(stream)


def normalise_mix_spec(spec: Mapping[str, float]) -> Dict[str, float]:
    """Normalise a stream mix specification using canonical keys."""

    result: Dict[str, float] = {}
    for raw_name, prob in spec.items():
        canonical = _canonical_stream_name(raw_name)
        if canonical is None:
            raise ValueError(f"Unknown stream '{raw_name}' in mix specification")
        if prob <= 0:
            continue
        if prob > 1:
            raise ValueError(f"Mixing probability must be <= 1 for stream '{raw_name}'")
        result[canonical] = float(prob)
    return result


def _apply_stream_mixing(
    merged: MutableMapping[str, Any],
    mix_streams: Mapping[str, float],
    *,
    generator: torch.Generator,
) -> None:
    if not mix_streams:
        return
    batch_size = None
    for key in ("face", "hand_l", "hand_r", "pose"):
        tensor = merged.get(key)
        if isinstance(tensor, torch.Tensor):
            batch_size = tensor.shape[0]
            break
    if batch_size is None or batch_size <= 1:
        return

    for stream, prob in mix_streams.items():
        group = _CANONICAL_STREAMS.get(stream)
        if not group:
            continue
        if prob < 1.0 and torch.rand((), generator=generator).item() > prob:
            continue
        permutation = torch.randperm(batch_size, generator=generator)
        for key in group:
            tensor = merged.get(key)
            if isinstance(tensor, torch.Tensor) and tensor.shape[0] == batch_size:
                merged[key] = tensor[permutation]


def build_collate(
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    mix_streams: Optional[Mapping[str, float]] = None,
    seed: Optional[int] = None,
):
    """Create a ``collate_fn`` wiring tokenisation and optional stream mixing."""

    mix_streams = normalise_mix_spec(mix_streams or {})
    base_generator = torch.Generator()
    if seed is not None:
        base_generator.manual_seed(seed)
    else:  # pragma: no cover - non deterministic branch
        base_generator.seed()
    worker_generators: Dict[int, torch.Generator] = {}

    def _resolve_generator() -> torch.Generator:
        info = get_worker_info()
        if info is None:
            return base_generator
        worker_id = info.id
        generator = worker_generators.get(worker_id)
        if generator is None:
            generator = torch.Generator()
            generator.manual_seed(info.seed)
            worker_generators[worker_id] = generator
        return generator

    def _collate(batch: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
        merged = collate_fn(batch)
        generator = _resolve_generator()
        _apply_stream_mixing(merged, mix_streams, generator=generator)
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
            "pose_conf_mask": merged["pose_conf_mask"],
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


def create_dataloader(
    dataset: LsaTMultiStream,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
    mix_streams: Optional[Mapping[str, float]] = None,
    seed: Optional[int] = None,
) -> DataLoader:
    """Instantiate a :class:`~torch.utils.data.DataLoader` for SLT datasets."""

    collate = build_collate(
        tokenizer,
        max_length=max_length,
        mix_streams=mix_streams,
        seed=seed,
    )
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
        generator=generator,
    )

