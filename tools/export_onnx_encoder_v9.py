#!/usr/bin/env python3
"""Export the multi-stream encoder stub to ONNX format."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Optional, Sequence

import torch
from torch.onnx import OperatorExportTypes

from slt.models import MultiStreamEncoder, ViTConfig


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the PyTorch checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="Destination ONNX file")
    parser.add_argument("--image-size", type=int, default=224, help="Input image resolution expected by the ViT backbones")
    parser.add_argument("--projector-dim", type=int, default=256, help="Dimensionality of the per-stream projectors")
    parser.add_argument("--d-model", type=int, default=512, help="Temporal encoder embedding dimension")
    parser.add_argument("--pose-landmarks", type=int, default=13, help="Number of pose landmarks present in the pose stream")
    parser.add_argument("--sequence-length", type=int, default=128, help="Temporal length used to build the dummy inputs")
    parser.add_argument("--projector-dropout", type=float, default=0.0, help="Dropout applied inside the projectors")
    parser.add_argument("--fusion-dropout", type=float, default=0.0, help="Dropout applied before stream fusion")
    parser.add_argument("--temporal-nhead", type=int, default=8, help="Number of attention heads in the temporal encoder")
    parser.add_argument("--temporal-layers", type=int, default=6, help="Number of transformer layers in the temporal encoder")
    parser.add_argument(
        "--temporal-dim-feedforward",
        type=int,
        default=2048,
        help="Feed-forward dimension inside the temporal encoder",
    )
    parser.add_argument("--temporal-dropout", type=float, default=0.1, help="Dropout probability in the temporal encoder")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device identifier used during export")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    return parser.parse_args(argv)


def _build_encoder(args: argparse.Namespace) -> MultiStreamEncoder:
    vit_config = ViTConfig(image_size=args.image_size)
    temporal_kwargs = {
        "nhead": args.temporal_nhead,
        "nlayers": args.temporal_layers,
        "dim_feedforward": args.temporal_dim_feedforward,
        "dropout": args.temporal_dropout,
    }
    encoder = MultiStreamEncoder(
        backbone_config=vit_config,
        projector_dim=args.projector_dim,
        d_model=args.d_model,
        pose_dim=3 * args.pose_landmarks,
        positional_num_positions=args.sequence_length,
        projector_dropout=args.projector_dropout,
        fusion_dropout=args.fusion_dropout,
        temporal_kwargs=temporal_kwargs,
    )
    return encoder


def _select_state_dict(state: Mapping[str, torch.Tensor], model: MultiStreamEncoder) -> OrderedDict[str, torch.Tensor]:
    expected_keys = set(model.state_dict().keys())
    provided_keys = set(state.keys())

    if provided_keys == expected_keys:
        return OrderedDict(state)

    prefixes = ["encoder.", "module.encoder."]
    for prefix in prefixes:
        filtered = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        if set(filtered.keys()) == expected_keys:
            return OrderedDict(filtered)

    raise RuntimeError(
        "Checkpoint does not contain weights compatible with MultiStreamEncoder. "
        "Expected keys matching encoder parameters."
    )


def _load_encoder_weights(path: Path, model: MultiStreamEncoder) -> None:
    checkpoint = torch.load(path, map_location="cpu")

    if isinstance(checkpoint, Mapping):
        for candidate in ("encoder_state", "model_state", "state_dict"):
            if candidate in checkpoint:
                checkpoint = checkpoint[candidate]
                break

    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Checkpoint must be a mapping containing model parameters")

    state_dict = _select_state_dict(checkpoint, model)
    model.load_state_dict(state_dict)


def _dummy_inputs(args: argparse.Namespace, device: torch.device) -> tuple:
    batch = 1
    seq = args.sequence_length
    image_size = args.image_size
    pose_dim = 3 * args.pose_landmarks

    face = torch.randn(batch, seq, 3, image_size, image_size, device=device)
    hand_l = torch.randn(batch, seq, 3, image_size, image_size, device=device)
    hand_r = torch.randn(batch, seq, 3, image_size, image_size, device=device)
    pose = torch.randn(batch, seq, pose_dim, device=device)
    pad_mask = torch.ones(batch, seq, dtype=torch.bool, device=device)
    miss_mask_hl = torch.zeros(batch, seq, dtype=torch.bool, device=device)
    miss_mask_hr = torch.zeros(batch, seq, dtype=torch.bool, device=device)

    return face, hand_l, hand_r, pose, pad_mask, miss_mask_hl, miss_mask_hr


def main_export(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    device = torch.device(args.device)
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")

    encoder = _build_encoder(args).to(device)
    _load_encoder_weights(args.checkpoint, encoder)
    encoder.eval()

    out_path = args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)

    inputs = _dummy_inputs(args, device)
    positional_args = inputs[:4]
    keyword_args = {
        "pad_mask": inputs[4],
        "miss_mask_hl": inputs[5],
        "miss_mask_hr": inputs[6],
    }

    dynamic_axes = {
        "face": {1: "T"},
        "hand_l": {1: "T"},
        "hand_r": {1: "T"},
        "pose": {1: "T"},
        "pad_mask": {1: "T"},
    }

    with torch.no_grad():
        torch.onnx.export(
            encoder,
            positional_args,
            out_path,
            kwargs=keyword_args,
            input_names=[
                "face",
                "hand_l",
                "hand_r",
                "pose",
                "pad_mask",
                "miss_mask_hl",
                "miss_mask_hr",
            ],
            output_names=["encoded"],
            dynamic_axes=dynamic_axes,
            opset_version=args.opset,
            fallback=True,
            operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
        )


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main_export()
