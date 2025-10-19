#!/usr/bin/env python3
"""Export the multi-stream encoder stub and heads to ONNX/TorchScript."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from pathlib import Path
from typing import Mapping, Optional, Sequence, Tuple

import torch
from torch.onnx import OperatorExportTypes

from slt.models import MultiStreamEncoder, ViTConfig


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to the PyTorch checkpoint")
    parser.add_argument("--onnx", type=Path, help="Destination ONNX file")
    parser.add_argument("--torchscript", type=Path, help="Destination TorchScript file")
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


class EncoderExportModule(torch.nn.Module):
    """Wrapper exposing the encoder output and intermediate heads."""

    def __init__(self, encoder: MultiStreamEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        face: torch.Tensor,
        hand_l: torch.Tensor,
        hand_r: torch.Tensor,
        pose: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        miss_mask_hl: Optional[torch.Tensor] = None,
        miss_mask_hr: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, ...]:
        face_feats = self.encoder._encode_backbone(self.encoder.face_backbone, face)
        hand_l_feats = self.encoder._encode_backbone(self.encoder.hand_backbone_left, hand_l)
        hand_r_feats = self.encoder._encode_backbone(self.encoder.hand_backbone_right, hand_r)

        hand_l_feats, hand_l_mask = self.encoder._apply_missing_mask(
            hand_l_feats, miss_mask_hl, stream="hand_left"
        )
        hand_r_feats, hand_r_mask = self.encoder._apply_missing_mask(
            hand_r_feats, miss_mask_hr, stream="hand_right"
        )

        face_proj = self.encoder.face_projector(face_feats)
        hand_l_proj = self.encoder.hand_left_projector(hand_l_feats)
        hand_r_proj = self.encoder.hand_right_projector(hand_r_feats)
        pose_proj = self.encoder.pose_projector(self.encoder._ensure_pose_shape(pose))

        combined_hand_mask = self.encoder._combine_missing_masks(hand_l_mask, hand_r_mask)
        fused = self.encoder._call_fusion(
            (face_proj, hand_l_proj, hand_r_proj, pose_proj), combined_hand_mask
        )
        fused = self.encoder.positional(fused)

        src_key_padding_mask = self.encoder._convert_padding_mask(pad_mask)
        encoded = self.encoder.temporal(
            fused, src_key_padding_mask=src_key_padding_mask
        )

        device = encoded.device
        batch, seq_len = encoded.shape[:2]

        if combined_hand_mask is None:
            combined_hand_mask = torch.zeros(batch, seq_len, device=device, dtype=torch.float32)
        else:
            combined_hand_mask = combined_hand_mask.to(device=device, dtype=torch.float32)

        if src_key_padding_mask is None:
            padding_mask = torch.zeros(batch, seq_len, device=device, dtype=torch.float32)
        else:
            padding_mask = src_key_padding_mask.to(device=device, dtype=torch.float32)

        return (
            encoded,
            face_proj,
            hand_l_proj,
            hand_r_proj,
            pose_proj,
            combined_hand_mask,
            padding_mask,
        )


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

    if args.onnx is None and args.torchscript is None:
        raise ValueError("At least one output (--onnx or --torchscript) must be provided")

    encoder = _build_encoder(args).to(device)
    _load_encoder_weights(args.checkpoint, encoder)
    encoder.eval()

    export_module = EncoderExportModule(encoder).to(device)
    inputs = _dummy_inputs(args, device)
    positional_args = inputs[:4]
    keyword_args = {
        "pad_mask": inputs[4],
        "miss_mask_hl": inputs[5],
        "miss_mask_hr": inputs[6],
    }

    if args.onnx is not None:
        args.onnx.parent.mkdir(parents=True, exist_ok=True)
        dynamic_axes = {
            "face": {1: "T"},
            "hand_l": {1: "T"},
            "hand_r": {1: "T"},
            "pose": {1: "T"},
            "pad_mask": {1: "T"},
            "miss_mask_hl": {1: "T"},
            "miss_mask_hr": {1: "T"},
            "encoded": {1: "T"},
            "face_head": {1: "T"},
            "hand_left_head": {1: "T"},
            "hand_right_head": {1: "T"},
            "pose_head": {1: "T"},
            "hand_mask": {1: "T"},
            "padding_mask": {1: "T"},
        }
        with torch.no_grad():
            torch.onnx.export(
                export_module,
                positional_args,
                args.onnx,
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
                output_names=[
                    "encoded",
                    "face_head",
                    "hand_left_head",
                    "hand_right_head",
                    "pose_head",
                    "hand_mask",
                    "padding_mask",
                ],
                dynamic_axes=dynamic_axes,
                opset_version=args.opset,
                fallback=True,
                operator_export_type=OperatorExportTypes.ONNX_FALLTHROUGH,
            )

    if args.torchscript is not None:
        args.torchscript.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            traced = torch.jit.trace(
                export_module,
                positional_args,
                check_trace=False,
                strict=False,
                example_kwarg_inputs=keyword_args,
            )
        traced.save(str(args.torchscript))


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main_export()
