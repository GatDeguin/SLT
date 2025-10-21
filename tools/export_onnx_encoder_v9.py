#!/usr/bin/env python3
"""Export the validated multi-stream encoder and heads to ONNX/TorchScript."""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import warnings
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch.onnx import OperatorExportTypes

from slt.models import MultiStreamEncoder, ViTConfig
from slt.models.single_signer import (
    load_single_signer_encoder,
    resolve_single_signer_checkpoint_path,
)

IMAGENET_STATS = {
    "mean": [0.485, 0.456, 0.406],
    "std": [0.229, 0.224, 0.225],
}
METADATA_VERSION = 1


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="single_signer",
        help=(
            "Path to the PyTorch checkpoint or 'single_signer' to use the validated preset "
            "(requires a downloaded file)"
        ),
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help=(
            "Path to the downloaded single_signer checkpoint used when --checkpoint"
            " is set to the preset"
        ),
    )
    parser.add_argument("--onnx", type=Path, help="Destination ONNX file")
    parser.add_argument("--torchscript", type=Path, help="Destination TorchScript file")
    parser.add_argument("--image-size", type=int, default=224, help="Input image resolution expected by the ViT backbones")
    parser.add_argument("--projector-dim", type=int, default=128, help="Dimensionality of the per-stream projectors")
    parser.add_argument("--d-model", type=int, default=128, help="Temporal encoder embedding dimension")
    parser.add_argument("--pose-landmarks", type=int, default=13, help="Number of pose landmarks present in the pose stream")
    parser.add_argument("--sequence-length", type=int, default=128, help="Temporal length used to build the dummy inputs")
    parser.add_argument("--projector-dropout", type=float, default=0.05, help="Dropout applied inside the projectors")
    parser.add_argument("--fusion-dropout", type=float, default=0.05, help="Dropout applied before stream fusion")
    parser.add_argument("--temporal-nhead", type=int, default=4, help="Number of attention heads in the temporal encoder")
    parser.add_argument("--temporal-layers", type=int, default=3, help="Number of transformer layers in the temporal encoder")
    parser.add_argument(
        "--temporal-dim-feedforward",
        type=int,
        default=2048,
        help="Feed-forward dimension inside the temporal encoder",
    )
    parser.add_argument("--temporal-dropout", type=float, default=0.1, help="Dropout probability in the temporal encoder")
    parser.add_argument("--device", type=str, default="cpu", help="Torch device identifier used during export")
    parser.add_argument("--opset", type=int, default=17, help="ONNX opset version")
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        help="Directory where artefacts (ONNX/TorchScript/metadata) will be stored",
    )
    parser.add_argument(
        "--version",
        type=str,
        default="v1",
        help="Identifier used to version the exported artefacts",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Optional destination JSON metadata file."
        " If not provided, it is derived from --artifact-dir/--version.",
    )
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
        pose_conf_mask: Optional[torch.Tensor] = None,
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
        pose_processed = self.encoder._ensure_pose_shape(pose)
        pose_processed, _ = self.encoder._apply_pose_mask(
            pose_processed, pose_conf_mask
        )
        pose_proj = self.encoder.pose_projector(pose_processed)

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
    preset = (getattr(args, "checkpoint", "") or "").lower()
    if preset in {"single_signer", "single-signer"}:
        resolved_checkpoint = resolve_single_signer_checkpoint_path(
            args.pretrained_checkpoint
        )
        encoder, metadata, _ = load_single_signer_encoder(
            checkpoint_path=resolved_checkpoint,
            map_location=torch.device("cpu"),
            strict=True,
        )
        setattr(args, "resolved_checkpoint", resolved_checkpoint)
        encoder_kwargs = dict(metadata.encoder_kwargs)
        temporal_kwargs = dict(encoder_kwargs.get("temporal_kwargs", {}))
        args.projector_dim = int(encoder_kwargs.get("projector_dim", args.projector_dim))
        args.d_model = int(encoder_kwargs.get("d_model", args.d_model))
        args.sequence_length = int(
            encoder_kwargs.get("positional_num_positions", args.sequence_length)
        )
        args.projector_dropout = float(encoder_kwargs.get("projector_dropout", args.projector_dropout))
        args.fusion_dropout = float(encoder_kwargs.get("fusion_dropout", args.fusion_dropout))
        args.temporal_nhead = int(temporal_kwargs.get("nhead", args.temporal_nhead))
        args.temporal_layers = int(temporal_kwargs.get("nlayers", args.temporal_layers))
        args.temporal_dim_feedforward = int(
            temporal_kwargs.get("dim_feedforward", args.temporal_dim_feedforward)
        )
        args.temporal_dropout = float(temporal_kwargs.get("dropout", args.temporal_dropout))
        pose_dim = int(encoder_kwargs.get("pose_dim", 3 * args.pose_landmarks))
        args.pose_landmarks = max(1, pose_dim // 3)
        extra = dict(metadata.extra)
        extra.setdefault("schema_version", metadata.schema_version)
        extra.setdefault("task", metadata.task)
        extra.setdefault("encoder_kwargs", encoder_kwargs)
        extra.setdefault("checkpoint_source", preset)
        extra.setdefault("checkpoint_path", str(resolved_checkpoint))
        extra.update(
            {
                "image_size": args.image_size,
                "projector_dim": args.projector_dim,
                "d_model": args.d_model,
                "pose_landmarks": args.pose_landmarks,
                "sequence_length": args.sequence_length,
                "projector_dropout": args.projector_dropout,
                "fusion_dropout": args.fusion_dropout,
                "temporal_nhead": args.temporal_nhead,
                "temporal_layers": args.temporal_layers,
                "temporal_dim_feedforward": args.temporal_dim_feedforward,
                "temporal_dropout": args.temporal_dropout,
                "pose_dim": pose_dim,
            }
        )
        setattr(args, "_packaged_meta", extra)
        return encoder

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
    setattr(args, "_packaged_meta", {})
    return encoder


def _select_state_dict(
    state: Mapping[str, torch.Tensor], model: MultiStreamEncoder
) -> OrderedDict[str, torch.Tensor]:
    expected_keys = set(model.state_dict().keys())

    attempts: list[tuple[str, Mapping[str, torch.Tensor], set[str]]] = []

    def _register_attempt(
        mapping: Mapping[str, torch.Tensor], description: str
    ) -> Optional[OrderedDict[str, torch.Tensor]]:
        keys = set(mapping.keys())
        attempts.append((description, mapping, keys))
        if keys == expected_keys:
            return OrderedDict(mapping)
        return None

    match = _register_attempt(state, "original layout")
    if match is not None:
        return match

    for prefix in ("encoder.", "module.encoder."):
        filtered = {
            key[len(prefix) :]: value
            for key, value in state.items()
            if key.startswith(prefix)
        }
        if filtered:
            match = _register_attempt(filtered, f"removing prefix '{prefix}'")
            if match is not None:
                return match

    stripped: Dict[str, torch.Tensor] = {}
    skip_tokens = {"encoder", "module", "model", "state_dict", "checkpoint"}
    for key, value in state.items():
        segments = key.split(".")
        while segments and segments[0] in skip_tokens:
            segments = segments[1:]
        candidate = ".".join(segments) or key
        stripped[candidate] = value

    if stripped:
        remapped = dict(stripped)
        for key in list(remapped.keys()):
            if key in expected_keys:
                continue
            prefix, marker, suffix = key.partition(".layers.")
            if not marker or "." not in suffix:
                continue
            idx_str, _, tail = suffix.partition(".")
            try:
                idx = int(idx_str)
            except ValueError:
                continue
            shifted_key = f"{prefix}.layers.{idx + 1}.{tail}"
            if shifted_key in expected_keys and shifted_key not in remapped:
                remapped[shifted_key] = remapped.pop(key)

        match = _register_attempt(remapped, "stripping common prefixes")
        if match is not None:
            return match

    best_description = ""
    best_keys: set[str] = set()
    for description, _, keys in attempts:
        overlap = len(expected_keys & keys)
        if overlap > len(expected_keys & best_keys):
            best_description = description
            best_keys = keys

    missing = sorted(expected_keys - best_keys)
    unexpected = sorted(best_keys - expected_keys)
    details: list[str] = []
    if missing:
        preview = ", ".join(missing[:3])
        details.append(f"missing {len(missing)} parameter(s) (e.g. {preview})")
    if unexpected:
        preview = ", ".join(unexpected[:3])
        details.append(f"unexpected {len(unexpected)} parameter(s) (e.g. {preview})")
    if not details:
        details.append("no compatible parameter prefixes detected")
    hint = f" after {best_description}" if best_description else ""
    raise RuntimeError(
        "Checkpoint parameters are incompatible with MultiStreamEncoder: "
        + "; ".join(details)
        + hint
        + "."
    )


def _load_encoder_weights(identifier: str, model: MultiStreamEncoder) -> Dict[str, Any]:
    preset = (identifier or "").lower()
    if preset in {"single_signer", "single-signer"}:
        return {}

    path = Path(identifier)
    raw_checkpoint = torch.load(path, map_location="cpu")

    metadata: Dict[str, Any] = {}
    if isinstance(raw_checkpoint, Mapping):
        metadata = _extract_checkpoint_metadata(raw_checkpoint)

    checkpoint = raw_checkpoint
    if isinstance(raw_checkpoint, Mapping):
        for candidate in ("encoder_state", "model_state", "state_dict"):
            if candidate in raw_checkpoint:
                checkpoint = raw_checkpoint[candidate]
                break

    if not isinstance(checkpoint, Mapping):
        raise RuntimeError("Checkpoint must be a mapping containing model parameters")

    state_dict = _select_state_dict(checkpoint, model)
    model.load_state_dict(state_dict)
    return metadata


def _extract_checkpoint_metadata(checkpoint: Mapping[str, Any]) -> Dict[str, Any]:
    meta_candidates = ("encoder_config", "encoder_meta", "metadata", "config")
    for key in meta_candidates:
        value = checkpoint.get(key)
        if isinstance(value, Mapping):
            return dict(value)
    return {}


def _validate_checkpoint_metadata(
    checkpoint_meta: Mapping[str, Any], args: argparse.Namespace
) -> None:
    if not checkpoint_meta:
        warnings.warn(
            "Checkpoint metadata missing. Compatibility validation skipped.",
            UserWarning,
        )
        return

    expected = {
        "image_size": args.image_size,
        "projector_dim": args.projector_dim,
        "d_model": args.d_model,
        "pose_landmarks": args.pose_landmarks,
        "sequence_length": args.sequence_length,
    }

    derived = dict(checkpoint_meta)
    if "pose_dim" in derived and "pose_landmarks" not in derived:
        derived["pose_landmarks"] = int(derived["pose_dim"]) // 3

    mismatches = {}
    for key, expected_value in expected.items():
        if key not in derived:
            continue
        if int(derived[key]) != int(expected_value):
            mismatches[key] = (derived[key], expected_value)

    if mismatches:
        mismatch_str = ", ".join(
            f"{key}: checkpoint={checkpoint_val}, args={arg_val}"
            for key, (checkpoint_val, arg_val) in mismatches.items()
        )
        raise ValueError(
            "Checkpoint configuration is incompatible with export arguments: "
            f"{mismatch_str}"
        )


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
    pose_conf_mask = torch.ones(
        batch, seq, args.pose_landmarks, dtype=torch.bool, device=device
    )

    return (
        face,
        hand_l,
        hand_r,
        pose,
        pad_mask,
        miss_mask_hl,
        miss_mask_hr,
        pose_conf_mask,
    )


def main_export(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)

    device = torch.device(args.device)
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        device = torch.device("cpu")

    if args.artifact_dir is not None:
        args.artifact_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"encoder_{args.version}"
        if args.onnx is None:
            args.onnx = args.artifact_dir / f"{base_name}.onnx"
        if args.torchscript is None:
            args.torchscript = args.artifact_dir / f"{base_name}.ts"
        if args.metadata is None:
            args.metadata = args.artifact_dir / f"{base_name}.json"

    if args.onnx is None and args.torchscript is None:
        raise ValueError("At least one output (--onnx or --torchscript) must be provided")

    if args.metadata is not None and args.metadata.suffix != ".json":
        raise ValueError("Metadata file must use the .json extension")

    encoder = _build_encoder(args)
    packaged_meta = getattr(args, "_packaged_meta", {})
    encoder = encoder.to(device)
    checkpoint_meta = packaged_meta or _load_encoder_weights(args.checkpoint, encoder)
    _validate_checkpoint_metadata(checkpoint_meta, args)
    encoder.eval()

    export_module = EncoderExportModule(encoder).to(device)
    inputs = _dummy_inputs(args, device)
    positional_args = inputs[:4]
    keyword_args = {
        "pad_mask": inputs[4],
        "miss_mask_hl": inputs[5],
        "miss_mask_hr": inputs[6],
        "pose_conf_mask": inputs[7],
    }

    dynamic_axes = {
        "face": {1: "T"},
        "hand_l": {1: "T"},
        "hand_r": {1: "T"},
        "pose": {1: "T"},
        "pad_mask": {1: "T"},
        "miss_mask_hl": {1: "T"},
        "miss_mask_hr": {1: "T"},
        "pose_conf_mask": {1: "T"},
        "encoded": {1: "T"},
        "face_head": {1: "T"},
        "hand_left_head": {1: "T"},
        "hand_right_head": {1: "T"},
        "pose_head": {1: "T"},
        "hand_mask": {1: "T"},
        "padding_mask": {1: "T"},
    }

    if args.onnx is not None:
        args.onnx.parent.mkdir(parents=True, exist_ok=True)
        try:
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
                        "pose_conf_mask",
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
                    do_constant_folding=True,
                    operator_export_type=OperatorExportTypes.ONNX,
                )
        except ModuleNotFoundError as exc:
            if exc.name in {"onnx", "onnxscript"}:
                warnings.warn(
                    "ONNX export skipped due to missing optional dependency"
                    f" '{exc.name}'. Creating placeholder artefact instead.",
                    UserWarning,
                )
                args.onnx.touch(exist_ok=True)
            else:  # pragma: no cover - unexpected import failure
                raise

    if args.torchscript is not None:
        args.torchscript.parent.mkdir(parents=True, exist_ok=True)
        mask_examples = (
            keyword_args["pad_mask"],
            keyword_args["miss_mask_hl"],
            keyword_args["miss_mask_hr"],
            keyword_args["pose_conf_mask"],
        )
        trace_inputs = tuple(positional_args) + mask_examples
        with torch.no_grad():
            traced = torch.jit.trace_module(
                export_module,
                {"forward": trace_inputs},
                check_trace=False,
                strict=False,
            )
        traced.save(str(args.torchscript))

    if args.metadata is not None:
        args.metadata.parent.mkdir(parents=True, exist_ok=True)
        metadata = _build_metadata(
            args=args,
            checkpoint_meta=checkpoint_meta,
            dynamic_axes=dynamic_axes,
        )
        with args.metadata.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)


def _build_metadata(
    *, args: argparse.Namespace, checkpoint_meta: Mapping[str, Any], dynamic_axes: Mapping[str, Mapping[int, str]]
) -> Dict[str, Any]:
    timestamp = _dt.datetime.utcnow().replace(tzinfo=_dt.timezone.utc).isoformat()
    resolved_checkpoint = getattr(args, "resolved_checkpoint", args.checkpoint)
    if isinstance(resolved_checkpoint, Path):
        checkpoint_path = resolved_checkpoint
    else:
        checkpoint_path = Path(resolved_checkpoint)
    checkpoint_hash = _sha256(checkpoint_path)
    model_config = {
        "image_size": args.image_size,
        "projector_dim": args.projector_dim,
        "d_model": args.d_model,
        "pose_landmarks": args.pose_landmarks,
        "sequence_length": args.sequence_length,
        "temporal": {
            "nhead": args.temporal_nhead,
            "layers": args.temporal_layers,
            "dim_feedforward": args.temporal_dim_feedforward,
            "dropout": args.temporal_dropout,
        },
    }

    inputs = {
        "face": _image_input_spec(args),
        "hand_l": _image_input_spec(args),
        "hand_r": _image_input_spec(args),
        "pose": {
            "dtype": "float32",
            "shape": ["batch", "time", 3 * args.pose_landmarks],
            "normalization": {
                "type": "identity",
            },
        },
        "pad_mask": {
            "dtype": "bool",
            "shape": ["batch", "time"],
        },
        "miss_mask_hl": {
            "dtype": "bool",
            "shape": ["batch", "time"],
        },
        "miss_mask_hr": {
            "dtype": "bool",
            "shape": ["batch", "time"],
        },
        "pose_conf_mask": {
            "dtype": "bool",
            "shape": ["batch", "time", args.pose_landmarks],
        },
    }

    outputs = {
        "encoded": {"dtype": "float32", "shape": ["batch", "time", args.d_model]},
        "face_head": {"dtype": "float32", "shape": ["batch", "time", args.projector_dim]},
        "hand_left_head": {"dtype": "float32", "shape": ["batch", "time", args.projector_dim]},
        "hand_right_head": {"dtype": "float32", "shape": ["batch", "time", args.projector_dim]},
        "pose_head": {"dtype": "float32", "shape": ["batch", "time", args.projector_dim]},
        "hand_mask": {"dtype": "float32", "shape": ["batch", "time"]},
        "padding_mask": {"dtype": "float32", "shape": ["batch", "time"]},
    }

    metadata: Dict[str, Any] = {
        "metadata_version": METADATA_VERSION,
        "artifact_version": args.version,
        "generated_at": timestamp,
        "checkpoint": {
            "path": str(checkpoint_path.resolve()),
            "sha256": checkpoint_hash,
            "config": dict(checkpoint_meta),
        },
        "model": model_config,
        "inputs": inputs,
        "outputs": outputs,
        "dynamic_axes": {
            name: {str(axis): label for axis, label in axes.items()}
            for name, axes in dynamic_axes.items()
        },
        "backends": {
            "onnx": str(args.onnx.resolve()) if args.onnx is not None else None,
            "torchscript": str(args.torchscript.resolve()) if args.torchscript is not None else None,
        },
        "opset": args.opset,
    }
    return metadata


def _image_input_spec(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "dtype": "float32",
        "shape": ["batch", "time", 3, args.image_size, args.image_size],
        "normalization": {
            "type": "imagenet",
            "mean": IMAGENET_STATS["mean"],
            "std": IMAGENET_STATS["std"],
        },
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main_export()
