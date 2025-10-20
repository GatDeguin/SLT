"""Tests for runtime helpers and export integration."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("cv2", exc_type=ImportError, reason="cv2 runtime dependencies not available")

from slt.runtime.realtime import (
    OnnxRuntimeEncoderRunner,
    TemporalBuffer,
    TorchScriptEncoderRunner,
    load_export_metadata,
)
from tools.export_onnx_encoder_v9 import EncoderExportModule, _build_encoder, main_export


def _encoder_args() -> SimpleNamespace:
    return SimpleNamespace(
        image_size=16,
        projector_dim=8,
        d_model=16,
        pose_landmarks=5,
        projector_dropout=0.0,
        fusion_dropout=0.0,
        temporal_nhead=2,
        temporal_layers=1,
        temporal_dim_feedforward=32,
        temporal_dropout=0.1,
        sequence_length=4,
        opset=17,
    )


def test_temporal_buffer_latency_masks() -> None:
    buffer = TemporalBuffer(
        sequence_length=4,
        image_size=8,
        pose_landmarks=3,
        target_latency_ms=50,
        frame_rate=25.0,
    )
    device = torch.device("cpu")

    for idx in range(4):
        image = torch.full((3, 8, 8), float(idx), dtype=torch.float32)
        pose = torch.full((3 * 3,), float(idx), dtype=torch.float32)
        buffer.append(
            face=image,
            hand_l=image,
            hand_r=image,
            pose=pose,
            missing_left=bool(idx % 2),
            missing_right=bool((idx + 1) % 2),
        )

    inputs = buffer.as_model_inputs(device, backend="torch")
    assert inputs is not None
    pad_mask = inputs["pad_mask"][0]
    assert pad_mask.sum().item() == buffer.window_size
    assert pad_mask.tolist() == [True, True, False, False]

    miss_left = inputs["miss_mask_hl"][0]
    miss_right = inputs["miss_mask_hr"][0]
    assert miss_left.tolist() == [False, True, False, False]
    assert miss_right.tolist() == [True, False, False, False]


def test_runtime_backends_integration(tmp_path: Path) -> None:
    encoder_args = _encoder_args()
    encoder = _build_encoder(encoder_args)
    encoder.eval()
    state_dict = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state": state_dict,
            "encoder_config": {
                "image_size": encoder_args.image_size,
                "projector_dim": encoder_args.projector_dim,
                "d_model": encoder_args.d_model,
                "pose_landmarks": encoder_args.pose_landmarks,
                "sequence_length": encoder_args.sequence_length,
            },
        },
        checkpoint,
    )

    artifact_dir = tmp_path / "artifacts"
    version = "vtest"
    argv = [
        "--checkpoint",
        str(checkpoint),
        "--artifact-dir",
        str(artifact_dir),
        "--version",
        version,
        "--image-size",
        str(encoder_args.image_size),
        "--sequence-length",
        str(encoder_args.sequence_length),
        "--projector-dim",
        str(encoder_args.projector_dim),
        "--d-model",
        str(encoder_args.d_model),
        "--pose-landmarks",
        str(encoder_args.pose_landmarks),
        "--temporal-nhead",
        str(encoder_args.temporal_nhead),
        "--temporal-layers",
        str(encoder_args.temporal_layers),
        "--temporal-dim-feedforward",
        str(encoder_args.temporal_dim_feedforward),
        "--temporal-dropout",
        str(encoder_args.temporal_dropout),
        "--device",
        "cpu",
    ]

    main_export(argv)

    metadata_path = artifact_dir / f"encoder_{version}.json"
    ts_path = artifact_dir / f"encoder_{version}.ts"
    onnx_path = artifact_dir / f"encoder_{version}.onnx"

    assert metadata_path.exists()
    assert ts_path.exists()
    assert onnx_path.exists()

    metadata = load_export_metadata(metadata_path)
    assert metadata["artifact_version"] == version
    assert metadata["inputs"]["face"]["normalization"]["mean"] == pytest.approx([0.485, 0.456, 0.406])

    device = torch.device("cpu")
    buffer = TemporalBuffer(
        sequence_length=encoder_args.sequence_length,
        image_size=encoder_args.image_size,
        pose_landmarks=encoder_args.pose_landmarks,
        target_latency_ms=40,
        frame_rate=60.0,
    )

    for step in range(encoder_args.sequence_length + 1):
        img = torch.randn(3, encoder_args.image_size, encoder_args.image_size)
        pose = torch.randn(3 * encoder_args.pose_landmarks)
        buffer.append(
            face=img,
            hand_l=img,
            hand_r=img,
            pose=pose,
            missing_left=bool(step % 2),
            missing_right=bool((step + 1) % 2),
        )

    inputs_torch = buffer.as_model_inputs(device, backend="torchscript")
    assert inputs_torch is not None

    model_eval = _build_encoder(SimpleNamespace(**encoder_args.__dict__))
    model_eval.load_state_dict(encoder.state_dict())
    model_eval.eval()
    export_module = EncoderExportModule(model_eval)

    ordered_args = [inputs_torch[name] for name in ("face", "hand_l", "hand_r", "pose")]
    kwargs = {
        "pad_mask": inputs_torch["pad_mask"],
        "miss_mask_hl": inputs_torch["miss_mask_hl"],
        "miss_mask_hr": inputs_torch["miss_mask_hr"],
    }

    with torch.no_grad():
        reference = export_module(*ordered_args, **kwargs)

    ts_runner = TorchScriptEncoderRunner(ts_path, metadata, device=device)
    ts_outputs = ts_runner(inputs_torch)
    assert len(ts_outputs) == len(reference)
    for ref_tensor, ts_tensor in zip(reference, ts_outputs):
        torch.testing.assert_close(ts_tensor, ref_tensor)

    pytest.importorskip("onnxruntime")
    inputs_onnx = buffer.as_model_inputs(device, backend="onnxruntime")
    assert inputs_onnx is not None
    onnx_runner = OnnxRuntimeEncoderRunner(onnx_path, metadata)
    onnx_outputs = onnx_runner(inputs_onnx)
    assert len(onnx_outputs) == len(reference)
    for ref_tensor, ort_out in zip(reference, onnx_outputs):
        np.testing.assert_allclose(ort_out, ref_tensor.detach().cpu().numpy(), rtol=1e-4, atol=1e-4)
