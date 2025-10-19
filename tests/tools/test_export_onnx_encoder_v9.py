"""Smoke tests for the ONNX export utility."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

torch = pytest.importorskip("torch")

from slt.models import MultiStreamEncoder, ViTConfig

from tools.export_onnx_encoder_v9 import main_export


def _build_encoder_state(image_size: int, sequence_length: int) -> MultiStreamEncoder:
    vit_config = ViTConfig(image_size=image_size)
    encoder = MultiStreamEncoder(
        backbone_config=vit_config,
        projector_dim=8,
        d_model=16,
        pose_dim=39,
        positional_num_positions=sequence_length,
        temporal_kwargs={"nhead": 4, "nlayers": 1, "dim_feedforward": 32, "dropout": 0.1},
    )
    return encoder


def test_main_export_writes_onnx_file(tmp_path: Path) -> None:
    image_size = 16
    sequence_length = 4

    encoder = _build_encoder_state(image_size, sequence_length)
    state_dict = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state": state_dict}, checkpoint)

    out_path = Path("/tmp") / f"encoder_{uuid4().hex}.onnx"
    if out_path.exists():
        out_path.unlink()

    args = [
        "--checkpoint",
        str(checkpoint),
        "--out",
        str(out_path),
        "--image-size",
        str(image_size),
        "--sequence-length",
        str(sequence_length),
        "--projector-dim",
        "8",
        "--d-model",
        "16",
        "--pose-landmarks",
        "13",
        "--temporal-nhead",
        "4",
        "--temporal-layers",
        "1",
        "--temporal-dim-feedforward",
        "32",
        "--temporal-dropout",
        "0.1",
    ]

    main_export(args)

    try:
        assert out_path.exists()
    finally:
        if out_path.exists():
            out_path.unlink()
