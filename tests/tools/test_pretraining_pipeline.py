"""Smoke tests for the lightweight pre-training utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from PIL import Image

from slt.models import MultiStreamEncoder, load_dinov2_backbone
from tools._pretrain_dino import run as run_pretraining
from tools.pretrain_utils import IBOTLoss, mask_patches


def _create_dummy_dataset(root: Path, *, num_images: int = 4, size: int = 64) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for index in range(num_images):
        array = np.random.randint(0, 255, size=(size, size, 3), dtype=np.uint8)
        Image.fromarray(array, mode="RGB").save(root / f"sample_{index}.png")


def test_dino_pretraining_and_multistream_integration(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    output_dir = tmp_path / "out"
    export_path = tmp_path / "export.pt"
    _create_dummy_dataset(data_dir, num_images=4, size=64)

    argv = [
        "--train-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--epochs",
        "1",
        "--batch-size",
        "2",
        "--num-workers",
        "0",
        "--learning-rate",
        "5e-4",
        "--vit-embed-dim",
        "64",
        "--vit-depth",
        "2",
        "--vit-num-heads",
        "4",
        "--vit-mlp-ratio",
        "2.0",
        "--image-size",
        "64",
        "--patch-size",
        "16",
        "--num-local-crops",
        "0",
        "--log-interval",
        "1",
        "--export-backbone",
        str(export_path),
    ]
    run_pretraining(argv, default_stream="face")

    assert export_path.exists()

    spec = f"file::{export_path}:slt_vitsmall_patch16"
    face_backbone = load_dinov2_backbone(spec)
    hand_backbone = load_dinov2_backbone(spec)
    encoder = MultiStreamEncoder(
        backbones={
            "face": face_backbone,
            "hand_left": hand_backbone,
            "hand_right": load_dinov2_backbone(spec),
        }
    )

    batch, time = 1, 2
    face = torch.randn(batch, time, 3, 64, 64)
    hand_l = torch.randn(batch, time, 3, 64, 64)
    hand_r = torch.randn(batch, time, 3, 64, 64)
    pose = torch.randn(batch, time, 39)
    output = encoder(face, hand_l, hand_r, pose)
    assert output.shape[0] == batch
    assert output.shape[1] == time


def test_ibot_loss_respects_mask() -> None:
    loss_fn = IBOTLoss(out_dim=8, global_weight=0.0, patch_weight=1.0)
    student_global = torch.randn(4, 8)
    teacher_global = torch.randn(4, 8)
    student_patches = torch.randn(4, 5, 8)
    teacher_patches = torch.randn(4, 5, 8)
    mask = mask_patches(student_patches.shape, mask_ratio=0.4, device=student_global.device)
    loss = loss_fn(student_global, teacher_global, student_patches, teacher_patches, mask)
    assert torch.isfinite(loss)
