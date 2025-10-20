"""Smoke tests for the lightweight pre-training utilities."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
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
    class DummyBackbone(torch.nn.Module):
        def __init__(self, embed_dim: int) -> None:
            super().__init__()
            self.embed_dim = embed_dim
            self.linear = torch.nn.Linear(3 * 64 * 64, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            flat = x.view(x.size(0), -1)
            return self.linear(flat)

        def forward_with_patches(self, x: torch.Tensor):
            cls = self.forward(x)
            patches = torch.zeros(x.size(0), 1, self.embed_dim, device=x.device, dtype=x.dtype)
            return cls, patches

    patches = pytest.MonkeyPatch()
    patches.setattr(
        "tools.pretrain_utils.build_vit_backbone",
        lambda cfg: DummyBackbone(cfg.embed_dim),
    )
    patches.setattr(
        "tools._pretrain_dino.build_vit_backbone",
        lambda cfg: DummyBackbone(cfg.embed_dim),
    )
    patches.setattr(
        "slt.models.backbones._instantiate_dinov2_architecture",
        lambda *args, **kwargs: DummyBackbone(64),
    )
    run_pretraining(argv, default_stream="face")
    patches.undo()

    assert export_path.exists()

    spec = f"file::{export_path}:dinov2_vits14"
    reload_patch = pytest.MonkeyPatch()
    reload_patch.setattr(
        "slt.models.backbones._instantiate_dinov2_architecture",
        lambda *args, **kwargs: DummyBackbone(64),
    )
    face_backbone = load_dinov2_backbone(spec)
    hand_backbone = load_dinov2_backbone(spec)
    encoder = MultiStreamEncoder(
        backbones={
            "face": face_backbone,
            "hand_left": hand_backbone,
            "hand_right": load_dinov2_backbone(spec),
        }
    )
    reload_patch.undo()

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
