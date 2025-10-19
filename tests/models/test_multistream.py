"""Tests for the :mod:`slt.models.multistream` module."""

from __future__ import annotations

import types

import pytest

torch = pytest.importorskip("torch")

from slt.models.backbones import ViTConfig, load_dinov2_backbone
from slt.models.multistream import MultiStreamEncoder


def _tiny_vit_config() -> ViTConfig:
    return ViTConfig(
        image_size=16,
        patch_size=8,
        in_channels=3,
        embed_dim=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
    )


def test_forward_pass_returns_temporal_sequence() -> None:
    config = _tiny_vit_config()
    model = MultiStreamEncoder(
        backbone_config=config,
        projector_dim=16,
        d_model=32,
        pose_dim=39,
        positional_num_positions=32,
        temporal_kwargs={"nhead": 4, "nlayers": 1, "dim_feedforward": 64},
    )

    batch, time = 2, 4
    face = torch.randn(batch, time, 3, config.image_size, config.image_size)
    hand_l = torch.randn_like(face)
    hand_r = torch.randn_like(face)
    pose = torch.randn(batch, time, 39)
    pad_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)

    captured = {}

    original_forward = model.temporal.forward

    def capture_forward(self, sequence, src_key_padding_mask=None):
        captured["mask"] = src_key_padding_mask
        return original_forward(sequence, src_key_padding_mask=src_key_padding_mask)

    model.temporal.forward = types.MethodType(capture_forward, model.temporal)

    output = model(
        face,
        hand_l,
        hand_r,
        pose,
        pad_mask=pad_mask,
        miss_mask_hl=None,
        miss_mask_hr=None,
    )

    assert output.shape == (batch, time, 32)
    assert captured["mask"].shape == pad_mask.shape
    # Padding mask should be flipped so padded (0) positions become True.
    expected_mask = torch.tensor([[False, False, False, False], [False, False, True, True]])
    assert torch.equal(captured["mask"], expected_mask)


def test_missing_hand_masks_trigger_hook() -> None:
    class RecordingEncoder(MultiStreamEncoder):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.calls = []

        def _mask_hand_features(self, features, mask, *, stream: str):
            self.calls.append((stream, mask.clone()))
            return features

    config = _tiny_vit_config()
    model = RecordingEncoder(
        backbone_config=config,
        projector_dim=8,
        d_model=16,
        pose_dim=39,
        positional_num_positions=16,
        temporal_kwargs={"nhead": 4, "nlayers": 1, "dim_feedforward": 32},
    )

    batch, time = 1, 2
    face = torch.randn(batch, time, 3, config.image_size, config.image_size)
    hand_l = torch.randn_like(face)
    hand_r = torch.randn_like(face)
    pose = torch.randn(batch, time, 39)
    miss_hl = torch.tensor([[1, 0]])
    miss_hr = torch.tensor([[0, 1]])

    _ = model(
        face,
        hand_l,
        hand_r,
        pose,
        pad_mask=None,
        miss_mask_hl=miss_hl,
        miss_mask_hr=miss_hr,
    )

    assert len(model.calls) == 2
    streams = [call[0] for call in model.calls]
    assert streams == ["hand_left", "hand_right"]
    for _, mask in model.calls:
        assert mask.dtype == torch.bool
        assert mask.shape == (batch, time)


def test_forward_with_external_backbones(tmp_path, monkeypatch) -> None:
    config = _tiny_vit_config()

    class DummyBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_dim = 32
            self.proj = torch.nn.Linear(3 * config.image_size * config.image_size, self.embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 4:
                raise AssertionError("Backbone inputs must be BCHW tensors")
            batch = x.size(0)
            flat = x.view(batch, -1)
            return self.proj(flat)

    reference = DummyBackbone()
    checkpoint_path = tmp_path / "dummy.pt"
    torch.save(reference.state_dict(), checkpoint_path)

    calls = []

    def fake_torchhub_load(repo, model, pretrained=True, trust_repo=True):
        calls.append((repo, model, pretrained))
        module = DummyBackbone()
        if pretrained:
            module.load_state_dict(reference.state_dict())
        return module

    monkeypatch.setattr(torch.hub, "load", fake_torchhub_load)

    spec = f"file::{checkpoint_path}:dummy"
    backbones = {
        "face": load_dinov2_backbone(spec),
        "hand_left": load_dinov2_backbone(spec),
        "hand_right": load_dinov2_backbone(spec, freeze=True),
    }

    encoder = MultiStreamEncoder(
        backbone_config=config,
        projector_dim=16,
        d_model=32,
        pose_dim=39,
        positional_num_positions=32,
        temporal_kwargs={"nhead": 4, "nlayers": 1, "dim_feedforward": 64},
        backbones=backbones,
    )

    batch, time = 2, 3
    face = torch.randn(batch, time, 3, config.image_size, config.image_size)
    hand_l = torch.randn_like(face)
    hand_r = torch.randn_like(face)
    pose = torch.randn(batch, time, 39)

    output = encoder(face, hand_l, hand_r, pose)

    assert output.shape == (batch, time, 32)
    assert len(calls) == 3
    assert all(param.requires_grad for param in backbones["face"].parameters())
    assert all(param.requires_grad for param in backbones["hand_left"].parameters())
    assert all(not param.requires_grad for param in backbones["hand_right"].parameters())
