"""Tests for the :mod:`slt.models.multistream` module."""

from __future__ import annotations

import types
from pathlib import Path
from typing import Dict, Optional

import pytest

torch = pytest.importorskip("torch")

from slt.models.backbones import load_dinov2_backbone
from slt.models.mska import MSKAEncoder
from slt.models.multistream import MultiStreamEncoder
from slt.models.single_signer import (
    CHECKPOINT_ENV_VAR,
    CHECKPOINT_FILENAME,
    build_single_signer_backbones,
)
from slt.models.temporal import TextSeq2SeqDecoder


IMAGE_SIZE = 32


class ConstantBackbone(torch.nn.Module):
    def __init__(self, value: float = 0.0, embed_dim: int = 384) -> None:
        super().__init__()
        self.value = value
        self.embed_dim = embed_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch = x.size(0)
        return torch.full((batch, self.embed_dim), self.value, device=x.device, dtype=x.dtype)


def _write_dummy_single_signer_checkpoint(path: Path) -> None:
    backbone_kwargs = {
        "in_channels": 3,
        "base_channels": 8,
        "features": 16,
        "dropout": 0.0,
    }
    encoder_kwargs = {
        "projector_dim": 8,
        "d_model": 16,
        "pose_dim": 39,
        "positional_num_positions": 16,
        "projector_dropout": 0.0,
        "fusion_dropout": 0.0,
        "temporal_kwargs": {"nhead": 2, "nlayers": 1, "dim_feedforward": 32, "dropout": 0.0},
    }
    backbones = build_single_signer_backbones(**backbone_kwargs)
    encoder = MultiStreamEncoder(backbones=backbones, **encoder_kwargs)

    decoder_kwargs = {
        "d_model": 16,
        "vocab_size": 32,
        "num_layers": 1,
        "num_heads": 2,
        "dropout": 0.0,
        "pad_token_id": 0,
        "eos_token_id": 1,
    }
    decoder = TextSeq2SeqDecoder(**decoder_kwargs)

    checkpoint = {
        "schema_version": "1.0",
        "task": "single_signer",
        "encoder": {
            "init_kwargs": encoder_kwargs,
            "backbone_kwargs": backbone_kwargs,
            "state_dict": encoder.state_dict(),
        },
        "decoder": {
            "init_kwargs": decoder_kwargs,
            "state_dict": decoder.state_dict(),
        },
        "tokenizer": {"pad_token_id": 0, "eos_token_id": 1},
        "metadata": {"dummy": True},
    }
    torch.save(checkpoint, path)


def _make_encoder(**kwargs) -> MultiStreamEncoder:
    backbones = {
        "face": ConstantBackbone(),
        "hand_left": ConstantBackbone(),
        "hand_right": ConstantBackbone(),
    }
    return MultiStreamEncoder(backbones=backbones, **kwargs)


def _make_mska(*, shared: bool) -> MSKAEncoder:
    return MSKAEncoder(
        input_dim=3,
        embed_dim=8,
        stream_names=("face", "hand_left", "hand_right", "pose"),
        num_heads=1,
        ff_multiplier=1,
        dropout=0.0,
        ctc_vocab_size=8,
        stream_attention_heads=1,
        stream_temporal_blocks=0,
        stream_temporal_kernel=3,
        stream_temporal_dilation=1,
        use_global_attention=True,
        global_attention_activation="identity",
        global_attention_mix=0.5,
        global_attention_shared=shared,
    )


def test_forward_pass_returns_temporal_sequence() -> None:
    model = _make_encoder(
        projector_dim=16,
        d_model=32,
        pose_dim=39,
        positional_num_positions=32,
        temporal_kwargs={"nhead": 4, "nlayers": 1, "dim_feedforward": 64},
    )

    batch, time = 2, 4
    face = torch.randn(batch, time, 3, IMAGE_SIZE, IMAGE_SIZE)
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

    model = RecordingEncoder(
        projector_dim=8,
        d_model=16,
        pose_dim=39,
        positional_num_positions=16,
        temporal_kwargs={"nhead": 4, "nlayers": 1, "dim_feedforward": 32},
        backbones={
            "face": ConstantBackbone(),
            "hand_left": ConstantBackbone(),
            "hand_right": ConstantBackbone(),
        },
    )

    batch, time = 1, 2
    face = torch.randn(batch, time, 3, IMAGE_SIZE, IMAGE_SIZE)
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


def test_missing_hand_frames_affect_only_masked_positions() -> None:
    model = _make_encoder(
        projector_dim=8,
        d_model=16,
        pose_dim=39,
        positional_num_positions=16,
        temporal_kwargs={"nhead": 4, "nlayers": 1, "dim_feedforward": 32},
    )

    dim = 6

    def fake_encode(self, backbone, stream):
        return stream

    model._encode_backbone = types.MethodType(fake_encode, model)

    class Identity(torch.nn.Module):
        def forward(self, x):
            return x

    class ZeroProjector(torch.nn.Module):
        def __init__(self, dim: int) -> None:
            super().__init__()
            self.dim = dim

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            shape = x.shape[:-1] + (self.dim,)
            return torch.zeros(shape, dtype=x.dtype, device=x.device)

    class DummyFusion(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.received_masks = []

        def forward(self, *streams, mask=None):
            self.received_masks.append(mask)
            total = torch.zeros_like(streams[0])
            for stream in streams:
                total = total + stream
            return total

    class TemporalIdentity(torch.nn.Module):
        def forward(self, sequence, src_key_padding_mask=None):
            return sequence

    model.face_projector = ZeroProjector(dim)
    model.hand_left_projector = Identity()
    model.hand_right_projector = Identity()
    model.pose_projector = ZeroProjector(dim)
    model.fusion = DummyFusion()
    model.positional = Identity()
    model.temporal = TemporalIdentity()

    batch, time = 1, 4
    face = torch.zeros(batch, time, dim)
    hand_l = torch.randn(batch, time, dim)
    hand_r = torch.randn(batch, time, dim)
    pose = torch.zeros(batch, time, 39)

    baseline = model(
        face,
        hand_l,
        hand_r,
        pose,
        pad_mask=None,
        miss_mask_hl=None,
        miss_mask_hr=None,
    )

    miss_hl = torch.tensor([[True, False, False, False]])
    miss_hr = torch.tensor([[False, False, True, False]])

    masked = model(
        face,
        hand_l,
        hand_r,
        pose,
        pad_mask=None,
        miss_mask_hl=miss_hl,
        miss_mask_hr=miss_hr,
    )

    combined_mask = miss_hl | miss_hr

    assert model._last_combined_hand_mask is not None
    assert torch.equal(model._last_combined_hand_mask, combined_mask)
    assert model.fusion.received_masks[-1] is not None
    assert torch.equal(model.fusion.received_masks[-1], combined_mask)

    delta = torch.abs(masked - baseline)
    unchanged = combined_mask.logical_not().unsqueeze(-1).expand_as(delta)
    changed = combined_mask.unsqueeze(-1).expand_as(delta)

    assert torch.all(delta[unchanged] < 1e-6)
    assert torch.any(delta[changed] > 1e-6)


def test_forward_with_external_backbones(tmp_path, monkeypatch) -> None:
    class DummyBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_dim = 384
            self.linear = torch.nn.Linear(3 * IMAGE_SIZE * IMAGE_SIZE, self.embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() != 4:
                raise AssertionError("Backbone inputs must be BCHW tensors")
            flat = x.view(x.size(0), -1)
            return self.linear(flat)

    reference = DummyBackbone()
    checkpoint_path = tmp_path / "dummy.pt"
    torch.save({"model": reference.state_dict()}, checkpoint_path)

    calls = []

    def fake_instantiate(model_name, *, map_location=None, trust_repo=True):
        calls.append((model_name, map_location))
        return DummyBackbone()

    monkeypatch.setattr(
        "slt.models.backbones._instantiate_dinov2_architecture", fake_instantiate
    )

    spec = f"file::{checkpoint_path}:dinov2_vits14"
    backbones = {
        "face": load_dinov2_backbone(spec),
        "hand_left": load_dinov2_backbone(spec),
        "hand_right": load_dinov2_backbone(spec, freeze=True),
    }

    encoder = MultiStreamEncoder(
        projector_dim=16,
        d_model=32,
        pose_dim=39,
        positional_num_positions=32,
        temporal_kwargs={"nhead": 4, "nlayers": 1, "dim_feedforward": 64},
        backbones=backbones,
    )

    batch, time = 2, 3
    face = torch.randn(batch, time, 3, IMAGE_SIZE, IMAGE_SIZE)
    hand_l = torch.randn_like(face)
    hand_r = torch.randn_like(face)
    pose = torch.randn(batch, time, 39)

    output = encoder(face, hand_l, hand_r, pose)

    assert output.shape == (batch, time, 32)
    assert len(calls) == 3
    assert all(param.requires_grad for param in backbones["face"].parameters())
    assert all(param.requires_grad for param in backbones["hand_left"].parameters())
    assert all(not param.requires_grad for param in backbones["hand_right"].parameters())


def test_register_and_switch_backbones() -> None:
    encoder = _make_encoder(
        projector_dim=8,
        d_model=16,
        pose_dim=39,
        positional_num_positions=16,
        temporal_kwargs={"nhead": 4, "nlayers": 1, "dim_feedforward": 32},
    )

    encoder.register_backbone("face", "const", lambda: ConstantBackbone(3.14))
    encoder.activate_backbone("face", "const")

    assert encoder.active_backbone_name("face") == "const"
    assert "const" in encoder.available_backbones("face")["face"]

    batch, time = 1, 2
    face = torch.randn(batch, time, 3, IMAGE_SIZE, IMAGE_SIZE)
    hand = torch.randn_like(face)
    pose = torch.randn(batch, time, 39)

    output = encoder(face, hand, hand, pose)
    assert output.shape == (batch, time, 16)


def test_hand_masks_emitted_and_tracked() -> None:
    encoder = _make_encoder(
        projector_dim=8,
        d_model=16,
        pose_dim=39,
        positional_num_positions=16,
        temporal_kwargs={"nhead": 4, "nlayers": 1, "dim_feedforward": 32},
    )

    recorded: Dict[str, torch.Tensor] = {}

    def hook(name: str, tensor: torch.Tensor) -> None:
        recorded[name] = tensor

    encoder.register_observer("hand_left.mask", hook)
    encoder.register_observer("hand_right.mask", hook)
    encoder.register_observer("hand.mask", hook)

    batch, time = 1, 3
    face = torch.randn(batch, time, 3, IMAGE_SIZE, IMAGE_SIZE)
    hand_l = torch.randn_like(face)
    hand_r = torch.randn_like(face)
    pose = torch.randn(batch, time, 39)

    miss_hl = torch.tensor([[True, False, True]])
    miss_hr = torch.tensor([[False, False, True]])

    _ = encoder(face, hand_l, hand_r, pose, miss_mask_hl=miss_hl, miss_mask_hr=miss_hr)

    masks = encoder.last_hand_masks
    assert torch.equal(masks["hand_left"], miss_hl)
    assert torch.equal(masks["hand_right"], miss_hr)
    assert "hand_left.mask" in recorded
    assert "hand_right.mask" in recorded
    assert "hand.mask" in recorded


def test_pose_conf_mask_zeroes_landmarks() -> None:
    encoder = _make_encoder(
        projector_dim=4,
        d_model=8,
        pose_dim=6,
        positional_num_positions=4,
        temporal_kwargs={"nhead": 2, "nlayers": 1, "dim_feedforward": 16},
    )

    class CaptureProjector(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.last_input: Optional[torch.Tensor] = None

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.last_input = x.detach()
            return x

    original_projector = encoder.pose_projector

    class WrappedProjector(CaptureProjector):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.last_input = x.detach()
            return original_projector(x)

    capture = WrappedProjector()
    encoder.pose_projector = capture

    batch, time = 1, 2
    landmarks = 2
    face = torch.randn(batch, time, 3, IMAGE_SIZE, IMAGE_SIZE)
    hand = torch.randn_like(face)
    pose = torch.arange(batch * time * landmarks * 3, dtype=torch.float32).view(batch, time, landmarks * 3)
    mask = torch.tensor([[[True, True], [False, False]]])

    _ = encoder(
        face,
        hand,
        hand,
        pose,
        pose_conf_mask=mask,
    )

    assert encoder.last_pose_mask is not None
    assert torch.equal(encoder.last_pose_mask, mask)
    assert capture.last_input is not None
    reshaped = capture.last_input.view(batch, time, landmarks, 3)
    assert torch.allclose(reshaped[0, 0], pose.view(batch, time, landmarks, 3)[0, 0])
    assert torch.all(reshaped[0, 1] == 0)


def test_stream_state_dict_roundtrip() -> None:
    encoder = _make_encoder(
        projector_dim=8,
        d_model=16,
        pose_dim=39,
        positional_num_positions=16,
        temporal_kwargs={"nhead": 4, "nlayers": 1, "dim_feedforward": 32},
    )

    face_state = encoder.stream_state_dict("face")
    assert face_state
    assert any(key.startswith("projector") for key in face_state)

    fresh = _make_encoder(
        projector_dim=8,
        d_model=16,
        pose_dim=39,
        positional_num_positions=16,
        temporal_kwargs={"nhead": 4, "nlayers": 1, "dim_feedforward": 32},
    )
    load_info = fresh.load_stream_state_dict("face", face_state)
    assert not load_info.missing_keys
    assert not load_info.unexpected_keys


def test_from_pretrained_returns_loaded_encoder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    checkpoint_path = tmp_path / CHECKPOINT_FILENAME
    _write_dummy_single_signer_checkpoint(checkpoint_path)
    monkeypatch.setenv(CHECKPOINT_ENV_VAR, str(checkpoint_path))

    encoder = MultiStreamEncoder.from_pretrained()

    assert isinstance(encoder, MultiStreamEncoder)
    assert hasattr(encoder, "pretrained_metadata")
    metadata = encoder.pretrained_metadata
    assert metadata.task == "single_signer"
    assert metadata.encoder_kwargs
    assert metadata.backbone_kwargs["features"] == 16
    monkeypatch.delenv(CHECKPOINT_ENV_VAR, raising=False)


def test_gloss_mlp_uses_leaky_relu_activation() -> None:
    class DummyMSKA(torch.nn.Module):
        def __init__(self, embed_dim: int) -> None:
            super().__init__()
            self.embed_dim = embed_dim
            self.stream_names = ("face", "hand_left", "hand_right", "pose")

    slope = 0.15
    dummy_mska = DummyMSKA(embed_dim=8)
    encoder = _make_encoder(
        projector_dim=8,
        d_model=16,
        pose_dim=39,
        positional_num_positions=16,
        temporal_kwargs={"nhead": 2, "nlayers": 1, "dim_feedforward": 32},
        mska=dummy_mska,
        mska_gloss_hidden_dim=12,
        mska_gloss_activation="leaky_relu",
        leaky_relu_negative_slope=slope,
    )

    mlp = encoder._mska_gloss_mlp
    assert mlp is not None
    activation = next(
        (module for module in mlp if isinstance(module, torch.nn.LeakyReLU)),
        None,
    )
    assert activation is not None
    assert abs(activation.negative_slope - slope) < 1e-6

def test_gloss_mlp_two_hidden_layers_produce_expected_sequence() -> None:
    class DummyMSKA(torch.nn.Module):
        def __init__(self, embed_dim: int, seq_len: int) -> None:
            super().__init__()
            self.embed_dim = embed_dim
            self.stream_names = ("face", "hand_left", "hand_right", "pose")
            self._seq_len = seq_len

        def forward(self, keypoint_streams):
            first_stream = next(iter(keypoint_streams.values()))
            batch = first_stream["points"].shape[0]
            fused_embedding = torch.ones(batch, self._seq_len, self.embed_dim)
            fused_mask = torch.ones(batch, self._seq_len, dtype=torch.bool)
            stream_embeddings = {
                name: torch.zeros(batch, self._seq_len, self.embed_dim)
                for name in self.stream_names
            }
            payload = {
                "stream_embeddings": stream_embeddings,
                "joint_embeddings": {},
                "stream_masks": {},
                "frame_masks": {},
                "fused_embedding": fused_embedding,
                "fused_mask": fused_mask,
                "attention": None,
            }
            return types.SimpleNamespace(**payload)

    batch, time = 2, 3
    projector_dim = 8
    d_model = 16
    first_hidden = 12
    second_hidden = 10
    dummy_mska = DummyMSKA(embed_dim=projector_dim, seq_len=time)
    encoder = _make_encoder(
        projector_dim=projector_dim,
        d_model=d_model,
        pose_dim=39,
        positional_num_positions=time,
        temporal_kwargs={"nhead": 2, "nlayers": 1, "dim_feedforward": 32},
        mska=dummy_mska,
        mska_gloss_hidden_dim=first_hidden,
        mska_gloss_second_hidden_dim=second_hidden,
        mska_gloss_dropout=0.25,
    )

    face = torch.randn(batch, time, 3, IMAGE_SIZE, IMAGE_SIZE)
    hand_l = torch.randn_like(face)
    hand_r = torch.randn_like(face)
    pose = torch.randn(batch, time, 39)
    keypoint_streams = {
        name: {"points": torch.randn(batch, time, 3)}
        for name in ("face", "hand_left", "hand_right", "pose")
    }

    _ = encoder(
        face,
        hand_l,
        hand_r,
        pose,
        keypoint_streams=keypoint_streams,
    )

    gloss_sequence = encoder.last_gloss_sequence
    gloss_mask = encoder.last_gloss_mask
    assert gloss_sequence is not None
    assert gloss_sequence.shape == (batch, time, d_model)
    assert gloss_mask is not None
    assert gloss_mask.shape == (batch, time)

    mlp = encoder._mska_gloss_mlp
    assert mlp is not None
    assert isinstance(mlp[0], torch.nn.Linear)
    assert isinstance(mlp[3], torch.nn.Linear)
    assert isinstance(mlp[-1], torch.nn.Linear)
    assert mlp[0].out_features == first_hidden
    assert mlp[3].out_features == second_hidden
    assert mlp[-1].in_features == second_hidden


def test_mska_encoder_sgr_shared_matrix_reused() -> None:
    torch.manual_seed(0)
    mska = _make_mska(shared=True)
    encoder = _make_encoder(
        projector_dim=8,
        d_model=16,
        pose_dim=39,
        positional_num_positions=8,
        temporal_kwargs={"nhead": 2, "nlayers": 1, "dim_feedforward": 32},
        mska=mska,
    )

    batch, time, joints = 2, 3, 4
    face = torch.randn(batch, time, 3, IMAGE_SIZE, IMAGE_SIZE)
    hand_l = torch.randn_like(face)
    hand_r = torch.randn_like(face)
    pose = torch.randn(batch, time, 39)
    keypoint_streams = {
        name: {"points": torch.randn(batch, time, joints, 3)}
        for name in encoder.mska_encoder.stream_names
    }

    output = encoder(
        face,
        hand_l,
        hand_r,
        pose,
        keypoint_streams=keypoint_streams,
    )

    store = encoder.mska_encoder._shared_global_store
    assert store is not None
    per_stream_stores = [
        stream_encoder.self_attention._global_store()
        for stream_encoder in encoder.mska_encoder.encoders.values()
    ]
    assert per_stream_stores
    assert all(candidate is store for candidate in per_stream_stores)

    encoder.zero_grad()
    loss = output.sum()
    loss.backward()

    matrix = store.matrix
    assert matrix is not None
    assert matrix.grad is not None
    assert matrix.grad.abs().sum() > 0


def test_mska_encoder_sgr_per_stream_independent_matrices() -> None:
    torch.manual_seed(0)
    mska = _make_mska(shared=False)
    encoder = _make_encoder(
        projector_dim=8,
        d_model=16,
        pose_dim=39,
        positional_num_positions=8,
        temporal_kwargs={"nhead": 2, "nlayers": 1, "dim_feedforward": 32},
        mska=mska,
    )

    batch, time, joints = 2, 3, 4
    face = torch.randn(batch, time, 3, IMAGE_SIZE, IMAGE_SIZE)
    hand_l = torch.randn_like(face)
    hand_r = torch.randn_like(face)
    pose = torch.randn(batch, time, 39)
    keypoint_streams = {
        name: {"points": torch.randn(batch, time, joints, 3)}
        for name in encoder.mska_encoder.stream_names
    }

    output = encoder(
        face,
        hand_l,
        hand_r,
        pose,
        keypoint_streams=keypoint_streams,
    )

    stores = []
    for stream_encoder in encoder.mska_encoder.encoders.values():
        store = stream_encoder.self_attention._global_store()
        assert store is not None
        stores.append(store)

    assert len({id(store) for store in stores}) == len(stores)

    encoder.zero_grad()
    loss = output.sum()
    loss.backward()

    for store in stores:
        matrix = store.matrix
        assert matrix is not None
        assert matrix.grad is not None
        assert matrix.grad.abs().sum() > 0

