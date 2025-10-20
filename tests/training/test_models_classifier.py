from __future__ import annotations

from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.nn.functional as F

from slt.training.configuration import ModelConfig
from slt.training.models import MultiStreamClassifier


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 8

    def __len__(self) -> int:  # pragma: no cover - compatibility shim
        return self.vocab_size


class DummyDecoder(torch.nn.Module):
    def __init__(
        self,
        *,
        d_model: int,
        vocab_size: int,
        pad_token_id: int,
        eos_token_id: int,
        **_: object,
    ) -> None:
        super().__init__()
        self.output = torch.nn.Linear(d_model, vocab_size)

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,
        *,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> SimpleNamespace:
        del encoder_attention_mask
        batch, seq_len, _ = encoder_hidden_states.shape
        if decoder_attention_mask is not None:
            target_len = int(decoder_attention_mask.shape[-1])
        elif labels is not None:
            target_len = int(labels.shape[-1])
        else:
            target_len = seq_len
        hidden = encoder_hidden_states[:, :target_len]
        logits = self.output(hidden)
        loss = None
        if labels is not None:
            flat_logits = logits.reshape(-1, logits.size(-1))
            flat_labels = labels.reshape(-1)
            mask = flat_labels.ne(-100)
            if mask.any():
                loss = F.cross_entropy(flat_logits[mask], flat_labels[mask], reduction="mean")
            else:
                loss = logits.sum() * 0
        return SimpleNamespace(logits=logits, loss=loss)

    def generate(
        self,
        encoder_hidden_states: torch.Tensor,
        *,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 2,
        **_: object,
    ) -> torch.LongTensor:
        del encoder_attention_mask
        batch = encoder_hidden_states.shape[0]
        return torch.zeros(batch, max_length, dtype=torch.long, device=encoder_hidden_states.device)


@pytest.mark.parametrize("batch_size,target_length", [(2, 2)])
def test_classifier_forward_backward_with_stream_backbones(monkeypatch, batch_size, target_length) -> None:
    image_size = 8
    embed_dim = 12

    class DummyBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_dim = embed_dim
            self.linear = torch.nn.Linear(3 * image_size * image_size, embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            flat = x.view(x.size(0), -1)
            return self.linear(flat)

    monkeypatch.setattr(
        "slt.training.models.load_dinov2_backbone",
        lambda spec, freeze=False: DummyBackbone(),
    )

    config = ModelConfig(
        image_size=image_size,
        projector_dim=8,
        d_model=16,
        pose_landmarks=1,
        projector_dropout=0.0,
        fusion_dropout=0.0,
        temporal_nhead=2,
        temporal_layers=1,
        temporal_dim_feedforward=32,
        temporal_dropout=0.0,
        sequence_length=2,
        decoder_layers=1,
        decoder_heads=2,
        decoder_dropout=0.0,
        decoder_class="tests.training.test_models_classifier.DummyDecoder",
        decoder_kwargs={},
        face_backbone="stub",
        hand_left_backbone="stub",
        hand_right_backbone="stub",
    )

    tokenizer = TinyTokenizer()
    model = MultiStreamClassifier(config, tokenizer)

    seq_len = config.sequence_length
    face = torch.randn(batch_size, seq_len, 3, image_size, image_size)
    hand_l = torch.randn_like(face)
    hand_r = torch.randn_like(face)
    pose = torch.randn(batch_size, seq_len, 3 * config.pose_landmarks)
    pose_conf_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    pose_conf_mask[:, -1] = torch.tensor([False] * batch_size)
    pad_mask = torch.ones(batch_size, seq_len, dtype=torch.bool)
    miss_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    labels = torch.randint(0, tokenizer.vocab_size, (batch_size, target_length))
    decoder_attention_mask = torch.ones(batch_size, target_length, dtype=torch.long)

    outputs = model(
        face=face,
        hand_l=hand_l,
        hand_r=hand_r,
        pose=pose,
        pose_conf_mask=pose_conf_mask,
        pad_mask=pad_mask,
        miss_mask_hl=miss_mask,
        miss_mask_hr=miss_mask,
        labels=labels,
        decoder_attention_mask=decoder_attention_mask,
    )

    assert outputs.logits.shape == (batch_size, target_length, tokenizer.vocab_size)
    assert outputs.loss is not None
    outputs.loss.backward()

    encoder_grads = [p.grad for p in model.encoder.parameters() if p.requires_grad]
    assert any(grad is not None for grad in encoder_grads)
    expected_mask = pose_conf_mask.unsqueeze(-1)
    assert torch.equal(model.encoder.last_pose_mask, expected_mask)
