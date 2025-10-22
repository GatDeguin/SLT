"""Tests for the combined multi-stream loss helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn.functional as F

torch = pytest.importorskip("torch")

from slt.models.mska import FusedCTCHead, StreamCTCHead
from slt.training.loops import multistream_loss


def test_multistream_loss_translation_only() -> None:
    logits = torch.zeros(1, 2, 3, dtype=torch.float32)
    labels = torch.tensor([[1, -100]], dtype=torch.long)

    loss, components = multistream_loss(
        logits,
        {"translation": labels},
        translation_weight=0.7,
        label_smoothing=0.0,
    )

    expected = F.cross_entropy(logits.view(-1, 3), labels.view(-1), ignore_index=-100)
    assert torch.allclose(components["loss_translation"], expected.detach())
    assert torch.allclose(components["loss_translation_weighted"], expected * 0.7)
    assert torch.allclose(loss, components["loss_translation_weighted"])


def test_multistream_loss_with_ctc_and_distillation() -> None:
    torch.manual_seed(0)
    translation = torch.tensor([[2, 3, -100]], dtype=torch.long)
    logits = torch.zeros(1, 3, 4, dtype=torch.float32)

    ctc_labels = torch.tensor([[1, 2, 0]], dtype=torch.long)
    ctc_mask = torch.tensor([[True, True, False]])
    ctc_lengths = torch.tensor([2], dtype=torch.long)

    time_steps = 4
    ctc_vocab = 5
    embed_dim = 6
    features = torch.randn(1, time_steps, embed_dim, dtype=torch.float32)
    stream_head = StreamCTCHead(embed_dim, ctc_vocab, dropout=0.0)
    fused_head = FusedCTCHead(embed_dim, ctc_vocab, dropout=0.0)
    stream_logits = stream_head(features)
    fused_logits = fused_head(features)
    fused_mask = torch.ones(1, time_steps, dtype=torch.bool)
    stream_mask = fused_mask.clone()
    teacher_logits = fused_logits.detach()

    fused_probs = torch.softmax(fused_logits, dim=-1)
    stream_probs = torch.softmax(stream_logits, dim=-1)
    teacher_probs = torch.softmax(teacher_logits, dim=-1)

    outputs = SimpleNamespace(
        logits=logits,
        auxiliary={
            "fused": {"logits": fused_logits, "mask": fused_mask, "probs": fused_probs},
            "stream": {"pose": stream_logits},
            "frame_masks": {"pose": stream_mask},
            "distillation": {"pose": teacher_logits},
            "probabilities": {
                "fused": fused_probs,
                "stream": {"pose": stream_probs},
                "distillation": {"pose": teacher_probs},
            },
        },
    )

    loss, components = multistream_loss(
        outputs,
        {
            "translation": translation,
            "ctc_labels": ctc_labels,
            "ctc_mask": ctc_mask,
            "ctc_lengths": ctc_lengths,
        },
        translation_weight=1.0,
        ctc_weight=0.25,
        distillation_weight=0.5,
        distillation_temperature=2.0,
    )

    assert "loss_ctc" in components
    assert "loss_distillation" in components
    assert components["loss_ctc"].item() >= 0.0
    assert components["loss_distillation"].item() >= 0.0

    targets = ctc_labels[ctc_mask]
    input_lengths = torch.tensor([time_steps], dtype=torch.long)
    target_lengths = ctc_lengths
    fused_expected = F.ctc_loss(
        fused_logits.log_softmax(dim=-1).transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        zero_infinity=True,
    )
    stream_expected = F.ctc_loss(
        stream_logits.log_softmax(dim=-1).transpose(0, 1),
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        zero_infinity=True,
    )
    expected_ctc = fused_expected + stream_expected
    assert torch.allclose(components["loss_ctc"], expected_ctc.detach(), atol=1e-6)

    temperature = 2.0
    student = stream_logits / temperature
    teacher = teacher_logits / temperature
    log_probs = F.log_softmax(student, dim=-1)
    teacher_probs_temp = F.softmax(teacher, dim=-1)
    per_token = F.kl_div(log_probs, teacher_probs_temp, reduction="none").sum(dim=-1)
    denom = stream_mask.to(dtype=per_token.dtype).sum().clamp_min(1.0)
    expected_dist = per_token.sum() * (temperature ** 2) / denom
    assert torch.allclose(
        components["loss_distillation"], expected_dist.detach(), atol=1e-6
    )

    expected_total = (
        components["loss_translation_weighted"]
        + components["loss_ctc_weighted"]
        + components["loss_distillation_weighted"]
    )
    assert torch.allclose(loss, expected_total)
