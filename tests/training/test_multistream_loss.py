"""Tests for the combined multi-stream loss helpers."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch.nn.functional as F

torch = pytest.importorskip("torch")

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
    translation = torch.tensor([[2, 3, -100]], dtype=torch.long)
    logits = torch.zeros(1, 3, 4, dtype=torch.float32)

    ctc_labels = torch.tensor([[1, 2, 0]], dtype=torch.long)
    ctc_mask = torch.tensor([[True, True, False]])
    ctc_lengths = torch.tensor([2], dtype=torch.long)

    time_steps = 4
    ctc_vocab = 5
    fused_logits = torch.zeros(1, time_steps, ctc_vocab, dtype=torch.float32)
    fused_mask = torch.ones(1, time_steps, dtype=torch.bool)
    stream_logits = torch.zeros(1, time_steps, ctc_vocab, dtype=torch.float32)
    stream_mask = torch.ones(1, time_steps, dtype=torch.bool)
    teacher_logits = torch.zeros(1, time_steps, ctc_vocab, dtype=torch.float32)

    outputs = SimpleNamespace(
        logits=logits,
        auxiliary={
            "fused": {"logits": fused_logits, "mask": fused_mask},
            "stream": {"pose": stream_logits},
            "frame_masks": {"pose": stream_mask},
            "distillation": {"pose": teacher_logits},
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

    expected_total = (
        components["loss_translation_weighted"]
        + components["loss_ctc_weighted"]
        + components["loss_distillation_weighted"]
    )
    assert torch.allclose(loss, expected_total)
