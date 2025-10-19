"""Tests for :class:`slt.models.temporal.TextSeq2SeqDecoder`."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from slt.models import TextSeq2SeqDecoder


def test_forward_pass_produces_logits_and_loss() -> None:
    vocab_size = 32
    decoder = TextSeq2SeqDecoder(
        d_model=32,
        vocab_size=vocab_size,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        pad_token_id=0,
        eos_token_id=1,
        config_kwargs={"d_ff": 64},
    )

    batch_size, encoder_steps, target_steps = 2, 3, 4
    encoder_hidden = torch.randn(batch_size, encoder_steps, 32)
    labels = torch.randint(0, vocab_size, (batch_size, target_steps))
    encoder_mask = torch.ones(batch_size, encoder_steps, dtype=torch.bool)
    decoder_mask = torch.ones(batch_size, target_steps, dtype=torch.long)

    outputs = decoder(
        encoder_hidden,
        encoder_attention_mask=encoder_mask,
        labels=labels,
        decoder_attention_mask=decoder_mask,
    )

    assert outputs.logits.shape == (batch_size, target_steps, vocab_size)
    assert outputs.loss is not None
