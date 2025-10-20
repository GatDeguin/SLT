"""Tests for :class:`slt.models.temporal.TextSeq2SeqDecoder`."""

from __future__ import annotations

from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import BartConfig, T5Config, T5ForConditionalGeneration

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


def test_pretrained_checkpoint_generation(tmp_path: Path) -> None:
    config = T5Config(
        vocab_size=32,
        d_model=32,
        d_kv=8,
        d_ff=64,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    model = T5ForConditionalGeneration(config)
    model.save_pretrained(tmp_path)

    decoder = TextSeq2SeqDecoder(
        d_model=32,
        pretrained_model_name_or_path=str(tmp_path),
    )

    batch_size, encoder_steps, target_steps = 2, 3, 4
    encoder_hidden = torch.randn(batch_size, encoder_steps, 32)
    labels = torch.randint(0, config.vocab_size, (batch_size, target_steps))
    mask = torch.ones(batch_size, encoder_steps, dtype=torch.long)
    decoder_mask = torch.ones(batch_size, target_steps, dtype=torch.long)

    outputs = decoder(
        encoder_hidden,
        encoder_attention_mask=mask,
        labels=labels,
        decoder_attention_mask=decoder_mask,
    )
    assert outputs.logits.shape == (batch_size, target_steps, config.vocab_size)

    generated = decoder.generate(
        encoder_hidden,
        encoder_attention_mask=mask,
        max_length=target_steps,
    )
    assert list(generated.shape) == [batch_size, target_steps]


def test_hidden_size_mismatch_raises(tmp_path: Path) -> None:
    config = T5Config(
        vocab_size=16,
        d_model=64,
        d_kv=16,
        d_ff=128,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        dropout_rate=0.0,
        pad_token_id=0,
        eos_token_id=1,
        decoder_start_token_id=0,
    )
    model = T5ForConditionalGeneration(config)
    model.save_pretrained(tmp_path)

    with pytest.raises(ValueError):
        TextSeq2SeqDecoder(d_model=32, pretrained_model_name_or_path=str(tmp_path))


def test_builds_bart_configuration() -> None:
    decoder = TextSeq2SeqDecoder(
        d_model=32,
        vocab_size=64,
        num_layers=1,
        num_heads=4,
        dropout=0.1,
        config_kwargs={
            "model_type": "bart",
            "encoder_layers": 2,
            "decoder_layers": 2,
            "encoder_attention_heads": 4,
            "decoder_attention_heads": 4,
        },
    )

    config = decoder.config
    assert isinstance(config, BartConfig)
    assert config.model_type == "bart"
    assert config.d_model == 32
