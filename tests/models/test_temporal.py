"""Unit tests for temporal encoder and decoder stubs."""

import sys
from pathlib import Path

import torch
from transformers import PretrainedConfig, T5ForConditionalGeneration

sys.path.append(str(Path(__file__).resolve().parents[2]))

from slt.models.temporal import TemporalEncoder, TextSeq2SeqDecoder


def test_temporal_encoder_output_shape():
    encoder = TemporalEncoder(d_model=16, nhead=4, nlayers=2, dim_feedforward=32, dropout=0.0)
    sequence = torch.randn(3, 5, 16)

    output = encoder(sequence)

    assert output.shape == sequence.shape


def test_temporal_encoder_respects_padding_mask():
    torch.manual_seed(0)
    encoder = TemporalEncoder(d_model=8, nhead=2, nlayers=1, dim_feedforward=16, dropout=0.0)

    full_sequence = torch.randn(1, 3, 8)
    truncated_sequence = full_sequence[:, :2, :]
    padding_mask = torch.tensor([[False, False, True]])  # last position is padding

    masked_output = encoder(full_sequence, src_key_padding_mask=padding_mask)
    truncated_output = encoder(truncated_sequence)

    torch.testing.assert_close(masked_output[:, :2, :], truncated_output, rtol=1e-4, atol=1e-4)


def test_text_seq2seq_decoder_forward_shapes():
    decoder = TextSeq2SeqDecoder(
        d_model=16,
        vocab_size=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        pad_token_id=0,
        eos_token_id=1,
    )

    enc_out = torch.randn(2, 3, 16)
    encoder_mask = torch.ones(2, 3, dtype=torch.long)
    labels = torch.tensor([[1, 2, 3, 4], [5, 6, 7, -100]])
    decoder_mask = (labels != -100).long()

    outputs = decoder(
        enc_out,
        encoder_attention_mask=encoder_mask,
        labels=labels,
        decoder_attention_mask=decoder_mask,
    )

    assert outputs.logits.shape == (2, labels.shape[1], 32)
    assert outputs.loss is not None


class CustomT5Decoder(T5ForConditionalGeneration):
    def __init__(self, config: PretrainedConfig) -> None:
        super().__init__(config)
        self.custom_attribute = True


def test_text_seq2seq_decoder_allows_custom_auto_model_and_half_precision():
    decoder = TextSeq2SeqDecoder(
        d_model=16,
        vocab_size=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        pad_token_id=0,
        eos_token_id=1,
        auto_model_cls=CustomT5Decoder,
        half_precision=True,
    )

    assert isinstance(decoder.model, CustomT5Decoder)
    assert getattr(decoder.model, "custom_attribute", False)
    for parameter in decoder.model.parameters():
        assert parameter.dtype == torch.float16


def test_text_seq2seq_decoder_ties_embeddings():
    decoder = TextSeq2SeqDecoder(
        d_model=16,
        vocab_size=32,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        pad_token_id=0,
        eos_token_id=1,
        tie_embeddings=True,
    )

    input_emb = decoder.model.get_input_embeddings()
    output_emb = decoder.model.get_output_embeddings()
    assert input_emb is not None and output_emb is not None
    assert input_emb.weight.data_ptr() == output_emb.weight.data_ptr()
