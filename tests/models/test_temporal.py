"""Unit tests for temporal encoder and decoder stubs."""

import sys
from pathlib import Path

import torch

sys.path.append(str(Path(__file__).resolve().parents[2]))

from slt.models.temporal import TemporalEncoder, TextDecoderStub


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


def test_text_decoder_stub_masked_mean():
    decoder = TextDecoderStub(d_model=4, vocab_size=10)

    enc_out = torch.tensor(
        [
            [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]],
            [[2.0, 4.0, 6.0, 8.0], [0.0, 0.0, 0.0, 0.0]],
        ]
    )
    padding_mask = torch.tensor([[False, False], [False, True]])

    logits = decoder(enc_out, padding_mask=padding_mask)

    # First sequence averages both steps, second ignores the padded second step
    expected_means = torch.tensor([[3.0, 4.0, 5.0, 6.0], [2.0, 4.0, 6.0, 8.0]])
    expected_logits = decoder.lm_head(expected_means)

    torch.testing.assert_close(logits, expected_logits)
