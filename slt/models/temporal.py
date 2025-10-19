"""Temporal encoder and lightweight decoder utilities."""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor, nn


class TemporalEncoder(nn.Module):
    """Thin wrapper around :class:`torch.nn.TransformerEncoder`.

    Parameters
    ----------
    d_model:
        Embedding dimension of the sequence tokens.
    nhead:
        Number of attention heads.
    nlayers:
        Number of encoder layers stacked.
    dim_feedforward:
        Dimension of the feedforward network within each encoder layer.
    dropout:
        Dropout probability applied inside the encoder layers.
    """

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        nlayers: int = 6,
        *,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(
        self,
        sequence: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode the temporal sequence.

        Parameters
        ----------
        sequence:
            Input tensor of shape ``(batch, time, d_model)``.
        src_key_padding_mask:
            Optional boolean mask of shape ``(batch, time)`` where ``True``
            indicates padded (ignored) positions.
        """

        return self.encoder(sequence, src_key_padding_mask=src_key_padding_mask)


class TextDecoderStub(nn.Module):
    """Minimal decoder that aggregates encoder outputs.

    The stub averages encoder hidden states over the temporal dimension and
    projects the resulting representation to a vocabulary-sized logits tensor.
    An optional padding mask can be provided to ignore padded positions when
    computing the mean.
    """

    def __init__(self, d_model: int = 512, vocab_size: int = 32_000) -> None:
        super().__init__()
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        enc_out: Tensor,
        padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Project the temporal average of ``enc_out`` into vocabulary logits."""

        if padding_mask is None:
            pooled = enc_out.mean(dim=1)
        else:
            if padding_mask.dtype != torch.bool:
                padding_mask = padding_mask.to(torch.bool)
            valid_mask = ~padding_mask
            weights = valid_mask.unsqueeze(-1)
            summed = (enc_out * weights).sum(dim=1)
            counts = valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
            pooled = summed / counts

        return self.lm_head(pooled)
