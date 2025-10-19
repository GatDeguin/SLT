"""Temporal encoder and lightweight decoder utilities."""

from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import Tensor, nn
from transformers import AutoModelForSeq2SeqLM, PretrainedConfig, T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput


class TemporalEncoder(nn.Module):
    """Thin wrapper around :class:`torch.nn.TransformerEncoder`."""

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
        """Encode the temporal sequence."""

        return self.encoder(sequence, src_key_padding_mask=src_key_padding_mask)


class TextSeq2SeqDecoder(nn.Module):
    """Seq2seq decoder backed by :mod:`transformers` models."""

    def __init__(
        self,
        *,
        d_model: int = 512,
        vocab_size: int = 32_000,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        pretrained_model_name_or_path: Optional[str] = None,
        config: Optional[PretrainedConfig] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads for the decoder configuration.")

        self.model = self._build_model(
            d_model=d_model,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config=config,
            config_kwargs=config_kwargs or {},
        )

    @staticmethod
    def _build_model(
        *,
        d_model: int,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        pad_token_id: int,
        eos_token_id: int,
        pretrained_model_name_or_path: Optional[str],
        config: Optional[PretrainedConfig],
        config_kwargs: Dict[str, Any],
    ) -> nn.Module:
        if config is not None:
            return AutoModelForSeq2SeqLM.from_config(config)

        if pretrained_model_name_or_path is not None:
            model = AutoModelForSeq2SeqLM.from_pretrained(pretrained_model_name_or_path)
            hidden_size = getattr(model.config, "d_model", None)
            if hidden_size is not None and hidden_size != d_model:
                raise ValueError(
                    "Loaded model hidden size does not match encoder dimensionality: "
                    f"expected {d_model}, got {hidden_size}."
                )
            return model

        default_config: Dict[str, Any] = {
            "vocab_size": vocab_size,
            "d_model": d_model,
            "d_kv": d_model // num_heads,
            "d_ff": config_kwargs.pop("d_ff", d_model * 4),
            "num_layers": num_layers,
            "num_decoder_layers": config_kwargs.pop("num_decoder_layers", num_layers),
            "num_heads": num_heads,
            "dropout_rate": dropout,
            "layer_norm_epsilon": config_kwargs.pop("layer_norm_epsilon", 1e-6),
            "initializer_factor": config_kwargs.pop("initializer_factor", 1.0),
            "pad_token_id": pad_token_id,
            "eos_token_id": eos_token_id,
            "decoder_start_token_id": config_kwargs.pop("decoder_start_token_id", pad_token_id),
        }
        default_config.update(config_kwargs)
        t5_config = T5Config(**default_config)
        return T5ForConditionalGeneration(t5_config)

    @property
    def config(self) -> PretrainedConfig:  # pragma: no cover - simple property
        return self.model.config

    def forward(
        self,
        encoder_hidden_states: Tensor,
        *,
        encoder_attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        **model_kwargs: Any,
    ) -> Seq2SeqLMOutput:
        """Forward pass delegating to the underlying HF model."""

        attention_mask = self._prepare_attention_mask(encoder_attention_mask)
        return self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            **model_kwargs,
        )

    def generate(
        self,
        encoder_hidden_states: Tensor,
        *,
        encoder_attention_mask: Optional[Tensor] = None,
        **generation_kwargs: Any,
    ) -> torch.LongTensor:
        """Generate sequences conditioned on the encoder hidden states."""

        attention_mask = self._prepare_attention_mask(encoder_attention_mask)
        return self.model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    @staticmethod
    def _prepare_attention_mask(mask: Optional[Tensor]) -> Optional[Tensor]:
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            return mask.to(dtype=torch.long)
        return mask
