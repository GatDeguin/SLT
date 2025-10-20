"""Reusable training-time model wrappers."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import AutoConfig, PreTrainedTokenizerBase

from slt.models import MultiStreamEncoder, TextSeq2SeqDecoder, ViTConfig, load_dinov2_backbone
from slt.models.single_signer import load_single_signer_components

from .configuration import ModelConfig


def _import_symbol(path: str) -> Any:
    if ":" in path:
        module_name, _, qualname = path.partition(":")
    else:
        module_name, _, qualname = path.rpartition(".")
    if not module_name or not qualname:
        raise ValueError(
            "Custom decoder class must be specified as 'module:ClassName' or 'module.ClassName'."
        )
    module = importlib.import_module(module_name)
    try:
        return getattr(module, qualname)
    except AttributeError as exc:
        raise ImportError(f"Unable to locate '{qualname}' in module '{module_name}'") from exc


class MultiStreamClassifier(nn.Module):
    """Convenience wrapper around the encoder/decoder pair."""

    def __init__(self, config: ModelConfig, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()

        pretrained = (config.pretrained or "").strip().lower()
        if pretrained and pretrained not in {"none", "false"}:
            if pretrained not in {"single_signer", "single-signer"}:
                raise ValueError(
                    "Unsupported pretrained identifier. Only 'single_signer' is available."
                )
            encoder, decoder, metadata = load_single_signer_components(
                tokenizer,
                checkpoint_path=config.pretrained_checkpoint,
                map_location=torch.device("cpu"),
                strict=True,
            )
            self.encoder = encoder
            self.decoder = decoder
            setattr(self, "pretrained_metadata", metadata)
            return

        vit_config = ViTConfig(image_size=config.image_size)
        temporal_kwargs = {
            "nhead": config.temporal_nhead,
            "nlayers": config.temporal_layers,
            "dim_feedforward": config.temporal_dim_feedforward,
            "dropout": config.temporal_dropout,
        }

        backbone_specs = {
            "face": (config.face_backbone, config.freeze_face_backbone),
            "hand_left": (config.hand_left_backbone, config.freeze_hand_left_backbone),
            "hand_right": (config.hand_right_backbone, config.freeze_hand_right_backbone),
        }
        external_backbones: Dict[str, torch.nn.Module] = {}
        for stream, (spec, freeze_flag) in backbone_specs.items():
            if spec:
                external_backbones[stream] = load_dinov2_backbone(spec, freeze=freeze_flag)

        self.encoder = MultiStreamEncoder(
            backbone_config=vit_config,
            projector_dim=config.projector_dim,
            d_model=config.d_model,
            pose_dim=3 * config.pose_landmarks,
            positional_num_positions=config.sequence_length,
            projector_dropout=config.projector_dropout,
            fusion_dropout=config.fusion_dropout,
            temporal_kwargs=temporal_kwargs,
            backbones=external_backbones if external_backbones else None,
        )
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if not vocab_size:
            vocab_size = len(tokenizer)

        decoder_config = None
        if config.decoder_config:
            decoder_config = AutoConfig.from_pretrained(config.decoder_config)
            hidden_size = TextSeq2SeqDecoder.hidden_size_from_config(decoder_config)
            if hidden_size is not None and hidden_size != config.d_model:
                raise ValueError(
                    "Decoder configuration hidden size does not match encoder dimensionality: "
                    f"expected {config.d_model}, got {hidden_size}."
                )

        decoder_model_name = None if decoder_config is not None else config.decoder_model

        base_kwargs = {
            "d_model": config.d_model,
            "vocab_size": int(vocab_size),
            "num_layers": config.decoder_layers,
            "num_heads": config.decoder_heads,
            "dropout": config.decoder_dropout,
            "pad_token_id": pad_id,
            "eos_token_id": eos_id,
        }

        if config.decoder_class:
            if decoder_model_name or decoder_config is not None:
                raise ValueError(
                    "decoder_class cannot be combined with decoder_model or decoder_config"
                )
            decoder_cls = _import_symbol(config.decoder_class)
            if not isinstance(decoder_cls, type) or not issubclass(decoder_cls, nn.Module):
                raise TypeError("Custom decoder class must inherit from torch.nn.Module")
            kwargs = dict(base_kwargs)
            kwargs.update(config.decoder_kwargs)
            self.decoder = decoder_cls(**kwargs)
        else:
            self.decoder = TextSeq2SeqDecoder(
                **base_kwargs,
                pretrained_model_name_or_path=decoder_model_name,
                config=decoder_config,
                config_kwargs=config.decoder_kwargs,
            )

    def forward(
        self,
        *,
        face: torch.Tensor,
        hand_l: torch.Tensor,
        hand_r: torch.Tensor,
        pose: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        miss_mask_hl: Optional[torch.Tensor] = None,
        miss_mask_hr: Optional[torch.Tensor] = None,
        pose_conf_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded = self.encoder(
            face,
            hand_l,
            hand_r,
            pose,
            pad_mask=pad_mask,
            miss_mask_hl=miss_mask_hl,
            miss_mask_hr=miss_mask_hr,
            pose_conf_mask=pose_conf_mask,
        )
        if encoder_attention_mask is None and pad_mask is not None:
            encoder_attention_mask = pad_mask.to(torch.long)
        return self.decoder(
            encoded,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

    def generate(
        self,
        *,
        face: torch.Tensor,
        hand_l: torch.Tensor,
        hand_r: torch.Tensor,
        pose: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        miss_mask_hl: Optional[torch.Tensor] = None,
        miss_mask_hr: Optional[torch.Tensor] = None,
        pose_conf_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs: Any,
    ) -> torch.LongTensor:
        encoded = self.encoder(
            face,
            hand_l,
            hand_r,
            pose,
            pad_mask=pad_mask,
            miss_mask_hl=miss_mask_hl,
            miss_mask_hr=miss_mask_hr,
            pose_conf_mask=pose_conf_mask,
        )
        if encoder_attention_mask is None and pad_mask is not None:
            encoder_attention_mask = pad_mask.to(torch.long)
        return self.decoder.generate(
            encoded,
            encoder_attention_mask=encoder_attention_mask,
            **generation_kwargs,
        )

