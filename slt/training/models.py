"""Reusable training-time model wrappers."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch import nn
from transformers import AutoConfig, PreTrainedTokenizerBase

from slt.models import MultiStreamEncoder, TextSeq2SeqDecoder, ViTConfig, load_dinov2_backbone
from slt.models.mska import MSKAEncoder
from slt.models.single_signer import load_single_signer_components

from .configuration import ModelConfig


@dataclass
class ClassifierOutput:
    """Wrapper combining decoder outputs with auxiliary logits."""

    decoder: Any
    auxiliary: Optional[Dict[str, Any]] = None

    @property
    def logits(self) -> Any:
        if hasattr(self.decoder, "logits"):
            return self.decoder.logits
        return self.decoder

    @property
    def loss(self) -> Any:
        return getattr(self.decoder, "loss", None)

    def __getattr__(self, name: str) -> Any:  # pragma: no cover - passthrough
        try:
            return getattr(self.decoder, name)
        except AttributeError as exc:  # pragma: no cover - defensive
            raise AttributeError(name) from exc


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
            self._mska_enabled = False
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

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if not vocab_size:
            vocab_size = len(tokenizer)

        mska_encoder: Optional[MSKAEncoder] = None
        if config.use_mska:
            mska_vocab = config.mska_ctc_vocab or vocab_size
            mska_encoder = MSKAEncoder(
                input_dim=config.mska_input_dim,
                embed_dim=config.projector_dim,
                stream_names=("face", "hand_left", "hand_right", "pose"),
                num_heads=config.mska_heads,
                ff_multiplier=config.mska_ff_multiplier,
                dropout=config.mska_dropout,
                ctc_vocab_size=int(mska_vocab),
                detach_teacher=config.mska_detach_teacher,
                stream_attention_heads=config.mska_stream_heads,
                stream_temporal_blocks=config.mska_temporal_blocks,
                stream_temporal_kernel=config.mska_temporal_kernel,
                stream_temporal_dilation=config.mska_temporal_dilation,
                use_global_attention=config.mska_use_sgr,
                global_attention_activation=config.mska_sgr_activation,
                global_attention_mix=config.mska_sgr_mix,
                global_attention_shared=config.mska_sgr_shared,
            )

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
            mska=mska_encoder,
            mska_gloss_hidden_dim=config.mska_gloss_hidden_dim,
            mska_gloss_activation=config.mska_gloss_activation,
            mska_gloss_dropout=config.mska_gloss_dropout,
        )
        self._mska_enabled = mska_encoder is not None
        self._mska_gloss_fusion = (config.mska_gloss_fusion or "add").strip().lower()
        if self._mska_gloss_fusion not in {"add", "concat", "none"}:
            raise ValueError(
                "mska_gloss_fusion must be one of {'add', 'concat', 'none'}"
            )

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

    @staticmethod
    def _prepare_keypoint_streams(
        payload: Dict[str, Any]
    ) -> Optional[Dict[str, Dict[str, torch.Tensor]]]:
        mapping = {
            "face": (
                "keypoints_face",
                "keypoints_face_mask",
                "keypoints_face_frame_mask",
            ),
            "hand_left": (
                "keypoints_hand_l",
                "keypoints_hand_l_mask",
                "keypoints_hand_l_frame_mask",
            ),
            "hand_right": (
                "keypoints_hand_r",
                "keypoints_hand_r_mask",
                "keypoints_hand_r_frame_mask",
            ),
            "pose": (
                "keypoints_body",
                "keypoints_body_mask",
                "keypoints_body_frame_mask",
            ),
        }

        streams: Dict[str, Dict[str, torch.Tensor]] = {}
        for name, keys in mapping.items():
            points = payload.pop(keys[0], None)
            mask = payload.pop(keys[1], None)
            frame_mask = payload.pop(keys[2], None)
            if points is None or not isinstance(points, torch.Tensor) or points.numel() == 0:
                continue
            stream_payload: Dict[str, torch.Tensor] = {"points": points}
            if isinstance(mask, torch.Tensor):
                stream_payload["mask"] = mask
            if isinstance(frame_mask, torch.Tensor):
                stream_payload["frame_mask"] = frame_mask
            streams[name] = stream_payload

        if "pose" not in streams:
            points = payload.pop("keypoints", None)
            if isinstance(points, torch.Tensor) and points.numel() != 0:
                mask = payload.pop("keypoints_mask", None)
                frame_mask = payload.pop("keypoints_frame_mask", None)
                stream_payload = {"points": points}
                if isinstance(mask, torch.Tensor):
                    stream_payload["mask"] = mask
                if isinstance(frame_mask, torch.Tensor):
                    stream_payload["frame_mask"] = frame_mask
                streams["pose"] = stream_payload

        return streams or None

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
        **extra_inputs: Any,
    ) -> ClassifierOutput:
        keypoint_streams = None
        encoder_kwargs = extra_inputs
        if self._mska_enabled:
            encoder_kwargs = dict(extra_inputs)
            keypoint_streams = self._prepare_keypoint_streams(encoder_kwargs)
        encoded = self.encoder(
            face,
            hand_l,
            hand_r,
            pose,
            pad_mask=pad_mask,
            miss_mask_hl=miss_mask_hl,
            miss_mask_hr=miss_mask_hr,
            pose_conf_mask=pose_conf_mask,
            keypoint_streams=keypoint_streams,
            **encoder_kwargs,
        )
        if encoder_attention_mask is None and pad_mask is not None:
            encoder_attention_mask = pad_mask.to(torch.long)
        gloss_sequence = getattr(self.encoder, "last_gloss_sequence", None)
        gloss_mask = getattr(self.encoder, "last_gloss_mask", None)
        encoded, encoder_attention_mask = self._merge_gloss_sequence(
            encoded, encoder_attention_mask, gloss_sequence, gloss_mask
        )
        decoder_output = self.decoder(
            encoded,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )
        auxiliary = None
        if self._mska_enabled and self.encoder.mska_encoder is not None:
            mska_output = self.encoder.last_mska_output
            if mska_output is not None:
                auxiliary = self.encoder.mska_encoder.auxiliary_logits(mska_output)
        return ClassifierOutput(decoder=decoder_output, auxiliary=auxiliary)

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
        **kwargs: Any,
    ) -> torch.LongTensor:
        encoder_extra_keys = {
            "keypoints",
            "keypoints_mask",
            "keypoints_frame_mask",
            "keypoints_body",
            "keypoints_body_mask",
            "keypoints_body_frame_mask",
            "keypoints_hand_l",
            "keypoints_hand_l_mask",
            "keypoints_hand_l_frame_mask",
            "keypoints_hand_r",
            "keypoints_hand_r_mask",
            "keypoints_hand_r_frame_mask",
            "keypoints_face",
            "keypoints_face_mask",
            "keypoints_face_frame_mask",
            "keypoints_lengths",
            "ctc_labels",
            "ctc_mask",
            "ctc_lengths",
            "gloss_sequences",
            "gloss_texts",
        }
        encoder_kwargs = {
            name: value for name, value in kwargs.items() if name in encoder_extra_keys
        }
        decoder_kwargs = {
            name: value for name, value in kwargs.items() if name not in encoder_extra_keys
        }
        keypoint_streams = None
        if self._mska_enabled:
            encoder_kwargs = dict(encoder_kwargs)
            keypoint_streams = self._prepare_keypoint_streams(encoder_kwargs)
        encoded = self.encoder(
            face,
            hand_l,
            hand_r,
            pose,
            pad_mask=pad_mask,
            miss_mask_hl=miss_mask_hl,
            miss_mask_hr=miss_mask_hr,
            pose_conf_mask=pose_conf_mask,
            keypoint_streams=keypoint_streams,
            **encoder_kwargs,
        )
        if encoder_attention_mask is None and pad_mask is not None:
            encoder_attention_mask = pad_mask.to(torch.long)
        gloss_sequence = getattr(self.encoder, "last_gloss_sequence", None)
        gloss_mask = getattr(self.encoder, "last_gloss_mask", None)
        encoded, encoder_attention_mask = self._merge_gloss_sequence(
            encoded, encoder_attention_mask, gloss_sequence, gloss_mask
        )
        return self.decoder.generate(
            encoded,
            encoder_attention_mask=encoder_attention_mask,
            **decoder_kwargs,
        )

    def _merge_gloss_sequence(
        self,
        encoder_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        gloss_states: Optional[torch.Tensor],
        gloss_mask: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        if gloss_states is None or self._mska_gloss_fusion == "none":
            return encoder_states, attention_mask

        if self._mska_gloss_fusion == "add":
            if gloss_states.shape != encoder_states.shape:
                raise ValueError(
                    "Gloss sequence must match encoder hidden states when fusion is 'add'"
                )
            merged = encoder_states + gloss_states
            if gloss_mask is None:
                return merged, attention_mask
            gloss_bool = gloss_mask.to(dtype=torch.bool)
            if attention_mask is None:
                return merged, gloss_bool
            base_bool = attention_mask.to(dtype=torch.bool)
            combined_bool = base_bool & gloss_bool
            if attention_mask.dtype == torch.bool:
                return merged, combined_bool
            return merged, combined_bool.to(dtype=attention_mask.dtype)

        if self._mska_gloss_fusion == "concat":
            merged = torch.cat([encoder_states, gloss_states], dim=1)
            batch, gloss_length, _ = gloss_states.shape
            device = merged.device
            if gloss_mask is None:
                if attention_mask is not None:
                    gloss_mask_tensor = torch.ones(
                        batch,
                        gloss_length,
                        dtype=attention_mask.dtype,
                        device=device,
                    )
                else:
                    gloss_mask_tensor = torch.ones(
                        batch,
                        gloss_length,
                        dtype=torch.bool,
                        device=device,
                    )
            else:
                gloss_mask_tensor = gloss_mask.to(device=device)
                if (
                    attention_mask is not None
                    and gloss_mask_tensor.dtype != attention_mask.dtype
                ):
                    gloss_mask_tensor = gloss_mask_tensor.to(dtype=attention_mask.dtype)
            if attention_mask is None:
                return merged, gloss_mask_tensor
            return (
                merged,
                torch.cat([attention_mask, gloss_mask_tensor], dim=1),
            )

        raise RuntimeError(f"Unsupported mska_gloss_fusion mode '{self._mska_gloss_fusion}'")

