"""High level multi-stream encoder stub for SLT."""

from __future__ import annotations

import inspect

from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor

from .backbones import ViTConfig, ViTSmallPatch16
from .modules import FuseConcatLinear, PositionalEncodingLearned, StreamProjector
from .temporal import TemporalEncoder

__all__ = ["MultiStreamEncoder"]


class MultiStreamEncoder(torch.nn.Module):
    """Composable encoder that processes face, hand and pose streams."""

    def __init__(
        self,
        *,
        backbone_config: Optional[ViTConfig] = None,
        projector_dim: int = 256,
        d_model: int = 512,
        pose_dim: int = 39,
        positional_num_positions: int = 512,
        projector_dropout: float = 0.0,
        fusion_dropout: float = 0.0,
        fusion_bias: bool = True,
        temporal_kwargs: Optional[Dict[str, Any]] = None,
        backbones: Optional[Mapping[str, torch.nn.Module]] = None,
    ) -> None:
        super().__init__()
        temporal_kwargs = temporal_kwargs or {}

        backbones = backbones or {}
        self.face_backbone = self._resolve_backbone(
            backbones, "face", backbone_config
        )
        self.hand_backbone_left = self._resolve_backbone(
            backbones, "hand_left", backbone_config
        )
        self.hand_backbone_right = self._resolve_backbone(
            backbones, "hand_right", backbone_config
        )

        backbone_dim = self._infer_backbone_dim(self.face_backbone)

        self.face_projector = StreamProjector(
            backbone_dim, projector_dim, dropout=projector_dropout
        )
        self.hand_left_projector = StreamProjector(
            backbone_dim, projector_dim, dropout=projector_dropout
        )
        self.hand_right_projector = StreamProjector(
            backbone_dim, projector_dim, dropout=projector_dropout
        )
        self.pose_projector = StreamProjector(
            pose_dim, projector_dim, dropout=projector_dropout
        )

        self.fusion = FuseConcatLinear(
            [projector_dim, projector_dim, projector_dim, projector_dim],
            d_model,
            dropout=fusion_dropout,
            bias=fusion_bias,
        )
        self.positional = PositionalEncodingLearned(positional_num_positions, d_model)
        self.temporal = TemporalEncoder(d_model=d_model, **temporal_kwargs)
        self._last_combined_hand_mask: Optional[Tensor] = None

    def forward(
        self,
        face: Tensor,
        hand_l: Tensor,
        hand_r: Tensor,
        pose: Tensor,
        pad_mask: Optional[Tensor] = None,
        miss_mask_hl: Optional[Tensor] = None,
        miss_mask_hr: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode multi-stream inputs returning a temporal sequence tensor."""

        face_feats = self._encode_backbone(self.face_backbone, face)
        hand_l_feats = self._encode_backbone(self.hand_backbone_left, hand_l)
        hand_r_feats = self._encode_backbone(self.hand_backbone_right, hand_r)

        hand_l_feats, hand_l_mask = self._apply_missing_mask(
            hand_l_feats, miss_mask_hl, stream="hand_left"
        )
        hand_r_feats, hand_r_mask = self._apply_missing_mask(
            hand_r_feats, miss_mask_hr, stream="hand_right"
        )

        face_proj = self.face_projector(face_feats)
        hand_l_proj = self.hand_left_projector(hand_l_feats)
        hand_r_proj = self.hand_right_projector(hand_r_feats)
        pose_proj = self.pose_projector(self._ensure_pose_shape(pose))

        combined_hand_mask = self._combine_missing_masks(hand_l_mask, hand_r_mask)
        self._last_combined_hand_mask = combined_hand_mask

        fused = self._call_fusion(
            (face_proj, hand_l_proj, hand_r_proj, pose_proj), combined_hand_mask
        )
        fused = self.positional(fused)

        src_key_padding_mask = self._convert_padding_mask(pad_mask)
        encoded = self.temporal(fused, src_key_padding_mask=src_key_padding_mask)
        return encoded

    @staticmethod
    def _encode_backbone(backbone: torch.nn.Module, stream: Tensor) -> Tensor:
        if stream.dim() != 5:
            raise ValueError(
                "Backbone inputs must have shape (batch, time, channels, height, width)."
            )
        batch, time = stream.shape[:2]
        flat = stream.view(batch * time, *stream.shape[2:])
        features = backbone(flat)
        return features.view(batch, time, -1)

    def _apply_missing_mask(
        self, features: Tensor, mask: Optional[Tensor], *, stream: str
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if mask is None:
            return features, None
        mask_bool = self._ensure_bool_mask(mask, features.shape[:2])
        masked = self._mask_hand_features(features, mask_bool, stream=stream)
        return masked, mask_bool

    @staticmethod
    def _ensure_bool_mask(mask: Tensor, expected_shape: Iterable[int]) -> Tensor:
        mask = mask.to(dtype=torch.bool)
        expected = tuple(expected_shape)
        if tuple(mask.shape[-2:]) == expected:
            return mask
        if mask.dim() == 2 and mask.shape[0] == expected[0] and mask.shape[1] == expected[1]:
            return mask
        raise ValueError(
            "Missing data masks must broadcast to (batch, time); received shape "
            f"{tuple(mask.shape)}"
        )

    def _mask_hand_features(
        self, features: Tensor, mask: Tensor, *, stream: str
    ) -> Tensor:  # pragma: no cover - extension hook
        if features.dim() < 2:
            raise ValueError(
                f"Masked features for stream '{stream}' must have at least two dimensions; "
                f"received shape {tuple(features.shape)}"
            )

        mask_device = mask.device if mask.device == features.device else features.device
        mask_bool = mask.to(device=mask_device, dtype=torch.bool)

        while mask_bool.dim() < features.dim():
            mask_bool = mask_bool.unsqueeze(-1)

        expanded_mask = mask_bool
        if any(m != f for m, f in zip(expanded_mask.shape, features.shape)):
            expanded_mask = expanded_mask.expand(features.shape)

        return features.masked_fill(expanded_mask, 0)

    def _combine_missing_masks(
        self, left: Optional[Tensor], right: Optional[Tensor]
    ) -> Optional[Tensor]:
        if left is None and right is None:
            return None
        if left is None:
            combined = right.clone()
        elif right is None:
            combined = left.clone()
        else:
            combined = torch.logical_or(left, right)
        return combined.to(dtype=torch.bool)

    def _call_fusion(
        self, streams: Sequence[Tensor], mask: Optional[Tensor]
    ) -> Tensor:
        if mask is not None:
            target_device = streams[0].device
            if mask.device != target_device:
                mask = mask.to(device=target_device)
            if self._fusion_accepts_mask():
                return self.fusion(*streams, mask=mask)
        return self.fusion(*streams)

    def _fusion_accepts_mask(self) -> bool:
        forward = self.fusion.forward  # type: ignore[assignment]
        signature = inspect.signature(forward)
        for parameter in signature.parameters.values():
            if parameter.name == "mask" and parameter.kind in (
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
                inspect.Parameter.KEYWORD_ONLY,
            ):
                return True
        return False

    @staticmethod
    def _resolve_backbone(
        backbones: Mapping[str, torch.nn.Module],
        stream: str,
        config: Optional[ViTConfig],
    ) -> torch.nn.Module:
        backbone = backbones.get(stream)
        if backbone is None:
            backbone = ViTSmallPatch16(config)
        if hasattr(backbone, "as_backbone") and callable(backbone.as_backbone):
            backbone = backbone.as_backbone()  # type: ignore[assignment]
        return backbone

    @staticmethod
    def _infer_backbone_dim(backbone: torch.nn.Module) -> int:
        if hasattr(backbone, "config") and hasattr(backbone.config, "embed_dim"):
            return int(backbone.config.embed_dim)
        if hasattr(backbone, "embed_dim"):
            return int(getattr(backbone, "embed_dim"))
        if hasattr(backbone, "num_features"):
            return int(getattr(backbone, "num_features"))
        raise AttributeError("Unable to infer backbone embedding dimension")

    @staticmethod
    def _ensure_pose_shape(pose: Tensor) -> Tensor:
        if pose.dim() != 3:
            raise ValueError(
                "Pose tensor must have shape (batch, time, pose_dim), received "
                f"{tuple(pose.shape)}"
            )
        return pose

    @staticmethod
    def _convert_padding_mask(mask: Optional[Tensor]) -> Optional[Tensor]:
        if mask is None:
            return None
        mask_bool = mask.to(dtype=torch.bool)
        return ~mask_bool
