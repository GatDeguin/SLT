"""High level multi-stream encoder stub for SLT."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

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
    ) -> None:
        super().__init__()
        temporal_kwargs = temporal_kwargs or {}

        self.face_backbone = ViTSmallPatch16(backbone_config)
        self.hand_backbone_left = ViTSmallPatch16(backbone_config)
        self.hand_backbone_right = ViTSmallPatch16(backbone_config)

        backbone_dim = self.face_backbone.config.embed_dim

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

    def forward(
        self,
        face: Tensor,
        hand_l: Tensor,
        hand_r: Tensor,
        pose: Tensor,
        *,
        pad_mask: Optional[Tensor] = None,
        miss_mask_hl: Optional[Tensor] = None,
        miss_mask_hr: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode multi-stream inputs returning a temporal sequence tensor."""

        face_feats = self._encode_backbone(self.face_backbone, face)
        hand_l_feats = self._encode_backbone(self.hand_backbone_left, hand_l)
        hand_r_feats = self._encode_backbone(self.hand_backbone_right, hand_r)

        hand_l_feats = self._apply_missing_mask(hand_l_feats, miss_mask_hl, stream="hand_left")
        hand_r_feats = self._apply_missing_mask(hand_r_feats, miss_mask_hr, stream="hand_right")

        face_proj = self.face_projector(face_feats)
        hand_l_proj = self.hand_left_projector(hand_l_feats)
        hand_r_proj = self.hand_right_projector(hand_r_feats)
        pose_proj = self.pose_projector(self._ensure_pose_shape(pose))

        fused = self.fusion(face_proj, hand_l_proj, hand_r_proj, pose_proj)
        fused = self.positional(fused)

        src_key_padding_mask = self._convert_padding_mask(pad_mask)
        encoded = self.temporal(fused, src_key_padding_mask=src_key_padding_mask)
        return encoded

    @staticmethod
    def _encode_backbone(backbone: ViTSmallPatch16, stream: Tensor) -> Tensor:
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
    ) -> Tensor:
        if mask is None:
            return features
        mask_bool = self._ensure_bool_mask(mask, features.shape[:2])
        return self._mask_hand_features(features, mask_bool, stream=stream)

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
        return features

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
