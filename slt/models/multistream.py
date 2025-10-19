"""High level multi-stream encoder stub for SLT."""

from __future__ import annotations

import inspect
import warnings
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, Iterable, Mapping, Optional, Sequence, Tuple

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
        self._last_padding_mask: Optional[Tensor] = None
        self._observers: DefaultDict[str, list[Callable[[str, Tensor], None]]] = defaultdict(list)

        self._auto_configure_components()

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
        self._emit("face.backbone", face_feats)
        hand_l_feats = self._encode_backbone(self.hand_backbone_left, hand_l)
        hand_r_feats = self._encode_backbone(self.hand_backbone_right, hand_r)

        hand_l_feats, hand_l_mask = self._apply_missing_mask(
            hand_l_feats, miss_mask_hl, stream="hand_left"
        )
        hand_r_feats, hand_r_mask = self._apply_missing_mask(
            hand_r_feats, miss_mask_hr, stream="hand_right"
        )
        self._emit("hand.left.backbone", hand_l_feats)
        self._emit("hand.right.backbone", hand_r_feats)

        face_proj = self.face_projector(face_feats)
        hand_l_proj = self.hand_left_projector(hand_l_feats)
        hand_r_proj = self.hand_right_projector(hand_r_feats)
        pose_proj = self.pose_projector(self._ensure_pose_shape(pose))
        self._emit("pose.projector", pose_proj)

        combined_hand_mask = self._combine_missing_masks(hand_l_mask, hand_r_mask)
        self._last_combined_hand_mask = combined_hand_mask
        if combined_hand_mask is not None:
            self._emit("hand.mask", combined_hand_mask.to(torch.float32))

        fused = self._call_fusion(
            (face_proj, hand_l_proj, hand_r_proj, pose_proj), combined_hand_mask
        )
        self._emit("fusion.output", fused)
        fused = self.positional(fused)

        src_key_padding_mask = self._convert_padding_mask(pad_mask)
        self._last_padding_mask = src_key_padding_mask
        if src_key_padding_mask is not None:
            self._emit("pad.mask", src_key_padding_mask.to(torch.float32))
        encoded = self.temporal(fused, src_key_padding_mask=src_key_padding_mask)
        self._emit("temporal.output", encoded)
        return encoded

    # ------------------------------------------------------------------
    # Observer API
    # ------------------------------------------------------------------
    def register_observer(self, name: str, hook: Callable[[str, Tensor], None]) -> None:
        """Register an observer that receives telemetry for ``name``."""

        self._observers[name].append(hook)

    def remove_observer(self, name: str, hook: Callable[[str, Tensor], None]) -> None:
        """Remove a previously registered observer."""

        hooks = self._observers.get(name)
        if not hooks:
            return
        try:
            hooks.remove(hook)
        except ValueError:  # pragma: no cover - defensive.
            return
        if not hooks:
            self._observers.pop(name, None)

    def clear_observers(self, name: Optional[str] = None) -> None:
        """Remove observers globally or for ``name``."""

        if name is None:
            self._observers.clear()
        else:
            self._observers.pop(name, None)

    @property
    def last_combined_hand_mask(self) -> Optional[Tensor]:
        """Return the most recent combined hand mask."""

        return self._last_combined_hand_mask

    @property
    def last_padding_mask(self) -> Optional[Tensor]:
        """Return the most recent padding mask used during attention."""

        return self._last_padding_mask

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _emit(self, name: str, tensor: Tensor) -> None:
        hooks = self._observers.get(name)
        if not hooks:
            return
        for hook in list(hooks):
            try:
                hook(name, tensor)
            except Exception as exc:  # pragma: no cover - telemetry hooks shouldn't raise
                warnings.warn(
                    f"Observer for '{name}' raised an exception: {exc}",
                    RuntimeWarning,
                    stacklevel=2,
                )

    def _auto_configure_components(self) -> None:
        projector_context = (
            (self.face_projector, self.face_backbone),
            (self.hand_left_projector, self.hand_backbone_left),
            (self.hand_right_projector, self.hand_backbone_right),
            (self.pose_projector, None),
        )
        for module, backbone in projector_context:
            self._maybe_install_dinov2(module, backbone)
        self._maybe_install_dinov2(self.fusion, None)
        self._maybe_install_dinov2(self.positional, self.face_backbone)

    def _maybe_install_dinov2(
        self, module: torch.nn.Module, backbone: Optional[torch.nn.Module]
    ) -> None:
        if not hasattr(module, "replace_with_dinov2"):
            return
        replace = getattr(module, "replace_with_dinov2")
        if not callable(replace):
            return
        signature = inspect.signature(replace)  # type: ignore[arg-type]
        kwargs: Dict[str, Any] = {}
        if "backbone" in signature.parameters and backbone is None:
            return
        if "backbone" in signature.parameters and backbone is not None:
            kwargs["backbone"] = backbone
        try:
            replace(**kwargs)
        except TypeError:
            # The implementation might expect additional arguments (e.g. variant).
            # In that case we silently skip auto-configuration and rely on manual wiring.
            return
        except (RuntimeError, ImportError, ValueError, AttributeError) as exc:
            warnings.warn(
                f"Unable to configure {module.__class__.__name__} with DINOv2 weights: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )

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
        fused = self.fusion(*streams)
        if mask is not None and not self._fusion_accepts_mask():
            mask_bool = mask.to(device=fused.device, dtype=torch.bool)
            while mask_bool.dim() < fused.dim():
                mask_bool = mask_bool.unsqueeze(-1)
            fused = fused.masked_fill(mask_bool, 0)
        return fused

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
        while mask_bool.dim() < 2:
            mask_bool = mask_bool.unsqueeze(0)
        if mask_bool.dim() > 2:
            mask_bool = mask_bool.view(mask_bool.shape[0], -1)
        return ~mask_bool
