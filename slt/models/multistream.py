"""High level multi-stream encoder stub for SLT."""

from __future__ import annotations

import math
import os
import inspect
import warnings
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Dict,
    Iterable,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import torch
import torch.nn.functional as F
from torch import Tensor

from .backbones import BackboneSpec, ViTConfig, load_backbone
from .modules import FuseConcatLinear, PositionalEncodingLearned, StreamProjector

if TYPE_CHECKING:
    from .mska import MSKAEncoder, MSKAOutput
from .temporal import TemporalEncoder

__all__ = ["MultiStreamEncoder"]


@dataclass(frozen=True)
class _StreamAssembly:
    """Book-keeping structure linking stream attributes."""

    name: str
    backbone_attr: Optional[str]
    projector_attr: str

    def modules(self, owner: "MultiStreamEncoder") -> Dict[str, torch.nn.Module]:
        modules: Dict[str, torch.nn.Module] = {}
        if self.backbone_attr is not None:
            backbone = getattr(owner, self.backbone_attr)
            modules["backbone"] = backbone
        projector = getattr(owner, self.projector_attr)
        modules["projector"] = projector
        return modules


class MultiStreamEncoder(torch.nn.Module):
    """Composable encoder that processes face, hand and pose streams."""

    def __init__(
        self,
        *,
        backbone_config: Optional[Union[ViTConfig, BackboneSpec]] = None,
        projector_dim: int = 256,
        d_model: int = 512,
        pose_dim: int = 39,
        positional_num_positions: int = 512,
        projector_dropout: float = 0.0,
        fusion_dropout: float = 0.0,
        fusion_bias: bool = True,
        temporal_kwargs: Optional[Dict[str, Any]] = None,
        backbones: Optional[Mapping[str, torch.nn.Module]] = None,
        mask_combiner: Optional[
            Callable[[Optional[Tensor], Optional[Tensor]], Optional[Tensor]]
        ] = None,
        mska: Optional["MSKAEncoder"] = None,
        mska_gloss_hidden_dim: Optional[int] = None,
        mska_gloss_activation: str = "gelu",
        mska_gloss_dropout: float = 0.0,
    ) -> None:
        super().__init__()
        temporal_kwargs = temporal_kwargs or {}

        backbones = backbones or {}
        self._stream_to_attr = {
            "face": "face_backbone",
            "hand_left": "hand_backbone_left",
            "hand_right": "hand_backbone_right",
        }
        self._stream_to_projector = {
            "face": "face_projector",
            "hand_left": "hand_left_projector",
            "hand_right": "hand_right_projector",
            "pose": "pose_projector",
        }
        self._backbone_factories: DefaultDict[
            str, Dict[str, Callable[[], torch.nn.Module]]
        ] = defaultdict(dict)
        self._backbone_instances: DefaultDict[str, Dict[str, torch.nn.Module]] = defaultdict(dict)
        self._backbone_indices: DefaultDict[str, Dict[str, int]] = defaultdict(dict)
        self._active_backbone_names: Dict[str, str] = {}
        self._last_hand_masks: Dict[str, Optional[Tensor]] = {"hand_left": None, "hand_right": None}
        self._observers: DefaultDict[str, list[Callable[[str, Tensor], None]]] = defaultdict(list)
        self._register_initial_backbones(backbones, backbone_config)

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
        self._last_pose_mask: Optional[Tensor] = None
        self._mask_combiner = mask_combiner or self._default_mask_combiner
        self._stream_assemblies: Dict[str, _StreamAssembly] = {
            stream: _StreamAssembly(
                name=stream,
                backbone_attr=self._stream_to_attr.get(stream),
                projector_attr=self._stream_to_projector[stream],
            )
            for stream in self._stream_to_projector
        }
        self._auto_configure_components()
        self.mska_encoder = mska
        self._last_mska_output: Optional["MSKAOutput"] = None
        self._mska_gloss_mlp: Optional[torch.nn.Sequential] = None
        self._last_gloss_sequence: Optional[Tensor] = None
        self._last_gloss_mask: Optional[Tensor] = None
        if self.mska_encoder is not None:
            if getattr(self.mska_encoder, "embed_dim", projector_dim) != projector_dim:
                raise ValueError(
                    "MSKA encoder output dimensionality must match projector_dim"
                )
            hidden_dim = int(mska_gloss_hidden_dim) if mska_gloss_hidden_dim else d_model
            if hidden_dim <= 0:
                raise ValueError("mska_gloss_hidden_dim must be positive")
            if mska_gloss_dropout < 0:
                raise ValueError("mska_gloss_dropout must be non-negative")
            activation = self._resolve_activation(mska_gloss_activation)
            layers: list[torch.nn.Module] = [
                torch.nn.Linear(projector_dim, hidden_dim),
                activation,
            ]
            if mska_gloss_dropout > 0.0:
                layers.append(torch.nn.Dropout(mska_gloss_dropout))
            layers.append(torch.nn.Linear(hidden_dim, d_model))
            self._mska_gloss_mlp = torch.nn.Sequential(*layers)
            self._mska_streams = {
                name: name
                for name in self.mska_encoder.stream_names
                if name in self._stream_to_projector
            }
        else:
            self._mska_streams = {}

    @classmethod
    def from_pretrained(
        cls,
        preset: str = "single_signer",
        *,
        checkpoint_path: Optional[Union[str, os.PathLike[str]]] = None,
        map_location: Optional[torch.device] = None,
        strict: bool = True,
    ) -> "MultiStreamEncoder":
        """Instantiate the encoder with the validated single-signer weights."""

        normalized = preset.replace("-", "_").strip().lower()
        if normalized != "single_signer":
            raise ValueError(
                f"Unknown pretrained preset '{preset}'. Available options: 'single_signer'"
            )
        from .single_signer import load_single_signer_encoder

        encoder, metadata, _ = load_single_signer_encoder(
            checkpoint_path=checkpoint_path, map_location=map_location, strict=strict
        )
        setattr(encoder, "pretrained_metadata", metadata)
        return encoder

    def forward(
        self,
        face: Tensor,
        hand_l: Tensor,
        hand_r: Tensor,
        pose: Tensor,
        pad_mask: Optional[Tensor] = None,
        miss_mask_hl: Optional[Tensor] = None,
        miss_mask_hr: Optional[Tensor] = None,
        pose_conf_mask: Optional[Tensor] = None,
        keypoint_streams: Optional[Mapping[str, Mapping[str, Tensor]]] = None,
        **unused_inputs: Any,
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
        pose_processed = self._ensure_pose_shape(pose)
        pose_processed, pose_mask = self._apply_pose_mask(pose_processed, pose_conf_mask)
        pose_proj = self.pose_projector(pose_processed)
        self._emit("pose.projector", pose_proj)
        if pose_mask is not None:
            self._emit("pose.mask", pose_mask.to(torch.float32))

        mska_output: Optional["MSKAOutput"] = None
        gloss_sequence: Optional[Tensor] = None
        gloss_mask: Optional[Tensor] = None
        if self.mska_encoder is not None and keypoint_streams:
            try:
                mska_output = self.mska_encoder(keypoint_streams)
            except ValueError:
                mska_output = None
            if mska_output is not None:
                self._emit("mska.fused", mska_output.fused_embedding)
                for stream_name, projector_attr in self._mska_streams.items():
                    stream_embedding = mska_output.stream_embeddings.get(stream_name)
                    if stream_embedding is None:
                        continue
                    if projector_attr == "face":
                        face_proj = face_proj + stream_embedding
                    elif projector_attr == "hand_left":
                        hand_l_proj = hand_l_proj + stream_embedding
                    elif projector_attr == "hand_right":
                        hand_r_proj = hand_r_proj + stream_embedding
                    elif projector_attr == "pose":
                        pose_proj = pose_proj + stream_embedding
                if self._mska_gloss_mlp is not None:
                    gloss_sequence = self._mska_gloss_mlp(mska_output.fused_embedding)
                    self._emit("mska.gloss", gloss_sequence)
                gloss_mask = mska_output.fused_mask
                if gloss_mask is not None:
                    gloss_mask = gloss_mask.to(dtype=torch.bool)
                    self._emit("mska.gloss_mask", gloss_mask.to(torch.float32))
            else:
                gloss_sequence = None
                gloss_mask = None
        
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
        self._last_mska_output = mska_output
        self._last_gloss_sequence = gloss_sequence
        self._last_gloss_mask = gloss_mask
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

    @property
    def last_hand_masks(self) -> Dict[str, Optional[Tensor]]:
        """Return the last per-hand masks applied during fusion."""

        return dict(self._last_hand_masks)

    @property
    def last_pose_mask(self) -> Optional[Tensor]:
        """Return the most recent pose confidence mask."""

        return self._last_pose_mask

    @property
    def last_mska_output(self) -> Optional["MSKAOutput"]:
        """Return the last MSKA output if available."""

        return self._last_mska_output

    @property
    def last_gloss_sequence(self) -> Optional[Tensor]:
        """Return the gloss representation produced by the MSKA MLP."""

        return self._last_gloss_sequence

    @property
    def last_gloss_mask(self) -> Optional[Tensor]:
        """Return the temporal mask aligned with the gloss sequence."""

        return self._last_gloss_mask

    def active_backbone_name(self, stream: str) -> Optional[str]:
        """Return the currently active backbone identifier for ``stream``."""

        return self._active_backbone_names.get(stream)

    def active_backbones(self) -> Dict[str, torch.nn.Module]:
        """Return a mapping with the currently instantiated backbones."""

        return {stream: getattr(self, attr) for stream, attr in self._stream_to_attr.items()}

    # ------------------------------------------------------------------
    # Backbone registry API
    # ------------------------------------------------------------------
    def stream_state_dict(self, stream: str) -> Dict[str, Tensor]:
        """Return a shallow state dictionary for a single ``stream``."""

        assembly = self._require_stream(stream)
        wrapper = torch.nn.Module()
        for name, module in assembly.modules(self).items():
            setattr(wrapper, name, module)
        return OrderedDict(wrapper.state_dict())

    def load_stream_state_dict(
        self,
        stream: str,
        state_dict: Mapping[str, Tensor],
        *,
        strict: bool = True,
    ) -> torch.nn.modules.module._IncompatibleKeys:
        """Load parameters for a particular ``stream``."""

        assembly = self._require_stream(stream)
        wrapper = torch.nn.Module()
        for name, module in assembly.modules(self).items():
            setattr(wrapper, name, module)
        return wrapper.load_state_dict(state_dict, strict=strict)

    def register_backbone(
        self,
        stream: str,
        name: str,
        spec: BackboneSpec,
        *,
        map_location: Optional[Union[str, torch.device]] = None,
        trust_repo: bool = True,
    ) -> None:
        """Register ``spec`` under ``name`` for ``stream``."""

        if stream not in self._stream_to_attr:
            raise ValueError(f"Unsupported stream '{stream}'")

        factory = self._build_backbone_factory(spec, map_location, trust_repo)
        self._backbone_factories[stream][name] = factory
        if name not in self._backbone_indices[stream]:
            self._backbone_indices[stream][name] = len(self._backbone_indices[stream])

    def activate_backbone(self, stream: str, name: str) -> torch.nn.Module:
        """Activate the registered backbone ``name`` for ``stream``."""

        if stream not in self._stream_to_attr:
            raise ValueError(f"Unsupported stream '{stream}'")
        if name not in self._backbone_factories[stream]:
            raise KeyError(f"No backbone named '{name}' registered for stream '{stream}'")

        instance = self._backbone_instances[stream].get(name)
        if instance is None:
            instance = self._backbone_factories[stream][name]()
            if hasattr(instance, "as_backbone") and callable(instance.as_backbone):
                instance = instance.as_backbone()  # type: ignore[assignment]
            self._backbone_instances[stream][name] = instance

        attr = self._stream_to_attr[stream]
        setattr(self, attr, instance)
        self._active_backbone_names[stream] = name
        self._emit_backbone_state(stream, name)
        return instance

    def available_backbones(self, stream: Optional[str] = None) -> Dict[str, Tuple[str, ...]]:
        """Return available backbone names for one or all streams."""

        if stream is not None:
            return {stream: tuple(self._backbone_factories[stream].keys())}
        return {
            stream_name: tuple(factory.keys())
            for stream_name, factory in self._backbone_factories.items()
        }

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
            if (
                isinstance(module, StreamProjector)
                and backbone is not None
            ):
                try:
                    backbone_dim = self._infer_backbone_dim(backbone)
                except AttributeError:
                    backbone_dim = None
                else:
                    if module.out_dim != backbone_dim:
                        continue
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
    def _extract_patch_size(
        backbone: torch.nn.Module,
    ) -> Optional[Tuple[int, int]]:
        patch_embed = getattr(backbone, "patch_embed", None)

        def _normalize(size: Any) -> Optional[Tuple[int, int]]:
            if size is None:
                return None
            if isinstance(size, (int, float)):
                value = int(size)
                if value > 0:
                    return value, value
                return None
            if isinstance(size, Sequence):
                values = [int(v) for v in size]
                if len(values) == 1:
                    values = values * 2
                if len(values) >= 2 and values[0] > 0 and values[1] > 0:
                    return values[0], values[1]
            return None

        if patch_embed is None:
            return None

        candidates = [patch_embed]
        for attr in ("proj", "projection"):
            candidate = getattr(patch_embed, attr, None)
            if candidate is not None:
                candidates.append(candidate)

        for candidate in candidates:
            kernel_size = getattr(candidate, "kernel_size", None)
            patch = _normalize(kernel_size)
            if patch is not None:
                return patch

        for owner in (patch_embed, backbone):
            patch = _normalize(getattr(owner, "patch_size", None))
            if patch is not None:
                return patch

        return None

    @staticmethod
    def _encode_backbone(backbone: torch.nn.Module, stream: Tensor) -> Tensor:
        if stream.dim() != 5:
            raise ValueError(
                "Backbone inputs must have shape (batch, time, channels, height, width)."
            )
        batch, time = stream.shape[:2]
        patch = MultiStreamEncoder._extract_patch_size(backbone)
        if patch is not None:
            patch_h, patch_w = patch
            if patch_h > 0 and patch_w > 0:
                height, width = stream.shape[-2:]
                target_h = math.ceil(height / patch_h) * patch_h
                target_w = math.ceil(width / patch_w) * patch_w
                if target_h != height or target_w != width:
                    pad_bottom = target_h - height
                    pad_right = target_w - width
                    if pad_bottom < 0 or pad_right < 0:
                        raise ValueError(
                            "Target spatial dimensions must be greater or equal to the input"
                        )
                    pad_spec = (0, pad_right, 0, pad_bottom, 0, 0)
                    stream = stream.permute(0, 2, 1, 3, 4)
                    stream = F.pad(stream, pad_spec, mode="replicate")
                    stream = stream.permute(0, 2, 1, 3, 4)
        flat = stream.view(batch * time, *stream.shape[2:])
        features = backbone(flat)
        return features.view(batch, time, -1)

    def _apply_missing_mask(
        self, features: Tensor, mask: Optional[Tensor], *, stream: str
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if mask is None:
            self._last_hand_masks[stream] = None
            return features, None
        mask_bool = self._ensure_bool_mask(mask, features.shape[:2])
        masked = self._mask_hand_features(features, mask_bool, stream=stream)
        self._last_hand_masks[stream] = mask_bool
        self._emit(f"{stream}.mask", mask_bool.to(torch.float32))
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

    def _apply_pose_mask(
        self, pose: Tensor, mask: Optional[Tensor]
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if mask is None:
            self._last_pose_mask = None
            return pose, None
        mask_bool = mask.to(dtype=torch.bool)
        expected_bt = pose.shape[:2]
        if mask_bool.shape[0] != expected_bt[0] or mask_bool.shape[1] != expected_bt[1]:
            raise ValueError(
                "Pose confidence masks must match batch/time dimensions; received "
                f"shape {tuple(mask_bool.shape)}"
            )
        landmarks = pose.size(-1) // 3
        if pose.size(-1) % 3 != 0:
            raise ValueError("Pose dimensionality must be divisible by 3")
        if mask_bool.dim() == 2:
            mask_bool = mask_bool.unsqueeze(-1).expand(-1, -1, landmarks)
        elif mask_bool.dim() == 3:
            if mask_bool.size(-1) != landmarks:
                raise ValueError(
                    "Pose mask landmark dimension does not match pose dimensionality"
                )
        else:
            raise ValueError("Pose confidence masks must have 2 or 3 dimensions")
        mask_bool = mask_bool.to(device=pose.device)
        masked = self._mask_pose_features(pose, mask_bool)
        self._last_pose_mask = mask_bool
        return masked, mask_bool

    def _mask_pose_features(self, pose: Tensor, mask: Tensor) -> Tensor:
        if pose.dim() != 3:
            raise ValueError("Pose features must have shape (batch, time, pose_dim)")
        pose_dim = pose.size(-1)
        if pose_dim % 3 != 0:
            raise ValueError("Pose dimensionality must be divisible by 3")
        if mask.dim() != 3:
            raise ValueError("Pose masks must have shape (batch, time, landmarks)")
        mask_bool = mask.to(device=pose.device, dtype=torch.bool)
        expanded = mask_bool.unsqueeze(-1)
        reshaped = pose.view(*pose.shape[:-1], pose_dim // 3, 3)
        zeros = torch.zeros_like(reshaped)
        reshaped = torch.where(expanded, reshaped, zeros)
        return reshaped.view_as(pose)

    def _combine_missing_masks(
        self, left: Optional[Tensor], right: Optional[Tensor]
    ) -> Optional[Tensor]:
        combined = self._mask_combiner(left, right)
        if combined is None:
            return None
        return combined.to(dtype=torch.bool)

    @staticmethod
    def _default_mask_combiner(
        left: Optional[Tensor], right: Optional[Tensor]
    ) -> Optional[Tensor]:
        if left is None and right is None:
            return None
        if left is None:
            return right.clone()
        if right is None:
            return left.clone()
        return torch.logical_or(left, right)

    def set_mask_combiner(
        self,
        combiner: Optional[
            Callable[[Optional[Tensor], Optional[Tensor]], Optional[Tensor]]
        ],
    ) -> None:
        """Configure how the left/right hand masks are combined."""

        if combiner is None:
            self._mask_combiner = self._default_mask_combiner
        elif not callable(combiner):
            raise TypeError("Mask combiner must be callable or None")
        else:
            self._mask_combiner = combiner

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

    @staticmethod
    def _resolve_activation(name: str) -> torch.nn.Module:
        lookup = {
            "relu": torch.nn.ReLU(),
            "gelu": torch.nn.GELU(),
            "silu": torch.nn.SiLU(),
            "tanh": torch.nn.Tanh(),
        }
        key = name.strip().lower()
        if key not in lookup:
            supported = ", ".join(sorted(lookup))
            raise ValueError(
                f"Unsupported activation '{name}'. Available options: {supported}"
            )
        return lookup[key]

    def _register_initial_backbones(
        self,
        backbones: Mapping[str, BackboneSpec],
        config: Optional[Union[ViTConfig, BackboneSpec]],
    ) -> None:
        for stream, attr in self._stream_to_attr.items():
            candidate: BackboneSpec
            if stream in backbones:
                candidate = backbones[stream]
            else:
                if isinstance(config, ViTConfig):
                    candidate = config.to_spec()
                elif isinstance(config, Mapping):
                    candidate = dict(config)
                elif config is not None:
                    candidate = config
                else:
                    candidate = ViTConfig().to_spec()
            self.register_backbone(stream, "default", candidate)
            self.activate_backbone(stream, "default")
        # Ensure bookkeeping emits signals for initial selection
        for stream in self._stream_to_attr:
            self._emit_backbone_state(stream, self._active_backbone_names[stream])

    def _build_backbone_factory(
        self,
        spec: BackboneSpec,
        map_location: Optional[Union[str, torch.device]],
        trust_repo: bool,
    ) -> Callable[[], torch.nn.Module]:
        if isinstance(spec, torch.nn.Module):
            module = spec

            def factory() -> torch.nn.Module:
                return module

            return factory

        if callable(spec) and not isinstance(spec, Mapping):
            def factory_callable() -> torch.nn.Module:
                candidate = spec()  # type: ignore[operator]
                if not isinstance(candidate, torch.nn.Module):
                    raise TypeError(
                        "Backbone factory callable must return an nn.Module instance"
                    )
                return candidate

            return factory_callable

        def factory_loader() -> torch.nn.Module:
            return load_backbone(spec, map_location=map_location, trust_repo=trust_repo)

        return factory_loader

    def _emit_backbone_state(self, stream: str, name: str) -> None:
        index = self._backbone_indices[stream].setdefault(
            name, len(self._backbone_indices[stream])
        )
        value = torch.tensor([float(index)], dtype=torch.float32)
        self._emit(f"{stream}.backbone", value)

    def _require_stream(self, stream: str) -> _StreamAssembly:
        if stream not in self._stream_assemblies:
            raise ValueError(f"Unknown stream '{stream}'")
        return self._stream_assemblies[stream]
