"""Reusable building blocks for SLT models."""
from __future__ import annotations

import copy
from typing import Callable, Iterable, Optional, Sequence

import torch
from torch import Tensor, nn

from .backbones import BackboneFreeze, _apply_freeze

try:  # pragma: no cover - optional dependency.
    from torchvision.models import get_model as tv_get_model  # type: ignore
    from torchvision.models import get_model_weights as tv_get_model_weights  # type: ignore
except Exception:  # pragma: no cover - optional dependency.
    tv_get_model = None  # type: ignore
    tv_get_model_weights = None  # type: ignore


def _resolve_dinov2_model(
    variant: Optional[str],
    *,
    weights: Optional[str] = None,
) -> Optional[nn.Module]:
    if variant is None:
        return None
    if tv_get_model is None or tv_get_model_weights is None:
        raise ImportError(
            "torchvision>=0.16 is required to build DINOv2 heads; install it to continue"
        )
    try:
        weights_enum = tv_get_model_weights(variant)
    except Exception:  # pragma: no cover - defensive.
        weights_enum = None

    resolved_weight = None
    if weights and weights.lower() not in {"", "none", "random"}:
        spec = weights.upper()
        if weights_enum is not None:
            if spec in weights_enum.__dict__:
                resolved_weight = getattr(weights_enum, spec)
            else:
                alt = f"{spec}_V1"
                if hasattr(weights_enum, alt):
                    resolved_weight = getattr(weights_enum, alt)
        if resolved_weight is None:
            raise ValueError(f"Unknown weight specification '{weights}' for variant '{variant}'")
    elif weights_enum is not None:
        resolved_weight = getattr(weights_enum, "DEFAULT", None)

    return tv_get_model(variant, weights=resolved_weight)


def _extract_module(source: nn.Module, names: Sequence[str]) -> Optional[nn.Module]:
    for name in names:
        component = getattr(source, name, None)
        if isinstance(component, nn.Module):
            return component
        if component is not None and hasattr(component, "projector"):
            projector = getattr(component, "projector")
            if isinstance(projector, nn.Module):
                return projector
    return None


def _clone_module(module: nn.Module) -> nn.Module:
    cloned = copy.deepcopy(module)
    parameters = list(cloned.parameters())
    if parameters:
        device = parameters[0].device
    else:
        device = torch.device("cpu")
    cloned.to(device)
    return cloned


def _expand_mask(mask: Tensor, reference: Tensor) -> Tensor:
    mask_bool = mask.to(device=reference.device, dtype=torch.bool)
    while mask_bool.dim() < reference.dim():
        mask_bool = mask_bool.unsqueeze(-1)
    target_shape = reference.shape
    if mask_bool.shape != target_shape:
        expand_shape = target_shape[:-1] + (mask_bool.size(-1),)
        mask_bool = mask_bool.expand(expand_shape)
        if mask_bool.shape != target_shape:
            mask_bool = mask_bool.expand(target_shape)
    return mask_bool


class StreamProjector(nn.Module):
    """Project per-stream embeddings into a common representation space."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dims: Optional[Sequence[int]] = None,
        dropout: float = 0.0,
        activation: Callable[[], nn.Module] = nn.GELU,
        layer_norm_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dims = list(hidden_dims or [out_dim])
        self.dropout = dropout
        self.activation = activation
        self.layer_norm_eps = layer_norm_eps
        self.layers = self._build_layers()

    def _build_layers(self) -> nn.Sequential:
        dims = [self.in_dim] + self.hidden_dims + [self.out_dim]
        sequential: list[nn.Module] = [nn.LayerNorm(self.in_dim, eps=self.layer_norm_eps)]
        for idx in range(len(dims) - 1):
            in_features, out_features = dims[idx], dims[idx + 1]
            sequential.append(nn.Linear(in_features, out_features))
            is_last = idx == len(dims) - 2
            if not is_last:
                sequential.append(self.activation())
                if self.dropout:
                    sequential.append(nn.Dropout(self.dropout))
        return nn.Sequential(*sequential)

    def forward(self, x: Tensor) -> Tensor:
        """Project the incoming tensor along its last dimension."""

        if x.size(-1) != self.in_dim:
            raise ValueError(
                f"Expected last dimension {self.in_dim}, received {x.size(-1)}"
            )
        return self.layers(x)

    def replace_with_dinov2(
        self,
        *,
        backbone: Optional[nn.Module] = None,
        variant: Optional[str] = None,
        weights: Optional[str] = None,
        freeze: BackboneFreeze = False,
    ) -> None:
        """Swap the projector with the official DINOv2 head when available."""

        projector: Optional[nn.Module] = None
        if backbone is not None:
            projector = _extract_module(backbone, ("projector", "head", "linear_head", "mlp_head"))
        if projector is None and variant is not None:
            model = _resolve_dinov2_model(variant, weights=weights)
            if model is not None:
                projector = _extract_module(model, ("projector", "head", "linear_head", "mlp_head"))
        if projector is None:
            raise RuntimeError(
                "Unable to locate a DINOv2 projector; provide a backbone or variant name"
            )

        self.layers = _clone_module(projector)
        if freeze:
            _apply_freeze(self.layers, freeze)


class FuseConcatLinear(nn.Module):
    """Fuse a list of tensors by concatenating and projecting them."""

    def __init__(
        self,
        in_dims: Iterable[int],
        out_dim: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
        hand_stream_indices: Sequence[int] = (1, 2),
    ) -> None:
        super().__init__()
        self.in_dims = list(in_dims)
        self.total_dim = int(sum(self.in_dims))
        if self.total_dim <= 0:
            raise ValueError("The concatenated dimensionality must be positive.")
        self.norm = nn.LayerNorm(self.total_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(self.total_dim, out_dim, bias=bias)
        self.hand_stream_indices = tuple(hand_stream_indices)
        self._offsets = self._compute_offsets()

    def _compute_offsets(self) -> list[tuple[int, int]]:
        offsets: list[tuple[int, int]] = []
        current = 0
        for dim in self.in_dims:
            start, end = current, current + dim
            offsets.append((start, end))
            current = end
        return offsets

    def forward(
        self,
        *streams: Tensor | Sequence[Tensor],
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Concatenate inputs along the last dimension and project the result."""

        if len(streams) == 1 and isinstance(streams[0], (list, tuple)):
            streams = tuple(streams[0])  # type: ignore[assignment]
        if not streams:
            raise ValueError("At least one tensor must be provided for fusion.")
        features = [self._ensure_tensor(stream) for stream in streams]
        base_shape = features[0].shape[:-1]
        for tensor in features:
            if tensor.shape[:-1] != base_shape:
                raise ValueError(
                    "All tensors must share the same leading dimensions; "
                    f"expected {base_shape}, received {tensor.shape[:-1]}"
                )

        if mask is not None and self.hand_stream_indices:
            expanded_mask = _expand_mask(mask, features[0])
            for index in self.hand_stream_indices:
                if 0 <= index < len(features):
                    features[index] = features[index].masked_fill(expanded_mask, 0)

        fused = torch.cat(features, dim=-1)
        fused = self.norm(fused)
        fused = self.dropout(fused)
        return self.proj(fused)

    @staticmethod
    def _ensure_tensor(tensor: Tensor) -> Tensor:
        if not isinstance(tensor, Tensor):
            raise TypeError("All streams must be PyTorch tensors.")
        return tensor

    def replace_with_dinov2(
        self,
        *,
        backbone: Optional[nn.Module] = None,
        variant: Optional[str] = None,
        weights: Optional[str] = None,
        freeze: BackboneFreeze = False,
    ) -> None:
        """Install the official DINOv2 fusion block when available."""

        fusion_module: Optional[nn.Module] = None
        if backbone is not None:
            fusion_module = _extract_module(backbone, ("fusion", "aggregator", "fusion_head"))
        if fusion_module is None and variant is not None:
            model = _resolve_dinov2_model(variant, weights=weights)
            if model is not None:
                fusion_module = _extract_module(model, ("fusion", "aggregator", "fusion_head"))
        if fusion_module is None:
            raise RuntimeError(
                "Unable to locate a DINOv2 fusion module; provide a backbone or variant name"
            )

        fusion_module = _clone_module(fusion_module)
        if isinstance(fusion_module, nn.Sequential):
            norm_layer = next((layer for layer in fusion_module if isinstance(layer, nn.LayerNorm)), None)
            proj_layer = next((layer for layer in fusion_module if isinstance(layer, nn.Linear)), None)
            if norm_layer is not None:
                self.norm.load_state_dict(norm_layer.state_dict())
            if proj_layer is not None:
                self.proj.load_state_dict(proj_layer.state_dict())
        elif isinstance(fusion_module, nn.Linear):
            self.proj.load_state_dict(fusion_module.state_dict())
        else:
            raise TypeError(
                "Unsupported fusion module type received from DINOv2; expected Sequential or Linear"
            )

        if freeze:
            _apply_freeze(self.proj, freeze)


class PositionalEncodingLearned(nn.Module):
    """Learned positional encodings for sequence data."""

    def __init__(
        self,
        num_positions: int,
        dim: int,
        *,
        padding_idx: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_positions, dim, padding_idx=padding_idx)
        self.dim = dim

    def forward(self, x: Tensor, positions: Optional[Tensor] = None) -> Tensor:
        """Add positional encodings to ``x``."""

        if x.size(-1) != self.dim:
            raise ValueError(
                f"Expected input feature dimension {self.dim}, received {x.size(-1)}"
            )

        batch_shape = x.shape[:-2]
        seq_len = x.size(-2)

        if positions is None:
            base_shape = (1,) * len(batch_shape) + (seq_len,)
            position_ids = torch.arange(seq_len, device=x.device).view(base_shape)
            expand_shape = batch_shape + (seq_len,)
            positions = position_ids.expand(expand_shape)
        else:
            positions = positions.to(x.device)
            if positions.shape[-1] != seq_len:
                raise ValueError(
                    f"Expected positions with last dimension {seq_len}, "
                    f"received {positions.shape[-1]}"
                )
            while positions.dim() < len(batch_shape) + 1:
                positions = positions.unsqueeze(0)
            expand_shape = batch_shape + (seq_len,)
            positions = positions.expand(expand_shape)

        pos_embed = self.embedding(positions.long())
        return x + pos_embed

    def replace_with_dinov2(
        self,
        *,
        backbone: Optional[nn.Module] = None,
        variant: Optional[str] = None,
        weights: Optional[str] = None,
    ) -> None:
        """Initialise the positional encoding with DINOv2 parameters."""

        model: Optional[nn.Module] = None
        if backbone is not None:
            model = backbone
        elif variant is not None:
            try:
                model = _resolve_dinov2_model(variant, weights=weights)
            except ImportError:
                model = None

        if model is None:
            raise RuntimeError(
                "Unable to load DINOv2 positional encodings; provide a backbone or variant name"
            )

        pos_embed = getattr(model, "pos_embed", None)
        if pos_embed is None and hasattr(model, "backbone"):
            pos_embed = getattr(model.backbone, "pos_embed", None)
        if pos_embed is None:
            raise AttributeError("Provided model does not expose positional embeddings")

        tensor = pos_embed.detach().clone()
        if tensor.dim() == 3 and tensor.size(0) == 1:
            tensor = tensor.squeeze(0)
        if tensor.dim() == 2 and tensor.size(0) > 0 and tensor.size(0) != self.embedding.num_embeddings:
            if tensor.size(0) - 1 == self.embedding.num_embeddings:
                tensor = tensor[1:]
            elif tensor.size(0) == self.embedding.num_embeddings + 1:
                tensor = tensor[1:]
            elif tensor.size(0) < self.embedding.num_embeddings:
                pad = tensor.new_zeros(self.embedding.num_embeddings - tensor.size(0), tensor.size(1))
                tensor = torch.cat([tensor, pad], dim=0)
            else:
                tensor = tensor[: self.embedding.num_embeddings]

        if tensor.size(0) != self.embedding.num_embeddings or tensor.size(1) != self.dim:
            raise ValueError(
                "DINOv2 positional embeddings are incompatible with the configured embedding size"
            )

        with torch.no_grad():
            self.embedding.weight.copy_(tensor)
