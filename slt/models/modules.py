"""Reusable building blocks for SLT models."""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import torch
from torch import Tensor, nn


class StreamProjector(nn.Module):
    """Project per-stream embeddings into a common representation space.

    Parameters
    ----------
    in_dim:
        Size of the incoming feature dimension.
    out_dim:
        Desired output projection size.
    hidden_dim:
        Optional intermediate dimension. When ``None`` the module uses
        ``out_dim`` which mimics the behaviour of the light-weight projectors
        used in the research prototypes. Production deployments can extend this
        class to use the official DINOv2 projection heads by overriding
        :meth:`forward`.
    dropout:
        Dropout probability applied after the activation function.
    activation:
        Activation module constructor. ``nn.GELU`` is used by default to match
        the ViT blocks.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        *,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        hidden_dim = hidden_dim or out_dim
        self.layers = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            activation(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        """Project the incoming tensor along its last dimension."""

        return self.layers(x)

    # ------------------------------------------------------------------
    # Extension hooks
    # ------------------------------------------------------------------
    def replace_with_dinov2(self) -> None:
        """Placeholder hook to swap the stub implementation for DINOv2.

        Production builds can override this method to install the official
        ``torchvision.models.dinov2`` projector heads. The research branch keeps
        a minimal dependency footprint by default.
        """

        raise NotImplementedError("Install DINOv2 projector weights in production.")


class FuseConcatLinear(nn.Module):
    """Fuse a list of tensors by concatenating and projecting them.

    Parameters
    ----------
    in_dims:
        Iterable with the dimensionality of each stream that will be fused.
    out_dim:
        Output dimensionality after fusion.
    dropout:
        Dropout probability applied prior to the final projection.
    bias:
        Whether to include a bias term in the output projection.
    """

    def __init__(
        self,
        in_dims: Iterable[int],
        out_dim: int,
        *,
        dropout: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.total_dim = int(sum(in_dims))
        if self.total_dim <= 0:
            raise ValueError("The concatenated dimensionality must be positive.")
        self.norm = nn.LayerNorm(self.total_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(self.total_dim, out_dim, bias=bias)

    def forward(self, *streams: Tensor | Sequence[Tensor]) -> Tensor:
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
        fused = torch.cat(features, dim=-1)
        fused = self.norm(fused)
        fused = self.dropout(fused)
        return self.proj(fused)

    @staticmethod
    def _ensure_tensor(tensor: Tensor) -> Tensor:
        if not isinstance(tensor, Tensor):
            raise TypeError("All streams must be PyTorch tensors.")
        return tensor

    def replace_with_dinov2(self) -> None:
        """Hook reserved for installing DINOv2 fusion blocks in production."""

        raise NotImplementedError("Replace with DINOv2 fusion in production builds.")


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
        """Add positional encodings to ``x``.

        Parameters
        ----------
        x:
            Input tensor expected to have shape ``(..., seq_len, dim)``.
        positions:
            Optional tensor with explicit position indices. When omitted the
            positions ``[0, 1, ..., seq_len - 1]`` are used.
        """

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

    def replace_with_dinov2(self) -> None:
        """Hook for injecting DINOv2 positional encodings in production."""

        raise NotImplementedError(
            "Swap the learned embedding with DINOv2 positional encodings in production."
        )
