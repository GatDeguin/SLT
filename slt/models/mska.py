"""Modules implementing the Multi-Stream Keypoint Attention encoder."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

__all__ = [
    "KeypointStreamEncoder",
    "KeypointStreamOutput",
    "MultiStreamKeypointAttention",
    "MSKAOutput",
    "MSKAEncoder",
    "StreamCTCHead",
    "FusedCTCHead",
]


@dataclass
class KeypointStreamOutput:
    """Container describing the encoded representation of a keypoint stream."""

    joint_embeddings: Tensor
    frame_embeddings: Tensor
    joint_mask: Optional[Tensor]
    frame_mask: Optional[Tensor]


class _GlobalAttentionStore(nn.Module):
    """Container for a learnable global attention matrix."""

    def __init__(self) -> None:
        super().__init__()
        self.register_parameter("matrix", None)
        self._size: int = 0

    def matrix_for(self, size: int, *, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Return an ``(size, size)`` matrix lazily initialised on ``device`` and ``dtype``."""

        if size <= 0:
            raise ValueError("Global attention size must be a positive integer")

        if self.matrix is None:
            self.matrix = nn.Parameter(torch.eye(size, device=device, dtype=dtype))
            self._size = size
        elif size > self._size:
            new_matrix = torch.eye(size, device=device, dtype=dtype)
            old_matrix = self.matrix.detach().to(device=device, dtype=dtype)
            new_matrix[: self._size, : self._size] = old_matrix
            self.matrix = nn.Parameter(new_matrix)
            self._size = size

        matrix = self.matrix
        if matrix.device != device:
            matrix = matrix.to(device=device)
        if matrix.dtype != dtype:
            matrix = matrix.to(dtype=dtype)
        return matrix[:size, :size]


class _TanhMultiHeadAttention(nn.Module):
    """Multi-head self-attention with ``tanh`` logits and optional global matrix."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        *,
        use_global: bool = False,
        global_activation: str = "softmax",
        global_mix: float = 0.5,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")
        if not 0.0 <= global_mix <= 1.0:
            raise ValueError("global_mix must be between 0 and 1 inclusive")

        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.embed_dim // self.num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.use_global = bool(use_global)
        self.global_mix = float(global_mix)
        self._global_activation_name = str(global_activation).lower()
        self._local_global_store: Optional[_GlobalAttentionStore] = (
            _GlobalAttentionStore() if self.use_global else None
        )
        self._shared_global_store: Optional[_GlobalAttentionStore] = None

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def set_global_store(self, store: _GlobalAttentionStore) -> None:
        """Attach a shared global attention store."""

        if not isinstance(store, _GlobalAttentionStore):
            raise TypeError("store must be an instance of _GlobalAttentionStore")
        self.use_global = True
        self._shared_global_store = store
        if self._local_global_store is not None:
            self._local_global_store = None

    def _global_store(self) -> Optional[_GlobalAttentionStore]:
        if self._shared_global_store is not None:
            return self._shared_global_store
        return self._local_global_store

    def _activate_global(self, matrix: Tensor) -> Tensor:
        activation = self._global_activation_name
        if activation == "softmax":
            return torch.softmax(matrix, dim=-1)
        if activation == "sigmoid":
            return torch.sigmoid(matrix)
        if activation == "tanh":
            return torch.tanh(matrix)
        if activation == "relu":
            return torch.relu(matrix)
        if activation in {"identity", "linear", "none"}:
            return matrix
        raise ValueError(f"Unsupported global activation '{self._global_activation_name}'")

    def forward(
        self,
        inputs: Tensor,
        *,
        key_padding_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if inputs.dim() != 3:
            raise ValueError("inputs must have shape (batch, length, embed_dim)")
        if inputs.size(-1) != self.embed_dim:
            raise ValueError(
                f"Expected embedding dimension {self.embed_dim}, got {inputs.size(-1)}"
            )

        batch, length, _ = inputs.shape
        query = self.q_proj(inputs)
        key = self.k_proj(inputs)
        value = self.v_proj(inputs)

        query = query.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores = torch.tanh(scores)

        mask: Optional[Tensor] = None
        if key_padding_mask is not None:
            if key_padding_mask.shape != (batch, length):
                raise ValueError(
                    "key_padding_mask must have shape (batch, length) matching the inputs"
                )
            mask = key_padding_mask.to(dtype=torch.bool)
            scores = scores.masked_fill(~mask[:, None, None, :], float("-inf"))

        finite_scores = torch.where(
            torch.isfinite(scores),
            scores,
            torch.full_like(scores, -torch.finfo(scores.dtype).max),
        )
        max_per_sample = finite_scores.amax(dim=(-1, -2, -3), keepdim=True)
        max_per_sample = torch.where(
            torch.isfinite(max_per_sample), max_per_sample, torch.zeros_like(max_per_sample)
        )
        stabilised = scores - max_per_sample
        exp_scores = torch.exp(stabilised)
        exp_scores = torch.where(torch.isfinite(scores), exp_scores, torch.zeros_like(exp_scores))
        if mask is not None:
            key_mask = mask[:, None, None, :].to(dtype=exp_scores.dtype)
            query_mask = mask[:, None, :, None].to(dtype=exp_scores.dtype)
            exp_scores = exp_scores * key_mask * query_mask
        denom = exp_scores.sum(dim=(-1, -2, -3), keepdim=True)
        denom = denom.clamp_min(torch.finfo(exp_scores.dtype).tiny)
        weights = exp_scores / denom
        weights = self.dropout(weights)
        renorm = weights.sum(dim=(-1, -2, -3), keepdim=True)
        renorm = renorm.clamp_min(torch.finfo(weights.dtype).tiny)
        weights = weights / renorm

        if self.use_global:
            store = self._global_store()
            if store is None:
                raise RuntimeError("Global attention requested but no store is available")
            global_matrix = store.matrix_for(length, device=inputs.device, dtype=inputs.dtype)
            activated = self._activate_global(global_matrix).clamp_min(0.0)
            global_scores = activated.unsqueeze(0).expand(batch, length, length)
            if mask is not None:
                mask_float = mask.to(dtype=global_scores.dtype)
                global_scores = (
                    global_scores
                    * mask_float[:, None, :]
                    * mask_float[:, :, None]
                )
            sums = global_scores.sum(dim=(-1, -2), keepdim=True)
            sums = sums.clamp_min(torch.finfo(global_scores.dtype).tiny)
            normalised = global_scores / sums
            normalised = normalised.unsqueeze(1).to(dtype=weights.dtype)
            weights = (1.0 - self.global_mix) * weights + self.global_mix * normalised

        attended = torch.matmul(weights, value)

        if mask is not None:
            query_mask = mask[:, None, :, None].to(dtype=attended.dtype)
            attended = attended * query_mask

        attended = attended.transpose(1, 2).contiguous().view(batch, length, self.embed_dim)
        output = self.out_proj(attended)
        return output, weights


class _TemporalConvBlock(nn.Module):
    """Temporal 2D convolutional block operating over time and joints."""

    def __init__(
        self,
        embed_dim: int,
        kernel_size: int,
        dropout: float,
        *,
        dilation: int = 1,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        if kernel_size <= 0:
            raise ValueError("kernel_size must be positive")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd to preserve temporal dimensions")
        if dilation <= 0:
            raise ValueError("dilation must be a positive integer")
        if negative_slope < 0:
            raise ValueError("negative_slope must be non-negative")

        padding = (dilation * (kernel_size - 1) // 2, 0)
        self.depthwise = nn.Conv2d(
            embed_dim,
            embed_dim,
            kernel_size=(kernel_size, 1),
            padding=padding,
            dilation=(dilation, 1),
            groups=embed_dim,
            bias=False,
        )
        self.pointwise = nn.Conv2d(embed_dim, embed_dim, kernel_size=1, bias=False)
        self.activation = nn.LeakyReLU(negative_slope=negative_slope, inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.dim() != 4:
            raise ValueError("inputs must have shape (batch, time, joints, embed_dim)")
        batch, time, joints, embed_dim = inputs.shape
        reshaped = inputs.permute(0, 3, 1, 2)
        conv = self.depthwise(reshaped)
        conv = self.activation(conv)
        conv = self.pointwise(conv)
        conv = self.dropout(conv)
        conv = conv.permute(0, 2, 3, 1)
        if conv.shape != (batch, time, joints, embed_dim):
            raise RuntimeError("Temporal convolution altered the input shape unexpectedly")
        return conv


class KeypointStreamEncoder(nn.Module):
    """Encode temporal keypoint sequences for a particular stream."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        *,
        num_heads: int = 4,
        temporal_blocks: int = 2,
        temporal_kernel: int = 3,
        temporal_dilation: int = 1,
        dropout: float = 0.0,
        use_global_attention: bool = False,
        global_attention_activation: str = "softmax",
        global_attention_mix: float = 0.5,
        leaky_relu_negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be a positive integer")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer")
        if temporal_blocks < 0:
            raise ValueError("temporal_blocks must be non-negative")
        if temporal_dilation <= 0:
            raise ValueError("temporal_dilation must be a positive integer")
        if leaky_relu_negative_slope < 0:
            raise ValueError("leaky_relu_negative_slope must be non-negative")

        self.in_dim = int(in_dim)
        self.embed_dim = int(embed_dim)
        self.num_heads = int(num_heads)
        self.temporal_blocks = int(temporal_blocks)
        self.temporal_kernel = int(temporal_kernel)
        self.temporal_dilation = int(temporal_dilation)
        self.leaky_relu_negative_slope = float(leaky_relu_negative_slope)

        self.input_norm = nn.LayerNorm(self.in_dim)
        self.input_projection = nn.Linear(self.in_dim, self.embed_dim, bias=False)
        self.projection_dropout = nn.Dropout(dropout)
        self.self_attention = _TanhMultiHeadAttention(
            self.embed_dim,
            self.num_heads,
            dropout,
            use_global=use_global_attention,
            global_activation=global_attention_activation,
            global_mix=global_attention_mix,
        )
        self.attention_norm = nn.LayerNorm(self.embed_dim)
        self.temporal_layers = nn.ModuleList(
            [
                _TemporalConvBlock(
                    self.embed_dim,
                    self.temporal_kernel,
                    dropout,
                    dilation=self.temporal_dilation,
                    negative_slope=self.leaky_relu_negative_slope,
                )
                for _ in range(self.temporal_blocks)
            ]
        )
        self.temporal_norms = nn.ModuleList(
            [nn.LayerNorm(self.embed_dim) for _ in range(self.temporal_blocks)]
        )
        self._last_attention_weights: Optional[Tensor] = None

    def set_global_attention_store(self, store: _GlobalAttentionStore) -> None:
        """Attach a shared global attention store to the underlying attention."""

        self.self_attention.set_global_store(store)

    @staticmethod
    def _masked_mean(
        values: Tensor,
        mask: Optional[Tensor],
        *,
        dim: int,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if mask is None:
            mean = values.mean(dim=dim)
            frame_mask = None
        else:
            mask = mask.to(dtype=torch.bool)
            if mask.dim() != values.dim() - 1:
                raise ValueError(
                    "Mask must match the batch/time/joint dimensions of the values"
                )
            mask_float = mask.to(dtype=values.dtype)
            while mask_float.dim() < values.dim():
                mask_float = mask_float.unsqueeze(-1)
            masked_sum = (values * mask_float).sum(dim=dim)
            denom = mask_float.sum(dim=dim).clamp_min(1.0)
            mean = masked_sum / denom
            frame_mask = mask.any(dim=dim)
        return mean, frame_mask

    def _joint_positional_encoding(
        self, joints: int, *, device: torch.device, dtype: torch.dtype
    ) -> Tensor:
        positions = torch.arange(joints, device=device, dtype=dtype).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.embed_dim, 2, device=device, dtype=dtype)
            * (-math.log(10000.0) / max(self.embed_dim, 1))
        )
        encoding = torch.zeros(joints, self.embed_dim, device=device, dtype=dtype)
        encoding[:, 0::2] = torch.sin(positions * div_term)
        if self.embed_dim % 2 == 1:
            encoding[:, 1::2] = torch.cos(positions * div_term[:-1])
        else:
            encoding[:, 1::2] = torch.cos(positions * div_term)
        return encoding

    def forward(
        self,
        keypoints: Tensor,
        *,
        mask: Optional[Tensor] = None,
        frame_mask: Optional[Tensor] = None,
    ) -> KeypointStreamOutput:
        """Encode ``keypoints`` with shape ``(batch, time, joints, dim)``."""

        if keypoints.dim() != 4:
            raise ValueError(
                "Keypoint streams must have four dimensions: (batch, time, joints, dim)"
            )
        if keypoints.size(-1) != self.in_dim:
            raise ValueError(
                f"Expected last dimension {self.in_dim}, received {keypoints.size(-1)}"
            )

        batch, time, joints, _ = keypoints.shape
        normed = self.input_norm(keypoints)
        projected = self.input_projection(normed)
        positional = self._joint_positional_encoding(
            joints, device=projected.device, dtype=projected.dtype
        )
        projected = projected + positional.view(1, 1, joints, self.embed_dim)
        projected = self.projection_dropout(projected)

        mask_bool: Optional[Tensor]
        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool)
            if mask_bool.shape[:3] != (batch, time, joints):
                raise ValueError(
                    "Mask must match the (batch, time, joints) dimensions of the keypoints"
                )
        else:
            mask_bool = None

        flat = projected.view(batch * time, joints, self.embed_dim)
        flat_mask = mask_bool.view(batch * time, joints) if mask_bool is not None else None

        attn_out, attn_weights = self.self_attention(flat, key_padding_mask=flat_mask)
        attn_out = attn_out.view(batch, time, joints, self.embed_dim)
        attn_out = self.attention_norm(attn_out + projected)

        hidden = attn_out
        for conv, norm in zip(self.temporal_layers, self.temporal_norms):
            conv_out = conv(hidden)
            hidden = norm(hidden + conv_out)

        if mask_bool is not None:
            hidden = hidden * mask_bool.unsqueeze(-1).to(dtype=hidden.dtype)

        frame_mean, inferred_frame_mask = self._masked_mean(hidden, mask_bool, dim=2)

        if frame_mask is not None:
            frame_mask = frame_mask.to(dtype=torch.bool)
            if frame_mask.shape[:2] != (batch, time):
                raise ValueError(
                    "Frame mask must match the (batch, time) dimensions of the keypoints"
                )
            if inferred_frame_mask is None:
                inferred_frame_mask = frame_mask
            else:
                inferred_frame_mask = torch.logical_and(inferred_frame_mask, frame_mask)

        self._last_attention_weights = attn_weights.detach().view(
            batch, time, self.self_attention.num_heads, joints, joints
        )

        return KeypointStreamOutput(
            joint_embeddings=hidden,
            frame_embeddings=frame_mean,
            joint_mask=mask_bool,
            frame_mask=inferred_frame_mask,
        )


@dataclass
class MultiStreamKeypointAttentionOutput:
    stream_embeddings: Dict[str, Tensor]
    frame_mask: Dict[str, Optional[Tensor]]
    fused_embedding: Tensor
    fused_mask: Optional[Tensor]
    attention: Optional[Tensor]


class MultiStreamKeypointAttention(nn.Module):
    """Attention block operating across multiple keypoint streams."""

    def __init__(
        self,
        embed_dim: int,
        *,
        num_heads: int = 4,
        ff_multiplier: int = 4,
        dropout: float = 0.0,
        stream_order: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__()
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.stream_order = tuple(stream_order) if stream_order is not None else None
        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=False,
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        ff_dim = max(embed_dim * ff_multiplier, embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.fuse_norm = nn.LayerNorm(embed_dim)
        self.fuse_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )

    def _ordered_items(
        self,
        streams: Mapping[str, KeypointStreamOutput],
    ) -> Sequence[Tuple[str, KeypointStreamOutput]]:
        if self.stream_order is None:
            return tuple(streams.items())
        ordered: list[Tuple[str, KeypointStreamOutput]] = []
        for name in self.stream_order:
            if name in streams:
                ordered.append((name, streams[name]))
        for name, output in streams.items():
            if all(name != existing for existing, _ in ordered):
                ordered.append((name, output))
        return tuple(ordered)

    def forward(
        self,
        streams: Mapping[str, KeypointStreamOutput],
    ) -> MultiStreamKeypointAttentionOutput:
        ordered = self._ordered_items(streams)
        if not ordered:
            raise ValueError("At least one keypoint stream is required for attention")

        names, outputs = zip(*ordered)
        frame_embeddings = torch.stack([item.frame_embeddings for item in outputs], dim=2)
        batch, time, stream_count, embed_dim = frame_embeddings.shape
        flat = frame_embeddings.view(batch * time, stream_count, embed_dim)
        transposed = flat.transpose(0, 1)

        padding_mask: Optional[Tensor]
        frame_masks: list[Optional[Tensor]] = []
        if any(item.frame_mask is not None for item in outputs):
            stacked_masks = []
            for item in outputs:
                if item.frame_mask is None:
                    stacked_masks.append(
                        torch.ones(batch, time, device=frame_embeddings.device, dtype=torch.bool)
                    )
                else:
                    stacked_masks.append(item.frame_mask.to(device=frame_embeddings.device))
                frame_masks.append(stacked_masks[-1])
            mask_tensor = torch.stack(stacked_masks, dim=2)
            padding_mask = torch.logical_not(mask_tensor).view(batch * time, stream_count)
        else:
            frame_masks.extend([None] * len(outputs))
            padding_mask = None

        attn_output, attn_weights = self.attention(
            transposed,
            transposed,
            transposed,
            key_padding_mask=padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )

        attn_output = attn_output.transpose(0, 1).reshape(batch, time, stream_count, embed_dim)
        residual = frame_embeddings
        attn_output = self.norm1(attn_output + residual)
        ff_output = self.ff(attn_output)
        attn_output = self.norm2(attn_output + ff_output)

        mask_tensor: Optional[Tensor]
        if padding_mask is not None:
            mask_tensor = torch.logical_not(padding_mask).view(batch, time, stream_count)
        else:
            mask_tensor = None

        if mask_tensor is None:
            fused = attn_output.mean(dim=2)
            fused_mask = None
        else:
            mask_float = mask_tensor.to(dtype=attn_output.dtype).unsqueeze(-1)
            summed = (attn_output * mask_float).sum(dim=2)
            denom = mask_float.sum(dim=2).clamp_min(1.0)
            fused = summed / denom
            fused_mask = mask_tensor.all(dim=2)

        fused = self.fuse_proj(self.fuse_norm(fused))

        if attn_weights is not None:
            attn_shape = (
                self.num_heads,
                batch,
                time,
                stream_count,
                stream_count,
            )
            attention_tensor = attn_weights.view(attn_shape).permute(1, 2, 0, 3, 4)
        else:
            attention_tensor = None

        stream_embeddings = {
            name: attn_output[:, :, idx]
            for idx, name in enumerate(names)
        }
        mask_mapping = {
            name: frame_masks[idx] if frame_masks[idx] is not None else None
            for idx, name in enumerate(names)
        }

        return MultiStreamKeypointAttentionOutput(
            stream_embeddings=stream_embeddings,
            frame_mask=mask_mapping,
            fused_embedding=fused,
            fused_mask=fused_mask,
            attention=attention_tensor,
        )


@dataclass
class MSKAOutput:
    """Structured output for the MSKA encoder."""

    stream_embeddings: Dict[str, Tensor]
    joint_embeddings: Dict[str, Tensor]
    stream_masks: Dict[str, Optional[Tensor]]
    frame_masks: Dict[str, Optional[Tensor]]
    fused_embedding: Tensor
    fused_mask: Optional[Tensor]
    attention: Optional[Tensor]


class StreamCTCHead(nn.Module):
    """CTC classification head with temporal convolutions for individual streams."""

    def __init__(
        self,
        in_dim: int,
        vocab_size: int,
        dropout: float = 0.0,
        *,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if negative_slope < 0:
            raise ValueError("negative_slope must be non-negative")
        self.in_dim = int(in_dim)
        self.vocab_size = int(vocab_size)
        self.hidden_dim = self.in_dim
        self.negative_slope = float(negative_slope)

        self.input_linear = nn.Linear(self.in_dim, self.hidden_dim)
        self.input_activation = nn.LeakyReLU(
            negative_slope=self.negative_slope, inplace=True
        )
        self.input_norm_1d = nn.BatchNorm1d(self.hidden_dim)
        self.input_norm_2d = nn.BatchNorm2d(self.hidden_dim)
        self.temporal_conv_1d = nn.Conv1d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            padding=1,
            bias=False,
        )
        self.temporal_norm_1d = nn.BatchNorm1d(self.hidden_dim)
        self.temporal_conv_2d = nn.Conv2d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=(3, 3),
            padding=(1, 1),
            bias=False,
        )
        self.temporal_norm_2d = nn.BatchNorm2d(self.hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(self.hidden_dim, self.vocab_size)

    def forward(self, features: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        logits, _ = self.forward_with_intermediate(features, mask=mask)
        return logits

    def forward_with_intermediate(
        self, features: Tensor, mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, Tensor]:
        """Return logits and temporal features preserving the time dimension."""

        temporal = self._temporal_features(features, mask=mask)
        logits = self.output_projection(temporal)
        return logits, temporal

    def _temporal_features(self, features: Tensor, mask: Optional[Tensor]) -> Tensor:
        if features.dim() not in {3, 4}:
            raise ValueError(
                "features must have shape (batch, time, dim) or (batch, time, joints, dim)"
            )
        if features.size(-1) != self.in_dim:
            raise ValueError(
                f"Expected input dimension {self.in_dim}, received {features.size(-1)}"
            )

        if features.dim() == 3:
            hidden = self.input_linear(features)
            hidden = hidden.transpose(1, 2).contiguous()
            hidden = self.input_norm_1d(hidden)
            hidden = self.input_activation(hidden)
            hidden = self.temporal_conv_1d(hidden)
            hidden = self.temporal_norm_1d(hidden)
            hidden = self.input_activation(hidden)
            hidden = self.dropout(hidden)
            return hidden.transpose(1, 2).contiguous()

        batch, time, joints, _ = features.shape
        hidden = self.input_linear(features)
        hidden = hidden.permute(0, 3, 1, 2).contiguous()
        hidden = self.input_norm_2d(hidden)
        hidden = self.input_activation(hidden)
        hidden = self.temporal_conv_2d(hidden)
        hidden = self.temporal_norm_2d(hidden)
        hidden = self.input_activation(hidden)
        hidden = self.dropout(hidden)

        if mask is not None:
            if mask.shape != (batch, time, joints):
                raise ValueError(
                    "mask must match the (batch, time, joints) dimensions of the inputs"
                )
            mask_float = mask.to(dtype=hidden.dtype).unsqueeze(1)
            hidden = hidden * mask_float
            denom = mask_float.sum(dim=-1).clamp_min(1.0)
            pooled = hidden.sum(dim=-1) / denom
        else:
            pooled = hidden.mean(dim=-1)

        return pooled.transpose(1, 2).contiguous()


class FusedCTCHead(StreamCTCHead):
    """CTC head operating on the fused stream representation with temporal context."""

    def __init__(
        self,
        in_dim: int,
        vocab_size: int,
        dropout: float = 0.0,
        *,
        negative_slope: float = 0.01,
    ) -> None:
        super().__init__(
            in_dim,
            vocab_size,
            dropout=dropout,
            negative_slope=negative_slope,
        )


class MSKAEncoder(nn.Module):
    """Multi-stream keypoint encoder producing auxiliary CTC logits."""

    def __init__(
        self,
        *,
        input_dim: int,
        embed_dim: int,
        stream_names: Sequence[str],
        num_heads: int,
        ff_multiplier: int,
        dropout: float,
        ctc_vocab_size: int,
        detach_teacher: bool = True,
        stream_attention_heads: int = 4,
        stream_temporal_blocks: int = 2,
        stream_temporal_kernel: int = 3,
        stream_temporal_dilation: int = 1,
        use_global_attention: bool = False,
        global_attention_activation: str = "softmax",
        global_attention_mix: float = 0.5,
        global_attention_shared: bool = False,
        leaky_relu_negative_slope: float = 0.01,
    ) -> None:
        super().__init__()
        if not stream_names:
            raise ValueError("stream_names must contain at least one entry")
        if not 0.0 <= global_attention_mix <= 1.0:
            raise ValueError("global_attention_mix must be between 0 and 1 inclusive")
        if leaky_relu_negative_slope < 0:
            raise ValueError("leaky_relu_negative_slope must be non-negative")

        self.embed_dim = embed_dim
        self.stream_names = tuple(stream_names)
        self.detach_teacher = detach_teacher
        self.leaky_relu_negative_slope = float(leaky_relu_negative_slope)
        self._shared_global_store: Optional[_GlobalAttentionStore]
        if use_global_attention and global_attention_shared:
            self._shared_global_store = _GlobalAttentionStore()
        else:
            self._shared_global_store = None
        self.encoders = nn.ModuleDict(
            {
                name: KeypointStreamEncoder(
                    input_dim,
                    embed_dim,
                    num_heads=stream_attention_heads,
                    temporal_blocks=stream_temporal_blocks,
                    temporal_kernel=stream_temporal_kernel,
                    temporal_dilation=stream_temporal_dilation,
                    dropout=dropout,
                    use_global_attention=use_global_attention,
                    global_attention_activation=global_attention_activation,
                    global_attention_mix=global_attention_mix,
                    leaky_relu_negative_slope=self.leaky_relu_negative_slope,
                )
                for name in self.stream_names
            }
        )
        if self._shared_global_store is not None:
            for encoder in self.encoders.values():
                encoder.set_global_attention_store(self._shared_global_store)
        self.attention = MultiStreamKeypointAttention(
            embed_dim,
            num_heads=num_heads,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
            stream_order=self.stream_names,
        )
        self.stream_heads = nn.ModuleDict(
            {
                name: StreamCTCHead(
                    embed_dim,
                    ctc_vocab_size,
                    dropout=dropout,
                    negative_slope=self.leaky_relu_negative_slope,
                )
                for name in self.stream_names
            }
        )
        self.fuse_head = FusedCTCHead(
            embed_dim,
            ctc_vocab_size,
            dropout=dropout,
            negative_slope=self.leaky_relu_negative_slope,
        )
        self._last_output: Optional[MSKAOutput] = None

    def forward(
        self,
        streams: Mapping[str, Mapping[str, Tensor]],
    ) -> MSKAOutput:
        """Encode a mapping of keypoint streams."""

        encoded_streams: Dict[str, KeypointStreamOutput] = {}
        joint_embeddings: Dict[str, Tensor] = {}
        stream_masks: Dict[str, Optional[Tensor]] = {}
        frame_masks: Dict[str, Optional[Tensor]] = {}

        for name in self.stream_names:
            payload = streams.get(name)
            if payload is None:
                continue
            points = payload.get("points")
            if points is None:
                continue
            mask = payload.get("mask")
            frame_mask = payload.get("frame_mask")
            encoded = self.encoders[name](points, mask=mask, frame_mask=frame_mask)
            encoded_streams[name] = encoded
            joint_embeddings[name] = encoded.joint_embeddings
            stream_masks[name] = encoded.joint_mask
            frame_masks[name] = encoded.frame_mask

        if not encoded_streams:
            raise ValueError("No valid keypoint streams provided to MSKAEncoder")

        attn_output = self.attention(encoded_streams)
        result = MSKAOutput(
            stream_embeddings=attn_output.stream_embeddings,
            joint_embeddings=joint_embeddings,
            stream_masks=stream_masks,
            frame_masks=attn_output.frame_mask,
            fused_embedding=attn_output.fused_embedding,
            fused_mask=attn_output.fused_mask,
            attention=attn_output.attention,
        )
        self._last_output = result
        return result

    def auxiliary_logits(
        self,
        output: Optional[MSKAOutput] = None,
        *,
        detach_teacher: Optional[bool] = None,
    ) -> Dict[str, Dict[str, Tensor]]:
        """Return logits for CTC supervision and distillation signals."""

        if output is None:
            if self._last_output is None:
                raise RuntimeError("No MSKA output available to compute auxiliary logits")
            output = self._last_output

        fused_logits, fused_temporal = self.fuse_head.forward_with_intermediate(
            output.fused_embedding
        )
        use_teacher = self.detach_teacher if detach_teacher is None else detach_teacher
        teacher_logits = fused_logits.detach() if use_teacher else fused_logits
        fused_probs = torch.softmax(fused_logits, dim=-1)
        teacher_probs = torch.softmax(teacher_logits, dim=-1)
        fused_temporal_probs = torch.softmax(fused_temporal, dim=1)
        teacher_temporal_source = fused_temporal.detach() if use_teacher else fused_temporal
        teacher_temporal_probs = torch.softmax(teacher_temporal_source, dim=1)

        stream_logits: Dict[str, Tensor] = {}
        distillation: Dict[str, Tensor] = {}
        stream_probs: Dict[str, Tensor] = {}
        stream_temporal_features: Dict[str, Tensor] = {}
        stream_temporal_probs: Dict[str, Tensor] = {}

        for name, embeddings in output.stream_embeddings.items():
            joint_embeddings = output.joint_embeddings.get(name)
            joint_mask = output.stream_masks.get(name)
            head_inputs = joint_embeddings if joint_embeddings is not None else embeddings
            logits, temporal_features = self.stream_heads[name].forward_with_intermediate(
                head_inputs, mask=joint_mask
            )
            stream_logits[name] = logits
            distillation[name] = teacher_logits
            stream_probs[name] = torch.softmax(logits, dim=-1)
            stream_temporal_features[name] = temporal_features
            stream_temporal_probs[name] = torch.softmax(temporal_features, dim=1)

        probabilities = {
            "fused": fused_probs,
            "stream": stream_probs,
            "distillation": {name: teacher_probs for name in stream_logits},
            "temporal": {
                "fused": fused_temporal_probs,
                "stream": stream_temporal_probs,
                "distillation": {
                    name: teacher_temporal_probs for name in stream_logits
                },
            },
        }

        return {
            "stream": stream_logits,
            "fused": {
                "logits": fused_logits,
                "mask": output.fused_mask,
                "probs": fused_probs,
                "temporal_probs": fused_temporal_probs,
            },
            "distillation": distillation,
            "frame_masks": output.frame_masks,
            "probabilities": probabilities,
            "temporal_features": {
                "fused": fused_temporal,
                "stream": stream_temporal_features,
            },
        }

    @property
    def last_output(self) -> Optional[MSKAOutput]:
        """Return the cached output from the most recent forward pass."""

        return self._last_output
