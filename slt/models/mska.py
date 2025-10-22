"""Modules implementing the Multi-Stream Keypoint Attention encoder."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple

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


class KeypointStreamEncoder(nn.Module):
    """Encode temporal keypoint sequences for a particular stream."""

    def __init__(
        self,
        in_dim: int,
        embed_dim: int,
        *,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        activation: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be a positive integer")
        if embed_dim <= 0:
            raise ValueError("embed_dim must be a positive integer")

        hidden_dim = hidden_dim or embed_dim
        if activation is None:
            activation = nn.GELU()

        self.in_dim = int(in_dim)
        self.embed_dim = int(embed_dim)
        self.hidden_dim = int(hidden_dim)
        self.norm = nn.LayerNorm(self.in_dim)
        self.projection = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_dim),
            activation,
            nn.Dropout(dropout),
            nn.Linear(self.hidden_dim, self.embed_dim),
        )

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

        normed = self.norm(keypoints)
        joint_embeddings = self.projection(normed)

        if mask is not None:
            mask_bool = mask.to(dtype=torch.bool)
            if mask_bool.shape[:3] != joint_embeddings.shape[:3]:
                raise ValueError(
                    "Mask must match the (batch, time, joints) dimensions of the keypoints"
                )
            expanded_mask = mask_bool.unsqueeze(-1).expand_as(joint_embeddings)
            joint_embeddings = joint_embeddings * expanded_mask.to(dtype=joint_embeddings.dtype)
        else:
            mask_bool = None

        frame_mean, inferred_frame_mask = self._masked_mean(
            joint_embeddings, mask_bool, dim=2
        )

        if frame_mask is not None:
            frame_mask = frame_mask.to(dtype=torch.bool)
            if frame_mask.shape[:2] != joint_embeddings.shape[:2]:
                raise ValueError(
                    "Frame mask must match the (batch, time) dimensions of the keypoints"
                )
            if inferred_frame_mask is None:
                inferred_frame_mask = frame_mask
            else:
                inferred_frame_mask = torch.logical_and(inferred_frame_mask, frame_mask)

        return KeypointStreamOutput(
            joint_embeddings=joint_embeddings,
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
    """CTC classification head for individual streams."""

    def __init__(self, in_dim: int, vocab_size: int, dropout: float = 0.0) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError("in_dim must be positive")
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        self.in_dim = in_dim
        self.vocab_size = vocab_size
        self.norm = nn.LayerNorm(in_dim)
        self.dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(in_dim, vocab_size)

    def forward(self, features: Tensor) -> Tensor:
        if features.size(-1) != self.in_dim:
            raise ValueError(
                f"Expected input dimension {self.in_dim}, received {features.size(-1)}"
            )
        hidden = self.dropout(self.norm(features))
        return self.proj(hidden)


class FusedCTCHead(StreamCTCHead):
    """CTC head operating on the fused stream representation."""

    pass


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
    ) -> None:
        super().__init__()
        if not stream_names:
            raise ValueError("stream_names must contain at least one entry")

        self.embed_dim = embed_dim
        self.stream_names = tuple(stream_names)
        self.detach_teacher = detach_teacher
        self.encoders = nn.ModuleDict(
            {
                name: KeypointStreamEncoder(
                    input_dim,
                    embed_dim,
                    hidden_dim=embed_dim,
                    dropout=dropout,
                )
                for name in self.stream_names
            }
        )
        self.attention = MultiStreamKeypointAttention(
            embed_dim,
            num_heads=num_heads,
            ff_multiplier=ff_multiplier,
            dropout=dropout,
            stream_order=self.stream_names,
        )
        self.stream_heads = nn.ModuleDict(
            {
                name: StreamCTCHead(embed_dim, ctc_vocab_size, dropout=dropout)
                for name in self.stream_names
            }
        )
        self.fuse_head = FusedCTCHead(embed_dim, ctc_vocab_size, dropout=dropout)
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

        fused_logits = self.fuse_head(output.fused_embedding)
        use_teacher = self.detach_teacher if detach_teacher is None else detach_teacher
        teacher_logits = fused_logits.detach() if use_teacher else fused_logits

        stream_logits: Dict[str, Tensor] = {}
        distillation: Dict[str, Tensor] = {}
        for name, embeddings in output.stream_embeddings.items():
            logits = self.stream_heads[name](embeddings)
            stream_logits[name] = logits
            distillation[name] = teacher_logits

        return {
            "stream": stream_logits,
            "fused": {"logits": fused_logits, "mask": output.fused_mask},
            "distillation": distillation,
            "frame_masks": output.frame_masks,
        }

    @property
    def last_output(self) -> Optional[MSKAOutput]:
        """Return the cached output from the most recent forward pass."""

        return self._last_output
