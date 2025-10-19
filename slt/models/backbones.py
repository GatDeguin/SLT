"""Backbone neural network definitions used by the SLT models."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
from torch import Tensor, nn


@dataclass
class ViTConfig:
    """Lightweight configuration container for :class:`ViTSmallPatch16`.

    Attributes
    ----------
    image_size:
        Expected input image resolution (height == width). The model can
        interpolate positional encodings at inference time for different
        resolutions, but the initial positional parameters are created using
        this value.
    patch_size:
        Square patch size used by the convolutional patch embedding layer.
    in_channels:
        Number of channels in the input image (``3`` for RGB inputs).
    embed_dim:
        Size of the token embedding.
    depth:
        Number of transformer encoder layers.
    num_heads:
        Number of attention heads per layer.
    mlp_ratio:
        Expansion factor applied to the feed-forward network hidden size.
    dropout:
        Dropout probability applied to token embeddings and feed-forward
        blocks.
    attention_dropout:
        Dropout probability used inside the attention mechanism.
    stochastic_dropout:
        Dropout probability applied to residual connections (stochastic depth).
    """

    image_size: int = 224
    patch_size: int = 16
    in_channels: int = 3
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    stochastic_dropout: float = 0.0


class PatchEmbed(nn.Module):
    """Convert an image into a sequence of patch embeddings."""

    def __init__(self, in_channels: int, embed_dim: int, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        if x.dim() != 4:
            raise ValueError(
                "PatchEmbed expects input in BCHW format, received tensor with "
                f"shape {tuple(x.shape)}"
            )
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class StochasticDepth(nn.Module):
    """Implement stochastic depth with per-sample masking."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: Tensor, residual: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x + residual
        keep_prob = 1.0 - self.p
        shape = (x.size(0),) + (1,) * (x.dim() - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return residual + x * random_tensor / keep_prob


class TransformerEncoderLayer(nn.Module):
    """A ViT-style encoder layer using ``nn.MultiheadAttention``."""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float,
        dropout: float,
        attention_dropout: float,
        stochastic_dropout: float,
    ) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=attention_dropout,
            batch_first=True,
        )
        self.drop_path = StochasticDepth(stochastic_dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, int(embed_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(embed_dim * mlp_ratio), embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        x = self.norm1(x)
        x, _ = self.attn(x, x, x)
        x = self.drop_path(x, residual)
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = self.drop_path(x, residual)
        return x


class ViTSmallPatch16(nn.Module):
    """A minimal ViT-S/16 backbone compatible with PyTorch.

    The implementation intentionally mirrors the behaviour of the ViT-S/16 model
    popularised by DINO/DINOv2 while keeping the dependency surface minimal.
    It exposes a small configuration dataclass to simplify experimentation and
    provides hooks that make swapping this stub for an actual DINOv2 backbone in
    production straightforward.
    """

    def __init__(self, config: Optional[ViTConfig] = None) -> None:
        super().__init__()
        self.config = config or ViTConfig()
        self.patch_embed = PatchEmbed(
            self.config.in_channels,
            self.config.embed_dim,
            self.config.patch_size,
        )
        num_patches = (self.config.image_size // self.config.patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, self.config.embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, self.config.embed_dim))
        self.pos_drop = nn.Dropout(self.config.dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerEncoderLayer(
                    self.config.embed_dim,
                    self.config.num_heads,
                    self.config.mlp_ratio,
                    self.config.dropout,
                    self.config.attention_dropout,
                    self.config.stochastic_dropout,
                )
                for _ in range(self.config.depth)
            ]
        )
        self.norm = nn.LayerNorm(self.config.embed_dim)

        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.zeros_(module.bias)
            nn.init.ones_(module.weight)

    def _interpolate_positional_encoding(self, x: Tensor) -> Tensor:
        n_patches = x.size(1) - 1
        n_pos_tokens = self.pos_embed.size(1) - 1
        if n_patches == n_pos_tokens:
            return self.pos_embed

        class_pos_embed = self.pos_embed[:, 0:1]
        patch_pos_embed = self.pos_embed[:, 1:]
        dim = x.size(-1)
        h = w = int(math.sqrt(n_patches))
        orig_h = orig_w = int(math.sqrt(n_pos_tokens))
        if h * w != n_patches:
            raise ValueError(
                "The number of patches must form a square grid for positional "
                f"interpolation, received {n_patches} patches."
            )
        patch_pos_embed = patch_pos_embed.reshape(1, orig_h, orig_w, dim).permute(0, 3, 1, 2)
        patch_pos_embed = torch.nn.functional.interpolate(
            patch_pos_embed,
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        )
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).reshape(1, h * w, dim)
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def forward_features(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(x.size(0), -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        pos_embed = self._interpolate_positional_encoding(x)
        x = x + pos_embed[:, : x.size(1)]
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x[:, 0], x[:, 1:]

    def forward(self, x: Tensor) -> Tensor:
        """Return the class token embedding for convenience."""
        cls_token, _ = self.forward_features(x)
        return cls_token

    def forward_with_patches(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        """Return both the pooled class token and patch embeddings."""
        return self.forward_features(x)

    # ---------------------------------------------------------------------
    # Extension hooks
    # ---------------------------------------------------------------------
    def load_pretrained_weights(self, state_dict: dict[str, Tensor]) -> None:
        """Load weights trained elsewhere.

        The method is intentionally lightweight so that production code can
        replace it with custom DINOv2 checkpoint loading logic.
        """

        self.load_state_dict(state_dict, strict=False)

    def as_backbone(self) -> "ViTSmallPatch16":
        """Return the module itself.

        This hook mirrors the API provided by DINOv2 models, making it trivial
        to swap the stub implementation with ``dinov2.vits14`` or similar
        architectures when running in production.
        """

        return self
