"""Lightweight TinyViT style stub module for DINO pre-training scripts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn


@dataclass
class TinyViTConfig:
    """Configuration container for :class:`TinyViTStub`.

    Attributes
    ----------
    in_channels:
        Number of channels in the input images.
    patch_size:
        Size of the convolutional patch embedding kernel.
    hidden_dim:
        Number of channels produced by the stem convolution.
    proj_dim:
        Dimension of the final projected embedding vector.
    """

    in_channels: int = 3
    patch_size: int = 16
    hidden_dim: int = 64
    proj_dim: int = 256


class TinyViTStub(nn.Module):
    """A tiny Vision Transformer inspired stub used for self-supervised experiments.

    The goal of this module is not to be a faithful TinyViT reproduction but a
    compact network that mimics the patch embedding and projection commonly used
    in DINO pipelines.  The architecture deliberately keeps the dependency
    footprint minimal so the accompanying scripts can run in constrained
    environments.
    """

    def __init__(self, config: TinyViTConfig | None = None) -> None:
        super().__init__()
        self.config = config or TinyViTConfig()

        self.patch_embed = nn.Conv2d(
            in_channels=self.config.in_channels,
            out_channels=self.config.hidden_dim,
            kernel_size=self.config.patch_size,
            stride=self.config.patch_size,
        )
        self.norm = nn.LayerNorm(self.config.hidden_dim)
        self.proj = nn.Sequential(
            nn.Linear(self.config.hidden_dim, self.config.proj_dim),
            nn.GELU(),
            nn.Linear(self.config.proj_dim, self.config.proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode an image batch into a latent vector.

        Parameters
        ----------
        x:
            Image batch shaped ``(batch, channels, height, width)``.

        Returns
        -------
        torch.Tensor
            Batch of projected embeddings shaped ``(batch, proj_dim)``.
        """

        patches = self.patch_embed(x)
        patches = patches.flatten(2).transpose(1, 2)
        normed = self.norm(patches)
        pooled = normed.mean(dim=1)
        return self.proj(pooled)


__all__: Tuple[str, ...] = ("TinyViTConfig", "TinyViTStub")
