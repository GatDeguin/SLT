"""Backbone neural network definitions used by the SLT models."""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Tuple, Union

import torch
from torch import Tensor, nn

try:  # pragma: no cover - optional dependency.
    from huggingface_hub import hf_hub_download  # type: ignore
except Exception:  # pragma: no cover - the dependency is optional.
    hf_hub_download = None  # type: ignore

try:  # pragma: no cover - optional dependency.
    from torchvision.models import get_model as tv_get_model  # type: ignore
    from torchvision.models import get_model_weights as tv_get_model_weights  # type: ignore
except Exception:  # pragma: no cover - optional dependency.
    tv_get_model = None  # type: ignore
    tv_get_model_weights = None  # type: ignore


BackboneFreeze = Union[bool, int, str, Iterable[str]]


def _apply_freeze(module: nn.Module, freeze: BackboneFreeze) -> None:
    """Freeze parameters in ``module`` based on ``freeze`` specification."""

    if isinstance(freeze, bool):
        if not freeze:
            return
        for parameter in module.parameters():
            parameter.requires_grad_(False)
        return

    if isinstance(freeze, int):
        _freeze_first_blocks(module, freeze)
        return

    if isinstance(freeze, str):
        directives = [part.strip() for part in freeze.replace(";", ",").split(",") if part.strip()]
        if not directives:
            return
        for directive in directives:
            if directive.lower() in {"all", "full", "backbone"}:
                _apply_freeze(module, True)
                continue
            if directive.lower() in {"embed", "patch", "patch_embed"}:
                _freeze_patch_embed(module)
                continue
            if directive.lower() in {"head", "classifier"}:
                _freeze_head(module)
                continue
            if directive.startswith("blocks[:") and directive.endswith("]"):
                count_str = directive[len("blocks[:") : -1]
                try:
                    count = int(count_str)
                except ValueError as exc:  # pragma: no cover - defensive.
                    raise ValueError(f"Invalid block freeze specification: {directive!r}") from exc
                _freeze_first_blocks(module, count)
                continue
            # Fallback to prefix-based freezing for unknown directives.
            _apply_freeze(module, [directive])
        return

    prefixes = tuple(freeze)
    if not prefixes:
        return
    for name, parameter in module.named_parameters():
        if name.startswith(prefixes):
            parameter.requires_grad_(False)


def _freeze_first_blocks(module: nn.Module, count: int) -> None:
    if count <= 0:
        return
    blocks = getattr(module, "blocks", None)
    if blocks is None:
        raise AttributeError(
            "Backbone does not expose a 'blocks' attribute required for block freezing"
        )
    frozen = 0
    for block in blocks:
        if frozen >= count:
            break
        for parameter in block.parameters():
            parameter.requires_grad_(False)
        frozen += 1


def _freeze_patch_embed(module: nn.Module) -> None:
    for attr in ("patch_embed", "patch_embedding", "stem"):
        submodule = getattr(module, attr, None)
        if isinstance(submodule, nn.Module):
            for parameter in submodule.parameters():
                parameter.requires_grad_(False)
        elif isinstance(submodule, Tensor):
            submodule.requires_grad_(False)
    for attr in ("pos_embed", "position_embedding", "cls_token", "class_token"):
        value = getattr(module, attr, None)
        if isinstance(value, nn.Parameter):
            value.requires_grad_(False)


def _freeze_head(module: nn.Module) -> None:
    for attr in ("head", "mlp_head", "linear_head", "classifier"):
        head_module = getattr(module, attr, None)
        if isinstance(head_module, nn.Module):
            for parameter in head_module.parameters():
                parameter.requires_grad_(False)


def _load_checkpoint(
    checkpoint: object,
    backbone: nn.Module,
    *,
    convert: Optional[callable] = None,
) -> None:
    """Load a checkpoint into ``backbone`` supporting common formats."""

    if isinstance(checkpoint, nn.Module):
        state_dict = checkpoint.state_dict()
    elif isinstance(checkpoint, Mapping):
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], Mapping):
            state_dict = checkpoint["state_dict"]
        elif "model" in checkpoint and isinstance(checkpoint["model"], Mapping):
            state_dict = checkpoint["model"]
        else:
            state_dict = checkpoint  # type: ignore[assignment]
    else:
        raise TypeError(
            "Unsupported checkpoint type. Expected an nn.Module or state_dict dictionary."
        )

    if convert is not None:
        state_dict = convert(state_dict)

    missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        warnings.warn(
            "Checkpoint loading reported missing or unexpected keys. "
            f"Missing keys: {missing}; unexpected keys: {unexpected}",
            RuntimeWarning,
            stacklevel=2,
        )


def _convert_dinov2_state_dict(state_dict: Mapping[str, Tensor]) -> Mapping[str, Tensor]:
    """Convert HuggingFace/local checkpoints to ``torchvision`` naming."""

    converted: dict[str, Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        for prefix in ("model.", "module.", "state_dict.", "backbone."):
            if new_key.startswith(prefix):
                new_key = new_key[len(prefix) :]
        if new_key.startswith("encoder."):
            new_key = new_key[len("encoder.") :]
        if new_key.startswith("linear_head."):
            new_key = "head." + new_key[len("linear_head.") :]
        if new_key.startswith("head.head."):
            new_key = new_key[len("head.") :]
        if new_key.startswith("trunk."):
            new_key = new_key[len("trunk.") :]
        converted[new_key] = value
    return converted


def _parse_backbone_spec(spec: str) -> tuple[str, dict[str, str]]:
    """Parse a ``load_dinov2_backbone`` specification string."""

    if "::" in spec:
        source, remainder = spec.split("::", 1)
    else:
        source, remainder = "torchvision", spec
    source = source.lower()

    if source in {"torchhub", "torchvision", "tv"}:
        parts = [part for part in remainder.split(":") if part]
        if not parts:
            raise ValueError("Torchvision specification must include a model name")
        model = parts[0]
        if len(parts) == 1:
            weight = "default"
        elif len(parts) == 2:
            weight = parts[1]
        else:
            raise ValueError(
                "Torchvision specification must follow 'model' or 'model:weights' format; "
                f"received {remainder!r}"
            )
        return "torchvision", {"model": model, "weights": weight}

    if source in {"hf", "huggingface"}:
        parts = [part for part in remainder.split(":") if part]
        if not parts:
            raise ValueError(
                "HuggingFace specification must provide at least the repository id"
            )
        if len(parts) == 1:
            repo_id = parts[0]
            filename = "pytorch_model.bin"
            model = "dinov2_vits14"
        elif len(parts) == 2:
            repo_id, filename = parts
            model = "dinov2_vits14"
        elif len(parts) == 3:
            repo_id, filename, model = parts
        else:
            raise ValueError(
                "HuggingFace specification must follow 'repo_id:filename:model' format"
            )
        return "hf", {"repo_id": repo_id, "filename": filename, "model": model}

    if source in {"file", "local"}:
        if not remainder:
            raise ValueError("File specification must include a checkpoint path")
        if ":" in remainder:
            path, model = remainder.split(":", 1)
        else:
            path = remainder
            model = "dinov2_vits14"
        return "file", {"path": path, "model": model}

    raise ValueError(
        "Backbone specification must start with 'torchhub::', 'hf::' or 'file::'."
    )


def load_dinov2_backbone(
    spec: str,
    *,
    freeze: BackboneFreeze = False,
    map_location: Optional[Union[str, torch.device]] = None,
    trust_repo: bool = True,
) -> nn.Module:
    """Load a DINOv2 backbone from Torch Hub, HuggingFace or local checkpoints."""

    source, parsed = _parse_backbone_spec(spec)

    if source == "torchvision":
        if tv_get_model is None or tv_get_model_weights is None:
            raise ImportError(
                "torchvision>=0.16 is required to load DINOv2 backbones via the torchvision API"
            )
        model_name = parsed["model"]
        weights_spec = parsed.get("weights", "default")
        weights = _resolve_torchvision_weights(model_name, weights_spec)
        try:
            backbone = tv_get_model(model_name, weights=weights)
        except Exception as exc:  # pragma: no cover - defensive.
            raise RuntimeError(
                f"Unable to instantiate torchvision model '{model_name}': {exc}"
            ) from exc
        _apply_freeze(backbone, freeze)
        return backbone

    if source == "hf":
        if hf_hub_download is None:
            raise ImportError(
                "huggingface_hub must be installed to download checkpoints from HuggingFace"
            )
        repo_id = parsed["repo_id"]
        filename = parsed["filename"]
        model = parsed["model"]
        checkpoint_path = hf_hub_download(repo_id, filename)
        checkpoint = torch.load(checkpoint_path, map_location=map_location or "cpu")
        backbone = _build_torchvision_model(model)
        _load_checkpoint(checkpoint, backbone, convert=_convert_dinov2_state_dict)
        _apply_freeze(backbone, freeze)
        return backbone

    if source == "file":
        checkpoint_path = Path(parsed["path"]).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=map_location or "cpu")
        backbone = _build_torchvision_model(parsed["model"])
        _load_checkpoint(checkpoint, backbone, convert=_convert_dinov2_state_dict)
        _apply_freeze(backbone, freeze)
        return backbone

    raise RuntimeError(f"Unsupported backbone source: {source}")


def _resolve_torchvision_weights(model_name: str, spec: str | None) -> Optional[object]:
    if tv_get_model_weights is None:
        return None
    if spec is None or spec.lower() in {"", "none", "random"}:
        return None
    try:
        weights_enum = tv_get_model_weights(model_name)
    except Exception as exc:  # pragma: no cover - defensive.
        raise RuntimeError(
            f"Unable to retrieve torchvision weights for '{model_name}': {exc}"
        ) from exc

    if spec.lower() in {"default", "pretrained"}:
        return getattr(weights_enum, "DEFAULT", None)

    candidate = spec.upper()
    if not candidate.endswith("_V1") and candidate not in weights_enum.__dict__:
        alt = f"{candidate}_V1"
        if hasattr(weights_enum, alt):
            candidate = alt

    if hasattr(weights_enum, candidate):
        return getattr(weights_enum, candidate)

    raise ValueError(f"Unknown weight specification '{spec}' for model '{model_name}'")


def _build_torchvision_model(model_name: str) -> nn.Module:
    if tv_get_model is None:
        raise ImportError(
            "torchvision>=0.16 is required to instantiate DINOv2 models from checkpoints"
        )
    try:
        return tv_get_model(model_name, weights=None)
    except Exception as exc:  # pragma: no cover - defensive.
        raise RuntimeError(
            f"Unable to instantiate torchvision model '{model_name}' without weights: {exc}"
        ) from exc


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
