"""Backbone neural network definitions used by the SLT models."""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional, Tuple, Union

import torch
import torch.hub
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


BackboneFreeze = Union[bool, int, str, Iterable[str], Mapping[str, Any]]
BackboneSpec = Union[
    str,
    Mapping[str, Any],
    nn.Module,
    Callable[[], nn.Module],
]


DINOV2_DEFAULT_REPO = "facebookresearch/dinov2"
DINOV2_DEFAULT_MODEL = "dinov2_vits14"
DINOV2_IMAGE_MEAN = (0.485, 0.456, 0.406)
DINOV2_IMAGE_STD = (0.229, 0.224, 0.225)


@dataclass
class ViTConfig:
    """Configuration container for DINOv2-style ViT backbones."""

    model_name: str = DINOV2_DEFAULT_MODEL
    repo: str = DINOV2_DEFAULT_REPO
    pretrained: bool = True
    image_size: int = 224
    patch_size: int = 14
    in_channels: int = 3
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    attention_dropout: float = 0.0
    stochastic_dropout: float = 0.0
    mean: Tuple[float, float, float] = DINOV2_IMAGE_MEAN
    std: Tuple[float, float, float] = DINOV2_IMAGE_STD

    def to_spec(self) -> str:
        """Return a :func:`load_dinov2_backbone` specification string."""

        mode = "pretrained" if self.pretrained else "random"
        return f"torchhub::{self.repo}:{self.model_name}:{mode}"


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

    if isinstance(freeze, Mapping):
        if not freeze:
            return
        if any(str(key).lower() in {"all", "full", "backbone"} for key in freeze):
            flag = freeze.get("all") or freeze.get("full") or freeze.get("backbone")
            if flag:
                _apply_freeze(module, True)
        blocks = freeze.get("blocks")
        if blocks is not None:
            _freeze_first_blocks(module, int(blocks))
        if freeze.get("patch_embed"):
            _freeze_patch_embed(module)
        if freeze.get("head"):
            _freeze_head(module)

        modules = freeze.get("modules") or freeze.get("layers")
        if modules:
            for name in modules:
                _freeze_named_module(module, name)

        prefixes = freeze.get("prefixes") or freeze.get("parameters")
        if prefixes:
            _apply_freeze(module, tuple(prefixes))

        names = freeze.get("names")
        if names:
            named_parameters = dict(module.named_parameters())
            for name in names:
                parameter = named_parameters.get(name)
                if parameter is None:
                    warnings.warn(
                        f"Freeze configuration referenced unknown parameter '{name}'",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                parameter.requires_grad_(False)

        excludes = freeze.get("exclude") or freeze.get("unfreeze")
        if excludes:
            exclude_prefixes = tuple(excludes)
            for name, parameter in module.named_parameters():
                if name.startswith(exclude_prefixes):
                    parameter.requires_grad_(True)
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


def _freeze_named_module(module: nn.Module, name: str) -> None:
    target = module
    for part in name.split("."):
        if not part:
            continue
        target = getattr(target, part, None)
        if target is None:
            warnings.warn(
                f"Freeze configuration referenced unknown module '{name}'",
                RuntimeWarning,
                stacklevel=2,
            )
            return
    if not isinstance(target, nn.Module):
        warnings.warn(
            f"Freeze configuration expected '{name}' to be a module; received {type(target)!r}",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    for parameter in target.parameters():
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


def _normalize_model_name(model_name: str) -> str:
    alias_map = {
        "slt_vitsmall_patch16": DINOV2_DEFAULT_MODEL,
        "dinov2-s/16": DINOV2_DEFAULT_MODEL,
        "dinov2_s16": DINOV2_DEFAULT_MODEL,
        "dinov2-s16": DINOV2_DEFAULT_MODEL,
    }
    return alias_map.get(model_name.lower(), model_name)


def _parse_backbone_spec(spec: str) -> tuple[str, dict[str, str]]:
    """Parse a ``load_dinov2_backbone`` specification string."""

    if "::" in spec:
        source, remainder = spec.split("::", 1)
    else:
        source, remainder = "torchhub", spec
    source = source.lower()

    if source in {"torchvision", "tv"}:
        parts = [part for part in remainder.split(":") if part]
        if not parts:
            raise ValueError("Torchvision specification must include a model name")
        model = _normalize_model_name(parts[0])
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

    if source in {"torchhub"}:
        parts = [part for part in remainder.split(":") if part]
        if not parts:
            raise ValueError("TorchHub specification must include a model name")
        if len(parts) == 1:
            repo = DINOV2_DEFAULT_REPO
            model = _normalize_model_name(parts[0])
            pretrained = "pretrained"
        elif len(parts) == 2:
            if parts[1].lower() in {"pretrained", "true", "false", "random", "0", "1", "yes", "no"}:
                repo = DINOV2_DEFAULT_REPO
                model = _normalize_model_name(parts[0])
                pretrained = parts[1]
            else:
                repo = parts[0]
                model = _normalize_model_name(parts[1])
                pretrained = "pretrained"
        elif len(parts) == 3:
            repo, model, pretrained = parts
            repo = repo or DINOV2_DEFAULT_REPO
            model = _normalize_model_name(model)
        else:
            raise ValueError(
                "TorchHub specification must follow 'model', 'model:mode' or 'repo:model:mode' format"
            )
        return "torchhub", {"repo": repo, "model": model, "pretrained": pretrained}

    if source in {"hf", "huggingface"}:
        parts = [part for part in remainder.split(":") if part]
        if not parts:
            raise ValueError(
                "HuggingFace specification must provide at least the repository id"
            )
        if len(parts) == 1:
            repo_id = parts[0]
            filename = "pytorch_model.bin"
            model = DINOV2_DEFAULT_MODEL
        elif len(parts) == 2:
            repo_id, filename = parts
            model = DINOV2_DEFAULT_MODEL
        elif len(parts) == 3:
            repo_id, filename, model = parts
            model = _normalize_model_name(model)
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
            model = DINOV2_DEFAULT_MODEL
        return "file", {"path": path, "model": _normalize_model_name(model)}

    raise ValueError(
        "Backbone specification must start with 'torchhub::', 'hf::' or 'file::'."
    )


def _resolve_map_location(map_location: Optional[Union[str, torch.device]]) -> Optional[torch.device]:
    if map_location is None:
        return None
    if isinstance(map_location, torch.device):
        return map_location
    return torch.device(map_location)


def _interpret_pretrained_flag(value: Optional[Union[str, bool, int]]) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return bool(value)
    if value is None:
        return True
    text = str(value).strip().lower()
    if text in {"", "pretrained", "true", "yes", "1"}:
        return True
    if text in {"false", "no", "0", "random", "none"}:
        return False
    raise ValueError(f"Unsupported pretrained flag: {value!r}")


def _attach_dinov2_normalization(
    backbone: nn.Module,
    *,
    mean: Iterable[float] = DINOV2_IMAGE_MEAN,
    std: Iterable[float] = DINOV2_IMAGE_STD,
) -> None:
    mean_tensor = torch.tensor(tuple(float(m) for m in mean), dtype=torch.float32).view(1, -1, 1, 1)
    std_tensor = torch.tensor(tuple(float(s) for s in std), dtype=torch.float32).view(1, -1, 1, 1)
    # Use unique buffer names to avoid interfering with the underlying model state dict.
    backbone.register_buffer("_pixel_mean", mean_tensor, persistent=False)
    backbone.register_buffer("_pixel_std", std_tensor, persistent=False)
    backbone.pixel_mean = mean_tensor  # type: ignore[assignment]
    backbone.pixel_std = std_tensor  # type: ignore[assignment]
    backbone.image_normalization = {
        "mean": tuple(float(m) for m in mean_tensor.flatten().tolist()),
        "std": tuple(float(s) for s in std_tensor.flatten().tolist()),
    }


def _instantiate_dinov2_architecture(
    model_name: str,
    *,
    map_location: Optional[Union[str, torch.device]] = None,
    trust_repo: bool = True,
) -> nn.Module:
    device = _resolve_map_location(map_location)
    if tv_get_model is not None:
        try:
            model = tv_get_model(model_name, weights=None)
        except Exception:
            model = None
        else:
            if device is not None:
                model = model.to(device)
            return model

    try:
        model = torch.hub.load(
            DINOV2_DEFAULT_REPO,
            model_name,
            pretrained=False,
            trust_repo=trust_repo,
        )
    except Exception as exc:  # pragma: no cover - defensive.
        raise RuntimeError(
            f"Unable to instantiate DINOv2 architecture '{model_name}': {exc}"
        ) from exc

    if device is not None:
        model = model.to(device)
    return model


def _load_dinov2_from_hub(
    repo: str,
    model_name: str,
    *,
    pretrained: bool,
    map_location: Optional[Union[str, torch.device]],
    trust_repo: bool,
) -> nn.Module:
    try:
        backbone = torch.hub.load(
            repo,
            model_name,
            pretrained=pretrained,
            trust_repo=trust_repo,
        )
    except Exception as exc:  # pragma: no cover - defensive.
        raise RuntimeError(
            f"Unable to load TorchHub model '{repo}:{model_name}': {exc}"
        ) from exc

    device = _resolve_map_location(map_location)
    if device is not None:
        backbone = backbone.to(device)
    return backbone


def load_dinov2_backbone(
    spec: str,
    *,
    freeze: BackboneFreeze = False,
    map_location: Optional[Union[str, torch.device]] = None,
    trust_repo: bool = True,
) -> nn.Module:
    """Load a DINOv2 backbone from Torch Hub, HuggingFace or local checkpoints."""

    source, parsed = _parse_backbone_spec(spec)
    device = _resolve_map_location(map_location)

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
        if device is not None:
            backbone = backbone.to(device)
        _attach_dinov2_normalization(backbone)
        _apply_freeze(backbone, freeze)
        return backbone

    if source == "torchhub":
        repo = parsed.get("repo", DINOV2_DEFAULT_REPO)
        model_name = parsed["model"]
        pretrained = _interpret_pretrained_flag(parsed.get("pretrained"))
        backbone = _load_dinov2_from_hub(
            repo,
            model_name,
            pretrained=pretrained,
            map_location=device,
            trust_repo=trust_repo,
        )
        _attach_dinov2_normalization(backbone)
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
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
        backbone = _instantiate_dinov2_architecture(
            model,
            map_location=device,
            trust_repo=trust_repo,
        )
        _load_checkpoint(checkpoint, backbone, convert=_convert_dinov2_state_dict)
        _attach_dinov2_normalization(backbone)
        _apply_freeze(backbone, freeze)
        return backbone

    if source == "file":
        checkpoint_path = Path(parsed["path"]).expanduser().resolve()
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device or "cpu")
        backbone = _instantiate_dinov2_architecture(
            parsed["model"],
            map_location=device,
            trust_repo=trust_repo,
        )
        _load_checkpoint(checkpoint, backbone, convert=_convert_dinov2_state_dict)
        _attach_dinov2_normalization(backbone)
        _apply_freeze(backbone, freeze)
        return backbone

    raise RuntimeError(f"Unsupported backbone source: {source}")


def load_backbone(
    spec: BackboneSpec,
    *,
    freeze: BackboneFreeze = False,
    map_location: Optional[Union[str, torch.device]] = None,
    trust_repo: bool = True,
) -> nn.Module:
    """Load a backbone from flexible specification formats.

    Parameters
    ----------
    spec:
        A string, mapping, callable or module describing the backbone. Strings
        follow the same syntax as :func:`load_dinov2_backbone`. Mappings can
        contain a ``type``/``source`` key with values ``torchvision``, ``hf`` or
        ``file`` as well as the corresponding parameters. If ``freeze`` is
        provided in the mapping it overrides the explicit ``freeze`` argument.
    freeze:
        Freeze configuration applied after instantiation.
    map_location:
        Optional location used when loading checkpoints.
    trust_repo:
        Whether to trust third-party repositories when using ``torch.hub``.
    """

    explicit_freeze = freeze

    if isinstance(spec, nn.Module):
        backbone = spec
    elif callable(spec):
        candidate = spec()
        if not isinstance(candidate, nn.Module):
            raise TypeError(
                "Backbone factory callable must return an nn.Module instance"
            )
        backbone = candidate
    elif isinstance(spec, Mapping):
        spec_mapping = dict(spec)
        mapping_freeze = spec_mapping.pop("freeze", explicit_freeze)
        source = spec_mapping.get("type") or spec_mapping.get("source") or "dinov2"
        source = str(source).lower()

        if source in {"torchvision", "torchhub", "tv"}:
            model = spec_mapping.get("model")
            if model is None:
                raise ValueError("Torchvision backbone mappings must specify 'model'")
            weights = spec_mapping.get("weights")
            tv_spec = f"{model}:{weights}" if weights not in (None, "") else str(model)
            backbone = load_dinov2_backbone(
                f"torchvision::{tv_spec}",
                freeze=False,
                map_location=map_location,
                trust_repo=trust_repo,
            )
        elif source in {"hf", "huggingface"}:
            repo_id = spec_mapping.get("repo_id") or spec_mapping.get("repository")
            if repo_id is None:
                raise ValueError("HuggingFace backbone mappings must include 'repo_id'")
            filename = spec_mapping.get("filename") or "pytorch_model.bin"
            model = (
                spec_mapping.get("model")
                or spec_mapping.get("architecture")
                or "dinov2_vits14"
            )
            backbone = load_dinov2_backbone(
                f"hf::{repo_id}:{filename}:{model}",
                freeze=False,
                map_location=map_location,
                trust_repo=trust_repo,
            )
        elif source in {"file", "local", "path"}:
            path = spec_mapping.get("path") or spec_mapping.get("checkpoint")
            if path is None:
                raise ValueError("File backbone mappings must include 'path'")
            model = (
                spec_mapping.get("model")
                or spec_mapping.get("architecture")
                or "dinov2_vits14"
            )
            backbone = load_dinov2_backbone(
                f"file::{path}:{model}",
                freeze=False,
                map_location=map_location,
                trust_repo=trust_repo,
            )
        elif source in {"dinov2", "default"}:
            dinov2_spec = spec_mapping.get("spec") or spec_mapping.get("identifier")
            if dinov2_spec is None:
                raise ValueError("DINOv2 mappings must provide 'spec'")
            backbone = load_dinov2_backbone(
                str(dinov2_spec),
                freeze=False,
                map_location=map_location,
                trust_repo=trust_repo,
            )
        else:
            raise ValueError(f"Unsupported backbone source '{source}'")
        explicit_freeze = mapping_freeze
    elif isinstance(spec, str):
        backbone = load_dinov2_backbone(
            spec,
            freeze=False,
            map_location=map_location,
            trust_repo=trust_repo,
        )
    else:
        raise TypeError(
            "Backbone specification must be a string, mapping, module or callable"
        )

    if explicit_freeze:
        _apply_freeze(backbone, explicit_freeze)

    return backbone


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


