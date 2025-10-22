"""Shared configuration helpers used across SLT training entry-points."""

from __future__ import annotations

import json
from dataclasses import MISSING, dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple, Type, TypeVar, Union, get_args, get_origin

T = TypeVar("T")


@dataclass
class ModelConfig:
    """Model hyper-parameters exposed in the CLIs."""

    image_size: int = 224
    projector_dim: int = 128
    d_model: int = 128
    pose_landmarks: int = 13
    projector_dropout: float = 0.05
    fusion_dropout: float = 0.05
    leaky_relu_negative_slope: float = 0.01
    temporal_nhead: int = 4
    temporal_layers: int = 3
    temporal_dim_feedforward: int = 384
    temporal_dropout: float = 0.05
    sequence_length: int = 128
    decoder_layers: int = 2
    decoder_heads: int = 4
    decoder_dropout: float = 0.1
    decoder_model: Optional[str] = None
    decoder_config: Optional[str] = None
    decoder_class: Optional[str] = None
    decoder_kwargs: Dict[str, Any] = field(default_factory=dict)
    face_backbone: Optional[str] = None
    hand_left_backbone: Optional[str] = None
    hand_right_backbone: Optional[str] = None
    freeze_face_backbone: bool = False
    freeze_hand_left_backbone: bool = False
    freeze_hand_right_backbone: bool = False
    pretrained: Optional[str] = "single_signer"
    pretrained_checkpoint: Optional[Path] = None
    use_mska: bool = False
    mska_heads: int = 4
    mska_ff_multiplier: int = 4
    mska_dropout: float = 0.1
    mska_input_dim: int = 3
    mska_ctc_vocab: Optional[int] = None
    mska_detach_teacher: bool = True
    mska_stream_heads: int = 4
    mska_temporal_blocks: int = 2
    mska_temporal_kernel: int = 3
    mska_temporal_dilation: int = 1
    mska_use_sgr: bool = False
    mska_sgr_shared: bool = False
    mska_sgr_activation: str = "softmax"
    mska_sgr_mix: float = 0.5
    mska_translation_weight: float = 1.0
    mska_ctc_weight: float = 0.0
    mska_distillation_weight: float = 0.0
    mska_distillation_temperature: float = 1.0
    mska_gloss_hidden_dim: Optional[int] = None
    mska_gloss_activation: str = "leaky_relu"
    mska_gloss_dropout: float = 0.0
    mska_gloss_fusion: str = "add"


@dataclass
class OptimConfig:
    """Optimisation related hyper-parameters."""

    optimizer: str = "adamw"
    lr: float = 1e-3
    weight_decay: float = 0.0
    scheduler: Optional[str] = None
    scheduler_step_size: int = 5
    scheduler_gamma: float = 0.5
    label_smoothing: float = 0.1
    grad_clip_norm: Optional[float] = None


@dataclass
class TrainingConfig:
    """Configuration of the outer training loop."""

    epochs: int = 40
    grad_accum_steps: int = 1
    compile: bool = False
    compile_mode: Optional[str] = None
    init_checkpoint: Optional[Path] = None


@dataclass
class DataConfig:
    """Paths and data loading configuration."""

    face_dir: Path
    hand_left_dir: Path
    hand_right_dir: Path
    pose_dir: Path
    metadata_csv: Path
    train_index: Path
    val_index: Path
    work_dir: Path
    keypoints_dir: Optional[Path] = None
    gloss_csv: Optional[Path] = None
    num_workers: int = 0
    batch_size: int = 4
    val_batch_size: Optional[int] = None
    seed: int = 1234
    device: str = "cuda"
    precision: str = "amp"
    tensorboard: Optional[Path] = None
    pin_memory: bool = True
    tokenizer: Optional[str] = None
    max_target_length: int = 128
    mix_streams: Dict[str, float] = field(default_factory=dict)
    keypoint_normalize_center: bool = True
    keypoint_scale_range: Optional[Tuple[float, float]] = None
    keypoint_translate_range: Optional[Tuple[float, float, float, float]] = None
    keypoint_rotate_range: Optional[Tuple[float, float]] = None
    keypoint_resample_range: Optional[Tuple[float, float]] = None


def dataclass_defaults(cls: Type[T]) -> Dict[str, Any]:
    """Return defaults for ``cls`` including ``default_factory`` values."""

    defaults: Dict[str, Any] = {}
    for field_obj in fields(cls):
        if field_obj.default is not MISSING:
            defaults[field_obj.name] = field_obj.default
        elif field_obj.default_factory is not MISSING:  # type: ignore[attr-defined]
            defaults[field_obj.name] = field_obj.default_factory()  # type: ignore[misc]
    return defaults


def _coerce_override_value(value: str) -> Any:
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        lowered = value.lower()
        if lowered in {"true", "false"}:
            return lowered == "true"
        if lowered == "null":
            return None
        try:
            return int(value)
        except ValueError:
            try:
                return float(value)
            except ValueError:
                return value


def apply_string_overrides(config: MutableMapping[str, Any], overrides: Iterable[str]) -> None:
    """Apply ``KEY=VALUE`` overrides to ``config`` using dotted paths."""

    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override format '{item}'. Expected KEY=VALUE")
        key, raw_value = item.split("=", 1)
        value = _coerce_override_value(raw_value)
        parts = key.split(".")
        cursor: MutableMapping[str, Any] = config
        for part in parts[:-1]:
            nested = cursor.get(part)
            if not isinstance(nested, MutableMapping):
                nested = {}
                cursor[part] = nested
            cursor = nested
        cursor[parts[-1]] = value


def deep_update(target: Mapping[str, Any], updates: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``updates`` into ``target`` returning a new mapping."""

    result = dict(target)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(result.get(key), Mapping):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config_template(path: Path) -> Dict[str, Any]:
    """Load a configuration template from JSON or YAML format."""

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    text = path.read_text(encoding="utf-8")
    suffix = path.suffix.lower()
    if suffix in {".yml", ".yaml"}:
        try:
            import yaml
        except Exception as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML is required to parse YAML configuration files") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)

    if not isinstance(data, Mapping):
        raise ValueError("Configuration root must be a mapping")
    return dict(data)


def _coerce_field_value(field_obj, value: Any) -> Any:
    if value is None:
        return None
    annotation = field_obj.type
    origin = get_origin(annotation)
    if origin is Union:
        args = get_args(annotation)
        non_none_args = [arg for arg in args if arg is not type(None)]  # noqa: E721
        if len(non_none_args) == 1:
            annotation = non_none_args[0]
            origin = get_origin(annotation)
    if annotation is Path and not isinstance(value, Path):
        return Path(value)
    if annotation is int and isinstance(value, str):
        return int(value)
    if annotation is float and isinstance(value, str):
        return float(value)
    if annotation is bool and isinstance(value, str):
        lowered = value.lower()
        if lowered in {"true", "1", "yes"}:
            return True
        if lowered in {"false", "0", "no"}:
            return False
        raise ValueError(f"Cannot coerce '{value}' to bool for field '{field_obj.name}'")
    if origin is dict and not isinstance(value, dict):
        raise TypeError(f"Field '{field_obj.name}' expects a mapping")
    return value


def instantiate_config(cls: Type[T], values: Mapping[str, Any]) -> T:
    """Instantiate ``cls`` from ``values`` coercing recognised types."""

    payload: Dict[str, Any] = {}
    for field_obj in fields(cls):
        key = field_obj.name
        if key not in values:
            continue
        payload[key] = _coerce_field_value(field_obj, values[key])
    return cls(**payload)


def resolve_configs(
    *,
    config_path: Optional[Path] = None,
    cli_overrides: Optional[Mapping[str, Mapping[str, Any]]] = None,
    set_overrides: Iterable[str] = (),
    base: Optional[Mapping[str, Any]] = None,
) -> Tuple[DataConfig, ModelConfig, OptimConfig, TrainingConfig, Dict[str, Any]]:
    """Resolve configuration sections returning dataclass instances."""

    defaults: Dict[str, Any] = {
        "data": dataclass_defaults(DataConfig),
        "model": dataclass_defaults(ModelConfig),
        "optim": dataclass_defaults(OptimConfig),
        "training": dataclass_defaults(TrainingConfig),
    }
    if base:
        defaults = deep_update(defaults, base)

    if config_path is not None:
        loaded = load_config_template(config_path)
        for key in ("data", "model", "optim", "training"):
            if key in loaded and isinstance(loaded[key], Mapping):
                defaults[key] = deep_update(defaults.get(key, {}), loaded[key])
        for key, value in loaded.items():
            if key not in defaults:
                defaults[key] = value

    if cli_overrides:
        for section, updates in cli_overrides.items():
            if not updates:
                continue
            current = defaults.get(section, {})
            if isinstance(current, Mapping):
                defaults[section] = deep_update(current, updates)
            else:
                defaults[section] = updates

    apply_string_overrides(defaults, set_overrides)

    raw_model_section = defaults.get("model", {})

    data_config = instantiate_config(DataConfig, defaults.get("data", {}))
    model_config = instantiate_config(ModelConfig, raw_model_section)
    optim_config = instantiate_config(OptimConfig, defaults.get("optim", {}))
    training_config = instantiate_config(TrainingConfig, defaults.get("training", {}))

    if data_config.val_batch_size is None:
        data_config.val_batch_size = data_config.batch_size

    scheduler_choice = optim_config.scheduler
    if scheduler_choice is not None:
        scheduler_choice = scheduler_choice.lower()
        if scheduler_choice == "none":
            optim_config.scheduler = None
        else:
            optim_config.scheduler = scheduler_choice

    if "mska" in raw_model_section and "use_mska" not in raw_model_section:
        model_config.use_mska = bool(raw_model_section["mska"])

    if model_config.use_mska:
        if data_config.keypoints_dir is None:
            raise ValueError("MSKA requires data.keypoints_dir to be configured")
        if model_config.mska_translation_weight < 0:
            raise ValueError("mska_translation_weight must be non-negative")
        if model_config.mska_ctc_weight < 0:
            raise ValueError("mska_ctc_weight must be non-negative")
        if model_config.mska_distillation_weight < 0:
            raise ValueError("mska_distillation_weight must be non-negative")
        if model_config.mska_distillation_temperature <= 0:
            raise ValueError("mska_distillation_temperature must be > 0")
        if model_config.mska_ctc_weight > 0 and data_config.gloss_csv is None:
            raise ValueError(
                "MSKA CTC supervision enabled but data.gloss_csv is missing"
            )
        if not 0.0 <= model_config.mska_sgr_mix <= 1.0:
            raise ValueError("mska_sgr_mix must be between 0 and 1 inclusive")
        valid_sgr_activations = {"softmax", "sigmoid", "tanh", "relu", "identity", "linear", "none"}
        if model_config.mska_sgr_activation.lower() not in valid_sgr_activations:
            raise ValueError(
                "mska_sgr_activation must be one of {'softmax', 'sigmoid', 'tanh', 'relu', 'identity', 'linear', 'none'}"
            )
    else:
        model_config.mska_translation_weight = max(0.0, model_config.mska_translation_weight)
        model_config.mska_ctc_weight = 0.0
        model_config.mska_distillation_weight = 0.0

    extra_optim = defaults.get("optim", {})
    if optim_config.scheduler == "cosine":
        tmax = extra_optim.get("scheduler_tmax")
        if tmax is None:
            tmax = optim_config.scheduler_step_size
        optim_config.scheduler_step_size = int(tmax)

    return data_config, model_config, optim_config, training_config, defaults

