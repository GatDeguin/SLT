"""Utilities to build optimizers and learning-rate schedulers."""
from __future__ import annotations

import importlib
from typing import Any, Dict, Iterable, Mapping, Optional, Tuple, Type

import torch
from torch.optim import Optimizer
from torch.optim import lr_scheduler as schedulers

OptimizerConfig = Mapping[str, Any]
SchedulerConfig = Mapping[str, Any]

_OPTIMIZER_ALIASES: Dict[str, str] = {
    "adamw": "AdamW",
    "adam": "Adam",
    "sgd": "SGD",
    "adamax": "Adamax",
    "adadelta": "Adadelta",
    "adagrad": "Adagrad",
    "rmsprop": "RMSprop",
}

_SCHEDULER_ALIASES: Dict[str, str] = {
    "steplr": "StepLR",
    "multistep": "MultiStepLR",
    "multisteplr": "MultiStepLR",
    "cosine": "CosineAnnealingLR",
    "cosineannealing": "CosineAnnealingLR",
    "cosineannealinglr": "CosineAnnealingLR",
    "onecycle": "OneCycleLR",
    "onecyclelr": "OneCycleLR",
}


def _import_object(path: str) -> Any:
    module_name, _, attribute = path.rpartition(".")
    if not module_name:
        raise ValueError(f"Fully qualified path required, got '{path}'")
    module = importlib.import_module(module_name)
    try:
        return getattr(module, attribute)
    except AttributeError as exc:  # pragma: no cover - defensive
        raise AttributeError(f"Module '{module_name}' has no attribute '{attribute}'") from exc


def _resolve_class(
    name: str,
    base_module: Any,
    aliases: Mapping[str, str],
) -> Type[Any]:
    candidate = name
    if "." in name:
        return _import_object(name)

    lookup = name.lower()
    if lookup in aliases:
        candidate = aliases[lookup]

    if hasattr(base_module, candidate):
        return getattr(base_module, candidate)

    for attr in dir(base_module):
        if attr.lower() == lookup:
            return getattr(base_module, attr)

    raise ValueError(f"Unknown class '{name}' in module '{base_module.__name__}'")


def _normalize_config(config: Optional[Mapping[str, Any]]) -> Mapping[str, Any]:
    if config is None:
        return {}
    if not isinstance(config, Mapping):
        raise TypeError("Configuration must be a mapping")
    return config


def _build_optimizer(
    params: Any,
    config: Mapping[str, Any],
) -> Optimizer:
    config = dict(config)
    opt_name = config.pop("target", config.pop("type", "adamw"))
    if not isinstance(opt_name, str):
        raise TypeError("Optimizer 'type' must be a string")

    optimizer_cls = _resolve_class(opt_name, torch.optim, _OPTIMIZER_ALIASES)

    param_groups = config.pop("param_groups", None)
    kwargs = config

    if param_groups is not None:
        if not isinstance(param_groups, Iterable):
            raise TypeError("'param_groups' must be an iterable of dicts")
        groups = []
        for group in param_groups:
            if not isinstance(group, Mapping):
                raise TypeError("Each param_group must be a mapping")
            group_dict = dict(group)
            if "params" not in group_dict:
                raise KeyError("Each param_group must define 'params'")
            groups.append(group_dict)
        return optimizer_cls(groups, **kwargs)

    return optimizer_cls(params, **kwargs)


def _build_scheduler(
    optimizer: Optimizer,
    config: Optional[Mapping[str, Any]],
):
    if not config:
        return None

    cfg = dict(config)
    sched_name = cfg.pop("target", cfg.pop("type", None))
    if sched_name is None:
        return None
    if not isinstance(sched_name, str):
        raise TypeError("Scheduler 'type' must be a string")

    scheduler_cls = _resolve_class(sched_name, schedulers, _SCHEDULER_ALIASES)
    kwargs = cfg
    return scheduler_cls(optimizer, **kwargs)


def build_optimizer_and_scheduler(
    params: Any,
    config: Optional[Mapping[str, Any]] = None,
) -> Tuple[Optimizer, Optional[Any]]:
    """Create optimizer and scheduler from a declarative configuration.

    Parameters
    ----------
    params:
        Iterable of parameters or parameter groups for the optimizer.
    config:
        Mapping with ``optimizer`` and optional ``scheduler`` sections. The
        structure is designed to be JSON/YAML friendly and supports either
        built-in optimizer names (``"adamw"``) or fully-qualified targets such
        as ``"torch.optim.AdamW"``.
    """

    config = _normalize_config(config)
    if "optimizer" in config:
        optimizer_cfg = config["optimizer"]
    else:
        optimizer_cfg = config

    optimizer = _build_optimizer(params, _normalize_config(optimizer_cfg))

    scheduler_cfg = config.get("scheduler") if isinstance(config, Mapping) else None
    scheduler = _build_scheduler(optimizer, _normalize_config(scheduler_cfg))

    return optimizer, scheduler


def create_optimizer(
    params: Any,
    config: Optional[OptimizerConfig] = None,
) -> Optimizer:
    """Instantiate an optimizer based on a configuration dictionary."""

    optimizer, _ = build_optimizer_and_scheduler(params, _normalize_config(config))
    return optimizer


def create_scheduler(
    optimizer: Optimizer,
    config: Optional[SchedulerConfig] = None,
):
    """Create a scheduler from configuration."""

    return _build_scheduler(optimizer, _normalize_config(config))
