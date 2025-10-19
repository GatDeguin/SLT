"""Utilities to build optimizers and learning-rate schedulers."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch.optim import Optimizer


def create_optimizer(
    params: Any,
    config: Optional[Dict[str, Any]] = None,
) -> Optimizer:
    """Instantiate an optimizer based on a configuration dictionary."""
    if config is None:
        config = {}
    opt_type = config.get("type", "adamw").lower()
    lr = config.get("lr", 1e-3)
    weight_decay = config.get("weight_decay", 0.0)

    if opt_type == "adamw":
        betas = config.get("betas", (0.9, 0.999))
        eps = config.get("eps", 1e-8)
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if opt_type == "adam":
        betas = config.get("betas", (0.9, 0.999))
        eps = config.get("eps", 1e-8)
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
    if opt_type == "sgd":
        momentum = config.get("momentum", 0.0)
        nesterov = config.get("nesterov", False)
        dampening = config.get("dampening", 0.0)
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
        )
    raise ValueError(f"Unsupported optimizer type: {config['type']!r}")


def create_scheduler(
    optimizer: Optimizer,
    config: Optional[Dict[str, Any]] = None,
):
    """Create a scheduler from configuration.

    Returns ``None`` when no scheduler type is supplied.
    """
    if not config:
        return None

    sched_type = config.get("type")
    if sched_type is None:
        return None
    sched_type = sched_type.lower()

    if sched_type == "steplr":
        step_size = config.get("step_size", 1)
        gamma = config.get("gamma", 0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    if sched_type == "multistep":
        milestones = config.get("milestones")
        if milestones is None:
            raise ValueError("'milestones' must be provided for MultiStepLR")
        gamma = config.get("gamma", 0.1)
        return torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    if sched_type == "cosine":
        t_max = config.get("t_max")
        if t_max is None:
            raise ValueError("'t_max' must be provided for CosineAnnealingLR")
        eta_min = config.get("eta_min", 0.0)
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max, eta_min=eta_min)
    if sched_type == "onecycle":
        max_lr = config.get("max_lr")
        steps_per_epoch = config.get("steps_per_epoch")
        epochs = config.get("epochs")
        if None in {max_lr, steps_per_epoch, epochs}:
            raise ValueError("'max_lr', 'steps_per_epoch', and 'epochs' are required for OneCycleLR")
        pct_start = config.get("pct_start", 0.3)
        anneal_strategy = config.get("anneal_strategy", "cos")
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=max_lr,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            pct_start=pct_start,
            anneal_strategy=anneal_strategy,
        )

    raise ValueError(f"Unsupported scheduler type: {config['type']!r}")
