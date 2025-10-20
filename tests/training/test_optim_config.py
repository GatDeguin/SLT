import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slt.training.optim import build_optimizer_and_scheduler


def test_build_optimizer_and_scheduler_with_aliases():
    model = nn.Linear(4, 2)
    config = {
        "optimizer": {"type": "adamw", "lr": 1e-3},
        "scheduler": {"type": "StepLR", "step_size": 10, "gamma": 0.5},
    }

    optimizer, scheduler = build_optimizer_and_scheduler(model.parameters(), config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)


def test_build_optimizer_with_target_path():
    model = nn.Linear(4, 2)
    config = {
        "optimizer": {
            "target": "torch.optim.SGD",
            "lr": 0.1,
            "momentum": 0.9,
        }
    }

    optimizer, scheduler = build_optimizer_and_scheduler(model.parameters(), config)

    assert isinstance(optimizer, torch.optim.SGD)
    assert scheduler is None
