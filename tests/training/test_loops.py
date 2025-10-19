import math
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slt.training.loops import eval_epoch, train_epoch
from slt.training.optim import create_optimizer


class LinearDataset(Dataset):
    def __init__(self, n_samples: int = 256, n_features: int = 4, noise: float = 0.05):
        generator = torch.Generator().manual_seed(0)
        self.x = torch.randn(n_samples, n_features, generator=generator)
        weights = torch.arange(1, n_features + 1, dtype=torch.float32)
        self.y = self.x.matmul(weights)[:, None]
        noise_tensor = torch.randn(self.y.shape, generator=generator)
        self.y += noise * noise_tensor

    def __len__(self):
        return self.x.size(0)

    def __getitem__(self, idx):
        return {"inputs": self.x[idx], "targets": self.y[idx]}


class SimpleRegressor(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.net(x)


def _run_training(device: torch.device):
    torch.manual_seed(0)
    dataset = LinearDataset()
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    model = SimpleRegressor(dataset.x.size(1)).to(device)
    optimizer = create_optimizer(model.parameters(), {"lr": 0.05, "type": "adamw"})
    loss_fn = nn.MSELoss()

    scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

    initial_loss = eval_epoch(model, loader, loss_fn, device=device)
    for _ in range(10):
        train_epoch(model, loader, optimizer, loss_fn, device=device, scaler=scaler)
    final_loss = eval_epoch(model, loader, loss_fn, device=device)

    return initial_loss, final_loss


def test_loss_decreases_cpu():
    device = torch.device("cpu")
    initial, final = _run_training(device)
    assert final < initial
    assert final < 0.2


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_loss_decreases_amp():
    device = torch.device("cuda")
    initial, final = _run_training(device)
    assert math.isfinite(initial)
    assert final < initial
    assert final < 0.1
