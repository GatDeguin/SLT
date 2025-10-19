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

from slt.training.loops import clip_gradients, eval_epoch, train_epoch
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


def test_clip_gradients_invokes_norm(monkeypatch):
    device = torch.device("cpu")
    dataset = LinearDataset(n_samples=64)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    model = SimpleRegressor(dataset.x.size(1)).to(device)
    optimizer = create_optimizer(model.parameters(), {"lr": 0.01, "type": "sgd"})
    loss_fn = nn.MSELoss()

    captured = {}

    def _fake_clip_gradients(optimizer, max_norm, *, scaler=None, parameters=None, norm_type=2.0):
        captured["max_norm"] = max_norm
        assert norm_type == 2.0
        return torch.tensor(1.0)

    monkeypatch.setattr("slt.training.loops.clip_gradients", _fake_clip_gradients)

    train_epoch(
        model,
        loader,
        optimizer,
        loss_fn,
        device=device,
        grad_clip_norm=0.5,
    )

    assert captured["max_norm"] == 0.5


def test_clip_gradients_helper():
    model = SimpleRegressor(4)
    optimizer = create_optimizer(model.parameters(), {"lr": 0.01, "type": "sgd"})
    for p in model.parameters():
        p.grad = torch.ones_like(p)

    norm = clip_gradients(optimizer, 0.1, parameters=model.parameters())
    assert norm is not None


def test_resume_from_checkpoint(tmp_path):
    device = torch.device("cpu")
    torch.manual_seed(0)
    dataset = LinearDataset()

    def make_loader():
        return DataLoader(dataset, batch_size=32, shuffle=False)

    model = SimpleRegressor(dataset.x.size(1)).to(device)
    optimizer = create_optimizer(model.parameters(), {"lr": 0.05, "type": "adamw"})
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    loss_fn = nn.MSELoss()

    pre_epochs = 3
    post_epochs = 2

    for _ in range(pre_epochs):
        train_epoch(model, make_loader(), optimizer, loss_fn, device=device, scaler=scaler)

    checkpoint_path = tmp_path / "checkpoint.pt"
    val_loss = eval_epoch(model, make_loader(), loss_fn, device=device)
    torch.save(
        {
            "epoch": pre_epochs,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict(),
            "val_loss": float(val_loss),
            "best_val": float(val_loss),
        },
        checkpoint_path,
    )

    for _ in range(post_epochs):
        train_epoch(model, make_loader(), optimizer, loss_fn, device=device, scaler=scaler)
    final_loss_expected = eval_epoch(model, make_loader(), loss_fn, device=device)

    model_resume = SimpleRegressor(dataset.x.size(1)).to(device)
    optimizer_resume = create_optimizer(model_resume.parameters(), {"lr": 0.05, "type": "adamw"})
    scaler_resume = torch.cuda.amp.GradScaler(enabled=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_resume.load_state_dict(checkpoint["model_state"])
    optimizer_resume.load_state_dict(checkpoint["optimizer_state"])
    scaler_resume.load_state_dict(checkpoint["scaler_state"])
    assert checkpoint["epoch"] == pre_epochs

    for _ in range(post_epochs):
        train_epoch(model_resume, make_loader(), optimizer_resume, loss_fn, device=device, scaler=scaler_resume)

    final_loss_resumed = eval_epoch(model_resume, make_loader(), loss_fn, device=device)
    assert final_loss_resumed == pytest.approx(final_loss_expected, rel=1e-6, abs=1e-6)
