import math
import sys
import time
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slt.training.loops import LoopResult, clip_gradients, eval_epoch, train_epoch
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

    initial_loss = eval_epoch(model, loader, loss_fn, device=device).loss
    for _ in range(10):
        train_epoch(model, loader, optimizer, loss_fn, device=device, scaler=scaler)
    final_loss = eval_epoch(model, loader, loss_fn, device=device).loss

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


def test_conditional_clip_callable(monkeypatch):
    device = torch.device("cpu")
    dataset = LinearDataset(n_samples=48)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = SimpleRegressor(dataset.x.size(1)).to(device)
    optimizer = create_optimizer(model.parameters(), {"lr": 0.01, "type": "sgd"})
    loss_fn = nn.MSELoss()

    recorded = []

    def _fake_clip_gradients(optimizer, max_norm, *, scaler=None, parameters=None, norm_type=2.0):
        recorded.append(max_norm)
        if max_norm is not None:
            return torch.tensor(max_norm)
        return None

    monkeypatch.setattr("slt.training.loops.clip_gradients", _fake_clip_gradients)

    def _clip_schedule(step: int, loss: torch.Tensor, opt: torch.optim.Optimizer):
        if step == 1:
            return None
        return 0.25 * step

    train_epoch(
        model,
        loader,
        optimizer,
        loss_fn,
        device=device,
        grad_clip_norm=_clip_schedule,
    )

    # The callable returns None for the first step (skip clipping) and positive afterwards.
    assert recorded[0] is None
    assert any(value and value > 0 for value in recorded[1:])


def test_clip_gradients_helper():
    model = SimpleRegressor(4)
    optimizer = create_optimizer(model.parameters(), {"lr": 0.01, "type": "sgd"})
    for p in model.parameters():
        p.grad = torch.ones_like(p)

    norm = clip_gradients(optimizer, 0.1, parameters=model.parameters())
    assert norm is not None


def test_gradient_accumulation_matches_large_batch():
    device = torch.device("cpu")
    torch.manual_seed(0)
    dataset = LinearDataset(n_samples=64)
    loader_large = DataLoader(dataset, batch_size=8, shuffle=False)
    loader_micro = DataLoader(dataset, batch_size=4, shuffle=False)

    model_large = SimpleRegressor(dataset.x.size(1)).to(device)
    model_accum = SimpleRegressor(dataset.x.size(1)).to(device)
    model_accum.load_state_dict(model_large.state_dict())

    optimizer_large = create_optimizer(model_large.parameters(), {"lr": 0.05, "type": "adamw"})
    optimizer_accum = create_optimizer(model_accum.parameters(), {"lr": 0.05, "type": "adamw"})
    loss_fn = nn.MSELoss()

    train_epoch(model_large, loader_large, optimizer_large, loss_fn, device=device)
    train_epoch(
        model_accum,
        loader_micro,
        optimizer_accum,
        loss_fn,
        device=device,
        grad_accum_steps=2,
    )

    for param_large, param_accum in zip(model_large.parameters(), model_accum.parameters()):
        assert torch.allclose(param_large, param_accum, atol=1e-6)


def test_dynamic_accumulation_strategy(monkeypatch):
    device = torch.device("cpu")
    torch.manual_seed(0)
    dataset = LinearDataset(n_samples=96)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = SimpleRegressor(dataset.x.size(1)).to(device)
    optimizer = create_optimizer(model.parameters(), {"lr": 0.05, "type": "adamw"})
    loss_fn = nn.MSELoss()

    step_calls = 0
    original_step = optimizer.step

    def counting_step(*args, **kwargs):
        nonlocal step_calls
        step_calls += 1
        return original_step(*args, **kwargs)

    monkeypatch.setattr(optimizer, "step", counting_step)

    def strategy(step: int, loss: torch.Tensor, batch: Any, pending: int):
        # Use a different accumulation length depending on the step index.
        if step == 1:
            return 2  # accumulate two micro-batches
        if step == 3:
            return (False, 3)  # start a window of three
        if step == 5:
            return True  # force an early step regardless of pending count
        return False

    result = train_epoch(
        model,
        loader,
        optimizer,
        loss_fn,
        device=device,
        grad_accum_steps=strategy,
    )

    assert isinstance(result, LoopResult)
    # We expect at least three optimizer steps: one for the initial pair, one for the forced
    # step, and a final flush.
    assert step_calls >= 3


def test_amp_overflow_recovery():
    device = torch.device("cpu")
    torch.manual_seed(0)
    dataset = LinearDataset(n_samples=32)
    loader = DataLoader(dataset, batch_size=8, shuffle=False)
    model = SimpleRegressor(dataset.x.size(1)).to(device)
    optimizer = create_optimizer(model.parameters(), {"lr": 0.05, "type": "sgd"})
    loss_fn = nn.MSELoss()

    class DummyScaler:
        def __init__(self, fail_on: int = 1):
            self.fail_on = fail_on
            self.calls = 0
            self.stepped = 0

        def is_enabled(self):
            return True

        def scale(self, loss):
            return loss

        def unscale_(self, optimizer):
            pass

        def step(self, optimizer):
            self.calls += 1
            if self.calls == self.fail_on:
                raise RuntimeError("inf or nan encountered in gradients")
            optimizer.step()
            self.stepped += 1

        def update(self):
            pass

    scaler = DummyScaler()

    weights_before = [param.detach().clone() for param in model.parameters()]

    train_epoch(
        model,
        loader,
        optimizer,
        loss_fn,
        device=device,
        scaler=scaler,  # type: ignore[arg-type]
        autocast_dtype=torch.float32,
    )

    weights_after = list(model.parameters())

    assert scaler.calls >= 1
    assert scaler.stepped >= 1
    assert any(not torch.allclose(before, after) for before, after in zip(weights_before, weights_after))


def test_train_epoch_throughput():
    device = torch.device("cpu")
    dataset = LinearDataset(n_samples=64)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = SimpleRegressor(dataset.x.size(1)).to(device)
    optimizer = create_optimizer(model.parameters(), {"lr": 0.05, "type": "adamw"})
    loss_fn = nn.MSELoss()

    start = time.perf_counter()
    result = train_epoch(model, loader, optimizer, loss_fn, device=device)
    elapsed = time.perf_counter() - start

    throughput = len(dataset) / max(elapsed, 1e-6)

    assert throughput > 0
    assert math.isfinite(result.loss)


def test_metric_aggregation():
    device = torch.device("cpu")
    dataset = LinearDataset(n_samples=32)
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    model = SimpleRegressor(dataset.x.size(1)).to(device)
    optimizer = create_optimizer(model.parameters(), {"lr": 0.1, "type": "adamw"})
    loss_fn = nn.MSELoss()

    def mean_target(_, targets):
        return targets.mean()

    result = train_epoch(
        model,
        loader,
        optimizer,
        loss_fn,
        device=device,
        metrics={"mean_target": mean_target},
    )

    assert isinstance(result, LoopResult)
    assert "mean_target" in result.metrics
    assert math.isfinite(result.metrics["mean_target"])

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
    val_loss = eval_epoch(model, make_loader(), loss_fn, device=device).loss
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
    final_loss_expected = eval_epoch(model, make_loader(), loss_fn, device=device).loss

    model_resume = SimpleRegressor(dataset.x.size(1)).to(device)
    optimizer_resume = create_optimizer(model_resume.parameters(), {"lr": 0.05, "type": "adamw"})
    scaler_resume = torch.cuda.amp.GradScaler(enabled=False)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_resume.load_state_dict(checkpoint["model_state"])
    optimizer_resume.load_state_dict(checkpoint["optimizer_state"])
    scaler_resume.load_state_dict(checkpoint["scaler_state"])
    assert checkpoint["epoch"] == pre_epochs

    for _ in range(post_epochs):
        train_epoch(
            model_resume,
            make_loader(),
            optimizer_resume,
            loss_fn,
            device=device,
            scaler=scaler_resume,
        )

    final_loss_resumed = eval_epoch(model_resume, make_loader(), loss_fn, device=device).loss
    assert final_loss_resumed == pytest.approx(final_loss_expected, rel=1e-6, abs=1e-6)
