"""Smoke tests for the high-level training loop helpers."""

from __future__ import annotations

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from slt.training.loops import LoopResult, eval_epoch, train_epoch


class TinyRegressionDataset(Dataset):
    def __init__(self, n_samples: int = 64, n_features: int = 3) -> None:
        generator = torch.Generator().manual_seed(2024)
        self.features = torch.randn(n_samples, n_features, generator=generator)
        weights = torch.arange(1, n_features + 1, dtype=torch.float32)
        noise = 0.05 * torch.randn(n_samples, generator=generator)
        self.targets = self.features.matmul(weights) + noise

    def __len__(self) -> int:  # pragma: no cover - trivial accessor
        return self.features.size(0)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return {
            "inputs": self.features[index],
            "targets": self.targets[index].unsqueeze(0),
        }


def _make_model(input_dim: int) -> nn.Module:
    return nn.Sequential(
        nn.Linear(input_dim, 16),
        nn.ReLU(),
        nn.Linear(16, 1),
    )


def test_train_epoch_returns_loop_result() -> None:
    dataset = TinyRegressionDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = _make_model(dataset.features.size(1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    initial = eval_epoch(model, loader, loss_fn, device="cpu")
    assert isinstance(initial, LoopResult)

    last_result: LoopResult = initial
    for _ in range(3):
        last_result = train_epoch(model, loader, optimizer, loss_fn, device="cpu")
        assert isinstance(last_result, LoopResult)

    final = eval_epoch(model, loader, loss_fn, device="cpu")
    assert isinstance(final, LoopResult)

    assert last_result.loss == pytest.approx(last_result.loss)
    assert final.loss < initial.loss
