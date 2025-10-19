import json
import random
import sys
from pathlib import Path

import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:  # pragma: no cover - numpy optional
    import numpy as np
except Exception:  # pragma: no cover
    np = None  # type: ignore

from slt.training.loops import eval_epoch, train_epoch
from slt.training.optim import create_optimizer
from tools.train_slt_multistream_v9 import (
    DataConfig,
    ModelConfig,
    OptimConfig,
    TrainingConfig,
    _load_rng_state,
    _save_checkpoint,
    _save_rng_state,
    _serialise_config,
)


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


def _reset_seeds(seed: int = 0) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if np is not None:
        np.random.seed(seed)


def test_rng_state_resume(tmp_path):
    device = torch.device("cpu")
    dataset = LinearDataset(n_samples=128)

    def make_loader():
        return DataLoader(dataset, batch_size=32, shuffle=False)

    _reset_seeds(0)
    model = SimpleRegressor(dataset.x.size(1)).to(device)
    optimizer = create_optimizer(model.parameters(), {"lr": 0.05, "type": "adamw"})
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    loss_fn = nn.MSELoss()

    pre_epochs = 2
    post_epochs = 3

    for _ in range(pre_epochs):
        train_epoch(model, make_loader(), optimizer, loss_fn, device=device, scaler=scaler)

    val_loss = eval_epoch(model, make_loader(), loss_fn, device=device).loss
    checkpoint_path = tmp_path / "last.pt"
    _save_checkpoint(
        checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=pre_epochs,
        val_loss=val_loss,
        scaler=scaler,
        best_val=val_loss,
        config={"data": {"batch_size": 32}},
    )
    _save_rng_state(tmp_path)

    for _ in range(post_epochs):
        train_epoch(model, make_loader(), optimizer, loss_fn, device=device, scaler=scaler)
    final_expected = eval_epoch(model, make_loader(), loss_fn, device=device).loss

    _reset_seeds(0)
    model_resumed = SimpleRegressor(dataset.x.size(1)).to(device)
    optimizer_resumed = create_optimizer(model_resumed.parameters(), {"lr": 0.05, "type": "adamw"})
    scaler_resumed = torch.cuda.amp.GradScaler(enabled=False)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_resumed.load_state_dict(checkpoint["model_state"])
    optimizer_resumed.load_state_dict(checkpoint["optimizer_state"])
    scaler_resumed.load_state_dict(checkpoint["scaler_state"])
    _load_rng_state(tmp_path)

    for _ in range(post_epochs):
        train_epoch(
            model_resumed,
            make_loader(),
            optimizer_resumed,
            loss_fn,
            device=device,
            scaler=scaler_resumed,
        )
    final_resumed = eval_epoch(model_resumed, make_loader(), loss_fn, device=device).loss

    assert final_resumed == pytest.approx(final_expected, rel=1e-6, abs=1e-6)


def test_serialise_config_outputs(tmp_path):
    (tmp_path / "meta.csv").write_text("", encoding="utf-8")
    (tmp_path / "train.csv").write_text("", encoding="utf-8")
    (tmp_path / "val.csv").write_text("", encoding="utf-8")

    data_config = DataConfig(
        face_dir=tmp_path,
        hand_left_dir=tmp_path,
        hand_right_dir=tmp_path,
        pose_dir=tmp_path,
        metadata_csv=tmp_path / "meta.csv",
        train_index=tmp_path / "train.csv",
        val_index=tmp_path / "val.csv",
        work_dir=tmp_path,
    )
    model_config = ModelConfig()
    optim_config = OptimConfig()
    training_config = TrainingConfig()
    merged = {
        "data": {"batch_size": data_config.batch_size},
        "model": {"d_model": model_config.d_model},
        "optim": {"lr": optim_config.lr},
        "training": {"epochs": training_config.epochs},
    }

    _serialise_config(tmp_path, data_config, model_config, optim_config, training_config, merged)

    config_path = tmp_path / "config.json"
    merged_path = tmp_path / "config.merged.json"
    assert config_path.exists()
    assert merged_path.exists()

    resolved = json.loads(config_path.read_text(encoding="utf-8"))
    assert resolved["training"]["epochs"] == training_config.epochs
    merged_data = json.loads(merged_path.read_text(encoding="utf-8"))
    assert merged_data["data"]["batch_size"] == data_config.batch_size
