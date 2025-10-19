#!/usr/bin/env python3
"""Command line entry-point to train the multi-stream SLT stub model.

The script wires together the reusable components from :mod:`slt` to
instantiate the dataset, encoder/decoder pair and the optimisation loop.  It is
geared towards experimentation and intentionally keeps the dependency surface
minimal so that it can run in environments where only the Python standard
library and PyTorch are available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, Optional

import torch
from torch import nn
from torch.utils.data import DataLoader

from slt.data import LsaTMultiStream, collate_fn
from slt.models import MultiStreamEncoder, TextDecoderStub, ViTConfig
from slt.training.loops import eval_epoch, train_epoch
from slt.training.optim import create_optimizer, create_scheduler
from slt.utils.general import set_seed

try:  # pragma: no cover - TensorBoard is an optional dependency.
    from torch.utils.tensorboard import SummaryWriter  # type: ignore
except Exception:  # pragma: no cover - if TensorBoard is not installed we noop.
    SummaryWriter = None  # type: ignore


@dataclass
class ModelConfig:
    """Model hyper-parameters exposed in the CLI."""

    image_size: int = 224
    projector_dim: int = 256
    d_model: int = 512
    pose_landmarks: int = 13
    projector_dropout: float = 0.0
    fusion_dropout: float = 0.0
    temporal_nhead: int = 8
    temporal_layers: int = 6
    temporal_dim_feedforward: int = 2048
    temporal_dropout: float = 0.1
    vocab_size: int = 32_000
    sequence_length: int = 128


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
    num_workers: int = 0
    batch_size: int = 4
    val_batch_size: Optional[int] = None
    seed: int = 1234
    device: str = "cuda"
    precision: str = "amp"
    tensorboard: Optional[Path] = None
    pin_memory: bool = True


class MultiStreamClassifier(nn.Module):
    """Convenience wrapper around the encoder/decoder pair."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()

        vit_config = ViTConfig(image_size=config.image_size)
        temporal_kwargs = {
            "nhead": config.temporal_nhead,
            "nlayers": config.temporal_layers,
            "dim_feedforward": config.temporal_dim_feedforward,
            "dropout": config.temporal_dropout,
        }

        self.encoder = MultiStreamEncoder(
            backbone_config=vit_config,
            projector_dim=config.projector_dim,
            d_model=config.d_model,
            pose_dim=3 * config.pose_landmarks,
            positional_num_positions=config.sequence_length,
            projector_dropout=config.projector_dropout,
            fusion_dropout=config.fusion_dropout,
            temporal_kwargs=temporal_kwargs,
        )
        self.decoder = TextDecoderStub(d_model=config.d_model, vocab_size=config.vocab_size)

    def forward(
        self,
        *,
        face: torch.Tensor,
        hand_l: torch.Tensor,
        hand_r: torch.Tensor,
        pose: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        miss_mask_hl: Optional[torch.Tensor] = None,
        miss_mask_hr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded = self.encoder(
            face,
            hand_l,
            hand_r,
            pose,
            pad_mask=pad_mask,
            miss_mask_hl=miss_mask_hl,
            miss_mask_hr=miss_mask_hr,
        )
        decoder_mask = None
        if pad_mask is not None:
            decoder_mask = (~pad_mask.to(torch.bool))
        return self.decoder(encoded, padding_mask=decoder_mask)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Data arguments
    parser.add_argument("--face-dir", type=Path, required=True, help="Directory with cropped face frames")
    parser.add_argument("--hand-left-dir", type=Path, required=True, help="Directory with left hand frames")
    parser.add_argument("--hand-right-dir", type=Path, required=True, help="Directory with right hand frames")
    parser.add_argument("--pose-dir", type=Path, required=True, help="Directory with pose .npz files")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="CSV file with video_id/text pairs")
    parser.add_argument("--train-index", type=Path, required=True, help="CSV file listing training video IDs")
    parser.add_argument("--val-index", type=Path, required=True, help="CSV file listing validation video IDs")
    parser.add_argument("--work-dir", type=Path, required=True, help="Directory where checkpoints/logs will be saved")

    parser.add_argument("--num-workers", type=int, default=0, help="Number of DataLoader worker processes")
    parser.add_argument("--batch-size", type=int, default=4, help="Training batch size")
    parser.add_argument("--val-batch-size", type=int, default=None, help="Validation batch size (defaults to training size)")
    parser.add_argument("--seed", type=int, default=1234, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Torch device identifier (e.g. cuda, cuda:0, cpu)")
    parser.add_argument(
        "--precision",
        choices=["fp32", "amp"],
        default="amp",
        help="Numerical precision. 'amp' enables automatic mixed precision on CUDA",
    )
    parser.add_argument(
        "--tensorboard",
        type=Path,
        default=None,
        help="Optional TensorBoard log directory. When omitted TensorBoard logging is disabled.",
    )
    parser.add_argument("--no-pin-memory", action="store_true", help="Disable pinned memory in the data loaders")

    # Model arguments
    parser.add_argument("--image-size", type=int, default=224, help="Input image resolution expected by the ViT backbones")
    parser.add_argument("--projector-dim", type=int, default=256, help="Dimensionality of the per-stream projectors")
    parser.add_argument("--d-model", type=int, default=512, help="Temporal encoder embedding dimension")
    parser.add_argument("--pose-landmarks", type=int, default=13, help="Number of pose landmarks in the NPZ files")
    parser.add_argument("--sequence-length", type=int, default=128, help="Temporal sequence length used during sampling")
    parser.add_argument("--projector-dropout", type=float, default=0.0, help="Dropout applied inside the projectors")
    parser.add_argument("--fusion-dropout", type=float, default=0.0, help="Dropout applied before stream fusion")
    parser.add_argument("--temporal-nhead", type=int, default=8, help="Number of attention heads in the temporal encoder")
    parser.add_argument("--temporal-layers", type=int, default=6, help="Number of transformer layers in the temporal encoder")
    parser.add_argument(
        "--temporal-dim-feedforward",
        type=int,
        default=2048,
        help="Feed-forward dimension inside the temporal encoder",
    )
    parser.add_argument("--temporal-dropout", type=float, default=0.1, help="Dropout used by the temporal encoder")
    parser.add_argument("--vocab-size", type=int, default=32_000, help="Size of the decoder output vocabulary")

    # Optimiser arguments
    parser.add_argument("--epochs", type=int, default=40, help="Number of training epochs")
    parser.add_argument("--optimizer", type=str, default="adamw", help="Optimizer type (adamw, adam, sgd)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="L2 weight decay")
    parser.add_argument(
        "--scheduler",
        choices=["none", "steplr", "cosine"],
        default="none",
        help="Optional learning rate scheduler",
    )
    parser.add_argument("--scheduler-step-size", type=int, default=5, help="Step size for StepLR scheduler")
    parser.add_argument("--scheduler-gamma", type=float, default=0.5, help="Gamma for StepLR scheduler")
    parser.add_argument(
        "--scheduler-tmax",
        type=int,
        default=10,
        help="T_max parameter for CosineAnnealingLR when scheduler=cosine",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="Label smoothing applied to the loss")

    return parser.parse_args()


def build_configs(args: argparse.Namespace) -> tuple[DataConfig, ModelConfig, OptimConfig]:
    data = DataConfig(
        face_dir=args.face_dir,
        hand_left_dir=args.hand_left_dir,
        hand_right_dir=args.hand_right_dir,
        pose_dir=args.pose_dir,
        metadata_csv=args.metadata_csv,
        train_index=args.train_index,
        val_index=args.val_index,
        work_dir=args.work_dir,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        val_batch_size=args.val_batch_size,
        seed=args.seed,
        device=args.device,
        precision=args.precision,
        tensorboard=args.tensorboard,
        pin_memory=not args.no_pin_memory,
    )

    model = ModelConfig(
        image_size=args.image_size,
        projector_dim=args.projector_dim,
        d_model=args.d_model,
        pose_landmarks=args.pose_landmarks,
        projector_dropout=args.projector_dropout,
        fusion_dropout=args.fusion_dropout,
        temporal_nhead=args.temporal_nhead,
        temporal_layers=args.temporal_layers,
        temporal_dim_feedforward=args.temporal_dim_feedforward,
        temporal_dropout=args.temporal_dropout,
        vocab_size=args.vocab_size,
        sequence_length=args.sequence_length,
    )

    scheduler_choice = args.scheduler.lower()
    if scheduler_choice == "none":
        scheduler_choice = None

    optim = OptimConfig(
        optimizer=args.optimizer,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scheduler=scheduler_choice,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        label_smoothing=args.label_smoothing,
    )

    if scheduler_choice == "cosine":
        optim.scheduler_step_size = args.scheduler_tmax

    return data, model, optim


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def _ensure_exists(path: Path, *, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} not found: {path}")


def _validate_paths(config: DataConfig) -> None:
    _ensure_exists(config.face_dir, kind="Face directory")
    _ensure_exists(config.hand_left_dir, kind="Left hand directory")
    _ensure_exists(config.hand_right_dir, kind="Right hand directory")
    _ensure_exists(config.pose_dir, kind="Pose directory")
    _ensure_exists(config.metadata_csv, kind="Metadata CSV")
    _ensure_exists(config.train_index, kind="Train index CSV")
    _ensure_exists(config.val_index, kind="Validation index CSV")
    config.work_dir.mkdir(parents=True, exist_ok=True)


def _texts_to_targets(texts: Iterable[str], vocab_size: int) -> torch.Tensor:
    targets = []
    for text in texts:
        normalized = text.strip().lower()
        digest = hashlib.sha1(normalized.encode("utf-8")).digest()
        value = int.from_bytes(digest[:8], "big") % vocab_size
        targets.append(value)
    if not targets:
        raise ValueError("Received an empty batch when building targets")
    return torch.tensor(targets, dtype=torch.long)


def _build_collate(vocab_size: int) -> Callable[[Iterable[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    def _collate(batch: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        merged = collate_fn(batch)
        inputs = {
            "face": merged["face"],
            "hand_l": merged["hand_l"],
            "hand_r": merged["hand_r"],
            "pose": merged["pose"],
            "pad_mask": merged["pad_mask"],
            "miss_mask_hl": merged["miss_mask_hl"],
            "miss_mask_hr": merged["miss_mask_hr"],
        }
        targets = _texts_to_targets(merged["texts"], vocab_size)
        return {"inputs": inputs, "targets": targets}

    return _collate


def _create_dataloader(
    dataset: LsaTMultiStream,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    vocab_size: int,
) -> DataLoader:
    collate = _build_collate(vocab_size)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate,
    )


def _serialise_config(work_dir: Path, data: DataConfig, model: ModelConfig, optim: OptimConfig) -> None:
    payload = {
        "data": asdict(data),
        "model": asdict(model),
        "optim": asdict(optim),
    }
    serialisable = json.loads(json.dumps(payload, default=str))
    config_path = work_dir / "config.json"
    with config_path.open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, ensure_ascii=False)


def _save_checkpoint(
    path: Path,
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
) -> None:
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": float(val_loss),
    }
    torch.save(state, path)


def main() -> None:
    args = parse_args()
    data_config, model_config, optim_config = build_configs(args)

    setup_logging()
    _validate_paths(data_config)

    set_seed(data_config.seed)

    device = torch.device(data_config.device)
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA requested but not available. Falling back to CPU.")
        device = torch.device("cpu")

    train_dataset = LsaTMultiStream(
        face_dir=str(data_config.face_dir),
        hand_l_dir=str(data_config.hand_left_dir),
        hand_r_dir=str(data_config.hand_right_dir),
        pose_dir=str(data_config.pose_dir),
        csv_path=str(data_config.metadata_csv),
        index_csv=str(data_config.train_index),
        T=model_config.sequence_length,
        img_size=model_config.image_size,
        lkp_count=model_config.pose_landmarks,
    )
    val_dataset = LsaTMultiStream(
        face_dir=str(data_config.face_dir),
        hand_l_dir=str(data_config.hand_left_dir),
        hand_r_dir=str(data_config.hand_right_dir),
        pose_dir=str(data_config.pose_dir),
        csv_path=str(data_config.metadata_csv),
        index_csv=str(data_config.val_index),
        T=model_config.sequence_length,
        img_size=model_config.image_size,
        lkp_count=model_config.pose_landmarks,
    )

    val_batch_size = data_config.val_batch_size or data_config.batch_size
    train_loader = _create_dataloader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        vocab_size=model_config.vocab_size,
    )
    val_loader = _create_dataloader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        vocab_size=model_config.vocab_size,
    )

    model = MultiStreamClassifier(model_config).to(device)
    optimizer = create_optimizer(
        model.parameters(),
        {
            "type": optim_config.optimizer,
            "lr": optim_config.lr,
            "weight_decay": optim_config.weight_decay,
        },
    )

    scheduler = None
    if optim_config.scheduler:
        if optim_config.scheduler == "steplr":
            scheduler = create_scheduler(
                optimizer,
                {
                    "type": "steplr",
                    "step_size": optim_config.scheduler_step_size,
                    "gamma": optim_config.scheduler_gamma,
                },
            )
        elif optim_config.scheduler == "cosine":
            scheduler = create_scheduler(
                optimizer,
                {
                    "type": "cosine",
                    "t_max": optim_config.scheduler_step_size,
                },
            )

    try:
        loss_fn = nn.CrossEntropyLoss(label_smoothing=optim_config.label_smoothing)
    except TypeError:  # pragma: no cover - older PyTorch versions may not support label smoothing.
        logging.warning("Label smoothing unsupported in this PyTorch version. Falling back to default loss.")
        loss_fn = nn.CrossEntropyLoss()

    use_amp = data_config.precision == "amp" and device.type == "cuda" and torch.cuda.is_available()
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    writer = None
    if data_config.tensorboard is not None:
        if SummaryWriter is None:
            logging.warning("TensorBoard requested but not available. Install tensorboard to enable logging.")
        else:
            data_config.tensorboard.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(data_config.tensorboard))

    _serialise_config(data_config.work_dir, data_config, model_config, optim_config)

    best_val = float("inf")
    best_path = data_config.work_dir / "best.pt"
    last_path = data_config.work_dir / "last.pt"

    logging.info("Starting training for %d epochs", args.epochs)
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            loss_fn,
            device=device,
            scaler=scaler,
        )
        val_loss = eval_epoch(model, val_loader, loss_fn, device=device)

        logging.info(
            "Epoch %d/%d - train_loss=%.4f - val_loss=%.4f",
            epoch,
            args.epochs,
            train_loss,
            val_loss,
        )

        if writer is not None:
            writer.add_scalar("loss/train", train_loss, epoch)
            writer.add_scalar("loss/val", val_loss, epoch)

        _save_checkpoint(last_path, model=model, optimizer=optimizer, epoch=epoch, val_loss=val_loss)

        if val_loss < best_val:
            best_val = val_loss
            logging.info("New best validation loss: %.4f", best_val)
            _save_checkpoint(best_path, model=model, optimizer=optimizer, epoch=epoch, val_loss=val_loss)

        if scheduler is not None:
            scheduler.step()

    if writer is not None:
        writer.close()

    logging.info("Training completed. Best validation loss: %.4f", best_val)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
