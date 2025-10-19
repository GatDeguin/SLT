#!/usr/bin/env python3
"""Command line entry-point to train the multi-stream SLT stub model.

The script wires together the reusable components from :mod:`slt` to
instantiate the dataset, encoder/decoder pair and the optimisation loop.
Tokenizer and decoder functionality rely on HuggingFace ``transformers`` so the
package must be installed alongside PyTorch.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from slt.data import LsaTMultiStream, collate_fn
from slt.models import MultiStreamEncoder, TextSeq2SeqDecoder, ViTConfig, load_dinov2_backbone
from slt.training.loops import eval_epoch, train_epoch
from slt.training.optim import create_optimizer, create_scheduler
from slt.utils.general import set_seed
from slt.utils.text import create_tokenizer, encode_batch

from transformers import AutoConfig, PreTrainedTokenizerBase

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
    sequence_length: int = 128
    decoder_layers: int = 2
    decoder_heads: int = 8
    decoder_dropout: float = 0.1
    decoder_model: Optional[str] = None
    decoder_config: Optional[str] = None
    face_backbone: Optional[str] = None
    hand_left_backbone: Optional[str] = None
    hand_right_backbone: Optional[str] = None
    freeze_face_backbone: bool = False
    freeze_hand_left_backbone: bool = False
    freeze_hand_right_backbone: bool = False


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
    tokenizer: Optional[str] = None
    max_target_length: int = 64


class MultiStreamClassifier(nn.Module):
    """Convenience wrapper around the encoder/decoder pair."""

    def __init__(self, config: ModelConfig, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()

        vit_config = ViTConfig(image_size=config.image_size)
        temporal_kwargs = {
            "nhead": config.temporal_nhead,
            "nlayers": config.temporal_layers,
            "dim_feedforward": config.temporal_dim_feedforward,
            "dropout": config.temporal_dropout,
        }

        backbone_specs = {
            "face": (config.face_backbone, config.freeze_face_backbone),
            "hand_left": (config.hand_left_backbone, config.freeze_hand_left_backbone),
            "hand_right": (config.hand_right_backbone, config.freeze_hand_right_backbone),
        }
        external_backbones: Dict[str, torch.nn.Module] = {}
        for stream, (spec, freeze_flag) in backbone_specs.items():
            if spec:
                external_backbones[stream] = load_dinov2_backbone(spec, freeze=freeze_flag)

        self.encoder = MultiStreamEncoder(
            backbone_config=vit_config,
            projector_dim=config.projector_dim,
            d_model=config.d_model,
            pose_dim=3 * config.pose_landmarks,
            positional_num_positions=config.sequence_length,
            projector_dropout=config.projector_dropout,
            fusion_dropout=config.fusion_dropout,
            temporal_kwargs=temporal_kwargs,
            backbones=external_backbones if external_backbones else None,
        )
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if not vocab_size:
            vocab_size = len(tokenizer)

        decoder_config = None
        if config.decoder_config:
            decoder_config = AutoConfig.from_pretrained(config.decoder_config)
            hidden_size = getattr(decoder_config, "d_model", None)
            if hidden_size is not None and hidden_size != config.d_model:
                raise ValueError(
                    "Decoder configuration hidden size does not match encoder dimensionality: "
                    f"expected {config.d_model}, got {hidden_size}."
                )

        decoder_model_name = None if decoder_config is not None else config.decoder_model

        self.decoder = TextSeq2SeqDecoder(
            d_model=config.d_model,
            vocab_size=int(vocab_size),
            num_layers=config.decoder_layers,
            num_heads=config.decoder_heads,
            dropout=config.decoder_dropout,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
            pretrained_model_name_or_path=decoder_model_name,
            config=decoder_config,
        )

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
        labels: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
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
        if encoder_attention_mask is None and pad_mask is not None:
            encoder_attention_mask = pad_mask.to(torch.long)
        return self.decoder(
            encoded,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
        )

    def generate(
        self,
        *,
        face: torch.Tensor,
        hand_l: torch.Tensor,
        hand_r: torch.Tensor,
        pose: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        miss_mask_hl: Optional[torch.Tensor] = None,
        miss_mask_hr: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
        **generation_kwargs: Any,
    ) -> torch.LongTensor:
        encoded = self.encoder(
            face,
            hand_l,
            hand_r,
            pose,
            pad_mask=pad_mask,
            miss_mask_hl=miss_mask_hl,
            miss_mask_hr=miss_mask_hr,
        )
        if encoder_attention_mask is None and pad_mask is not None:
            encoder_attention_mask = pad_mask.to(torch.long)
        return self.decoder.generate(
            encoded,
            encoder_attention_mask=encoder_attention_mask,
            **generation_kwargs,
        )


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
    parser.add_argument(
        "--decoder-layers",
        type=int,
        default=2,
        help="Number of layers in the seq2seq decoder",
    )
    parser.add_argument(
        "--decoder-heads",
        type=int,
        default=8,
        help="Number of attention heads in the seq2seq decoder",
    )
    parser.add_argument(
        "--decoder-dropout",
        type=float,
        default=0.1,
        help="Dropout probability inside the seq2seq decoder",
    )
    parser.add_argument(
        "--face-backbone",
        type=str,
        default=None,
        help=(
            "Backbone specification for the face stream. The value is forwarded to "
            "slt.models.backbones.load_dinov2_backbone, e.g. 'torchhub::facebookresearch/dinov2:dinov2_vits14'."
        ),
    )
    parser.add_argument(
        "--hand-left-backbone",
        type=str,
        default=None,
        help=(
            "Backbone specification for the left hand stream (see --face-backbone for format)."
        ),
    )
    parser.add_argument(
        "--hand-right-backbone",
        type=str,
        default=None,
        help=(
            "Backbone specification for the right hand stream (see --face-backbone for format)."
        ),
    )
    parser.add_argument(
        "--freeze-face-backbone",
        action="store_true",
        help="Freeze all parameters in the face backbone after loading external weights.",
    )
    parser.add_argument(
        "--freeze-hand-left-backbone",
        action="store_true",
        help="Freeze all parameters in the left hand backbone after loading external weights.",
    )
    parser.add_argument(
        "--freeze-hand-right-backbone",
        action="store_true",
        help="Freeze all parameters in the right hand backbone after loading external weights.",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="HuggingFace tokenizer identifier or local path. When omitted the decoder checkpoint is used.",
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=128,
        help="Maximum length of the tokenised target sequences",
    )
    parser.add_argument(
        "--decoder-model",
        type=str,
        default=None,
        help="Optional pretrained decoder model name or path passed to TextSeq2SeqDecoder.",
    )
    parser.add_argument(
        "--decoder-config",
        type=str,
        default=None,
        help="Optional decoder configuration name or path used to initialise TextSeq2SeqDecoder.",
    )

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
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=None,
        help="Optional maximum norm for gradient clipping before each optimisation step.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint path to resume training from (loads model, optimiser and AMP scaler state).",
    )

    args = parser.parse_args()
    if args.decoder_config and args.decoder_model:
        parser.error("--decoder-model and --decoder-config are mutually exclusive")
    if not args.tokenizer and not args.decoder_model:
        parser.error("Provide --tokenizer when --decoder-model is not specified")
    if args.clip_grad_norm is not None and args.clip_grad_norm <= 0:
        logging.warning("Ignoring non-positive --clip-grad-norm value: %s", args.clip_grad_norm)
        args.clip_grad_norm = None
    return args


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
        tokenizer=args.tokenizer,
        max_target_length=args.max_target_length,
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
        sequence_length=args.sequence_length,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_dropout=args.decoder_dropout,
        decoder_model=args.decoder_model,
        decoder_config=args.decoder_config,
        face_backbone=args.face_backbone,
        hand_left_backbone=args.hand_left_backbone,
        hand_right_backbone=args.hand_right_backbone,
        freeze_face_backbone=args.freeze_face_backbone,
        freeze_hand_left_backbone=args.freeze_hand_left_backbone,
        freeze_hand_right_backbone=args.freeze_hand_right_backbone,
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
        grad_clip_norm=args.clip_grad_norm,
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


def _build_collate(
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
) -> Callable[[Iterable[Dict[str, torch.Tensor]]], Dict[str, torch.Tensor]]:
    def _collate(batch: Iterable[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        merged = collate_fn(batch)
        tokenized = encode_batch(
            tokenizer,
            merged["texts"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
        )
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        inputs: Dict[str, torch.Tensor] = {
            "face": merged["face"],
            "hand_l": merged["hand_l"],
            "hand_r": merged["hand_r"],
            "pose": merged["pose"],
            "pad_mask": merged["pad_mask"],
            "lengths": merged["lengths"],
            "miss_mask_hl": merged["miss_mask_hl"],
            "miss_mask_hr": merged["miss_mask_hr"],
            "labels": labels,
            "decoder_attention_mask": attention_mask,
            "encoder_attention_mask": merged["pad_mask"].to(torch.long),
        }
        return {"inputs": inputs, "labels": labels, "video_ids": merged["video_ids"]}

    return _collate


def _create_dataloader(
    dataset: LsaTMultiStream,
    *,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    pin_memory: bool,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> DataLoader:
    collate = _build_collate(tokenizer, max_length=max_length)
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
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    best_val: Optional[float] = None,
) -> None:
    state = {
        "epoch": epoch,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "val_loss": float(val_loss),
    }
    if scaler is not None:
        state["scaler_state"] = scaler.state_dict()
    if best_val is not None:
        state["best_val"] = float(best_val)
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

    tokenizer_source = data_config.tokenizer or model_config.decoder_model
    if tokenizer_source is None:
        raise ValueError("Tokenizer source could not be resolved.")
    tokenizer = create_tokenizer(tokenizer_source)

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
        tokenizer=tokenizer,
        max_length=data_config.max_target_length,
    )
    val_loader = _create_dataloader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        tokenizer=tokenizer,
        max_length=data_config.max_target_length,
    )

    model = MultiStreamClassifier(model_config, tokenizer).to(device)
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

    def _loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        vocab = logits.size(-1)
        try:
            return F.cross_entropy(
                logits.view(-1, vocab),
                targets.view(-1),
                ignore_index=-100,
                label_smoothing=optim_config.label_smoothing,
            )
        except TypeError:  # pragma: no cover - fallback when label smoothing unsupported
            logging.warning(
                "Label smoothing unsupported in this PyTorch version. Falling back to default loss.")
            return F.cross_entropy(
                logits.view(-1, vocab),
                targets.view(-1),
                ignore_index=-100,
            )

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
    start_epoch = 0

    if args.resume is not None:
        checkpoint_path = args.resume
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer_state = checkpoint.get("optimizer_state")
        if optimizer_state is not None:
            optimizer.load_state_dict(optimizer_state)
        if scaler is not None and "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        best_val = float(checkpoint.get("best_val", checkpoint.get("val_loss", best_val)))
        start_epoch = int(checkpoint.get("epoch", 0))
        logging.info("Resumed training from %s at epoch %d", checkpoint_path, start_epoch)

    if args.epochs <= start_epoch:
        logging.info(
            "Checkpoint epoch (%d) is greater than or equal to requested epochs (%d). Nothing to do.",
            start_epoch,
            args.epochs,
        )
        if writer is not None:
            writer.close()
        return

    logging.info("Starting training for %d epochs", args.epochs)
    for epoch in range(start_epoch + 1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            _loss_fn,
            device=device,
            scaler=scaler,
            grad_clip_norm=optim_config.grad_clip_norm,
        )
        val_loss = eval_epoch(model, val_loader, _loss_fn, device=device)

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

        _save_checkpoint(
            last_path,
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            val_loss=val_loss,
            scaler=scaler,
            best_val=best_val,
        )

        if val_loss < best_val:
            best_val = val_loss
            logging.info("New best validation loss: %.4f", best_val)
            _save_checkpoint(
                best_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                scaler=scaler,
                best_val=best_val,
            )

        if scheduler is not None:
            scheduler.step()

    if writer is not None:
        writer.close()

    logging.info("Training completed. Best validation loss: %.4f", best_val)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
