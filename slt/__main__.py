"""Demo de entrenamiento corto para ``python -m slt``.

Este módulo reproduce el ejemplo del archivo ``Proyecto`` utilizando la
implementación modular del paquete. Instancia el dataset multi-stream,
construye *DataLoaders* y ejecuta un bucle de entrenamiento/validación de
unas pocas épocas para verificar la integración de los componentes stub.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .data import LsaTMultiStream, collate_fn
from .models import MultiStreamEncoder, TextDecoderStub, ViTConfig
from .training.loops import eval_epoch, train_epoch
from .utils.general import set_seed


class _DemoModel(nn.Module):
    """Encoder + decoder stub utilizados durante la demostración."""

    def __init__(
        self,
        *,
        image_size: int = 224,
        sequence_length: int = 64,
        projector_dim: int = 256,
        d_model: int = 512,
        pose_landmarks: int = 13,
        vocab_size: int = 32_000,
    ) -> None:
        super().__init__()

        vit_config = ViTConfig(image_size=image_size)
        temporal_kwargs = {
            "nhead": 8,
            "nlayers": 4,
            "dim_feedforward": 2048,
            "dropout": 0.1,
        }
        self.encoder = MultiStreamEncoder(
            backbone_config=vit_config,
            projector_dim=projector_dim,
            d_model=d_model,
            pose_dim=3 * pose_landmarks,
            positional_num_positions=sequence_length,
            temporal_kwargs=temporal_kwargs,
        )
        self.decoder = TextDecoderStub(d_model=d_model, vocab_size=vocab_size)

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
            decoder_mask = ~pad_mask.to(torch.bool)
        return self.decoder(encoded, padding_mask=decoder_mask)


def _texts_to_targets(texts: Iterable[str], vocab_size: int) -> torch.Tensor:
    """Genera etiquetas deterministas a partir de los textos del dataset."""

    targets = []
    for text in texts:
        normalized = text.strip().lower()
        digest = hashlib.sha1(normalized.encode("utf-8")).digest()
        targets.append(int.from_bytes(digest[:8], "big") % vocab_size)
    if not targets:
        raise ValueError("El batch de textos no puede estar vacío.")
    return torch.tensor(targets, dtype=torch.long)


def _build_collate(vocab_size: int):
    """Crea una función ``collate_fn`` compatible con los bucles de entrenamiento."""

    def _collate(batch):
        merged = collate_fn(batch)
        inputs: Dict[str, torch.Tensor] = {
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


def _create_loader(
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


def _select_device(device_flag: str) -> torch.device:
    if device_flag.lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_flag)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento corto del stub multi-stream (demo).",
    )
    parser.add_argument("--face-dir", type=Path, required=True, help="Carpeta con frames de rostro")
    parser.add_argument("--hand-left-dir", type=Path, required=True, help="Carpeta con frames de mano izquierda")
    parser.add_argument("--hand-right-dir", type=Path, required=True, help="Carpeta con frames de mano derecha")
    parser.add_argument("--pose-dir", type=Path, required=True, help="Carpeta con archivos .npz de pose")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="CSV con columnas video_id;texto")
    parser.add_argument("--train-index", type=Path, required=True, help="CSV con lista de video_id para entrenamiento")
    parser.add_argument("--val-index", type=Path, required=True, help="CSV con lista de video_id para validación")
    parser.add_argument("--work-dir", type=Path, default=Path("work_dirs/demo"), help="Directorio donde guardar checkpoints")
    parser.add_argument("--batch-size", type=int, default=2, help="Tamaño de batch")
    parser.add_argument("--epochs", type=int, default=2, help="Cantidad de épocas de entrenamiento")
    parser.add_argument("--sequence-length", type=int, default=64, help="Número de frames muestreados por clip")
    parser.add_argument("--image-size", type=int, default=224, help="Resolución de entrada para los backbones")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate del optimizador AdamW")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers de DataLoader")
    parser.add_argument("--no-pin-memory", action="store_true", help="Deshabilita pinned memory en los loaders")
    parser.add_argument("--device", type=str, default="auto", help="Dispositivo torch (auto, cpu, cuda, cuda:0, ...)")
    parser.add_argument("--seed", type=int, default=1234, help="Semilla aleatoria")
    parser.add_argument("--vocab-size", type=int, default=32_000, help="Tamaño del vocabulario del decoder stub")
    parser.add_argument("--no-amp", action="store_true", help="Desactiva AMP incluso si hay GPU disponible")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    device = _select_device(args.device)
    pin_memory = not args.no_pin_memory
    use_amp = device.type == "cuda" and not args.no_amp and torch.cuda.is_available()

    train_dataset = LsaTMultiStream(
        str(args.face_dir),
        str(args.hand_left_dir),
        str(args.hand_right_dir),
        str(args.pose_dir),
        str(args.metadata_csv),
        str(args.train_index),
        T=args.sequence_length,
        img_size=args.image_size,
    )
    val_dataset = LsaTMultiStream(
        str(args.face_dir),
        str(args.hand_left_dir),
        str(args.hand_right_dir),
        str(args.pose_dir),
        str(args.metadata_csv),
        str(args.val_index),
        T=args.sequence_length,
        img_size=args.image_size,
    )

    train_loader = _create_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        vocab_size=args.vocab_size,
    )
    val_loader = _create_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        vocab_size=args.vocab_size,
    )

    model = _DemoModel(
        image_size=args.image_size,
        sequence_length=args.sequence_length,
        vocab_size=args.vocab_size,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scaler: Optional["torch.cuda.amp.GradScaler"] = None
    autocast_dtype: Optional[torch.dtype] = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        autocast_dtype = torch.float16

    def _loss_fn(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, targets, label_smoothing=0.1)

    args.work_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            _loss_fn,
            device=device,
            scaler=scaler,
            autocast_dtype=autocast_dtype,
        )
        val_loss = eval_epoch(
            model,
            val_loader,
            _loss_fn,
            device=device,
        )
        torch.save(model.state_dict(), args.work_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.work_dir / "best.pt")
        print({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    print("Entrenamiento demo completado. Sustituye los stubs por modelos reales para producción.")


if __name__ == "__main__":
    main()
