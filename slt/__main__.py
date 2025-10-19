"""Demo de entrenamiento corto para ``python -m slt``.

Este módulo reproduce el ejemplo del archivo ``Proyecto`` utilizando la
implementación modular del paquete. Instancia el dataset multi-stream,
construye *DataLoaders* y ejecuta un bucle de entrenamiento/validación de
unas pocas épocas para verificar la integración de los componentes stub.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader

from .data import LsaTMultiStream, collate_fn
from transformers import PreTrainedTokenizerBase

from .models import MultiStreamEncoder, TextSeq2SeqDecoder, ViTConfig
from .training.loops import eval_epoch, train_epoch
from .utils.general import set_seed
from .utils.text import create_tokenizer, encode_batch


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
        tokenizer: PreTrainedTokenizerBase,
        decoder_layers: int = 2,
        decoder_heads: int = 8,
        decoder_dropout: float = 0.1,
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
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if not vocab_size:
            vocab_size = len(tokenizer)
        self.decoder = TextSeq2SeqDecoder(
            d_model=d_model,
            vocab_size=int(vocab_size),
            num_layers=decoder_layers,
            num_heads=decoder_heads,
            dropout=decoder_dropout,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
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


def _build_collate(tokenizer: PreTrainedTokenizerBase, *, max_length: int):
    """Crea una función ``collate_fn`` que incluye etiquetas tokenizadas."""

    def _collate(batch: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
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
            "miss_mask_hl": merged["miss_mask_hl"],
            "miss_mask_hr": merged["miss_mask_hr"],
            "labels": labels,
            "decoder_attention_mask": attention_mask,
            "encoder_attention_mask": merged["pad_mask"].to(torch.long),
        }

        return {"inputs": inputs, "labels": labels, "video_ids": merged["video_ids"]}

    return _collate


def _create_loader(
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
    parser.add_argument("--tokenizer", type=str, required=True, help="Identificador o ruta a un tokenizer de HuggingFace")
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=64,
        help="Longitud máxima de las secuencias de texto tokenizadas",
    )
    parser.add_argument(
        "--decoder-layers",
        type=int,
        default=2,
        help="Capas del decoder seq2seq utilizado durante la demo",
    )
    parser.add_argument(
        "--decoder-heads",
        type=int,
        default=8,
        help="Número de cabezas de atención en el decoder seq2seq",
    )
    parser.add_argument(
        "--decoder-dropout",
        type=float,
        default=0.1,
        help="Dropout aplicado dentro del decoder seq2seq",
    )
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

    tokenizer = create_tokenizer(args.tokenizer)

    train_loader = _create_loader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        tokenizer=tokenizer,
        max_length=args.max_target_length,
    )
    val_loader = _create_loader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        tokenizer=tokenizer,
        max_length=args.max_target_length,
    )

    model = _DemoModel(
        image_size=args.image_size,
        sequence_length=args.sequence_length,
        tokenizer=tokenizer,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_dropout=args.decoder_dropout,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    scaler: Optional["torch.cuda.amp.GradScaler"] = None
    autocast_dtype: Optional[torch.dtype] = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        autocast_dtype = torch.float16

    def _loss_fn(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if hasattr(outputs, "logits"):
            logits = outputs.logits
        else:
            logits = outputs
        vocab = logits.size(-1)
        return F.cross_entropy(
            logits.view(-1, vocab),
            targets.view(-1),
            ignore_index=-100,
            label_smoothing=0.1,
        )

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
