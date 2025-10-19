#!/usr/bin/env python3
"""Pre-entrena el TinyViT stub con recortes de manos unificados.

Este script complementa a :mod:`tools.pretrain_dino_face` combinando los
recorridos de ambas manos en un único ``Dataset`` plano.  Comparte el mismo
``TinyViTStub`` y bucle de entrenamiento sencillo basado en una pérdida L2 sobre
las proyecciones producidas por el modelo.

Parámetros
----------
--left-dir : Path
    Directorio con los recortes de la mano izquierda.
--right-dir : Path
    Directorio con los recortes de la mano derecha.
--epochs : int, opcional
    Número de épocas de entrenamiento; por defecto 10.
--batch-size : int, opcional
    Tamaño del batch; por defecto 32.
--lr : float, opcional
    Tasa de aprendizaje del optimizador Adam; por defecto 1e-3.
--image-size : int, opcional
    Resolución cuadrada de entrada; por defecto 224.
--device : str, opcional
    Dispositivo Torch donde ejecutar el entrenamiento.  Si no se especifica se
    selecciona automáticamente ``cuda`` cuando esté disponible.
--num-workers : int, opcional
    Número de workers del ``DataLoader``; por defecto 0.
--checkpoint-path : Path, opcional
    Ruta para guardar el checkpoint final del modelo.
--log-level : str, opcional
    Nivel de logging mostrado por consola; por defecto ``INFO``.

Salida
------
Igual que el script de rostros, la pérdida media por época se informa mediante
logs y opcionalmente se persiste un checkpoint con los pesos resultantes.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from tools._tinyvit_stub import TinyViTConfig, TinyViTStub

SUPPORTED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}


class UnifiedHandsDataset(Dataset[torch.Tensor]):
    """Concatena de forma plana las imágenes de ambas manos."""

    def __init__(self, left_dir: Path, right_dir: Path, image_size: int = 224) -> None:
        self.left_dir = left_dir
        self.right_dir = right_dir
        self.image_size = image_size
        self.images: List[Path] = self._collect_paths(left_dir) + self._collect_paths(right_dir)
        if not self.images:
            raise FileNotFoundError(
                f"No se encontraron imágenes soportadas en {left_dir} ni en {right_dir}."
            )

    def _collect_paths(self, directory: Path) -> List[Path]:
        return sorted(
            path
            for path in directory.rglob("*")
            if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file()
        )

    def __len__(self) -> int:  # pragma: no cover - getter trivial
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.images[index]).convert("RGB")
        if self.image_size:
            image = ImageOps.fit(image, (self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)


def create_dataloader(left_dir: Path, right_dir: Path, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    dataset = UnifiedHandsDataset(left_dir, right_dir, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def train_epoch(model: TinyViTStub, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Ejecuta una época completa de entrenamiento y reporta la pérdida media."""

    model.train()
    criterion = lambda x: (x.pow(2).mean())
    total_loss = 0.0
    total_samples = 0

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad(set_to_none=True)
        embeddings = model(batch)
        loss = criterion(embeddings)
        loss.backward()
        optimizer.step()

        batch_size = batch.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-dir", type=Path, required=True, help="Recortes de mano izquierda")
    parser.add_argument("--right-dir", type=Path, required=True, help="Recortes de mano derecha")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas de entrenamiento")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño del batch")
    parser.add_argument("--lr", type=float, default=1e-3, help="Tasa de aprendizaje")
    parser.add_argument("--image-size", type=int, default=224, help="Resolución cuadrada de entrada")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo de entrenamiento (auto por defecto)")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers del DataLoader")
    parser.add_argument("--checkpoint-path", type=Path, default=None, help="Ruta donde guardar el checkpoint final")
    parser.add_argument("--log-level", type=str, default="INFO", help="Nivel de logging")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    setup_logging(args.log_level)

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    logging.info("Usando dispositivo %s", device)

    loader = create_dataloader(args.left_dir, args.right_dir, args.image_size, args.batch_size, args.num_workers)
    logging.info("Dataset unificado con %d imágenes", len(loader.dataset))

    config = TinyViTConfig()
    model = TinyViTStub(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, loader, optimizer, device)
        logging.info("Época %d/%d - pérdida media: %.6f", epoch, args.epochs, loss)

    if args.checkpoint_path:
        args.checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "config": config.__dict__}, args.checkpoint_path)
        logging.info("Checkpoint guardado en %s", args.checkpoint_path)


if __name__ == "__main__":  # pragma: no cover - punto de entrada CLI
    main()
