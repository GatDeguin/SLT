#!/usr/bin/env python3
"""Pre-entrena un TinyViT de juguete con recortes de rostro.

El objetivo del script es proporcionar una receta autocontenida para probar
pipelines de DINO sin depender de implementaciones externas.  Trabaja con un
``Dataset`` plano que recorre recursivamente un directorio con imágenes y entrena
un ``TinyViTStub`` de :mod:`tools._tinyvit_stub` optimizando una pérdida simple.

Parámetros
----------
--data-dir : Path
    Ruta al directorio que contiene las imágenes (png, jpg, jpeg, bmp, tiff).
--epochs : int, opcional
    Número de épocas de entrenamiento; por defecto 10.
--batch-size : int, opcional
    Tamaño del batch a utilizar; por defecto 32.
--lr : float, opcional
    Tasa de aprendizaje del optimizador Adam; por defecto 1e-3.
--image-size : int, opcional
    Tamaño cuadrado al que se redimensionan las imágenes antes de alimentar al
    modelo; por defecto 224.
--device : str, opcional
    Dispositivo Torch donde ejecutar el entrenamiento; por defecto ``cuda`` si
    está disponible y ``cpu`` en caso contrario.
--num-workers : int, opcional
    Número de procesos auxiliares para el ``DataLoader``; por defecto 0.
--checkpoint-path : Path, opcional
    Si se especifica, el estado final del modelo se guarda en esta ruta.
--log-level : str, opcional
    Nivel de log mostrado por consola; por defecto ``INFO``.

Salida
------
El script registra en el log el valor medio de la pérdida en cada época y, si se
solicita, guarda un checkpoint del modelo entrenado.
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


class FlatImageDataset(Dataset[torch.Tensor]):
    """Dataset plano que recorre un directorio y devuelve tensores normalizados."""

    def __init__(self, root: Path, image_size: int = 224) -> None:
        self.root = root
        self.image_size = image_size
        self.images: List[Path] = sorted(
            path
            for path in root.rglob("*")
            if path.suffix.lower() in SUPPORTED_EXTENSIONS and path.is_file()
        )
        if not self.images:
            raise FileNotFoundError(f"No se encontraron imágenes soportadas en {root}.")

    def __len__(self) -> int:  # pragma: no cover - getter trivial
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        image = Image.open(self.images[index]).convert("RGB")
        if self.image_size:
            image = ImageOps.fit(image, (self.image_size, self.image_size))
        array = np.asarray(image, dtype=np.float32) / 255.0
        tensor = torch.from_numpy(array).permute(2, 0, 1)
        return tensor


def create_dataloader(data_dir: Path, image_size: int, batch_size: int, num_workers: int) -> DataLoader:
    dataset = FlatImageDataset(data_dir, image_size=image_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)


def train_epoch(model: TinyViTStub, loader: DataLoader, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    """Entrena una época completa y devuelve la pérdida media."""

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
    parser.add_argument("--data-dir", type=Path, required=True, help="Directorio con recortes de rostro")
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

    loader = create_dataloader(args.data_dir, args.image_size, args.batch_size, args.num_workers)
    logging.info("Dataset listo con %d imágenes", len(loader.dataset))

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
