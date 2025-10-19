#!/usr/bin/env python3
"""Pre-entrenamiento DINO/iBOT especializado para recortes de rostro."""

from __future__ import annotations

from typing import Iterable

from tools._pretrain_dino import run as _run


def main(argv: Iterable[str] | None = None) -> None:
    """Punto de entrada CLI.

    El wrapper delega la lógica completa en :mod:`tools._pretrain_dino` usando
    ``face`` como stream por defecto para los metadatos y la exportación de
    checkpoints.
    """

    _run(argv, default_stream="face")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
