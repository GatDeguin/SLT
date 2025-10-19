#!/usr/bin/env python3
"""Pre-entrenamiento DINO/iBOT para recortes de manos."""

from __future__ import annotations

from typing import Iterable

from tools._pretrain_dino import run as _run


def main(argv: Iterable[str] | None = None) -> None:
    """Punto de entrada CLI con configuración por defecto para manos."""

    _run(argv, default_stream="hand")


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
