#!/usr/bin/env python3
"""Pre-entrenamiento DINO/iBOT combinando rostro y manos."""

from __future__ import annotations

from typing import Iterable

from tools._pretrain_dino import run_multistream as _run_multistream

_STREAMS = ("face", "hand_left", "hand_right")


def main(argv: Iterable[str] | None = None) -> None:
    """Punto de entrada CLI para el modo multi-stream."""

    _run_multistream(argv, streams=_STREAMS)


if __name__ == "__main__":  # pragma: no cover - CLI
    main()
