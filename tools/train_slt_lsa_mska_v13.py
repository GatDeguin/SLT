"""Deprecated wrapper for the legacy MSKA training pipeline."""

from __future__ import annotations

import warnings

from tools.train_slt_multistream_v9 import main as multistream_main


def main() -> None:
    """Forward execution to :mod:`tools.train_slt_multistream_v9` with a warning."""

    warnings.warn(
        "'tools/train_slt_lsa_mska_v13.py' is deprecated. "
        "Use 'tools/train_slt_multistream_v9.py --use-mska' instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    multistream_main()


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
