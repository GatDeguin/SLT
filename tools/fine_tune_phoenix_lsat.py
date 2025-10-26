#!/usr/bin/env python3
"""Launch a 5k-step fine-tuning run from Phoenix weights into the LSA-T pipeline."""

from __future__ import annotations

import argparse
import logging
import shlex
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CHECKPOINT = _REPO_ROOT / "work_dirs" / "phoenix" / "best.pth"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Wrapper around train_slt_multistream_v9.py to fine-tune Phoenix checkpoints "
            "with differentiated learning rates and a 5k-step schedule."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Configuration template passed to train_slt_multistream_v9.py",
    )
    parser.add_argument(
        "--work-dir",
        type=Path,
        help="Optional work directory override for the fine-tuning run",
    )
    parser.add_argument(
        "--phoenix-checkpoint",
        type=Path,
        default=_DEFAULT_CHECKPOINT,
        help="Checkpoint initialisation path (defaults to work_dirs/phoenix/best.pth)",
    )
    parser.add_argument(
        "--lr-encoder",
        type=float,
        default=5e-5,
        help="Learning rate assigned to the encoder parameter group",
    )
    parser.add_argument(
        "--lr-decoder",
        type=float,
        default=1e-4,
        help="Learning rate assigned to the decoder parameter group",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=5000,
        help="Upper bound of training iterations to execute",
    )
    parser.add_argument(
        "--subset-size",
        type=int,
        default=5000,
        help="Optional cap on the number of training samples to iterate",
    )
    parser.add_argument(
        "extras",
        nargs=argparse.REMAINDER,
        help=(
            "Additional flags forwarded to train_slt_multistream_v9.py. Prefix them with "
            "'--' to separate from this wrapper's options."
        ),
    )
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.lr_encoder <= 0 or args.lr_decoder <= 0:
        raise ValueError("Learning rates must be positive")
    if args.max_train_steps <= 0 or args.subset_size <= 0:
        raise ValueError("Step and subset limits must be positive integers")
    checkpoint = args.phoenix_checkpoint
    if not checkpoint.exists():
        raise FileNotFoundError(f"Phoenix checkpoint not found: {checkpoint}")
    args.phoenix_checkpoint = checkpoint
    if args.work_dir is not None:
        args.work_dir = args.work_dir.expanduser().resolve()
    args.config = args.config.expanduser().resolve()


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _validate_args(args)

    cli_path = _REPO_ROOT / "tools" / "train_slt_multistream_v9.py"
    command = [
        sys.executable,
        str(cli_path),
        "--config",
        str(args.config),
        "--init-checkpoint",
        str(args.phoenix_checkpoint),
        "--lr-encoder",
        str(args.lr_encoder),
        "--lr-decoder",
        str(args.lr_decoder),
        "--max-train-steps",
        str(args.max_train_steps),
        "--subset-size",
        str(args.subset_size),
    ]
    if args.work_dir is not None:
        command.extend(["--work-dir", str(args.work_dir)])
    if args.extras:
        command.extend(args.extras)

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
    logging.info("Executing %s", " ".join(shlex.quote(part) for part in command))
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
