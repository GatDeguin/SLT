"""CI helper to compare smoke-test losses against README expectations."""

from __future__ import annotations

import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from torch import nn
from torch.utils.data import DataLoader

from slt.training.loops import eval_epoch, train_epoch
from tests.training.test_short_loop import TinyRegressionDataset, _make_model

README_PATH = Path(__file__).resolve().parent.parent / "README.md"
REL_TOLERANCE = 0.15
ABS_TOLERANCE = 0.75


def _parse_expected_metrics(readme_text: str) -> dict[str, float]:
    initial = None
    final = None
    for line in readme_text.splitlines():
        if "Pérdida inicial" in line:
            match = re.search(r"≈\s*([0-9]+(?:[.,][0-9]+)?)", line)
            if match:
                initial = float(match.group(1).replace(",", "."))
        if "Pérdida final" in line:
            match = re.search(r"≈\s*([0-9]+(?:[.,][0-9]+)?)", line)
            if match:
                final = float(match.group(1).replace(",", "."))
    if initial is None or final is None:
        raise RuntimeError("No se encontraron las métricas esperadas en README.md")
    return {"initial": initial, "final": final}


def _run_reference_training() -> dict[str, float]:
    torch.manual_seed(0)
    dataset = TinyRegressionDataset()
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    model = _make_model(dataset.features.size(1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.05)
    loss_fn = nn.MSELoss()

    initial = eval_epoch(model, loader, loss_fn, device="cpu")
    for _ in range(3):
        train_epoch(model, loader, optimizer, loss_fn, device="cpu")
    final = eval_epoch(model, loader, loss_fn, device="cpu")
    return {"initial": initial.loss, "final": final.loss}


def main() -> int:
    readme = README_PATH.read_text(encoding="utf-8")
    expected = _parse_expected_metrics(readme)
    observed = _run_reference_training()

    print("=== Verificación de métricas de entrenamiento sintético ===")
    status_ok = True
    for key in ("initial", "final"):
        exp_val = expected[key]
        obs_val = observed[key]
        abs_diff = abs(obs_val - exp_val)
        rel_diff = abs_diff / exp_val if exp_val else abs_diff
        print(
            f"{key.title()} → observado={obs_val:.3f} | esperado≈{exp_val:.3f} | "
            f"Δabs={abs_diff:.3f} | Δrel={rel_diff:.2%}"
        )
        if abs_diff > ABS_TOLERANCE and rel_diff > REL_TOLERANCE:
            status_ok = False
    if not status_ok:
        print("ERROR: las métricas observadas se desvían de los valores documentados.")
        return 1
    print("Métricas dentro de tolerancia.")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
