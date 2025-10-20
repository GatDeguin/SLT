"""Utilities to interpret evaluation metrics and export them to dashboards.

Este script sirve como referencia rápida para analistas y equipos de producto
que necesiten consumir los reportes generados por ``tools/eval_slt_multistream_v9.py``.
Incluye helpers para cargar los archivos ``report.json``/``report.csv`` y
producir payloads listos para ser enviados a los dashboards internos.

Uso rápido
---------
$ python docs/metrics_dashboard_integration.py /ruta/al/directorio_de_reportes

El script imprimirá un resumen de las métricas agregadas y mostrará cómo
estructurar la carga útil JSON esperada por los dashboards corporativos.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

AGGREGATE_KEY = "aggregate"
CHECKPOINT_KEY = "checkpoints"


@dataclass
class DashboardPayload:
    """Payload básico para dashboards internos.

    Attributes
    ----------
    aggregate:
        Métricas agregadas (media, desvío estándar, etc.).
    checkpoints:
        Métricas individuales de cada checkpoint evaluado.
    """

    aggregate: Mapping[str, Mapping[str, float]]
    checkpoints: Sequence[Mapping[str, object]]

    def to_json(self) -> str:
        """Serializa el payload a JSON indentado."""

        return json.dumps({
            "aggregate": self.aggregate,
            "checkpoints": list(self.checkpoints),
        }, ensure_ascii=False, indent=2)


def load_report(directory: Path) -> DashboardPayload:
    """Carga ``report.json`` generado por la herramienta de evaluación."""

    report_path = directory / "report.json"
    if not report_path.exists():
        raise FileNotFoundError(f"No se encontró report.json en {directory}")

    with report_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    aggregate = data.get(AGGREGATE_KEY, {})
    checkpoints = data.get(CHECKPOINT_KEY, [])
    return DashboardPayload(aggregate=aggregate, checkpoints=checkpoints)


def explain_metrics(payload: DashboardPayload) -> None:
    """Imprime una explicación legible de las métricas disponibles."""

    if not payload.aggregate:
        print("No se encontraron métricas agregadas en el reporte.")
    else:
        print("=== Métricas agregadas ===")
        for metric, stats in payload.aggregate.items():
            mean = float(stats.get("mean", 0.0) or 0.0)
            std = float(stats.get("std", 0.0) or 0.0)
            pmin = float(stats.get("min", 0.0) or 0.0)
            pmax = float(stats.get("max", 0.0) or 0.0)
            count = int(stats.get("count", 0) or 0)
            print(
                f"- {metric}: media={mean:.3f} | std={std:.3f} | min={pmin:.3f} | max={pmax:.3f} | n={count}"
            )
        print()

    if not payload.checkpoints:
        print("No se encontraron checkpoints individuales.")
        return

    print("=== Checkpoints ===")
    for item in payload.checkpoints:
        checkpoint_name = Path(item.get("checkpoint", "")).name or "desconocido"
        metrics = item.get("metrics", {})
        print(f"Checkpoint: {checkpoint_name}")
        print(f"  BLEU: {float(metrics.get('bleu', 0.0) or 0.0):.3f}")
        print(f"  ChrF: {float(metrics.get('chrf', 0.0) or 0.0):.3f}")
        print(f"  CER:  {float(metrics.get('cer', 0.0) or 0.0):.3f}%")
        print(f"  WER:  {float(metrics.get('wer', 0.0) or 0.0):.3f}%")
        print(f"  Latencia media: {float(metrics.get('avg_latency_ms', 0.0) or 0.0):.3f} ms")
        print(f"  Latencia p95:  {float(metrics.get('p95_latency_ms', 0.0) or 0.0):.3f} ms")
        examples = item.get("examples", [])[:3]
        if examples:
            print("  Ejemplos:")
            for example in examples:
                vid = example.get("video_id", "?")
                pred = example.get("prediction", "")
                ref = example.get("reference", "")
                latency = float(example.get("latency_ms", 0.0) or 0.0)
                print(f"    - {vid}: pred='{pred}' | ref='{ref}' | latencia={latency:.3f} ms")
        print()


def export_for_dashboard(payload: DashboardPayload, output: Path) -> None:
    """Exporta un CSV simplificado listo para ingesta automática."""

    fieldnames = [
        "checkpoint",
        "metric",
        "value",
    ]
    with output.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for item in payload.checkpoints:
            checkpoint_name = Path(item.get("checkpoint", "")).name or "desconocido"
            metrics = item.get("metrics", {})
            for metric, value in metrics.items():
                writer.writerow({
                    "checkpoint": checkpoint_name,
                    "metric": metric,
                    "value": value,
                })


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "report_dir",
        type=Path,
        help="Directorio que contiene report.json/report.csv generados por la evaluación",
    )
    parser.add_argument(
        "--dashboard-csv",
        type=Path,
        help="Ruta opcional donde exportar un CSV listo para dashboards",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    payload = load_report(args.report_dir)
    explain_metrics(payload)
    if args.dashboard_csv:
        export_for_dashboard(payload, args.dashboard_csv)
        print(f"Archivo para dashboard exportado en {args.dashboard_csv}")
    else:
        print("Sugerencia: use --dashboard-csv para exportar métricas automáticamente.")


if __name__ == "__main__":  # pragma: no cover - utilidad CLI
    main()
