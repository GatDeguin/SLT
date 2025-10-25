#!/usr/bin/env python3
"""Run batch evaluations across multiple SLT models and aggregate metrics."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import statistics
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import torch

from slt.utils.cli import parse_range_pair, parse_translation_range
from slt.utils.text import (
    TokenizerValidationError,
    create_tokenizer,
    validate_tokenizer,
)

from tools.eval_slt_multistream_v9 import (
    _build_model,
    _compute_latency_stats,
    _compute_metrics,
    _create_dataloader,
    _load_checkpoint,
    _predict,
    _select_device,
    _setup_logging,
)


@dataclass
class ModelSpec:
    """Descriptor for a model to be evaluated."""

    name: str
    kind: str
    value: str


@dataclass
class ModelResult:
    """Aggregated metrics produced after evaluating a model."""

    name: str
    kind: str
    source: str
    metrics: Dict[str, Any]


def _parse_model_spec(raw: str) -> ModelSpec:
    if "=" not in raw:
        raise argparse.ArgumentTypeError("Each --model must follow the NAME=VALUE format")
    name, payload = raw.split("=", 1)
    if not name.strip():
        raise argparse.ArgumentTypeError("Model name cannot be empty")
    if ":" in payload:
        kind, _, value = payload.partition(":")
    else:
        kind, value = "checkpoint", payload
    kind = kind.strip().lower()
    value = value.strip()
    if not value:
        raise argparse.ArgumentTypeError(f"Model '{name}' is missing a payload")
    if kind not in {"checkpoint", "command"}:
        raise argparse.ArgumentTypeError(
            "Model kind must be 'checkpoint' or 'command'"
        )
    return ModelSpec(name=name.strip(), kind=kind, value=value)


def _normalise_metrics(metrics: Mapping[str, Any]) -> Dict[str, Any]:
    normalised: Dict[str, Any] = {}
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            normalised[key] = float(value)
            continue
        if isinstance(value, str):
            trimmed = value.strip()
            if not trimmed:
                continue
            try:
                normalised[key] = float(trimmed)
                continue
            except ValueError:
                normalised[key] = trimmed
                continue
        normalised[key] = value
    return normalised


def _summarise(results: Sequence[ModelResult]) -> Dict[str, Dict[str, Any]]:
    grouped: Dict[str, list[tuple[str, float]]] = {}
    for result in results:
        for metric_name, raw_value in result.metrics.items():
            if isinstance(raw_value, (int, float)):
                grouped.setdefault(metric_name, []).append(
                    (result.name, float(raw_value))
                )
    summary: Dict[str, Dict[str, Any]] = {}
    for metric_name, entries in grouped.items():
        values = [value for _, value in entries]
        if not values:
            continue
        mean = statistics.fmean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        min_entry = min(entries, key=lambda item: item[1])
        max_entry = max(entries, key=lambda item: item[1])
        summary[metric_name] = {
            "mean": float(mean),
            "std": float(std),
            "min": float(min_entry[1]),
            "min_model": min_entry[0],
            "max": float(max_entry[1]),
            "max_model": max_entry[0],
            "count": len(entries),
        }
    return summary


def _run_command(spec: ModelSpec, context: Mapping[str, str]) -> Dict[str, Any]:
    try:
        command = spec.value.format(**context)
    except KeyError as exc:
        missing = str(exc).strip("'")
        raise KeyError(
            f"La plantilla del modelo '{spec.name}' hace referencia a "
            f"una variable desconocida: {missing}"
        ) from exc
    logging.info("Ejecutando comando externo para %s: %s", spec.name, command)
    completed = subprocess.run(
        command,
        shell=True,
        check=True,
        text=True,
        capture_output=True,
    )
    stdout = completed.stdout.strip()
    if not stdout:
        raise RuntimeError(
            f"El comando asociado a '{spec.name}' no produjo salida JSON en stdout"
        )
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:  # pragma: no cover - defensivo
        raise RuntimeError(
            f"No se pudo interpretar la salida JSON de '{spec.name}': {exc}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            f"La salida de '{spec.name}' debe ser un objeto JSON con métricas"
        )
    return dict(payload)


def _slugify(name: str) -> str:
    pieces = [ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in name]
    slug = "".join(pieces).strip("_")
    return slug or "model"


def _ensure_path(path: Path, *, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} inexistente: {path}")


def _validate_paths(args: argparse.Namespace) -> None:
    required = (
        ("face_dir", "Directorio de rostro"),
        ("hand_left_dir", "Directorio de mano izquierda"),
        ("hand_right_dir", "Directorio de mano derecha"),
        ("pose_dir", "Directorio de pose"),
        ("metadata_csv", "CSV de metadata"),
        ("eval_index", "CSV de índices"),
    )
    for attr, label in required:
        value = getattr(args, attr)
        _ensure_path(Path(value), label=label)
    if args.keypoints_dir is not None:
        _ensure_path(args.keypoints_dir, label="Directorio de keypoints")
    if args.gloss_csv is not None:
        _ensure_path(args.gloss_csv, label="CSV de glosas")
    if args.pretrained_checkpoint is not None:
        _ensure_path(args.pretrained_checkpoint, label="Checkpoint pre-entrenado")


def _evaluate_checkpoint(
    spec: ModelSpec,
    args: argparse.Namespace,
    tokenizer,
    loader,
    device: torch.device,
    output_dir: Path,
) -> Dict[str, Any]:
    model = _build_model(args, tokenizer).to(device)
    checkpoint_path = Path(spec.value)
    _load_checkpoint(model, checkpoint_path, device)
    predictions, latencies = _predict(
        model,
        loader,
        device,
        tokenizer,
        max_length=args.max_target_length,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )
    references = [item.reference for item in predictions]
    texts = [item.prediction for item in predictions]
    metrics = _compute_latency_stats(latencies)
    if args.compute_metrics and references and texts:
        bleu, chrf, cer, wer = _compute_metrics(references, texts)
        metrics.update({
            "bleu": bleu,
            "chrf": chrf,
            "cer": cer,
            "wer": wer,
        })
    slug = _slugify(spec.name or checkpoint_path.stem)
    predictions_path = output_dir / f"{slug}_predictions.csv"
    with predictions_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["video_id", "prediction", "reference"])
        for item in predictions:
            writer.writerow([item.video_id, item.prediction, item.reference])
    logging.info(
        "Predicciones del checkpoint %s guardadas en %s",
        checkpoint_path.name,
        predictions_path,
    )
    model.cpu()
    del model
    return metrics


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--face-dir", type=Path, required=True, help="Directorio con frames de rostro")
    parser.add_argument("--hand-left-dir", type=Path, required=True, help="Directorio con frames de mano izquierda")
    parser.add_argument("--hand-right-dir", type=Path, required=True, help="Directorio con frames de mano derecha")
    parser.add_argument("--pose-dir", type=Path, required=True, help="Directorio con poses .npz")
    parser.add_argument("--keypoints-dir", type=Path, help="Directorio opcional con keypoints MediaPipe")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="CSV con columnas video_id/texto")
    parser.add_argument("--eval-index", type=Path, required=True, help="CSV con la lista de video_id a evaluar")
    parser.add_argument("--gloss-csv", type=Path, help="CSV opcional con glosas y etiquetas CTC")
    parser.add_argument("--sequence-length", type=int, default=128, help="Cantidad de frames muestreados")
    parser.add_argument("--image-size", type=int, default=224, help="Tamaño de imagen esperado por el encoder")
    parser.add_argument("--pose-landmarks", type=int, default=13, help="Cantidad de landmarks en el stream de pose")
    parser.add_argument("--batch-size", type=int, default=4, help="Tamaño de batch para la evaluación interna")
    parser.add_argument("--num-workers", type=int, default=0, help="Número de workers del DataLoader")
    parser.add_argument("--device", type=str, default="cuda", help="Dispositivo torch (cuda o cpu)")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false", help="Deshabilita pin_memory en DataLoader")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", help="Habilita pin_memory en DataLoader")
    parser.set_defaults(pin_memory=True)
    parser.add_argument("--tokenizer", type=str, required=True, help="Nombre o ruta del tokenizer a utilizar")
    parser.add_argument("--max-target-length", type=int, default=128, help="Longitud máxima de las secuencias destino")
    parser.add_argument("--num-beams", type=int, default=1, help="Cantidad de beams para decodificación autoregresiva")
    parser.add_argument("--max-new-tokens", type=int, default=None, help="Tokens nuevos máximos generados por modelo")
    parser.add_argument("--projector-dim", type=int, default=128, help="Dimensión de los proyectores de stream")
    parser.add_argument("--d-model", type=int, default=128, help="Dimensión del espacio fusionado")
    parser.add_argument("--projector-dropout", type=float, default=0.05, help="Dropout aplicado a proyectores")
    parser.add_argument("--fusion-dropout", type=float, default=0.05, help="Dropout aplicado tras la fusión")
    parser.add_argument("--temporal-nhead", type=int, default=4, help="Número de cabezales de atención temporal")
    parser.add_argument("--temporal-layers", type=int, default=3, help="Capas del codificador temporal")
    parser.add_argument("--temporal-dim-feedforward", type=int, default=384, help="Dimensión del feedforward temporal")
    parser.add_argument("--temporal-dropout", type=float, default=0.05, help="Dropout del codificador temporal")
    parser.add_argument("--decoder-layers", type=int, default=2, help="Capas del decoder seq2seq")
    parser.add_argument("--decoder-heads", type=int, default=4, help="Cabezas de atención del decoder seq2seq")
    parser.add_argument("--decoder-dropout", type=float, default=0.1, help="Dropout interno del decoder seq2seq")
    parser.add_argument("--pretrained", type=str, default="single_signer", help="Preset pre-entrenado a utilizar")
    parser.add_argument("--pretrained-checkpoint", type=Path, help="Ruta al checkpoint single_signer ya descargado")
    parser.add_argument("--use-mska", dest="use_mska", action="store_true", help="Habilita la rama MSKA durante la evaluación")
    parser.add_argument("--no-mska", dest="use_mska", action="store_false", help="Desactiva la rama MSKA aunque el checkpoint la incluya")
    parser.set_defaults(use_mska=None)
    parser.add_argument("--mska-heads", type=int, help="Número de cabezales de MSKA")
    parser.add_argument("--mska-ff-multiplier", type=int, help="Multiplicador del feedforward MSKA")
    parser.add_argument("--mska-dropout", type=float, help="Dropout aplicado en MSKA")
    parser.add_argument("--leaky-relu-negative-slope", dest="leaky_relu_negative_slope", type=float, help="Coeficiente de fuga para LeakyReLU")
    parser.add_argument("--mska-input-dim", type=int, help="Dimensión de entrada para MSKA")
    parser.add_argument("--mska-ctc-vocab", type=int, help="Tamaño de vocabulario CTC de MSKA")
    parser.add_argument("--mska-use-sgr", dest="mska_use_sgr", action="store_true", help="Activa atención global SGR en MSKA")
    parser.add_argument("--no-mska-use-sgr", dest="mska_use_sgr", action="store_false", help="Desactiva atención global SGR")
    parser.set_defaults(mska_use_sgr=None)
    parser.add_argument("--mska-sgr-shared", dest="mska_sgr_shared", action="store_true", help="Comparte proyecciones en SGR")
    parser.add_argument("--no-mska-sgr-shared", dest="mska_sgr_shared", action="store_false", help="No comparte proyecciones en SGR")
    parser.set_defaults(mska_sgr_shared=None)
    parser.add_argument("--mska-sgr-activation", type=str, help="Función de activación usada por SGR")
    parser.add_argument("--mska-sgr-mix", type=float, help="Factor de mezcla entre streams en SGR")
    parser.add_argument("--mska-detach-teacher", dest="mska_detach_teacher", action="store_true", help="Detiene gradiente hacia el maestro MSKA")
    parser.add_argument("--no-mska-detach-teacher", dest="mska_detach_teacher", action="store_false", help="Propaga gradiente hacia el maestro MSKA")
    parser.set_defaults(mska_detach_teacher=None)
    parser.add_argument("--mska-gloss-hidden-dim", type=int, help="Dimensión oculta para la rama de glosas")
    parser.add_argument("--mska-gloss-second-hidden-dim", type=int, help="Segunda dimensión oculta para glosas")
    parser.add_argument("--mska-gloss-activation", type=str, help="Activación en la fusión de glosas")
    parser.add_argument("--mska-gloss-dropout", type=float, help="Dropout aplicado a glosas")
    parser.add_argument("--mska-gloss-fusion", type=str, help="Modo de fusión de glosas (add/concat/none)")
    parser.add_argument("--keypoint-normalize-center", dest="keypoint_normalize_center", action="store_true", help="Normaliza keypoints alrededor del centro")
    parser.add_argument("--no-keypoint-normalize-center", dest="keypoint_normalize_center", action="store_false", help="Desactiva la normalización al centro")
    parser.set_defaults(keypoint_normalize_center=None)
    parser.add_argument("--keypoint-scale-range", type=str, help="Rango de escala aplicado a los keypoints")
    parser.add_argument("--keypoint-translate-range", type=str, help="Traslación aplicada a los keypoints")
    parser.add_argument("--keypoint-rotate-range", type=str, help="Ángulo mínimo y máximo para rotar keypoints")
    parser.add_argument("--keypoint-resample-range", type=str, help="Rango de remuestreo temporal para keypoints")
    parser.add_argument("--compute-metrics", dest="compute_metrics", action="store_true", help="Calcula métricas BLEU/ChrF/CER/WER si hay referencias")
    parser.add_argument("--skip-metrics", dest="compute_metrics", action="store_false", help="Omite métricas de texto para acelerar la evaluación")
    parser.set_defaults(compute_metrics=True)
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        help=(
            "Especificación del modelo en formato "
            "nombre=checkpoint:/ruta o nombre=command:<plantilla>"
        ),
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directorio donde guardar reportes agregados")
    parser.add_argument("--overwrite", action="store_true", help="Permite reutilizar el directorio de salida si existe")
    parser.add_argument("--log-level", default="INFO", help="Nivel de logging (INFO, DEBUG, ...)")

    args = parser.parse_args(argv)

    if getattr(args, "keypoint_scale_range", None) is not None:
        raw = args.keypoint_scale_range.strip()
        if raw.lower() in {"none", "off"}:
            args.keypoint_scale_range = None
        else:
            try:
                args.keypoint_scale_range = parse_range_pair(
                    raw,
                    positive=True,
                    symmetric_single=False,
                )
            except ValueError as exc:
                parser.error(f"--keypoint-scale-range: {exc}")
    if getattr(args, "keypoint_translate_range", None) is not None:
        raw = args.keypoint_translate_range.strip()
        if raw.lower() in {"none", "off"}:
            args.keypoint_translate_range = None
        else:
            try:
                args.keypoint_translate_range = parse_translation_range(raw)
            except ValueError as exc:
                parser.error(f"--keypoint-translate-range: {exc}")
    if getattr(args, "keypoint_rotate_range", None) is not None:
        raw = args.keypoint_rotate_range.strip()
        if raw.lower() in {"none", "off"}:
            args.keypoint_rotate_range = None
        else:
            try:
                args.keypoint_rotate_range = parse_range_pair(
                    raw,
                    positive=False,
                    symmetric_single=True,
                )
            except ValueError as exc:
                parser.error(f"--keypoint-rotate-range: {exc}")
    if getattr(args, "keypoint_resample_range", None) is not None:
        raw = args.keypoint_resample_range.strip()
        if raw.lower() in {"none", "off"}:
            args.keypoint_resample_range = None
        else:
            try:
                args.keypoint_resample_range = parse_range_pair(
                    raw,
                    positive=True,
                    symmetric_single=False,
                )
            except ValueError as exc:
                parser.error(f"--keypoint-resample-range: {exc}")

    args.model = [_parse_model_spec(item) for item in args.model]
    return args


def _build_context(args: argparse.Namespace, output_dir: Path) -> Dict[str, str]:
    context = {
        "face_dir": str(args.face_dir),
        "hand_left_dir": str(args.hand_left_dir),
        "hand_right_dir": str(args.hand_right_dir),
        "pose_dir": str(args.pose_dir),
        "keypoints_dir": str(args.keypoints_dir) if args.keypoints_dir else "",
        "metadata_csv": str(args.metadata_csv),
        "eval_index": str(args.eval_index),
        "gloss_csv": str(args.gloss_csv) if args.gloss_csv else "",
        "tokenizer": args.tokenizer,
        "sequence_length": str(args.sequence_length),
        "image_size": str(args.image_size),
        "pose_landmarks": str(args.pose_landmarks),
        "batch_size": str(args.batch_size),
        "num_workers": str(args.num_workers),
        "device": args.device,
        "max_target_length": str(args.max_target_length),
        "num_beams": str(args.num_beams),
        "max_new_tokens": "" if args.max_new_tokens is None else str(args.max_new_tokens),
        "output_dir": str(output_dir),
        "pretrained": args.pretrained,
        "pretrained_checkpoint": ""
        if args.pretrained_checkpoint is None
        else str(args.pretrained_checkpoint),
    }
    return context


def _write_reports(
    output_dir: Path,
    *,
    args: argparse.Namespace,
    results: Sequence[ModelResult],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "device": args.device,
        "tokenizer": args.tokenizer,
        "dataset": {
            "face_dir": str(args.face_dir),
            "hand_left_dir": str(args.hand_left_dir),
            "hand_right_dir": str(args.hand_right_dir),
            "pose_dir": str(args.pose_dir),
            "metadata_csv": str(args.metadata_csv),
            "eval_index": str(args.eval_index),
        },
    }
    payload = {
        "metadata": metadata,
        "models": [
            {
                "name": result.name,
                "kind": result.kind,
                "source": result.source,
                "metrics": result.metrics,
            }
            for result in results
        ],
        "summary": _summarise(results),
    }
    json_path = output_dir / "batch_eval_report.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    csv_path = output_dir / "batch_eval_report.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["model", "kind", "metric", "value"])
        for result in results:
            for metric, value in result.metrics.items():
                writer.writerow([result.name, result.kind, metric, value])


def run(argv: Optional[Sequence[str]] = None) -> Sequence[ModelResult]:
    args = parse_args(argv)
    _setup_logging(args.log_level)
    _validate_paths(args)
    context = _build_context(args, args.output_dir)
    if args.output_dir.exists() and not args.overwrite:
        existing = list(args.output_dir.glob("*"))
        if existing:
            raise RuntimeError(
                f"El directorio {args.output_dir} ya contiene archivos. Usa --overwrite para continuar."
            )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = _select_device(args.device)
    tokenizer = create_tokenizer(args.tokenizer)
    if hasattr(tokenizer, "encode"):
        try:
            validate_tokenizer(tokenizer, allow_empty_decode=True)
        except TokenizerValidationError as exc:
            raise RuntimeError(f"Tokenizer inválido: {exc}") from exc
    loader = _create_dataloader(args)
    results: list[ModelResult] = []
    for spec in args.model:
        logging.info("Evaluando modelo %s (%s)", spec.name, spec.kind)
        if spec.kind == "checkpoint":
            metrics = _evaluate_checkpoint(spec, args, tokenizer, loader, device, args.output_dir)
            source = str(Path(spec.value).resolve())
        else:
            metrics = _run_command(spec, context)
            source = spec.value
        normalised = _normalise_metrics(metrics)
        results.append(ModelResult(spec.name, spec.kind, source, normalised))
    _write_reports(args.output_dir, args=args, results=results)
    logging.info("Reportes guardados en %s", args.output_dir)
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover - punto de entrada CLI
    raise SystemExit(main())
