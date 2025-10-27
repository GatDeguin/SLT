"""Preparar crops de LSA-T reutilizando el pipeline de `extract_rois_v2.py`."""

from __future__ import annotations

import argparse
import csv
import glob
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

from extract_rois_v2 import (
    _append_metadata,
    _metadata_path,
    _read_metadata_index,
    ensure_dir,
    process_video,
)

from slt.utils.metadata import sanitize_time_value


_VALID_SUFFIXES = {".mp4", ".mov", ".mkv"}

logger = logging.getLogger(__name__)


@dataclass
class _ClipSummary:
    clip_id: str
    video: str
    start: float
    end: float
    duration: float
    span: float
    delta: float
    row: Dict[str, object]


@dataclass
class _MetaLoadResult:
    grouped: Dict[str, Dict[str, object]]
    clips: List[_ClipSummary]
    fieldnames: List[str]


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Genera crops de rostro/manos/pose para LSA-T y datasets externos "
            "utilizando MediaPipe."
        )
    )
    parser.add_argument(
        "--lsa-root",
        type=Path,
        required=True,
        help="Directorio base del corpus LSA-T con archivos de video (*.mp4).",
    )
    parser.add_argument(
        "--meta-csv",
        type=Path,
        default=Path("meta.csv"),
        help="CSV principal (separado por ';') con la metadata de LSA-T.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/single_signer/processed_lsat"),
        help="Ruta destino para escribir los crops (face/hand_l/hand_r/pose).",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Archivo JSONL para registrar métricas. Por defecto dentro de out_root.",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=25,
        help="FPS objetivo tras el muestreo temporal.",
    )
    parser.add_argument(
        "--fps-limit",
        type=float,
        help="FPS máximo leído del video original antes de muestrear.",
    )
    parser.add_argument(
        "--face-blur",
        action="store_true",
        help="Activa el desenfoque facial conservando ojos y boca.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Omite videos ya procesados con éxito según el archivo de metadata.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Desordena la cola de videos antes de procesar (reproducible con seed).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1337,
        help="Seed utilizada al mezclar videos con --shuffle.",
    )
    parser.add_argument(
        "--target-crops",
        type=int,
        help="Detiene la ejecución una vez alcanzados N frames escritos (aprox.).",
    )
    parser.add_argument(
        "--extra-datasets",
        nargs="*",
        default=(),
        help=(
            "Patrones glob (archivos o carpetas) con videos adicionales a mezclar. "
            "Se validan contra meta.csv."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Lista la información descubierta sin ejecutar MediaPipe.",
    )
    parser.add_argument(
        "--duration-threshold",
        type=float,
        default=20.0,
        help=(
            "Duración máxima en segundos antes de marcar un clip como outlier. "
            "Use 0 para desactivar este control."
        ),
    )
    parser.add_argument(
        "--delta-threshold",
        type=float,
        default=0.5,
        help=(
            "Diferencia máxima aceptada (segundos) entre end-start y duration. "
            "Use 0 para desactivar este control."
        ),
    )
    parser.add_argument(
        "--fail-on-outliers",
        action="store_true",
        help=(
            "Finaliza la ejecución con código 1 si hay clips fuera de los "
            "umbrales configurados."
        ),
    )
    return parser.parse_args(argv)


def _load_meta(meta_csv: Path) -> _MetaLoadResult:
    if not meta_csv.exists():
        raise FileNotFoundError(f"No se encontró el CSV de metadata: {meta_csv}")

    with meta_csv.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter=";")
        if reader.fieldnames is None:
            raise ValueError("El CSV de metadata no posee encabezado válido")

        fieldnames = list(reader.fieldnames)
        missing_counts = {"start": 0, "end": 0, "duration": 0}
        missing_rows: List[Dict[str, object]] = []
        missing_label = "__missing_fields__"

        grouped: Dict[str, Dict[str, object]] = defaultdict(
            lambda: {"clip_count": 0, "total_span": 0.0}
        )
        clips: List[_ClipSummary] = []
        for row in reader:
            video = (row.get("video") or "").strip()
            if not video:
                continue

            grouped_entry = grouped[video]

            start = sanitize_time_value(row.get("start"))
            end = sanitize_time_value(row.get("end"))
            duration = sanitize_time_value(row.get("duration"))

            missing_fields = []
            if start is None:
                missing_counts["start"] += 1
                missing_fields.append("start")
            if end is None:
                missing_counts["end"] += 1
                missing_fields.append("end")
            if duration is None:
                missing_counts["duration"] += 1
                missing_fields.append("duration")

            if missing_fields:
                clip_id = (row.get("id") or video or "desconocido").strip()
                logger.warning(
                    "Omitiendo clip %s por metadata incompleta (%s)",
                    clip_id,
                    ", ".join(missing_fields),
                )
                missing_row = dict(row)
                missing_row[missing_label] = ",".join(missing_fields)
                missing_rows.append(missing_row)
                continue

            grouped_entry["clip_count"] = int(grouped_entry["clip_count"]) + 1
            span = max(0.0, end - start)
            grouped_entry["total_span"] = float(grouped_entry["total_span"]) + span

            clip_id = (row.get("id") or video or "desconocido").strip()
            clean_row = {key: row.get(key) for key in fieldnames}
            clean_row.update({"start": start, "end": end, "duration": duration})
            clips.append(
                _ClipSummary(
                    clip_id=clip_id,
                    video=video,
                    start=float(start),
                    end=float(end),
                    duration=float(duration),
                    span=float(span),
                    delta=abs(float(duration) - float(span)),
                    row=clean_row,
                )
            )

        if missing_rows:
            missing_total = len(missing_rows)
            summary = ", ".join(
                f"{key}={value}" for key, value in missing_counts.items() if value
            )
            if not summary:
                summary = "sin_detalle"
            missing_path = meta_csv.with_name("meta_missing.csv")
            wrote_detail = False
            try:
                with missing_path.open("w", encoding="utf-8", newline="") as out:
                    writer = csv.DictWriter(
                        out,
                        fieldnames=[*fieldnames, missing_label],
                        delimiter=";",
                    )
                    writer.writeheader()
                    writer.writerows(missing_rows)
                wrote_detail = True
            except OSError as exc:
                logger.error(
                    "No se pudo escribir meta_missing.csv en %s: %s",
                    missing_path,
                    exc,
                )
            message = (
                f"Se omitieron {missing_total} filas con metadata incompleta ({summary})."
            )
            if wrote_detail:
                message += f" Detalle en {missing_path}."
            logger.info(message)

        return _MetaLoadResult(
            grouped={key: dict(value) for key, value in grouped.items()},
            clips=clips,
            fieldnames=fieldnames,
        )


def _iter_videos(root: Path) -> Iterator[Path]:
    if not root.exists():
        raise FileNotFoundError(f"El directorio de videos no existe: {root}")
    yield from root.rglob("*.mp4")


def _write_outliers(
    meta_csv: Path,
    result: _MetaLoadResult,
    duration_threshold: Optional[float],
    delta_threshold: Optional[float],
) -> Tuple[int, Path]:
    duration_limit = None if duration_threshold in (None, 0) else duration_threshold
    delta_limit = None if delta_threshold in (None, 0) else delta_threshold

    outliers: List[Dict[str, object]] = []
    for clip in result.clips:
        reasons = []
        if duration_limit is not None and clip.duration > duration_limit:
            reasons.append("duration")
        if delta_limit is not None and clip.delta > delta_limit:
            reasons.append("delta")
        if not reasons:
            continue

        row = dict(clip.row)
        row["clean_duration"] = f"{clip.duration:.3f}"
        row["span_seconds"] = f"{clip.span:.3f}"
        row["duration_delta"] = f"{clip.delta:.3f}"
        row["outlier_reasons"] = ",".join(reasons)
        outliers.append(row)

    outlier_path = meta_csv.with_name("meta_outliers.csv")
    fieldnames = list(result.fieldnames)
    for extra in ["clean_duration", "span_seconds", "duration_delta", "outlier_reasons"]:
        if extra not in fieldnames:
            fieldnames.append(extra)

    try:
        with outlier_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter=";")
            writer.writeheader()
            writer.writerows(outliers)
    except OSError as exc:
        logger.error(
            "No se pudo escribir meta_outliers.csv en %s: %s", outlier_path, exc
        )

    return len(outliers), outlier_path


def _summarize_durations(
    meta_csv: Path,
    result: _MetaLoadResult,
    duration_threshold: Optional[float],
    delta_threshold: Optional[float],
) -> Tuple[int, Path]:
    if not result.clips:
        logger.warning("meta.csv no contiene filas con tiempos completos tras limpiarlo.")
        return _write_outliers(meta_csv, result, duration_threshold, delta_threshold)

    durations = [clip.duration for clip in result.clips]
    deltas = [clip.delta for clip in result.clips]

    logger.info(
        (
            "Duraciones limpias por clip: total=%d, media=%.2fs, min=%.2fs, "
            "max=%.2fs, delta_max=%.2fs"
        ),
        len(durations),
        mean(durations),
        min(durations),
        max(durations),
        max(deltas),
    )

    outlier_count, outlier_path = _write_outliers(
        meta_csv, result, duration_threshold, delta_threshold
    )

    duration_msg = (
        "sin_límite"
        if duration_threshold in (None, 0)
        else f">{duration_threshold:.2f}s"
    )
    delta_msg = (
        "sin_límite"
        if delta_threshold in (None, 0)
        else f">{delta_threshold:.2f}s"
    )

    if outlier_count:
        logger.warning(
            (
                "Se detectaron %d clips fuera de rango (duración %s, delta %s). "
                "Detalle en %s"
            ),
            outlier_count,
            duration_msg,
            delta_msg,
            outlier_path,
        )
    else:
        logger.info(
            "Sin outliers según los umbrales (duración %s, delta %s). Detalle en %s",
            duration_msg,
            delta_msg,
            outlier_path,
        )

    return outlier_count, outlier_path


def _expand_extra(patterns: Sequence[str]) -> List[Path]:
    expanded: List[Path] = []
    for pattern in patterns:
        matches = glob.glob(pattern, recursive=True)
        if not matches:
            raise FileNotFoundError(f"El patrón no encontró rutas: {pattern}")
        for raw in matches:
            path = Path(raw)
            if path.is_dir():
                for suffix in _VALID_SUFFIXES:
                    expanded.extend(path.rglob(f"*{suffix}"))
            else:
                if path.suffix.lower() not in _VALID_SUFFIXES:
                    continue
                expanded.append(path)
    return expanded


def _collect_videos(
    lsa_root: Path, extra_patterns: Sequence[str]
) -> Dict[str, Tuple[Path, str]]:
    videos: Dict[str, Tuple[Path, str]] = {}
    for item in _iter_videos(lsa_root):
        stem = item.stem
        if stem in videos and videos[stem][0] != item:
            raise ValueError(f"Video duplicado detectado para ID {stem}: {item}")
        videos[stem] = (item, "lsa_t")

    for extra in _expand_extra(extra_patterns):
        stem = extra.stem
        if stem in videos and videos[stem][0] != extra:
            raise ValueError(f"Video duplicado detectado al mezclar extras: {extra}")
        videos[stem] = (extra, "extra")

    return videos


def _validate_sources(
    videos: Dict[str, Tuple[Path, str]], meta: Dict[str, Dict[str, object]]
) -> None:
    video_ids = set(videos)
    meta_ids = set(meta)

    missing_meta = sorted(video_ids - meta_ids)
    if missing_meta:
        joined = ", ".join(missing_meta[:5])
        raise ValueError(
            "Los siguientes videos no aparecen en meta.csv: "
            f"{joined}{'...' if len(missing_meta) > 5 else ''}"
        )

    missing_files = sorted(meta_ids - video_ids)
    if missing_files:
        joined = ", ".join(missing_files[:5])
        print(
            "Advertencia: meta.csv contiene videos sin archivo disponible: "
            f"{joined}{'...' if len(missing_files) > 5 else ''}",
            file=sys.stderr,
        )


def _resolve_metadata_path(out_root: Path, metadata_flag: Optional[Path]) -> Path:
    if metadata_flag:
        ensure_dir(metadata_flag.parent)
        return metadata_flag
    return _metadata_path(out_root, None)


def _build_queue(
    videos: Dict[str, Tuple[Path, str]],
    meta: Dict[str, Dict[str, object]],
    shuffle: bool,
    seed: int,
) -> List[Tuple[str, Path, Dict[str, object], str]]:
    queue = [
        (video_id, path, meta[video_id], dataset)
        for video_id, (path, dataset) in videos.items()
        if video_id in meta
    ]
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(queue)
    else:
        queue.sort(key=lambda item: item[0])
    return queue


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)

    meta_result = _load_meta(args.meta_csv)
    outlier_count, outlier_path = _summarize_durations(
        args.meta_csv, meta_result, args.duration_threshold, args.delta_threshold
    )
    if args.fail_on_outliers and outlier_count:
        logger.error(
            (
                "Se encontraron %d clips fuera de los umbrales configurados. "
                "Revise %s antes de continuar."
            ),
            outlier_count,
            outlier_path,
        )
        return 1

    meta = meta_result.grouped
    videos = _collect_videos(args.lsa_root, args.extra_datasets)
    _validate_sources(videos, meta)

    out_root = ensure_dir(args.output_root)
    out_dirs = {
        "face": str(ensure_dir(out_root / "face")),
        "hand_l": str(ensure_dir(out_root / "hand_l")),
        "hand_r": str(ensure_dir(out_root / "hand_r")),
    }
    pose_dir = str(ensure_dir(out_root / "pose"))

    metadata_path = _resolve_metadata_path(out_root, args.metadata)
    index = _read_metadata_index(metadata_path)

    total_frames = sum(
        entry.get("frames_written", 0) or 0 for entry in index.values() if entry.get("success")
    )

    queue = _build_queue(videos, meta, args.shuffle, args.seed)

    if args.dry_run:
        print("=== Resumen (dry-run) ===")
        print(f"Videos detectados: {len(queue)}")
        print(f"Frames ya exportados: {total_frames}")
        for video_id, path, info, dataset in queue[:10]:
            print(
                "- {video_id} -> {path} (clips={clips}, span={span:.2f}s, "
                "dataset={dataset})".format(
                    video_id=video_id,
                    path=path,
                    clips=info["clip_count"],
                    span=info["total_span"],
                    dataset=dataset,
                )
            )
        return 0

    processed = 0
    skipped = 0
    for video_id, video_path, info, dataset in queue:
        entry = index.get(video_path.name)
        if args.resume and entry and entry.get("success"):
            skipped += 1
            continue

        result = process_video(
            str(video_path),
            out_dirs,
            pose_dir,
            fps_target=args.fps,
            face_blur=args.face_blur,
            fps_limit=args.fps_limit,
        )
        result.update(
            {
                "video": video_path.name,
                "video_id": video_id,
                "video_path": str(video_path),
                "clip_count": info.get("clip_count"),
                "total_span": info.get("total_span"),
                "dataset": dataset,
            }
        )

        _append_metadata(metadata_path, result)

        if result.get("success"):
            processed += 1
            total_frames += int(result.get("frames_written") or 0)
        else:
            print(
                f"Fallo procesando {video_path.name}: {result.get('error')}",
                file=sys.stderr,
            )

        if args.target_crops and total_frames >= args.target_crops:
            print(
                f"Objetivo alcanzado: {total_frames} frames >= {args.target_crops}."
            )
            break

    print(
        f"Resumen: OK={processed}, omitidos={skipped}, frames acumulados={total_frames}."
    )
    print(f"Metadata almacenada en: {metadata_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover - ejecución manual
    raise SystemExit(main())
