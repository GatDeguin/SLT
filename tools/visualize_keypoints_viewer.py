#!/usr/bin/env python3
"""Visor interactivo para validar video, keypoints y subtítulos en sincronía."""

from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from slt.data.lsa_t_multistream import _resolve_mediapipe_layout
from slt.utils.metadata import SplitSegment, parse_split_column, sanitize_time_value


@dataclass
class SubtitleConfig:
    """Configuración básica para interpretar el CSV de subtítulos."""

    csv_path: Path
    delimiter: str = ";"
    id_column: str = "id"
    video_column: str = "video"
    text_column: str = "text"
    start_column: str = "start"
    end_column: str = "end"
    split_column: Optional[str] = "split"
    target_id: Optional[str] = None
    target_video: Optional[str] = None
    absolute_times: bool = False


@dataclass
class SubtitleEntry:
    """Segmento individual preparado para mostrar en pantalla."""

    text: str
    start: float
    end: float

    def contains(self, timestamp: float) -> bool:
        return self.start <= timestamp <= self.end


@dataclass
class ViewerConfig:
    """Parámetros del visor."""

    window_name: str = "SLT keypoint viewer"
    wait_time_ms: int = 1
    loop: bool = False
    display_scale: float = 1.0
    font_scale: float = 0.8
    font_thickness: int = 2
    subtitle_margin: int = 24
    subtitle_max_width: int = 900
    confidence_threshold: float = 0.2
    normalised_keypoints: bool = True
    video_offset: float = 0.0
    keypoints_offset: float = 0.0
    seek_to_start: bool = True


@dataclass
class KeypointData:
    """Estructura con los keypoints y metadatos auxiliares."""

    frames: np.ndarray
    layout: Dict[str, List[int]]
    fps: float


VIDEO_EXTENSIONS = (".mp4", ".mkv", ".mov", ".avi", ".webm")


def _wrap_text(
    text: str,
    font: int,
    font_scale: float,
    thickness: int,
    max_width: int,
) -> List[str]:
    """Divide el subtítulo en líneas que quepan en ``max_width`` píxeles."""

    words = text.split()
    if not words:
        return [""]

    lines: List[str] = []
    current: List[str] = []

    for word in words:
        tentative = " ".join(current + [word])
        width, _ = cv2.getTextSize(tentative, font, font_scale, thickness)[0]
        if width <= max_width or not current:
            current.append(word)
            continue
        lines.append(" ".join(current))
        current = [word]

    if current:
        lines.append(" ".join(current))

    return lines


def _draw_subtitles(
    frame: np.ndarray,
    subtitle: str,
    viewer_cfg: ViewerConfig,
) -> None:
    """Superpone el subtítulo activo sobre ``frame``."""

    font = cv2.FONT_HERSHEY_SIMPLEX
    lines = _wrap_text(
        subtitle,
        font,
        viewer_cfg.font_scale,
        viewer_cfg.font_thickness,
        min(viewer_cfg.subtitle_max_width, frame.shape[1] - 2 * viewer_cfg.subtitle_margin),
    )

    if not lines:
        return

    line_height = int(30 * viewer_cfg.font_scale)
    total_height = len(lines) * line_height + viewer_cfg.subtitle_margin
    y_base = frame.shape[0] - viewer_cfg.subtitle_margin
    x_margin = viewer_cfg.subtitle_margin

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (x_margin - 10, y_base - total_height),
        (frame.shape[1] - x_margin + 10, y_base + viewer_cfg.subtitle_margin // 2),
        (0, 0, 0),
        thickness=-1,
    )
    alpha = 0.55
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, dst=frame)

    for idx, line in enumerate(lines[::-1]):
        y = y_base - idx * line_height
        cv2.putText(
            frame,
            line,
            (x_margin, y),
            font,
            viewer_cfg.font_scale,
            (255, 255, 255),
            viewer_cfg.font_thickness,
            lineType=cv2.LINE_AA,
        )


def _load_subtitles(cfg: SubtitleConfig) -> Tuple[List[SubtitleEntry], Optional[float]]:
    """Carga los subtítulos y retorna segmentos más el inicio sugerido."""

    if not cfg.csv_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV: {cfg.csv_path}")

    with cfg.csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=cfg.delimiter)
        rows = list(reader)

    if not rows:
        raise ValueError(f"El CSV {cfg.csv_path} no contiene filas.")

    filtered: List[Dict[str, str]] = []
    for row in rows:
        row_norm = {
            key: (value.strip() if isinstance(value, str) else value)
            for key, value in row.items()
        }
        if cfg.target_id and row_norm.get(cfg.id_column) != cfg.target_id:
            continue
        if cfg.target_video and row_norm.get(cfg.video_column) != cfg.target_video:
            continue
        filtered.append(row_norm)

    if not filtered:
        target = cfg.target_id or cfg.target_video or "<sin filtro>"
        raise ValueError(f"No se hallaron filas que coincidan con {target!r} en {cfg.csv_path}.")

    segments: List[SubtitleEntry] = []
    clip_start: Optional[float] = None

    for row in filtered:
        start_raw = row.get(cfg.start_column)
        end_raw = row.get(cfg.end_column)
        start = sanitize_time_value(start_raw) or 0.0
        end = sanitize_time_value(end_raw) or start
        if clip_start is None:
            clip_start = start
        split_raw = row.get(cfg.split_column) if cfg.split_column else None
        if split_raw:
            parsed = parse_split_column(split_raw)
            source_segments: Iterable[SplitSegment] = parsed
        else:
            source_segments = [SplitSegment(row.get(cfg.text_column, ""), start, end)]

        for segment in source_segments:
            rel_start = segment.start
            rel_end = segment.end
            if not cfg.absolute_times and clip_start is not None:
                rel_start -= clip_start
                rel_end -= clip_start
            rel_start = max(rel_start, 0.0)
            rel_end = max(rel_end, rel_start)
            segments.append(SubtitleEntry(segment.text, rel_start, rel_end))

    segments.sort(key=lambda item: (item.start, item.end))
    return segments, (None if cfg.absolute_times else clip_start)


def _load_keypoints(path: Path, fps: Optional[float]) -> KeypointData:
    """Lee el archivo de keypoints y arma un layout estándar."""

    if not path.exists():
        raise FileNotFoundError(f"No se encontró el archivo de keypoints: {path}")

    ext = path.suffix.lower()
    layout_name: Optional[str] = None

    if ext == ".npz":
        with np.load(path, allow_pickle=True) as data:
            if "keypoints" not in data:
                raise KeyError("El .npz no contiene la clave 'keypoints'.")
            frames = data["keypoints"]
            if "layout" in data:
                layout_name = str(data["layout"])
    else:
        frames = np.load(path)

    if frames.ndim != 3:
        raise ValueError(f"Los keypoints deben tener forma (T, N, C); recibido {frames.shape}.")

    num_landmarks = frames.shape[1]
    layout = _resolve_mediapipe_layout(num_landmarks)

    fps_value = float(fps) if fps and fps > 0 else math.nan
    return KeypointData(frames=frames.astype(np.float32), layout=layout, fps=fps_value)


def _resolve_path_by_stem(
    directory: Path,
    stem: str,
    allowed_suffixes: Optional[Sequence[str]] = None,
) -> Path:
    """Devuelve el primer archivo en ``directory`` cuyo stem coincide con ``stem``."""

    if allowed_suffixes:
        normalised = tuple(ext.lower() for ext in allowed_suffixes)
        allowed_suffixes = normalised
        for ext in normalised:
            candidate = directory / f"{stem}{ext}"
            if candidate.exists():
                return candidate
    for candidate in directory.glob(f"{stem}.*"):
        if not candidate.is_file():
            continue
        if allowed_suffixes and candidate.suffix.lower() not in allowed_suffixes:
            continue
        return candidate
    raise FileNotFoundError(
        f"No se encontró un archivo con stem '{stem}' dentro de {directory}."
    )


def _iter_clip_resources(
    videos_dir: Path,
    keypoints_dir: Path,
    subtitle_cfg: SubtitleConfig,
) -> Iterator[Tuple[Path, Path, SubtitleConfig, str]]:
    """Itera los clips presentes en ``meta.csv`` resolviendo rutas relativas."""

    if not videos_dir.is_dir():
        raise FileNotFoundError(f"El directorio de videos no existe: {videos_dir}")
    if not keypoints_dir.is_dir():
        raise FileNotFoundError(f"El directorio de keypoints no existe: {keypoints_dir}")

    with subtitle_cfg.csv_path.open("r", encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter=subtitle_cfg.delimiter)
        rows = list(reader)

    if not rows:
        raise ValueError(f"El CSV {subtitle_cfg.csv_path} no contiene filas.")

    filtered: List[Dict[str, str]] = []
    for row in rows:
        row_norm = {
            key: (value.strip() if isinstance(value, str) else value)
            for key, value in row.items()
        }
        if (
            subtitle_cfg.target_id
            and row_norm.get(subtitle_cfg.id_column) != subtitle_cfg.target_id
        ):
            continue
        if (
            subtitle_cfg.target_video
            and row_norm.get(subtitle_cfg.video_column) != subtitle_cfg.target_video
        ):
            continue
        filtered.append(row_norm)

    if not filtered:
        target = subtitle_cfg.target_id or subtitle_cfg.target_video or "<sin filtro>"
        raise ValueError(
            f"No se hallaron filas que coincidan con {target!r} en {subtitle_cfg.csv_path}."
        )

    def _prepare_entry(
        row: Dict[str, str],
        resolved_video: Optional[Path],
    ) -> Tuple[Path, Path, SubtitleConfig, str]:
        clip_id = row.get(subtitle_cfg.id_column)
        if not clip_id:
            raise ValueError(
                f"La fila {row} no contiene la columna {subtitle_cfg.id_column!r}."
            )

        video_value = row.get(subtitle_cfg.video_column)
        video_candidates = [clip_id]
        if video_value and video_value not in video_candidates:
            video_candidates.append(video_value)

        video_path = resolved_video
        if video_path is None:
            for stem in video_candidates:
                try:
                    video_path = _resolve_path_by_stem(
                        videos_dir,
                        stem,
                        VIDEO_EXTENSIONS,
                    )
                    break
                except FileNotFoundError:
                    continue
        if video_path is None:
            raise FileNotFoundError(
                f"No se encontró el video asociado a {clip_id!r} dentro de {videos_dir}."
            )

        try:
            keypoints_path = _resolve_path_by_stem(
                keypoints_dir,
                clip_id,
                (".npy", ".npz"),
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"No se encontró el archivo de keypoints para {clip_id!r} en {keypoints_dir}."
            ) from exc

        clip_cfg = replace(
            subtitle_cfg,
            target_id=clip_id,
            target_video=video_value or subtitle_cfg.target_video,
        )

        return video_path, keypoints_path, clip_cfg, clip_id

    if subtitle_cfg.target_id or subtitle_cfg.target_video:
        for row in filtered:
            yield _prepare_entry(row, None)
        return

    video_files = sorted(
        path
        for path in videos_dir.iterdir()
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    )

    rows_by_id: Dict[str, Dict[str, str]] = {}
    rows_by_video: Dict[str, List[Dict[str, str]]] = {}
    for row in filtered:
        clip_id = row.get(subtitle_cfg.id_column)
        if clip_id:
            rows_by_id[clip_id] = row
        video_value = row.get(subtitle_cfg.video_column)
        if video_value:
            rows_by_video.setdefault(video_value, []).append(row)

    yielded: set[str] = set()
    for video_path in video_files:
        stem = video_path.stem
        matched_rows: List[Dict[str, str]] = []
        row_by_id = rows_by_id.get(stem)
        if row_by_id:
            matched_rows.append(row_by_id)
        else:
            matched_rows.extend(rows_by_video.get(stem, []))

        if not matched_rows:
            print(
                f"Advertencia: no se hallaron filas en {subtitle_cfg.csv_path} "
                f"para el video {stem!r}."
            )
            continue

        for row in matched_rows:
            clip_id = row.get(subtitle_cfg.id_column)
            if not clip_id or clip_id in yielded:
                continue
            yielded.add(clip_id)
            yield _prepare_entry(row, video_path)

    missing_rows = [
        row
        for row in filtered
        if row.get(subtitle_cfg.id_column) and row.get(subtitle_cfg.id_column) not in yielded
    ]
    for row in missing_rows:
        clip_id = row.get(subtitle_cfg.id_column) or "<sin id>"
        print(
            f"Advertencia: no se encontró un video en {videos_dir} para el clip {clip_id!r}."
        )


def _select_subtitle(segments: Sequence[SubtitleEntry], timestamp: float) -> str:
    """Encuentra el subtítulo activo para ``timestamp``."""

    for segment in segments:
        if segment.contains(timestamp):
            return segment.text
    return ""


def _draw_keypoints(
    frame: np.ndarray,
    keypoints: np.ndarray,
    layout: Dict[str, List[int]],
    viewer_cfg: ViewerConfig,
    visible_mask: Optional[np.ndarray] = None,
) -> None:
    """Pinta los keypoints sobre la imagen."""

    colors = {
        "body": (0, 255, 0),
        "face": (255, 200, 0),
        "hand_l": (0, 165, 255),
        "hand_r": (255, 105, 180),
    }

    for name, indices in layout.items():
        color = colors.get(name, (255, 255, 255))
        for idx in indices:
            if idx >= keypoints.shape[0]:
                continue
            x, y = keypoints[idx, :2]
            confidence = keypoints[idx, 2] if keypoints.shape[1] >= 3 else 1.0
            if viewer_cfg.confidence_threshold > 0 and confidence < viewer_cfg.confidence_threshold:
                continue
            if visible_mask is not None and not visible_mask[idx]:
                continue
            cv2.circle(frame, (int(x), int(y)), 3, color, thickness=-1)


def _resize_frame(frame: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return frame
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)


def run_viewer(
    video_path: Path,
    keypoints_path: Path,
    subtitle_cfg: SubtitleConfig,
    viewer_cfg: ViewerConfig,
    video_fps: Optional[float] = None,
    keypoints_fps: Optional[float] = None,
) -> None:
    """Ejecuta el bucle principal del visor."""

    subtitles, clip_start = _load_subtitles(subtitle_cfg)
    keypoints_data = _load_keypoints(keypoints_path, keypoints_fps)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    fps_video = video_fps or cap.get(cv2.CAP_PROP_FPS) or 0.0
    if fps_video <= 0:
        raise RuntimeError("No fue posible inferir el FPS del video. Usa --fps para forzarlo.")

    fps_keypoints = (
        keypoints_fps
        or (None if math.isnan(keypoints_data.fps) else keypoints_data.fps)
        or fps_video
    )

    if viewer_cfg.seek_to_start and clip_start and clip_start > 0:
        cap.set(cv2.CAP_PROP_POS_MSEC, clip_start * 1000)

    frame_index = 0
    total_keypoint_frames = keypoints_data.frames.shape[0]
    clip_reference = clip_start or 0.0
    playback_start = time.perf_counter()

    cv2.namedWindow(viewer_cfg.window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                if viewer_cfg.loop:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_index = 0
                    playback_start = time.perf_counter()
                    continue
                break

            original_frame = frame.copy()
            frame = _resize_frame(frame, viewer_cfg.display_scale)

            raw_pos_ms = cap.get(cv2.CAP_PROP_POS_MSEC)
            video_pos = raw_pos_ms / 1000.0 if raw_pos_ms > 0 else float("nan")
            if math.isnan(video_pos) or video_pos <= 0:
                video_pos = frame_index / fps_video
                if viewer_cfg.seek_to_start and clip_start:
                    video_pos += clip_start

            relative_time = max(0.0, video_pos - clip_reference)
            subtitle_time = (
                video_pos if subtitle_cfg.absolute_times else relative_time
            ) + viewer_cfg.video_offset
            subtitle_text = _select_subtitle(subtitles, subtitle_time)

            if total_keypoint_frames > 0:
                kp_time = relative_time + viewer_cfg.keypoints_offset
                kp_frame = int(round(kp_time * fps_keypoints))
                kp_frame = max(0, min(kp_frame, total_keypoint_frames - 1))
                kp_array = keypoints_data.frames[kp_frame].copy()

                if viewer_cfg.normalised_keypoints:
                    height, width = original_frame.shape[:2]
                    kp_array[:, 0] *= width
                    kp_array[:, 1] *= height

                if viewer_cfg.display_scale != 1.0:
                    scale = viewer_cfg.display_scale
                    kp_array[:, :2] *= scale

                visibility = None
                if kp_array.shape[1] > 3:
                    visibility = kp_array[:, 3] > 0.0

                _draw_keypoints(frame, kp_array, keypoints_data.layout, viewer_cfg, visibility)

            if subtitle_text:
                _draw_subtitles(frame, subtitle_text, viewer_cfg)

            info = (
                f"t={relative_time:0.2f}s | video={video_pos:0.2f}s | frame={frame_index}"
            )
            cv2.putText(
                frame,
                info,
                (viewer_cfg.subtitle_margin, viewer_cfg.subtitle_margin + 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

            elapsed = time.perf_counter() - playback_start
            target_elapsed = relative_time
            remaining = target_elapsed - elapsed
            dynamic_wait_ms = 0
            if viewer_cfg.wait_time_ms != 0 and remaining > 0:
                dynamic_wait_ms = int(round(remaining * 1000))

            base_wait = viewer_cfg.wait_time_ms
            if base_wait < 0:
                base_wait = 0

            if base_wait == 0:
                wait_arg = 0
            else:
                wait_arg = max(base_wait, dynamic_wait_ms, 1)

            cv2.imshow(viewer_cfg.window_name, frame)
            key = cv2.waitKey(wait_arg) & 0xFF
            if key in (27, ord("q")):
                break

            frame_index += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualiza video, keypoints MediaPipe y subtítulos en tiempo real.",
    )
    parser.add_argument("--video", type=Path, help="Ruta al video base.")
    parser.add_argument(
        "--keypoints",
        type=Path,
        help="Archivo .npz/.npy con los keypoints MediaPipe (forma (T, N, C)).",
    )
    parser.add_argument(
        "--csv",
        required=True,
        type=Path,
        help="CSV con subtítulos y columnas de tiempo (ej. meta.csv).",
    )
    parser.add_argument(
        "--videos-dir",
        type=Path,
        help="Directorio base que contiene los videos segmentados (alternativa a --video).",
    )
    parser.add_argument(
        "--keypoints-dir",
        type=Path,
        help="Directorio con los keypoints MediaPipe por clip (alternativa a --keypoints).",
    )
    parser.add_argument("--segment-id", help="Valor de la columna 'id' a visualizar.")
    parser.add_argument("--video-id", help="Filtra filas por la columna 'video'.")
    parser.add_argument("--delimiter", default=";", help="Delimitador utilizado en el CSV.")
    parser.add_argument("--id-column", default="id", help="Nombre de la columna con IDs únicos.")
    parser.add_argument(
        "--video-column",
        default="video",
        help="Nombre de la columna que identifica el video fuente.",
    )
    parser.add_argument(
        "--text-column",
        default="text",
        help="Columna con el texto del subtítulo cuando no hay 'split'.",
    )
    parser.add_argument(
        "--start-column",
        default="start",
        help="Columna con el timestamp inicial del clip en segundos.",
    )
    parser.add_argument(
        "--end-column",
        default="end",
        help="Columna con el timestamp final del clip en segundos.",
    )
    parser.add_argument(
        "--split-column",
        default="split",
        help="Columna con la lista de segmentos parciales (literal de Python).",
    )
    parser.add_argument(
        "--absolute-times",
        action="store_true",
        help="No restar el inicio del clip; usa tiempos absolutos del CSV.",
    )
    parser.add_argument(
        "--window-name",
        default="SLT keypoint viewer",
        help="Nombre de la ventana de visualización.",
    )
    parser.add_argument(
        "--wait-ms",
        type=int,
        default=1,
        help="Tiempo de espera para cv2.waitKey (ms). Usa 0 para avanzar manualmente.",
    )
    parser.add_argument(
        "--loop",
        action="store_true",
        help="Reinicia el video automáticamente al llegar al final.",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=1.0,
        help="Factor de escala aplicado al frame mostrado.",
    )
    parser.add_argument(
        "--font-scale",
        type=float,
        default=0.8,
        help="Escala de fuente para los subtítulos.",
    )
    parser.add_argument(
        "--font-thickness",
        type=int,
        default=2,
        help="Grosor de fuente para subtítulos.",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.2,
        help="Confianza mínima para dibujar un keypoint.",
    )
    parser.add_argument(
        "--subtitle-width",
        type=int,
        default=900,
        help="Ancho máximo (px) reservado para subtítulos.",
    )
    parser.add_argument(
        "--subtitle-margin",
        type=int,
        default=24,
        help="Margen en píxeles alrededor de los subtítulos.",
    )
    kp_norm_group = parser.add_mutually_exclusive_group()
    kp_norm_group.add_argument(
        "--normalised-keypoints",
        dest="normalised_keypoints",
        action="store_true",
        default=True,
        help=(
            "Interpreta los keypoints en coordenadas normalizadas [0, 1]. "
            "Usa --absolute-keypoints para desactivarlo."
        ),
    )
    kp_norm_group.add_argument(
        "--absolute-keypoints",
        dest="normalised_keypoints",
        action="store_false",
    )
    parser.add_argument(
        "--video-offset",
        type=float,
        default=0.0,
        help="Offset temporal (s) aplicado al video antes de mostrar subtítulos.",
    )
    parser.add_argument(
        "--keypoints-offset",
        type=float,
        default=0.0,
        help="Offset temporal (s) aplicado a los keypoints respecto del video.",
    )
    parser.add_argument(
        "--fps",
        type=float,
        help="FPS del video si no puede inferirse automáticamente.",
    )
    parser.add_argument(
        "--keypoints-fps",
        type=float,
        help="FPS de los keypoints cuando difiere del video.",
    )
    parser.add_argument(
        "--no-seek",
        action="store_true",
        help="No posicionar el video en el inicio del clip según el CSV.",
    )

    args = parser.parse_args()

    has_directories = args.videos_dir is not None or args.keypoints_dir is not None
    if has_directories:
        if not args.videos_dir or not args.keypoints_dir:
            parser.error("Debe especificar --videos-dir y --keypoints-dir para el modo por lotes.")
    else:
        missing = [
            flag
            for flag, value in (("--video", args.video), ("--keypoints", args.keypoints))
            if value is None
        ]
        if missing:
            parser.error(
                "Los argumentos --video y --keypoints son obligatorios cuando no se utilizan "
                "--videos-dir/--keypoints-dir."
            )

    return args


def main() -> None:
    args = _parse_args()

    subtitle_cfg = SubtitleConfig(
        csv_path=args.csv,
        delimiter=args.delimiter,
        id_column=args.id_column,
        video_column=args.video_column,
        text_column=args.text_column,
        start_column=args.start_column,
        end_column=args.end_column,
        split_column=args.split_column,
        target_id=args.segment_id,
        target_video=args.video_id,
        absolute_times=args.absolute_times,
    )

    viewer_cfg = ViewerConfig(
        window_name=args.window_name,
        wait_time_ms=args.wait_ms,
        loop=args.loop,
        display_scale=args.display_scale,
        font_scale=args.font_scale,
        font_thickness=args.font_thickness,
        subtitle_margin=args.subtitle_margin,
        subtitle_max_width=args.subtitle_width,
        confidence_threshold=args.confidence_threshold,
        normalised_keypoints=args.normalised_keypoints,
        video_offset=args.video_offset,
        keypoints_offset=args.keypoints_offset,
        seek_to_start=not args.no_seek,
    )

    if args.videos_dir and args.keypoints_dir:
        clips = list(_iter_clip_resources(args.videos_dir, args.keypoints_dir, subtitle_cfg))
        total = len(clips)
        for index, (video_path, keypoints_path, clip_cfg, clip_id) in enumerate(
            clips, start=1
        ):
            print(
                f"[{index}/{total}] Visualizando clip {clip_id} "
                f"({video_path.name}, {keypoints_path.name})."
            )
            try:
                run_viewer(
                    video_path=video_path,
                    keypoints_path=keypoints_path,
                    subtitle_cfg=clip_cfg,
                    viewer_cfg=viewer_cfg,
                    video_fps=args.fps,
                    keypoints_fps=args.keypoints_fps,
                )
            except KeyboardInterrupt:
                print("Interrupción del usuario. Finalizando.")
                break

            if index < total:
                response = input(
                    "Presiona Enter para continuar con el siguiente clip o escribe 'q' para salir: "
                )
                if response.strip().lower().startswith("q"):
                    print("Finalizando por solicitud del usuario.")
                    break
    else:
        run_viewer(
            video_path=args.video,
            keypoints_path=args.keypoints,
            subtitle_cfg=subtitle_cfg,
            viewer_cfg=viewer_cfg,
            video_fps=args.fps,
            keypoints_fps=args.keypoints_fps,
        )


if __name__ == "__main__":
    main()
