#!/usr/bin/env python3
"""Valida el pipeline en tiempo real con videos pregrabados."""

from __future__ import annotations

import argparse
import contextlib
from pathlib import Path
from typing import Iterable, Optional

import cv2
import torch

try:  # pragma: no cover - dependencia opcional para la demo
    import mediapipe as mp
except Exception:  # pragma: no cover - MediaPipe es opcional
    mp = None  # type: ignore[assignment]

from slt.runtime import HolisticFrameProcessor, TemporalBuffer
from tools.demo_realtime_multistream import (
    DemoConfig,
    ModelRunner,
    add_model_cli_arguments,
    build_tokenizer,
    decode_sequences,
    draw_overlays,
)


def run_offline(args: argparse.Namespace) -> None:
    if mp is None:
        raise RuntimeError(
            "MediaPipe no está disponible. Instala el paquete 'mediapipe' para ejecutar la prueba offline."
        )
    video_path = Path(args.video)
    if not video_path.exists():
        raise FileNotFoundError(f"No se encontró el video: {video_path}")

    config = DemoConfig(
        sequence_length=args.sequence_length,
        pose_landmarks=args.pose_landmarks,
    )
    if args.max_tokens:
        config.max_tokens = args.max_tokens
    config.beam_size = args.beam_size
    device = torch.device(args.device)

    tokenizer = build_tokenizer(args)

    runner = ModelRunner(
        config=config,
        device=device,
        model_path=args.model,
        model_format=args.model_format,
        max_tokens=config.max_tokens,
        beam_size=config.beam_size,
        onnx_provider=args.onnx_provider,
    )

    buffer = TemporalBuffer(config.sequence_length, config.image_size, config.pose_landmarks)
    processor = HolisticFrameProcessor(
        image_size=config.image_size,
        pose_landmarks=config.pose_landmarks,
        bbox_scale=args.bbox_scale,
        smoothing=args.smoothing,
        max_misses=args.max_misses,
    )

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir el video: {video_path}")

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        writer = cv2.VideoWriter(str(args.output), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"No se pudo inicializar el VideoWriter para {args.output}")

    if args.display:
        cv2.namedWindow("Offline SLT", cv2.WINDOW_NORMAL)

    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)

    mediapipe_contexts = [face_mesh, hands, pose]

    @contextlib.contextmanager
    def closing_all(contexts: Iterable) -> Iterable:
        try:
            yield contexts
        finally:
            for ctx in contexts:
                ctx.close()

    last_text = ""
    frame_idx = 0
    with closing_all(mediapipe_contexts):
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                frame_idx += 1

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                face_result = face_mesh.process(rgb)
                hands_result = hands.process(rgb)
                pose_result = pose.process(rgb)

                face_tensor, hand_l_tensor, hand_r_tensor, pose_tensor, detections, boxes = processor.process(
                    frame,
                    face_result=face_result,
                    hands_result=hands_result,
                    pose_result=pose_result,
                )

                buffer.append(
                    face_tensor,
                    hand_l_tensor,
                    hand_r_tensor,
                    pose_tensor,
                    detected_left=detections.hand_l,
                    detected_right=detections.hand_r,
                )

                inputs = buffer.as_model_inputs(device, backend=runner.backend)
                if inputs is not None:
                    with torch.no_grad():
                        sequences = runner(inputs)
                    text = decode_sequences(sequences, tokenizer)
                    if text and text != last_text:
                        print(f"[{frame_idx:06d}] {text}")
                        last_text = text

                if args.display or writer:
                    draw_overlays(frame, boxes, detections, last_text)
                    if writer:
                        writer.write(frame)
                    if args.display:
                        cv2.imshow("Offline SLT", frame)
                        key = cv2.waitKey(args.wait_ms) & 0xFF
                        if key in (ord("q"), 27):
                            break
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()

    if not args.display and writer is None and last_text:
        print(last_text)


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path, help="Ruta al video MP4/MKV a procesar")
    parser.add_argument("--output", type=Path, default=None, help="Ruta para guardar un video con overlay")
    parser.add_argument("--display", action="store_true", help="Muestra el resultado en una ventana de OpenCV")
    parser.add_argument("--wait-ms", type=int, default=1, help="Retardo en ms para cv2.waitKey cuando --display está activo")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=32,
        help="Número de frames a acumular antes de invocar el modelo",
    )
    parser.add_argument(
        "--pose-landmarks",
        type=int,
        default=13,
        help="Cantidad de landmarks de pose a considerar (MediaPipe Holistic)",
    )
    parser.add_argument("--bbox-scale", type=float, default=1.2, help="Factor de expansión del bounding box")
    parser.add_argument("--smoothing", type=float, default=0.4, help="Factor de suavizado para el tracking de ROI")
    parser.add_argument("--max-misses", type=int, default=5, help="Frames sin detección antes de reiniciar la ROI")

    add_model_cli_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    run_offline(args)


if __name__ == "__main__":  # pragma: no cover - ejecución manual
    main()
