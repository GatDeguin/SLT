"""Extracción de regiones de interés (cara y manos) con MediaPipe.

Este script procesa videos y guarda crops cuadrados de cara y manos junto
con la pose superior. Sirve como versión funcional del pseudocódigo incluido
como referencia en la documentación del proyecto.
"""
from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

try:  # pragma: no cover - dependencias opcionales
    import mediapipe as mp
except Exception:  # pragma: no cover - dependencias opcionales
    mp = None  # type: ignore[assignment]


_MP_WARNING = (
    "MediaPipe no está disponible. Instala el paquete `mediapipe` para poder "
    "extraer regiones de interés."
)


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    """Crea el directorio indicado (y padres) si no existe."""

    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def expand_clamp_bbox(
    x: int,
    y: int,
    w: int,
    h: int,
    scale: float,
    frame_width: int,
    frame_height: int,
) -> Tuple[int, int, int, int]:
    """Amplía un *bounding box* y lo clampa al tamaño del frame."""

    if w <= 0 or h <= 0:
        return x, y, 0, 0

    cx = x + w / 2.0
    cy = y + h / 2.0
    new_w = w * scale
    new_h = h * scale

    x1 = max(0, int(round(cx - new_w / 2.0)))
    y1 = max(0, int(round(cy - new_h / 2.0)))
    x2 = min(frame_width, int(round(cx + new_w / 2.0)))
    y2 = min(frame_height, int(round(cy + new_h / 2.0)))

    return x1, y1, max(0, x2 - x1), max(0, y2 - y1)


def crop_square(
    image: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    out_size: int = 224,
) -> np.ndarray:
    """Extrae un recorte cuadrado centrado en la región indicada."""

    if image.size == 0 or w <= 0 or h <= 0:
        return np.zeros((out_size, out_size, 3), dtype=image.dtype if image.size else np.uint8)

    height, width = image.shape[:2]
    side = int(round(max(w, h)))
    cx, cy = x + w // 2, y + h // 2

    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=image.dtype)

    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


def blur_face_preserve_eyes_mouth(
    frame: np.ndarray,
    face_landmarks: Optional[object],
    keep_radius: int = 6,
) -> np.ndarray:
    """Aplica desenfoque a la cara conservando ojos y boca."""

    if face_landmarks is None:
        return frame

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    height, width = frame.shape[:2]

    # Índices aproximados basados en la malla MediaPipe Face Mesh.
    mouth_idxs = list(range(0, 18)) + list(range(61, 91))
    left_eye_idxs = list(range(33, 134))
    right_eye_idxs = list(range(263, 363))
    keep_indices = set(mouth_idxs + left_eye_idxs + right_eye_idxs)

    for idx, landmark in enumerate(face_landmarks.landmark):
        if idx not in keep_indices:
            continue
        px = int(round(landmark.x * width))
        py = int(round(landmark.y * height))
        if 0 <= px < width and 0 <= py < height:
            cv2.circle(mask, (px, py), keep_radius, 255, -1)

    if mask.max() == 0:
        return frame

    blurred = cv2.GaussianBlur(frame, (31, 31), 0)
    mask3 = cv2.merge([mask, mask, mask])
    inv_mask = cv2.bitwise_not(mask3)
    return cv2.bitwise_and(blurred, inv_mask) + cv2.bitwise_and(frame, mask3)


def _ensure_mediapipe_available() -> bool:
    if mp is None:  # pragma: no cover - dependencias opcionales
        warnings.warn(_MP_WARNING)
        return False
    return True


def process_video(
    video_path: str,
    out_dirs: Dict[str, str],
    pose_dir: str,
    fps_target: int = 25,
    face_blur: bool = False,
) -> bool:
    """Procesa un único video y guarda los ROIs correspondientes."""

    if not _ensure_mediapipe_available():  # pragma: no cover - dependencias opcionales
        return False

    mp_face = mp.solutions.face_mesh
    mp_hands = mp.solutions.hands
    mp_pose = mp.solutions.pose

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        warnings.warn(f"No se pudo abrir el video: {video_path}")
        return False

    basename = Path(video_path).stem
    face_out = ensure_dir(out_dirs.get("face", os.path.join(pose_dir, "face")))
    hand_l_out = ensure_dir(out_dirs.get("hand_l", os.path.join(pose_dir, "hand_l")))
    hand_r_out = ensure_dir(out_dirs.get("hand_r", os.path.join(pose_dir, "hand_r")))
    pose_out = ensure_dir(pose_dir)

    fps = cap.get(cv2.CAP_PROP_FPS) or fps_target
    stride = max(1, int(round(fps / fps_target)))

    pose_frames: List[np.ndarray] = []

    with (
        mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True) as face_mesh,
        mp_hands.Hands(static_image_mode=False, max_num_hands=2) as hands,
        mp_pose.Pose(static_image_mode=False, model_complexity=1) as pose,
    ):
        frame_index = 0
        out_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break

            if frame_index % stride != 0:
                frame_index += 1
                continue

            height, width = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_result = face_mesh.process(rgb)
            hands_result = hands.process(rgb)
            pose_result = pose.process(rgb)

            # Cara
            face_crop = np.zeros((224, 224, 3), dtype=frame.dtype)
            if face_result.multi_face_landmarks:
                face_landmarks = face_result.multi_face_landmarks[0]
                xs = [int(landmark.x * width) for landmark in face_landmarks.landmark]
                ys = [int(landmark.y * height) for landmark in face_landmarks.landmark]
                x1 = max(0, min(xs))
                y1 = max(0, min(ys))
                x2 = min(width, max(xs))
                y2 = min(height, max(ys))
                x, y, w, h = expand_clamp_bbox(x1, y1, x2 - x1, y2 - y1, 1.2, width, height)

                source_frame = blur_face_preserve_eyes_mouth(frame, face_landmarks) if face_blur else frame
                patch = source_frame[y : y + h, x : x + w]
                face_crop = crop_square(patch, 0, 0, patch.shape[1], patch.shape[0], 224)

            cv2.imwrite(str(face_out / f"{basename}_f{out_index:06d}.jpg"), face_crop)

            # Manos
            left_crop = np.zeros((224, 224, 3), dtype=frame.dtype)
            right_crop = np.zeros((224, 224, 3), dtype=frame.dtype)
            if hands_result.multi_hand_landmarks and hands_result.multi_handedness:
                for landmarks, handedness in zip(
                    hands_result.multi_hand_landmarks, hands_result.multi_handedness
                ):
                    xs = [int(l.x * width) for l in landmarks.landmark]
                    ys = [int(l.y * height) for l in landmarks.landmark]
                    x1 = max(0, min(xs))
                    y1 = max(0, min(ys))
                    x2 = min(width, max(xs))
                    y2 = min(height, max(ys))
                    x, y, w, h = expand_clamp_bbox(x1, y1, x2 - x1, y2 - y1, 1.2, width, height)
                    crop = crop_square(frame, x, y, w, h, 224)
                    label = handedness.classification[0].label.lower()
                    if label.startswith("left"):
                        left_crop = crop
                    else:
                        right_crop = crop

            cv2.imwrite(str(hand_l_out / f"{basename}_f{out_index:06d}.jpg"), left_crop)
            cv2.imwrite(str(hand_r_out / f"{basename}_f{out_index:06d}.jpg"), right_crop)

            # Pose
            pose_vec = np.zeros((17, 3), dtype=np.float32)
            if pose_result.pose_landmarks:
                for idx, landmark in enumerate(pose_result.pose_landmarks.landmark[:17]):
                    pose_vec[idx, 0] = landmark.x
                    pose_vec[idx, 1] = landmark.y
                    pose_vec[idx, 2] = landmark.visibility
            pose_frames.append(pose_vec.reshape(-1))

            out_index += 1
            frame_index += 1

    cap.release()

    pose_array = np.asarray(pose_frames, dtype=np.float32)
    np.savez_compressed(pose_out / f"{basename}.npz", pose=pose_array)
    return True


def run_bulk(
    videos_dir: str,
    out_root: str,
    fps_target: int = 25,
    face_blur: bool = False,
) -> None:
    """Procesa todos los videos *.mp4 en ``videos_dir``."""

    if not _ensure_mediapipe_available():  # pragma: no cover - dependencias opcionales
        return

    videos_path = Path(videos_dir)
    if not videos_path.exists():
        warnings.warn(f"Directorio no encontrado: {videos_dir}")
        return

    out_dirs = {
        "face": str(ensure_dir(Path(out_root) / "face")),
        "hand_l": str(ensure_dir(Path(out_root) / "hand_l")),
        "hand_r": str(ensure_dir(Path(out_root) / "hand_r")),
    }
    pose_dir = str(ensure_dir(Path(out_root) / "pose"))

    for video_path in sorted(videos_path.glob("*.mp4")):
        print(f"Procesando {video_path.name}")
        process_video(str(video_path), out_dirs, pose_dir, fps_target=fps_target, face_blur=face_blur)


if __name__ == "__main__":  # pragma: no cover - ejecución manual
    import argparse

    parser = argparse.ArgumentParser(description="Extracción de ROIs (cara/manos/pose)")
    parser.add_argument("videos_dir", help="Directorio con videos .mp4")
    parser.add_argument("out_root", help="Directorio destino para los recortes")
    parser.add_argument("--fps", type=int, default=25, help="FPS de muestreo")
    parser.add_argument(
        "--face-blur",
        action="store_true",
        help="Aplica desenfoque de cara conservando ojos y boca",
    )

    args = parser.parse_args()
    run_bulk(args.videos_dir, args.out_root, fps_target=args.fps, face_blur=args.face_blur)
