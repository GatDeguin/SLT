"""Componentes reutilizables para demos en tiempo real."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional, Tuple

import numpy as np
import torch
import cv2

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


@dataclass
class FrameDetections:
    """Representa el estado de las detecciones de un frame."""

    face: bool
    hand_l: bool
    hand_r: bool


class TemporalBuffer:
    """Acumula ventanas temporales listas para inferencia."""

    def __init__(self, sequence_length: int, image_size: int, pose_landmarks: int) -> None:
        self.sequence_length = sequence_length
        self.image_size = image_size
        self.pose_landmarks = pose_landmarks

        self.face: Deque[torch.Tensor] = deque(maxlen=sequence_length)
        self.hand_l: Deque[torch.Tensor] = deque(maxlen=sequence_length)
        self.hand_r: Deque[torch.Tensor] = deque(maxlen=sequence_length)
        self.pose: Deque[torch.Tensor] = deque(maxlen=sequence_length)
        self.miss_hl: Deque[bool] = deque(maxlen=sequence_length)
        self.miss_hr: Deque[bool] = deque(maxlen=sequence_length)

    def append(
        self,
        face: torch.Tensor,
        hand_l: torch.Tensor,
        hand_r: torch.Tensor,
        pose: torch.Tensor,
        *,
        detected_left: bool,
        detected_right: bool,
    ) -> None:
        self.face.append(face)
        self.hand_l.append(hand_l)
        self.hand_r.append(hand_r)
        self.pose.append(pose)
        self.miss_hl.append(detected_left)
        self.miss_hr.append(detected_right)

    def _pad_stream(self, stream: Deque[torch.Tensor]) -> torch.Tensor:
        if stream:
            stacked = torch.stack(list(stream), dim=0)
        else:
            stacked = torch.zeros((0, 3, self.image_size, self.image_size), dtype=torch.float32)

        if stacked.shape[0] == self.sequence_length:
            return stacked

        pad_frames = self.sequence_length - stacked.shape[0]
        if stacked.shape[0] == 0:
            return torch.zeros((pad_frames, 3, self.image_size, self.image_size), dtype=torch.float32)

        padding = torch.zeros((pad_frames, *stacked.shape[1:]), dtype=stacked.dtype, device=stacked.device)
        return torch.cat([stacked, padding], dim=0)

    def _pad_pose(self, stream: Deque[torch.Tensor]) -> torch.Tensor:
        if stream:
            stacked = torch.stack(list(stream), dim=0)
        else:
            stacked = torch.zeros((0, 3 * self.pose_landmarks), dtype=torch.float32)
        if stacked.shape[0] == self.sequence_length:
            return stacked
        pad_frames = self.sequence_length - stacked.shape[0]
        if stacked.shape[0] == 0:
            return torch.zeros((pad_frames, 3 * self.pose_landmarks), dtype=torch.float32)
        padding = torch.zeros((pad_frames, stacked.shape[1]), dtype=stacked.dtype, device=stacked.device)
        return torch.cat([stacked, padding], dim=0)

    def _pad_mask(self, stream: Deque[bool]) -> torch.Tensor:
        mask = torch.zeros(self.sequence_length, dtype=torch.bool)
        valid_len = len(stream)
        if valid_len:
            mask[:valid_len] = torch.tensor(list(stream), dtype=torch.bool)
        return mask

    def as_model_inputs(
        self,
        device: torch.device,
        *,
        backend: str = "torch",
    ) -> Optional[Dict[str, torch.Tensor | np.ndarray]]:
        if not self.face:
            return None

        face = self._pad_stream(self.face)
        hand_l = self._pad_stream(self.hand_l)
        hand_r = self._pad_stream(self.hand_r)
        pose = self._pad_pose(self.pose)

        pad_mask = torch.zeros(self.sequence_length, dtype=torch.bool)
        valid_len = len(self.face)
        if valid_len:
            pad_mask[:valid_len] = True

        miss_mask_hl = self._pad_mask(self.miss_hl)
        miss_mask_hr = self._pad_mask(self.miss_hr)

        if backend == "torch":
            return {
                "face": face.unsqueeze(0).to(device),
                "hand_l": hand_l.unsqueeze(0).to(device),
                "hand_r": hand_r.unsqueeze(0).to(device),
                "pose": pose.unsqueeze(0).to(device),
                "pad_mask": pad_mask.unsqueeze(0).to(device),
                "miss_mask_hl": miss_mask_hl.unsqueeze(0).to(device),
                "miss_mask_hr": miss_mask_hr.unsqueeze(0).to(device),
            }

        face_np = face.unsqueeze(0).cpu().numpy().astype(np.float32)
        hand_l_np = hand_l.unsqueeze(0).cpu().numpy().astype(np.float32)
        hand_r_np = hand_r.unsqueeze(0).cpu().numpy().astype(np.float32)
        pose_np = pose.unsqueeze(0).cpu().numpy().astype(np.float32)
        pad_mask_np = pad_mask.unsqueeze(0).cpu().numpy().astype(np.bool_)
        miss_hl_np = miss_mask_hl.unsqueeze(0).cpu().numpy().astype(np.bool_)
        miss_hr_np = miss_mask_hr.unsqueeze(0).cpu().numpy().astype(np.bool_)

        return {
            "face": face_np,
            "hand_l": hand_l_np,
            "hand_r": hand_r_np,
            "pose": pose_np,
            "pad_mask": pad_mask_np,
            "miss_mask_hl": miss_hl_np,
            "miss_mask_hr": miss_hr_np,
        }


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

def preprocess_crop(crop: np.ndarray) -> torch.Tensor:
    """Convierte un recorte BGR en un tensor normalizado CHW."""

    import cv2  # import local para evitar dependencias obligatorias

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(crop_rgb.transpose(2, 0, 1)).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def extract_pose_vector(result: object, count: int) -> torch.Tensor:
    """Devuelve un vector plano con ``count`` landmarks de pose."""

    pose = np.zeros((count, 3), dtype=np.float32)
    if result and getattr(result, "pose_landmarks", None):
        for idx, landmark in enumerate(result.pose_landmarks.landmark[:count]):
            pose[idx, 0] = landmark.x
            pose[idx, 1] = landmark.y
            pose[idx, 2] = landmark.visibility
    return torch.from_numpy(pose.reshape(-1))


class RoiTracker:
    """Mantiene un *bounding box* estable entre frames."""

    def __init__(self, *, smoothing: float = 0.4, max_misses: int = 5) -> None:
        self.smoothing = float(np.clip(smoothing, 0.0, 1.0))
        self.max_misses = max(0, int(max_misses))
        self._bbox: Optional[Tuple[int, int, int, int]] = None
        self._misses = 0

    @staticmethod
    def _blend(prev: Tuple[int, int, int, int], new: Tuple[int, int, int, int], alpha: float) -> Tuple[int, int, int, int]:
        if alpha <= 0.0:
            return prev
        if alpha >= 1.0:
            return new
        beta = 1.0 - alpha
        return tuple(int(round(beta * p + alpha * n)) for p, n in zip(prev, new))

    def update(self, bbox: Optional[Tuple[int, int, int, int]]) -> bool:
        if bbox is not None and bbox[2] > 0 and bbox[3] > 0:
            if self._bbox is None:
                self._bbox = bbox
            else:
                self._bbox = self._blend(self._bbox, bbox, self.smoothing)
            self._misses = 0
        else:
            self._misses += 1
            if self._misses > self.max_misses:
                self._bbox = None
        return self._bbox is not None

    @property
    def bbox(self) -> Optional[Tuple[int, int, int, int]]:
        return self._bbox

    @property
    def active(self) -> bool:
        return self._bbox is not None

    def extract_crop(self, frame: np.ndarray, out_size: int) -> np.ndarray:
        if not self._bbox:
            return np.zeros((out_size, out_size, 3), dtype=frame.dtype)
        x, y, w, h = self._bbox
        return crop_square(frame, x, y, w, h, out_size)


class HolisticFrameProcessor:
    """Procesa frames utilizando las salidas de MediaPipe Holistic."""

    def __init__(
        self,
        *,
        image_size: int,
        pose_landmarks: int,
        bbox_scale: float = 1.2,
        smoothing: float = 0.4,
        max_misses: int = 5,
    ) -> None:
        self.image_size = image_size
        self.pose_landmarks = pose_landmarks
        self.bbox_scale = bbox_scale
        self.face_tracker = RoiTracker(smoothing=smoothing, max_misses=max_misses)
        self.hand_left_tracker = RoiTracker(smoothing=smoothing, max_misses=max_misses)
        self.hand_right_tracker = RoiTracker(smoothing=smoothing, max_misses=max_misses)

    def _update_face(
        self,
        frame: np.ndarray,
        face_result: object,
    ) -> bool:
        if face_result and getattr(face_result, "multi_face_landmarks", None):
            face_landmarks = face_result.multi_face_landmarks[0]
            height, width = frame.shape[:2]
            xs = [int(landmark.x * width) for landmark in face_landmarks.landmark]
            ys = [int(landmark.y * height) for landmark in face_landmarks.landmark]
            x1 = max(0, min(xs))
            y1 = max(0, min(ys))
            x2 = min(width, max(xs))
            y2 = min(height, max(ys))
            bbox = expand_clamp_bbox(x1, y1, x2 - x1, y2 - y1, self.bbox_scale, width, height)
            return self.face_tracker.update(bbox)
        return self.face_tracker.update(None)

    def _update_hands(self, frame: np.ndarray, hands_result: object) -> None:
        if not hands_result:
            self.hand_left_tracker.update(None)
            self.hand_right_tracker.update(None)
            return

        landmarks_list = getattr(hands_result, "multi_hand_landmarks", None)
        handedness_list = getattr(hands_result, "multi_handedness", None)
        if not landmarks_list or not handedness_list:
            self.hand_left_tracker.update(None)
            self.hand_right_tracker.update(None)
            return

        height, width = frame.shape[:2]
        detected_left = False
        detected_right = False
        for landmarks, handedness in zip(landmarks_list, handedness_list):
            xs = [int(l.x * width) for l in landmarks.landmark]
            ys = [int(l.y * height) for l in landmarks.landmark]
            x1 = max(0, min(xs))
            y1 = max(0, min(ys))
            x2 = min(width, max(xs))
            y2 = min(height, max(ys))
            bbox = expand_clamp_bbox(x1, y1, x2 - x1, y2 - y1, self.bbox_scale, width, height)
            label = handedness.classification[0].label.lower()
            if label.startswith("left"):
                detected_left = self.hand_left_tracker.update(bbox) or detected_left
            else:
                detected_right = self.hand_right_tracker.update(bbox) or detected_right

        if not detected_left:
            self.hand_left_tracker.update(None)
        if not detected_right:
            self.hand_right_tracker.update(None)

    def process(
        self,
        frame: np.ndarray,
        *,
        face_result: object,
        hands_result: object,
        pose_result: object,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, FrameDetections, Dict[str, Optional[Tuple[int, int, int, int]]]]:
        self._update_face(frame, face_result)
        self._update_hands(frame, hands_result)

        face_crop = self.face_tracker.extract_crop(frame, self.image_size)
        hand_l_crop = self.hand_left_tracker.extract_crop(frame, self.image_size)
        hand_r_crop = self.hand_right_tracker.extract_crop(frame, self.image_size)

        face_tensor = preprocess_crop(face_crop)
        hand_l_tensor = preprocess_crop(hand_l_crop)
        hand_r_tensor = preprocess_crop(hand_r_crop)
        pose_tensor = extract_pose_vector(pose_result, self.pose_landmarks)

        detections = FrameDetections(
            face=self.face_tracker.active,
            hand_l=self.hand_left_tracker.active,
            hand_r=self.hand_right_tracker.active,
        )
        boxes = {
            "face": self.face_tracker.bbox,
            "hand_l": self.hand_left_tracker.bbox,
            "hand_r": self.hand_right_tracker.bbox,
        }
        return face_tensor, hand_l_tensor, hand_r_tensor, pose_tensor, detections, boxes
