#!/usr/bin/env python3
"""Real-time demo for the multi-stream SLT stub model.

This script captures frames from a webcam, extracts crops for the face and
hands using MediaPipe and feeds a temporal window to the :class:`MultiStreamSLT`
model. The decoder is a placeholder implemented via :class:`TextDecoderStub` and
therefore the textual output only represents token identifiers.
"""

from __future__ import annotations

import argparse
import contextlib
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch

try:  # pragma: no cover - optional dependency for the demo.
    import mediapipe as mp
except Exception:  # pragma: no cover - MediaPipe is optional.
    mp = None  # type: ignore

from slt.models import MultiStreamEncoder, TextDecoderStub, ViTConfig


@dataclass
class DemoConfig:
    """Hyper-parameters used by the demo application."""

    image_size: int = 224
    sequence_length: int = 32
    pose_landmarks: int = 13
    projector_dim: int = 256
    d_model: int = 512
    temporal_nhead: int = 8
    temporal_layers: int = 6
    temporal_dim_feedforward: int = 2048
    temporal_dropout: float = 0.1
    vocab_size: int = 32_000


class MultiStreamSLT(torch.nn.Module):
    """Thin wrapper combining the encoder and the placeholder decoder."""

    def __init__(self, config: DemoConfig) -> None:
        super().__init__()

        vit_config = ViTConfig(image_size=config.image_size)
        temporal_kwargs = {
            "nhead": config.temporal_nhead,
            "nlayers": config.temporal_layers,
            "dim_feedforward": config.temporal_dim_feedforward,
            "dropout": config.temporal_dropout,
        }

        self.encoder = MultiStreamEncoder(
            backbone_config=vit_config,
            projector_dim=config.projector_dim,
            d_model=config.d_model,
            pose_dim=3 * config.pose_landmarks,
            positional_num_positions=config.sequence_length,
            temporal_kwargs=temporal_kwargs,
        )
        self.decoder = TextDecoderStub(d_model=config.d_model, vocab_size=config.vocab_size)

    def forward(
        self,
        *,
        face: torch.Tensor,
        hand_l: torch.Tensor,
        hand_r: torch.Tensor,
        pose: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        miss_mask_hl: Optional[torch.Tensor] = None,
        miss_mask_hr: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded = self.encoder(
            face,
            hand_l,
            hand_r,
            pose,
            pad_mask=pad_mask,
            miss_mask_hl=miss_mask_hl,
            miss_mask_hr=miss_mask_hr,
        )
        decoder_mask = None
        if pad_mask is not None:
            decoder_mask = ~pad_mask.to(torch.bool)
        return self.decoder(encoded, padding_mask=decoder_mask)


class TemporalBuffer:
    """Maintains a temporal window of preprocessed frames for each stream."""

    def __init__(self, config: DemoConfig) -> None:
        self.config = config
        self.maxlen = config.sequence_length
        self.face: Deque[torch.Tensor] = deque(maxlen=self.maxlen)
        self.hand_l: Deque[torch.Tensor] = deque(maxlen=self.maxlen)
        self.hand_r: Deque[torch.Tensor] = deque(maxlen=self.maxlen)
        self.pose: Deque[torch.Tensor] = deque(maxlen=self.maxlen)
        self.miss_hl: Deque[bool] = deque(maxlen=self.maxlen)
        self.miss_hr: Deque[bool] = deque(maxlen=self.maxlen)

    def append(
        self,
        face: torch.Tensor,
        hand_l: torch.Tensor,
        hand_r: torch.Tensor,
        pose: torch.Tensor,
        *,
        miss_left: bool,
        miss_right: bool,
    ) -> None:
        self.face.append(face)
        self.hand_l.append(hand_l)
        self.hand_r.append(hand_r)
        self.pose.append(pose)
        self.miss_hl.append(miss_left)
        self.miss_hr.append(miss_right)

    def _pad_stream(self, stream: Deque[torch.Tensor]) -> torch.Tensor:
        if stream:
            stacked = torch.stack(list(stream), dim=0)
        else:
            stacked = torch.zeros(
                (0, 3, self.config.image_size, self.config.image_size), dtype=torch.float32
            )

        if stacked.shape[0] == self.maxlen:
            return stacked

        pad_frames = self.maxlen - stacked.shape[0]
        if stacked.shape[0] == 0:
            pad_tensor = torch.zeros(
                (pad_frames, 3, self.config.image_size, self.config.image_size), dtype=torch.float32
            )
            return pad_tensor

        padding = torch.zeros(
            (pad_frames, *stacked.shape[1:]), dtype=stacked.dtype, device=stacked.device
        )
        return torch.cat([stacked, padding], dim=0)

    def _pad_pose(self, stream: Deque[torch.Tensor]) -> torch.Tensor:
        if stream:
            stacked = torch.stack(list(stream), dim=0)
        else:
            stacked = torch.zeros((0, 3 * self.config.pose_landmarks), dtype=torch.float32)
        if stacked.shape[0] == self.maxlen:
            return stacked
        pad_frames = self.maxlen - stacked.shape[0]
        if stacked.shape[0] == 0:
            return torch.zeros((pad_frames, 3 * self.config.pose_landmarks), dtype=torch.float32)
        padding = torch.zeros((pad_frames, stacked.shape[1]), dtype=stacked.dtype, device=stacked.device)
        return torch.cat([stacked, padding], dim=0)

    def as_model_inputs(self, device: torch.device) -> Optional[Dict[str, torch.Tensor]]:
        if not self.face:
            return None

        face = self._pad_stream(self.face).unsqueeze(0).to(device)
        hand_l = self._pad_stream(self.hand_l).unsqueeze(0).to(device)
        hand_r = self._pad_stream(self.hand_r).unsqueeze(0).to(device)
        pose = self._pad_pose(self.pose).unsqueeze(0).to(device)

        pad_mask = torch.zeros(self.maxlen, dtype=torch.bool)
        miss_mask_hl = torch.zeros(self.maxlen, dtype=torch.bool)
        miss_mask_hr = torch.zeros(self.maxlen, dtype=torch.bool)

        valid_len = len(self.face)
        if valid_len > 0:
            pad_mask[:valid_len] = True
        if self.miss_hl:
            miss_mask_hl[: valid_len] = torch.tensor(list(self.miss_hl), dtype=torch.bool)
        if self.miss_hr:
            miss_mask_hr[: valid_len] = torch.tensor(list(self.miss_hr), dtype=torch.bool)

        return {
            "face": face,
            "hand_l": hand_l,
            "hand_r": hand_r,
            "pose": pose,
            "pad_mask": pad_mask.unsqueeze(0).to(device),
            "miss_mask_hl": miss_mask_hl.unsqueeze(0).to(device),
            "miss_mask_hr": miss_mask_hr.unsqueeze(0).to(device),
        }


def expand_clamp_bbox(x: int, y: int, w: int, h: int, scale: float, width: int, height: int) -> Tuple[int, int, int, int]:
    """Expand a bounding box around ``(x, y, w, h)`` and clamp to image bounds."""

    cx, cy = x + w / 2.0, y + h / 2.0
    w2, h2 = w * scale, h * scale
    x1 = int(max(0, cx - w2 / 2.0))
    y1 = int(max(0, cy - h2 / 2.0))
    x2 = int(min(width, cx + w2 / 2.0))
    y2 = int(min(height, cy + h2 / 2.0))
    return x1, y1, max(1, x2 - x1), max(1, y2 - y1)


def crop_square(image: np.ndarray, x: int, y: int, w: int, h: int, out_size: int) -> np.ndarray:
    """Extract a square crop centred on the bounding box."""

    height, width = image.shape[:2]
    side = max(w, h)
    cx, cy = x + w // 2, y + h // 2
    x1 = max(0, cx - side // 2)
    y1 = max(0, cy - side // 2)
    x2 = min(width, x1 + side)
    y2 = min(height, y1 + side)
    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((out_size, out_size, 3), dtype=image.dtype)
    return cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def preprocess_crop(crop: np.ndarray) -> torch.Tensor:
    """Convert a BGR crop into a normalized CHW tensor."""

    crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(crop_rgb.transpose(2, 0, 1)).float() / 255.0
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def extract_pose_vector(result: object, count: int) -> torch.Tensor:
    """Return a flattened pose vector with ``count`` landmarks."""

    pose = np.zeros((count, 3), dtype=np.float32)
    if result.pose_landmarks:
        for idx, landmark in enumerate(result.pose_landmarks.landmark[:count]):
            pose[idx, 0] = landmark.x
            pose[idx, 1] = landmark.y
            pose[idx, 2] = landmark.visibility
    return torch.from_numpy(pose.reshape(-1))


def decode_logits_stub(logits: torch.Tensor) -> str:
    """Placeholder decoding that maps the argmax token to a label string."""

    token_id = int(torch.argmax(logits, dim=-1).item())
    return f"<token_{token_id}>"


def run_demo(args: argparse.Namespace) -> None:
    if mp is None:
        raise RuntimeError(
            "MediaPipe no está disponible. Instala el paquete 'mediapipe' para ejecutar la demo."
        )

    config = DemoConfig(sequence_length=args.sequence_length, pose_landmarks=args.pose_landmarks)
    device = torch.device(args.device)

    model = MultiStreamSLT(config).to(device)
    model.eval()

    buffer = TemporalBuffer(config)

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara con índice {args.camera}")

    window_name = "MultiStream SLT Demo"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

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

    with closing_all(mediapipe_contexts):
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_result = face_mesh.process(rgb)
                hands_result = hands.process(rgb)
                pose_result = pose.process(rgb)

                face_crop = np.zeros((config.image_size, config.image_size, 3), dtype=frame.dtype)
                if face_result.multi_face_landmarks:
                    face_landmarks = face_result.multi_face_landmarks[0]
                    xs = [int(landmark.x * width) for landmark in face_landmarks.landmark]
                    ys = [int(landmark.y * height) for landmark in face_landmarks.landmark]
                    x1 = max(0, min(xs))
                    y1 = max(0, min(ys))
                    x2 = min(width, max(xs))
                    y2 = min(height, max(ys))
                    x, y, w, h = expand_clamp_bbox(x1, y1, x2 - x1, y2 - y1, 1.2, width, height)
                    face_crop = crop_square(frame, x, y, w, h, config.image_size)
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                left_crop = np.zeros_like(face_crop)
                right_crop = np.zeros_like(face_crop)
                left_detected = False
                right_detected = False
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
                        crop = crop_square(frame, x, y, w, h, config.image_size)
                        label = handedness.classification[0].label.lower()
                        if label.startswith("left"):
                            left_crop = crop
                            left_detected = True
                            color = (255, 0, 0)
                        else:
                            right_crop = crop
                            right_detected = True
                            color = (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

                pose_vec = extract_pose_vector(pose_result, config.pose_landmarks)

                face_tensor = preprocess_crop(face_crop)
                hand_l_tensor = preprocess_crop(left_crop)
                hand_r_tensor = preprocess_crop(right_crop)

                buffer.append(
                    face_tensor,
                    hand_l_tensor,
                    hand_r_tensor,
                    pose_vec,
                    miss_left=left_detected,
                    miss_right=right_detected,
                )

                inputs = buffer.as_model_inputs(device)
                if inputs is not None:
                    with torch.no_grad():
                        logits = model(
                            face=inputs["face"],
                            hand_l=inputs["hand_l"],
                            hand_r=inputs["hand_r"],
                            pose=inputs["pose"],
                            pad_mask=inputs["pad_mask"],
                            miss_mask_hl=inputs["miss_mask_hl"],
                            miss_mask_hr=inputs["miss_mask_hr"],
                        )
                    decoded = decode_logits_stub(logits)
                    cv2.putText(
                        frame,
                        decoded,
                        (10, height - 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                cv2.imshow(window_name, frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
        finally:
            cap.release()
            cv2.destroyAllWindows()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Índice de la cámara a utilizar")
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
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    run_demo(args)


if __name__ == "__main__":  # pragma: no cover - ejecución manual
    main()
