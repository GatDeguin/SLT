from __future__ import annotations

"""Tests for face fallback and preprocessing helpers in extract_rois_v2."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np

try:  # pragma: no cover - entorno sin dependencias nativas
    import cv2  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback de pruebas
    fake_cv2 = ModuleType("cv2")

    fake_cv2.COLOR_BGR2GRAY = 0
    fake_cv2.COLOR_GRAY2BGR = 1
    fake_cv2.COLOR_BGR2RGB = 2

    def _ensure_uint8(arr: np.ndarray) -> np.ndarray:
        return np.clip(arr, 0, 255).astype(np.uint8)

    def _cvt_color(image: np.ndarray, code: int) -> np.ndarray:
        if code == fake_cv2.COLOR_BGR2GRAY:
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("BGR input required")
            weights = np.array([0.114, 0.587, 0.299], dtype=np.float32)
            gray = np.tensordot(image.astype(np.float32), weights, axes=([2], [0]))
            return _ensure_uint8(gray)
        if code == fake_cv2.COLOR_GRAY2BGR:
            if image.ndim != 2:
                raise ValueError("Gray input required")
            return np.stack([image, image, image], axis=-1)
        if code == fake_cv2.COLOR_BGR2RGB:
            if image.ndim != 3 or image.shape[2] != 3:
                raise ValueError("BGR input required")
            return image[..., ::-1]
        raise NotImplementedError(f"Unsupported code {code}")

    def _gaussian_blur(image: np.ndarray, ksize: tuple[int, int], sigma: float) -> np.ndarray:
        kernel_x, kernel_y = ksize
        pad_x = kernel_x // 2
        pad_y = kernel_y // 2
        kernel = np.ones((kernel_y, kernel_x), dtype=np.float32)
        kernel /= kernel.sum()

        if image.ndim == 2:
            padded = np.pad(image.astype(np.float32), ((pad_y, pad_y), (pad_x, pad_x)), mode="edge")
            out = np.zeros_like(image, dtype=np.float32)
            for row in range(image.shape[0]):
                for col in range(image.shape[1]):
                    region = padded[row : row + kernel_y, col : col + kernel_x]
                    out[row, col] = float(np.sum(region * kernel))
            return _ensure_uint8(out)

        if image.ndim == 3:
            channels = [_gaussian_blur(image[:, :, idx], ksize, sigma) for idx in range(image.shape[2])]
            return np.stack(channels, axis=-1)

        raise ValueError("Imagen no soportada para blur")

    def _merge(channels: list[np.ndarray]) -> np.ndarray:
        return np.stack(channels, axis=-1)

    def _bitwise_not(image: np.ndarray) -> np.ndarray:
        return np.bitwise_not(image)

    def _bitwise_and(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.bitwise_and(a, b)

    def _add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return _ensure_uint8(a.astype(np.int32) + b.astype(np.int32))

    def _circle(image: np.ndarray, center: tuple[int, int], radius: int, color: int, thickness: int) -> None:
        cx, cy = center
        yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        image[mask] = color

    def _resize(image: np.ndarray, size: tuple[int, int], interpolation: int | None = None) -> np.ndarray:
        out_w, out_h = size
        if image.ndim == 2:
            channels = [image]
        else:
            channels = [image[:, :, idx] for idx in range(image.shape[2])]
        resized_channels = []
        for channel in channels:
            ys = np.linspace(0, channel.shape[0] - 1, out_h)
            xs = np.linspace(0, channel.shape[1] - 1, out_w)
            grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
            coords_y = np.clip(np.round(grid_y).astype(int), 0, channel.shape[0] - 1)
            coords_x = np.clip(np.round(grid_x).astype(int), 0, channel.shape[1] - 1)
            resized_channels.append(channel[coords_y, coords_x])
        stacked = np.stack(resized_channels, axis=-1)
        if image.ndim == 2:
            return stacked[:, :, 0]
        return stacked.astype(image.dtype)

    class _VideoCapture:
        def __init__(self, *args: object, **kwargs: object) -> None:
            self._opened = False

        def isOpened(self) -> bool:
            return self._opened

        def read(self) -> tuple[bool, np.ndarray]:
            return False, np.empty((0, 0, 3), dtype=np.uint8)

        def release(self) -> None:
            return None

        def get(self, *_: object) -> float:
            return 0.0

    def _imwrite(*_: object, **__: object) -> bool:
        return True

    fake_cv2.cvtColor = _cvt_color
    fake_cv2.GaussianBlur = _gaussian_blur
    fake_cv2.merge = _merge
    fake_cv2.bitwise_not = _bitwise_not
    fake_cv2.bitwise_and = _bitwise_and
    fake_cv2.add = _add
    fake_cv2.circle = _circle
    fake_cv2.resize = _resize
    fake_cv2.VideoCapture = _VideoCapture
    fake_cv2.imwrite = _imwrite
    fake_cv2.CAP_PROP_FPS = 5

    sys.modules["cv2"] = fake_cv2

from tools.extract_rois_v2 import (
    apply_face_partial_grayscale,
    blur_face_preserve_eyes_mouth,
    build_face_keep_mask,
    resolve_face_bbox,
)


class _DummyLandmark(SimpleNamespace):
    def __init__(self, x: float, y: float) -> None:
        super().__init__(x=x, y=y)


def _make_pose_landmarks(points: list[tuple[float, float]]) -> SimpleNamespace:
    return SimpleNamespace(landmark=[_DummyLandmark(x, y) for x, y in points])


def test_resolve_face_bbox_reuses_previous_before_pose() -> None:
    width, height = 128, 96
    prev_bbox = (20, 16, 30, 28)
    pose_points = [
        (-0.1, 0.05),
        (0.2, 0.05),
        (0.4, 0.1),
        (0.6, 0.15),
        (0.8, 0.2),
        (0.9, 0.3),
        (1.0, 0.4),
        (1.1, 0.5),
        (0.7, 0.55),
        (0.5, 0.6),
        (0.3, 0.65),
    ]
    pose_landmarks = _make_pose_landmarks(pose_points)

    bbox_prev, source_prev = resolve_face_bbox(
        None,
        pose_landmarks,
        prev_bbox,
        width,
        height,
    )
    assert bbox_prev == prev_bbox
    assert source_prev == "previous"

    bbox_pose, source_pose = resolve_face_bbox(
        None,
        pose_landmarks,
        None,
        width,
        height,
    )
    assert source_pose == "pose"
    assert bbox_pose is not None
    x, y, w, h = bbox_pose
    assert 0 <= x <= width
    assert 0 <= y <= height
    assert w > 0 and h > 0
    assert x + w <= width
    assert y + h <= height


def test_face_partial_grayscale_preserves_mask_regions() -> None:
    frame_height, frame_width = 60, 60
    bbox = (20, 20, 20, 20)
    # Build landmarks covering eyes and mouth so the mask has a colored core.
    center_norm = (bbox[0] + bbox[2] / 2) / frame_width
    center_point = _DummyLandmark(center_norm, center_norm)
    # Populate enough entries to satisfy the keep indices set.
    face_landmarks = SimpleNamespace(landmark=[center_point for _ in range(400)])

    mask = build_face_keep_mask(face_landmarks, bbox, (frame_height, frame_width), keep_radius=3)
    assert mask.shape == (bbox[3], bbox[2])
    assert np.any(mask == 255)
    assert np.any(mask == 0)

    h, w = bbox[3], bbox[2]
    x_vals = np.linspace(0, 255, w, dtype=np.uint8)
    y_vals = np.linspace(255, 0, h, dtype=np.uint8)
    red = np.tile(x_vals, (h, 1))
    green = np.tile(y_vals[:, None], (1, w))
    blue = ((red.astype(np.int32) + green.astype(np.int32)) // 2).astype(np.uint8)
    patch = np.stack([blue, green, red], axis=-1)

    gray_patch = apply_face_partial_grayscale(patch, mask)
    outside = mask == 0
    assert np.all(gray_patch[outside][:, 0] == gray_patch[outside][:, 1])
    assert np.all(gray_patch[outside][:, 1] == gray_patch[outside][:, 2])

    inside = mask == 255
    assert np.all(gray_patch[inside] == patch[inside])

    blurred = blur_face_preserve_eyes_mouth(gray_patch, mask)
    assert np.all(blurred[inside] == gray_patch[inside])
    assert np.any(blurred[outside] != gray_patch[outside])
    assert np.all(blurred[outside][:, 0] == blurred[outside][:, 1])
    assert np.all(blurred[outside][:, 1] == blurred[outside][:, 2])
