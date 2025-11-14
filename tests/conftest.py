"""Test configuration ensuring project root is importable and cv2 is stubbed."""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from pathlib import Path
from types import ModuleType

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _install_fake_cv2() -> None:
    """Install a lightweight OpenCV stub for environments without cv2."""

    fake_cv2 = ModuleType("cv2")
    fake_cv2.__spec__ = ModuleSpec("cv2", loader=None)

    # Constants consumed by the ROI extractor utilities.
    fake_cv2.COLOR_BGR2GRAY = 0
    fake_cv2.COLOR_GRAY2BGR = 1
    fake_cv2.COLOR_BGR2RGB = 2
    fake_cv2.INTER_LINEAR = 1
    fake_cv2.CAP_PROP_FPS = 0

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
            return image[..., ::-1].copy()
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
            channels = [
                _gaussian_blur(image[:, :, idx], ksize, sigma)
                for idx in range(image.shape[2])
            ]
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

    def _circle(
        image: np.ndarray,
        center: tuple[int, int],
        radius: int,
        color: int,
        thickness: int,
    ) -> None:
        cx, cy = center
        yy, xx = np.ogrid[: image.shape[0], : image.shape[1]]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2
        image[mask] = color

    def _resize(
        image: np.ndarray,
        size: tuple[int, int],
        interpolation: int | None = None,
    ) -> np.ndarray:
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

        def release(self) -> None:  # pragma: no cover - no side effects
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

    sys.modules["cv2"] = fake_cv2


try:  # pragma: no cover - depends on optional native dependency
    import cv2  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback for CI environments
    _install_fake_cv2()
