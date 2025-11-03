from __future__ import annotations

"""Tests for face fallback and preprocessing helpers in extract_rois_v2."""

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

try:  # pragma: no cover - entorno sin dependencias nativas
    import cv2  # type: ignore  # noqa: F401
except Exception:  # pragma: no cover - fallback de pruebas
    fake_cv2 = ModuleType("cv2")

    fake_cv2.COLOR_BGR2GRAY = 0
    fake_cv2.COLOR_GRAY2BGR = 1
    fake_cv2.COLOR_BGR2RGB = 2
    fake_cv2.INTER_LINEAR = 1

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

import tools.extract_rois_v2 as extract_rois_v2

from tools.extract_rois_v2 import (
    _normalise_format,
    _normalise_streams,
    apply_face_partial_grayscale,
    blur_face_preserve_eyes_mouth,
    build_face_keep_mask,
    KEYPOINT_BODY_LANDMARKS,
    KEYPOINT_FACE_LANDMARKS,
    KEYPOINT_HAND_LANDMARKS,
    KEYPOINT_LAYOUT_NAME,
    KEYPOINT_TOTAL_LANDMARKS,
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


@pytest.mark.parametrize("keypoints_format", ["npz", "npy"])
def test_process_video_exports_keypoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, keypoints_format: str
) -> None:
    frames = [np.full((8, 8, 3), 32, dtype=np.uint8), np.full((8, 8, 3), 48, dtype=np.uint8)]

    class _VideoCaptureStub:
        def __init__(self, *_: object, **__: object) -> None:
            self._frames = [frame.copy() for frame in frames]
            self._index = 0

        def isOpened(self) -> bool:
            return True

        def read(self) -> tuple[bool, np.ndarray]:
            if self._index < len(self._frames):
                frame = self._frames[self._index]
                self._index += 1
                return True, frame.copy()
            return False, np.empty((0, 0, 3), dtype=np.uint8)

        def release(self) -> None:
            return None

        def get(self, *_: object) -> float:
            return 25.0

    monkeypatch.setattr(extract_rois_v2.cv2, "VideoCapture", _VideoCaptureStub)
    monkeypatch.setattr(extract_rois_v2, "_ensure_mediapipe_available", lambda: True)

    def _landmarks(count: int, base_x: float, base_y: float, conf: float) -> list[SimpleNamespace]:
        return [
            SimpleNamespace(x=base_x + idx * 0.001, y=base_y + idx * 0.001, visibility=conf)
            for idx in range(count)
        ]

    face_landmarks = SimpleNamespace(
        landmark=_landmarks(KEYPOINT_FACE_LANDMARKS + 5, 0.3, 0.4, 0.0)
    )
    left_landmarks = SimpleNamespace(landmark=_landmarks(KEYPOINT_HAND_LANDMARKS, 0.6, 0.2, 0.0))
    right_landmarks = SimpleNamespace(landmark=_landmarks(KEYPOINT_HAND_LANDMARKS, 0.8, 0.2, 0.0))
    pose_landmarks = SimpleNamespace(
        landmark=_landmarks(KEYPOINT_BODY_LANDMARKS, 0.1, 0.2, 0.8)
    )

    frame_results = [
        (
            SimpleNamespace(multi_face_landmarks=[face_landmarks]),
            SimpleNamespace(
                multi_hand_landmarks=[left_landmarks, right_landmarks],
                multi_handedness=[
                    SimpleNamespace(classification=[SimpleNamespace(label="Left")]),
                    SimpleNamespace(classification=[SimpleNamespace(label="Right")]),
                ],
            ),
            SimpleNamespace(pose_landmarks=pose_landmarks),
        ),
        (
            SimpleNamespace(multi_face_landmarks=[]),
            SimpleNamespace(multi_hand_landmarks=[], multi_handedness=[]),
            SimpleNamespace(pose_landmarks=None),
        ),
    ]

    class _PipelineStub:
        def __init__(self) -> None:
            self._index = 0

        def __enter__(self) -> "_PipelineStub":
            return self

        def __exit__(self, *_: object) -> None:
            self.close()

        def close(self) -> None:
            return None

        def process(
            self, _: np.ndarray
        ) -> tuple[SimpleNamespace, SimpleNamespace, SimpleNamespace]:
            result = frame_results[min(self._index, len(frame_results) - 1)]
            self._index += 1
            return result

    monkeypatch.setattr(extract_rois_v2, "_create_pipeline", lambda *args, **kwargs: _PipelineStub())

    video_path = tmp_path / "sample.mp4"
    video_path.write_bytes(b"")

    metadata = extract_rois_v2.process_video(
        str(video_path),
        {},
        tmp_path / "pose",
        tmp_path / "keypoints",
        fps_target=25,
        streams={"pose"},
        export_keypoints=True,
        keypoints_format=keypoints_format,
    )

    assert metadata["success"] is True
    assert metadata["keypoints_frames"] == 2
    assert metadata["keypoints_format"] == keypoints_format
    keypoints_file = tmp_path / "keypoints" / f"{video_path.stem}.{keypoints_format}"
    assert keypoints_file.exists()

    if keypoints_format == "npz":
        loaded = np.load(keypoints_file)
        keypoints = loaded["keypoints"]
        assert loaded["layout"].item() == KEYPOINT_LAYOUT_NAME
    else:
        keypoints = np.load(keypoints_file)

    assert keypoints.shape == (2, KEYPOINT_TOTAL_LANDMARKS, 3)

    body_end = KEYPOINT_BODY_LANDMARKS
    face_end = body_end + KEYPOINT_FACE_LANDMARKS
    hand_l_end = face_end + KEYPOINT_HAND_LANDMARKS

    body_section = keypoints[0, :body_end]
    np.testing.assert_allclose(body_section[:3, 0], [0.1, 0.101, 0.102], atol=1e-6)
    np.testing.assert_allclose(body_section[:3, 1], [0.2, 0.201, 0.202], atol=1e-6)
    assert np.all(body_section[:, 2] > 0.7)

    face_section = keypoints[0, body_end:face_end]
    np.testing.assert_allclose(face_section[0, :2], [0.3, 0.4], atol=1e-6)
    np.testing.assert_allclose(
        face_section[-1, :2],
        [
            0.3 + (KEYPOINT_FACE_LANDMARKS - 1) * 0.001,
            0.4 + (KEYPOINT_FACE_LANDMARKS - 1) * 0.001,
        ],
        atol=1e-6,
    )
    np.testing.assert_allclose(face_section[:, 2], 1.0, atol=1e-6)

    left_section = keypoints[0, face_end:hand_l_end]
    right_section = keypoints[0, hand_l_end : hand_l_end + KEYPOINT_HAND_LANDMARKS]
    np.testing.assert_allclose(left_section[:, 2], 1.0, atol=1e-6)
    np.testing.assert_allclose(right_section[:, 2], 1.0, atol=1e-6)

    assert not keypoints[1].any()


def test_normalise_streams_accepts_aliases() -> None:
    resolved = _normalise_streams(["hands", "face"])
    assert resolved == {"hand_l", "hand_r", "face"}


def test_normalise_format_accepts_jpeg() -> None:
    assert _normalise_format("JPEG") == "jpg"


def test_resolve_delegate_models_uses_mediapipe_defaults(tmp_path, monkeypatch) -> None:
    package_root = tmp_path / "mediapipe_pkg"
    (package_root / "modules" / "face_landmarker").mkdir(parents=True)
    (package_root / "modules" / "hand_landmarker").mkdir(parents=True)
    (package_root / "modules" / "pose_landmarker").mkdir(parents=True)

    for relative in (
        ("modules", "face_landmarker", "face_landmarker.task"),
        ("modules", "hand_landmarker", "hand_landmarker.task"),
        ("modules", "pose_landmarker", "pose_landmarker_full.task"),
    ):
        target = package_root.joinpath(*relative)
        target.write_bytes(b"")

    init_file = package_root / "__init__.py"
    init_file.write_text("# dummy mediapipe package")

    fake_mp = SimpleNamespace(__file__=str(init_file))
    monkeypatch.setattr(extract_rois_v2, "mp", fake_mp)

    face_model, hand_model, pose_model = extract_rois_v2._resolve_delegate_models(
        extract_rois_v2._DELEGATE_GPU,
        None,
        None,
        None,
    )

    expected_face = package_root / "modules" / "face_landmarker" / "face_landmarker.task"
    expected_hand = package_root / "modules" / "hand_landmarker" / "hand_landmarker.task"
    expected_pose = package_root / "modules" / "pose_landmarker" / "pose_landmarker_full.task"

    assert face_model == str(expected_face)
    assert hand_model == str(expected_hand)
    assert pose_model == str(expected_pose)


def test_resolve_delegate_models_respects_custom_paths(tmp_path, monkeypatch) -> None:
    package_root = tmp_path / "mediapipe_pkg"
    package_root.mkdir()
    init_file = package_root / "__init__.py"
    init_file.write_text("# dummy mediapipe package")

    fake_mp = SimpleNamespace(__file__=str(init_file))
    monkeypatch.setattr(extract_rois_v2, "mp", fake_mp)

    face_override = tmp_path / "face_override.task"
    hand_override = tmp_path / "hand_override.task"
    pose_override = tmp_path / "pose_override.task"
    for path in (face_override, hand_override, pose_override):
        path.write_bytes(b"")

    face_model, hand_model, pose_model = extract_rois_v2._resolve_delegate_models(
        extract_rois_v2._DELEGATE_GPU,
        str(face_override),
        str(hand_override),
        str(pose_override),
    )

    assert face_model == str(face_override)
    assert hand_model == str(hand_override)
    assert pose_model == str(pose_override)


def test_normalise_streams_raises_on_unknown() -> None:
    with pytest.raises(ValueError):
        _normalise_streams(["unknown"])
