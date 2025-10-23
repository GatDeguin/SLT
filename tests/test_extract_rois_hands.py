"""Pruebas unitarias para ``extract_rois_v2`` centradas en manos."""

from __future__ import annotations

from dataclasses import dataclass

from tools.extract_rois_v2 import hand_bbox_from_pose, resolve_hand_bbox


@dataclass
class _DummyLandmark:
    x: float
    y: float
    visibility: float = 1.0


@dataclass
class _DummyPose:
    landmark: list[_DummyLandmark]


def _build_pose(points: list[tuple[float, float]]) -> _DummyPose:
    landmarks: list[_DummyLandmark] = []
    for idx in range(33):
        if idx < len(points):
            x, y = points[idx]
            landmarks.append(_DummyLandmark(x, y))
        else:
            landmarks.append(_DummyLandmark(0.0, 0.0))
    return _DummyPose(landmark=landmarks)


def test_hand_bbox_from_pose_returns_square_box() -> None:
    pose = _build_pose([(0.0, 0.0)] * 33)
    pose.landmark[17] = _DummyLandmark(0.30, 0.40)
    pose.landmark[19] = _DummyLandmark(0.35, 0.45)
    pose.landmark[21] = _DummyLandmark(0.32, 0.43)

    bbox = hand_bbox_from_pose(pose, (17, 19, 21), 640, 480)

    assert bbox is not None
    x, y, w, h = bbox
    assert (x, y, w, h) == (189, 189, 38, 38)
    assert w == h


def test_resolve_hand_bbox_prefers_pose_over_previous() -> None:
    pose = _build_pose([(0.0, 0.0)] * 33)
    pose.landmark[18] = _DummyLandmark(0.60, 0.20)
    pose.landmark[20] = _DummyLandmark(0.62, 0.22)
    pose.landmark[22] = _DummyLandmark(0.64, 0.24)

    bbox, source = resolve_hand_bbox(
        detected_bbox=None,
        pose_landmarks=pose,
        indices=(18, 20, 22),
        prev_bbox=(10, 10, 15, 15),
        frame_width=640,
        frame_height=480,
    )

    assert source == "pose"
    assert bbox is not None
    assert bbox[2] == bbox[3]


def test_resolve_hand_bbox_uses_previous_when_no_pose() -> None:
    bbox, source = resolve_hand_bbox(
        detected_bbox=None,
        pose_landmarks=None,
        indices=(17, 19, 21),
        prev_bbox=(5, 5, 20, 20),
        frame_width=320,
        frame_height=240,
    )

    assert source == "previous"
    assert bbox == (5, 5, 20, 20)


def test_resolve_hand_bbox_black_when_no_sources() -> None:
    bbox, source = resolve_hand_bbox(
        detected_bbox=None,
        pose_landmarks=None,
        indices=(17, 19, 21),
        prev_bbox=None,
        frame_width=320,
        frame_height=240,
    )

    assert source == "black"
    assert bbox is None
