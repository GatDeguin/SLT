"""Utilities to generate synthetic multi-stream datasets for tests and CI."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import numpy as np
from PIL import Image

__all__ = [
    "SyntheticDatasetPaths",
    "SyntheticDatasetSpec",
    "generate_multistream_dataset",
]


@dataclass
class SyntheticDatasetSpec:
    """Specification for a tiny multi-stream dataset."""

    sequence_length: int
    image_size: int
    pose_landmarks: int
    frames_per_video: int
    num_train: int
    num_val: int
    fps: float = 25.0
    base_text: str = "hola mundo"
    seed: int = 1234

    def total_videos(self) -> int:
        return self.num_train + self.num_val


@dataclass
class SyntheticDatasetPaths:
    """Paths generated for the synthetic dataset."""

    face_dir: Path
    hand_left_dir: Path
    hand_right_dir: Path
    pose_dir: Path
    keypoints_dir: Path
    metadata_csv: Path
    train_index: Path
    val_index: Path
    gloss_csv: Path
    video_ids: Sequence[str]


def _write_image(path: Path, image_size: int, value: int) -> None:
    arr = np.full((image_size, image_size, 3), value, dtype=np.uint8)
    Image.fromarray(arr, mode="RGB").save(path)


def _write_pose(path: Path, frames: int, landmarks: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    pose = rng.random((frames, 3 * landmarks), dtype=np.float32)
    conf = rng.uniform(0.7, 1.0, size=(frames, landmarks)).astype(np.float32)
    pose[:, 2::3] = conf.reshape(frames, landmarks)
    np.savez_compressed(
        path,
        pose=pose,
        pose_norm=np.asarray("signing_space_v1", dtype=np.str_),
    )


def _write_keypoints(path: Path, frames: int, seed: int, total_landmarks: int = 79) -> None:
    rng = np.random.default_rng(seed)
    coords = rng.random((frames, total_landmarks, 3), dtype=np.float32)
    conf = rng.uniform(0.7, 1.0, size=(frames, total_landmarks)).astype(np.float32)
    coords[:, :, 2] = conf
    np.savez_compressed(path, keypoints=coords)


def _video_ids(prefix: str, count: int) -> List[str]:
    return [f"{prefix}{idx:03d}" for idx in range(1, count + 1)]


def generate_multistream_dataset(root: Path, spec: SyntheticDatasetSpec) -> SyntheticDatasetPaths:
    """Create a dataset layout that matches :class:`LsaTMultiStream` expectations."""

    face_dir = root / "face"
    hand_left_dir = root / "hand_l"
    hand_right_dir = root / "hand_r"
    pose_dir = root / "pose"
    keypoints_dir = root / "keypoints"
    for directory in (face_dir, hand_left_dir, hand_right_dir, pose_dir, keypoints_dir):
        directory.mkdir(parents=True, exist_ok=True)

    video_ids = _video_ids("vid", spec.total_videos())
    frame_indices = range(spec.frames_per_video)

    for vid_idx, video_id in enumerate(video_ids):
        intensity = int(40 + 10 * vid_idx)
        for frame in frame_indices:
            frame_tag = f"{video_id}_f{frame:06d}.jpg"
            _write_image(face_dir / frame_tag, spec.image_size, intensity)
            _write_image(hand_left_dir / frame_tag, spec.image_size, intensity + 5)
            _write_image(hand_right_dir / frame_tag, spec.image_size, intensity + 10)
        _write_pose(
            pose_dir / f"{video_id}.npz",
            spec.frames_per_video,
            spec.pose_landmarks,
            seed=spec.seed + vid_idx,
        )
        _write_keypoints(
            keypoints_dir / f"{video_id}.npz",
            spec.frames_per_video,
            seed=spec.seed + 100 + vid_idx,
        )

    metadata_rows: List[str] = ["video_id;texto;fps;duration;frame_count"]
    for video_id in video_ids:
        duration = spec.frames_per_video / spec.fps
        metadata_rows.append(f"{video_id};{spec.base_text};{spec.fps};{duration:.6f};{spec.frames_per_video}")

    metadata_csv = root / "metadata.csv"
    metadata_csv.write_text("\n".join(metadata_rows) + "\n", encoding="utf-8")

    train_ids = video_ids[: spec.num_train]
    val_ids = video_ids[spec.num_train : spec.num_train + spec.num_val]

    def _write_index(path: Path, ids: Iterable[str]) -> None:
        path.write_text("video_id\n" + "\n".join(ids) + ("\n" if ids else ""), encoding="utf-8")

    train_index = root / "train.csv"
    val_index = root / "val.csv"
    _write_index(train_index, train_ids)
    _write_index(val_index, val_ids)

    gloss_rows = ["video_id;gloss;ctc_labels"]
    for idx, video_id in enumerate(video_ids, start=1):
        gloss = f"g{idx}a g{idx}b"
        labels = " ".join(str(n) for n in range(1, 1 + len(gloss.split())))
        gloss_rows.append(f"{video_id};{gloss};{labels}")
    gloss_csv = root / "gloss.csv"
    gloss_csv.write_text("\n".join(gloss_rows) + "\n", encoding="utf-8")

    return SyntheticDatasetPaths(
        face_dir=face_dir,
        hand_left_dir=hand_left_dir,
        hand_right_dir=hand_right_dir,
        pose_dir=pose_dir,
        keypoints_dir=keypoints_dir,
        metadata_csv=metadata_csv,
        train_index=train_index,
        val_index=val_index,
        gloss_csv=gloss_csv,
        video_ids=video_ids,
    )
