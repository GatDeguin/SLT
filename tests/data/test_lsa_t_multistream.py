import random
from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("numpy")
pytest.importorskip("PIL")

import torch
import numpy as np
from PIL import Image

from slt.data import LsaTMultiStream, SampleItem, collate_fn


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> dict:
    random.seed(0)
    np.random.seed(0)

    face_dir = tmp_path / "face"
    hand_l_dir = tmp_path / "hand_l"
    hand_r_dir = tmp_path / "hand_r"
    pose_dir = tmp_path / "pose"
    for d in [face_dir, hand_l_dir, hand_r_dir, pose_dir]:
        d.mkdir(parents=True, exist_ok=True)

    video_id = "vid001"
    textos_path = tmp_path / "subs.csv"
    split_path = tmp_path / "split.csv"

    def save_frame(directory: Path, idx: int) -> None:
        arr = (np.random.rand(32, 32, 3) * 255).astype("uint8")
        Image.fromarray(arr, mode="RGB").save(directory / f"{video_id}_f{idx:06d}.jpg")

    for i in range(5):
        save_frame(face_dir, i)
    for i in range(3):
        save_frame(hand_l_dir, i)
    # sin mano derecha para probar máscara de ausencia

    pose = np.random.rand(7, 3 * 13).astype("float32")
    np.savez(pose_dir / f"{video_id}.npz", pose=pose)

    textos_path.write_text("video_id;texto\nvid001;hola mundo\n", encoding="utf-8")
    split_path.write_text("video_id\nvid001\n", encoding="utf-8")

    return {
        "face_dir": str(face_dir),
        "hand_l_dir": str(hand_l_dir),
        "hand_r_dir": str(hand_r_dir),
        "pose_dir": str(pose_dir),
        "csv_path": str(textos_path),
        "index_csv": str(split_path),
    }


def test_sample_item_structure(synthetic_dataset: dict) -> None:
    ds = LsaTMultiStream(T=4, img_size=32, **synthetic_dataset)
    assert len(ds) == 1

    random.seed(0)
    sample = ds[0]
    assert isinstance(sample, SampleItem)
    assert sample.face.shape == (4, 3, 32, 32)
    assert sample.hand_l.shape == (4, 3, 32, 32)
    assert sample.hand_r.shape == (4, 3, 32, 32)
    assert sample.pose.shape == (4, 39)
    assert sample.pad_mask.dtype == torch.bool
    assert sample.miss_mask_hr.sum() == 0
    assert sample.miss_mask_hl.sum() == 4
    assert sample.text == "hola mundo"
    assert sample.video_id == "vid001"


def test_collate_fn_outputs(synthetic_dataset: dict) -> None:
    ds = LsaTMultiStream(T=4, img_size=32, **synthetic_dataset)
    random.seed(1)
    batch = [ds[0], ds[0]]
    data = collate_fn(batch)

    assert data["face"].shape == (2, 4, 3, 32, 32)
    assert data["pose"].shape == (2, 4, 39)
    assert data["pad_mask"].dtype == torch.bool
    assert data["texts"] == ["hola mundo", "hola mundo"]
    assert data["video_ids"] == ["vid001", "vid001"]


def test_forced_flip_swaps_streams_and_pose(synthetic_dataset: dict) -> None:
    ds_no_flip = LsaTMultiStream(T=4, img_size=32, flip_prob=0.0, **synthetic_dataset)
    ds_flip = LsaTMultiStream(T=4, img_size=32, flip_prob=1.0, **synthetic_dataset)

    random.seed(42)
    sample_no = ds_no_flip[0]
    random.seed(42)
    sample_flip = ds_flip[0]

    assert torch.allclose(sample_flip.face, torch.flip(sample_no.face, dims=[3]))
    assert torch.allclose(sample_flip.hand_l, torch.flip(sample_no.hand_r, dims=[3]))
    assert torch.allclose(sample_flip.hand_r, torch.flip(sample_no.hand_l, dims=[3]))
    assert torch.equal(sample_flip.miss_mask_hl, sample_no.miss_mask_hr)
    assert torch.equal(sample_flip.miss_mask_hr, sample_no.miss_mask_hl)

    expected_pose = ds_flip._flip_pose_tensor(sample_no.pose)
    assert torch.allclose(sample_flip.pose, expected_pose)
