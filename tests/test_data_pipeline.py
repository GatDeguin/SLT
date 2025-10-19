import numpy as np
import pytest

from slt.data.lsa_t_multistream import LsaTMultiStream, collate_fn


@pytest.fixture()
def sample_dataset(tmp_path):
    root = tmp_path / "dataset"
    face = root / "face"
    hl = root / "hand_l"
    hr = root / "hand_r"
    pose_dir = root / "pose"
    for d in [face, hl, hr, pose_dir]:
        d.mkdir(parents=True)

    def _write_image(path):
        from PIL import Image

        arr = np.full((224, 224, 3), 127, dtype=np.uint8)
        Image.fromarray(arr).save(path)

    # Frames para video principal
    for idx in range(3):
        _write_image(face / f"vid1_f{idx:06d}.jpg")
        _write_image(hl / f"vid1_f{idx:06d}.jpg")
    # Mano derecha pierde último frame
    for idx in range(2):
        _write_image(hr / f"vid1_f{idx:06d}.jpg")

    pose = np.zeros((3, 17 * 3), dtype=np.float32)
    pose[:, 2::3] = 0.9
    np.savez_compressed(pose_dir / "vid1.npz", pose=pose)

    csv_path = root / "clips.csv"
    csv_path.write_text(
        "video_id;texto;fps;duration;frame_count\nvid1;hola;25;0.12;3\n",
        encoding="utf-8",
    )
    index_path = root / "index.csv"
    index_path.write_text("video_id\nvid1\n", encoding="utf-8")

    ds = LsaTMultiStream(
        face_dir=str(face),
        hand_l_dir=str(hl),
        hand_r_dir=str(hr),
        pose_dir=str(pose_dir),
        csv_path=str(csv_path),
        index_csv=str(index_path),
        T=4,
        lkp_count=17,
        flip_prob=0.0,
        enable_flip=False,
        quality_checks=True,
        quality_strict=False,
        fps_tolerance=2.0,
    )
    return ds


def test_dataset_sample_shapes(sample_dataset):
    sample = sample_dataset[0]

    assert sample.face.shape == (4, 3, 224, 224)
    assert sample.pose.shape[0] == 4
    assert sample.pose_conf_mask.shape == (4, sample_dataset.lkp_count)
    assert sample.pad_mask.sum().item() == sample.length.item()
    # Mano derecha pierde un frame, longitud efectiva 2
    assert sample.length.item() == 2
    assert sample.quality["effective_length"] == 2
    assert "hand_r" in sample.quality["missing_frames"]

    fps_info = sample.quality["fps"]
    assert fps_info["expected"] == 25
    assert fps_info["ok"] is True


def test_collate_includes_quality(sample_dataset):
    sample = sample_dataset[0]
    batch = collate_fn([sample, sample])

    assert batch["face"].shape[0] == 2
    assert len(batch["quality"]) == 2
    assert batch["quality"][0]["effective_length"] == 2


def test_quality_strict_raises(tmp_path, sample_dataset):
    # Reutilizamos data, pero activamos modo estricto
    ds = LsaTMultiStream(
        face_dir=str(tmp_path / "dataset" / "face"),
        hand_l_dir=str(tmp_path / "dataset" / "hand_l"),
        hand_r_dir=str(tmp_path / "dataset" / "hand_r"),
        pose_dir=str(tmp_path / "dataset" / "pose"),
        csv_path=str(tmp_path / "dataset" / "clips.csv"),
        index_csv=str(tmp_path / "dataset" / "index.csv"),
        T=4,
        lkp_count=17,
        flip_prob=0.0,
        enable_flip=False,
        quality_checks=True,
        quality_strict=True,
    )

    with pytest.raises(ValueError):
        _ = ds[0]
