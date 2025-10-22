import random
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
pytest.importorskip("PIL")
from PIL import Image  # type: ignore

from slt.data import LsaTMultiStream, SampleItem, collate_fn


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> dict:
    random.seed(0)
    np.random.seed(0)

    face_dir = tmp_path / "face"
    hand_l_dir = tmp_path / "hand_l"
    hand_r_dir = tmp_path / "hand_r"
    pose_dir = tmp_path / "pose"
    keypoints_dir = tmp_path / "keypoints"
    for d in [face_dir, hand_l_dir, hand_r_dir, pose_dir, keypoints_dir]:
        d.mkdir(parents=True, exist_ok=True)

    video_id = "vid001"
    textos_path = tmp_path / "subs.csv"
    split_path = tmp_path / "split.csv"
    gloss_path = tmp_path / "gloss.csv"

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

    keypoints = np.zeros((6, 79, 3), dtype="float32")
    for frame in range(6):
        value = frame / 10.0
        keypoints[frame, :, 0] = value
        keypoints[frame, :, 1] = value + 0.5
        keypoints[frame, :, 2] = 0.9
    np.savez(keypoints_dir / f"{video_id}.npz", keypoints=keypoints)

    textos_path.write_text("video_id;texto\nvid001;hola mundo\n", encoding="utf-8")
    split_path.write_text("video_id\nvid001\n", encoding="utf-8")
    gloss_path.write_text("video_id;gloss;ctc_labels\nvid001;ga gb;1 2\n", encoding="utf-8")

    return {
        "face_dir": str(face_dir),
        "hand_l_dir": str(hand_l_dir),
        "hand_r_dir": str(hand_r_dir),
        "pose_dir": str(pose_dir),
        "keypoints_dir": str(keypoints_dir),
        "csv_path": str(textos_path),
        "index_csv": str(split_path),
        "gloss_csv": str(gloss_path),
    }


def test_sample_item_structure(synthetic_dataset: dict) -> None:
    ds = LsaTMultiStream(T=4, img_size=32, flip_prob=0.0, **synthetic_dataset)
    assert len(ds) == 1

    random.seed(0)
    sample = ds[0]
    assert isinstance(sample, SampleItem)
    assert sample.face.shape == (4, 3, 32, 32)
    assert sample.hand_l.shape == (4, 3, 32, 32)
    assert sample.hand_r.shape == (4, 3, 32, 32)
    assert sample.pose.shape == (4, 39)
    assert sample.pose_conf_mask.shape == (4, 13)
    assert sample.pose_conf_mask.dtype == torch.bool
    assert sample.pad_mask.dtype == torch.bool
    assert torch.equal(sample.pad_mask, torch.ones(4, dtype=torch.bool))
    assert sample.length.dtype == torch.long
    assert sample.length.item() == 4
    assert sample.miss_mask_hr.sum() == 0
    assert sample.miss_mask_hl.sum() == 4
    assert sample.keypoints.shape == (4, 79, 3)
    assert sample.keypoints_mask.shape == (4, 79)
    assert sample.keypoints_body.shape[1] == 33
    assert sample.keypoints_hand_l.shape[1] == 21
    assert sample.keypoints_hand_r.shape[1] == 21
    assert sample.keypoints_face.shape[1] == 4
    assert sample.keypoints_lengths.shape == (5,)
    assert sample.ctc_labels.dtype == torch.long
    assert sample.ctc_mask.dtype == torch.bool
    assert sample.gloss_text == "ga gb"
    assert sample.gloss_sequence == ["ga", "gb"]
    assert sample.text == "hola mundo"
    assert sample.video_id == "vid001"


def test_collate_fn_outputs(synthetic_dataset: dict) -> None:
    ds = LsaTMultiStream(T=4, img_size=32, flip_prob=0.0, **synthetic_dataset)
    random.seed(1)
    batch = [ds[0], ds[0]]
    data = collate_fn(batch)

    assert data["face"].shape == (2, 4, 3, 32, 32)
    assert data["pose"].shape == (2, 4, 39)
    assert data["pose_conf_mask"].shape == (2, 4, 13)
    assert data["pose_conf_mask"].dtype == torch.bool
    assert data["pad_mask"].dtype == torch.bool
    assert torch.equal(data["lengths"], torch.tensor([4, 4], dtype=torch.long))
    assert data["texts"] == ["hola mundo", "hola mundo"]
    assert data["keypoints"].shape == (2, 4, 79, 3)
    assert data["keypoints_lengths"].shape == (2, 5)
    assert data["ctc_labels"].shape[0] == 2
    assert data["gloss_sequences"][0] == ["ga", "gb"]
    assert data["video_ids"] == ["vid001", "vid001"]


def test_pad_mask_marks_padding_when_clip_shorter(synthetic_dataset: dict) -> None:
    ds = LsaTMultiStream(T=8, img_size=32, flip_prob=0.0, **synthetic_dataset)
    sample = ds[0]

    expected_mask = torch.tensor([1, 1, 1, 1, 1, 0, 0, 0], dtype=torch.bool)
    assert torch.equal(sample.pad_mask, expected_mask)
    assert sample.length.item() == 5


class DummyTokenizer:
    model_max_length = 16

    def __call__(
        self,
        text_list,
        *,
        max_length,
        padding,
        truncation,
        return_tensors,
    ):
        batch_size = len(text_list)
        input_ids = torch.arange(batch_size * max_length, dtype=torch.long).view(batch_size, max_length)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_cli_collate_propagates_masks_and_lengths(synthetic_dataset: dict) -> None:
    from slt.__main__ import _build_collate

    ds = LsaTMultiStream(T=8, img_size=32, flip_prob=0.0, **synthetic_dataset)
    sample = ds[0]
    collate = _build_collate(DummyTokenizer(), max_length=6)

    batch = collate([sample])

    inputs = batch["inputs"]
    assert torch.equal(inputs["pad_mask"], sample.pad_mask.unsqueeze(0))
    assert torch.equal(inputs["encoder_attention_mask"], sample.pad_mask.unsqueeze(0).to(torch.long))
    assert torch.equal(inputs["lengths"], sample.length.unsqueeze(0))
    assert torch.equal(inputs["pose_conf_mask"], sample.pose_conf_mask.unsqueeze(0))
    assert torch.equal(inputs["keypoints_mask"], sample.keypoints_mask.unsqueeze(0))
    assert inputs["gloss_sequences"][0] == sample.gloss_sequence


def test_training_collate_includes_pose_mask(synthetic_dataset: dict) -> None:
    from slt.training.data import build_collate

    ds = LsaTMultiStream(T=6, img_size=32, flip_prob=0.0, **synthetic_dataset)
    sample = ds[0]
    tokenizer = DummyTokenizer()
    collate = build_collate(tokenizer, max_length=5)

    batch = collate([sample])
    inputs = batch["inputs"]
    assert torch.equal(inputs["pose_conf_mask"], sample.pose_conf_mask.unsqueeze(0))
    assert torch.equal(inputs["keypoints"], sample.keypoints.unsqueeze(0))
    assert torch.equal(inputs["ctc_labels"], sample.ctc_labels.unsqueeze(0))


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

    expected_kp_hand_l = sample_no.keypoints_hand_r.clone()
    expected_kp_hand_l[:, :, 0] = 1.0 - expected_kp_hand_l[:, :, 0]
    expected_kp_hand_r = sample_no.keypoints_hand_l.clone()
    expected_kp_hand_r[:, :, 0] = 1.0 - expected_kp_hand_r[:, :, 0]


def test_keypoints_sampling_tracks_frame_indices(synthetic_dataset: dict) -> None:
    ds = LsaTMultiStream(T=4, img_size=32, flip_prob=0.0, **synthetic_dataset)
    sample = ds[0]

    expected = torch.tensor([0.0, 0.2, 0.4, 0.5], dtype=torch.float32)
    assert torch.allclose(sample.keypoints[:, 0, 0], expected, atol=1e-6)

    frame_mask = sample.keypoints_mask.any(dim=1)
    assert torch.equal(sample.keypoints_frame_mask, frame_mask)
    assert sample.keypoints_lengths[0].item() == int(frame_mask.sum().item())


def test_keypoint_view_masks_track_lengths(synthetic_dataset: dict) -> None:
    ds = LsaTMultiStream(T=4, img_size=32, flip_prob=0.0, **synthetic_dataset)
    sample = ds[0]

    views = [
        (sample.keypoints_body_mask, sample.keypoints_body_frame_mask, sample.keypoints_lengths[1]),
        (sample.keypoints_hand_l_mask, sample.keypoints_hand_l_frame_mask, sample.keypoints_lengths[2]),
        (sample.keypoints_hand_r_mask, sample.keypoints_hand_r_frame_mask, sample.keypoints_lengths[3]),
        (sample.keypoints_face_mask, sample.keypoints_face_frame_mask, sample.keypoints_lengths[4]),
    ]

    for mask, frame_mask, length in views:
        if mask.numel() == 0:
            assert length.item() == 0
            assert not frame_mask.any()
            continue
        expected = mask.any(dim=1)
        assert torch.equal(frame_mask, expected)
        assert length.item() == int(expected.sum().item())


def test_pose_low_confidence_zeroing(synthetic_dataset: dict) -> None:
    pose_path = Path(synthetic_dataset["pose_dir"]) / "vid001.npz"

    low_conf_pose = np.zeros((3, 3 * 13), dtype="float32")
    # Frame 0: first landmark below threshold, second above
    low_conf_pose[0, 0:3] = [0.2, 0.3, 0.1]
    low_conf_pose[0, 3:6] = [0.4, 0.5, 0.8]
    # Frame 1: both above threshold
    low_conf_pose[1, 0:3] = [0.6, 0.7, 0.6]
    low_conf_pose[1, 3:6] = [0.8, 0.2, 0.9]
    # Frame 2: first below, second above
    low_conf_pose[2, 0:3] = [0.9, 0.1, 0.2]
    low_conf_pose[2, 3:6] = [0.3, 0.4, 0.95]

    np.savez(pose_path, pose=low_conf_pose)

    ds = LsaTMultiStream(T=3, img_size=32, min_conf=0.5, flip_prob=0.0, **synthetic_dataset)
    random.seed(0)
    sample = ds[0]

    pose = sample.pose.view(3, 13, 3)
    mask = sample.pose_conf_mask

    # Frame 0: first landmark filtered, coordinates zeroed but confidence preserved
    assert torch.allclose(pose[0, 0, :2], torch.zeros(2))
    assert pytest.approx(pose[0, 0, 2].item()) == 0.1
    assert mask[0, 0].item() is False
    assert torch.allclose(pose[0, 1, :2], torch.tensor([0.4, 0.5]))
    assert pytest.approx(pose[0, 1, 2].item()) == 0.8
    assert mask[0, 1].item() is True

    # Frame 1: both valid
    assert mask[1, 0].item() is True
    assert mask[1, 1].item() is True
    assert torch.allclose(pose[1, 0, :2], torch.tensor([0.6, 0.7]))

    # Frame 2: first invalid again, second valid
    assert mask[2, 0].item() is False
    assert mask[2, 1].item() is True
    assert torch.allclose(pose[2, 1, :2], torch.tensor([0.3, 0.4]))


def test_missing_metadata_columns_raise(tmp_path: Path) -> None:
    face_dir = tmp_path / "face"
    hand_l_dir = tmp_path / "hand_l"
    hand_r_dir = tmp_path / "hand_r"
    pose_dir = tmp_path / "pose"
    for directory in (face_dir, hand_l_dir, hand_r_dir, pose_dir):
        directory.mkdir(parents=True, exist_ok=True)

    csv_path = tmp_path / "subs.csv"
    csv_path.write_text("video_id;text\nvid001;hola\n", encoding="utf-8")
    index_path = tmp_path / "index.csv"
    index_path.write_text("identifier\nvid001\n", encoding="utf-8")

    with pytest.raises(ValueError):
        LsaTMultiStream(
            str(face_dir),
            str(hand_l_dir),
            str(hand_r_dir),
            str(pose_dir),
            str(csv_path),
            str(index_path),
        )


def test_keypoint_translation_updates_masks(synthetic_dataset: dict) -> None:
    keypoint_path = Path(synthetic_dataset["keypoints_dir"]) / "vid001.npz"
    shifted = np.zeros((4, 79, 3), dtype="float32")
    shifted[:, :, 0] = 0.8
    shifted[:, :, 1] = 0.2
    shifted[:, :, 2] = 0.9
    np.savez(keypoint_path, keypoints=shifted)

    ds = LsaTMultiStream(
        T=4,
        img_size=32,
        flip_prob=0.0,
        keypoint_translate_range=(0.5, 0.5, 0.0, 0.0),
        **synthetic_dataset,
    )

    random.seed(0)
    sample = ds[0]

    assert sample.keypoints_mask.sum().item() == 0
    assert not bool(sample.keypoints_frame_mask.any())
    assert sample.keypoints_lengths[0].item() == 0
    assert not bool(sample.keypoints_body_mask.any())
    assert not bool(sample.keypoints_hand_l_mask.any())
    assert not bool(sample.keypoints_hand_r_mask.any())
    assert not bool(sample.keypoints_face_mask.any())


def test_keypoint_resample_affects_sampling_sequence(synthetic_dataset: dict) -> None:
    keypoint_path = Path(synthetic_dataset["keypoints_dir"]) / "vid001.npz"
    frames = 6
    pattern = np.zeros((frames, 79, 3), dtype="float32")
    for idx in range(frames):
        value = idx / (frames - 1)
        pattern[idx, :, 0] = value
        pattern[idx, :, 1] = 0.5
        pattern[idx, :, 2] = 1.0
    np.savez(keypoint_path, keypoints=pattern)

    ds_baseline = LsaTMultiStream(
        T=4,
        img_size=32,
        flip_prob=0.0,
        **synthetic_dataset,
    )
    ds_resampled = LsaTMultiStream(
        T=4,
        img_size=32,
        flip_prob=0.0,
        keypoint_resample_range=(0.5, 0.5),
        **synthetic_dataset,
    )

    random.seed(1)
    sample_default = ds_baseline[0]
    random.seed(1)
    sample_resampled = ds_resampled[0]

    seq_default = sample_default.keypoints[:, 0, 0]
    seq_resampled = sample_resampled.keypoints[:, 0, 0]

    assert seq_resampled.shape == seq_default.shape == (4,)
    assert not torch.allclose(seq_default, seq_resampled)
    assert pytest.approx(seq_resampled[0].item()) == seq_default[0].item()
    assert pytest.approx(seq_resampled[-1].item()) == 1.0
    assert torch.all(seq_resampled[1:] >= seq_resampled[:-1])
    assert sample_resampled.keypoints_lengths[0].item() == int(
        sample_resampled.keypoints_frame_mask.sum().item()
    )


def test_spatial_transforms_keep_valid_points_in_bounds(synthetic_dataset: dict) -> None:
    keypoint_path = Path(synthetic_dataset["keypoints_dir"]) / "vid001.npz"
    base = np.zeros((5, 79, 3), dtype="float32")
    base[:, :, 0] = 0.4
    base[:, :, 1] = 0.6
    base[:, :, 2] = 0.9
    np.savez(keypoint_path, keypoints=base)

    ds = LsaTMultiStream(
        T=4,
        img_size=32,
        flip_prob=0.0,
        keypoint_scale_range=(1.1, 1.1),
        keypoint_translate_range=(-0.1, -0.1, 0.05, 0.05),
        keypoint_rotate_range=(30.0, 30.0),
        **synthetic_dataset,
    )

    random.seed(2)
    sample = ds[0]

    assert sample.keypoints.shape == (4, 79, 3)
    mask = sample.keypoints_mask
    assert bool(mask.any())
    coords = sample.keypoints[:, :, :2][mask]
    assert torch.all((coords >= 0.0) & (coords <= 1.0))
    assert sample.keypoints_lengths[0].item() == int(sample.keypoints_frame_mask.sum().item())
