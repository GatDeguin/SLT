import csv
import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("PIL")
from PIL import Image  # type: ignore

np = pytest.importorskip("numpy")
torch = pytest.importorskip("torch")
tokenizers = pytest.importorskip("tokenizers")
transformers = pytest.importorskip("transformers")

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture()
def synthetic_dataset(tmp_path: Path) -> dict:
    np.random.seed(0)

    face_dir = tmp_path / "face"
    hand_l_dir = tmp_path / "hand_l"
    hand_r_dir = tmp_path / "hand_r"
    pose_dir = tmp_path / "pose"
    for directory in [face_dir, hand_l_dir, hand_r_dir, pose_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    video_id = "vid001"
    textos_path = tmp_path / "subs.csv"
    split_path = tmp_path / "split.csv"

    def save_frame(directory: Path, idx: int) -> None:
        arr = (np.random.rand(32, 32, 3) * 255).astype("uint8")
        Image.fromarray(arr, mode="RGB").save(directory / f"{video_id}_f{idx:06d}.jpg")

    for i in range(4):
        save_frame(face_dir, i)
        save_frame(hand_l_dir, i)
        save_frame(hand_r_dir, i)

    pose = np.random.rand(4, 3 * 13).astype("float32")
    np.savez(pose_dir / f"{video_id}.npz", pose=pose)

    textos_path.write_text("video_id;texto\nvid001;hola mundo\n", encoding="utf-8")
    split_path.write_text("video_id\nvid001\n", encoding="utf-8")

    return {
        "face_dir": face_dir,
        "hand_l_dir": hand_l_dir,
        "hand_r_dir": hand_r_dir,
        "pose_dir": pose_dir,
        "metadata_csv": textos_path,
        "index_csv": split_path,
    }


@pytest.fixture()
def tiny_tokenizer_dir(tmp_path: Path) -> Path:
    save_dir = tmp_path / "tokenizer"
    save_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = tokenizers.Tokenizer(tokenizers.models.WordLevel({"<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "hola": 4, "mundo": 5}, unk_token="<unk>"))
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    tokenizer_path = save_dir / "tokenizer.json"
    tokenizer.save(str(tokenizer_path))

    hf_tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_path),
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
    )
    hf_tokenizer.save_pretrained(save_dir)
    return save_dir


def _load_eval_module():
    module_path = PROJECT_ROOT / "tools" / "eval_slt_multistream_v9.py"
    spec = importlib.util.spec_from_file_location("eval_slt_multistream_v9", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_eval_script_generates_stub_csv(
    tmp_path: Path, synthetic_dataset: dict, tiny_tokenizer_dir: Path
) -> None:
    module = _load_eval_module()

    config = module.ModelConfig(
        image_size=32,
        projector_dim=64,
        d_model=128,
        pose_landmarks=13,
        projector_dropout=0.0,
        fusion_dropout=0.0,
        temporal_nhead=2,
        temporal_layers=1,
        temporal_dim_feedforward=128,
        temporal_dropout=0.0,
        sequence_length=4,
        decoder_layers=1,
        decoder_heads=2,
        decoder_dropout=0.0,
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(str(tiny_tokenizer_dir))
    class DummyEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, face, hand_l, hand_r, pose, *, pad_mask=None, miss_mask_hl=None, miss_mask_hr=None):
            batch, time = face.shape[:2]
            return torch.zeros(batch, time, config.d_model, device=face.device)

    patch = pytest.MonkeyPatch()
    patch.setattr(module, "MultiStreamEncoder", lambda *args, **kwargs: DummyEncoder())
    model = module.MultiStreamClassifier(config, tokenizer)

    checkpoint_path = tmp_path / "checkpoint.pt"
    torch.save({"model_state": model.state_dict()}, checkpoint_path)

    output_csv = tmp_path / "predicciones.csv"

    args = [
        "--face-dir",
        str(synthetic_dataset["face_dir"]),
        "--hand-left-dir",
        str(synthetic_dataset["hand_l_dir"]),
        "--hand-right-dir",
        str(synthetic_dataset["hand_r_dir"]),
        "--pose-dir",
        str(synthetic_dataset["pose_dir"]),
        "--metadata-csv",
        str(synthetic_dataset["metadata_csv"]),
        "--eval-index",
        str(synthetic_dataset["index_csv"]),
        "--checkpoint",
        str(checkpoint_path),
        "--output-csv",
        str(output_csv),
        "--image-size",
        "32",
        "--projector-dim",
        "64",
        "--d-model",
        "128",
        "--pose-landmarks",
        "13",
        "--temporal-nhead",
        "2",
        "--temporal-layers",
        "1",
        "--temporal-dim-feedforward",
        "128",
        "--temporal-dropout",
        "0.0",
        "--sequence-length",
        "4",
        "--decoder-layers",
        "1",
        "--decoder-heads",
        "2",
        "--decoder-dropout",
        "0.0",
        "--tokenizer",
        str(tiny_tokenizer_dir),
        "--max-target-length",
        "6",
        "--batch-size",
        "1",
        "--device",
        "cpu",
        "--no-pin-memory",
    ]

    exit_code = module.main(args)
    assert exit_code == 0
    assert output_csv.exists()

    with output_csv.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        rows = list(reader)

    assert rows[0] == ["video_id", "prediction", "reference", "latency_ms"]
    assert len(rows) == 2
    assert rows[1][0] == "vid001"
    assert isinstance(rows[1][1], str)
    assert rows[1][2] == "hola mundo"
    assert float(rows[1][3]) >= 0.0
    patch.undo()
