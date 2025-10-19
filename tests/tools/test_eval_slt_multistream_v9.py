"""Tests for :mod:`tools.eval_slt_multistream_v9`."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from typing import List

import pytest

torch = pytest.importorskip("torch")

from tools import eval_slt_multistream_v9 as eval_module


@dataclass
class DummyTokenizer:
    pad_token_id: int = 0
    eos_token_id: int = 1

    def batch_decode(self, sequences, skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = True) -> List[str]:
        mapping = {2: "uno", 3: "dos", 4: "tres", 5: "cuatro", 6: "cinco"}
        texts: List[str] = []
        for seq in sequences:
            tokens: List[str] = []
            for token in seq:
                if token == self.eos_token_id:
                    break
                if token == self.pad_token_id and skip_special_tokens:
                    continue
                tokens.append(mapping.get(int(token), f"<{int(token)}>") if skip_special_tokens else str(int(token)))
            texts.append(" ".join(tokens))
        return texts

    def __len__(self) -> int:  # pragma: no cover - not used but mimics HF API
        return 10


def test_decode_from_logits_greedy_and_beam():
    tokenizer = DummyTokenizer()
    logits = torch.tensor(
        [
            [
                [0.1, -1.0, 3.0, 0.5, 0.0, -0.5, -0.2],
                [0.2, 2.5, 0.2, 3.5, -1.0, -0.7, -0.4],
                [0.3, 4.0, -0.3, -0.5, 0.0, 0.1, -0.2],
            ],
            [
                [0.0, -2.0, 1.5, 0.1, 2.0, 1.0, -0.1],
                [0.2, -1.0, 0.5, 0.3, 1.7, 1.8, 0.0],
                [0.1, 3.5, -0.2, 0.4, 0.0, 0.1, -0.3],
            ],
        ],
        dtype=torch.float32,
    )

    greedy = eval_module._decode_from_logits(logits, tokenizer, num_beams=1)
    assert greedy == ["uno dos", "tres cuatro"]

    beam = eval_module._decode_from_logits(logits, tokenizer, num_beams=3)
    # Con beam > 1 debería mantener las mismas predicciones válidas
    assert beam == ["uno dos", "tres cuatro"]


def test_run_generates_predictions_and_metrics(tmp_path, monkeypatch):
    face_dir = tmp_path / "face"
    hand_l_dir = tmp_path / "hand_l"
    hand_r_dir = tmp_path / "hand_r"
    pose_dir = tmp_path / "pose"
    for directory in [face_dir, hand_l_dir, hand_r_dir, pose_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    metadata = tmp_path / "metadata.csv"
    metadata.write_text("video_id;texto\nvid1;hola mundo\nvid2;adios mundo\n", encoding="utf-8")

    eval_index = tmp_path / "index.csv"
    eval_index.write_text("video_id\nvid1\nvid2\n", encoding="utf-8")

    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_text("stub", encoding="utf-8")

    class FakeTokenizer(DummyTokenizer):
        def batch_decode(self, sequences, skip_special_tokens: bool = True, clean_up_tokenization_spaces: bool = True) -> List[str]:
            mapping = {2: "hola", 3: "mundo", 4: "adios"}
            texts: List[str] = []
            for seq in sequences:
                tokens: List[str] = []
                for token in seq:
                    if token == self.eos_token_id:
                        break
                    if token == self.pad_token_id and skip_special_tokens:
                        continue
                    tokens.append(mapping.get(int(token), ""))
                texts.append(" ".join(filter(None, tokens)))
            return texts

    class FakeModel:
        def to(self, device):  # pragma: no cover - passthrough
            return self

        def eval(self):  # pragma: no cover - passthrough
            return self

        def generate(self, **_: object):
            return torch.tensor(
                [
                    [2, 3, 1, 0],
                    [4, 3, 1, 0],
                ],
                dtype=torch.long,
            )

    batch = {
        "face": torch.zeros(2, 1, 3, 4, 4),
        "hand_l": torch.zeros(2, 1, 3, 4, 4),
        "hand_r": torch.zeros(2, 1, 3, 4, 4),
        "pose": torch.zeros(2, 1, 39),
        "pad_mask": torch.ones(2, 1, dtype=torch.bool),
        "miss_mask_hl": torch.ones(2, 1, dtype=torch.bool),
        "miss_mask_hr": torch.ones(2, 1, dtype=torch.bool),
        "texts": ["hola mundo", "adios mundo"],
        "video_ids": ["vid1", "vid2"],
    }

    monkeypatch.setattr(eval_module, "create_tokenizer", lambda _: FakeTokenizer())
    monkeypatch.setattr(eval_module, "_create_dataloader", lambda args: [batch])
    monkeypatch.setattr(eval_module, "_build_model", lambda args, tok: FakeModel())
    monkeypatch.setattr(eval_module, "_load_checkpoint", lambda model, path, device: None)

    output_csv = tmp_path / "predictions" / "preds.csv"

    argv = [
        "--face-dir",
        str(face_dir),
        "--hand-left-dir",
        str(hand_l_dir),
        "--hand-right-dir",
        str(hand_r_dir),
        "--pose-dir",
        str(pose_dir),
        "--metadata-csv",
        str(metadata),
        "--eval-index",
        str(eval_index),
        "--checkpoint",
        str(checkpoint),
        "--output-csv",
        str(output_csv),
        "--device",
        "cpu",
        "--tokenizer",
        "stub",
        "--sequence-length",
        "1",
    ]

    predictions = eval_module.run(argv)

    assert len(predictions) == 2
    assert [item.prediction for item in predictions] == ["hola mundo", "adios mundo"]

    assert output_csv.exists()
    with output_csv.open() as csv_file:
        rows = list(csv.reader(csv_file))
    assert rows[0] == ["video_id", "prediction"]
    assert rows[1:] == [["vid1", "hola mundo"], ["vid2", "adios mundo"]]

    metrics_json = output_csv.parent / "metrics.json"
    metrics_csv = output_csv.parent / "metrics.csv"
    assert metrics_json.exists()
    assert metrics_csv.exists()

    data = json.loads(metrics_json.read_text(encoding="utf-8"))
    assert data["bleu"] == pytest.approx(100.0)
    assert data["chrf"] == pytest.approx(100.0)

    with metrics_csv.open() as csv_file:
        reader = list(csv.reader(csv_file))
    assert reader[0] == ["metric", "value"]
    assert [row[0] for row in reader[1:]] == ["bleu", "chrf"]
    assert [float(row[1]) for row in reader[1:]] == [pytest.approx(100.0), pytest.approx(100.0)]
