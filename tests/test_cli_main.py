"""Integration tests for the ``python -m slt`` entrypoint."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

pytest.importorskip("numpy")
pytest.importorskip("PIL")

import numpy as np
from PIL import Image

from slt.training.loops import LoopResult


@pytest.fixture()
def cli_dataset(tmp_path: Path) -> SimpleNamespace:
    face_dir = tmp_path / "face"
    hand_l_dir = tmp_path / "hand_l"
    hand_r_dir = tmp_path / "hand_r"
    pose_dir = tmp_path / "pose"
    for directory in (face_dir, hand_l_dir, hand_r_dir, pose_dir):
        directory.mkdir(parents=True, exist_ok=True)

    video_id = "vid001"
    (tmp_path / "subs.csv").write_text("video_id;texto\nvid001;hola mundo\n", encoding="utf-8")
    (tmp_path / "train.csv").write_text("video_id\nvid001\n", encoding="utf-8")
    (tmp_path / "val.csv").write_text("video_id\nvid001\n", encoding="utf-8")

    frame = np.zeros((32, 32, 3), dtype="uint8")
    for directory in (face_dir, hand_l_dir, hand_r_dir):
        for idx in range(2):
            path = directory / f"{video_id}_f{idx:06d}.jpg"
            Image.fromarray(frame, mode="RGB").save(path)

    pose = np.random.RandomState(0).rand(3, 3 * 13).astype("float32")
    np.savez(pose_dir / f"{video_id}.npz", pose=pose)

    return SimpleNamespace(
        face_dir=face_dir,
        hand_l_dir=hand_l_dir,
        hand_r_dir=hand_r_dir,
        pose_dir=pose_dir,
        metadata_csv=tmp_path / "subs.csv",
        train_index=tmp_path / "train.csv",
        val_index=tmp_path / "val.csv",
    )


class _DummyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    model_max_length = 8

    def __len__(self) -> int:  # pragma: no cover - trivial
        return 16

    def __call__(
        self,
        texts,
        *,
        max_length,
        padding,
        truncation,
        return_tensors,
    ):
        batch_size = len(list(texts))
        input_ids = torch.zeros(batch_size, max_length, dtype=torch.long)
        attention_mask = torch.ones_like(input_ids)
        return {"input_ids": input_ids, "attention_mask": attention_mask}


class _DummyModel(nn.Module):
    def __init__(self, *_, **__):
        super().__init__()
        self.param = nn.Parameter(torch.zeros(1))

    def forward(self, *args, **kwargs):  # pragma: no cover - not invoked
        return torch.zeros(1)


def test_main_runs_training_loop(
    monkeypatch: pytest.MonkeyPatch, cli_dataset: SimpleNamespace, tmp_path: Path
) -> None:
    from slt import __main__ as slt_main

    outputs = {"train": [], "eval": [], "saves": []}

    args = SimpleNamespace(
        face_dir=cli_dataset.face_dir,
        hand_left_dir=cli_dataset.hand_l_dir,
        hand_right_dir=cli_dataset.hand_r_dir,
        pose_dir=cli_dataset.pose_dir,
        metadata_csv=cli_dataset.metadata_csv,
        train_index=cli_dataset.train_index,
        val_index=cli_dataset.val_index,
        work_dir=tmp_path / "work",
        batch_size=1,
        epochs=2,
        sequence_length=4,
        image_size=32,
        lr=1e-3,
        num_workers=0,
        no_pin_memory=True,
        device="cpu",
        seed=0,
        tokenizer="dummy",
        max_target_length=6,
        decoder_layers=1,
        decoder_heads=2,
        decoder_dropout=0.0,
        no_amp=True,
    )

    def fake_parse_args():
        return args

    monkeypatch.setattr(slt_main, "parse_args", fake_parse_args)
    monkeypatch.setattr(slt_main, "create_tokenizer", lambda _: _DummyTokenizer())
    monkeypatch.setattr(slt_main, "_DemoModel", _DummyModel)

    def fake_train_epoch(*args, **kwargs):
        outputs["train"].append(True)
        return LoopResult(loss=0.5 / len(outputs["train"]), metrics={})

    def fake_eval_epoch(*args, **kwargs):
        outputs["eval"].append(True)
        return LoopResult(loss=0.4 / len(outputs["eval"]), metrics={})

    monkeypatch.setattr(slt_main, "train_epoch", fake_train_epoch)
    monkeypatch.setattr(slt_main, "eval_epoch", fake_eval_epoch)

    def fake_save(state, path):
        outputs["saves"].append(Path(path))

    monkeypatch.setattr(torch, "save", fake_save)

    slt_main.main()

    assert len(outputs["train"]) == 2
    assert len(outputs["eval"]) == 2
    save_names = [path.name for path in outputs["saves"]]
    assert save_names.count("last.pt") == args.epochs
    assert save_names.count("best.pt") >= 1
