"""Smoke tests for the unified CLI workflow with MSKA enabled."""

from __future__ import annotations
"""Smoke tests for the unified CLI workflow with MSKA enabled."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from slt.training.loops import LoopResult
from tools import train_slt_multistream_v9 as train_module
from tools import _pretrain_dino as pretrain_module
from tests._synthetic import SyntheticDatasetSpec, generate_multistream_dataset


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __init__(self, vocab_size: int, max_length: int) -> None:
        self.vocab_size = vocab_size
        self.model_max_length = max_length

    def __len__(self) -> int:  # pragma: no cover - compatibility shim
        return self.vocab_size

    def __call__(
        self,
        texts,
        *,
        max_length: int,
        padding: str,
        truncation: bool,
        return_tensors: str,
    ):
        del padding, truncation, return_tensors
        batch = list(texts)
        input_ids = torch.full((len(batch), max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, text in enumerate(batch):
            tokens = max(1, len(text.split()))
            length = min(max_length, tokens)
            input_ids[row, : length - 1] = 2
            input_ids[row, length - 1] = self.eos_token_id
            attention_mask[row, :length] = 1
        return {"input_ids": input_ids, "attention_mask": attention_mask}

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        tokens = []
        for token in token_ids:
            if token == self.eos_token_id:
                break
            if skip_special_tokens and token == self.pad_token_id:
                continue
            tokens.append(f"tok{int(token)}")
        return " ".join(tokens)


class DummyModel(torch.nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.dummy = torch.nn.Parameter(torch.zeros(1))
        self.encoder = torch.nn.Sequential(torch.nn.Linear(1, 1))
        self.encoder.mska_encoder = torch.nn.Linear(1, 1)  # type: ignore[attr-defined]
        self.decoder = torch.nn.Linear(1, vocab_size)


def _run_epoch(
    loader,
    loss_fn,
    *,
    device,
    metrics,
    vocab_size: int,
    ctc_vocab: int,
) -> LoopResult:
    batch = next(iter(loader))
    inputs = batch["inputs"]
    targets = batch["targets"]

    assert "keypoints" in inputs
    assert inputs["keypoints"].shape[-2:] == (79, 3)
    assert inputs["ctc_labels"].shape[0] == inputs["face"].shape[0]

    if isinstance(device, str):
        device = torch.device(device)

    batch_size = targets["translation"].shape[0]
    target_len = targets["translation"].shape[1]
    time_steps = 3

    outputs = SimpleNamespace(
        logits=torch.zeros(batch_size, target_len, vocab_size, device=device),
        auxiliary={
            "fused": {
                "logits": torch.zeros(batch_size, time_steps, ctc_vocab, device=device),
                "mask": torch.ones(batch_size, time_steps, dtype=torch.bool, device=device),
            },
            "stream": {
                "pose": torch.zeros(batch_size, time_steps, ctc_vocab, device=device),
            },
            "frame_masks": {
                "pose": torch.ones(batch_size, time_steps, dtype=torch.bool, device=device),
            },
            "distillation": {
                "pose": torch.zeros(batch_size, time_steps, ctc_vocab, device=device),
            },
        },
    )

    loss, components = loss_fn(outputs, targets)
    metrics_out = {
        "loss_translation_weighted": float(
            components["loss_translation_weighted"].detach().cpu().item()
        )
    }

    if "loss_ctc_weighted" in components:
        metrics_out["loss_ctc_weighted"] = float(
            components["loss_ctc_weighted"].detach().cpu().item()
        )
    if "loss_distillation_weighted" in components:
        metrics_out["loss_distillation_weighted"] = float(
            components["loss_distillation_weighted"].detach().cpu().item()
        )

    if metrics:
        for name, metric_fn in metrics.items():
            metrics_out[name] = float(metric_fn(outputs, targets["translation"]))

    return LoopResult(float(loss.detach().cpu().item()), metrics_out)


def test_unified_cli_pipeline_with_mska(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset_spec = SyntheticDatasetSpec(
        sequence_length=4,
        image_size=16,
        pose_landmarks=5,
        frames_per_video=6,
        num_train=2,
        num_val=1,
    )
    data_root = tmp_path / "data"
    paths = generate_multistream_dataset(data_root, dataset_spec)

    work_dir = tmp_path / "work"
    tokenizer = TinyTokenizer(vocab_size=8, max_length=6)
    ctc_vocab = 6

    config = {
        "data": {
            "face_dir": str(paths.face_dir),
            "hand_left_dir": str(paths.hand_left_dir),
            "hand_right_dir": str(paths.hand_right_dir),
            "pose_dir": str(paths.pose_dir),
            "keypoints_dir": str(paths.keypoints_dir),
            "metadata_csv": str(paths.metadata_csv),
            "train_index": str(paths.train_index),
            "val_index": str(paths.val_index),
            "gloss_csv": str(paths.gloss_csv),
            "work_dir": str(work_dir),
            "batch_size": 1,
            "val_batch_size": 1,
            "num_workers": 0,
            "device": "cpu",
            "precision": "fp32",
            "tokenizer": "stub",
            "max_target_length": 6,
        },
        "model": {
            "image_size": dataset_spec.image_size,
            "projector_dim": 8,
            "d_model": 16,
            "pose_landmarks": dataset_spec.pose_landmarks,
            "sequence_length": dataset_spec.sequence_length,
            "decoder_layers": 1,
            "decoder_heads": 2,
            "decoder_dropout": 0.0,
            "pretrained": "none",
            "use_mska": True,
            "mska_ctc_vocab": ctc_vocab,
            "mska_translation_weight": 0.7,
            "mska_ctc_weight": 0.2,
            "mska_distillation_weight": 0.1,
            "mska_distillation_temperature": 2.0,
            "mska_gloss_hidden_dim": 12,
            "mska_gloss_activation": "leaky_relu",
            "mska_gloss_dropout": 0.1,
            "mska_gloss_fusion": "add",
        },
        "optim": {
            "lr": 1e-3,
            "label_smoothing": 0.0,
        },
        "training": {
            "epochs": 1,
        },
    }

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    monkeypatch.setattr(train_module, "create_tokenizer", lambda _: tokenizer)
    monkeypatch.setattr(
        train_module,
        "MultiStreamClassifier",
        lambda *args, **kwargs: DummyModel(vocab_size=tokenizer.vocab_size),
    )

    def _train_epoch(model, loader, optimizer, loss_fn, *, device, metrics=None, **kwargs):
        return _run_epoch(
            loader,
            loss_fn,
            device=device,
            metrics=metrics,
            vocab_size=tokenizer.vocab_size,
            ctc_vocab=ctc_vocab,
        )

    def _eval_epoch(model, loader, loss_fn, *, device, metrics=None, **kwargs):
        return _run_epoch(
            loader,
            loss_fn,
            device=device,
            metrics=metrics,
            vocab_size=tokenizer.vocab_size,
            ctc_vocab=ctc_vocab,
        )

    monkeypatch.setattr(train_module, "train_epoch", _train_epoch)
    monkeypatch.setattr(train_module, "eval_epoch", _eval_epoch)

    saved_paths: list[Path] = []

    def _fake_save_checkpoint(path: Path, **kwargs) -> None:
        saved_paths.append(path)

    monkeypatch.setattr(train_module, "_save_checkpoint", _fake_save_checkpoint)

    def _validate_paths(config) -> None:
        required = [
            ("face_dir", "Face directory"),
            ("hand_left_dir", "Left hand directory"),
            ("hand_right_dir", "Right hand directory"),
            ("pose_dir", "Pose directory"),
            ("metadata_csv", "Metadata CSV"),
            ("train_index", "Train index CSV"),
            ("val_index", "Validation index CSV"),
        ]
        for attr, kind in required:
            path_obj = Path(getattr(config, attr))
            if not path_obj.exists():
                raise FileNotFoundError(f"{kind} not found: {path_obj}")
            setattr(config, attr, path_obj)

        if config.keypoints_dir:
            keypoints_path = Path(config.keypoints_dir)
            if not keypoints_path.exists():
                raise FileNotFoundError(f"Keypoints directory not found: {keypoints_path}")
            config.keypoints_dir = keypoints_path
        if config.gloss_csv:
            gloss_path = Path(config.gloss_csv)
            if not gloss_path.exists():
                raise FileNotFoundError(f"Gloss CSV not found: {gloss_path}")
            config.gloss_csv = gloss_path

        work_dir_path = Path(config.work_dir)
        work_dir_path.mkdir(parents=True, exist_ok=True)
        config.work_dir = work_dir_path

    monkeypatch.setattr(train_module, "_validate_paths", _validate_paths)

    monkeypatch.setattr(sys, "argv", ["train", "--config", str(config_path)])
    train_module.main()

    assert saved_paths
    assert any(path.name == "last.pt" for path in saved_paths)
    assert work_dir.exists()


def test_pretrain_cli_minimal(tmp_path: Path) -> None:
    data_dir = tmp_path / "images"
    output_dir = tmp_path / "outputs"
    data_dir.mkdir()
    output_dir.mkdir()

    argv = [
        "--train-dir",
        str(data_dir),
        "--output-dir",
        str(output_dir),
        "--koleo-weight",
        "0.25",
        "--koleo-epsilon",
        "5e-4",
    ]

    args = pretrain_module._parse_args(argv, default_stream="face")

    assert args.koleo_weight == pytest.approx(0.25)
    assert args.koleo_epsilon == pytest.approx(5e-4)
