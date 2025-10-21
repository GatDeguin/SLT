"""End-to-end smoke tests spanning data loading, training and export."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Tuple

import numpy as np
import pytest
import torch
from torch import nn
from transformers.tokenization_utils_base import BatchEncoding

onnx = pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")

from slt.data.lsa_t_multistream import LsaTMultiStream
from slt.training.configuration import ModelConfig
from slt.training.data import create_dataloader
from slt.training.loops import LoopResult, eval_epoch, train_epoch
from slt.training.models import MultiStreamClassifier
from tools.export_onnx_encoder_v9 import EncoderExportModule, main_export

from tests._synthetic import SyntheticDatasetSpec, generate_multistream_dataset


class _TinyTokenizer:
    """Minimal tokenizer compatible with :mod:`slt.training.data`."""

    def __init__(self, vocab_size: int, max_length: int) -> None:
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.model_max_length = max_length

    def __len__(self) -> int:  # pragma: no cover - trivial
        return self.vocab_size

    def __call__(
        self,
        texts: Iterable[str],
        *,
        max_length: int,
        padding: str,
        truncation: bool,
        return_tensors: str,
    ) -> BatchEncoding:
        del padding, truncation, return_tensors  # parameters required by protocol
        batch = list(texts)
        input_ids = torch.full((len(batch), max_length), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, text in enumerate(batch):
            length = min(max_length, max(1, len(text.split())))
            input_ids[row, : length - 1] = 2
            input_ids[row, length - 1] = self.eos_token_id
            attention_mask[row, :length] = 1
        return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})


@pytest.mark.parametrize(
    "dataset_spec, training_cfg",
    [
        (
            SyntheticDatasetSpec(
                sequence_length=4,
                image_size=16,
                pose_landmarks=5,
                frames_per_video=6,
                num_train=2,
                num_val=1,
                base_text="hola uno",
            ),
            {
                "batch_size": 1,
                "target_length": 6,
                "projector_dim": 8,
                "d_model": 16,
                "temporal_nhead": 2,
                "temporal_dim_feedforward": 32,
                "decoder_heads": 2,
                "lr": 5e-3,
                "epochs": 4,
            },
        ),
        (
            SyntheticDatasetSpec(
                sequence_length=6,
                image_size=24,
                pose_landmarks=7,
                frames_per_video=8,
                num_train=3,
                num_val=1,
                base_text="hola dos",
            ),
            {
                "batch_size": 2,
                "target_length": 8,
                "projector_dim": 12,
                "d_model": 24,
                "temporal_nhead": 3,
                "temporal_dim_feedforward": 48,
                "decoder_heads": 3,
                "lr": 1e-2,
                "epochs": 5,
            },
        ),
        (
            SyntheticDatasetSpec(
                sequence_length=5,
                image_size=20,
                pose_landmarks=6,
                frames_per_video=5,
                num_train=2,
                num_val=1,
                base_text="hola tres",
            ),
            {
                "batch_size": 1,
                "target_length": 6,
                "projector_dim": 10,
                "d_model": 16,
                "temporal_nhead": 2,
                "temporal_dim_feedforward": 32,
                "decoder_heads": 2,
                "lr": 7e-3,
                "epochs": 3,
            },
        ),
    ],
    ids=["tiny-seq4", "tiny-seq6", "tiny-non-multiple"],
)
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_synthetic_pipeline_roundtrip(
    tmp_path: Path, dataset_spec: SyntheticDatasetSpec, training_cfg: Mapping[str, float]
) -> None:
    assert dataset_spec.image_size % 14 != 0
    data_root = tmp_path / "dataset"
    paths = generate_multistream_dataset(data_root, dataset_spec)

    train_dataset = LsaTMultiStream(
        face_dir=str(paths.face_dir),
        hand_l_dir=str(paths.hand_left_dir),
        hand_r_dir=str(paths.hand_right_dir),
        pose_dir=str(paths.pose_dir),
        csv_path=str(paths.metadata_csv),
        index_csv=str(paths.train_index),
        T=dataset_spec.sequence_length,
        img_size=dataset_spec.image_size,
        lkp_count=dataset_spec.pose_landmarks,
        flip_prob=0.0,
        enable_flip=False,
        quality_checks=True,
        quality_strict=False,
    )

    tokenizer = _TinyTokenizer(vocab_size=32, max_length=training_cfg["target_length"])
    loader = create_dataloader(
        train_dataset,
        batch_size=training_cfg["batch_size"],
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        tokenizer=tokenizer,
        max_length=training_cfg["target_length"],
    )

    model_config = ModelConfig(
        image_size=dataset_spec.image_size,
        projector_dim=training_cfg["projector_dim"],
        d_model=training_cfg["d_model"],
        pose_landmarks=dataset_spec.pose_landmarks,
        projector_dropout=0.0,
        fusion_dropout=0.0,
        temporal_nhead=training_cfg["temporal_nhead"],
        temporal_layers=1,
        temporal_dim_feedforward=training_cfg["temporal_dim_feedforward"],
        temporal_dropout=0.0,
        sequence_length=dataset_spec.sequence_length,
        decoder_layers=1,
        decoder_heads=training_cfg["decoder_heads"],
        decoder_dropout=0.0,
        pretrained="none",
    )

    torch.manual_seed(0)
    model = MultiStreamClassifier(model_config, tokenizer)
    optimizer = torch.optim.AdamW(model.parameters(), lr=training_cfg["lr"])

    def _loss_fn(outputs, targets):
        del targets
        if isinstance(outputs, nn.Module):  # pragma: no cover - defensive
            raise TypeError("Loss function expected output tensor-like result")
        if hasattr(outputs, "loss") and outputs.loss is not None:
            return outputs.loss
        if isinstance(outputs, torch.Tensor):
            return outputs.mean()
        raise TypeError("Unsupported output type for loss computation")

    initial = eval_epoch(model, loader, _loss_fn, device="cpu")
    assert isinstance(initial, LoopResult)

    for _ in range(training_cfg["epochs"]):
        train_result = train_epoch(model, loader, optimizer, _loss_fn, device="cpu")
        assert isinstance(train_result, LoopResult)

    final = eval_epoch(model, loader, _loss_fn, device="cpu")
    assert isinstance(final, LoopResult)
    assert final.loss < initial.loss

    checkpoint = tmp_path / "checkpoint.pt"
    encoder_state = {f"encoder.{k}": v.cpu() for k, v in model.encoder.state_dict().items()}
    torch.save({"model_state": encoder_state}, checkpoint)

    onnx_path = tmp_path / "encoder.onnx"
    ts_path = tmp_path / "encoder.ts"
    argv = [
        "--checkpoint",
        str(checkpoint),
        "--onnx",
        str(onnx_path),
        "--torchscript",
        str(ts_path),
        "--image-size",
        str(dataset_spec.image_size),
        "--sequence-length",
        str(dataset_spec.sequence_length),
        "--projector-dim",
        str(training_cfg["projector_dim"]),
        "--d-model",
        str(training_cfg["d_model"]),
        "--pose-landmarks",
        str(dataset_spec.pose_landmarks),
        "--temporal-nhead",
        str(training_cfg["temporal_nhead"]),
        "--temporal-layers",
        "1",
        "--temporal-dim-feedforward",
        str(training_cfg["temporal_dim_feedforward"]),
        "--temporal-dropout",
        "0.0",
        "--projector-dropout",
        "0.0",
        "--fusion-dropout",
        "0.0",
        "--device",
        "cpu",
    ]
    main_export(argv)

    assert onnx_path.exists()
    assert ts_path.exists()

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)

    export_module = EncoderExportModule(model.encoder.eval())
    batch = next(iter(loader))
    inputs = batch["inputs"]
    positional: Tuple[torch.Tensor, ...] = (
        inputs["face"],
        inputs["hand_l"],
        inputs["hand_r"],
        inputs["pose"],
    )
    kwargs = {
        "pad_mask": inputs["pad_mask"],
        "miss_mask_hl": inputs["miss_mask_hl"],
        "miss_mask_hr": inputs["miss_mask_hr"],
        "pose_conf_mask": inputs["pose_conf_mask"],
    }

    scripted = torch.jit.load(str(ts_path))
    scripted.eval()

    with torch.no_grad():
        reference = export_module(*positional, **kwargs)
        scripted_out = scripted(*positional, **kwargs)

    assert isinstance(scripted_out, (tuple, list))
    assert len(scripted_out) == len(reference)
    for ref, out in zip(reference, scripted_out):
        torch.testing.assert_close(out, ref, atol=1e-4, rtol=1e-4)

    session = onnxruntime.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    onnx_inputs = {
        "face": positional[0].detach().cpu().numpy(),
        "hand_l": positional[1].detach().cpu().numpy(),
        "hand_r": positional[2].detach().cpu().numpy(),
        "pose": positional[3].detach().cpu().numpy(),
        "pad_mask": kwargs["pad_mask"].detach().cpu().numpy(),
        "miss_mask_hl": kwargs["miss_mask_hl"].detach().cpu().numpy(),
        "miss_mask_hr": kwargs["miss_mask_hr"].detach().cpu().numpy(),
        "pose_conf_mask": kwargs["pose_conf_mask"].detach().cpu().numpy(),
    }
    onnx_outputs = session.run(None, onnx_inputs)
    assert len(onnx_outputs) == len(reference)
    for ref, ort_out in zip(reference, onnx_outputs):
        ort_tensor = torch.from_numpy(np.asarray(ort_out))
        torch.testing.assert_close(ort_tensor, ref, atol=1e-4, rtol=1e-4)
