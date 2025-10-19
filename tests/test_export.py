"""Integration tests for export utilities."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

onnx = pytest.importorskip("onnx")
torch = pytest.importorskip("torch")

from slt.models import MultiStreamEncoder, ViTConfig
from tools.export_onnx_encoder_v9 import (
    EncoderExportModule,
    _build_encoder,
    _dummy_inputs,
    main_export,
)


@pytest.fixture()
def encoder_args() -> SimpleNamespace:
    return SimpleNamespace(
        image_size=16,
        projector_dim=8,
        d_model=16,
        pose_landmarks=13,
        projector_dropout=0.0,
        fusion_dropout=0.0,
        temporal_nhead=4,
        temporal_layers=1,
        temporal_dim_feedforward=32,
        temporal_dropout=0.1,
        sequence_length=4,
        opset=17,
    )


def _build_state(args: SimpleNamespace) -> MultiStreamEncoder:
    vit_config = ViTConfig(image_size=args.image_size)
    encoder = MultiStreamEncoder(
        backbone_config=vit_config,
        projector_dim=args.projector_dim,
        d_model=args.d_model,
        pose_dim=3 * args.pose_landmarks,
        positional_num_positions=args.sequence_length,
        projector_dropout=args.projector_dropout,
        fusion_dropout=args.fusion_dropout,
        temporal_kwargs={
            "nhead": args.temporal_nhead,
            "nlayers": args.temporal_layers,
            "dim_feedforward": args.temporal_dim_feedforward,
            "dropout": args.temporal_dropout,
        },
    )
    return encoder


def test_main_export_generates_onnx_and_torchscript(tmp_path: Path, encoder_args: SimpleNamespace) -> None:
    encoder = _build_state(encoder_args)
    state_dict = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save({"model_state": state_dict}, checkpoint)

    onnx_path = tmp_path / f"encoder_{uuid4().hex}.onnx"
    ts_path = tmp_path / f"encoder_{uuid4().hex}.ts"

    argv = [
        "--checkpoint",
        str(checkpoint),
        "--onnx",
        str(onnx_path),
        "--torchscript",
        str(ts_path),
        "--image-size",
        str(encoder_args.image_size),
        "--sequence-length",
        str(encoder_args.sequence_length),
        "--projector-dim",
        str(encoder_args.projector_dim),
        "--d-model",
        str(encoder_args.d_model),
        "--pose-landmarks",
        str(encoder_args.pose_landmarks),
        "--projector-dropout",
        str(encoder_args.projector_dropout),
        "--fusion-dropout",
        str(encoder_args.fusion_dropout),
        "--temporal-nhead",
        str(encoder_args.temporal_nhead),
        "--temporal-layers",
        str(encoder_args.temporal_layers),
        "--temporal-dim-feedforward",
        str(encoder_args.temporal_dim_feedforward),
        "--temporal-dropout",
        str(encoder_args.temporal_dropout),
        "--device",
        "cpu",
        "--opset",
        str(encoder_args.opset),
    ]

    main_export(argv)

    assert onnx_path.exists()
    assert ts_path.exists()

    model_args = SimpleNamespace(**encoder_args.__dict__)
    encoder_eval = _build_encoder(model_args)
    encoder_eval.load_state_dict(encoder.state_dict())
    encoder_eval.eval()

    export_module = EncoderExportModule(encoder_eval)
    inputs = _dummy_inputs(model_args, torch.device("cpu"))
    positional = inputs[:4]
    kwargs = {
        "pad_mask": inputs[4],
        "miss_mask_hl": inputs[5],
        "miss_mask_hr": inputs[6],
    }

    with torch.no_grad():
        reference_outputs = export_module(*positional, **kwargs)

    scripted = torch.jit.load(str(ts_path))
    scripted.eval()
    with torch.no_grad():
        ts_outputs = scripted(*positional, **kwargs)

    assert isinstance(ts_outputs, (tuple, list))
    assert len(ts_outputs) == len(reference_outputs)
    for ref, scripted_out in zip(reference_outputs, ts_outputs):
        torch.testing.assert_close(scripted_out, ref)

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    output_names = [output.name for output in onnx_model.graph.output]
    assert output_names == [
        "encoded",
        "face_head",
        "hand_left_head",
        "hand_right_head",
        "pose_head",
        "hand_mask",
        "padding_mask",
    ]
