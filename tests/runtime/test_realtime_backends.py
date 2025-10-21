"""Tests for runtime helpers and export integration."""

from __future__ import annotations

import ctypes
import math
import sys
import types
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Sequence, Tuple

import numpy as np
import pytest
import torch

if "cv2" not in sys.modules:
    mock_cv2 = types.ModuleType("cv2")
    mock_cv2.COLOR_BGR2RGB = 0
    mock_cv2.INTER_LINEAR = 1
    mock_cv2.__spec__ = types.SimpleNamespace()

    def _mock_cvt_color(image: np.ndarray, code: int) -> np.ndarray:
        if code != mock_cv2.COLOR_BGR2RGB:  # pragma: no cover - defensive
            raise ValueError("Unsupported conversion code")
        return image[..., ::-1]

    def _mock_resize(image: np.ndarray, size: Sequence[int], interpolation: int | None = None) -> np.ndarray:
        width, height = size
        if image.ndim == 3:
            channels = image.shape[2]
            return np.zeros((height, width, channels), dtype=image.dtype)
        return np.zeros((height, width), dtype=image.dtype)

    mock_cv2.cvtColor = _mock_cvt_color
    mock_cv2.resize = _mock_resize
    sys.modules["cv2"] = mock_cv2

from slt.runtime.realtime import (
    OnnxRuntimeEncoderRunner,
    TemporalBuffer,
    TensorRTEncoderRunner,
    TorchScriptEncoderRunner,
    load_export_metadata,
)


def _encoder_args() -> SimpleNamespace:
    return SimpleNamespace(
        image_size=16,
        projector_dim=8,
        d_model=16,
        pose_landmarks=5,
        projector_dropout=0.0,
        fusion_dropout=0.0,
        temporal_nhead=2,
        temporal_layers=1,
        temporal_dim_feedforward=32,
        temporal_dropout=0.1,
        sequence_length=4,
        opset=17,
    )


class _AssertingBackbone(torch.nn.Module):
    def __init__(self, *, patch_size: int = 7, embed_dim: int = 8) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_embed = torch.nn.Module()
        self.patch_embed.proj = torch.nn.Conv2d(
            3,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )
        self.last_input_shape: Tuple[int, ...] | None = None

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        self.last_input_shape = tuple(images.shape)
        kernel_h, kernel_w = self.patch_embed.proj.kernel_size
        if images.shape[-2] % kernel_h != 0 or images.shape[-1] % kernel_w != 0:
            raise AssertionError("Backbone received incompatible spatial dimensions")
        tokens = self.patch_embed.proj(images)
        return tokens.flatten(2).mean(dim=-1)


def test_temporal_buffer_latency_masks() -> None:
    buffer = TemporalBuffer(
        sequence_length=4,
        image_size=8,
        pose_landmarks=3,
        target_latency_ms=50,
        frame_rate=25.0,
    )
    device = torch.device("cpu")

    for idx in range(4):
        image = torch.full((3, 8, 8), float(idx), dtype=torch.float32)
        pose = torch.full((3 * 3,), float(idx), dtype=torch.float32)
        buffer.append(
            face=image,
            hand_l=image,
            hand_r=image,
            pose=pose,
            missing_left=bool(idx % 2),
            missing_right=bool((idx + 1) % 2),
        )

    inputs = buffer.as_model_inputs(device, backend="torch")
    assert inputs is not None
    pad_mask = inputs["pad_mask"][0]
    assert pad_mask.sum().item() == buffer.window_size
    assert pad_mask.tolist() == [True, True, False, False]

    miss_left = inputs["miss_mask_hl"][0]
    miss_right = inputs["miss_mask_hr"][0]
    assert miss_left.tolist() == [False, True, False, False]
    assert miss_right.tolist() == [True, False, False, False]


def test_runtime_backends_integration(tmp_path: Path) -> None:
    pytest.importorskip("cv2", exc_type=ImportError, reason="cv2 runtime dependencies not available")
    try:
        from tools.export_onnx_encoder_v9 import (
            EncoderExportModule,
            _build_encoder,
            main_export,
        )
    except Exception as exc:  # pragma: no cover - entorno sin dependencias de exportación
        pytest.skip(f"Export tooling unavailable: {exc}")
    encoder_args = _encoder_args()
    encoder = _build_encoder(encoder_args)
    encoder.eval()
    state_dict = {f"encoder.{k}": v for k, v in encoder.state_dict().items()}

    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model_state": state_dict,
            "encoder_config": {
                "image_size": encoder_args.image_size,
                "projector_dim": encoder_args.projector_dim,
                "d_model": encoder_args.d_model,
                "pose_landmarks": encoder_args.pose_landmarks,
                "sequence_length": encoder_args.sequence_length,
            },
        },
        checkpoint,
    )

    artifact_dir = tmp_path / "artifacts"
    version = "vtest"
    argv = [
        "--checkpoint",
        str(checkpoint),
        "--artifact-dir",
        str(artifact_dir),
        "--version",
        version,
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
    ]

    main_export(argv)

    metadata_path = artifact_dir / f"encoder_{version}.json"
    ts_path = artifact_dir / f"encoder_{version}.ts"
    onnx_path = artifact_dir / f"encoder_{version}.onnx"

    assert metadata_path.exists()
    assert ts_path.exists()
    assert onnx_path.exists()

    metadata = load_export_metadata(metadata_path)
    assert metadata["artifact_version"] == version
    assert metadata["inputs"]["face"]["normalization"]["mean"] == pytest.approx([0.485, 0.456, 0.406])

    device = torch.device("cpu")
    buffer = TemporalBuffer(
        sequence_length=encoder_args.sequence_length,
        image_size=encoder_args.image_size,
        pose_landmarks=encoder_args.pose_landmarks,
        target_latency_ms=40,
        frame_rate=60.0,
    )

    for step in range(encoder_args.sequence_length + 1):
        img = torch.randn(3, encoder_args.image_size, encoder_args.image_size)
        pose = torch.randn(3 * encoder_args.pose_landmarks)
        buffer.append(
            face=img,
            hand_l=img,
            hand_r=img,
            pose=pose,
            missing_left=bool(step % 2),
            missing_right=bool((step + 1) % 2),
        )

    inputs_torch = buffer.as_model_inputs(device, backend="torchscript")
    assert inputs_torch is not None

    model_eval = _build_encoder(SimpleNamespace(**encoder_args.__dict__))
    model_eval.load_state_dict(encoder.state_dict())
    model_eval.eval()
    export_module = EncoderExportModule(model_eval)

    ordered_args = [inputs_torch[name] for name in ("face", "hand_l", "hand_r", "pose")]
    kwargs = {
        "pad_mask": inputs_torch["pad_mask"],
        "miss_mask_hl": inputs_torch["miss_mask_hl"],
        "miss_mask_hr": inputs_torch["miss_mask_hr"],
    }

    with torch.no_grad():
        reference = export_module(*ordered_args, **kwargs)

    ts_runner = TorchScriptEncoderRunner(ts_path, metadata, device=device)
    ts_outputs = ts_runner(inputs_torch)
    assert len(ts_outputs) == len(reference)
    for ref_tensor, ts_tensor in zip(reference, ts_outputs):
        torch.testing.assert_close(ts_tensor, ref_tensor)

    pytest.importorskip("onnxruntime")
    inputs_onnx = buffer.as_model_inputs(device, backend="onnxruntime")
    assert inputs_onnx is not None
    onnx_runner = OnnxRuntimeEncoderRunner(onnx_path, metadata)
    onnx_outputs = onnx_runner(inputs_onnx)
    assert len(onnx_outputs) == len(reference)
    for ref_tensor, ort_out in zip(reference, onnx_outputs):
        np.testing.assert_allclose(ort_out, ref_tensor.detach().cpu().numpy(), rtol=1e-4, atol=1e-4)


def test_torchscript_runner_handles_non_divisible_image_size(tmp_path: Path) -> None:
    try:
        from tools.export_onnx_encoder_v9 import EncoderExportModule
    except Exception as exc:  # pragma: no cover - entorno sin dependencias de exportación
        pytest.skip(f"Export tooling unavailable: {exc}")

    from slt.models import MultiStreamEncoder

    args = _encoder_args()
    patch_size = 7

    def _make_backbone() -> _AssertingBackbone:
        return _AssertingBackbone(patch_size=patch_size, embed_dim=args.projector_dim)

    encoder = MultiStreamEncoder(
        backbones={
            "face": _make_backbone(),
            "hand_left": _make_backbone(),
            "hand_right": _make_backbone(),
        },
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
    encoder.eval()
    export_module = EncoderExportModule(encoder)

    device = torch.device("cpu")
    batch = 1
    seq = args.sequence_length
    image_size = args.image_size
    pose_dim = 3 * args.pose_landmarks

    face = torch.randn(batch, seq, 3, image_size, image_size, device=device)
    hand_l = torch.randn_like(face)
    hand_r = torch.randn_like(face)
    pose = torch.randn(batch, seq, pose_dim, device=device)
    pad_mask = torch.ones(batch, seq, dtype=torch.bool, device=device)
    miss_mask = torch.zeros(batch, seq, dtype=torch.bool, device=device)

    example_inputs = (face, hand_l, hand_r, pose, pad_mask, miss_mask, miss_mask)

    with torch.no_grad():
        reference = export_module(*example_inputs)

    target_dim = math.ceil(image_size / patch_size) * patch_size
    for attr in ("face_backbone", "hand_backbone_left", "hand_backbone_right"):
        recorded = getattr(encoder, attr).last_input_shape
        assert recorded is not None
        assert recorded[-2:] == (target_dim, target_dim)

    traced = torch.jit.trace(export_module, example_inputs, check_trace=False)
    ts_path = tmp_path / "encoder_padding.ts"
    traced.save(str(ts_path))

    metadata = {
        "inputs": {
            "face": {"dtype": "float32"},
            "hand_l": {"dtype": "float32"},
            "hand_r": {"dtype": "float32"},
            "pose": {"dtype": "float32"},
            "pad_mask": {"dtype": "bool"},
            "miss_mask_hl": {"dtype": "bool"},
            "miss_mask_hr": {"dtype": "bool"},
        }
    }

    runner_inputs = {
        "face": face,
        "hand_l": hand_l,
        "hand_r": hand_r,
        "pose": pose,
        "pad_mask": pad_mask,
        "miss_mask_hl": miss_mask,
        "miss_mask_hr": miss_mask,
    }

    ts_runner = TorchScriptEncoderRunner(ts_path, metadata, device=device)
    ts_outputs = ts_runner(runner_inputs)

    assert len(ts_outputs) == len(reference)
    for ref_tensor, ts_tensor in zip(reference, ts_outputs):
        torch.testing.assert_close(ts_tensor, ref_tensor)


def test_tensorrt_runner_with_mock_engine(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ejecuta el runner TensorRT con motores simulados y verifica conversiones."""

    memory: Dict[int, bytearray] = {}
    next_ptr = 1

    cuda_pkg = types.ModuleType("cuda")
    cudart_mod = types.ModuleType("cuda.cudart")

    class _MemcpyKind:
        cudaMemcpyHostToDevice = 1
        cudaMemcpyDeviceToHost = 2

    def _cuda_stream_create() -> Tuple[int, int]:
        return 0, 7

    def _cuda_stream_sync(stream: int) -> int:
        assert stream == 7
        return 0

    def _cuda_malloc(size: int) -> Tuple[int, int]:
        nonlocal next_ptr
        ptr = next_ptr
        next_ptr += 1
        memory[ptr] = bytearray(size)
        return 0, ptr

    def _cuda_free(ptr: int) -> int:
        memory.pop(ptr, None)
        return 0

    def _cuda_memcpy_async(dst: int, src: int, size: int, kind: int, stream: int) -> int:
        assert stream == 7
        if kind == _MemcpyKind.cudaMemcpyHostToDevice:
            data = ctypes.string_at(src, size)
            memory[dst][:] = data
        elif kind == _MemcpyKind.cudaMemcpyDeviceToHost:
            data = bytes(memory[src][:size])
            ctypes.memmove(dst, data, size)
        else:  # pragma: no cover - defensive
            raise AssertionError("Unsupported memcpy direction")
        return 0

    cudart_mod.cudaMemcpyKind = _MemcpyKind
    cudart_mod.cudaStreamCreate = _cuda_stream_create
    cudart_mod.cudaStreamSynchronize = _cuda_stream_sync
    cudart_mod.cudaMalloc = _cuda_malloc
    cudart_mod.cudaFree = _cuda_free
    cudart_mod.cudaMemcpyAsync = _cuda_memcpy_async
    cudart_mod._memory = memory
    cuda_pkg.cudart = cudart_mod

    monkeypatch.setitem(sys.modules, "cuda", cuda_pkg)
    monkeypatch.setitem(sys.modules, "cuda.cudart", cudart_mod)

    trt_mod = types.ModuleType("tensorrt")

    class _FakeLogger:
        WARNING = 1

        def __init__(self, level: int) -> None:
            self.level = level

    class _FakeRuntime:
        def __init__(self, logger: _FakeLogger) -> None:
            self.logger = logger

        def deserialize_cuda_engine(self, _: bytes) -> "_FakeEngine":
            return _FakeEngine()

    class _DataType:
        FLOAT = "float32"
        BOOL = "bool"

    def _nptype(dtype: str) -> np.dtype[Any]:
        return np.float32 if dtype == _DataType.FLOAT else np.bool_

    def _volume(shape: Sequence[int]) -> int:
        prod = 1
        for dim in shape:
            prod *= dim
        return prod

    class _FakeContext:
        def __init__(self, engine: "_FakeEngine") -> None:
            self.engine = engine
            self._binding_shapes = dict(engine._binding_shapes)
            self.captured_inputs: Dict[str, np.ndarray] = {}

        def set_binding_shape(self, index: int, shape: Sequence[int]) -> bool:
            self._binding_shapes[index] = tuple(shape)
            return True

        def get_binding_shape(self, index: int) -> Tuple[int, ...]:
            return self._binding_shapes[index]

        def execute_async_v2(self, bindings: Sequence[int], stream_handle: int) -> bool:
            assert stream_handle == 7
            for index, (name, is_input, dtype, _) in enumerate(self.engine._bindings):
                buffer = cudart_mod._memory[bindings[index]]
                np_dtype = _nptype(dtype)
                shape = self._binding_shapes[index]
                if is_input:
                    array = np.frombuffer(buffer, dtype=np_dtype).reshape(shape).copy()
                    self.captured_inputs[name] = array
                else:
                    value = np.full(shape, index, dtype=np_dtype)
                    buffer[: value.nbytes] = value.tobytes()
            return True

    class _FakeEngine:
        _bindings = [
            ("face", True, _DataType.FLOAT, (1, 4, 3, 8, 8)),
            ("hand_l", True, _DataType.FLOAT, (1, 4, 3, 8, 8)),
            ("hand_r", True, _DataType.FLOAT, (1, 4, 3, 8, 8)),
            ("pose", True, _DataType.FLOAT, (1, 4, 15)),
            ("pad_mask", True, _DataType.BOOL, (1, 4)),
            ("miss_mask_hl", True, _DataType.BOOL, (1, 4)),
            ("miss_mask_hr", True, _DataType.BOOL, (1, 4)),
            ("encoded", False, _DataType.FLOAT, (1, 4, 16)),
            ("face_head", False, _DataType.FLOAT, (1, 4, 8)),
            ("hand_left_head", False, _DataType.FLOAT, (1, 4, 8)),
            ("hand_right_head", False, _DataType.FLOAT, (1, 4, 8)),
            ("pose_head", False, _DataType.FLOAT, (1, 4, 8)),
            ("hand_mask", False, _DataType.FLOAT, (1, 4)),
            ("padding_mask", False, _DataType.FLOAT, (1, 4)),
        ]

        def __init__(self) -> None:
            self.num_bindings = len(self._bindings)
            self._binding_shapes = {
                index: binding[3] for index, binding in enumerate(self._bindings)
            }

        def create_execution_context(self) -> _FakeContext:
            return _FakeContext(self)

        def get_binding_name(self, index: int) -> str:
            return self._bindings[index][0]

        def binding_is_input(self, index: int) -> bool:
            return self._bindings[index][1]

        def get_binding_dtype(self, index: int) -> str:
            return self._bindings[index][2]

        def get_binding_shape(self, index: int) -> Tuple[int, ...]:
            return self._binding_shapes[index]

    trt_mod.Logger = _FakeLogger
    trt_mod.Runtime = _FakeRuntime
    trt_mod.DataType = _DataType
    trt_mod.nptype = _nptype
    trt_mod.volume = _volume

    monkeypatch.setitem(sys.modules, "tensorrt", trt_mod)

    engine_path = tmp_path / "encoder.plan"
    engine_path.write_bytes(b"fake-tensorrt-engine")

    metadata = {
        "inputs": {
            "face": {"dtype": "float32"},
            "hand_l": {"dtype": "float32"},
            "hand_r": {"dtype": "float32"},
            "pose": {"dtype": "float32"},
            "pad_mask": {"dtype": "bool"},
            "miss_mask_hl": {"dtype": "bool"},
            "miss_mask_hr": {"dtype": "bool"},
        },
        "outputs": {
            "encoded": {"dtype": "float32"},
            "face_head": {"dtype": "float32"},
            "hand_left_head": {"dtype": "float32"},
            "hand_right_head": {"dtype": "float32"},
            "pose_head": {"dtype": "float32"},
            "hand_mask": {"dtype": "float32"},
            "padding_mask": {"dtype": "float32"},
        },
    }

    runner = TensorRTEncoderRunner(engine_path, metadata)

    inputs = {
        "face": torch.randn(1, 4, 3, 8, 8, dtype=torch.float32),
        "hand_l": torch.randn(1, 4, 3, 8, 8, dtype=torch.float32),
        "hand_r": torch.randn(1, 4, 3, 8, 8, dtype=torch.float32),
        "pose": torch.randn(1, 4, 15, dtype=torch.float32),
        "pad_mask": torch.tensor([[1, 1, 0, 0]], dtype=torch.int32),
        "miss_mask_hl": torch.tensor([[0, 1, 0, 1]], dtype=torch.int32),
        "miss_mask_hr": torch.tensor([[1, 0, 1, 0]], dtype=torch.int32),
    }

    outputs = runner(inputs)

    assert len(outputs) == len(metadata["outputs"])
    np.testing.assert_array_equal(
        outputs[0],
        np.full((1, 4, 16), 7, dtype=np.float32),
    )
    assert outputs[-1].dtype == np.float32

    captured_pad = runner.context.captured_inputs["pad_mask"]
    assert captured_pad.dtype == np.bool_
    np.testing.assert_array_equal(
        captured_pad,
        np.array([[True, True, False, False]], dtype=np.bool_),
    )
    assert not memory
