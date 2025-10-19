#!/usr/bin/env python3
"""Demo en tiempo real para el modelo multi-stream de SLT."""

from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch

try:  # pragma: no cover - dependencia opcional para la demo
    import mediapipe as mp
except Exception:  # pragma: no cover - MediaPipe es opcional
    mp = None  # type: ignore[assignment]

from slt.models import MultiStreamEncoder, TextSeq2SeqDecoder, ViTConfig
from slt.runtime import FrameDetections, HolisticFrameProcessor, TemporalBuffer
from slt.utils.text import create_tokenizer, decode


@dataclass
class DemoConfig:
    """Hyper-parámetros usados por la demo."""

    image_size: int = 224
    sequence_length: int = 32
    pose_landmarks: int = 13
    projector_dim: int = 256
    d_model: int = 512
    temporal_nhead: int = 8
    temporal_layers: int = 6
    temporal_dim_feedforward: int = 2048
    temporal_dropout: float = 0.1
    vocab_size: int = 32_000
    decoder_layers: int = 1
    decoder_heads: int = 4
    decoder_dropout: float = 0.0
    max_tokens: int = 8
    beam_size: int = 1


class MultiStreamSLT(torch.nn.Module):
    """Wrapper que combina encoder y decoder stub."""

    def __init__(self, config: DemoConfig) -> None:
        super().__init__()

        vit_config = ViTConfig(image_size=config.image_size)
        temporal_kwargs = {
            "nhead": config.temporal_nhead,
            "nlayers": config.temporal_layers,
            "dim_feedforward": config.temporal_dim_feedforward,
            "dropout": config.temporal_dropout,
        }

        self.encoder = MultiStreamEncoder(
            backbone_config=vit_config,
            projector_dim=config.projector_dim,
            d_model=config.d_model,
            pose_dim=3 * config.pose_landmarks,
            positional_num_positions=config.sequence_length,
            temporal_kwargs=temporal_kwargs,
        )
        self.decoder = TextSeq2SeqDecoder(
            d_model=config.d_model,
            vocab_size=config.vocab_size,
            num_layers=config.decoder_layers,
            num_heads=config.decoder_heads,
            dropout=config.decoder_dropout,
            pad_token_id=0,
            eos_token_id=1,
        )
        self.max_tokens = config.max_tokens
        self.beam_size = config.beam_size

    def forward(
        self,
        *,
        face: torch.Tensor,
        hand_l: torch.Tensor,
        hand_r: torch.Tensor,
        pose: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        miss_mask_hl: Optional[torch.Tensor] = None,
        miss_mask_hr: Optional[torch.Tensor] = None,
    ) -> torch.LongTensor:
        return self.generate(
            face=face,
            hand_l=hand_l,
            hand_r=hand_r,
            pose=pose,
            pad_mask=pad_mask,
            miss_mask_hl=miss_mask_hl,
            miss_mask_hr=miss_mask_hr,
        )

    def generate(
        self,
        *,
        face: torch.Tensor,
        hand_l: torch.Tensor,
        hand_r: torch.Tensor,
        pose: torch.Tensor,
        pad_mask: Optional[torch.Tensor] = None,
        miss_mask_hl: Optional[torch.Tensor] = None,
        miss_mask_hr: Optional[torch.Tensor] = None,
        **generation_kwargs,
    ) -> torch.LongTensor:
        encoded = self.encoder(
            face,
            hand_l,
            hand_r,
            pose,
            pad_mask=pad_mask,
            miss_mask_hl=miss_mask_hl,
            miss_mask_hr=miss_mask_hr,
        )
        encoder_attention_mask = pad_mask.to(torch.long) if pad_mask is not None else None
        max_length = generation_kwargs.pop("max_length", self.max_tokens)
        num_beams = generation_kwargs.pop("num_beams", self.beam_size)
        return self.decoder.generate(
            encoded,
            encoder_attention_mask=encoder_attention_mask,
            max_length=max_length,
            num_beams=num_beams,
            **generation_kwargs,
        )


class ModelRunner:
    """Manejador de modelos TorchScript/ONNX para la demo."""

    def __init__(
        self,
        *,
        config: DemoConfig,
        device: torch.device,
        model_path: Optional[Path],
        model_format: str,
        max_tokens: int,
        beam_size: int,
        onnx_provider: Optional[str] = None,
    ) -> None:
        self.device = device
        self.max_tokens = max_tokens
        self.beam_size = beam_size

        resolved_format = infer_model_format(model_path, model_format)
        self.backend = "torch" if resolved_format in {"stub", "torchscript"} else "onnx"

        if resolved_format == "stub":
            self.model = MultiStreamSLT(config).to(device)
            self.model.eval()
            self.generate_method = self.model.generate
        elif resolved_format == "torchscript":
            if model_path is None:
                raise ValueError("--model es obligatorio cuando se utiliza TorchScript")
            self.model = torch.jit.load(str(model_path), map_location=device)
            self.model.eval()
            self.generate_method = getattr(self.model, "generate", None)
        elif resolved_format == "onnx":
            if model_path is None:
                raise ValueError("--model es obligatorio para modelos ONNX")
            try:
                import onnxruntime as ort
            except ImportError as exc:  # pragma: no cover - dependencia opcional
                raise RuntimeError(
                    "onnxruntime no está disponible. Instala el extra 'export' o la librería manualmente."
                ) from exc

            providers = None
            if onnx_provider:
                providers = [onnx_provider]
            else:
                if device.type == "cuda":
                    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
                else:
                    providers = ["CPUExecutionProvider"]
            self.session = ort.InferenceSession(str(model_path), providers=providers)
            self.input_names = [inp.name for inp in self.session.get_inputs()]
            self.generate_method = None
        else:  # pragma: no cover - formato inesperado
            raise ValueError(f"Formato de modelo no soportado: {resolved_format}")

    def __call__(self, inputs: Dict[str, torch.Tensor | np.ndarray]) -> torch.LongTensor:
        if self.backend == "onnx":
            ort_inputs = {}
            for name in self.input_names:
                if name not in inputs:
                    raise KeyError(f"El modelo ONNX espera la entrada '{name}'")
                value = inputs[name]
                if isinstance(value, torch.Tensor):
                    array = value.detach().cpu().numpy()
                else:
                    array = value
                if array.dtype == np.bool_:
                    ort_inputs[name] = array
                else:
                    ort_inputs[name] = array.astype(np.float32, copy=False)
            outputs = self.session.run(None, ort_inputs)
            return torch.from_numpy(outputs[0])

        tensor_inputs = {}
        for key, value in inputs.items():
            tensor_inputs[key] = value if isinstance(value, torch.Tensor) else torch.from_numpy(value).to(self.device)

        if self.generate_method is not None:
            try:
                output = self.generate_method(
                    **tensor_inputs,
                    max_length=self.max_tokens,
                    num_beams=self.beam_size,
                )
            except TypeError:
                output = self.generate_method(**tensor_inputs)  # type: ignore[misc]
        else:
            output = self.model(**tensor_inputs)

        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, (list, tuple)) and output:
            return torch.as_tensor(output[0])
        raise RuntimeError("El modelo TorchScript devolvió una salida inesperada")


def decode_token_ids_stub(sequences: torch.Tensor) -> str:
    """Decodificador placeholder que expone los IDs generados."""

    if sequences.dim() == 2:
        seq = sequences[0]
    else:
        seq = sequences
    for token_id in seq.tolist():
        if token_id not in (0,):
            return f"<token_{int(token_id)}>"
    return "<token_0>"


def decode_sequences(sequences: torch.Tensor, tokenizer) -> str:
    if tokenizer is None:
        return decode_token_ids_stub(sequences)
    texts = decode(tokenizer, sequences, skip_special_tokens=True)
    return texts[0] if texts else ""


def draw_overlays(frame: np.ndarray, boxes: Dict[str, Optional[Tuple[int, int, int, int]]], detections: FrameDetections, text: str) -> None:
    colors = {
        "face": (0, 255, 0),
        "hand_l": (255, 0, 0),
        "hand_r": (0, 0, 255),
    }
    for key, bbox in boxes.items():
        if not bbox:
            continue
        x, y, w, h = bbox
        color = colors.get(key, (0, 255, 255))
        thickness = 3 if getattr(detections, key, False) else 1
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, thickness)
    if text:
        height = frame.shape[0]
        cv2.putText(
            frame,
            text,
            (10, height - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )


def build_tokenizer(args: argparse.Namespace):
    if not getattr(args, "tokenizer", None):
        return None
    return create_tokenizer(args.tokenizer, revision=getattr(args, "tokenizer_revision", None))


def infer_model_format(model_path: Optional[Path], model_format: str) -> str:
    if model_format != "auto":
        return model_format
    if model_path is None:
        return "stub"
    suffix = model_path.suffix.lower()
    if suffix in {".pt", ".pth", ".ts"}:
        return "torchscript"
    if suffix == ".onnx":
        return "onnx"
    raise ValueError("No se pudo inferir el formato del modelo. Usa --model-format explícitamente.")


def add_model_cli_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", type=Path, help="Ruta al modelo TorchScript/ONNX exportado", default=None)
    parser.add_argument(
        "--model-format",
        choices=("auto", "stub", "torchscript", "onnx"),
        default="auto",
        help="Formato del modelo especificado en --model",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=None,
        help="Longitud máxima para generación autoregresiva",
    )
    parser.add_argument("--beam-size", type=int, default=1, help="Número de beams para la búsqueda autoregresiva")
    parser.add_argument(
        "--onnx-provider",
        type=str,
        default=None,
        help="Proveedor preferido de ejecución para onnxruntime (por ejemplo CUDAExecutionProvider)",
    )
    parser.add_argument("--tokenizer", type=str, help="Nombre o ruta del tokenizador HuggingFace", default=None)
    parser.add_argument("--tokenizer-revision", type=str, help="Revisión del tokenizador (tag/commit)", default=None)


def run_demo(args: argparse.Namespace) -> None:
    if mp is None:
        raise RuntimeError(
            "MediaPipe no está disponible. Instala el paquete 'mediapipe' para ejecutar la demo."
        )

    config = DemoConfig(
        sequence_length=args.sequence_length,
        pose_landmarks=args.pose_landmarks,
    )
    if args.max_tokens:
        config.max_tokens = args.max_tokens
    config.beam_size = args.beam_size
    device = torch.device(args.device)

    tokenizer = build_tokenizer(args)

    runner = ModelRunner(
        config=config,
        device=device,
        model_path=args.model,
        model_format=args.model_format,
        max_tokens=config.max_tokens,
        beam_size=config.beam_size,
        onnx_provider=args.onnx_provider,
    )

    buffer = TemporalBuffer(config.sequence_length, config.image_size, config.pose_landmarks)
    processor = HolisticFrameProcessor(
        image_size=config.image_size,
        pose_landmarks=config.pose_landmarks,
        bbox_scale=args.bbox_scale,
        smoothing=args.smoothing,
        max_misses=args.max_misses,
    )

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la cámara con índice {args.camera}")

    window_name = "MultiStream SLT Demo"
    if not args.no_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    face_mesh = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)
    hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2)
    pose = mp.solutions.pose.Pose(static_image_mode=False, model_complexity=1)

    mediapipe_contexts = [face_mesh, hands, pose]

    @contextlib.contextmanager
    def closing_all(contexts: Iterable) -> Iterable:
        try:
            yield contexts
        finally:
            for ctx in contexts:
                ctx.close()

    last_text = ""
    with closing_all(mediapipe_contexts):
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                face_result = face_mesh.process(rgb)
                hands_result = hands.process(rgb)
                pose_result = pose.process(rgb)

                face_tensor, hand_l_tensor, hand_r_tensor, pose_tensor, detections, boxes = processor.process(
                    frame,
                    face_result=face_result,
                    hands_result=hands_result,
                    pose_result=pose_result,
                )

                buffer.append(
                    face_tensor,
                    hand_l_tensor,
                    hand_r_tensor,
                    pose_tensor,
                    detected_left=detections.hand_l,
                    detected_right=detections.hand_r,
                )

                inputs = buffer.as_model_inputs(device, backend=runner.backend)
                if inputs is not None:
                    with torch.no_grad():
                        sequences = runner(inputs)
                    text = decode_sequences(sequences, tokenizer)
                    if text and text != last_text:
                        print(text)
                        last_text = text
                else:
                    text = last_text

                if not args.no_window:
                    draw_overlays(frame, boxes, detections, last_text)
                    cv2.imshow(window_name, frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        break
            if args.no_window and last_text:
                print(last_text)
        finally:
            cap.release()
            cv2.destroyAllWindows()


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--camera", type=int, default=0, help="Índice de la cámara a utilizar")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=32,
        help="Número de frames a acumular antes de invocar el modelo",
    )
    parser.add_argument(
        "--pose-landmarks",
        type=int,
        default=13,
        help="Cantidad de landmarks de pose a considerar (MediaPipe Holistic)",
    )
    parser.add_argument("--bbox-scale", type=float, default=1.2, help="Factor de expansión del bounding box")
    parser.add_argument("--smoothing", type=float, default=0.4, help="Factor de suavizado para el tracking de ROI")
    parser.add_argument("--max-misses", type=int, default=5, help="Cantidad de frames sin detección antes de descartar la ROI")
    parser.add_argument("--no-window", action="store_true", help="Ejecuta la demo sin mostrar overlay de OpenCV")

    add_model_cli_arguments(parser)
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    run_demo(args)


if __name__ == "__main__":  # pragma: no cover - ejecución manual
    main()
