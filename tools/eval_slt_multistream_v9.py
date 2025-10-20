#!/usr/bin/env python3
"""Evalúa el modelo multi-stream generando predicciones y métricas."""

import argparse
import csv
import json
import logging
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from slt.data import LsaTMultiStream, collate_fn
from slt.models import MultiStreamEncoder, TextSeq2SeqDecoder, ViTConfig
from slt.models.single_signer import load_single_signer_components
from slt.utils.text import (
    TokenizerValidationError,
    character_error_rate,
    create_tokenizer,
    decode,
    validate_tokenizer,
    word_error_rate,
)

from transformers import PreTrainedTokenizerBase

try:  # pragma: no cover - import opcional
    from sacrebleu.metrics import BLEU, CHRF
except ImportError:  # pragma: no cover - fallback para entornos sin extra
    BLEU = None  # type: ignore[assignment]
    CHRF = None  # type: ignore[assignment]


@dataclass
class PredictionItem:
    """Predicción textual asociada a un video."""

    video_id: str
    prediction: str
    reference: str
    latency_ms: Optional[float] = None


@dataclass
class ModelConfig:
    """Configuración del modelo utilizada durante la evaluación."""

    image_size: int = 224
    projector_dim: int = 128
    d_model: int = 128
    pose_landmarks: int = 13
    projector_dropout: float = 0.05
    fusion_dropout: float = 0.05
    temporal_nhead: int = 4
    temporal_layers: int = 3
    temporal_dim_feedforward: int = 384
    temporal_dropout: float = 0.05
    sequence_length: int = 128
    decoder_layers: int = 2
    decoder_heads: int = 4
    decoder_dropout: float = 0.1
    pretrained: Optional[str] = "single_signer"
    pretrained_checkpoint: Optional[Path] = None


@dataclass
class EvaluationResult:
    """Resumen de evaluación asociado a un checkpoint específico."""

    checkpoint: Path
    predictions: List[PredictionItem]
    metrics: Dict[str, float]
    examples: List[PredictionItem]


class MultiStreamClassifier(nn.Module):
    """Ensamble encoder/decoder utilizado tanto en entrenamiento como en inferencia."""

    def __init__(self, config: ModelConfig, tokenizer: PreTrainedTokenizerBase) -> None:
        super().__init__()

        pretrained = (config.pretrained or "").strip().lower()
        if pretrained and pretrained not in {"none", "false"}:
            if pretrained not in {"single_signer", "single-signer"}:
                raise ValueError(
                    "Identificador de pesos no soportado. Usa 'single_signer' o 'none'."
                )
            encoder, decoder, metadata = load_single_signer_components(
                tokenizer,
                checkpoint_path=config.pretrained_checkpoint,
                map_location=torch.device("cpu"),
                strict=True,
            )
            self.encoder = encoder
            self.decoder = decoder
            setattr(self, "pretrained_metadata", metadata)
            return

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
            projector_dropout=config.projector_dropout,
            fusion_dropout=config.fusion_dropout,
            temporal_kwargs=temporal_kwargs,
        )
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
        vocab_size = getattr(tokenizer, "vocab_size", None)
        if not vocab_size:
            vocab_size = len(tokenizer)
        self.decoder = TextSeq2SeqDecoder(
            d_model=config.d_model,
            vocab_size=int(vocab_size),
            num_layers=config.decoder_layers,
            num_heads=config.decoder_heads,
            dropout=config.decoder_dropout,
            pad_token_id=pad_id,
            eos_token_id=eos_id,
        )

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
        labels: Optional[torch.Tensor] = None,
        decoder_attention_mask: Optional[torch.Tensor] = None,
        encoder_attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        encoded = self.encoder(
            face,
            hand_l,
            hand_r,
            pose,
            pad_mask=pad_mask,
            miss_mask_hl=miss_mask_hl,
            miss_mask_hr=miss_mask_hr,
        )
        if encoder_attention_mask is None and pad_mask is not None:
            encoder_attention_mask = pad_mask.to(torch.long)
        return self.decoder(
            encoded,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
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
        encoder_attention_mask: Optional[torch.Tensor] = None,
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
        if encoder_attention_mask is None and pad_mask is not None:
            encoder_attention_mask = pad_mask.to(torch.long)
        return self.decoder.generate(
            encoded,
            encoder_attention_mask=encoder_attention_mask,
            **generation_kwargs,
        )


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--face-dir", type=Path, required=True, help="Directorio con frames de rostro")
    parser.add_argument("--hand-left-dir", type=Path, required=True, help="Directorio con frames de mano izquierda")
    parser.add_argument("--hand-right-dir", type=Path, required=True, help="Directorio con frames de mano derecha")
    parser.add_argument("--pose-dir", type=Path, required=True, help="Directorio con poses .npz")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="CSV con columnas video_id/texto")
    parser.add_argument("--eval-index", type=Path, required=True, help="CSV con la lista de video_id a evaluar")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        nargs="+",
        required=True,
        help="Ruta(s) a uno o varios checkpoints a cargar",
    )
    parser.add_argument("--output-csv", type=Path, required=True, help="Archivo de salida para escribir predicciones")

    parser.add_argument("--batch-size", type=int, default=4, help="Tamaño de batch para la evaluación")
    parser.add_argument("--num-workers", type=int, default=0, help="Número de workers del DataLoader")
    parser.add_argument("--device", type=str, default="cuda", help="Dispositivo torch (cuda o cpu)")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false", help="Deshabilita pin_memory en DataLoader")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", help="Habilita pin_memory en DataLoader")
    parser.set_defaults(pin_memory=True)

    parser.add_argument("--image-size", type=int, default=224, help="Tamaño de imagen esperado por el encoder")
    parser.add_argument("--projector-dim", type=int, default=128, help="Dimensión de los proyectores de stream")
    parser.add_argument("--d-model", type=int, default=128, help="Dimensión del espacio fusionado")
    parser.add_argument("--pose-landmarks", type=int, default=13, help="Cantidad de landmarks en el stream de pose")
    parser.add_argument("--projector-dropout", type=float, default=0.05, help="Dropout aplicado a proyectores")
    parser.add_argument("--fusion-dropout", type=float, default=0.05, help="Dropout aplicado tras la fusión")
    parser.add_argument("--temporal-nhead", type=int, default=4, help="Número de cabezales de atención temporal")
    parser.add_argument("--temporal-layers", type=int, default=3, help="Capas del codificador temporal")
    parser.add_argument(
        "--temporal-dim-feedforward",
        type=int,
        default=384,
        help="Dimensión del feedforward en el codificador temporal",
    )
    parser.add_argument("--temporal-dropout", type=float, default=0.05, help="Dropout del codificador temporal")
    parser.add_argument("--sequence-length", type=int, default=128, help="Cantidad de frames muestreados por video")
    parser.add_argument(
        "--decoder-layers",
        type=int,
        default=2,
        help="Capas del decoder seq2seq",
    )
    parser.add_argument(
        "--decoder-heads",
        type=int,
        default=4,
        help="Cabezas de atención del decoder seq2seq",
    )
    parser.add_argument(
        "--decoder-dropout",
        type=float,
        default=0.1,
        help="Dropout interno del decoder seq2seq",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        required=True,
        help="Identificador o ruta de un tokenizer de HuggingFace",
    )
    parser.add_argument(
        "--max-target-length",
        type=int,
        default=128,
        help="Longitud máxima utilizada al generar predicciones",
    )

    parser.add_argument(
        "--pretrained",
        type=str,
        default="single_signer",
        help="Pesos pre-entrenados a cargar (single_signer o none)",
    )
    parser.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help=(
            "Ruta al checkpoint single_signer descargado. Solo aplica cuando --pretrained"
            " está activo."
        ),
    )
    parser.add_argument("--num-beams", type=int, default=1, help="Cantidad de beams para la decodificación")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Cantidad máxima de tokens generados autoregresivamente",
    )

    parser.add_argument("--log-level", default="INFO", help="Nivel de logging (INFO, DEBUG, ...)")
    parser.add_argument(
        "--skip-metrics",
        dest="compute_metrics",
        action="store_false",
        help="Omite el cálculo de métricas de calidad de texto",
    )
    parser.set_defaults(compute_metrics=True)

    return parser.parse_args(argv)


def _setup_logging(level: str) -> None:
    logging.basicConfig(level=level.upper(), format="%(asctime)s - %(levelname)s - %(message)s")


def _ensure_path(path: Path, *, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{kind} inexistente: {path}")


def _validate_inputs(args: argparse.Namespace) -> None:
    _ensure_path(args.face_dir, kind="Directorio de rostro")
    _ensure_path(args.hand_left_dir, kind="Directorio de mano izquierda")
    _ensure_path(args.hand_right_dir, kind="Directorio de mano derecha")
    _ensure_path(args.pose_dir, kind="Directorio de pose")
    _ensure_path(args.metadata_csv, kind="CSV de metadata")
    _ensure_path(args.eval_index, kind="CSV de índices")
    for ckpt in args.checkpoint:
        _ensure_path(ckpt, kind="Checkpoint")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)


def _select_device(identifier: str) -> torch.device:
    device = torch.device(identifier)
    if device.type.startswith("cuda") and not torch.cuda.is_available():
        logging.warning("CUDA no disponible, usando CPU en su lugar")
        return torch.device("cpu")
    return device


def _build_model(args: argparse.Namespace, tokenizer: PreTrainedTokenizerBase) -> MultiStreamClassifier:
    config = ModelConfig(
        image_size=args.image_size,
        projector_dim=args.projector_dim,
        d_model=args.d_model,
        pose_landmarks=args.pose_landmarks,
        projector_dropout=args.projector_dropout,
        fusion_dropout=args.fusion_dropout,
        temporal_nhead=args.temporal_nhead,
        temporal_layers=args.temporal_layers,
        temporal_dim_feedforward=args.temporal_dim_feedforward,
        temporal_dropout=args.temporal_dropout,
        sequence_length=args.sequence_length,
        decoder_layers=args.decoder_layers,
        decoder_heads=args.decoder_heads,
        decoder_dropout=args.decoder_dropout,
        pretrained=args.pretrained,
        pretrained_checkpoint=args.pretrained_checkpoint,
    )
    return MultiStreamClassifier(config, tokenizer)


def _load_checkpoint(model: nn.Module, path: Path, device: torch.device) -> None:
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get("model_state") or checkpoint.get("state_dict")
        if state_dict is None:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)


def _create_dataloader(args: argparse.Namespace) -> DataLoader:
    dataset = LsaTMultiStream(
        face_dir=str(args.face_dir),
        hand_l_dir=str(args.hand_left_dir),
        hand_r_dir=str(args.hand_right_dir),
        pose_dir=str(args.pose_dir),
        csv_path=str(args.metadata_csv),
        index_csv=str(args.eval_index),
        T=args.sequence_length,
        img_size=args.image_size,
        lkp_count=args.pose_landmarks,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        collate_fn=collate_fn,
    )


def _prepare_inputs(batch: dict, device: torch.device) -> dict:
    tensor_keys = ["face", "hand_l", "hand_r", "pose", "pad_mask", "miss_mask_hl", "miss_mask_hr"]
    inputs = {key: batch[key].to(device) for key in tensor_keys}
    inputs["encoder_attention_mask"] = batch["pad_mask"].to(device=device, dtype=torch.long)
    return inputs


def _clean_token_sequences(
    sequences: Iterable[Sequence[int]],
    *,
    pad_token_id: int,
    eos_token_id: int,
) -> List[List[int]]:
    cleaned: List[List[int]] = []
    for seq in sequences:
        tokens: List[int] = []
        for token in seq:
            if token == eos_token_id:
                break
            if token == pad_token_id:
                continue
            tokens.append(int(token))
        cleaned.append(tokens)
    return cleaned


def _predict(
    model: MultiStreamClassifier,
    loader: DataLoader,
    device: torch.device,
    tokenizer: PreTrainedTokenizerBase,
    *,
    max_length: int,
    num_beams: int,
    max_new_tokens: Optional[int],
) -> Tuple[List[PredictionItem], List[float]]:
    results: List[PredictionItem] = []
    latencies: List[float] = []
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            inputs = _prepare_inputs(batch, device)
            pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
            eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else pad_id
            generation_kwargs = {
                "num_beams": max(1, num_beams),
                "early_stopping": num_beams > 1,
                "pad_token_id": pad_id,
                "eos_token_id": eos_id,
            }
            if max_new_tokens is not None:
                generation_kwargs["max_new_tokens"] = max_new_tokens
            else:
                generation_kwargs["max_length"] = max_length
            generation_kwargs.setdefault("min_length", 2)
            start = time.perf_counter()
            sequences = model.generate(**inputs, **generation_kwargs)
            elapsed_ms = (time.perf_counter() - start) * 1000.0
            if hasattr(sequences, "sequences"):
                sequences_tensor = sequences.sequences  # type: ignore[attr-defined]
            else:
                sequences_tensor = sequences
            decoded = decode(tokenizer, sequences_tensor)
            references = batch.get("texts") or [""] * len(decoded)
            batch_size = len(decoded) if decoded else 1
            sample_latency = elapsed_ms / max(batch_size, 1)
            latencies.extend([sample_latency] * len(decoded))
            for video_id, text, ref in zip(batch["video_ids"], decoded, references):
                results.append(
                    PredictionItem(
                        video_id=video_id,
                        prediction=text,
                        reference=ref,
                        latency_ms=sample_latency,
                    )
                )
    return results, latencies


def _write_csv(path: Path, rows: Iterable[PredictionItem]) -> None:
    temp_file = None
    with NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(path.parent)) as tmp:
        writer = csv.writer(tmp)
        writer.writerow(["video_id", "prediction", "reference", "latency_ms"])
        for row in rows:
            latency = "" if row.latency_ms is None else f"{row.latency_ms:.6f}"
            writer.writerow([row.video_id, row.prediction, row.reference, latency])
        temp_file = Path(tmp.name)
    if temp_file is None:
        raise RuntimeError("No se pudo escribir el archivo temporal de predicciones")
    temp_file.replace(path)


def _compute_metrics(
    references: Sequence[str], predictions: Sequence[str]
) -> Tuple[float, float, float, float]:
    if not references or not predictions:
        return 0.0, 0.0, 0.0, 0.0
    if BLEU is None or CHRF is None:
        logging.warning("sacrebleu no disponible, las métricas BLEU/ChrF se devolverán en 0.0")
        cer = character_error_rate(references, predictions)
        wer = word_error_rate(references, predictions)
        return 0.0, 0.0, cer, wer
    bleu_metric = BLEU(tokenize="13a")
    chrf_metric = CHRF()
    bleu_score = bleu_metric.corpus_score(predictions, [references]).score
    chrf_score = chrf_metric.corpus_score(predictions, [references]).score
    cer_score = character_error_rate(references, predictions)
    wer_score = word_error_rate(references, predictions)
    return float(bleu_score), float(chrf_score), float(cer_score), float(wer_score)


def _percentile(data: Sequence[float], percentile: float) -> float:
    if not data:
        return 0.0
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("percentile debe estar entre 0.0 y 1.0")
    sorted_data = sorted(data)
    if len(sorted_data) == 1:
        return sorted_data[0]
    index = (len(sorted_data) - 1) * percentile
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return sorted_data[int(index)]
    lower_value = sorted_data[lower]
    upper_value = sorted_data[upper]
    weight = index - lower
    return lower_value + (upper_value - lower_value) * weight


def _compute_latency_stats(latencies_ms: Sequence[float]) -> Dict[str, float]:
    if not latencies_ms:
        return {
            "avg_latency_ms": 0.0,
            "std_latency_ms": 0.0,
            "p50_latency_ms": 0.0,
            "p95_latency_ms": 0.0,
            "min_latency_ms": 0.0,
            "max_latency_ms": 0.0,
        }
    avg = statistics.fmean(latencies_ms)
    std = statistics.pstdev(latencies_ms) if len(latencies_ms) > 1 else 0.0
    return {
        "avg_latency_ms": float(avg),
        "std_latency_ms": float(std),
        "p50_latency_ms": float(_percentile(latencies_ms, 0.5)),
        "p95_latency_ms": float(_percentile(latencies_ms, 0.95)),
        "min_latency_ms": float(min(latencies_ms)),
        "max_latency_ms": float(max(latencies_ms)),
    }


def _aggregate_metric_sets(metric_sets: Sequence[Mapping[str, float]]) -> Dict[str, Dict[str, float]]:
    aggregated: Dict[str, Dict[str, float]] = {}
    if not metric_sets:
        return aggregated
    grouped: Dict[str, List[float]] = {}
    for metrics in metric_sets:
        for key, value in metrics.items():
            grouped.setdefault(key, []).append(float(value))
    for key, values in grouped.items():
        mean = statistics.fmean(values)
        std = statistics.pstdev(values) if len(values) > 1 else 0.0
        aggregated[key] = {
            "mean": float(mean),
            "std": float(std),
            "min": float(min(values)),
            "max": float(max(values)),
            "count": int(len(values)),
        }
    return aggregated


def _select_examples(predictions: Sequence[PredictionItem], limit: int = 5) -> List[PredictionItem]:
    return list(predictions[: min(limit, len(predictions))])


def _sanitize_checkpoint_name(checkpoint: Path) -> str:
    name = checkpoint.stem or checkpoint.name
    sanitized = [
        ch if ch.isalnum() or ch in {"-", "_"} else "_"
        for ch in name
    ]
    return "".join(sanitized).strip("_") or "checkpoint"


def _resolve_output_path(base_path: Path, checkpoint: Path, multiple: bool) -> Path:
    if not multiple:
        return base_path
    suffix = base_path.suffix
    stem = base_path.stem
    checkpoint_name = _sanitize_checkpoint_name(checkpoint)
    return base_path.with_name(f"{stem}__{checkpoint_name}{suffix}")


def _write_reports(
    output_dir: Path,
    *,
    aggregate: Mapping[str, Mapping[str, float]],
    results: Sequence[EvaluationResult],
) -> None:
    report = {
        "aggregate": aggregate,
        "checkpoints": [
            {
                "checkpoint": str(result.checkpoint),
                "metrics": {key: float(value) for key, value in result.metrics.items()},
                "examples": [
                    {
                        "video_id": item.video_id,
                        "prediction": item.prediction,
                        "reference": item.reference,
                        "latency_ms": item.latency_ms,
                    }
                    for item in result.examples
                ],
            }
            for result in results
        ],
    }

    json_tmp: Optional[Path] = None
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=str(output_dir)) as tmp:
        json.dump(report, tmp, ensure_ascii=False, indent=2)
        json_tmp = Path(tmp.name)
    if json_tmp is None:
        raise RuntimeError("No se pudo escribir el archivo JSON de métricas")
    json_tmp.replace(output_dir / "report.json")

    csv_tmp: Optional[Path] = None
    with NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(output_dir)) as tmp:
        writer = csv.writer(tmp)
        writer.writerow(["type", "checkpoint", "name", "value", "reference"])
        for metric_name, stats in aggregate.items():
            for stat_name, value in stats.items():
                writer.writerow(["aggregate", "ALL", f"{metric_name}_{stat_name}", value, ""])
        for result in results:
            for metric_name, value in result.metrics.items():
                writer.writerow(["metric", result.checkpoint.name, metric_name, value, ""])
            for item in result.examples:
                writer.writerow([
                    "example",
                    result.checkpoint.name,
                    item.video_id,
                    item.prediction,
                    item.reference,
                ])
        csv_tmp = Path(tmp.name)
    if csv_tmp is None:
        raise RuntimeError("No se pudo escribir el archivo CSV de métricas")
    csv_tmp.replace(output_dir / "report.csv")


def run(argv: Optional[Sequence[str]] = None) -> List[PredictionItem]:
    args = parse_args(argv)
    _setup_logging(args.log_level)
    _validate_inputs(args)
    device = _select_device(args.device)

    tokenizer = create_tokenizer(args.tokenizer)
    if hasattr(tokenizer, "encode"):
        try:
            validate_tokenizer(tokenizer, allow_empty_decode=True)
        except TokenizerValidationError as exc:
            raise RuntimeError(f"Tokenizer inválido: {exc}") from exc

    loader = _create_dataloader(args)
    multiple_checkpoints = len(args.checkpoint) > 1
    evaluation_results: List[EvaluationResult] = []

    for checkpoint_path in args.checkpoint:
        logging.info("Evaluando checkpoint %s", checkpoint_path)
        model = _build_model(args, tokenizer).to(device)
        _load_checkpoint(model, checkpoint_path, device)
        predictions, latencies = _predict(
            model,
            loader,
            device,
            tokenizer,
            max_length=args.max_target_length,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
        )

        output_csv = _resolve_output_path(args.output_csv, checkpoint_path, multiple_checkpoints)
        _write_csv(output_csv, predictions)
        references = [item.reference for item in predictions]
        texts = [item.prediction for item in predictions]
        latency_metrics = _compute_latency_stats(latencies)
        metrics = dict(latency_metrics)
        if args.compute_metrics:
            bleu, chrf, cer, wer = _compute_metrics(references, texts)
            metrics.update({
                "bleu": bleu,
                "chrf": chrf,
                "cer": cer,
                "wer": wer,
            })
        else:
            bleu = chrf = cer = wer = float("nan")
        examples = _select_examples(predictions)
        evaluation_results.append(
            EvaluationResult(
                checkpoint=checkpoint_path,
                predictions=predictions,
                metrics=metrics,
                examples=examples,
            )
        )
        if args.compute_metrics:
            logging.info(
                "Métricas %s - BLEU: %.2f, ChrF: %.2f, CER: %.2f, WER: %.2f, Latencia media: %.2f ms",
                checkpoint_path.name,
                bleu,
                chrf,
                cer,
                wer,
                metrics["avg_latency_ms"],
            )
        else:
            logging.info(
                "Métricas %s - métricas de texto omitidas, latencia media: %.2f ms",
                checkpoint_path.name,
                metrics["avg_latency_ms"],
            )
        logging.info("Predicciones guardadas en %s", output_csv)
        del model

    aggregate = _aggregate_metric_sets([result.metrics for result in evaluation_results])
    report_dir = args.output_csv.parent
    _write_reports(report_dir, aggregate=aggregate, results=evaluation_results)
    logging.info("Reportes agregados guardados en %s", report_dir)
    if aggregate:
        for metric_name, stats in aggregate.items():
            logging.info(
                "Resumen %s -> media=%.4f, std=%.4f, min=%.4f, max=%.4f (n=%d)",
                metric_name,
                stats.get("mean", 0.0),
                stats.get("std", 0.0),
                stats.get("min", 0.0),
                stats.get("max", 0.0),
                int(stats.get("count", 0)),
            )

    return evaluation_results[0].predictions if evaluation_results else []


def main(argv: Optional[Sequence[str]] = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover - punto de entrada CLI
    raise SystemExit(main())
