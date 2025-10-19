#!/usr/bin/env python3
"""Evalúa el modelo multi-stream generando predicciones y métricas."""

import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from slt.data import LsaTMultiStream, collate_fn
from slt.models import MultiStreamEncoder, TextSeq2SeqDecoder, ViTConfig
from slt.utils.text import create_tokenizer, decode

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


@dataclass
class ModelConfig:
    """Configuración del modelo stub utilizada durante la evaluación."""

    image_size: int = 224
    projector_dim: int = 256
    d_model: int = 512
    pose_landmarks: int = 13
    projector_dropout: float = 0.0
    fusion_dropout: float = 0.0
    temporal_nhead: int = 8
    temporal_layers: int = 6
    temporal_dim_feedforward: int = 2048
    temporal_dropout: float = 0.1
    sequence_length: int = 128
    decoder_layers: int = 2
    decoder_heads: int = 8
    decoder_dropout: float = 0.1


class MultiStreamClassifier(nn.Module):
    """Ensamble encoder/decoder utilizado tanto en entrenamiento como en inferencia."""

    def __init__(self, config: ModelConfig, tokenizer: PreTrainedTokenizerBase) -> None:
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
    parser.add_argument("--checkpoint", type=Path, required=True, help="Ruta al checkpoint a cargar")
    parser.add_argument("--output-csv", type=Path, required=True, help="Archivo de salida para escribir predicciones")

    parser.add_argument("--batch-size", type=int, default=4, help="Tamaño de batch para la evaluación")
    parser.add_argument("--num-workers", type=int, default=0, help="Número de workers del DataLoader")
    parser.add_argument("--device", type=str, default="cuda", help="Dispositivo torch (cuda o cpu)")
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false", help="Deshabilita pin_memory en DataLoader")
    parser.add_argument("--pin-memory", dest="pin_memory", action="store_true", help="Habilita pin_memory en DataLoader")
    parser.set_defaults(pin_memory=True)

    parser.add_argument("--image-size", type=int, default=224, help="Tamaño de imagen esperado por el ViT stub")
    parser.add_argument("--projector-dim", type=int, default=256, help="Dimensión de los proyectores de stream")
    parser.add_argument("--d-model", type=int, default=512, help="Dimensión del espacio fusionado")
    parser.add_argument("--pose-landmarks", type=int, default=13, help="Cantidad de landmarks en el stream de pose")
    parser.add_argument("--projector-dropout", type=float, default=0.0, help="Dropout aplicado a proyectores")
    parser.add_argument("--fusion-dropout", type=float, default=0.0, help="Dropout aplicado tras la fusión")
    parser.add_argument("--temporal-nhead", type=int, default=8, help="Número de cabezales de atención temporal")
    parser.add_argument("--temporal-layers", type=int, default=6, help="Capas del codificador temporal")
    parser.add_argument(
        "--temporal-dim-feedforward",
        type=int,
        default=2048,
        help="Dimensión del feedforward en el codificador temporal",
    )
    parser.add_argument("--temporal-dropout", type=float, default=0.1, help="Dropout del codificador temporal")
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
        default=8,
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

    parser.add_argument("--num-beams", type=int, default=1, help="Cantidad de beams para la decodificación")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Cantidad máxima de tokens generados autoregresivamente",
    )

    parser.add_argument("--log-level", default="INFO", help="Nivel de logging (INFO, DEBUG, ...)")

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
    _ensure_path(args.checkpoint, kind="Checkpoint")
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
) -> List[PredictionItem]:
    results: List[PredictionItem] = []
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
            sequences = model.generate(**inputs, **generation_kwargs)
            if hasattr(sequences, "sequences"):
                sequences_tensor = sequences.sequences  # type: ignore[attr-defined]
            else:
                sequences_tensor = sequences
            decoded = decode(tokenizer, sequences_tensor)
            references = batch.get("texts") or [""] * len(decoded)
            for video_id, text, ref in zip(batch["video_ids"], decoded, references):
                results.append(PredictionItem(video_id=video_id, prediction=text, reference=ref))
    return results


def _write_csv(path: Path, rows: Iterable[PredictionItem]) -> None:
    temp_file = None
    with NamedTemporaryFile("w", encoding="utf-8", newline="", delete=False, dir=str(path.parent)) as tmp:
        writer = csv.writer(tmp)
        writer.writerow(["video_id", "prediction", "reference"])
        for row in rows:
            writer.writerow([row.video_id, row.prediction, row.reference])
        temp_file = Path(tmp.name)
    if temp_file is None:
        raise RuntimeError("No se pudo escribir el archivo temporal de predicciones")
    temp_file.replace(path)


def _levenshtein_distance(reference: str, prediction: str) -> int:
    if reference == prediction:
        return 0
    if not reference:
        return len(prediction)
    if not prediction:
        return len(reference)
    prev_row = list(range(len(prediction) + 1))
    for i, ref_char in enumerate(reference, start=1):
        current = [i]
        for j, pred_char in enumerate(prediction, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (ref_char != pred_char)
            current.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = current
    return prev_row[-1]


def _character_error_rate(references: Sequence[str], predictions: Sequence[str]) -> float:
    total_distance = 0
    total_length = 0
    for ref, pred in zip(references, predictions):
        ref_text = ref or ""
        pred_text = pred or ""
        total_distance += _levenshtein_distance(ref_text, pred_text)
        total_length += max(len(ref_text), 1)
    if total_length == 0:
        return 0.0
    return (total_distance / total_length) * 100.0


def _compute_metrics(references: Sequence[str], predictions: Sequence[str]) -> Tuple[float, float, float]:
    if not references or not predictions:
        return 0.0, 0.0, 0.0
    if BLEU is None or CHRF is None:
        logging.warning("sacrebleu no disponible, las métricas BLEU/ChrF se devolverán en 0.0")
        cer = _character_error_rate(references, predictions)
        return 0.0, 0.0, cer
    bleu_metric = BLEU(tokenize="13a")
    chrf_metric = CHRF()
    bleu_score = bleu_metric.corpus_score(predictions, [references]).score
    chrf_score = chrf_metric.corpus_score(predictions, [references]).score
    cer_score = _character_error_rate(references, predictions)
    return float(bleu_score), float(chrf_score), float(cer_score)


def _select_examples(predictions: Sequence[PredictionItem], limit: int = 5) -> List[PredictionItem]:
    return list(predictions[: min(limit, len(predictions))])


def _write_reports(
    output_dir: Path,
    *,
    metrics: Mapping[str, float],
    examples: Sequence[PredictionItem],
) -> None:
    report = {
        "metrics": {key: float(value) for key, value in metrics.items()},
        "examples": [
            {
                "video_id": item.video_id,
                "prediction": item.prediction,
                "reference": item.reference,
            }
            for item in examples
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
        writer.writerow(["type", "name", "value", "reference"])
        for key in sorted(metrics):
            writer.writerow(["metric", key, metrics[key], ""])
        for item in examples:
            writer.writerow(["example", item.video_id, item.prediction, item.reference])
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
    loader = _create_dataloader(args)
    model = _build_model(args, tokenizer).to(device)
    _load_checkpoint(model, args.checkpoint, device)

    predictions = _predict(
        model,
        loader,
        device,
        tokenizer,
        max_length=args.max_target_length,
        num_beams=args.num_beams,
        max_new_tokens=args.max_new_tokens,
    )
    _write_csv(args.output_csv, predictions)
    references = [item.reference for item in predictions]
    texts = [item.prediction for item in predictions]
    bleu, chrf, cer = _compute_metrics(references, texts)
    metrics = {"bleu": bleu, "chrf": chrf, "cer": cer}
    examples = _select_examples(predictions)
    _write_reports(args.output_csv.parent, metrics=metrics, examples=examples)
    logging.info("Predicciones guardadas en %s", args.output_csv)
    logging.info("Métricas - BLEU: %.2f, ChrF: %.2f, CER: %.2f", bleu, chrf, cer)
    return predictions


def main(argv: Optional[Sequence[str]] = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":  # pragma: no cover - punto de entrada CLI
    raise SystemExit(main())
