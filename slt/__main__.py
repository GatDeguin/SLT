"""Entrada de demostración para ``python -m slt``.

El módulo ejecuta un entrenamiento corto utilizando los mismos componentes que
la CLI completa de ``tools/train_slt_multistream_v9.py``. Acepta archivos de
configuración externos (JSON/YAML) y permite sobreescribir parámetros mediante
la bandera ``--set`` para facilitar experimentos rápidos. De forma
predeterminada el modelo se inicializa con los pesos validados para el flujo
``single_signer`` siempre que el checkpoint haya sido descargado y esté
disponible localmente."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional

import torch
import torch.nn.functional as F

from .data import LsaTMultiStream, collate_fn
from .training.configuration import resolve_configs
from .training.data import create_dataloader, normalise_mix_spec
from .training.loops import eval_epoch, train_epoch
from .training.models import MultiStreamClassifier
from .training.optim import create_optimizer
from .utils.general import set_seed
from .utils.text import create_tokenizer

_DemoModel = MultiStreamClassifier


def _build_collate(tokenizer, *, max_length: int):
    """Return a collate function that tokenizes targets using ``tokenizer``."""

    def collate(batch):
        collated = collate_fn(batch)
        inputs = {
            "face": collated["face"],
            "hand_l": collated["hand_l"],
            "hand_r": collated["hand_r"],
            "pose": collated["pose"],
            "pose_conf_mask": collated["pose_conf_mask"],
            "pad_mask": collated["pad_mask"],
            "encoder_attention_mask": collated["pad_mask"].to(torch.long),
            "lengths": collated["lengths"],
            "miss_mask_hl": collated.get("miss_mask_hl"),
            "miss_mask_hr": collated.get("miss_mask_hr"),
            "keypoints": collated["keypoints"],
            "keypoints_mask": collated["keypoints_mask"],
            "keypoints_frame_mask": collated["keypoints_frame_mask"],
            "keypoints_body": collated["keypoints_body"],
            "keypoints_body_mask": collated["keypoints_body_mask"],
            "keypoints_body_frame_mask": collated["keypoints_body_frame_mask"],
            "keypoints_hand_l": collated["keypoints_hand_l"],
            "keypoints_hand_l_mask": collated["keypoints_hand_l_mask"],
            "keypoints_hand_l_frame_mask": collated["keypoints_hand_l_frame_mask"],
            "keypoints_hand_r": collated["keypoints_hand_r"],
            "keypoints_hand_r_mask": collated["keypoints_hand_r_mask"],
            "keypoints_hand_r_frame_mask": collated["keypoints_hand_r_frame_mask"],
            "keypoints_face": collated["keypoints_face"],
            "keypoints_face_mask": collated["keypoints_face_mask"],
            "keypoints_face_frame_mask": collated["keypoints_face_frame_mask"],
            "keypoints_lengths": collated["keypoints_lengths"],
            "ctc_labels": collated["ctc_labels"],
            "ctc_mask": collated["ctc_mask"],
            "ctc_lengths": collated["ctc_lengths"],
        }
        tokenized = tokenizer(
            collated["texts"],
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        targets = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }
        inputs["gloss_sequences"] = collated["gloss_sequences"]
        inputs["gloss_texts"] = collated["gloss_texts"]
        return {
            "inputs": inputs,
            "targets": targets,
            "texts": collated["texts"],
            "video_ids": collated["video_ids"],
        }

    return collate


def _select_device(device_flag: str) -> torch.device:
    if device_flag.lower() == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_flag)


def _build_cli_overrides(
    args: argparse.Namespace,
    mix_streams: dict[str, float] | None,
) -> dict[str, dict[str, object]]:
    overrides: dict[str, dict[str, object]] = {
        "data": {
            "face_dir": args.face_dir,
            "hand_left_dir": args.hand_left_dir,
            "hand_right_dir": args.hand_right_dir,
            "pose_dir": args.pose_dir,
            "metadata_csv": args.metadata_csv,
            "train_index": args.train_index,
            "val_index": args.val_index,
            "work_dir": args.work_dir,
        },
        "model": {},
        "optim": {},
        "training": {},
    }
    data_section = overrides["data"]
    if args.batch_size is not None:
        data_section["batch_size"] = args.batch_size
        data_section["val_batch_size"] = args.batch_size
    if getattr(args, "val_batch_size", None) is not None:
        data_section["val_batch_size"] = args.val_batch_size
    if args.num_workers is not None:
        data_section["num_workers"] = args.num_workers
    if args.no_pin_memory:
        data_section["pin_memory"] = False
    if args.tokenizer is not None:
        data_section["tokenizer"] = args.tokenizer
    if args.max_target_length is not None:
        data_section["max_target_length"] = args.max_target_length
    if getattr(args, "keypoints_dir", None) is not None:
        data_section["keypoints_dir"] = args.keypoints_dir
    if getattr(args, "gloss_csv", None) is not None:
        data_section["gloss_csv"] = args.gloss_csv
    if args.device is not None:
        data_section["device"] = args.device
    if getattr(args, "precision", None) is not None:
        precision_flag = args.precision
        if precision_flag == "float32":
            precision_flag = "fp32"
        data_section["precision"] = precision_flag
    if args.seed is not None:
        data_section["seed"] = args.seed
    if mix_streams:
        data_section["mix_streams"] = mix_streams

    model_section = overrides["model"]
    if args.sequence_length is not None:
        model_section["sequence_length"] = args.sequence_length
    if args.image_size is not None:
        model_section["image_size"] = args.image_size
    if args.decoder_layers is not None:
        model_section["decoder_layers"] = args.decoder_layers
    if args.decoder_heads is not None:
        model_section["decoder_heads"] = args.decoder_heads
    if args.decoder_dropout is not None:
        model_section["decoder_dropout"] = args.decoder_dropout
    pretrained = getattr(args, "pretrained", None)
    if pretrained is not None:
        model_section["pretrained"] = pretrained
    checkpoint = getattr(args, "pretrained_checkpoint", None)
    if checkpoint is not None:
        model_section["pretrained_checkpoint"] = checkpoint

    if args.lr is not None:
        overrides["optim"]["lr"] = args.lr
    if args.epochs is not None:
        overrides["training"]["epochs"] = args.epochs
    return overrides


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Entrenamiento corto del modelo multi-stream validado (demo).",
    )
    parser.add_argument(
        "--config",
        type=Path,
        help="Plantilla de configuración JSON o YAML",
    )
    parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Sobrescribe valores utilizando claves con puntos (ej. data.batch_size=4)",
    )

    config_group = parser.add_argument_group(
        "Menú de configuración",
        "Define rutas base e importación de CSV para el demo.",
    )
    config_group.add_argument(
        "--face-dir",
        type=Path,
        required=True,
        help="Carpeta con frames de rostro",
    )
    config_group.add_argument(
        "--hand-left-dir",
        type=Path,
        required=True,
        help="Carpeta con frames de mano izquierda",
    )
    config_group.add_argument(
        "--hand-right-dir",
        type=Path,
        required=True,
        help="Carpeta con frames de mano derecha",
    )
    config_group.add_argument(
        "--pose-dir",
        type=Path,
        required=True,
        help="Carpeta con archivos .npz de pose",
    )
    config_group.add_argument(
        "--keypoints-dir",
        type=Path,
        help="Carpeta con keypoints MediaPipe (.npy/.npz)",
    )
    config_group.add_argument(
        "--metadata-csv",
        type=Path,
        required=True,
        help="CSV con columnas video_id;texto",
    )
    config_group.add_argument(
        "--train-index",
        type=Path,
        required=True,
        help="CSV con lista de video_id para entrenamiento",
    )
    config_group.add_argument(
        "--val-index",
        type=Path,
        required=True,
        help="CSV con lista de video_id para validación",
    )
    config_group.add_argument(
        "--gloss-csv",
        type=Path,
        help="CSV opcional con columnas video_id;gloss;ctc_labels",
    )
    config_group.add_argument(
        "--work-dir",
        type=Path,
        default=Path("work_dirs/demo"),
        help="Directorio donde guardar checkpoints",
    )

    training_group = parser.add_argument_group(
        "Entrenamiento",
        "Opciones principales del loop de entrenamiento.",
    )
    training_group.add_argument("--batch-size", type=int, help="Tamaño de batch")
    training_group.add_argument(
        "--val-batch-size",
        type=int,
        help="Batch de validación",
    )
    training_group.add_argument(
        "--epochs",
        type=int,
        help="Cantidad de épocas de entrenamiento",
    )
    training_group.add_argument(
        "--lr",
        type=float,
        help="Learning rate del optimizador",
    )
    training_group.add_argument(
        "--num-workers",
        type=int,
        help="Workers de DataLoader",
    )
    training_group.add_argument(
        "--no-pin-memory",
        action="store_true",
        help="Deshabilita pinned memory en los loaders",
    )
    training_group.add_argument("--seed", type=int, help="Semilla aleatoria")
    training_group.add_argument(
        "--no-amp",
        action="store_true",
        help="Desactiva AMP incluso si hay GPU disponible",
    )

    model_group = parser.add_argument_group(
        "Modelo",
        "Hiper-parámetros del encoder y decoder utilizados en la demo.",
    )
    model_group.add_argument(
        "--sequence-length",
        type=int,
        help="Número de frames muestreados por clip",
    )
    model_group.add_argument(
        "--image-size",
        type=int,
        help="Resolución de entrada para los backbones",
    )
    model_group.add_argument(
        "--tokenizer",
        type=str,
        help="Identificador o ruta a un tokenizer de HuggingFace",
    )
    model_group.add_argument(
        "--max-target-length",
        type=int,
        default=None,
        help="Longitud máxima de las secuencias de texto tokenizadas",
    )
    model_group.add_argument(
        "--decoder-layers",
        type=int,
        default=None,
        help="Capas del decoder seq2seq utilizado durante la demo",
    )
    model_group.add_argument(
        "--decoder-heads",
        type=int,
        default=None,
        help="Número de cabezas de atención en el decoder seq2seq",
    )
    model_group.add_argument(
        "--decoder-dropout",
        type=float,
        default=None,
        help="Dropout aplicado dentro del decoder seq2seq",
    )
    model_group.add_argument(
        "--pretrained",
        type=str,
        default=None,
        help="Identificador de pesos pre-entrenados (single_signer o none)",
    )
    model_group.add_argument(
        "--pretrained-checkpoint",
        type=Path,
        default=None,
        help=(
            "Ruta al checkpoint single_signer ya descargado. Se usa cuando "
            "--pretrained está activo."
        ),
    )
    model_group.add_argument(
        "--mix-stream",
        dest="mix_streams",
        action="append",
        default=[],
        metavar="STREAM[:P]",
        help=(
            "Permuta aleatoriamente streams individuales con probabilidad P "
            "(face, hand-left, hand-right, pose)"
        ),
    )

    runtime_group = parser.add_argument_group(
        "Ejecución",
        "Control del dispositivo, precisión numérica y flags relacionados.",
    )
    runtime_group.add_argument(
        "--device",
        type=str,
        help="Dispositivo torch (auto, cpu, cuda, cuda:0, ...)",
    )
    runtime_group.add_argument(
        "--precision",
        choices=["amp", "fp32", "float32"],
        help="Precisión numérica. 'amp' usa mixed precision en GPU",
    )
    return parser.parse_args()


def _parse_mix_streams(raw: list[str]) -> dict[str, float]:
    if not raw:
        return {}
    mix_spec: dict[str, float] = {}
    for entry in raw:
        name, _, prob_text = entry.partition(":")
        name = name.strip()
        prob = 1.0
        if prob_text:
            try:
                prob = float(prob_text)
            except ValueError as exc:
                raise ValueError(f"Probabilidad inválida para --mix-stream '{entry}'") from exc
        mix_spec[name] = prob
    return normalise_mix_spec(mix_spec)


def main() -> None:
    args = parse_args()
    mix_streams_arg = getattr(args, "mix_streams", None)
    mix_streams = _parse_mix_streams(mix_streams_arg) if mix_streams_arg else None

    base_defaults = {
        "data": {
            "batch_size": 2,
            "val_batch_size": 2,
            "num_workers": 0,
            "pin_memory": True,
            "max_target_length": 64,
            "device": "auto",
            "seed": 1234,
        },
        "model": {
            "sequence_length": 64,
            "image_size": 224,
        },
        "training": {
            "epochs": 2,
        },
    }

    cli_overrides = _build_cli_overrides(args, mix_streams)

    data_config, model_config, optim_config, training_config, _ = resolve_configs(
        config_path=getattr(args, "config", None),
        cli_overrides=cli_overrides,
        set_overrides=getattr(args, "overrides", []),
        base=base_defaults,
    )

    try:
        data_config.mix_streams = normalise_mix_spec(data_config.mix_streams or {})
    except ValueError as exc:  # pragma: no cover - configuration error
        raise ValueError(f"Configuración de mezcla inválida: {exc}") from exc

    set_seed(data_config.seed)

    device = _select_device(data_config.device)
    precision_flag = (data_config.precision or "amp").lower()
    if precision_flag == "float32":
        precision_flag = "fp32"
    use_amp = (
        precision_flag == "amp"
        and device.type == "cuda"
        and torch.cuda.is_available()
        and not getattr(args, "no_amp", False)
    )
    if device.type.startswith("cuda") and not torch.cuda.is_available():  # pragma: no cover
        device = torch.device("cpu")

    train_dataset = LsaTMultiStream(
        face_dir=str(data_config.face_dir),
        hand_l_dir=str(data_config.hand_left_dir),
        hand_r_dir=str(data_config.hand_right_dir),
        pose_dir=str(data_config.pose_dir),
        csv_path=str(data_config.metadata_csv),
        index_csv=str(data_config.train_index),
        keypoints_dir=str(data_config.keypoints_dir)
        if data_config.keypoints_dir
        else None,
        gloss_csv=str(data_config.gloss_csv) if data_config.gloss_csv else None,
        T=model_config.sequence_length,
        img_size=model_config.image_size,
        lkp_count=model_config.pose_landmarks,
    )
    val_dataset = LsaTMultiStream(
        face_dir=str(data_config.face_dir),
        hand_l_dir=str(data_config.hand_left_dir),
        hand_r_dir=str(data_config.hand_right_dir),
        pose_dir=str(data_config.pose_dir),
        csv_path=str(data_config.metadata_csv),
        index_csv=str(data_config.val_index),
        keypoints_dir=str(data_config.keypoints_dir)
        if data_config.keypoints_dir
        else None,
        gloss_csv=str(data_config.gloss_csv) if data_config.gloss_csv else None,
        T=model_config.sequence_length,
        img_size=model_config.image_size,
        lkp_count=model_config.pose_landmarks,
    )

    tokenizer_source = (
        data_config.tokenizer or model_config.decoder_model or args.tokenizer
    )
    if tokenizer_source is None:
        raise ValueError(
            "Debe especificarse un tokenizer en la CLI o en el archivo de "
            "configuración"
        )
    tokenizer = create_tokenizer(tokenizer_source)

    train_loader = create_dataloader(
        train_dataset,
        batch_size=data_config.batch_size,
        shuffle=True,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        tokenizer=tokenizer,
        max_length=data_config.max_target_length,
        mix_streams=data_config.mix_streams,
        seed=data_config.seed,
    )
    val_loader = create_dataloader(
        val_dataset,
        batch_size=data_config.val_batch_size or data_config.batch_size,
        shuffle=False,
        num_workers=data_config.num_workers,
        pin_memory=data_config.pin_memory,
        tokenizer=tokenizer,
        max_length=data_config.max_target_length,
        mix_streams=None,
        seed=data_config.seed,
    )

    model = _DemoModel(model_config, tokenizer).to(device)

    optim_cfg = {
        "type": optim_config.optimizer,
        "lr": optim_config.lr,
        "weight_decay": optim_config.weight_decay,
    }
    optimizer = create_optimizer(model.parameters(), optim_cfg)

    scaler: torch.cuda.amp.GradScaler | None = None
    autocast_dtype: torch.dtype | None = None
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        autocast_dtype = torch.float16

    def _loss_fn(
        outputs: Any,
        targets: torch.Tensor,
        _: Optional[Mapping[str, Any]] = None,
    ) -> torch.Tensor:
        logits = outputs.logits if hasattr(outputs, "logits") else outputs
        vocab = logits.size(-1)
        return F.cross_entropy(
            logits.view(-1, vocab),
            targets.view(-1),
            ignore_index=-100,
            label_smoothing=optim_config.label_smoothing,
        )

    data_config.work_dir.mkdir(parents=True, exist_ok=True)

    best_val = float("inf")
    for epoch in range(1, training_config.epochs + 1):
        train_result = train_epoch(
            model,
            train_loader,
            optimizer,
            _loss_fn,
            device=device,
            scaler=scaler,
            autocast_dtype=autocast_dtype,
            grad_clip_norm=optim_config.grad_clip_norm,
            grad_accum_steps=training_config.grad_accum_steps,
        )
        val_result = eval_epoch(
            model,
            val_loader,
            _loss_fn,
            device=device,
        )
        train_loss = getattr(train_result, "loss", train_result)
        val_loss = getattr(val_result, "loss", val_result)
        torch.save(model.state_dict(), data_config.work_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), data_config.work_dir / "best.pt")
        print({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})

    print("Entrenamiento demo completado. Sustituye los stubs por modelos reales para producción.")


if __name__ == "__main__":
    main()
