"""Shared entry point for DINO/iBOT pre-training scripts."""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

try:  # pragma: no cover - Python 3.10 compatibility
    import tomllib  # type: ignore[attr-defined]
except ModuleNotFoundError:  # pragma: no cover - fallback for Python <=3.10
    import tomli as tomllib  # type: ignore

import torch
from torch import nn

from tools.pretrain_utils import (
    AugmentationConfig,
    BackboneConfig,
    CheckpointState,
    DINOLoss,
    DINOMultiCropAugmentation,
    IBOTLoss,
    ProjectionHead,
    apply_ema,
    build_dataloader,
    build_vit_backbone,
    cosine_scheduler,
    export_for_dinov2,
    ExperimentTracker,
    load_checkpoint,
    mask_patches,
    save_checkpoint,
)

LOGGER = logging.getLogger(__name__)


def _ensure_mapping(value: object) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    return {}


def _load_cli_config(path: Path) -> dict[str, Any]:
    suffix = path.suffix.lower()
    if suffix in {".json"}:
        with path.open("r", encoding="utf8") as handle:
            data = json.load(handle)
    elif suffix in {".toml", ".tml"}:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    else:  # pragma: no cover - defensive
        raise ValueError(
            "Unsupported configuration format. Use JSON or TOML files."
        )

    if not isinstance(data, Mapping):
        raise ValueError("Configuration file must define a mapping at the top level")

    normalized: dict[str, Any] = {}

    simple_keys = {
        "train_dir",
        "output_dir",
        "algorithm",
        "epochs",
        "batch_size",
        "num_workers",
        "learning_rate",
        "weight_decay",
        "warmup_epochs",
        "teacher_momentum_base",
        "teacher_momentum_final",
        "clip_grad_norm",
        "out_dim",
        "head_hidden_dim",
        "patch_out_dim",
        "patch_mask_ratio",
        "patch_loss_weight",
        "student_temp",
        "teacher_temp",
        "center_momentum",
        "seed",
        "device",
        "resume",
        "export_backbone",
        "log_interval",
        "image_size",
        "patch_size",
        "vit_embed_dim",
        "vit_depth",
        "vit_num_heads",
        "vit_mlp_ratio",
        "num_local_crops",
        "local_crop_size",
        "global_crop_scale",
        "local_crop_scale",
        "stream",
    }
    for key in simple_keys:
        if key in data:
            normalized[key] = data[key]

    dataset_section = _ensure_mapping(data.get("dataset") or data.get("dataloader"))
    for key in ("batch_size", "num_workers"):
        if key in dataset_section and key not in normalized:
            normalized[key] = dataset_section[key]
    for key in ("shuffle", "pin_memory", "persistent_workers"):
        if key in dataset_section:
            normalized[f"dataset_{key}"] = dataset_section[key]

    augmentation_section = _ensure_mapping(
        data.get("augmentation") or data.get("augmentations")
    )
    aug_map = {
        "brightness": "aug_brightness",
        "contrast": "aug_contrast",
        "saturation": "aug_saturation",
        "hue": "aug_hue",
        "color_jitter_prob": "aug_color_jitter_prob",
        "grayscale_prob": "aug_grayscale_prob",
        "gaussian_blur_prob": "aug_gaussian_blur_prob",
        "solarize_prob": "aug_solarize_prob",
        "hflip_prob": "aug_hflip_prob",
        "mean": "aug_mean",
        "std": "aug_std",
        "global_crop_scale": "global_crop_scale",
        "local_crop_scale": "local_crop_scale",
        "image_size": "image_size",
        "local_crop_size": "local_crop_size",
        "num_local_crops": "num_local_crops",
    }
    for key, target in aug_map.items():
        if key in augmentation_section and target not in normalized:
            normalized[target] = augmentation_section[key]

    checkpoint_section = _ensure_mapping(
        data.get("checkpoint") or data.get("checkpoints") or data.get("checkpointing")
    )
    checkpoint_map = {
        "best_name": "checkpoint_best_name",
        "last_name": "checkpoint_last_name",
        "history_file": "checkpoint_history_file",
    }
    for key, target in checkpoint_map.items():
        if key in checkpoint_section:
            normalized[target] = checkpoint_section[key]

    experiment_section = _ensure_mapping(data.get("experiment"))
    if "name" in experiment_section:
        normalized["experiment_name"] = experiment_section["name"]
    if "notes" in experiment_section:
        normalized["experiment_notes"] = experiment_section["notes"]
    if "tags" in experiment_section:
        tags = experiment_section["tags"]
        if isinstance(tags, (list, tuple, set)):
            normalized["experiment_tag"] = list(tags)
        else:
            normalized["experiment_tag"] = [str(tags)]

    return normalized


def _resolve_path(value: Path | str, base: Path | None) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute() and base is not None:
        return (base / path).resolve()
    return path.resolve()


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _momentum_schedule(base: float, final: float, total_steps: int) -> list[float]:
    schedule = []
    for step in range(total_steps):
        progress = step / max(total_steps - 1, 1)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        schedule.append(final - (final - base) * cosine)
    return schedule


def _update_teacher(
    student_backbone: nn.Module,
    teacher_backbone: nn.Module,
    student_head: nn.Module,
    teacher_head: nn.Module,
    momentum: float,
    *,
    student_patch_head: nn.Module | None = None,
    teacher_patch_head: nn.Module | None = None,
) -> None:
    apply_ema(student_backbone, teacher_backbone, momentum)
    apply_ema(student_head, teacher_head, momentum)
    if student_patch_head is not None and teacher_patch_head is not None:
        apply_ema(student_patch_head, teacher_patch_head, momentum)


def _compute_dino_loss(
    teacher_outputs: Sequence[torch.Tensor],
    student_outputs: Sequence[torch.Tensor],
    loss_fn: DINOLoss,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=student_outputs[0].device)
    terms = 0
    for iq, teacher_view in enumerate(teacher_outputs):
        for iv, student_view in enumerate(student_outputs):
            if iq == iv:
                continue
            loss = loss + loss_fn(student_view, teacher_view)
            terms += 1
    if terms:
        loss = loss / terms
    return loss


def _parse_args(argv: Iterable[str] | None, default_stream: str) -> argparse.Namespace:
    base_parser = argparse.ArgumentParser(add_help=False)
    base_parser.add_argument(
        "--config",
        type=Path,
        help="Archivo TOML/JSON con parámetros por defecto",
    )

    if argv is None:
        parsed_config, remaining = base_parser.parse_known_args()
    else:
        parsed_config, remaining = base_parser.parse_known_args(list(argv))

    defaults = {}
    if parsed_config.config:
        defaults = _load_cli_config(parsed_config.config)

    parser = argparse.ArgumentParser(
        description="Self-supervised DINO/iBOT pre-training",
        parents=[base_parser],
    )
    parser.set_defaults(stream=default_stream)
    if defaults:
        parser.set_defaults(**defaults)

    parser.add_argument(
        "--train-dir",
        nargs="+",
        type=Path,
        default=None,
        help="Carpetas con recortes del stream",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio donde escribir checkpoints y logs",
    )
    parser.add_argument(
        "--algorithm",
        choices=["dino", "ibot"],
        default="dino",
        help="Algoritmo de entrenamiento a utilizar",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño del batch")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers del DataLoader")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Tasa de aprendizaje inicial")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Decaimiento L2")
    parser.add_argument("--warmup-epochs", type=float, default=1.0, help="Épocas de warmup lineal")
    parser.add_argument(
        "--teacher-momentum-base",
        type=float,
        default=0.996,
        help="Momentum inicial del maestro",
    )
    parser.add_argument(
        "--teacher-momentum-final",
        type=float,
        default=1.0,
        help="Momentum final del maestro",
    )
    parser.add_argument(
        "--clip-grad-norm",
        type=float,
        default=1.0,
        help="Norma máxima para el gradiente (0 desactiva)",
    )
    parser.add_argument("--out-dim", type=int, default=1024, help="Dimensión del proyector DINO")
    parser.add_argument(
        "--head-hidden-dim",
        type=int,
        default=2048,
        help="Dimensión oculta del proyector",
    )
    parser.add_argument(
        "--patch-out-dim",
        type=int,
        default=256,
        help="Dimensión de logits de parches para iBOT",
    )
    parser.add_argument(
        "--patch-mask-ratio",
        type=float,
        default=0.3,
        help="Proporción de parches enmascarados para iBOT",
    )
    parser.add_argument(
        "--patch-loss-weight",
        type=float,
        default=1.0,
        help="Peso relativo de la pérdida de parches en iBOT",
    )
    parser.add_argument("--student-temp", type=float, default=0.1, help="Temperatura del estudiante")
    parser.add_argument("--teacher-temp", type=float, default=0.04, help="Temperatura del maestro")
    parser.add_argument("--center-momentum", type=float, default=0.9, help="Momentum del centro para DINO")
    parser.add_argument("--seed", type=int, default=42, help="Semilla PRNG")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo a utilizar")
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Checkpoint para reanudar entrenamiento",
    )
    parser.add_argument(
        "--export-backbone",
        type=Path,
        default=None,
        help="Ruta donde exportar el backbone compatible con load_dinov2_backbone",
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        help="Intervalo en iteraciones para registro de pérdidas",
    )

    parser.add_argument("--image-size", type=int, default=224, help="Resolución de entrada del ViT")
    parser.add_argument("--patch-size", type=int, default=16, help="Tamaño de parche del ViT")
    parser.add_argument("--vit-embed-dim", type=int, default=384, help="Dimensión de embedding del ViT")
    parser.add_argument("--vit-depth", type=int, default=12, help="Número de capas del ViT")
    parser.add_argument(
        "--vit-num-heads",
        type=int,
        default=6,
        help="Número de cabezales de atención del ViT",
    )
    parser.add_argument(
        "--vit-mlp-ratio",
        type=float,
        default=4.0,
        help="Factor de expansión MLP del ViT",
    )
    parser.add_argument("--num-local-crops", type=int, default=4, help="Número de crops locales")
    parser.add_argument("--local-crop-size", type=int, default=96, help="Tamaño de crops locales")
    parser.add_argument(
        "--global-crop-scale",
        type=float,
        nargs=2,
        default=(0.4, 1.0),
        help="Rango de escala para crops globales",
    )
    parser.add_argument(
        "--local-crop-scale",
        type=float,
        nargs=2,
        default=(0.05, 0.4),
        help="Rango de escala para crops locales",
    )
    parser.add_argument(
        "--stream",
        type=str,
        default=default_stream,
        help="Nombre del stream registrado en metadatos",
    )

    parser.set_defaults(dataset_shuffle=True)
    parser.add_argument(
        "--no-dataset-shuffle",
        action="store_false",
        dest="dataset_shuffle",
        help="Desactiva el barajado del DataLoader",
    )
    parser.add_argument(
        "--dataset-pin-memory",
        action="store_true",
        help="Activa pin_memory en el DataLoader",
    )
    parser.add_argument(
        "--dataset-persistent-workers",
        action="store_true",
        help="Activa persistent_workers cuando num_workers>0",
    )

    default_aug = AugmentationConfig()
    parser.add_argument(
        "--aug-brightness",
        type=float,
        default=default_aug.brightness,
        help="Rango de brillo para color jitter",
    )
    parser.add_argument(
        "--aug-contrast",
        type=float,
        default=default_aug.contrast,
        help="Rango de contraste para color jitter",
    )
    parser.add_argument(
        "--aug-saturation",
        type=float,
        default=default_aug.saturation,
        help="Rango de saturación para color jitter",
    )
    parser.add_argument(
        "--aug-hue",
        type=float,
        default=default_aug.hue,
        help="Rango de matiz para color jitter",
    )
    parser.add_argument(
        "--aug-color-jitter-prob",
        type=float,
        default=default_aug.color_jitter_prob,
        help="Probabilidad de aplicar color jitter",
    )
    parser.add_argument(
        "--aug-grayscale-prob",
        type=float,
        default=default_aug.grayscale_prob,
        help="Probabilidad de escala de grises",
    )
    parser.add_argument(
        "--aug-gaussian-blur-prob",
        type=float,
        default=default_aug.gaussian_blur_prob,
        help="Probabilidad de blur gaussiano",
    )
    parser.add_argument(
        "--aug-solarize-prob",
        type=float,
        default=default_aug.solarize_prob,
        help="Probabilidad de solarización",
    )
    parser.add_argument(
        "--aug-hflip-prob",
        type=float,
        default=default_aug.hflip_prob,
        help="Probabilidad de flip horizontal",
    )
    parser.add_argument(
        "--aug-mean",
        type=float,
        nargs=3,
        default=list(default_aug.mean),
        help="Media de normalización RGB",
    )
    parser.add_argument(
        "--aug-std",
        type=float,
        nargs=3,
        default=list(default_aug.std),
        help="Desvío estándar de normalización RGB",
    )

    parser.add_argument(
        "--checkpoint-best-name",
        type=str,
        default="checkpoint_best.pt",
        help="Nombre de archivo para el mejor checkpoint",
    )
    parser.add_argument(
        "--checkpoint-last-name",
        type=str,
        default="checkpoint_last.pt",
        help="Nombre de archivo para el último checkpoint",
    )
    parser.add_argument(
        "--checkpoint-history-file",
        type=str,
        default="metrics.jsonl",
        help="Archivo JSONL donde registrar métricas",
    )

    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Nombre amigable para el experimento",
    )
    parser.add_argument(
        "--experiment-notes",
        type=str,
        default=None,
        help="Notas libres asociadas al experimento",
    )
    parser.add_argument(
        "--experiment-tag",
        action="append",
        default=None,
        help="Etiqueta opcional (puede repetirse)",
    )

    args = parser.parse_args(remaining)

    if not args.train_dir:
        parser.error("Debe especificarse al menos un --train-dir o declararlo en la configuración")
    if not args.output_dir:
        parser.error("Debe indicar --output-dir o declararlo en la configuración")

    if args.experiment_tag is None:
        args.experiment_tag = []

    return args


def _build_models(
    args: argparse.Namespace, device: torch.device
) -> tuple[nn.Module, nn.Module, nn.Module, nn.Module, nn.Module | None, nn.Module | None]:
    backbone_cfg = BackboneConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.vit_embed_dim,
        depth=args.vit_depth,
        num_heads=args.vit_num_heads,
        mlp_ratio=args.vit_mlp_ratio,
    )
    student_backbone = build_vit_backbone(backbone_cfg).to(device)
    teacher_backbone = build_vit_backbone(backbone_cfg).to(device)
    teacher_backbone.load_state_dict(student_backbone.state_dict())
    for param in teacher_backbone.parameters():
        param.requires_grad_(False)

    student_head = ProjectionHead(args.vit_embed_dim, args.out_dim, args.head_hidden_dim).to(device)
    teacher_head = ProjectionHead(args.vit_embed_dim, args.out_dim, args.head_hidden_dim).to(device)
    teacher_head.load_state_dict(student_head.state_dict())
    for param in teacher_head.parameters():
        param.requires_grad_(False)

    student_patch_head: nn.Module | None = None
    teacher_patch_head: nn.Module | None = None
    if args.algorithm == "ibot":
        student_patch_head = nn.Linear(args.vit_embed_dim, args.patch_out_dim).to(device)
        teacher_patch_head = nn.Linear(args.vit_embed_dim, args.patch_out_dim).to(device)
        teacher_patch_head.load_state_dict(student_patch_head.state_dict())
        for param in teacher_patch_head.parameters():
            param.requires_grad_(False)

    return (
        student_backbone,
        teacher_backbone,
        student_head,
        teacher_head,
        student_patch_head,
        teacher_patch_head,
    )


def _resume_if_needed(
    args: argparse.Namespace,
    student_backbone: nn.Module,
    teacher_backbone: nn.Module,
    student_head: nn.Module,
    teacher_head: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None,
    student_patch_head: nn.Module | None,
    teacher_patch_head: nn.Module | None,
) -> tuple[int, float, int]:
    if not args.resume:
        return 1, float("inf"), 0
    state = load_checkpoint(args.resume)
    student_backbone.load_state_dict(state.student)
    teacher_backbone.load_state_dict(state.teacher)
    student_head.load_state_dict(state.head)
    if state.teacher_head:
        teacher_head.load_state_dict(state.teacher_head)
    if student_patch_head is not None and teacher_patch_head is not None and "student_patch" in state.metadata:
        student_patch_head.load_state_dict(state.metadata["student_patch"])  # type: ignore[arg-type]
        teacher_patch_head.load_state_dict(state.metadata["teacher_patch"])  # type: ignore[arg-type]
    optimizer.load_state_dict(state.optimizer)
    if scheduler and state.scheduler:
        scheduler.load_state_dict(state.scheduler)
    LOGGER.info("Reanudando entrenamiento desde la época %d", state.epoch)
    best_loss = float(state.metadata.get("best_loss", float("inf")))
    global_step = int(state.metadata.get("global_step", 0))
    return state.epoch + 1, best_loss, global_step


def run(argv: Iterable[str] | None = None, *, default_stream: str) -> None:
    args = _parse_args(argv, default_stream)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    if getattr(args, "config", None):
        args.config = args.config.expanduser().resolve()
        config_dir: Path | None = args.config.parent
    else:
        config_dir = None

    train_dirs = [_resolve_path(path, config_dir) for path in args.train_dir]
    output_dir = _resolve_path(args.output_dir, config_dir)
    resume_path = _resolve_path(args.resume, config_dir) if args.resume else None
    export_path = (
        _resolve_path(args.export_backbone, config_dir) if args.export_backbone else None
    )

    args.train_dir = train_dirs
    args.output_dir = output_dir
    args.resume = resume_path
    args.export_backbone = export_path

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info("Usando dispositivo %s", device)
    _seed_everything(args.seed)

    output_dir.mkdir(parents=True, exist_ok=True)

    dataloader = build_dataloader(
        train_dirs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=bool(args.dataset_shuffle),
        pin_memory=bool(args.dataset_pin_memory),
        persistent_workers=bool(args.dataset_persistent_workers),
    )
    augment_cfg = AugmentationConfig(
        image_size=args.image_size,
        global_crop_scale=tuple(float(x) for x in args.global_crop_scale),
        local_crop_size=args.local_crop_size,
        local_crop_scale=tuple(float(x) for x in args.local_crop_scale),
        num_local_crops=args.num_local_crops,
        brightness=args.aug_brightness,
        contrast=args.aug_contrast,
        saturation=args.aug_saturation,
        hue=args.aug_hue,
        color_jitter_prob=args.aug_color_jitter_prob,
        grayscale_prob=args.aug_grayscale_prob,
        gaussian_blur_prob=args.aug_gaussian_blur_prob,
        solarize_prob=args.aug_solarize_prob,
        hflip_prob=args.aug_hflip_prob,
        mean=tuple(float(x) for x in args.aug_mean),
        std=tuple(float(x) for x in args.aug_std),
    )
    augmenter = DINOMultiCropAugmentation(augment_cfg)

    backbone_meta = BackboneConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        embed_dim=args.vit_embed_dim,
        depth=args.vit_depth,
        num_heads=args.vit_num_heads,
        mlp_ratio=args.vit_mlp_ratio,
    )

    tracker = ExperimentTracker(
        output_dir,
        history_filename=args.checkpoint_history_file,
    )
    experiment_info = {
        "name": args.experiment_name,
        "notes": args.experiment_notes,
        "tags": list(args.experiment_tag or []),
    }
    experiment_info = {
        key: value
        for key, value in experiment_info.items()
        if value not in (None, "", [])
    }
    tracker_payload: dict[str, Any] = {
        "algorithm": args.algorithm,
        "stream": args.stream,
        "train_dirs": [str(path) for path in train_dirs],
        "output_dir": str(output_dir),
        "epochs": args.epochs,
        "seed": args.seed,
        "optimizer": {
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_epochs": args.warmup_epochs,
        },
        "teacher_momentum": {
            "base": args.teacher_momentum_base,
            "final": args.teacher_momentum_final,
        },
        "dataset": {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "shuffle": bool(args.dataset_shuffle),
            "pin_memory": bool(args.dataset_pin_memory),
            "persistent_workers": bool(args.dataset_persistent_workers),
        },
        "augmentation": asdict(augment_cfg),
        "backbone": asdict(backbone_meta),
        "checkpointing": {
            "best": args.checkpoint_best_name,
            "last": args.checkpoint_last_name,
            "history_file": args.checkpoint_history_file,
        },
    }
    if experiment_info:
        tracker_payload["experiment"] = experiment_info
    if args.config:
        tracker_payload["config_file"] = str(args.config)
    if resume_path:
        tracker_payload["resume"] = str(resume_path)
    if export_path:
        tracker_payload["export_backbone"] = str(export_path)
    tracker.log_params(tracker_payload)

    (
        student_backbone,
        teacher_backbone,
        student_head,
        teacher_head,
        student_patch_head,
        teacher_patch_head,
    ) = _build_models(args, device)

    parameters = list(student_backbone.parameters()) + list(student_head.parameters())
    if args.algorithm == "ibot" and student_patch_head is not None:
        parameters += list(student_patch_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)

    scheduler = None
    total_steps = len(dataloader) * args.epochs
    if total_steps == 0:
        raise RuntimeError("El dataset está vacío; no es posible entrenar")
    warmup_steps = int(len(dataloader) * args.warmup_epochs)
    lr_schedule = list(cosine_scheduler(args.learning_rate, warmup_steps, total_steps))
    momentum_schedule = _momentum_schedule(
        args.teacher_momentum_base,
        args.teacher_momentum_final,
        total_steps,
    )

    start_epoch, best_loss, global_step = _resume_if_needed(
        args,
        student_backbone,
        teacher_backbone,
        student_head,
        teacher_head,
        optimizer,
        scheduler,
        student_patch_head,
        teacher_patch_head,
    )

    dino_loss_fn = DINOLoss(
        args.out_dim,
        student_temp=args.student_temp,
        teacher_temp=args.teacher_temp,
        center_momentum=args.center_momentum,
    ).to(device)
    ibot_loss_fn = (
        IBOTLoss(
            args.patch_out_dim,
            student_temp=args.student_temp,
            teacher_temp=args.teacher_temp,
            center_momentum=args.center_momentum,
            global_weight=0.0,
            patch_weight=args.patch_loss_weight,
        ).to(device)
        if args.algorithm == "ibot"
        else None
    )

    best_checkpoint = output_dir / args.checkpoint_best_name
    last_checkpoint = output_dir / args.checkpoint_last_name

    for epoch in range(start_epoch, args.epochs + 1):
        student_backbone.train()
        student_head.train()
        if student_patch_head is not None:
            student_patch_head.train()
        epoch_loss = 0.0
        for step, images in enumerate(dataloader, start=1):
            optimizer.zero_grad(set_to_none=True)
            teacher_backbone.eval()
            teacher_head.eval()
            if teacher_patch_head is not None:
                teacher_patch_head.eval()

            global_views_per_image = []
            local_views_per_image = []
            for image in images:
                globals_, locals_ = augmenter(image)
                global_views_per_image.append(globals_)
                local_views_per_image.append(locals_)

            global_views = [torch.stack(crop).to(device) for crop in zip(*global_views_per_image)]
            local_views = []
            if local_views_per_image and local_views_per_image[0]:
                local_views = [torch.stack(crop).to(device) for crop in zip(*local_views_per_image)]

            with torch.no_grad():
                teacher_outputs = []
                teacher_patch_outputs = []
                for crop in global_views:
                    cls_tokens, patch_tokens = teacher_backbone.forward_with_patches(crop)
                    teacher_outputs.append(teacher_head(cls_tokens))
                    if teacher_patch_head is not None:
                        teacher_patch_outputs.append(teacher_patch_head(patch_tokens))

            student_outputs = []
            student_patch_outputs = []
            for idx, crop in enumerate(global_views + local_views):
                cls_tokens, patch_tokens = student_backbone.forward_with_patches(crop)
                student_outputs.append(student_head(cls_tokens))
                if student_patch_head is not None and idx < len(global_views):
                    student_patch_outputs.append(student_patch_head(patch_tokens))

            loss = _compute_dino_loss(teacher_outputs, student_outputs, dino_loss_fn)

            if (
                ibot_loss_fn is not None
                and student_patch_outputs
                and teacher_patch_outputs
            ):
                student_patches = torch.cat(student_patch_outputs, dim=0)
                teacher_patches = torch.cat(teacher_patch_outputs, dim=0)
                mask = mask_patches(student_patches.shape, args.patch_mask_ratio, device)
                loss = loss + ibot_loss_fn(
                    student_patches.mean(dim=1),
                    teacher_patches.mean(dim=1),
                    student_patches,
                    teacher_patches,
                    mask,
                )

            loss.backward()
            if args.clip_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(parameters, args.clip_grad_norm)

            schedule_idx = min(global_step, len(lr_schedule) - 1)
            lr = lr_schedule[schedule_idx]
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.step()

            momentum = momentum_schedule[schedule_idx]
            _update_teacher(
                student_backbone,
                teacher_backbone,
                student_head,
                teacher_head,
                momentum,
                student_patch_head=student_patch_head,
                teacher_patch_head=teacher_patch_head,
            )

            epoch_loss += loss.item()
            global_step += 1
            if step % args.log_interval == 0:
                LOGGER.info("Época %d - iter %d/%d - pérdida %.4f", epoch, step, len(dataloader), loss.item())
                tracker.log_metrics(
                    global_step,
                    {"loss": float(loss.item()), "epoch": epoch, "step": step},
                    context="train_step",
                )

        avg_loss = epoch_loss / max(len(dataloader), 1)
        LOGGER.info("Época %d completada - pérdida media %.4f", epoch, avg_loss)
        tracker.log_metrics(
            global_step,
            {"loss": float(avg_loss), "epoch": epoch},
            context="train_epoch",
        )

        new_best = avg_loss < best_loss
        best_loss = min(best_loss, avg_loss)
        metadata = {
            "epoch": epoch,
            "algorithm": args.algorithm,
            "stream": args.stream,
            "backbone": {
                "image_size": args.image_size,
                "patch_size": args.patch_size,
                "embed_dim": args.vit_embed_dim,
                "depth": args.vit_depth,
                "num_heads": args.vit_num_heads,
                "mlp_ratio": args.vit_mlp_ratio,
            },
            "best_loss": best_loss,
            "global_step": global_step,
        }
        if student_patch_head is not None and teacher_patch_head is not None:
            metadata["student_patch"] = student_patch_head.state_dict()
            metadata["teacher_patch"] = teacher_patch_head.state_dict()

        state = CheckpointState(
            epoch=epoch,
            student=student_backbone.state_dict(),
            teacher=teacher_backbone.state_dict(),
            head=student_head.state_dict(),
            teacher_head=teacher_head.state_dict(),
            optimizer=optimizer.state_dict(),
            scheduler=None,
            metadata=metadata,
        )
        save_checkpoint(last_checkpoint, state)
        tracker.register_artifact(
            last_checkpoint,
            name="checkpoint_last",
            metadata={"epoch": epoch, "loss": float(avg_loss)},
        )
        if new_best:
            LOGGER.info("Nueva mejor pérdida %.4f en época %d", avg_loss, epoch)
            save_checkpoint(best_checkpoint, state)
            tracker.register_artifact(
                best_checkpoint,
                name="checkpoint_best",
                metadata={"epoch": epoch, "loss": float(avg_loss)},
            )
            tracker.log_metrics(
                global_step,
                {"loss": float(avg_loss), "epoch": epoch},
                context="train_best",
            )

    if args.export_backbone:
        export_metadata = {
            "stream": args.stream,
            "algorithm": args.algorithm,
            "best_loss": best_loss,
            "epochs": args.epochs,
            "global_step": global_step,
            "backbone": {
                "image_size": args.image_size,
                "patch_size": args.patch_size,
                "embed_dim": args.vit_embed_dim,
                "depth": args.vit_depth,
                "num_heads": args.vit_num_heads,
                "mlp_ratio": args.vit_mlp_ratio,
            },
        }
        export_for_dinov2(args.export_backbone, backbone=teacher_backbone, metadata=export_metadata)
        LOGGER.info("Backbone exportado a %s", args.export_backbone)
        tracker.register_artifact(
            args.export_backbone,
            name="exported_backbone",
            metadata=export_metadata,
        )
