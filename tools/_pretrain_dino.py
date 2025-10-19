"""Shared entry point for DINO/iBOT pre-training scripts."""

from __future__ import annotations

import argparse
import logging
import math
import random
from pathlib import Path
from typing import Iterable, Sequence

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
    load_checkpoint,
    mask_patches,
    save_checkpoint,
)

LOGGER = logging.getLogger(__name__)


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
    parser = argparse.ArgumentParser(description="Self-supervised DINO/iBOT pre-training")
    parser.add_argument("--train-dir", nargs="+", type=Path, required=True, help="Carpetas con recortes del stream")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directorio donde escribir checkpoints y logs")
    parser.add_argument("--algorithm", choices=["dino", "ibot"], default="dino", help="Algoritmo de entrenamiento a utilizar")
    parser.add_argument("--epochs", type=int, default=10, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=32, help="Tamaño del batch")
    parser.add_argument("--num-workers", type=int, default=0, help="Workers del DataLoader")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Tasa de aprendizaje inicial")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Decaimiento L2")
    parser.add_argument("--warmup-epochs", type=float, default=1.0, help="Épocas de warmup lineal")
    parser.add_argument("--teacher-momentum-base", type=float, default=0.996, help="Momentum inicial del maestro")
    parser.add_argument("--teacher-momentum-final", type=float, default=1.0, help="Momentum final del maestro")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0, help="Norma máxima para el gradiente (0 desactiva)")
    parser.add_argument("--out-dim", type=int, default=1024, help="Dimensión del proyector DINO")
    parser.add_argument("--head-hidden-dim", type=int, default=2048, help="Dimensión oculta del proyector")
    parser.add_argument("--patch-out-dim", type=int, default=256, help="Dimensión de logits de parches para iBOT")
    parser.add_argument("--patch-mask-ratio", type=float, default=0.3, help="Proporción de parches enmascarados para iBOT")
    parser.add_argument("--patch-loss-weight", type=float, default=1.0, help="Peso relativo de la pérdida de parches en iBOT")
    parser.add_argument("--student-temp", type=float, default=0.1, help="Temperatura del estudiante")
    parser.add_argument("--teacher-temp", type=float, default=0.04, help="Temperatura del maestro")
    parser.add_argument("--center-momentum", type=float, default=0.9, help="Momentum del centro para DINO")
    parser.add_argument("--seed", type=int, default=42, help="Semilla PRNG")
    parser.add_argument("--device", type=str, default=None, help="Dispositivo a utilizar")
    parser.add_argument("--resume", type=Path, default=None, help="Checkpoint para reanudar entrenamiento")
    parser.add_argument("--export-backbone", type=Path, default=None, help="Ruta donde exportar el backbone compatible con load_dinov2_backbone")
    parser.add_argument("--log-interval", type=int, default=10, help="Intervalo en iteraciones para registro de pérdidas")
    parser.add_argument("--image-size", type=int, default=224, help="Resolución de entrada del ViT")
    parser.add_argument("--patch-size", type=int, default=16, help="Tamaño de parche del ViT")
    parser.add_argument("--vit-embed-dim", type=int, default=384, help="Dimensión de embedding del ViT")
    parser.add_argument("--vit-depth", type=int, default=12, help="Número de capas del ViT")
    parser.add_argument("--vit-num-heads", type=int, default=6, help="Número de cabezales de atención del ViT")
    parser.add_argument("--vit-mlp-ratio", type=float, default=4.0, help="Factor de expansión MLP del ViT")
    parser.add_argument("--num-local-crops", type=int, default=4, help="Número de crops locales")
    parser.add_argument("--local-crop-size", type=int, default=96, help="Tamaño de crops locales")
    parser.add_argument("--global-crop-scale", type=float, nargs=2, default=(0.4, 1.0), help="Rango de escala para crops globales")
    parser.add_argument("--local-crop-scale", type=float, nargs=2, default=(0.05, 0.4), help="Rango de escala para crops locales")
    parser.add_argument("--stream", type=str, default=default_stream, help="Nombre del stream registrado en metadatos")
    return parser.parse_args(argv)


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
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    LOGGER.info("Usando dispositivo %s", device)
    _seed_everything(args.seed)

    dataloader = build_dataloader(
        args.train_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
    )
    augment_cfg = AugmentationConfig(
        image_size=args.image_size,
        global_crop_scale=tuple(args.global_crop_scale),
        local_crop_size=args.local_crop_size,
        local_crop_scale=tuple(args.local_crop_scale),
        num_local_crops=args.num_local_crops,
    )
    augmenter = DINOMultiCropAugmentation(augment_cfg)

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

    best_checkpoint = args.output_dir / "checkpoint_best.pt"
    last_checkpoint = args.output_dir / "checkpoint_last.pt"

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

        avg_loss = epoch_loss / max(len(dataloader), 1)
        LOGGER.info("Época %d completada - pérdida media %.4f", epoch, avg_loss)

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
        if new_best:
            LOGGER.info("Nueva mejor pérdida %.4f en época %d", avg_loss, epoch)
            save_checkpoint(best_checkpoint, state)

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
