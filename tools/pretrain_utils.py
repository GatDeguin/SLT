"""Shared utilities for self-supervised DINO/iBOT pre-training scripts.

The helpers in this module provide a light-weight alternative to the
``torchvision`` training recipes used by the original DINO/iBOT projects.  They
avoid optional third-party dependencies while still exposing the building
blocks required by the command line tools that live in :mod:`tools`.

The implementation intentionally focuses on approachability: datasets operate on
simple image folders, augmentations are implemented with the Python Imaging
Library and :mod:`numpy`, and the losses follow the formulations published in
`Emerging Properties in Self-Supervised Vision Transformers`_ and
`Masked Siamese Networks for Label-Efficient Learning`_.

The resulting utilities make it possible to quickly spin self-supervised
experiments on CPU-only environments which is essential for the smoke tests
included in this repository.

.. _Emerging Properties in Self-Supervised Vision Transformers: https://arxiv.org/abs/2104.14294
.. _Masked Siamese Networks for Label-Efficient Learning: https://arxiv.org/abs/2204.07141
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset

from slt.models.backbones import ViTConfig, ViTSmallPatch16

SUPPORTED_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def _serialize_json(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    if isinstance(value, datetime):
        return value.isoformat()
    raise TypeError(f"Unsupported value for JSON serialization: {value!r}")


# ---------------------------------------------------------------------------
# Dataset helpers
# ---------------------------------------------------------------------------


@dataclass
class DatasetConfig:
    roots: Sequence[Path | str]
    batch_size: int = 32
    num_workers: int = 0
    shuffle: bool = True
    pin_memory: bool = False
    persistent_workers: bool = False


class ImageFolderDataset(Dataset[Image.Image]):
    """Iterate over images stored in one or multiple directory trees."""

    def __init__(self, roots: Sequence[Path | str]) -> None:
        self.roots = [Path(root).expanduser().resolve() for root in roots]
        self._files = self._collect_files()
        if not self._files:
            formatted = ", ".join(str(root) for root in self.roots)
            raise FileNotFoundError(
                f"No supported image files were found under: {formatted}"
            )

    def _collect_files(self) -> list[Path]:
        files: list[Path] = []
        for root in self.roots:
            if root.is_file() and root.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                files.append(root)
                continue
            if not root.exists():
                raise FileNotFoundError(f"Dataset path does not exist: {root}")
            for path in root.rglob("*"):
                if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS:
                    files.append(path)
        files.sort()
        return files

    def __len__(self) -> int:  # pragma: no cover - trivial
        return len(self._files)

    def __getitem__(self, index: int) -> Image.Image:
        path = self._files[index]
        with Image.open(path) as image:
            return image.convert("RGB")


# ---------------------------------------------------------------------------
# DataLoader helpers
# ---------------------------------------------------------------------------


def pil_collate(batch: Sequence[Image.Image]) -> list[Image.Image]:
    """Default collation function that keeps PIL images as Python objects."""

    return list(batch)


# ---------------------------------------------------------------------------
# Augmentation utilities
# ---------------------------------------------------------------------------


def _pil_to_tensor(image: Image.Image) -> Tensor:
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1)
    return tensor


def _normalize(tensor: Tensor, mean: Sequence[float], std: Sequence[float]) -> Tensor:
    mean_t = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    std_t = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor - mean_t) / std_t


def _random_resized_crop(
    image: Image.Image,
    size: int,
    *,
    scale: Tuple[float, float],
    ratio: Tuple[float, float] = (3.0 / 4.0, 4.0 / 3.0),
) -> Image.Image:
    width, height = image.size
    area = width * height
    log_ratio = (math.log(ratio[0]), math.log(ratio[1]))
    for _ in range(10):
        target_area = random.uniform(scale[0], scale[1]) * area
        aspect = math.exp(random.uniform(*log_ratio))
        crop_w = int(round(math.sqrt(target_area * aspect)))
        crop_h = int(round(math.sqrt(target_area / aspect)))
        if crop_w <= width and crop_h <= height:
            x = random.randint(0, width - crop_w)
            y = random.randint(0, height - crop_h)
            cropped = image.crop((x, y, x + crop_w, y + crop_h))
            return cropped.resize((size, size), Image.BICUBIC)
    return ImageOps.fit(image, (size, size), method=Image.BICUBIC)


def _apply_color_jitter(
    image: Image.Image,
    *,
    brightness: float,
    contrast: float,
    saturation: float,
    hue: float,
) -> Image.Image:
    if brightness > 0:
        factor = random.uniform(max(0, 1 - brightness), 1 + brightness)
        image = ImageEnhance.Brightness(image).enhance(factor)
    if contrast > 0:
        factor = random.uniform(max(0, 1 - contrast), 1 + contrast)
        image = ImageEnhance.Contrast(image).enhance(factor)
    if saturation > 0:
        factor = random.uniform(max(0, 1 - saturation), 1 + saturation)
        image = ImageEnhance.Color(image).enhance(factor)
    if hue > 0:
        hsv = np.array(image.convert("HSV"), dtype=np.uint8)
        shift = int((random.uniform(-hue, hue) * 255) % 255)
        hsv[..., 0] = (hsv[..., 0].astype(int) + shift) % 255
        image = Image.fromarray(hsv, mode="HSV").convert("RGB")
    return image


def _random_grayscale(image: Image.Image, p: float) -> Image.Image:
    if random.random() < p:
        return ImageOps.grayscale(image).convert("RGB")
    return image


def _random_gaussian_blur(image: Image.Image, p: float, radius: Tuple[float, float]) -> Image.Image:
    if random.random() < p:
        rad = random.uniform(*radius)
        return image.filter(ImageFilter.GaussianBlur(radius=rad))
    return image


def _random_solarize(image: Image.Image, p: float, threshold: int = 128) -> Image.Image:
    if random.random() < p:
        lut = [i if i < threshold else 255 - i for i in range(256)]
        return image.point(lut * 3)
    return image


@dataclass
class AugmentationConfig:
    image_size: int = 224
    global_crop_scale: Tuple[float, float] = (0.4, 1.0)
    local_crop_size: int = 96
    local_crop_scale: Tuple[float, float] = (0.05, 0.4)
    num_local_crops: int = 4
    brightness: float = 0.4
    contrast: float = 0.4
    saturation: float = 0.2
    hue: float = 0.1
    color_jitter_prob: float = 0.8
    grayscale_prob: float = 0.2
    gaussian_blur_prob: float = 0.5
    solarize_prob: float = 0.1
    hflip_prob: float = 0.5
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406)
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)


class DINOMultiCropAugmentation:
    """Generate multiple crops for DINO style pre-training."""

    def __init__(self, config: AugmentationConfig) -> None:
        self.config = config

    def _base_transform(self, image: Image.Image, size: int, scale: Tuple[float, float]) -> Tensor:
        image = _random_resized_crop(image, size, scale=scale)
        if random.random() < self.config.hflip_prob:
            image = ImageOps.mirror(image)
        if random.random() < self.config.color_jitter_prob:
            image = _apply_color_jitter(
                image,
                brightness=self.config.brightness,
                contrast=self.config.contrast,
                saturation=self.config.saturation,
                hue=self.config.hue,
            )
        image = _random_grayscale(image, self.config.grayscale_prob)
        image = _random_gaussian_blur(image, self.config.gaussian_blur_prob, (0.1, 2.0))
        image = _random_solarize(image, self.config.solarize_prob)
        tensor = _pil_to_tensor(image)
        tensor = _normalize(tensor, self.config.mean, self.config.std)
        return tensor

    def __call__(self, image: Image.Image) -> tuple[list[Tensor], list[Tensor]]:
        globals: list[Tensor] = []
        locals_: list[Tensor] = []
        for _ in range(2):
            globals.append(
                self._base_transform(
                    image,
                    self.config.image_size,
                    self.config.global_crop_scale,
                )
            )
        for _ in range(self.config.num_local_crops):
            locals_.append(
                self._base_transform(
                    image,
                    self.config.local_crop_size,
                    self.config.local_crop_scale,
                )
            )
        return globals, locals_


# ---------------------------------------------------------------------------
# Losses and model heads
# ---------------------------------------------------------------------------


class ProjectionHead(nn.Module):
    """Simple MLP projection head used by the student and teacher."""

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.mlp(x)


class DINOLoss(nn.Module):
    """Cross-entropy loss between student and teacher distributions."""

    def __init__(
        self,
        out_dim: int,
        *,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
    ) -> None:
        super().__init__()
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(out_dim), persistent=True)

    def forward(self, student: Tensor, teacher: Tensor) -> Tensor:
        teacher = teacher.detach()
        student_logprob = torch.nn.functional.log_softmax(student / self.student_temp, dim=-1)
        teacher_prob = torch.nn.functional.softmax(
            (teacher - self.center) / self.teacher_temp, dim=-1
        )
        loss = -(teacher_prob * student_logprob).sum(dim=-1).mean()
        with torch.no_grad():
            batch_center = teacher.mean(dim=0)
            self.center.mul_(self.center_momentum).add_(batch_center * (1 - self.center_momentum))
        return loss


class IBOTLoss(nn.Module):
    """Combined global and patch level loss for iBOT style training."""

    def __init__(
        self,
        out_dim: int,
        *,
        student_temp: float = 0.1,
        teacher_temp: float = 0.04,
        center_momentum: float = 0.9,
        global_weight: float = 1.0,
        patch_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.global_loss = DINOLoss(
            out_dim,
            student_temp=student_temp,
            teacher_temp=teacher_temp,
            center_momentum=center_momentum,
        )
        self.global_weight = global_weight
        self.patch_weight = patch_weight

    def forward(
        self,
        student_global: Tensor,
        teacher_global: Tensor,
        student_patch_logits: Tensor,
        teacher_patch_logits: Tensor,
        mask: Tensor,
    ) -> Tensor:
        global_loss = (
            self.global_loss(student_global, teacher_global)
            if self.global_weight
            else torch.tensor(0.0, device=student_global.device)
        )
        masked_student = student_patch_logits[mask]
        masked_teacher = teacher_patch_logits[mask].detach()
        if masked_student.numel() == 0:
            patch_loss = torch.tensor(0.0, device=student_global.device)
        else:
            patch_loss = torch.nn.functional.mse_loss(masked_student, masked_teacher)
        return self.global_weight * global_loss + self.patch_weight * patch_loss


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


@dataclass
class BackboneConfig:
    image_size: int = 224
    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 12
    num_heads: int = 6
    mlp_ratio: float = 4.0

    def to_vit_config(self) -> ViTConfig:
        return ViTConfig(
            image_size=self.image_size,
            patch_size=self.patch_size,
            embed_dim=self.embed_dim,
            depth=self.depth,
            num_heads=self.num_heads,
            mlp_ratio=self.mlp_ratio,
        )


@dataclass
class CheckpointState:
    epoch: int
    student: dict[str, Tensor]
    teacher: dict[str, Tensor]
    head: dict[str, Tensor]
    teacher_head: Optional[dict[str, Tensor]]
    optimizer: dict[str, Tensor]
    scheduler: Optional[dict[str, Tensor]]
    metadata: dict[str, object]


def build_vit_backbone(config: BackboneConfig) -> ViTSmallPatch16:
    return ViTSmallPatch16(config.to_vit_config())


def cosine_scheduler(base_lr: float, warmup_steps: int, total_steps: int) -> Iterator[float]:
    for step in range(total_steps):
        if warmup_steps and step < warmup_steps:
            yield base_lr * (step + 1) / warmup_steps
        else:
            progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            yield base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def save_checkpoint(path: Path, state: CheckpointState) -> None:
    payload = {
        "epoch": state.epoch,
        "student": state.student,
        "teacher": state.teacher,
        "head": state.head,
        "teacher_head": state.teacher_head,
        "optimizer": state.optimizer,
        "scheduler": state.scheduler,
        "metadata": state.metadata,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> CheckpointState:
    checkpoint = torch.load(path, map_location=map_location)
    return CheckpointState(
        epoch=int(checkpoint.get("epoch", 0)),
        student=dict(checkpoint.get("student", {})),
        teacher=dict(checkpoint.get("teacher", {})),
        head=dict(checkpoint.get("head", {})),
        teacher_head=dict(checkpoint.get("teacher_head", {})) or None,
        optimizer=dict(checkpoint.get("optimizer", {})),
        scheduler=dict(checkpoint.get("scheduler")) if checkpoint.get("scheduler") else None,
        metadata=dict(checkpoint.get("metadata", {})),
    )


def export_for_dinov2(
    path: Path,
    *,
    backbone: nn.Module,
    metadata: Optional[dict[str, object]] = None,
) -> None:
    """Persist a backbone state dict compatible with :func:`load_dinov2_backbone`."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": backbone.state_dict(),
        "metadata": metadata or {},
    }
    with path.open("wb") as handle:
        torch.save(payload, handle)
    if metadata:
        with path.with_suffix(".json").open("w", encoding="utf8") as handle:
            json.dump(metadata, handle, indent=2)


def build_dataloader(
    roots: Sequence[Path | str],
    *,
    batch_size: int,
    num_workers: int,
    shuffle: bool = True,
    pin_memory: bool = False,
    persistent_workers: bool = False,
    collate_fn: Optional[callable] = None,
) -> DataLoader:
    dataset = ImageFolderDataset(roots)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn or pil_collate,
        pin_memory=pin_memory,
        persistent_workers=bool(persistent_workers and num_workers > 0),
    )


def apply_ema(student: nn.Module, teacher: nn.Module, momentum: float) -> None:
    with torch.no_grad():
        for student_param, teacher_param in zip(student.parameters(), teacher.parameters()):
            teacher_param.data.mul_(momentum).add_(student_param.data * (1.0 - momentum))


def mask_patches(shape: Tuple[int, int, int], mask_ratio: float, device: torch.device) -> Tensor:
    batch, num_patches, _ = shape
    mask = torch.zeros((batch, num_patches), dtype=torch.bool, device=device)
    num_mask = max(1, int(num_patches * mask_ratio))
    for idx in range(batch):
        perm = torch.randperm(num_patches, device=device)
        mask[idx, perm[:num_mask]] = True
    return mask


# ---------------------------------------------------------------------------
# Experiment tracking helpers
# ---------------------------------------------------------------------------


class ExperimentTracker:
    """Minimal experiment tracking helper used by the CLI wrappers."""

    def __init__(
        self,
        root: Path,
        *,
        history_filename: str = "metrics.jsonl",
        params_filename: str = "params.json",
        artifacts_filename: str = "artifacts.json",
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.history_path = self.root / history_filename
        self.params_path = self.root / params_filename
        self.artifacts_path = self.root / artifacts_filename

    def log_params(self, params: Mapping[str, object]) -> None:
        payload: dict[str, object] = {}
        if self.params_path.exists():
            with self.params_path.open("r", encoding="utf8") as handle:
                try:
                    payload = json.load(handle)
                except json.JSONDecodeError:
                    payload = {}
        payload.update(params)
        payload.setdefault("timestamp", datetime.now(timezone.utc).isoformat())
        with self.params_path.open("w", encoding="utf8") as handle:
            json.dump(payload, handle, indent=2, default=_serialize_json)

    def log_metrics(self, step: int, metrics: Mapping[str, object], *, context: str | None = None) -> None:
        record: dict[str, object] = {
            "step": step,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if context:
            record["context"] = context
        record.update(metrics)
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        with self.history_path.open("a", encoding="utf8") as handle:
            handle.write(json.dumps(record, default=_serialize_json) + "\n")

    def register_artifact(
        self,
        path: Path,
        *,
        name: str | None = None,
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        artifact_entry: dict[str, object] = {
            "name": name or Path(path).name,
            "path": str(Path(path)),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if metadata:
            artifact_entry["metadata"] = dict(metadata)

        entries: list[dict[str, object]] = []
        if self.artifacts_path.exists():
            with self.artifacts_path.open("r", encoding="utf8") as handle:
                try:
                    existing = json.load(handle)
                except json.JSONDecodeError:
                    existing = []
            if isinstance(existing, list):
                entries = list(existing)

        replaced = False
        for idx, current in enumerate(entries):
            if current.get("name") == artifact_entry["name"]:
                entries[idx] = artifact_entry
                replaced = True
                break
        if not replaced:
            entries.append(artifact_entry)

        with self.artifacts_path.open("w", encoding="utf8") as handle:
            json.dump(entries, handle, indent=2, default=_serialize_json)
