"""Training and evaluation loops for SLT models."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, MutableMapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_

try:  # pragma: no cover - CUDA may be unavailable in CI
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
except Exception:  # pragma: no cover - GradScaler/autocast optional
    GradScaler = None  # type: ignore

    @contextmanager
    def autocast(*args, **kwargs):  # type: ignore
        yield

Batch = Union[Dict[str, Any], Sequence[Any]]
Inputs = Union[torch.Tensor, Sequence[Any], MutableMapping[str, Any]]
LossFn = Callable[[torch.Tensor, Any], torch.Tensor]


def _move_to_device(data: Any, device: Union[str, torch.device]) -> Any:
    """Recursively move tensors in *data* to *device*."""
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, MutableMapping):
        return type(data)({k: _move_to_device(v, device) for k, v in data.items()})
    if isinstance(data, (list, tuple)):
        return type(data)(_move_to_device(v, device) for v in data)
    return data


def _split_batch(batch: Batch) -> Tuple[Inputs, Any]:
    """Split a batch into inputs and targets."""
    if isinstance(batch, MutableMapping):
        if "targets" in batch:
            inputs = {k: v for k, v in batch.items() if k != "targets"}
            if "inputs" in inputs and len(inputs) == 1:
                inputs = inputs["inputs"]
            return inputs, batch["targets"]
        if "labels" in batch:
            inputs = {k: v for k, v in batch.items() if k not in {"labels", "targets"}}
            if "inputs" in inputs and len(inputs) == 1:
                inputs = inputs["inputs"]
            return inputs, batch["labels"]
        if "inputs" in batch and "y" in batch:
            return batch["inputs"], batch["y"]
        raise KeyError("Dictionary batch must contain 'targets' or 'labels'.")
    if isinstance(batch, (list, tuple)):
        if len(batch) < 2:
            raise ValueError("Iterable batch must contain at least inputs and targets.")
        inputs, targets = batch[0], batch[1]
        return inputs, targets
    raise TypeError(f"Unsupported batch type: {type(batch)!r}")


def _call_model(model: nn.Module, inputs: Inputs) -> torch.Tensor:
    if isinstance(inputs, MutableMapping):
        return model(**inputs)
    if isinstance(inputs, (list, tuple)):
        return model(*inputs)
    return model(inputs)


def _count_items(targets: Any) -> int:
    if isinstance(targets, torch.Tensor):
        if targets.dim() == 0:
            return 1
        if targets.dtype in {torch.int64, torch.int32, torch.long} and targets.dim() > 1:
            valid = targets.ne(-100)
            count = int(valid.sum().item())
            if count > 0:
                return count
        return targets.shape[0]
    if isinstance(targets, (list, tuple)):
        return len(targets)
    if hasattr(targets, "__len__"):
        return len(targets)  # type: ignore[arg-type]
    return 1


def clip_gradients(
    optimizer: torch.optim.Optimizer,
    max_norm: Optional[float],
    *,
    scaler: Optional["GradScaler"] = None,
    parameters: Optional[Iterable[torch.Tensor]] = None,
    norm_type: Union[float, int] = 2.0,
) -> Optional[torch.Tensor]:
    """Clip gradients to the provided ``max_norm`` if specified.

    When ``scaler`` is provided and enabled, the gradients are first unscaled to
    ensure clipping operates on the true values. The function returns the total
    norm as reported by :func:`torch.nn.utils.clip_grad_norm_` when clipping is
    performed, otherwise ``None``.
    """

    if max_norm is None or max_norm <= 0:
        return None

    if parameters is None:
        parameters = (p for group in optimizer.param_groups for p in group["params"])

    grads = [p for p in parameters if p is not None and p.requires_grad]
    if not grads:
        return None

    if scaler is not None and getattr(scaler, "is_enabled", lambda: False)():
        scaler.unscale_(optimizer)

    return clip_grad_norm_(grads, max_norm, norm_type=norm_type)


def train_epoch(
    model: nn.Module,
    loader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    *,
    device: Union[str, torch.device] = "cuda",
    scaler: Optional["GradScaler"] = None,
    autocast_dtype: Optional[torch.dtype] = torch.float16,
    grad_clip_norm: Optional[float] = None,
    grad_clip_norm_type: Union[float, int] = 2.0,
) -> float:
    """Run a single training epoch and return the average loss."""

    model.train()
    total_loss = 0.0
    total_items = 0

    use_amp = scaler is not None and getattr(scaler, "is_enabled", lambda: False)()

    for batch in loader:
        inputs, targets = _split_batch(batch)
        inputs = _move_to_device(inputs, device)
        targets = _move_to_device(targets, device)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and autocast_dtype is not None:
            with autocast(dtype=autocast_dtype):
                outputs = _call_model(model, inputs)
                loss = loss_fn(outputs, targets)
            scaler.scale(loss).backward()  # type: ignore[arg-type]
            clip_gradients(
                optimizer,
                grad_clip_norm,
                scaler=scaler,
                parameters=model.parameters(),
                norm_type=grad_clip_norm_type,
            )
            scaler.step(optimizer)  # type: ignore[arg-type]
            scaler.update()  # type: ignore[arg-type]
        else:
            outputs = _call_model(model, inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
            clip_gradients(
                optimizer,
                grad_clip_norm,
                scaler=None,
                parameters=model.parameters(),
                norm_type=grad_clip_norm_type,
            )
            optimizer.step()

        batch_items = _count_items(targets)
        total_loss += loss.detach().item() * batch_items
        total_items += batch_items

    if total_items == 0:
        raise RuntimeError("Empty training loader provided.")

    return total_loss / total_items


def eval_epoch(
    model: nn.Module,
    loader: Iterable[Batch],
    loss_fn: LossFn,
    *,
    device: Union[str, torch.device] = "cuda",
) -> float:
    """Evaluate the model for a single epoch and return the average loss."""

    model.eval()
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for batch in loader:
            inputs, targets = _split_batch(batch)
            inputs = _move_to_device(inputs, device)
            targets = _move_to_device(targets, device)

            outputs = _call_model(model, inputs)
            loss = loss_fn(outputs, targets)

            batch_items = _count_items(targets)
            total_loss += loss.detach().item() * batch_items
            total_items += batch_items

    if total_items == 0:
        raise RuntimeError("Empty evaluation loader provided.")

    return total_loss / total_items
