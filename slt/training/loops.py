"""Training and evaluation loops for SLT models."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterable, MutableMapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn

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
        return targets.shape[0] if targets.dim() > 0 else 1
    if isinstance(targets, (list, tuple)):
        return len(targets)
    if hasattr(targets, "__len__"):
        return len(targets)  # type: ignore[arg-type]
    return 1


def train_epoch(
    model: nn.Module,
    loader: Iterable[Batch],
    optimizer: torch.optim.Optimizer,
    loss_fn: LossFn,
    *,
    device: Union[str, torch.device] = "cuda",
    scaler: Optional["GradScaler"] = None,
    autocast_dtype: Optional[torch.dtype] = torch.float16,
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
            scaler.step(optimizer)  # type: ignore[arg-type]
            scaler.update()  # type: ignore[arg-type]
        else:
            outputs = _call_model(model, inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()
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
