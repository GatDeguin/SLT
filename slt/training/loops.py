"""Training and evaluation loops for SLT models."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
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
MetricFn = Callable[[Any, Any], Union[float, torch.Tensor]]


@dataclass
class LoopResult:
    """Container with the aggregated loss and metrics for a loop."""

    loss: float
    metrics: Dict[str, float]


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
    return _execute_forward(model, inputs)


def _execute_forward(callable_obj: Callable[..., torch.Tensor], inputs: Inputs) -> torch.Tensor:
    if isinstance(inputs, MutableMapping):
        return callable_obj(**inputs)
    if isinstance(inputs, (list, tuple)):
        return callable_obj(*inputs)
    return callable_obj(inputs)


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


def _update_metric_sums(
    metric_sums: Dict[str, float],
    outputs: Any,
    targets: Any,
    metric_fns: Optional[Dict[str, MetricFn]],
    batch_items: int,
) -> None:
    if not metric_fns:
        return
    for name, fn in metric_fns.items():
        value = fn(outputs, targets)
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"Metric '{name}' must return a scalar value, got shape {tuple(value.shape)}."
                )
            value = float(value.item())
        metric_sums[name] = metric_sums.get(name, 0.0) + float(value) * batch_items


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
    grad_accum_steps: int = 1,
    metrics: Optional[Dict[str, MetricFn]] = None,
    forward_fn: Optional[Callable[..., torch.Tensor]] = None,
) -> LoopResult:
    """Run a single training epoch and return aggregated metrics."""

    model.train()
    total_loss = 0.0
    total_items = 0
    metric_sums: Dict[str, float] = {}

    if grad_accum_steps <= 0:
        raise ValueError("grad_accum_steps must be a positive integer")

    use_amp = scaler is not None and getattr(scaler, "is_enabled", lambda: False)()

    optimizer.zero_grad(set_to_none=True)
    step_index = 0

    for step_index, batch in enumerate(loader, start=1):
        inputs, targets = _split_batch(batch)
        inputs = _move_to_device(inputs, device)
        targets = _move_to_device(targets, device)

        batch_items = _count_items(targets)

        if use_amp and autocast_dtype is not None:
            with autocast(dtype=autocast_dtype):
                outputs = (
                    _call_model(model, inputs)
                    if forward_fn is None
                    else _execute_forward(forward_fn, inputs)
                )
                raw_loss = loss_fn(outputs, targets)
            loss = raw_loss / grad_accum_steps
            scaler.scale(loss).backward()  # type: ignore[arg-type]
            if step_index % grad_accum_steps == 0:
                clip_gradients(
                    optimizer,
                    grad_clip_norm,
                    scaler=scaler,
                    parameters=model.parameters(),
                    norm_type=grad_clip_norm_type,
                )
                scaler.step(optimizer)  # type: ignore[arg-type]
                scaler.update()  # type: ignore[arg-type]
                optimizer.zero_grad(set_to_none=True)
        else:
            outputs = (
                _call_model(model, inputs)
                if forward_fn is None
                else _execute_forward(forward_fn, inputs)
            )
            raw_loss = loss_fn(outputs, targets)
            loss = raw_loss / grad_accum_steps
            loss.backward()
            if step_index % grad_accum_steps == 0:
                clip_gradients(
                    optimizer,
                    grad_clip_norm,
                    scaler=None,
                    parameters=model.parameters(),
                    norm_type=grad_clip_norm_type,
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

        total_loss += raw_loss.detach().item() * batch_items
        total_items += batch_items
        _update_metric_sums(metric_sums, outputs, targets, metrics, batch_items)

    remainder = step_index % grad_accum_steps if step_index else 0
    if remainder != 0:
        if use_amp:
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
            clip_gradients(
                optimizer,
                grad_clip_norm,
                scaler=None,
                parameters=model.parameters(),
                norm_type=grad_clip_norm_type,
            )
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    if total_items == 0:
        raise RuntimeError("Empty training loader provided.")

    averaged_metrics = {name: value / total_items for name, value in metric_sums.items()}

    return LoopResult(total_loss / total_items, averaged_metrics)


def eval_epoch(
    model: nn.Module,
    loader: Iterable[Batch],
    loss_fn: LossFn,
    *,
    device: Union[str, torch.device] = "cuda",
    metrics: Optional[Dict[str, MetricFn]] = None,
    forward_fn: Optional[Callable[..., torch.Tensor]] = None,
) -> LoopResult:
    """Evaluate the model for a single epoch and return aggregated metrics."""

    model.eval()
    total_loss = 0.0
    total_items = 0
    metric_sums: Dict[str, float] = {}

    with torch.no_grad():
        for batch in loader:
            inputs, targets = _split_batch(batch)
            inputs = _move_to_device(inputs, device)
            targets = _move_to_device(targets, device)

            outputs = (
                _call_model(model, inputs)
                if forward_fn is None
                else _execute_forward(forward_fn, inputs)
            )
            loss = loss_fn(outputs, targets)

            batch_items = _count_items(targets)
            total_loss += loss.detach().item() * batch_items
            total_items += batch_items
            _update_metric_sums(metric_sums, outputs, targets, metrics, batch_items)

    if total_items == 0:
        raise RuntimeError("Empty evaluation loader provided.")

    averaged_metrics = {name: value / total_items for name, value in metric_sums.items()}

    return LoopResult(total_loss / total_items, averaged_metrics)
