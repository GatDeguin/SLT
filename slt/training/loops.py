"""Training and evaluation loops for SLT models."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, MutableMapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer

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
GradClipValue = Optional[Union[float, Callable[[int, torch.Tensor, Optimizer], Optional[float]]]]
GradAccumulation = Union[int, Callable[[int, torch.Tensor, Batch, int], bool]]
AmpFailureHandler = Callable[[RuntimeError, int], bool]


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
        metadata_keys = {"video_ids"}
        if "inputs" in batch:
            if "targets" in batch:
                return batch["inputs"], batch["targets"]
            if "labels" in batch:
                return batch["inputs"], batch["labels"]
            if "y" in batch:
                return batch["inputs"], batch["y"]
        if "targets" in batch:
            blocked = {"targets"} | metadata_keys
            inputs = {k: v for k, v in batch.items() if k not in blocked}
            return inputs, batch["targets"]
        if "labels" in batch:
            blocked = {"labels", "targets"} | metadata_keys
            inputs = {k: v for k, v in batch.items() if k not in blocked}
            return inputs, batch["labels"]
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
    grad_clip_norm: GradClipValue = None,
    grad_clip_norm_type: Union[float, int] = 2.0,
    grad_accum_steps: GradAccumulation = 1,
    metrics: Optional[Dict[str, MetricFn]] = None,
    forward_fn: Optional[Callable[..., torch.Tensor]] = None,
    amp_failure_handler: Optional[AmpFailureHandler] = None,
) -> LoopResult:
    """Run a single training epoch and return aggregated metrics."""

    model.train()
    total_loss = 0.0
    total_items = 0
    metric_sums: Dict[str, float] = {}

    if isinstance(grad_accum_steps, int):
        if grad_accum_steps <= 0:
            raise ValueError("grad_accum_steps must be a positive integer")
        current_accum_target = grad_accum_steps
    elif callable(grad_accum_steps):
        current_accum_target = 1
    else:
        raise TypeError("grad_accum_steps must be an int or a callable strategy")

    use_amp = scaler is not None and getattr(scaler, "is_enabled", lambda: False)()

    optimizer.zero_grad(set_to_none=True)
    step_index = 0
    pending_backward = 0
    last_raw_loss: Optional[torch.Tensor] = None

    def _resolve_clip_value(step: int, loss_value: torch.Tensor) -> Optional[float]:
        value: Optional[float]
        if callable(grad_clip_norm):
            value_candidate = grad_clip_norm(step, loss_value, optimizer)
            if value_candidate is None:
                return None
            value = float(value_candidate)
        else:
            value = grad_clip_norm
        if value is None:
            return None
        if value <= 0:
            return None
        return value

    def _parse_accumulation_directive(
        step: int,
        loss_value: torch.Tensor,
        batch_obj: Batch,
        pending_after: int,
    ) -> Tuple[bool, int]:
        nonlocal current_accum_target
        if isinstance(grad_accum_steps, int):
            return False, current_accum_target

        directive = grad_accum_steps(step, loss_value, batch_obj, pending_after)
        should_step = False
        if isinstance(directive, tuple):
            if len(directive) != 2:
                raise ValueError(
                    "grad_accum_steps callable must return (should_step, accumulation_steps)"
                )
            should_step = bool(directive[0])
            if directive[1] is not None:
                current_accum_target = int(directive[1])
        elif isinstance(directive, bool):
            should_step = directive
        elif isinstance(directive, int):
            current_accum_target = directive
        else:
            raise TypeError(
                "grad_accum_steps callable must return bool, int or (bool, int)"
            )

        if current_accum_target <= 0:
            raise ValueError("Accumulation steps provided by strategy must be positive")

        return should_step, current_accum_target

    def _handle_amp_failure(error: RuntimeError) -> bool:
        handled = True
        if amp_failure_handler is not None:
            handled = bool(amp_failure_handler(error, step_index))
        return handled

    def _is_amp_overflow(error: RuntimeError) -> bool:
        message = str(error).lower()
        return "inf" in message or "nan" in message

    def _perform_step(loss_value: torch.Tensor) -> None:
        clip_value = _resolve_clip_value(step_index, loss_value)
        if use_amp:
            clip_gradients(
                optimizer,
                clip_value,
                scaler=scaler,
                parameters=model.parameters(),
                norm_type=grad_clip_norm_type,
            )
            try:
                scaler.step(optimizer)  # type: ignore[arg-type]
            except RuntimeError as error:
                if _is_amp_overflow(error) and _handle_amp_failure(error):
                    scaler.update()  # type: ignore[arg-type]
                    optimizer.zero_grad(set_to_none=True)
                    return
                raise
            scaler.update()  # type: ignore[arg-type]
        else:
            clip_gradients(
                optimizer,
                clip_value,
                scaler=None,
                parameters=model.parameters(),
                norm_type=grad_clip_norm_type,
            )
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

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
        else:
            outputs = (
                _call_model(model, inputs)
                if forward_fn is None
                else _execute_forward(forward_fn, inputs)
            )
            raw_loss = loss_fn(outputs, targets)

        pending_after = pending_backward + 1
        should_step, current_target = _parse_accumulation_directive(
            step_index, raw_loss, batch, pending_after
        )

        loss_divisor = current_target
        if use_amp and autocast_dtype is not None:
            loss = raw_loss / loss_divisor
            scaler.scale(loss).backward()  # type: ignore[arg-type]
        else:
            loss = raw_loss / loss_divisor
            loss.backward()

        pending_backward = pending_after
        last_raw_loss = raw_loss

        if not should_step:
            should_step = pending_backward >= current_target

        if should_step:
            _perform_step(raw_loss)
            pending_backward = 0

        total_loss += raw_loss.detach().item() * batch_items
        total_items += batch_items
        _update_metric_sums(metric_sums, outputs, targets, metrics, batch_items)

    if pending_backward > 0 and last_raw_loss is not None:
        _perform_step(last_raw_loss)

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
