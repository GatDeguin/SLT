"""Training and evaluation loops for SLT models."""
from __future__ import annotations

import math
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

import torch
from torch import nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Optimizer
import torch.nn.functional as F

try:  # pragma: no cover - CUDA may be unavailable in CI
    from torch.cuda.amp import GradScaler, autocast  # type: ignore
except Exception:  # pragma: no cover - GradScaler/autocast optional
    GradScaler = None  # type: ignore

    @contextmanager
    def autocast(*args, **kwargs):  # type: ignore
        yield

Batch = Union[Dict[str, Any], Sequence[Any]]
Inputs = Union[torch.Tensor, Sequence[Any], MutableMapping[str, Any]]
LossFn = Callable[[Any, Any, Optional[Inputs]], Union[torch.Tensor, Tuple[torch.Tensor, Mapping[str, Any]]]]
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


def _mask_compatible(mask: torch.Tensor, tensor: torch.Tensor) -> bool:
    if mask.numel() == 0 or tensor.numel() == 0:
        return True
    if mask.shape == tensor.shape:
        return True
    if mask.dim() == 0 or tensor.dim() == 0:
        return True
    if mask.shape[0] == tensor.shape[0]:
        return True
    if mask.numel() == tensor.numel():
        return True
    return False


def _count_from_mask(mask: torch.Tensor) -> int:
    if mask.dtype == torch.bool:
        valid = mask
    else:
        valid = mask.ne(0)
    return int(valid.sum().item())


def _count_tensor_items(tensor: torch.Tensor, mask: Optional[torch.Tensor] = None) -> int:
    if mask is not None and _mask_compatible(mask, tensor):
        return _count_from_mask(mask)
    if tensor.dim() == 0:
        return 1
    if tensor.dtype in {torch.int64, torch.int32, torch.long} and tensor.dim() > 1:
        valid = tensor.ne(-100)
        count = int(valid.sum().item())
        if count > 0:
            return count
    return tensor.shape[0]


def _candidate_mask_keys(key: str) -> Tuple[str, ...]:
    overrides: Dict[str, Tuple[str, ...]] = {
        "translation": (
            "translation_attention_mask",
            "decoder_attention_mask",
            "attention_mask",
        ),
        "labels": (
            "labels_attention_mask",
            "decoder_attention_mask",
            "attention_mask",
        ),
        "ctc_labels": ("ctc_mask",),
        "scores": ("attention_mask",),
        "logits": ("attention_mask",),
    }
    base = list(overrides.get(key, ()))
    if key.endswith("_labels"):
        prefix = key[: -len("_labels")]
        base.extend((f"{prefix}_mask", f"{prefix}_attention_mask"))
    for suffix in ("_attention_mask", "_mask", "_masks"):
        base.append(f"{key}{suffix}")
    base.extend(("attention_mask", "mask"))
    seen: set[str] = set()
    ordered: list[str] = []
    for candidate in base:
        if candidate and candidate not in seen:
            seen.add(candidate)
            ordered.append(candidate)
    return tuple(ordered)


def _find_mask_for_key(
    mapping: Mapping[str, Any], key: str, tensor: torch.Tensor
) -> Optional[torch.Tensor]:
    for candidate in _candidate_mask_keys(key):
        if candidate == key:
            continue
        mask = mapping.get(candidate)
        if isinstance(mask, torch.Tensor) and _mask_compatible(mask, tensor):
            return mask
    return None


def _iter_tensor_candidates(
    mapping: Mapping[str, Any]
) -> Iterable[Tuple[str, torch.Tensor]]:
    primary: list[Tuple[str, torch.Tensor]] = []
    masks: list[Tuple[str, torch.Tensor]] = []
    for key, value in mapping.items():
        if not isinstance(value, torch.Tensor) or key == "translation":
            continue
        lower = key.lower()
        if (
            lower.endswith("mask")
            or lower.endswith("masks")
            or lower.endswith("attention_mask")
        ):
            masks.append((key, value))
        else:
            primary.append((key, value))
    yield from primary
    yield from masks


def _count_items(targets: Any) -> int:
    if isinstance(targets, torch.Tensor):
        return _count_tensor_items(targets)
    if isinstance(targets, Mapping):
        translation = targets.get("translation")
        if isinstance(translation, torch.Tensor):
            mask = _find_mask_for_key(targets, "translation", translation)
            return _count_tensor_items(translation, mask=mask)
        for key, tensor in _iter_tensor_candidates(targets):
            mask = _find_mask_for_key(targets, key, tensor)
            return _count_tensor_items(tensor, mask=mask)
        for value in targets.values():
            if isinstance(value, Mapping):
                nested = _count_items(value)
                if nested:
                    return nested
            elif isinstance(value, (list, tuple)):
                for element in value:
                    nested = _count_items(element)
                    if nested:
                        return nested
        return 1
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


def _accumulate_named_values(
    metric_sums: Dict[str, float], values: Optional[Mapping[str, Any]], batch_items: int
) -> None:
    if not values:
        return
    for name, value in values.items():
        if value is None:
            continue
        if isinstance(value, torch.Tensor):
            if value.numel() != 1:
                raise ValueError(
                    f"Loss component '{name}' must be a scalar tensor, got shape {tuple(value.shape)}."
                )
            scalar = float(value.detach().cpu().item())
        else:
            scalar = float(value)
        metric_sums[name] = metric_sums.get(name, 0.0) + scalar * batch_items


def _unwrap_loss_output(
    result: Union[torch.Tensor, Tuple[torch.Tensor, Mapping[str, Any]]]
) -> Tuple[torch.Tensor, Optional[Mapping[str, Any]]]:
    if isinstance(result, tuple):
        if len(result) != 2:
            raise ValueError("Loss function tuples must contain (loss, components)")
        loss_value, components = result
    else:
        loss_value, components = result, None
    if not isinstance(loss_value, torch.Tensor):
        raise TypeError("Loss function must return a torch.Tensor or (tensor, mapping)")
    return loss_value, components


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
                try:
                    loss_result = loss_fn(outputs, targets, inputs)
                except TypeError:
                    loss_result = loss_fn(outputs, targets)
                raw_loss, loss_components = _unwrap_loss_output(loss_result)
        else:
            outputs = (
                _call_model(model, inputs)
                if forward_fn is None
                else _execute_forward(forward_fn, inputs)
            )
            try:
                loss_result = loss_fn(outputs, targets, inputs)
            except TypeError:
                loss_result = loss_fn(outputs, targets)
            raw_loss, loss_components = _unwrap_loss_output(loss_result)

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
        _accumulate_named_values(metric_sums, loss_components, batch_items)
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
            try:
                loss_result = loss_fn(outputs, targets, inputs)
            except TypeError:
                loss_result = loss_fn(outputs, targets)
            loss, loss_components = _unwrap_loss_output(loss_result)

            batch_items = _count_items(targets)
            total_loss += loss.detach().item() * batch_items
            total_items += batch_items
            _accumulate_named_values(metric_sums, loss_components, batch_items)
            _update_metric_sums(metric_sums, outputs, targets, metrics, batch_items)

    if total_items == 0:
        raise RuntimeError("Empty evaluation loader provided.")

    averaged_metrics = {name: value / total_items for name, value in metric_sums.items()}

    return LoopResult(total_loss / total_items, averaged_metrics)


def _sequence_cross_entropy(
    logits: torch.Tensor, labels: torch.Tensor, *, label_smoothing: float
) -> torch.Tensor:
    vocab = logits.size(-1)
    flat_logits = logits.reshape(-1, vocab)
    flat_labels = labels.reshape(-1)
    kwargs = {"ignore_index": -100}
    if label_smoothing > 0:
        kwargs["label_smoothing"] = float(label_smoothing)
    try:
        return F.cross_entropy(flat_logits, flat_labels, **kwargs)
    except TypeError:  # pragma: no cover - legacy PyTorch fallback
        kwargs.pop("label_smoothing", None)
        return F.cross_entropy(flat_logits, flat_labels, **kwargs)


def _prepare_ctc_targets(
    labels: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    lengths: Optional[torch.Tensor],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    if labels is None or labels.numel() == 0:
        return None, None
    device = labels.device
    if mask is not None:
        mask_tensor = mask.to(device=device, dtype=torch.bool)
    else:
        mask_tensor = labels.ge(0)
    target_lengths = mask_tensor.sum(dim=1, dtype=torch.long)
    if lengths is not None:
        lengths = lengths.to(device=device, dtype=torch.long)
        target_lengths = torch.minimum(target_lengths, lengths)
    pieces: list[torch.Tensor] = []
    for row, row_mask, length in zip(labels, mask_tensor, target_lengths):
        valid = row[row_mask]
        if length.item() > 0:
            pieces.append(valid[: int(length.item())])
    if not pieces:
        return None, target_lengths
    concatenated = torch.cat(pieces).to(device=device, dtype=torch.long)
    return concatenated, target_lengths


def _resolve_input_lengths(
    mask: Optional[torch.Tensor], logits: torch.Tensor
) -> torch.Tensor:
    batch, time = logits.shape[:2]
    if mask is None:
        return torch.full((batch,), time, dtype=torch.long, device=logits.device)
    mask_tensor = mask.to(device=logits.device, dtype=torch.bool)
    lengths = mask_tensor.sum(dim=1, dtype=torch.long)
    return torch.clamp(lengths, min=1)


def _ctc_loss_from_logits(
    logits: torch.Tensor,
    mask: Optional[torch.Tensor],
    targets: Optional[torch.Tensor],
    target_lengths: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    if targets is None or target_lengths is None:
        return None
    if target_lengths.max().item() == 0:
        return logits.new_zeros(())
    input_lengths = _resolve_input_lengths(mask, logits)
    if torch.any(input_lengths < target_lengths):
        input_lengths = torch.maximum(input_lengths, target_lengths)
    log_probs = logits.log_softmax(dim=-1).transpose(0, 1)
    return F.ctc_loss(
        log_probs,
        targets,
        input_lengths,
        target_lengths,
        blank=0,
        zero_infinity=True,
    )


def _log_sum_exp(values: Iterable[float]) -> float:
    filtered = [value for value in values if value != -math.inf]
    if not filtered:
        return -math.inf
    maximum = max(filtered)
    if maximum == -math.inf:
        return -math.inf
    total = math.fsum(math.exp(value - maximum) for value in filtered)
    return maximum + math.log(total)


def _update_beam_entry(
    store: dict[tuple[int, ...], tuple[float, float]],
    key: tuple[int, ...],
    *,
    blank_log_prob: float | None = None,
    non_blank_log_prob: float | None = None,
) -> None:
    previous_blank, previous_non_blank = store.get(key, (-math.inf, -math.inf))
    if blank_log_prob is not None:
        previous_blank = _log_sum_exp((previous_blank, blank_log_prob))
    if non_blank_log_prob is not None:
        previous_non_blank = _log_sum_exp((previous_non_blank, non_blank_log_prob))
    store[key] = (previous_blank, previous_non_blank)


def _prune_beam(
    beam: dict[tuple[int, ...], tuple[float, float]], width: int
) -> dict[tuple[int, ...], tuple[float, float]]:
    if len(beam) <= width:
        return beam
    ranked = sorted(
        beam.items(),
        key=lambda item: _log_sum_exp(item[1]),
        reverse=True,
    )
    return dict(ranked[:width])


def ctc_beam_search(
    logits: torch.Tensor,
    mask: torch.Tensor | None = None,
    *,
    beam_width: int = 5,
    blank: int = 0,
) -> list[list[int]]:
    """Decode CTC logits using a configurable beam search."""

    if beam_width <= 0:
        raise ValueError("beam_width must be a positive integer")
    if beam_width == 1:
        return _ctc_greedy_decode(logits, mask, blank=blank)

    log_probs = logits.detach().log_softmax(dim=-1).cpu()
    batch, time, vocab = log_probs.shape
    mask_cpu: Optional[torch.Tensor] = None
    if mask is not None and mask.numel() != 0:
        mask_cpu = mask.to(device=log_probs.device, dtype=torch.bool).cpu()

    results: list[list[int]] = []
    for batch_index in range(batch):
        length = (
            int(mask_cpu[batch_index].sum().item())
            if mask_cpu is not None
            else time
        )
        if length <= 0:
            results.append([])
            continue
        beam: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -math.inf)}
        for step_index in range(length):
            step_log_probs = log_probs[batch_index, step_index]
            next_beam: dict[tuple[int, ...], tuple[float, float]] = {}
            blank_log_prob = float(step_log_probs[blank].item())
            for prefix, (blank_score, non_blank_score) in beam.items():
                blank_update = _log_sum_exp(
                    (blank_score + blank_log_prob, non_blank_score + blank_log_prob)
                )
                _update_beam_entry(
                    next_beam,
                    prefix,
                    blank_log_prob=blank_update,
                )
                total_prefix_score = _log_sum_exp((blank_score, non_blank_score))
                for symbol in range(vocab):
                    if symbol == blank:
                        continue
                    symbol_log_prob = float(step_log_probs[symbol].item())
                    if prefix and prefix[-1] == symbol:
                        non_blank_update = blank_score + symbol_log_prob
                        _update_beam_entry(
                            next_beam,
                            prefix,
                            non_blank_log_prob=non_blank_update,
                        )
                    else:
                        extension = prefix + (symbol,)
                        non_blank_update = total_prefix_score + symbol_log_prob
                        _update_beam_entry(
                            next_beam,
                            extension,
                            non_blank_log_prob=non_blank_update,
                        )
            beam = _prune_beam(next_beam, beam_width)
        ordered = sorted(
            beam.items(), key=lambda item: _log_sum_exp(item[1]), reverse=True
        )
        best_sequence = list(ordered[0][0]) if ordered else []
        results.append(best_sequence)
    return results


def _ctc_greedy_decode(
    logits: torch.Tensor, mask: Optional[torch.Tensor], blank: int = 0
) -> list[list[int]]:
    predictions = logits.detach().softmax(dim=-1).argmax(dim=-1)
    predictions_cpu = predictions.cpu()
    mask_cpu = None
    if mask is not None and mask.numel() != 0:
        mask_cpu = mask.to(device=predictions.device, dtype=torch.bool).cpu()
    batch, time = predictions_cpu.shape
    sequences: list[list[int]] = []
    for idx in range(batch):
        if mask_cpu is not None:
            length = int(mask_cpu[idx].sum().item())
        else:
            length = time
        tokens = predictions_cpu[idx, :length].tolist()
        collapsed: list[int] = []
        previous: Optional[int] = None
        for token in tokens:
            if token == blank:
                previous = None
                continue
            if previous != token:
                collapsed.append(token)
            previous = token
        sequences.append(collapsed)
    return sequences


def _distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: Optional[torch.Tensor],
    mask: Optional[torch.Tensor],
    temperature: float,
) -> Optional[torch.Tensor]:
    if teacher_logits is None:
        return None
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    student = student_logits / temperature
    teacher = teacher_logits / temperature
    log_probs = F.log_softmax(student, dim=-1)
    teacher_probs = F.softmax(teacher, dim=-1)
    per_token = F.kl_div(log_probs, teacher_probs, reduction="none").sum(dim=-1)
    if mask is not None:
        mask_tensor = mask.to(device=per_token.device, dtype=per_token.dtype)
        per_token = per_token * mask_tensor
        denom = mask_tensor.sum().clamp_min(1.0)
    else:
        denom = float(per_token.numel())
    return per_token.sum() * (temperature ** 2) / denom


def multistream_loss(
    outputs: Any,
    targets: Mapping[str, Any],
    *,
    label_smoothing: float = 0.0,
    translation_weight: float = 1.0,
    ctc_weight: float = 0.0,
    distillation_weight: float = 0.0,
    distillation_temperature: float = 1.0,
    translation_key: str = "translation",
    ctc_decode_beams: int = 5,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    labels = targets.get(translation_key)
    if labels is None:
        raise KeyError(
            f"Targets missing '{translation_key}' required for translation loss"
        )
    logits = getattr(outputs, "logits", outputs)
    translation_loss = _sequence_cross_entropy(
        logits, labels, label_smoothing=label_smoothing
    )
    weighted_translation = translation_loss * float(translation_weight)
    total_loss = weighted_translation
    components: Dict[str, Any] = {
        "loss_translation": translation_loss.detach(),
        "loss_translation_weighted": weighted_translation.detach(),
    }

    auxiliary = getattr(outputs, "auxiliary", None)
    combined_entry = auxiliary.get("combined") if auxiliary else None
    ctc_loss_value = None
    if ctc_weight != 0 and auxiliary:
        ctc_labels = targets.get("ctc_labels")
        ctc_mask = targets.get("ctc_mask")
        ctc_lengths = targets.get("ctc_lengths")
        targets_concat, target_lengths = _prepare_ctc_targets(
            ctc_labels, ctc_mask, ctc_lengths
        )
        if targets_concat is not None and target_lengths is not None:
            fused = auxiliary.get("fused", {})
            fused_logits = fused.get("logits")
            fused_mask = fused.get("mask")
            streams = auxiliary.get("stream", {})
            frame_masks = auxiliary.get("frame_masks", {})
            losses: list[torch.Tensor] = []
            if fused_logits is not None:
                fused_loss = _ctc_loss_from_logits(
                    fused_logits, fused_mask, targets_concat, target_lengths
                )
                if fused_loss is not None:
                    losses.append(fused_loss)
            for name, stream_logits in streams.items():
                stream_mask = frame_masks.get(name)
                stream_loss = _ctc_loss_from_logits(
                    stream_logits, stream_mask, targets_concat, target_lengths
                )
                if stream_loss is not None:
                    losses.append(stream_loss)
            if losses:
                ctc_loss_value = sum(losses)
                weighted_ctc = ctc_loss_value * float(ctc_weight)
                total_loss = total_loss + weighted_ctc
                components["loss_ctc"] = ctc_loss_value.detach()
                components["loss_ctc_weighted"] = weighted_ctc.detach()

    distillation_loss_value = None
    if distillation_weight != 0 and auxiliary:
        streams = auxiliary.get("stream", {})
        teachers = auxiliary.get("distillation", {})
        frame_masks = auxiliary.get("frame_masks", {})
        default_teacher = None
        if combined_entry is not None:
            default_teacher = combined_entry.get("logits")
        losses: list[torch.Tensor] = []
        for name, stream_logits in streams.items():
            teacher_logits = teachers.get(name, default_teacher)
            mask = frame_masks.get(name)
            dist = _distillation_loss(
                stream_logits,
                teacher_logits,
                mask,
                float(distillation_temperature),
            )
            if dist is not None:
                losses.append(dist)
        if losses:
            distillation_loss_value = sum(losses) / len(losses)
            weighted_dist = distillation_loss_value * float(distillation_weight)
            total_loss = total_loss + weighted_dist
            components["loss_distillation"] = distillation_loss_value.detach()
            components["loss_distillation_weighted"] = weighted_dist.detach()

    if combined_entry is not None:
        combined_logits = combined_entry.get("logits")
        if combined_logits is not None:
            components["ctc_ensemble_logits"] = combined_logits.detach()
            combined_mask = combined_entry.get("mask")
            beam_width = max(1, int(ctc_decode_beams))
            components["ctc_ensemble_sequence"] = ctc_beam_search(
                combined_logits, combined_mask, beam_width=beam_width
            )

    return total_loss, components
