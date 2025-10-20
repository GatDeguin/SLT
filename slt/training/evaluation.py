"""Evaluation helpers built on top of :mod:`slt.training.loops`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Union

import torch
from torch import nn
from transformers import PreTrainedTokenizerBase

from .loops import LoopResult

MetricFn = Callable[[Any, Any], Union[float, torch.Tensor]]

_GENERATION_INPUT_KEYS = {
    "face",
    "hand_l",
    "hand_r",
    "pose",
    "pad_mask",
    "miss_mask_hl",
    "miss_mask_hr",
    "pose_conf_mask",
    "encoder_attention_mask",
}


@dataclass
class EvaluationOutputs:
    """Aggregated metrics and decoded predictions for an evaluation run."""

    loop_result: LoopResult
    predictions: Sequence[str]
    references: Sequence[str]
    video_ids: Sequence[str]


def _move_to_device(data: Any, device: Union[str, torch.device]) -> Any:
    if isinstance(data, torch.Tensor):
        return data.to(device)
    if isinstance(data, Mapping):
        return {k: _move_to_device(v, device) for k, v in data.items()}
    if isinstance(data, (list, tuple)):
        return type(data)(_move_to_device(v, device) for v in data)
    return data


def _count_items(targets: Any) -> int:
    if isinstance(targets, torch.Tensor):
        if targets.dim() == 0:
            return 1
        if targets.dtype in {torch.long, torch.int64, torch.int32} and targets.dim() > 1:
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
    metric_fns: Optional[Mapping[str, MetricFn]],
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


def _prepare_generation_inputs(batch_inputs: Mapping[str, Any]) -> Dict[str, Any]:
    return {key: batch_inputs[key] for key in _GENERATION_INPUT_KEYS if key in batch_inputs}


def _decode_labels(
    labels: torch.Tensor,
    tokenizer: PreTrainedTokenizerBase,
    *,
    skip_special_tokens: bool,
) -> List[str]:
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    clean = labels.detach().to(torch.long).clone()
    clean[clean == -100] = pad_token_id
    return tokenizer.batch_decode(clean.cpu(), skip_special_tokens=skip_special_tokens)


def evaluate_model(
    model: nn.Module,
    loader: Iterable[Mapping[str, Any]],
    loss_fn: Callable[[Any, Any], torch.Tensor],
    tokenizer: PreTrainedTokenizerBase,
    *,
    device: Union[str, torch.device] = "cuda",
    metrics: Optional[Mapping[str, MetricFn]] = None,
    generate_kwargs: Optional[Mapping[str, Any]] = None,
    skip_special_tokens: bool = True,
) -> EvaluationOutputs:
    """Run evaluation and decode predictions using ``tokenizer``."""

    model.eval()
    total_loss = 0.0
    total_items = 0
    metric_sums: Dict[str, float] = {}
    predictions: List[str] = []
    references: List[str] = []
    collected_ids: List[str] = []
    generation_options = dict(generate_kwargs or {})

    with torch.no_grad():
        for batch in loader:
            batch_inputs = batch.get("inputs")
            if batch_inputs is None:
                raise KeyError("Evaluation batch must include an 'inputs' mapping")
            labels = batch_inputs.get("labels", batch.get("labels"))
            if labels is None:
                raise KeyError("Evaluation batch must include labels for loss computation")

            moved_inputs = _move_to_device(batch_inputs, device)
            outputs = model(**moved_inputs)
            labels_device = moved_inputs.get("labels")
            if labels_device is None:
                labels_device = _move_to_device(labels, device)

            loss = loss_fn(outputs, labels_device)
            batch_items = _count_items(labels_device)
            total_loss += loss.detach().item() * batch_items
            total_items += batch_items
            _update_metric_sums(metric_sums, outputs, labels_device, metrics, batch_items)

            generation_inputs = _prepare_generation_inputs(moved_inputs)
            token_ids = model.generate(**generation_inputs, **generation_options)
            predictions.extend(
                tokenizer.batch_decode(
                    token_ids.detach().cpu(), skip_special_tokens=skip_special_tokens
                )
            )
            references.extend(
                _decode_labels(labels_device, tokenizer, skip_special_tokens=skip_special_tokens)
            )
            batch_ids = batch.get("video_ids")
            if batch_ids:
                collected_ids.extend(str(item) for item in batch_ids)

    if total_items == 0:
        raise RuntimeError("Empty evaluation loader provided.")

    averaged_metrics = {name: value / total_items for name, value in metric_sums.items()}
    loop_result = LoopResult(total_loss / total_items, averaged_metrics)
    return EvaluationOutputs(loop_result, predictions, references, collected_ids)
