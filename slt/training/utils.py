"""Utility helpers for model preparation during training."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Union

from transformers import AutoTokenizer, PreTrainedTokenizerBase

import torch.nn as nn


def _matches_prefix(name: str, prefixes: Optional[Iterable[str]]) -> bool:
    if not prefixes:
        return True
    for prefix in prefixes:
        if name.startswith(prefix):
            return True
    return False


def freeze_module(
    module: nn.Module,
    *,
    parameter_prefixes: Optional[Iterable[str]] = None,
    exclude_prefixes: Optional[Iterable[str]] = None,
) -> None:
    """Freeze parameters in ``module`` matching the provided prefixes."""

    exclude = tuple(exclude_prefixes or ())
    for name, param in module.named_parameters():
        if param is None:
            continue
        if exclude and any(name.startswith(prefix) for prefix in exclude):
            continue
        if _matches_prefix(name, parameter_prefixes):
            param.requires_grad = False


def unfreeze_module(
    module: nn.Module,
    *,
    parameter_prefixes: Optional[Iterable[str]] = None,
    exclude_prefixes: Optional[Iterable[str]] = None,
) -> None:
    """Unfreeze parameters in ``module`` matching the provided prefixes."""

    exclude = tuple(exclude_prefixes or ())
    for name, param in module.named_parameters():
        if param is None:
            continue
        if exclude and any(name.startswith(prefix) for prefix in exclude):
            continue
        if _matches_prefix(name, parameter_prefixes):
            param.requires_grad = True


def _unique_tokens(tokens: Iterable[str]) -> Sequence[str]:
    seen = set()
    ordered = []
    for token in tokens:
        if not token:
            continue
        if token in seen:
            continue
        seen.add(token)
        ordered.append(token)
    return ordered


def load_tokenizer(
    name_or_path: str,
    *,
    use_fast: bool = True,
    extra_tokens: Optional[Sequence[str]] = None,
    extra_tokens_file: Optional[Union[str, Path]] = None,
    special_tokens: Optional[Mapping[str, Union[str, Sequence[str]]]] = None,
    **kwargs: object,
) -> PreTrainedTokenizerBase:
    """Load a tokenizer and optionally extend its vocabulary."""

    tokenizer = AutoTokenizer.from_pretrained(name_or_path, use_fast=use_fast, **kwargs)
    tokens: Sequence[str] = ()
    if extra_tokens_file is not None:
        lines = Path(extra_tokens_file).read_text(encoding="utf-8").splitlines()
        tokens = tuple(line.strip() for line in lines if line.strip())
    if extra_tokens:
        tokens = (*tokens, *extra_tokens)
    tokens = _unique_tokens(tokens)
    if special_tokens:
        tokenizer.add_special_tokens(dict(special_tokens))
    if tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": list(tokens)})
    return tokenizer
