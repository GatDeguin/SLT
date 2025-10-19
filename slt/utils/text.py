"""Tokenization helpers built on top of Hugging Face tokenizers."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase


def create_tokenizer(
    name_or_path: str,
    *,
    use_fast: bool = True,
    revision: Optional[str] = None,
    **kwargs,
) -> PreTrainedTokenizerBase:
    """Instantiate a :class:`~transformers.PreTrainedTokenizerBase`.

    Parameters
    ----------
    name_or_path:
        Identifier passed to :func:`transformers.AutoTokenizer.from_pretrained`.
    use_fast:
        Whether to request the fast tokenizer implementation when available.
    revision:
        Optional model revision (branch, tag or commit hash).
    kwargs:
        Additional keyword arguments forwarded to ``from_pretrained``.
    """

    return AutoTokenizer.from_pretrained(
        name_or_path,
        use_fast=use_fast,
        revision=revision,
        **kwargs,
    )


def encode_batch(
    tokenizer: PreTrainedTokenizerBase,
    texts: Iterable[str],
    *,
    max_length: Optional[int] = None,
    padding: Union[bool, str] = "longest",
    truncation: bool = True,
    return_tensors: str = "pt",
) -> "BatchEncoding":
    """Encode a batch of texts using the provided tokenizer."""

    text_list = list(texts)
    if not text_list:
        raise ValueError("encode_batch received an empty iterable of texts.")

    if max_length is None:
        max_length = tokenizer.model_max_length

    return tokenizer(
        text_list,
        max_length=max_length,
        padding=padding,
        truncation=truncation,
        return_tensors=return_tensors,
    )


def decode(
    tokenizer: PreTrainedTokenizerBase,
    sequences: Union[torch.Tensor, Sequence[Sequence[int]], Sequence[int]],
    *,
    skip_special_tokens: bool = True,
    clean_up_tokenization_spaces: bool = True,
) -> List[str]:
    """Decode sequences of token IDs into strings."""

    if isinstance(sequences, torch.Tensor):
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        seq_list: List[Sequence[int]] = sequences.detach().cpu().tolist()
    elif isinstance(sequences, Sequence) and sequences and isinstance(sequences[0], int):
        seq_list = [sequences]  # type: ignore[list-item]
    else:
        seq_list = list(sequences)  # type: ignore[list-item]

    return tokenizer.batch_decode(
        seq_list,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    )
