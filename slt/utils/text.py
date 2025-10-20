"""Tokenization helpers built on top of Hugging Face tokenizers."""

from __future__ import annotations

from typing import Callable, Iterable, List, Optional, Sequence, Union

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

__all__ = [
    "TokenizerValidationError",
    "create_tokenizer",
    "decode",
    "encode_batch",
    "validate_tokenizer",
    "character_error_rate",
    "word_error_rate",
    "levenshtein_distance",
]


class TokenizerValidationError(ValueError):
    """Raised when a tokenizer does not meet the expected requirements."""



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


def validate_tokenizer(
    tokenizer: PreTrainedTokenizerBase,
    *,
    required_special_tokens: Sequence[str] = ("pad_token_id", "eos_token_id"),
    probe_text: str = "tokenizer validation sample",
    allow_empty_decode: bool = False,
) -> None:
    """Validate that a tokenizer can be safely used during inference.

    The checks implemented here are intentionally lightweight so they can be
    executed in CLI utilities before the heavy evaluation loop runs.  They aim
    to catch the most common configuration errors such as missing special
    tokens or models distributed without a proper vocabulary file.

    Parameters
    ----------
    tokenizer:
        Tokenizer instance to validate.
    required_special_tokens:
        Iterable of attribute names that must resolve to non-``None`` IDs.
    probe_text:
        Text used to perform a round-trip encode/decode check.
    allow_empty_decode:
        When ``False`` the decoded probe text must contain at least one
        non-space character.
    """

    missing_tokens = [
        attr
        for attr in required_special_tokens
        if getattr(tokenizer, attr, None) is None
    ]
    if missing_tokens:
        raise TokenizerValidationError(
            f"Tokenizer is missing required special tokens: {', '.join(missing_tokens)}"
        )

    vocab_size = getattr(tokenizer, "vocab_size", None) or len(tokenizer)
    if not isinstance(vocab_size, int) or vocab_size <= 0:
        raise TokenizerValidationError("Tokenizer does not expose a valid vocabulary size.")

    try:
        encoded = tokenizer.encode(probe_text, add_special_tokens=True)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise TokenizerValidationError("Tokenizer failed to encode probe text.") from exc

    if not encoded:
        raise TokenizerValidationError("Tokenizer returned an empty encoding for the probe text.")

    try:
        decoded = tokenizer.decode(encoded, skip_special_tokens=True)
    except Exception as exc:  # pragma: no cover - defensive guard
        raise TokenizerValidationError("Tokenizer failed to decode probe ids.") from exc

    if not allow_empty_decode and decoded.strip() == "":
        raise TokenizerValidationError(
            "Decoded probe text is empty; check special token configuration."
        )

    # ensure the tokenizer exposes integer IDs for the configured attributes
    for attr in required_special_tokens:
        value = getattr(tokenizer, attr, None)
        if not isinstance(value, int):
            raise TokenizerValidationError(f"Tokenizer attribute '{attr}' must be an integer ID.")
        if value < 0:
            raise TokenizerValidationError(
                f"Tokenizer attribute '{attr}' must be a non-negative integer ID."
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
    if any(text is None for text in text_list):
        raise ValueError("encode_batch received a None entry; please filter inputs beforehand.")

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

    if not seq_list:
        return []
    if any(seq is None for seq in seq_list):
        raise ValueError("decode received a None entry; please filter inputs beforehand.")

    return tokenizer.batch_decode(
        seq_list,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_tokenization_spaces,
    )


def levenshtein_distance(reference: Sequence[str], prediction: Sequence[str]) -> int:
    if reference == prediction:
        return 0
    if not reference:
        return len(prediction)
    if not prediction:
        return len(reference)
    prev_row = list(range(len(prediction) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, pred_item in enumerate(prediction, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = prev_row[j] + 1
            replace_cost = prev_row[j - 1] + (ref_item != pred_item)
            current.append(min(insert_cost, delete_cost, replace_cost))
        prev_row = current
    return prev_row[-1]


def _error_rate(
    references: Sequence[str],
    predictions: Sequence[str],
    *,
    tokenizer: Callable[[str], Sequence[str]],
) -> float:
    if not references or not predictions:
        return 0.0

    total_distance = 0
    total_length = 0
    for ref, pred in zip(references, predictions):
        ref_seq = tokenizer(ref or "")
        pred_seq = tokenizer(pred or "")
        total_distance += levenshtein_distance(ref_seq, pred_seq)
        total_length += max(len(ref_seq), 1)

    if total_length == 0:
        return 0.0
    return (total_distance / total_length) * 100.0


def character_error_rate(references: Sequence[str], predictions: Sequence[str]) -> float:
    """Compute the character error rate (CER) expressed as a percentage."""

    return _error_rate(references, predictions, tokenizer=list)


def word_error_rate(references: Sequence[str], predictions: Sequence[str]) -> float:
    """Compute the word error rate (WER) expressed as a percentage."""

    return _error_rate(references, predictions, tokenizer=lambda text: text.split())
