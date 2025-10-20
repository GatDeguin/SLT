"""Tests for the realtime demo helpers."""

from __future__ import annotations

import argparse
import importlib.machinery
import sys
import types
from pathlib import Path

import pytest
import torch

if "cv2" not in sys.modules:
    cv2_stub = types.ModuleType("cv2")
    cv2_stub.__spec__ = importlib.machinery.ModuleSpec("cv2", loader=None)  # type: ignore[attr-defined]
    sys.modules["cv2"] = cv2_stub

from tools import demo_realtime_multistream as demo


class _DummyTokenizer:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    def batch_decode(self, sequences, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        return list(self._texts)


def _make_args(**kwargs) -> argparse.Namespace:
    defaults = {"tokenizer": None, "tokenizer_revision": None}
    defaults.update(kwargs)
    return argparse.Namespace(**defaults)


def test_decode_sequences_returns_first_text() -> None:
    tokenizer = _DummyTokenizer(["hola", "chau"])
    sequences = torch.tensor([[1, 2, 3]])
    assert demo.decode_sequences(sequences, tokenizer) == "hola"


def test_decode_sequences_requires_tokenizer() -> None:
    with pytest.raises(ValueError, match="tokenizador"):
        demo.decode_sequences(torch.tensor([1, 2, 3]), None)


def test_build_tokenizer_requires_identifier() -> None:
    args = _make_args(tokenizer=None)
    with pytest.raises(ValueError, match="--tokenizer"):
        demo.build_tokenizer(args)


def test_build_tokenizer_wraps_creation_error(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _make_args(tokenizer="dummy")

    def _fail(*_args, **_kwargs):
        raise OSError("not found")

    monkeypatch.setattr(demo, "create_tokenizer", _fail)

    with pytest.raises(RuntimeError, match="No se pudo cargar el tokenizador"):
        demo.build_tokenizer(args)


def test_build_tokenizer_validates_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _make_args(tokenizer="dummy")
    tokenizer = _DummyTokenizer(["demo"])

    monkeypatch.setattr(demo, "create_tokenizer", lambda *_args, **_kwargs: tokenizer)

    def _fail_validation(_tokenizer):
        raise demo.TokenizerValidationError("missing pad_token_id")

    monkeypatch.setattr(demo, "validate_tokenizer", _fail_validation)

    with pytest.raises(RuntimeError, match="missing pad_token_id"):
        demo.build_tokenizer(args)


def test_build_tokenizer_returns_valid_instance(monkeypatch: pytest.MonkeyPatch) -> None:
    args = _make_args(tokenizer="dummy", tokenizer_revision="main")
    tokenizer = _DummyTokenizer(["demo"])

    monkeypatch.setattr(demo, "create_tokenizer", lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(demo, "validate_tokenizer", lambda _tokenizer: None)

    assert demo.build_tokenizer(args) is tokenizer


def test_validate_model_arguments_requires_model_path() -> None:
    with pytest.raises(ValueError, match="--model"):
        demo.validate_model_arguments(None, "auto")


def test_validate_model_arguments_requires_path_for_exported_formats() -> None:
    with pytest.raises(ValueError, match="torchscript"):
        demo.validate_model_arguments(None, "torchscript")


def test_validate_model_arguments_accepts_stub_without_model() -> None:
    demo.validate_model_arguments(None, "stub")


def test_validate_model_arguments_accepts_path(tmp_path: Path) -> None:
    model_path = tmp_path / "model.ts"
    model_path.write_text("stub")
    demo.validate_model_arguments(model_path, "torchscript")
