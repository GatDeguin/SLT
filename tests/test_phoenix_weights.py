"""Tests for Phoenix checkpoint helpers."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

from collections import OrderedDict

import torch

from slt.models import phoenix


class _DummyEncoder:
    def __init__(self) -> None:
        self._state = OrderedDict({"face_backbone.weight": torch.tensor([1.0])})

    def state_dict(self) -> OrderedDict[str, torch.Tensor]:
        return self._state


def test_extract_encoder_state_from_nested_model() -> None:
    encoder = _DummyEncoder()
    payload = {
        "model": {
            "state_dict": {
                "model.encoder.face_backbone.weight": torch.tensor([1.0])
            }
        }
    }

    extracted = phoenix._extract_encoder_state(payload, encoder=encoder)

    assert set(extracted.keys()) == set(encoder.state_dict().keys())
    assert torch.equal(extracted["face_backbone.weight"], torch.tensor([1.0]))


def test_extract_encoder_state_handles_encoder_prefix() -> None:
    encoder = _DummyEncoder()
    payload = {
        "state_dict": {
            "custom.module.encoder.face_backbone.weight": torch.tensor([2.0])
        }
    }

    extracted = phoenix._extract_encoder_state(payload, encoder=encoder)

    assert torch.equal(extracted["face_backbone.weight"], torch.tensor([2.0]))
