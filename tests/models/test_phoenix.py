"""Tests for Phoenix checkpoint helpers."""

from __future__ import annotations

from collections import OrderedDict
import sys
import types

import pytest

if "transformers" not in sys.modules:
    transformers_stub = types.ModuleType("transformers")

    class _DummyPretrainedConfig:
        pass

    class _DummyAutoConfig(_DummyPretrainedConfig):
        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> "_DummyAutoConfig":
            return cls()

    class _DummyAutoModel:
        def __init__(self) -> None:
            self.config = _DummyPretrainedConfig()

        @classmethod
        def from_config(cls, config: _DummyPretrainedConfig) -> "_DummyAutoModel":
            instance = cls()
            instance.config = config
            return instance

        @classmethod
        def from_pretrained(cls, *args: object, **kwargs: object) -> "_DummyAutoModel":
            return cls()

    class _DummyT5(_DummyAutoModel):
        pass

    transformers_stub.AutoConfig = _DummyAutoConfig
    transformers_stub.AutoModelForSeq2SeqLM = _DummyAutoModel
    transformers_stub.PretrainedConfig = _DummyPretrainedConfig
    transformers_stub.T5ForConditionalGeneration = _DummyT5

    outputs_stub = types.ModuleType("transformers.modeling_outputs")
    outputs_stub.BaseModelOutput = type("BaseModelOutput", (), {})
    outputs_stub.Seq2SeqLMOutput = type("Seq2SeqLMOutput", (), {})

    sys.modules["transformers"] = transformers_stub
    sys.modules["transformers.modeling_outputs"] = outputs_stub

from slt.models.phoenix import _extract_encoder_state


class _DummyEncoder:
    def __init__(self) -> None:
        self._state = OrderedDict(
            (
                ("layer.weight", 1),
                ("layer.bias", 2),
            )
        )

    def state_dict(self) -> OrderedDict[str, int]:  # pragma: no cover - helper behaviour
        return self._state


@pytest.fixture()
def dummy_encoder() -> _DummyEncoder:
    return _DummyEncoder()


def test_extract_encoder_state_from_model_state(dummy_encoder: _DummyEncoder) -> None:
    payload = {
        "model_state": {
            "layer.weight": 10,
            "layer.bias": 20,
            "decoder.weight": 30,
        }
    }

    state = _extract_encoder_state(payload, encoder=dummy_encoder)

    assert dict(state) == {"layer.weight": 10, "layer.bias": 20}


def test_extract_encoder_state_from_state_dict(dummy_encoder: _DummyEncoder) -> None:
    payload = {
        "state_dict": {
            "encoder.layer.weight": 3,
            "encoder.layer.bias": 4,
            "encoder.extra": 5,
            "decoder.layer.weight": 6,
        }
    }

    state = _extract_encoder_state(payload, encoder=dummy_encoder)

    assert dict(state) == {"layer.weight": 3, "layer.bias": 4}


def test_extract_encoder_state_missing(dummy_encoder: _DummyEncoder) -> None:
    with pytest.raises(ValueError):
        _extract_encoder_state({}, encoder=dummy_encoder)
