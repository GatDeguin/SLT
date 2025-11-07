"""Tests for Phoenix checkpoint helpers."""

from __future__ import annotations

import sys
import types
from collections import OrderedDict

import pytest

if "torch" not in sys.modules:
    torch_stub = types.ModuleType("torch")

    def _unsupported_torch_load(*args: object, **kwargs: object) -> None:
        raise RuntimeError("torch.load is not available in the test environment")

    torch_stub.load = _unsupported_torch_load
    torch_stub.hub = types.ModuleType("torch.hub")
    torch_stub.hub.load = _unsupported_torch_load

    class _DummyTensor:
        pass

    class _DummyModule:
        def __init__(self, *args: object, **kwargs: object) -> None:
            return

        def parameters(self) -> tuple[object, ...]:  # pragma: no cover - helper compatibility
            return ()

        def requires_grad_(self, flag: bool) -> "_DummyModule":  # pragma: no cover
            return self

        def __call__(self, *args: object, **kwargs: object) -> "_DummyModule":
            return self

    class _DummySequential(_DummyModule):
        pass

    nn_stub = types.ModuleType("torch.nn")
    nn_stub.Module = _DummyModule
    nn_stub.Sequential = _DummySequential
    nn_stub.LayerNorm = _DummyModule
    nn_stub.Linear = _DummyModule
    nn_stub.Dropout = _DummyModule
    nn_stub.GELU = _DummyModule

    torch_stub.Tensor = _DummyTensor
    torch_stub.nn = nn_stub
    nn_functional_stub = types.ModuleType("torch.nn.functional")
    nn_stub.functional = nn_functional_stub
    sys.modules["torch"] = torch_stub
    sys.modules["torch.hub"] = torch_stub.hub
    sys.modules["torch.nn"] = nn_stub
    sys.modules["torch.nn.functional"] = nn_functional_stub

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


def test_extract_encoder_state_from_direct_mapping(
    dummy_encoder: _DummyEncoder,
) -> None:
    payload = OrderedDict(
        (
            ("layer.weight", 11),
            ("layer.bias", 22),
        )
    )

    state = _extract_encoder_state(payload, encoder=dummy_encoder)

    assert dict(state) == {"layer.weight": 11, "layer.bias": 22}


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
