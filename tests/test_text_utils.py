from typing import Any, Dict, Iterable, List, Sequence

import pytest

torch = pytest.importorskip("torch")

from slt.utils.text import (
    TokenizerValidationError,
    character_error_rate,
    decode,
    encode_batch,
    levenshtein_distance,
    validate_tokenizer,
    word_error_rate,
)


class DummyTokenizer:
    """Minimal tokenizer emulating the subset of Hugging Face used in tests."""

    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.model_max_length = 16
        self.vocab: Dict[str, int] = {"<pad>": 0, "</s>": 1, "hola": 2, "mundo": 3}
        self._last_call: Dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.vocab)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        tokens = []
        for word in text.split():
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab)
            tokens.append(self.vocab[word])
        if add_special_tokens:
            return [self.pad_token_id, *tokens, self.eos_token_id]
        return tokens

    def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
        id_to_token = {idx: token for token, idx in self.vocab.items()}
        decoded: List[str] = []
        for idx in token_ids:
            if skip_special_tokens and idx in {self.pad_token_id, self.eos_token_id}:
                continue
            decoded.append(id_to_token.get(idx, f"unk_{idx}"))
        return " ".join(decoded)

    def batch_decode(
        self,
        sequences: Sequence[Sequence[int]],
        skip_special_tokens: bool = True,
        clean_up_tokenization_spaces: bool = True,
    ) -> List[str]:
        del clean_up_tokenization_spaces
        return [self.decode(seq, skip_special_tokens=skip_special_tokens) for seq in sequences]

    def __call__(self, batch: Iterable[str], **kwargs: Any) -> Dict[str, Any]:
        self._last_call = {"batch": list(batch), **kwargs}
        return {"encoded": [self.encode(text) for text in self._last_call["batch"]]}


def test_validate_tokenizer_success() -> None:
    tokenizer = DummyTokenizer()
    validate_tokenizer(tokenizer)


def test_validate_tokenizer_missing_required_token() -> None:
    tokenizer = DummyTokenizer()
    tokenizer.pad_token_id = None  # type: ignore[assignment]
    with pytest.raises(TokenizerValidationError):
        validate_tokenizer(tokenizer)


def test_validate_tokenizer_rejects_negative_ids() -> None:
    tokenizer = DummyTokenizer()
    tokenizer.eos_token_id = -1
    with pytest.raises(TokenizerValidationError):
        validate_tokenizer(tokenizer)


def test_validate_tokenizer_rejects_empty_decode() -> None:
    class BlankTokenizer(DummyTokenizer):
        def decode(self, token_ids: Sequence[int], skip_special_tokens: bool = True) -> str:
            return "  "

    tokenizer = BlankTokenizer()
    with pytest.raises(TokenizerValidationError):
        validate_tokenizer(tokenizer)


def test_encode_batch_forwards_parameters() -> None:
    tokenizer = DummyTokenizer()
    batch = ["hola", "hola mundo"]
    result = encode_batch(
        tokenizer,
        batch,
        max_length=8,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    assert result["encoded"]
    assert tokenizer._last_call == {
        "batch": batch,
        "max_length": 8,
        "padding": True,
        "truncation": False,
        "return_tensors": "pt",
    }


def test_encode_batch_rejects_empty_iterable() -> None:
    tokenizer = DummyTokenizer()
    with pytest.raises(ValueError):
        encode_batch(tokenizer, [])


def test_decode_handles_tensor_inputs() -> None:
    tokenizer = DummyTokenizer()
    tensor = torch.tensor([[0, 2, 1], [0, 3, 1]], dtype=torch.long)
    decoded = decode(tokenizer, tensor)
    assert decoded == ["hola", "mundo"]


def test_decode_rejects_none_entries() -> None:
    tokenizer = DummyTokenizer()
    with pytest.raises(ValueError):
        decode(tokenizer, [None])  # type: ignore[list-item]


def test_character_error_rate_matches_expected_percentage() -> None:
    references = ["hola"]
    predictions = ["bola"]
    cer = character_error_rate(references, predictions)
    assert cer == pytest.approx(25.0)


def test_word_error_rate_matches_expected_percentage() -> None:
    references = ["hola mundo"]
    predictions = ["hola"]
    wer = word_error_rate(references, predictions)
    assert wer == pytest.approx(50.0)


def test_levenshtein_distance_symmetry() -> None:
    assert levenshtein_distance(["hola"], ["hola"]) == 0
    forward = levenshtein_distance(["hola"], ["hola", "mundo"])
    backward = levenshtein_distance(["hola", "mundo"], ["hola"])
    assert forward == backward == 1
    assert levenshtein_distance([], ["hola"]) == 1


def test_error_rates_handle_mismatched_lengths() -> None:
    references = ["hola", "mundo"]
    predictions = ["hola"]
    cer = character_error_rate(references, predictions)
    wer = word_error_rate(references, predictions)
    assert cer == pytest.approx(55.5555, rel=1e-4)
    assert wer == pytest.approx(50.0)


def test_error_rates_penalize_extra_predictions() -> None:
    references = []
    predictions = ["hola"]
    cer = character_error_rate(references, predictions)
    wer = word_error_rate(references, predictions)
    assert cer == pytest.approx(100.0)
    assert wer == pytest.approx(100.0)

