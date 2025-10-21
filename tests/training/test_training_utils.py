"""Tests for :mod:`slt.training.utils` and evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import List

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("tokenizers")

from torch import nn
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from slt.models import TextSeq2SeqDecoder
from slt.training.evaluation import EvaluationOutputs, evaluate_model
from slt.training.loops import LoopResult, _split_batch
from slt.training.utils import freeze_module, load_tokenizer, unfreeze_module


class TinyModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.norm = nn.LayerNorm(4)

    def forward(self, tensor: torch.Tensor) -> torch.Tensor:
        return self.norm(self.linear(tensor))


class DummySeq2Seq(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.decoder = TextSeq2SeqDecoder(
            d_model=d_model,
            vocab_size=vocab_size,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
        )

    def forward(self, **inputs: torch.Tensor):
        return self.decoder(
            inputs["face"],
            encoder_attention_mask=inputs.get("encoder_attention_mask"),
            labels=inputs.get("labels"),
            decoder_attention_mask=inputs.get("decoder_attention_mask"),
        )

    def generate(self, **inputs: torch.Tensor):
        hidden = inputs["face"]
        encoder_attention_mask = inputs.get("encoder_attention_mask")
        blocked = {
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
        generation_kwargs = {k: v for k, v in inputs.items() if k not in blocked}
        return self.decoder.generate(
            hidden,
            encoder_attention_mask=encoder_attention_mask,
            **generation_kwargs,
        )


def _build_tokenizer(directory: Path) -> PreTrainedTokenizerFast:
    directory.mkdir(parents=True, exist_ok=True)
    vocab = {"[PAD]": 0, "[UNK]": 1, "[EOS]": 2, "hola": 3}
    tokenizer = Tokenizer(WordLevel(vocab, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = Whitespace()
    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        pad_token="[PAD]",
        eos_token="[EOS]",
        unk_token="[UNK]",
    )
    fast_tokenizer.model_max_length = 8
    fast_tokenizer.save_pretrained(directory)
    return fast_tokenizer


def test_freeze_and_unfreeze_module() -> None:
    module = TinyModule()
    freeze_module(module, parameter_prefixes=("linear",))
    assert all(not p.requires_grad for p in module.linear.parameters())
    assert any(p.requires_grad for p in module.norm.parameters())

    unfreeze_module(module, parameter_prefixes=("linear",))
    assert all(p.requires_grad for p in module.linear.parameters())

    freeze_module(module)
    assert all(not p.requires_grad for p in module.parameters())


def test_load_tokenizer_supports_extra_tokens(tmp_path: Path) -> None:
    directory = tmp_path / "tok"
    _build_tokenizer(directory)
    extra_file = tmp_path / "special.txt"
    extra_file.write_text("<s>\n</s>\n", encoding="utf-8")

    tokenizer = load_tokenizer(
        str(directory),
        extra_tokens=("<extra>",),
        extra_tokens_file=extra_file,
        special_tokens={
            "bos_token": "<s>",
            "additional_special_tokens": ["</s>"],
        },
    )

    vocab = tokenizer.get_vocab()
    assert "<extra>" in vocab
    assert "<s>" in tokenizer.all_special_tokens


def _build_evaluation_batch(
    tokenizer: PreTrainedTokenizerFast,
    *,
    batch_size: int,
    encoder_steps: int,
    target_steps: int,
    d_model: int,
) -> List[dict]:
    face = torch.randn(batch_size, encoder_steps, d_model)
    mask = torch.ones(batch_size, encoder_steps, dtype=torch.long)
    decoder_mask = torch.ones(batch_size, target_steps, dtype=torch.long)
    labels = torch.randint(0, tokenizer.vocab_size, (batch_size, target_steps))
    labels[:, -1] = tokenizer.eos_token_id or 0
    masked_labels = labels.clone()
    masked_labels[:, -1] = -100
    batch = {
        "inputs": {
            "face": face,
            "hand_l": torch.zeros_like(face),
            "hand_r": torch.zeros_like(face),
            "pose": torch.zeros_like(face),
            "pose_conf_mask": torch.ones(batch_size, encoder_steps),
            "pad_mask": mask.bool(),
            "miss_mask_hl": torch.zeros(batch_size, encoder_steps, dtype=torch.bool),
            "miss_mask_hr": torch.zeros(batch_size, encoder_steps, dtype=torch.bool),
            "encoder_attention_mask": mask,
            "decoder_attention_mask": decoder_mask,
            "labels": masked_labels,
        },
        "labels": masked_labels,
        "video_ids": [f"video-{i}" for i in range(batch_size)],
    }
    return [batch]


def test_evaluate_model_returns_predictions(tmp_path: Path) -> None:
    directory = tmp_path / "tok"
    tokenizer = _build_tokenizer(directory)
    model = DummySeq2Seq(d_model=32, vocab_size=tokenizer.vocab_size)
    batch = _build_evaluation_batch(
        tokenizer,
        batch_size=2,
        encoder_steps=3,
        target_steps=4,
        d_model=model.decoder.config.d_model,
    )

    result = evaluate_model(
        model,
        batch,
        lambda outputs, _: outputs.loss,
        tokenizer,
        device="cpu",
        generate_kwargs={"max_length": 4},
    )

    assert isinstance(result, EvaluationOutputs)
    assert isinstance(result.loop_result, LoopResult)
    assert len(result.predictions) == len(result.references) == 2
    assert list(result.video_ids) == ["video-0", "video-1"]


def test_split_batch_prefers_inputs_and_ignores_metadata() -> None:
    batch_inputs = {"face": torch.randn(2, 3, 4)}
    labels = torch.randint(0, 10, (2, 5))
    batch = {
        "inputs": batch_inputs,
        "labels": labels,
        "video_ids": ["video-0", "video-1"],
    }

    inputs, targets = _split_batch(batch)

    assert inputs is batch_inputs
    assert targets is labels
    assert "video_ids" not in inputs
