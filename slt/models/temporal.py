"""Temporal encoder and lightweight decoder utilities."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type

import torch
from torch import Tensor, nn
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    PretrainedConfig,
    T5ForConditionalGeneration,
)
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

try:  # pragma: no cover - optional dependency in transformers
    from huggingface_hub import hf_hub_download
except Exception:  # pragma: no cover - used only when available
    hf_hub_download = None  # type: ignore

try:  # pragma: no cover - optional helper
    from huggingface_hub.utils import OfflineModeIsEnabled
except Exception:  # pragma: no cover - fallback when helper missing
    OfflineModeIsEnabled = type("OfflineModeIsEnabled", (), {})  # type: ignore

try:  # pragma: no cover - optional dependency for error detection
    import requests
except Exception:  # pragma: no cover - requests may be absent in some envs
    requests = None  # type: ignore


def _path_exists(path: Path) -> bool:
    try:
        return path.expanduser().exists()
    except OSError:  # pragma: no cover - defensive guard
        return False


def _is_connectivity_error(exc: Exception) -> bool:
    if isinstance(exc, OfflineModeIsEnabled):  # pragma: no cover - depends on hub
        return True
    if requests is not None and isinstance(exc, requests.exceptions.RequestException):
        return True
    text = str(exc).lower()
    return any(keyword in text for keyword in {"offline", "connection", "network"})


class TemporalEncoder(nn.Module):
    """Thin wrapper around :class:`torch.nn.TransformerEncoder`."""

    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        nlayers: int = 6,
        *,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=nlayers)

    def forward(
        self,
        sequence: Tensor,
        src_key_padding_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Encode the temporal sequence."""

        return self.encoder(sequence, src_key_padding_mask=src_key_padding_mask)


class TextSeq2SeqDecoder(nn.Module):
    """Seq2seq decoder backed by :mod:`transformers` models."""

    def __init__(
        self,
        *,
        d_model: int = 512,
        vocab_size: int = 32_000,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        pad_token_id: int = 0,
        eos_token_id: int = 1,
        pretrained_model_name_or_path: Optional[str] = None,
        config: Optional[PretrainedConfig] = None,
        config_kwargs: Optional[Dict[str, Any]] = None,
        auto_model_cls: Type[AutoModelForSeq2SeqLM] = AutoModelForSeq2SeqLM,
        tie_embeddings: bool = True,
        half_precision: bool = False,
        prompt_length: int = 0,
        prompt_init: str = "normal",
        prompt_init_std: float = 0.02,
        prompt_init_range: float = 0.5,
        prompt_init_tokens: Optional[Sequence[int]] = None,
        local_files_only: bool = False,
        local_paths: Optional[Sequence[str]] = None,
        env_var_paths: Optional[Sequence[str]] = None,
        hf_hub_download_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                "d_model must be divisible by num_heads for the decoder configuration."
            )

        config_kwargs = dict(config_kwargs or {})

        self.model = self._build_model(
            d_model=d_model,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            pretrained_model_name_or_path=pretrained_model_name_or_path,
            config=config,
            config_kwargs=config_kwargs,
            auto_model_cls=auto_model_cls,
            local_files_only=local_files_only,
            local_paths=local_paths,
            env_var_paths=env_var_paths,
            hf_hub_download_kwargs=hf_hub_download_kwargs,
        )

        if tie_embeddings:
            self._tie_embeddings()

        self._prompt_length = int(prompt_length)
        self._prompt: Optional[nn.Parameter] = None
        self._prompt_init = (prompt_init or "normal").strip().lower()
        self._prompt_init_std = float(prompt_init_std)
        self._prompt_init_range = float(prompt_init_range)
        self._prompt_token_ids: Optional[Tuple[int, ...]]
        if prompt_init_tokens is not None:
            self._prompt_token_ids = tuple(int(token) for token in prompt_init_tokens)
        else:
            self._prompt_token_ids = None

        if self._prompt_length < 0:
            raise ValueError("prompt_length must be non-negative")

        if self._uses_decoder_prompt():
            hidden_size = self.hidden_size_from_config(self.model.config)
            if hidden_size is None:
                raise ValueError("Unable to infer decoder hidden size for prompt initialisation")
            self._prompt = nn.Parameter(torch.empty(self._prompt_length, hidden_size))
            self._initialise_prompt_embeddings()
        elif self._prompt_length > 0:
            logging.warning(
                "Decoder prompt requested but the underlying model (%s) does not "
                "correspond to a T5 architecture. The prompt will be ignored.",
                type(self.model).__name__,
            )

        if half_precision:
            self.model = self.model.to(dtype=torch.float16)
            if self._prompt is not None:
                self._prompt.data = self._prompt.data.to(dtype=torch.float16)

    @staticmethod
    def _build_model(
        *,
        d_model: int,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        pad_token_id: int,
        eos_token_id: int,
        pretrained_model_name_or_path: Optional[str],
        config: Optional[PretrainedConfig],
        config_kwargs: Dict[str, Any],
        auto_model_cls: Type[AutoModelForSeq2SeqLM],
        local_files_only: bool,
        local_paths: Optional[Sequence[str]],
        env_var_paths: Optional[Sequence[str]],
        hf_hub_download_kwargs: Optional[Dict[str, Any]],
    ) -> nn.Module:
        if config is not None:
            TextSeq2SeqDecoder._validate_hidden_size(config, d_model)
            return auto_model_cls.from_config(config)

        if pretrained_model_name_or_path is not None:
            return TextSeq2SeqDecoder._load_pretrained_model(
                pretrained_model_name_or_path,
                auto_model_cls=auto_model_cls,
                d_model=d_model,
                local_files_only=local_files_only,
                local_paths=local_paths,
                env_var_paths=env_var_paths,
                hf_hub_download_kwargs=hf_hub_download_kwargs,
            )

        model_type = config_kwargs.pop("model_type", "t5").lower()
        default_kwargs = TextSeq2SeqDecoder._build_default_config_kwargs(
            model_type=model_type,
            d_model=d_model,
            vocab_size=vocab_size,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            overrides=config_kwargs,
        )
        try:
            auto_config = AutoConfig.for_model(model_type, **default_kwargs)
        except ValueError as exc:  # pragma: no cover - error depends on transformers internals
            raise ValueError(
                f"Unsupported model_type '{model_type}' for AutoConfig"
            ) from exc
        TextSeq2SeqDecoder._validate_hidden_size(auto_config, d_model)
        if (
            issubclass(auto_model_cls, T5ForConditionalGeneration)
            and auto_config.model_type == "t5"
        ):
            return auto_model_cls(auto_config)
        return auto_model_cls.from_config(auto_config)

    @staticmethod
    def _load_pretrained_model(
        identifier: str,
        *,
        auto_model_cls: Type[AutoModelForSeq2SeqLM],
        d_model: int,
        local_files_only: bool,
        local_paths: Optional[Sequence[str]],
        env_var_paths: Optional[Sequence[str]],
        hf_hub_download_kwargs: Optional[Dict[str, Any]],
    ) -> nn.Module:
        candidates = TextSeq2SeqDecoder._collect_candidate_paths(
            identifier,
            extra_paths=local_paths,
            env_var_paths=env_var_paths,
        )

        if hf_hub_download_kwargs and hf_hub_download:
            download_kwargs = dict(hf_hub_download_kwargs)
            download_kwargs.setdefault("local_files_only", local_files_only)
            if "repo_id" in download_kwargs and "filename" in download_kwargs:
                try:
                    hf_hub_download(**download_kwargs)
                except Exception as exc:  # pragma: no cover - depends on hub
                    if _is_connectivity_error(exc):
                        logging.warning(
                            "hf_hub_download could not reach the Hub (%s)."
                            " Falling back to local files.",
                            exc,
                        )
                    else:
                        raise
            else:
                logging.warning(
                    "hf_hub_download_kwargs require 'repo_id' and 'filename'. Skipping."
                )

        for path in candidates:
            try:
                model = TextSeq2SeqDecoder._from_local_path(
                    path,
                    auto_model_cls=auto_model_cls,
                    local_files_only=True,
                )
                TextSeq2SeqDecoder._validate_hidden_size(model.config, d_model)
                return model
            except Exception:  # pragma: no cover - defensive guard
                continue

        try:
            model = auto_model_cls.from_pretrained(
                identifier,
                local_files_only=local_files_only,
            )
            TextSeq2SeqDecoder._validate_hidden_size(model.config, d_model)
            return model
        except Exception as exc:
            if not _is_connectivity_error(exc):
                raise

            for path in candidates:
                try:
                    model = TextSeq2SeqDecoder._from_local_path(
                        path,
                        auto_model_cls=auto_model_cls,
                        local_files_only=True,
                    )
                    TextSeq2SeqDecoder._validate_hidden_size(model.config, d_model)
                    return model
                except Exception:  # pragma: no cover - defensive guard
                    continue
            message = (
                "Decoder weights could not be downloaded due to connectivity issues. "
                "Provide a local checkpoint with --decoder-model or "
                "set SLT_DECODER_PATH."
            )
            raise OSError(message) from exc

        raise RuntimeError("Unable to load decoder model.")  # pragma: no cover

    @staticmethod
    def _collect_candidate_paths(
        identifier: str,
        *,
        extra_paths: Optional[Sequence[str]],
        env_var_paths: Optional[Sequence[str]],
    ) -> List[str]:
        candidates: List[str] = []
        potential = Path(identifier)
        if _path_exists(potential):
            candidates.append(str(potential))

        if extra_paths:
            for path in extra_paths:
                if path:
                    candidates.append(path)

        env_sources = env_var_paths or ("SLT_DECODER_PATH", "SLT_DECODER_DIR")
        for var in env_sources:
            if not var:
                continue
            value = os.environ.get(var)
            if not value:
                continue
            segments = [segment for segment in value.split(os.pathsep) if segment]
            candidates.extend(segments or [value])

        seen: set[str] = set()
        resolved: List[str] = []
        for raw in candidates:
            expanded = str(Path(raw).expanduser())
            if expanded in seen:
                continue
            seen.add(expanded)
            if _path_exists(Path(expanded)):
                resolved.append(expanded)
        return resolved

    @staticmethod
    def _from_local_path(
        path: str,
        *,
        auto_model_cls: Type[AutoModelForSeq2SeqLM],
        local_files_only: bool,
    ) -> nn.Module:
        resolved = Path(path)
        if resolved.is_file():
            state_dict = torch.load(resolved, map_location="cpu")
            if not isinstance(state_dict, dict):
                raise TypeError(
                    f"Local decoder weights at {path} do not contain a state_dict."
                )
            parent = resolved.parent
            return auto_model_cls.from_pretrained(
                str(parent),
                local_files_only=local_files_only,
                state_dict=state_dict,
            )
        return auto_model_cls.from_pretrained(
            str(resolved),
            local_files_only=local_files_only,
        )

    @staticmethod
    def _build_default_config_kwargs(
        *,
        model_type: str,
        d_model: int,
        vocab_size: int,
        num_layers: int,
        num_heads: int,
        dropout: float,
        pad_token_id: int,
        eos_token_id: int,
        overrides: Dict[str, Any],
    ) -> Dict[str, Any]:
        overrides = dict(overrides)
        if model_type in {"bart", "mbart"}:
            bos_token_id = overrides.pop("bos_token_id", eos_token_id)
            base: Dict[str, Any] = {
                "vocab_size": vocab_size,
                "d_model": d_model,
                "encoder_layers": overrides.pop("encoder_layers", num_layers),
                "decoder_layers": overrides.pop("decoder_layers", num_layers),
                "encoder_attention_heads": overrides.pop("encoder_attention_heads", num_heads),
                "decoder_attention_heads": overrides.pop("decoder_attention_heads", num_heads),
                "dropout": overrides.pop("dropout", dropout),
                "attention_dropout": overrides.pop("attention_dropout", dropout),
                "activation_dropout": overrides.pop("activation_dropout", 0.0),
                "pad_token_id": pad_token_id,
                "bos_token_id": bos_token_id,
                "eos_token_id": eos_token_id,
                "decoder_start_token_id": overrides.pop(
                    "decoder_start_token_id", bos_token_id
                ),
            }
        else:
            base = {
                "vocab_size": vocab_size,
                "d_model": d_model,
                "d_kv": overrides.pop("d_kv", d_model // num_heads),
                "d_ff": overrides.pop("d_ff", d_model * 4),
                "num_layers": overrides.pop("num_layers", num_layers),
                "num_decoder_layers": overrides.pop(
                    "num_decoder_layers", num_layers
                ),
                "num_heads": overrides.pop("num_heads", num_heads),
                "dropout_rate": overrides.pop("dropout_rate", dropout),
                "layer_norm_epsilon": overrides.pop("layer_norm_epsilon", 1e-6),
                "initializer_factor": overrides.pop("initializer_factor", 1.0),
                "pad_token_id": pad_token_id,
                "eos_token_id": eos_token_id,
                "decoder_start_token_id": overrides.pop(
                    "decoder_start_token_id", pad_token_id
                ),
            }
        base.update(overrides)
        return base

    @staticmethod
    def _validate_hidden_size(config: PretrainedConfig, expected: int) -> None:
        hidden_size = TextSeq2SeqDecoder.hidden_size_from_config(config)
        if hidden_size is not None and hidden_size != expected:
            raise ValueError(
                "Loaded model hidden size does not match encoder dimensionality: "
                f"expected {expected}, got {hidden_size}."
            )

    @staticmethod
    def hidden_size_from_config(config: PretrainedConfig) -> Optional[int]:
        for attr in ("d_model", "hidden_size", "model_dim"):
            value = getattr(config, attr, None)
            if isinstance(value, int):
                return value
        return None

    def _tie_embeddings(self) -> None:
        tie_fn = getattr(self.model, "tie_weights", None)
        if callable(tie_fn):  # pragma: no cover - depends on model implementation
            tie_fn()
            return

        input_embeddings = getattr(self.model, "get_input_embeddings", None)
        output_embeddings = getattr(self.model, "get_output_embeddings", None)
        if callable(input_embeddings) and callable(output_embeddings):
            in_emb = input_embeddings()
            out_emb = output_embeddings()
            if in_emb is not None and out_emb is not None:
                out_emb.weight = in_emb.weight

    def _uses_decoder_prompt(self) -> bool:
        return (
            self._prompt_length > 0
            and isinstance(self.model, T5ForConditionalGeneration)
        )

    def _initialise_prompt_embeddings(self) -> None:
        if self._prompt is None:
            return
        init_mode = self._prompt_init
        if init_mode in {"none", "default"}:
            init_mode = "normal"

        weight = self.model.get_input_embeddings().weight
        if init_mode == "zero":
            nn.init.zeros_(self._prompt)
            return
        if init_mode == "normal":
            std = max(self._prompt_init_std, 1e-6)
            nn.init.normal_(self._prompt, mean=0.0, std=std)
            return
        if init_mode == "uniform":
            limit = max(self._prompt_init_range, 1e-6)
            nn.init.uniform_(self._prompt, -limit, limit)
            return
        if init_mode == "tokens":
            if self._prompt_token_ids is None or len(self._prompt_token_ids) == 0:
                raise ValueError(
                    "prompt_init_tokens must be provided when prompt_init='tokens'"
                )
            vectors = weight.new_zeros(self._prompt.shape)
            token_ids = torch.tensor(
                self._prompt_token_ids,
                dtype=torch.long,
                device=weight.device,
            )
            gathered = weight.index_select(0, token_ids)
            if gathered.size(0) >= self._prompt_length:
                vectors.copy_(gathered[: self._prompt_length])
            else:
                repeats = (self._prompt_length + gathered.size(0) - 1) // gathered.size(0)
                tiled = gathered.repeat(repeats, 1)
                vectors.copy_(tiled[: self._prompt_length])
            self._prompt.data.copy_(vectors.to(dtype=self._prompt.dtype))
            return
        if init_mode == "vocab":
            if weight.size(0) < self._prompt_length:
                expanded = weight.mean(dim=0, keepdim=True).repeat(self._prompt_length, 1)
            else:
                indices = torch.randperm(weight.size(0), device=weight.device)[: self._prompt_length]
                expanded = weight.index_select(0, indices)
            self._prompt.data.copy_(expanded.to(dtype=self._prompt.dtype))
            return
        raise ValueError(
            "Unsupported prompt_init value. Expected one of {'zero', 'normal', "
            "'uniform', 'tokens', 'vocab'}"
        )

    def _expand_prompt(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> Tensor:
        if self._prompt is None:
            raise RuntimeError("Prompt embeddings requested but not initialised")
        prompt = self._prompt.to(device=device, dtype=dtype)
        return prompt.unsqueeze(0).expand(batch_size, -1, -1)

    def _apply_decoder_prompt(
        self,
        hidden_states: Tensor,
        attention_mask: Optional[Tensor],
    ) -> tuple[Tensor, Optional[Tensor]]:
        if not self._uses_decoder_prompt():
            return hidden_states, attention_mask
        batch = hidden_states.size(0)
        prompt = self._expand_prompt(batch, hidden_states.device, hidden_states.dtype)
        augmented_hidden = torch.cat((prompt, hidden_states), dim=1)
        if attention_mask is not None:
            prompt_mask = torch.ones(
                batch,
                self._prompt_length,
                dtype=attention_mask.dtype,
                device=attention_mask.device,
            )
            augmented_mask = torch.cat((prompt_mask, attention_mask), dim=1)
        else:
            augmented_mask = torch.ones(
                batch,
                self._prompt_length + hidden_states.size(1),
                dtype=torch.long,
                device=hidden_states.device,
            )
        return augmented_hidden, augmented_mask

    def prepare_decoder_input_ids(self, labels: Tensor) -> Tensor:
        shift_right = getattr(self.model, "_shift_right", None)
        if callable(shift_right):
            return shift_right(labels)
        pad_token_id = getattr(self.config, "pad_token_id", None)
        decoder_start_token_id = getattr(self.config, "decoder_start_token_id", pad_token_id)
        if decoder_start_token_id is None:
            decoder_start_token_id = pad_token_id if pad_token_id is not None else 0
        input_ids = labels.new_full(labels.shape, int(decoder_start_token_id))
        if labels.size(1) > 1:
            input_ids[:, 1:] = labels[:, :-1].clone()
        input_ids.masked_fill_(input_ids == -100, int(decoder_start_token_id))
        return input_ids

    @property
    def config(self) -> PretrainedConfig:  # pragma: no cover - simple property
        return self.model.config

    def forward(
        self,
        encoder_hidden_states: Tensor,
        *,
        encoder_attention_mask: Optional[Tensor] = None,
        labels: Optional[Tensor] = None,
        decoder_attention_mask: Optional[Tensor] = None,
        **model_kwargs: Any,
    ) -> Seq2SeqLMOutput:
        """Forward pass delegating to the underlying HF model."""

        attention_mask = self._prepare_attention_mask(encoder_attention_mask)
        encoder_hidden_states, attention_mask = self._apply_decoder_prompt(
            encoder_hidden_states, attention_mask
        )
        return self.model(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask,
            **model_kwargs,
        )

    def generate(
        self,
        encoder_hidden_states: Tensor,
        *,
        encoder_attention_mask: Optional[Tensor] = None,
        **generation_kwargs: Any,
    ) -> torch.LongTensor:
        """Generate sequences conditioned on the encoder hidden states."""

        attention_mask = self._prepare_attention_mask(encoder_attention_mask)
        encoder_hidden_states, attention_mask = self._apply_decoder_prompt(
            encoder_hidden_states, attention_mask
        )
        return self.model.generate(
            encoder_outputs=BaseModelOutput(last_hidden_state=encoder_hidden_states),
            attention_mask=attention_mask,
            **generation_kwargs,
        )

    @staticmethod
    def _prepare_attention_mask(mask: Optional[Tensor]) -> Optional[Tensor]:
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            return mask.to(dtype=torch.long)
        return mask
