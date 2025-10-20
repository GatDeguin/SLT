from __future__ import annotations

import pytest

transformers = pytest.importorskip("transformers")
T5Config = transformers.T5Config

torch = pytest.importorskip("torch")

from tools import train_slt_multistream_v9 as train_module


class TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1
    vocab_size = 8

    def __len__(self) -> int:  # pragma: no cover - compatibility shim
        return self.vocab_size


@pytest.mark.parametrize("batch_size,target_length", [(1, 3)])
def test_multistream_classifier_forward_with_custom_decoder_config(tmp_path, batch_size, target_length):
    config_dir = tmp_path / "decoder"
    config_dir.mkdir()

    decoder_config = T5Config(
        vocab_size=TinyTokenizer.vocab_size,
        d_model=16,
        d_kv=4,
        d_ff=32,
        num_layers=1,
        num_decoder_layers=1,
        num_heads=4,
        dropout_rate=0.0,
        pad_token_id=TinyTokenizer.pad_token_id,
        eos_token_id=TinyTokenizer.eos_token_id,
        decoder_start_token_id=TinyTokenizer.pad_token_id,
    )
    decoder_config.save_pretrained(config_dir)

    model_config = train_module.ModelConfig(
        image_size=16,
        projector_dim=8,
        d_model=16,
        pose_landmarks=1,
        projector_dropout=0.0,
        fusion_dropout=0.0,
        temporal_nhead=4,
        temporal_layers=1,
        temporal_dim_feedforward=32,
        temporal_dropout=0.0,
        sequence_length=2,
        decoder_layers=1,
        decoder_heads=4,
        decoder_dropout=0.0,
        decoder_config=str(config_dir),
    )

    class DummyEncoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()

        def forward(self, face, hand_l, hand_r, pose, *, pad_mask=None, miss_mask_hl=None, miss_mask_hr=None):
            batch, time = face.shape[:2]
            return torch.zeros(batch, time, model_config.d_model, device=face.device)

    patch = pytest.MonkeyPatch()
    patch.setattr("slt.training.models.MultiStreamEncoder", lambda *args, **kwargs: DummyEncoder())
    tokenizer = TinyTokenizer()
    model = train_module.MultiStreamClassifier(model_config, tokenizer)
    patch.undo()

    sequence_length = model_config.sequence_length
    image_size = model_config.image_size

    face = torch.zeros(batch_size, sequence_length, 3, image_size, image_size)
    hand_l = torch.zeros(batch_size, sequence_length, 3, image_size, image_size)
    hand_r = torch.zeros(batch_size, sequence_length, 3, image_size, image_size)
    pose = torch.zeros(batch_size, sequence_length, 3 * model_config.pose_landmarks)
    pad_mask = torch.ones(batch_size, sequence_length, dtype=torch.bool)
    miss_mask = torch.zeros(batch_size, sequence_length, dtype=torch.bool)

    labels = torch.zeros(batch_size, target_length, dtype=torch.long)
    decoder_attention_mask = torch.ones(batch_size, target_length, dtype=torch.long)

    outputs = model(
        face=face,
        hand_l=hand_l,
        hand_r=hand_r,
        pose=pose,
        pad_mask=pad_mask,
        miss_mask_hl=miss_mask,
        miss_mask_hr=miss_mask,
        labels=labels,
        decoder_attention_mask=decoder_attention_mask,
    )

    assert outputs.logits.shape == (batch_size, target_length, TinyTokenizer.vocab_size)
    assert model.decoder.model.config.vocab_size == TinyTokenizer.vocab_size
