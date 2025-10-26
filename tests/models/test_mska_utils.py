from pathlib import Path

import torch

from slt.models.mska import MSKAEncoder
from slt.models.utils import load_mska_encoder_state


def _constant_state(module: torch.nn.Module, value: float) -> dict[str, torch.Tensor]:
    state = {}
    for name, tensor in module.state_dict().items():
        filled = torch.full_like(tensor, value)
        state[name] = filled
    return state


def test_load_mska_encoder_state(tmp_path: Path) -> None:
    encoder = MSKAEncoder(
        input_dim=3,
        embed_dim=8,
        stream_names=("face", "hand_left", "hand_right"),
        num_heads=2,
        ff_multiplier=2,
        dropout=0.0,
        ctc_vocab_size=5,
    )

    face_state = _constant_state(encoder.encoders["face"], 0.5)
    hand_state = _constant_state(encoder.encoders["hand_left"], -0.25)

    for module in encoder.encoders.values():
        for param in module.parameters():
            torch.nn.init.zeros_(param)

    face_file = tmp_path / "face.pt"
    hand_file = tmp_path / "hand.pt"
    torch.save({"stream": "face", "state_dict": face_state}, face_file)
    torch.save({"stream": "hand", "state_dict": hand_state}, hand_file)

    result = load_mska_encoder_state(
        encoder,
        face_checkpoint=face_file,
        hand_checkpoint=hand_file,
    )

    assert set(result) == {"face", "hand_left", "hand_right"}
    loaded_face = encoder.encoders["face"].state_dict()
    for name, tensor in face_state.items():
        assert torch.allclose(loaded_face[name], tensor)

    for stream in ("hand_left", "hand_right"):
        loaded = encoder.encoders[stream].state_dict()
        for name, tensor in hand_state.items():
            assert torch.allclose(loaded[name], tensor)
