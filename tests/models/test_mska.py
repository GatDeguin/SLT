import torch

from slt.models.mska import (
    KeypointStreamEncoder,
    MSKAEncoder,
    MultiStreamKeypointAttention,
)


def test_keypoint_stream_encoder_masks() -> None:
    encoder = KeypointStreamEncoder(3, 6, hidden_dim=4, dropout=0.0)
    points = torch.randn(1, 3, 4, 3)
    joint_mask = torch.tensor(
        [[[True, False, True, False], [False, False, False, False], [True, True, True, True]]]
    )
    frame_mask = torch.tensor([[True, False, True]])
    output = encoder(points, mask=joint_mask, frame_mask=frame_mask)

    assert output.joint_embeddings.shape == (1, 3, 4, 6)
    assert output.frame_embeddings.shape == (1, 3, 6)
    assert torch.equal(output.joint_mask, joint_mask)
    expected_frame = torch.tensor([[True, False, True]])
    assert torch.equal(output.frame_mask, expected_frame)


def test_multi_stream_keypoint_attention_shapes() -> None:
    embed_dim = 8
    encoder = KeypointStreamEncoder(3, embed_dim, hidden_dim=embed_dim, dropout=0.0)
    batch, time, joints = 2, 3, 5
    frame_mask_face = torch.tensor(
        [[True, True, False], [True, False, True]], dtype=torch.bool
    )
    frame_mask_hand = torch.tensor(
        [[True, True, True], [True, True, False]], dtype=torch.bool
    )
    joint_mask_face = frame_mask_face.unsqueeze(-1).expand(batch, time, joints)
    joint_mask_hand = frame_mask_hand.unsqueeze(-1).expand(batch, time, joints)

    face_output = encoder(
        torch.randn(batch, time, joints, 3),
        mask=joint_mask_face,
        frame_mask=frame_mask_face,
    )
    hand_output = encoder(
        torch.randn(batch, time, joints, 3),
        mask=joint_mask_hand,
        frame_mask=frame_mask_hand,
    )

    attention = MultiStreamKeypointAttention(
        embed_dim,
        num_heads=2,
        ff_multiplier=2,
        stream_order=("face", "hand_left"),
    )
    result = attention({"face": face_output, "hand_left": hand_output})

    assert set(result.stream_embeddings.keys()) == {"face", "hand_left"}
    for tensor in result.stream_embeddings.values():
        assert tensor.shape == (batch, time, embed_dim)
    assert result.fused_embedding.shape == (batch, time, embed_dim)
    assert result.attention is not None
    assert result.attention.shape == (batch, time, 2, 2, 2)
    expected_fused_mask = frame_mask_face & frame_mask_hand
    assert torch.equal(result.fused_mask, expected_fused_mask)


def test_mska_encoder_auxiliary_logits_and_gradients() -> None:
    torch.manual_seed(0)
    mska = MSKAEncoder(
        input_dim=3,
        embed_dim=10,
        stream_names=("face", "hand_left", "pose"),
        num_heads=2,
        ff_multiplier=2,
        dropout=0.0,
        ctc_vocab_size=7,
    )
    batch, time, joints = 2, 4, 6
    streams = {}
    parameters = []
    for name in ("face", "hand_left", "pose"):
        points = torch.randn(batch, time, joints, 3, requires_grad=True)
        mask = torch.ones(batch, time, joints, dtype=torch.bool)
        frame_mask = torch.ones(batch, time, dtype=torch.bool)
        streams[name] = {
            "points": points,
            "mask": mask,
            "frame_mask": frame_mask,
        }
        parameters.append(points)

    output = mska(streams)
    auxiliary = mska.auxiliary_logits(output)

    assert set(auxiliary["stream"].keys()) == {"face", "hand_left", "pose"}
    for logits in auxiliary["stream"].values():
        assert logits.shape == (batch, time, 7)
    fused = auxiliary["fused"]
    assert fused["logits"].shape == (batch, time, 7)
    assert fused["mask"].shape == (batch, time)

    loss = fused["logits"].sum()
    for logits in auxiliary["stream"].values():
        loss = loss + logits.sum()
    loss.backward()

    grads = [tensor.grad for tensor in parameters]
    assert all(grad is not None for grad in grads)
    assert all(grad.abs().sum() > 0 for grad in grads)
