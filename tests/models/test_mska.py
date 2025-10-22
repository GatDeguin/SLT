import torch
from torch import nn

from slt.models.mska import (
    FusedCTCHead,
    KeypointStreamEncoder,
    MSKAEncoder,
    MultiStreamKeypointAttention,
    StreamCTCHead,
)


def test_keypoint_stream_encoder_tanh_attention_weights() -> None:
    encoder = KeypointStreamEncoder(
        in_dim=4,
        embed_dim=4,
        num_heads=1,
        temporal_blocks=0,
        temporal_kernel=3,
        dropout=0.0,
    )
    encoder.input_norm = nn.Identity()
    encoder.attention_norm = nn.Identity()
    with torch.no_grad():
        encoder.input_projection.weight.copy_(torch.eye(4))
        for linear in (
            encoder.self_attention.q_proj,
            encoder.self_attention.k_proj,
            encoder.self_attention.v_proj,
            encoder.self_attention.out_proj,
        ):
            linear.weight.copy_(torch.eye(4))

    keypoints = torch.tensor([[[[6.0, -2.0, 0.5, 1.0], [1.0, 0.0, 0.0, 0.0]]]])
    output = encoder(keypoints)

    assert output.joint_embeddings.shape == (1, 1, 2, 4)
    assert output.frame_embeddings.shape == (1, 1, 4)

    weights = encoder._last_attention_weights
    assert weights is not None

    positional = encoder._joint_positional_encoding(
        joints=2, device=keypoints.device, dtype=keypoints.dtype
    ).view(1, 1, 2, 4)
    projected = keypoints + positional
    flat = projected.view(1, 2, 4)
    raw_scores = torch.matmul(flat, flat.transpose(-2, -1)) * encoder.self_attention.scale
    tanh_scores = torch.tanh(raw_scores)
    expected_weights = torch.softmax(tanh_scores, dim=-1)

    actual_weights = weights[0, 0, 0]
    assert torch.allclose(actual_weights, expected_weights[0], atol=1e-5)

    softmax_without_tanh = torch.softmax(raw_scores, dim=-1)
    assert not torch.allclose(softmax_without_tanh, expected_weights, atol=1e-5)


def test_keypoint_stream_encoder_masks_and_gradients() -> None:
    torch.manual_seed(0)
    encoder = KeypointStreamEncoder(
        in_dim=3,
        embed_dim=6,
        num_heads=2,
        temporal_blocks=2,
        temporal_kernel=3,
        dropout=0.0,
    )

    points = torch.randn(2, 3, 4, 3, requires_grad=True)
    joint_mask = torch.tensor(
        [
            [
                [True, True, False, False],
                [True, False, True, False],
                [False, False, False, False],
            ],
            [
                [True, True, True, True],
                [True, False, True, False],
                [True, True, True, True],
            ],
        ],
        dtype=torch.bool,
    )
    frame_mask = torch.tensor(
        [[True, True, False], [True, True, True]],
        dtype=torch.bool,
    )

    output = encoder(points, mask=joint_mask, frame_mask=frame_mask)

    assert output.joint_embeddings.shape == (2, 3, 4, 6)
    assert output.frame_embeddings.shape == (2, 3, 6)
    assert torch.equal(output.joint_mask, joint_mask)

    expected_frame = torch.logical_and(frame_mask, joint_mask.any(dim=2))
    assert torch.equal(output.frame_mask, expected_frame)

    hidden = output.joint_embeddings
    assert torch.allclose(hidden[~joint_mask], torch.zeros_like(hidden[~joint_mask]))

    loss = hidden.sum() + output.frame_embeddings.sum()
    loss.backward()

    grads = points.grad
    assert grads is not None
    assert grads[joint_mask].abs().sum() > 0


def test_multi_stream_keypoint_attention_shapes() -> None:
    embed_dim = 8
    encoder = KeypointStreamEncoder(
        in_dim=3,
        embed_dim=embed_dim,
        num_heads=2,
        temporal_blocks=1,
        temporal_kernel=3,
        dropout=0.0,
    )
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


def test_ctc_heads_preserve_temporal_dimension() -> None:
    torch.manual_seed(0)
    batch, time, dim, vocab = 2, 5, 6, 9
    features = torch.randn(batch, time, dim)

    stream_head = StreamCTCHead(dim, vocab, dropout=0.0)
    fused_head = FusedCTCHead(dim, vocab, dropout=0.0)

    stream_logits = stream_head(features)
    fused_logits = fused_head(features)

    assert stream_logits.shape == (batch, time, vocab)
    assert fused_logits.shape == (batch, time, vocab)

    stream_convs = [module for module in stream_head.modules() if isinstance(module, nn.Conv1d)]
    fused_convs = [module for module in fused_head.modules() if isinstance(module, nn.Conv1d)]
    assert stream_convs and stream_convs[0].kernel_size == (3,)
    assert fused_convs and fused_convs[0].kernel_size == (3,)

    stream_probs = torch.softmax(stream_logits, dim=-1)
    fused_probs = torch.softmax(fused_logits, dim=-1)
    stream_sums = stream_probs.sum(dim=-1)
    fused_sums = fused_probs.sum(dim=-1)
    assert torch.allclose(stream_sums, torch.ones_like(stream_sums), atol=1e-5)
    assert torch.allclose(fused_sums, torch.ones_like(fused_sums), atol=1e-5)


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
        stream_attention_heads=2,
        stream_temporal_blocks=1,
        stream_temporal_kernel=3,
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
    assert "probs" in fused
    assert fused["probs"].shape == (batch, time, 7)

    probabilities = auxiliary["probabilities"]
    assert torch.allclose(fused["probs"], probabilities["fused"], atol=1e-6)

    stream_probs = probabilities["stream"]
    assert set(stream_probs.keys()) == {"face", "hand_left", "pose"}
    for probs in stream_probs.values():
        assert probs.shape == (batch, time, 7)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)

    teacher_probs = probabilities["distillation"]
    assert set(teacher_probs.keys()) == {"face", "hand_left", "pose"}
    for probs in teacher_probs.values():
        assert probs.shape == (batch, time, 7)
        sums = probs.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    assert not teacher_probs["face"].requires_grad

    loss = fused["logits"].sum()
    for logits in auxiliary["stream"].values():
        loss = loss + logits.sum()
    loss.backward()

    grads = [tensor.grad for tensor in parameters]
    assert all(grad is not None for grad in grads)
    assert all(grad.abs().sum() > 0 for grad in grads)
