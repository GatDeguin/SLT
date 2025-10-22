import torch
from torch import nn

from slt.models.mska import (
    FusedCTCHead,
    KeypointStreamEncoder,
    MSKAEncoder,
    MultiStreamKeypointAttention,
    StreamCTCHead,
)


def test_keypoint_stream_encoder_global_attention_weights() -> None:
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
    max_score = tanh_scores.max()
    exp_scores = torch.exp(tanh_scores - max_score)
    expected_weights = exp_scores / exp_scores.sum()

    actual_weights = weights[0, 0, 0]
    assert torch.allclose(actual_weights, expected_weights, atol=1e-5)

    global_sum = weights.sum(dim=(-1, -2, -3, -4))
    assert torch.allclose(global_sum, torch.ones_like(global_sum), atol=1e-6)

    row_softmax = torch.softmax(tanh_scores, dim=-1)
    assert not torch.allclose(row_softmax[0], expected_weights, atol=1e-5)


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

    attn_weights = encoder._last_attention_weights
    assert attn_weights is not None
    attn_sum = attn_weights.sum(dim=(-1, -2, -3))
    valid_frames = joint_mask.any(dim=2)
    assert torch.allclose(attn_sum[valid_frames], torch.ones_like(attn_sum[valid_frames]), atol=1e-5)
    assert torch.allclose(attn_sum[~valid_frames], torch.zeros_like(attn_sum[~valid_frames]))
    key_mask = joint_mask.view(2 * 3, 4)
    attn_weights_flat = attn_weights.view(2 * 3, encoder.self_attention.num_heads, 4, 4)
    padded_cols = ~key_mask.unsqueeze(1).unsqueeze(-2)
    masked_values = attn_weights_flat.masked_select(padded_cols)
    assert torch.allclose(masked_values, torch.zeros_like(masked_values))
    padded_rows = ~key_mask.unsqueeze(1).unsqueeze(-1)
    masked_rows = attn_weights_flat.masked_select(padded_rows)
    assert torch.allclose(masked_rows, torch.zeros_like(masked_rows))

    loss = hidden.sum() + output.frame_embeddings.sum()
    loss.backward()

    grads = points.grad
    assert grads is not None
    assert grads[joint_mask].abs().sum() > 0


def test_keypoint_stream_encoder_sgr_parameter_and_gradients() -> None:
    torch.manual_seed(0)
    encoder = KeypointStreamEncoder(
        in_dim=3,
        embed_dim=4,
        num_heads=1,
        temporal_blocks=0,
        temporal_kernel=3,
        dropout=0.0,
        use_global_attention=True,
        global_attention_activation="identity",
        global_attention_mix=0.5,
    )

    points = torch.randn(2, 1, 3, 3)
    output = encoder(points)

    store = encoder.self_attention._global_store()
    assert store is not None
    matrix = store.matrix
    assert matrix is not None
    assert matrix.shape == (3, 3)

    loss = output.joint_embeddings.sum()
    loss.backward()

    grad = matrix.grad
    assert grad is not None
    assert grad.abs().sum() > 0



def test_keypoint_stream_encoder_sgr_modifies_output() -> None:
    torch.manual_seed(0)
    base = KeypointStreamEncoder(
        in_dim=3,
        embed_dim=4,
        num_heads=1,
        temporal_blocks=0,
        temporal_kernel=3,
        dropout=0.0,
    )
    sgr = KeypointStreamEncoder(
        in_dim=3,
        embed_dim=4,
        num_heads=1,
        temporal_blocks=0,
        temporal_kernel=3,
        dropout=0.0,
        use_global_attention=True,
        global_attention_activation="identity",
        global_attention_mix=1.0,
    )

    with torch.no_grad():
        base.input_projection.weight.copy_(torch.eye(4, 3))
        sgr.input_projection.weight.copy_(torch.eye(4, 3))
        for layer in (
            base.self_attention.q_proj,
            base.self_attention.k_proj,
            base.self_attention.v_proj,
            base.self_attention.out_proj,
        ):
            layer.weight.copy_(torch.eye(4))
        for layer in (
            sgr.self_attention.q_proj,
            sgr.self_attention.k_proj,
            sgr.self_attention.v_proj,
            sgr.self_attention.out_proj,
        ):
            layer.weight.copy_(torch.eye(4))

    points = torch.tensor([[[[0.2, -0.4, 0.6], [0.5, 0.1, -0.3], [-0.2, 0.7, 0.4]]]], dtype=torch.float32)
    _ = sgr(points)
    store = sgr.self_attention._global_store()
    assert store is not None and store.matrix is not None
    with torch.no_grad():
        store.matrix.fill_(1.0)

    base_output = base(points)
    sgr_output = sgr(points)

    assert not torch.allclose(base_output.joint_embeddings, sgr_output.joint_embeddings)
    assert not torch.allclose(base_output.frame_embeddings, sgr_output.frame_embeddings)



def test_mska_encoder_shared_sgr_parameter() -> None:
    torch.manual_seed(0)
    mska = MSKAEncoder(
        input_dim=3,
        embed_dim=4,
        stream_names=("pose", "hand"),
        num_heads=1,
        ff_multiplier=1,
        dropout=0.0,
        ctc_vocab_size=5,
        stream_attention_heads=1,
        stream_temporal_blocks=0,
        stream_temporal_kernel=3,
        stream_temporal_dilation=1,
        use_global_attention=True,
        global_attention_activation="identity",
        global_attention_mix=0.75,
        global_attention_shared=True,
    )

    store = mska._shared_global_store
    assert store is not None
    for encoder in mska.encoders.values():
        assert encoder.self_attention._global_store() is store

    mska.zero_grad()
    streams = {
        "pose": {"points": torch.randn(2, 1, 3, 3)},
        "hand": {"points": torch.randn(2, 1, 3, 3)},
    }
    output = mska(streams)
    loss = output.fused_embedding.sum()
    loss.backward()

    matrix = store.matrix
    assert matrix is not None
    assert matrix.grad is not None
    assert matrix.grad.abs().sum() > 0


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

    stream_logits, stream_temporal = stream_head.forward_with_intermediate(features)
    fused_logits, fused_temporal = fused_head.forward_with_intermediate(features)

    assert stream_logits.shape == (batch, time, vocab)
    assert fused_logits.shape == (batch, time, vocab)
    assert stream_temporal.shape == (batch, time, dim)
    assert fused_temporal.shape == (batch, time, dim)

    stream_convs = [module for module in stream_head.modules() if isinstance(module, nn.Conv1d)]
    fused_convs = [module for module in fused_head.modules() if isinstance(module, nn.Conv1d)]
    stream_convs2d = [module for module in stream_head.modules() if isinstance(module, nn.Conv2d)]
    assert stream_convs and stream_convs[0].kernel_size == (3,)
    assert fused_convs and fused_convs[0].kernel_size == (3,)
    assert stream_convs2d and stream_convs2d[0].kernel_size == (3, 3)

    stream_probs = torch.softmax(stream_logits, dim=-1)
    fused_probs = torch.softmax(fused_logits, dim=-1)
    stream_sums = stream_probs.sum(dim=-1)
    fused_sums = fused_probs.sum(dim=-1)
    assert torch.allclose(stream_sums, torch.ones_like(stream_sums), atol=1e-5)
    assert torch.allclose(fused_sums, torch.ones_like(fused_sums), atol=1e-5)

    joints = 4
    joint_features = torch.randn(batch, time, joints, dim)
    joint_mask = torch.ones(batch, time, joints, dtype=torch.bool)
    logits_4d, temporal_4d = stream_head.forward_with_intermediate(
        joint_features, mask=joint_mask
    )
    assert logits_4d.shape == (batch, time, vocab)
    assert temporal_4d.shape == (batch, time, dim)
    temporal_probs = torch.softmax(temporal_4d, dim=1)
    temporal_sums = temporal_probs.sum(dim=1)
    assert torch.allclose(temporal_sums, torch.ones_like(temporal_sums), atol=1e-5)


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
    assert "temporal_probs" in fused
    assert fused["temporal_probs"].shape == (batch, time, 10)
    fused_temporal_sums = fused["temporal_probs"].sum(dim=1)
    assert torch.allclose(fused_temporal_sums, torch.ones_like(fused_temporal_sums), atol=1e-5)

    temporal_features = auxiliary["temporal_features"]
    assert temporal_features["fused"].shape == (batch, time, 10)
    stream_temporal_feats = temporal_features["stream"]
    assert set(stream_temporal_feats.keys()) == {"face", "hand_left", "pose"}
    for features_tensor in stream_temporal_feats.values():
        assert features_tensor.shape == (batch, time, 10)

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

    temporal_probs = probabilities["temporal"]
    assert torch.allclose(temporal_probs["fused"], fused["temporal_probs"], atol=1e-6)
    for probs in temporal_probs["stream"].values():
        assert probs.shape == (batch, time, 10)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
    for probs in temporal_probs["distillation"].values():
        assert probs.shape == (batch, time, 10)
        sums = probs.sum(dim=1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)
        assert not probs.requires_grad

    loss = fused["logits"].sum()
    for logits in auxiliary["stream"].values():
        loss = loss + logits.sum()
    loss.backward()

    grads = [tensor.grad for tensor in parameters]
    assert all(grad is not None for grad in grads)
    assert all(grad.abs().sum() > 0 for grad in grads)
