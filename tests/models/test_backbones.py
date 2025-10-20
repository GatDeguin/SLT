import pytest

torch = pytest.importorskip("torch")

from slt.models.backbones import ViTConfig, ViTSmallPatch16, load_backbone


def _tiny_config() -> ViTConfig:
    return ViTConfig(
        image_size=16,
        patch_size=8,
        in_channels=3,
        embed_dim=32,
        depth=1,
        num_heads=4,
        mlp_ratio=2.0,
    )


def test_load_backbone_from_local_spec(tmp_path) -> None:
    config = _tiny_config()
    reference = ViTSmallPatch16(config)
    checkpoint = tmp_path / "vit.pt"
    torch.save(
        {
            "model": reference.state_dict(),
            "metadata": {"backbone": vars(config)},
        },
        checkpoint,
    )

    loaded = load_backbone(
        {
            "type": "file",
            "path": str(checkpoint),
            "model": "slt_vitsmall_patch16",
        },
        map_location="cpu",
    )

    assert isinstance(loaded, ViTSmallPatch16)
    for key, value in reference.state_dict().items():
        assert torch.allclose(loaded.state_dict()[key], value)


def test_load_backbone_freeze_mapping() -> None:
    config = _tiny_config()
    backbone = load_backbone(
        ViTSmallPatch16(config),
        freeze={
            "patch_embed": True,
            "head": True,
            "blocks": 1,
            "prefixes": ["norm"],
            "exclude": ["blocks.0.attn"],
        },
    )

    def requires_grad(prefix: str) -> bool:
        return any(
            param.requires_grad
            for name, param in backbone.named_parameters()
            if name.startswith(prefix)
        )

    assert not requires_grad("patch_embed")
    assert not requires_grad("blocks.0.mlp")
    assert not requires_grad("head") if hasattr(backbone, "head") else True
    assert requires_grad("blocks.0.attn")
    assert not requires_grad("norm")


def test_load_backbone_huggingface_spec(monkeypatch, tmp_path) -> None:
    config = _tiny_config()
    reference = ViTSmallPatch16(config)
    checkpoint = tmp_path / "hf.bin"
    torch.save(
        {
            "model": reference.state_dict(),
            "metadata": {"backbone": vars(config)},
        },
        checkpoint,
    )

    calls = {}

    def fake_download(repo_id, filename):
        calls["repo"] = repo_id
        calls["filename"] = filename
        return checkpoint

    monkeypatch.setattr("slt.models.backbones.hf_hub_download", fake_download)

    loaded = load_backbone(
        {
            "type": "hf",
            "repo_id": "dummy/repo",
            "filename": "hf.bin",
            "model": "slt_vitsmall_patch16",
            "freeze": {"head": True},
        },
        map_location="cpu",
    )

    assert isinstance(loaded, ViTSmallPatch16)
    assert calls == {"repo": "dummy/repo", "filename": "hf.bin"}
    if hasattr(loaded, "head"):
        assert all(not p.requires_grad for p in loaded.head.parameters())
