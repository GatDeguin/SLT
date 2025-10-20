import pytest

torch = pytest.importorskip("torch")

from slt.models.backbones import ViTConfig, load_backbone


class DummyBackbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_dim = 384
        self.patch_embed = torch.nn.Conv2d(3, 384, kernel_size=1)
        block = torch.nn.ModuleDict(
            {
                "attn": torch.nn.Linear(384, 384),
                "mlp": torch.nn.Linear(384, 384),
            }
        )
        self.blocks = torch.nn.ModuleList([block])
        self.head = torch.nn.Linear(384, 10)
        self.norm = torch.nn.LayerNorm(384)


def test_load_backbone_from_local_spec(tmp_path) -> None:
    reference = DummyBackbone()
    checkpoint = tmp_path / "vit.pt"
    torch.save(
        {
            "model": reference.state_dict(),
            "metadata": {"backbone": {}},
        },
        checkpoint,
    )

    def fake_instantiate(model, *, map_location=None, trust_repo=True):
        assert model == "dinov2_vits14"
        return DummyBackbone()

    import slt.models.backbones as backbones

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(backbones, "_instantiate_dinov2_architecture", fake_instantiate)
    try:
        loaded = load_backbone(
            {
                "type": "file",
                "path": str(checkpoint),
                "model": "dinov2_vits14",
            },
            map_location="cpu",
        )
    finally:
        monkeypatch.undo()

    assert isinstance(loaded, DummyBackbone)
    for key, value in reference.state_dict().items():
        assert torch.allclose(loaded.state_dict()[key], value)
    expected = tuple(ViTConfig().mean)
    assert tuple(round(v, 6) for v in loaded.image_normalization["mean"]) == expected


def test_load_backbone_freeze_mapping() -> None:
    backbone = load_backbone(
        DummyBackbone(),
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
    reference = DummyBackbone()
    checkpoint = tmp_path / "hf.bin"
    torch.save(
        {
            "model": reference.state_dict(),
            "metadata": {"backbone": {}},
        },
        checkpoint,
    )

    calls = {}

    def fake_download(repo_id, filename):
        calls["repo"] = repo_id
        calls["filename"] = filename
        return checkpoint

    monkeypatch.setattr("slt.models.backbones.hf_hub_download", fake_download)
    monkeypatch.setattr(
        "slt.models.backbones._instantiate_dinov2_architecture",
        lambda *args, **kwargs: DummyBackbone(),
    )

    loaded = load_backbone(
        {
            "type": "hf",
            "repo_id": "dummy/repo",
            "filename": "hf.bin",
            "model": "dinov2_vits14",
            "freeze": {"head": True},
        },
        map_location="cpu",
    )

    assert isinstance(loaded, DummyBackbone)
    assert calls == {"repo": "dummy/repo", "filename": "hf.bin"}
    if hasattr(loaded, "head"):
        assert all(not p.requires_grad for p in loaded.head.parameters())
