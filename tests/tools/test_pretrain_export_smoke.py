import json
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slt.models import load_dinov2_backbone
from tools.pretrain_utils import BackboneConfig, build_vit_backbone, export_for_dinov2


def test_exported_backbone_loads_with_stub(tmp_path: Path) -> None:
    class DummyBackbone(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed_dim = 384
            self.linear = torch.nn.Linear(10, self.embed_dim)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.linear(x)

    config = BackboneConfig()

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "tools.pretrain_utils.load_dinov2_backbone", lambda *args, **kwargs: DummyBackbone()
    )
    backbone = build_vit_backbone(config)
    monkeypatch.undo()
    export_path = tmp_path / "backbone.pt"

    metadata = {"backbone": asdict(config), "note": "test"}
    export_for_dinov2(export_path, backbone=backbone, metadata=metadata)

    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        "slt.models.backbones._instantiate_dinov2_architecture",
        lambda *args, **kwargs: DummyBackbone(),
    )
    reloaded = load_dinov2_backbone(f"file::{export_path}:dinov2_vits14", map_location="cpu")
    monkeypatch.undo()

    assert set(backbone.state_dict().keys()) == set(reloaded.state_dict().keys())
    for key, value in backbone.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], value)

    meta_file = export_path.with_suffix(".json")
    assert meta_file.exists()
    exported_meta = json.loads(meta_file.read_text(encoding="utf8"))
    assert exported_meta["backbone"]["image_size"] == config.image_size
