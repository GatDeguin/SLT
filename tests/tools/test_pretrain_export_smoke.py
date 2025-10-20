import json
from dataclasses import asdict
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from slt.models import load_dinov2_backbone
from tools.pretrain_utils import BackboneConfig, build_vit_backbone, export_for_dinov2


def test_exported_backbone_loads_with_stub(tmp_path: Path) -> None:
    config = BackboneConfig(image_size=32, patch_size=8, embed_dim=64, depth=4, num_heads=4, mlp_ratio=2.0)
    backbone = build_vit_backbone(config)
    export_path = tmp_path / "backbone.pt"

    metadata = {"backbone": asdict(config), "note": "test"}
    export_for_dinov2(export_path, backbone=backbone, metadata=metadata)

    reloaded = load_dinov2_backbone(f"file::{export_path}:slt_vitsmall_patch16", map_location="cpu")

    assert set(backbone.state_dict().keys()) == set(reloaded.state_dict().keys())
    for key, value in backbone.state_dict().items():
        torch.testing.assert_close(reloaded.state_dict()[key], value)

    meta_file = export_path.with_suffix(".json")
    assert meta_file.exists()
    exported_meta = json.loads(meta_file.read_text(encoding="utf8"))
    assert exported_meta["backbone"]["image_size"] == config.image_size
