import logging
import sys
from pathlib import Path
from types import ModuleType

import pytest


if "extract_rois_v2" not in sys.modules:
    dummy = ModuleType("extract_rois_v2")
    dummy._append_metadata = lambda *args, **kwargs: None
    dummy._metadata_path = lambda *args, **kwargs: Path("metadata.jsonl")
    dummy._read_metadata_index = lambda *args, **kwargs: {}
    dummy.ensure_dir = lambda *args, **kwargs: None
    dummy.process_video = lambda *args, **kwargs: None
    sys.modules["extract_rois_v2"] = dummy

from tools.prepare_lsat_crops import _load_meta


def test_load_meta_sanitizes_and_skips_invalid(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    csv_path = tmp_path / "meta.csv"
    csv_path.write_text(
        "id;video;start;end\n"
        "clip_ok;video1;0;6.359.999.999.999.990\n"
        "clip_skip;video1;1,0;\n",
        encoding="utf-8",
    )

    with caplog.at_level(logging.WARNING):
        meta = _load_meta(csv_path)

    assert set(meta) == {"video1"}
    assert meta["video1"]["clip_count"] == 1
    assert meta["video1"]["total_span"] == pytest.approx(6.35999999999999)
    assert any("clip_skip" in record.message for record in caplog.records)
