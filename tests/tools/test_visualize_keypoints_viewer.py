"""Tests rápidos para el visor de keypoints."""

from __future__ import annotations

import importlib
import sys
import types
from typing import Dict


class _Cv2Stub(types.ModuleType):
    """Stub sencillo para evitar depender de la librería ``cv2``."""

    def __getattr__(self, name: str):  # type: ignore[override]
        def _missing(*_args, **_kwargs):
            raise RuntimeError(f"La función cv2.{name} no está disponible en pruebas.")

        return _missing


def _stub_module(name: str, module: types.ModuleType, registry: Dict[str, types.ModuleType]) -> None:
    """Registrar ``module`` en ``sys.modules`` preservando su estado previo."""

    if name in sys.modules:
        registry[name] = sys.modules[name]
    sys.modules[name] = module


def test_load_font_returns_dejavu_sans() -> None:
    """Verificar que la fuente empaquetada está disponible."""

    previous_modules: Dict[str, types.ModuleType] = {}
    created_modules: set[str] = set()

    def register(name: str, module: types.ModuleType) -> None:
        if name not in sys.modules:
            created_modules.add(name)
        _stub_module(name, module, previous_modules)

    try:
        register("cv2", _Cv2Stub("cv2"))

        slt_stub = types.ModuleType("slt")
        register("slt", slt_stub)

        slt_data_stub = types.ModuleType("slt.data")
        register("slt.data", slt_data_stub)
        slt_stub.data = slt_data_stub

        slt_data_lsa_stub = types.ModuleType("slt.data.lsa_t_multistream")

        def _empty_dict(*_args, **_kwargs) -> Dict[str, str]:
            return {}

        slt_data_lsa_stub._resolve_mediapipe_connections = _empty_dict  # type: ignore[attr-defined]
        slt_data_lsa_stub._resolve_mediapipe_layout = _empty_dict  # type: ignore[attr-defined]
        register("slt.data.lsa_t_multistream", slt_data_lsa_stub)
        slt_data_stub.lsa_t_multistream = slt_data_lsa_stub

        slt_utils_stub = types.ModuleType("slt.utils")
        register("slt.utils", slt_utils_stub)
        metadata_stub = types.ModuleType("slt.utils.metadata")

        class _SplitSegment:  # noqa: D401 - stub mínimo
            """Stub para ``SplitSegment``."""

        metadata_stub.SplitSegment = _SplitSegment  # type: ignore[attr-defined]

        def _empty_list(*_args, **_kwargs) -> list[str]:
            return []

        metadata_stub.parse_split_column = _empty_list  # type: ignore[attr-defined]

        def _identity(value):
            return value

        metadata_stub.sanitize_time_value = _identity  # type: ignore[attr-defined]
        register("slt.utils.metadata", metadata_stub)
        slt_utils_stub.metadata = metadata_stub

        module = importlib.import_module("tools.visualize_keypoints_viewer")
        font = module._load_font(16)
    finally:
        for name, module in previous_modules.items():
            sys.modules[name] = module
        for name in created_modules - previous_modules.keys():
            sys.modules.pop(name, None)

    family, _style = font.getname()
    assert family == "DejaVu Sans"
