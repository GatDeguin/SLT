"""Run synthetic dataset checks mirroring ``docs/data_contract.md`` guidance."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from slt.data.lsa_t_multistream import LsaTMultiStream
from tests._synthetic import SyntheticDatasetSpec, generate_multistream_dataset

EXPECTED_SPEC = SyntheticDatasetSpec(
    sequence_length=6,
    image_size=20,
    pose_landmarks=8,
    frames_per_video=7,
    num_train=2,
    num_val=1,
    base_text="guia contrato",
)


def _validate_sample(sample, spec: SyntheticDatasetSpec) -> None:
    effective_length = sample.quality.get("effective_length")
    target_length = min(spec.sequence_length, spec.frames_per_video)
    if effective_length != target_length:
        raise RuntimeError(
            f"Longitud efectiva inesperada: {effective_length}, se esperaba {target_length}."
        )

    fps_info = sample.quality.get("fps", {})
    if int(fps_info.get("expected", 0)) != int(spec.fps) or not fps_info.get("ok", False):
        raise RuntimeError("Chequeo de FPS no coincide con la metadata esperada del contrato.")

    missing = sample.quality.get("missing_frames", {})
    if any(values for values in missing.values()):
        raise RuntimeError(f"Se detectaron frames faltantes: {missing}")

    if sample.pad_mask.sum().item() != sample.length.item():
        raise RuntimeError("pad_mask y length no coinciden, revisar normalización del dataset.")



def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        paths = generate_multistream_dataset(root, EXPECTED_SPEC)
        dataset = LsaTMultiStream(
            face_dir=str(paths.face_dir),
            hand_l_dir=str(paths.hand_left_dir),
            hand_r_dir=str(paths.hand_right_dir),
            pose_dir=str(paths.pose_dir),
            csv_path=str(paths.metadata_csv),
            index_csv=str(paths.train_index),
            T=EXPECTED_SPEC.sequence_length,
            img_size=EXPECTED_SPEC.image_size,
            lkp_count=EXPECTED_SPEC.pose_landmarks,
            flip_prob=0.0,
            enable_flip=False,
            quality_checks=True,
            quality_strict=True,
        )
        sample = dataset[0]
        _validate_sample(sample, EXPECTED_SPEC)
        print("Dataset sintético cumple con el contrato de datos.")
    return 0


if __name__ == "__main__":  # pragma: no cover - entry point
    sys.exit(main())
