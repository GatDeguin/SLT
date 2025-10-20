"""Dataset multi-stream para LSA-T."""
from __future__ import annotations

import importlib
import os
import random
import warnings
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional

import torch
from torch.utils.data import Dataset


@lru_cache(maxsize=None)
def _lazy_import(name: str):
    """Importación perezosa con memoización."""
    return importlib.import_module(name)


def _get_pandas():  # pragma: no cover - función sencilla
    return _lazy_import("pandas")


def _get_numpy():  # pragma: no cover - función sencilla
    return _lazy_import("numpy")


def _get_pil_image():  # pragma: no cover - función sencilla
    return _lazy_import("PIL.Image")


@dataclass
class SampleItem:
    """Estructura con los tensores normalizados por clip."""

    face: torch.Tensor
    hand_l: torch.Tensor
    hand_r: torch.Tensor
    pose: torch.Tensor
    pose_conf_mask: torch.Tensor
    pad_mask: torch.Tensor
    length: torch.Tensor
    miss_mask_hl: torch.Tensor
    miss_mask_hr: torch.Tensor
    quality: Dict[str, Any]
    text: str
    video_id: str


class LsaTMultiStream(Dataset):
    """Dataset multi-stream para clips del corpus LSA-T.

    Espera la siguiente estructura de carpetas:

    - ``face/<video_id>_fXXXXXX.jpg``
    - ``hand_l/<video_id>_fXXXXXX.jpg``
    - ``hand_r/<video_id>_fXXXXXX.jpg``
    - ``pose/<video_id>.npz`` con clave ``pose``

    Además necesita un CSV con columnas ``video_id`` y ``texto`` (separadas
    por ``;``) y un CSV adicional con la lista de IDs pertenecientes al split
    a utilizar.
    """

    def __init__(
        self,
        face_dir: str,
        hand_l_dir: str,
        hand_r_dir: str,
        pose_dir: str,
        csv_path: str,
        index_csv: str,
        T: int = 128,
        img_size: int = 224,
        lkp_count: int = 13,
        min_conf: float = 0.25,
        flip_prob: float = 0.2,
        enable_flip: bool = True,
        quality_checks: bool = True,
        quality_strict: bool = False,
        fps_tolerance: float = 1.0,
    ) -> None:
        pd = _get_pandas()
        np = _get_numpy()

        self.face_dir = face_dir
        self.hand_l_dir = hand_l_dir
        self.hand_r_dir = hand_r_dir
        self.pose_dir = pose_dir
        self.img_size = img_size
        self.T = T
        self.lkp_count = lkp_count
        self.min_conf = min_conf
        self.flip_prob = flip_prob
        self.enable_flip = enable_flip
        self.quality_checks = quality_checks
        self.quality_strict = quality_strict
        self.fps_tolerance = fps_tolerance
        self._np = np

        df = pd.read_csv(csv_path, sep=";")
        df.columns = [c.strip().lower() for c in df.columns]
        if "video_id" not in df.columns or "texto" not in df.columns:
            raise ValueError("El CSV principal debe contener columnas 'video_id' y 'texto'.")

        idx = pd.read_csv(index_csv)
        idx.columns = [c.strip().lower() for c in idx.columns]
        if "video_id" not in idx.columns:
            raise ValueError("El CSV de índices debe contener la columna 'video_id'.")

        self.df = df.merge(idx[["video_id"]], on="video_id", how="inner")
        self.ids = self.df["video_id"].tolist()
        self.texts = dict(zip(df["video_id"], df["texto"]))

        def _coerce(value: Any) -> Optional[float]:
            if value is None:
                return None
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(str(value).strip())
            except (TypeError, ValueError):
                return None

        self.meta = {}
        for vid in self.ids:
            vid_meta: Dict[str, Any] = {}
            rows = self.df.loc[self.df["video_id"] == vid]
            if rows.empty:
                continue
            row = rows.iloc[0]
            if "fps" in row:
                vid_meta["fps"] = _coerce(row["fps"])
            if "duration" in row:
                vid_meta["duration"] = _coerce(row["duration"])
            if "frame_count" in row:
                vid_meta["frame_count"] = _coerce(row["frame_count"])
            self.meta[vid] = vid_meta

    def __len__(self) -> int:  # pragma: no cover - simple
        return len(self.ids)

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------
    def _read_image(self, path: str) -> torch.Tensor:
        """Lee y normaliza una imagen RGB en rango ``[0, 1]``."""

        Image = _get_pil_image()
        np = self._np

        with Image.open(path) as img:
            img = img.convert("RGB").resize((self.img_size, self.img_size))
            arr = np.asarray(img, dtype="float32") / 255.0
        arr = arr.transpose(2, 0, 1)
        return torch.from_numpy(arr)

    def _list_frames(self, base_dir: str, vid: str) -> List[str]:
        if not os.path.isdir(base_dir):
            return []
        prefix = f"{vid}_f"
        files = [
            os.path.join(base_dir, name)
            for name in sorted(os.listdir(base_dir))
            if name.startswith(prefix) and name.endswith(".jpg")
        ]
        return files

    def _sample_indices(self, T0: int) -> List[int]:
        """Devuelve índices equiespaciados para muestrear ``T0`` frames."""

        if self.T <= 0:
            return []
        if T0 <= 0:
            return [0] * self.T

        np = self._np
        last_index = max(T0 - 1, 0)

        if T0 == 1:
            return [0] * self.T

        # ``linspace`` garantiza alineación consistente sin depender de ``random``
        # para mantener reproducibilidad.
        positions = np.linspace(0.0, float(last_index), num=self.T)
        idxs = np.rint(positions).astype("int64").tolist()
        return [max(0, min(last_index, int(i))) for i in idxs]

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> SampleItem:
        vid = self.ids[index]
        text = str(self.texts.get(vid, "")).strip()

        face_frames = self._list_frames(self.face_dir, vid)
        hl_frames = self._list_frames(self.hand_l_dir, vid)
        hr_frames = self._list_frames(self.hand_r_dir, vid)

        stream_lengths = {
            "face": len(face_frames),
            "hand_l": len(hl_frames),
            "hand_r": len(hr_frames),
        }

        pose_path = os.path.join(self.pose_dir, f"{vid}.npz")
        pose = self._load_pose(pose_path)
        pose_length = pose.shape[0] if hasattr(pose, "shape") else 0
        stream_lengths["pose"] = pose_length

        T0 = max(stream_lengths.values()) if stream_lengths else 0
        idxs = self._sample_indices(T0)

        def safe_get(frames: List[str], j: int) -> Optional[str]:
            if not frames:
                return None
            return frames[min(j, len(frames) - 1)]

        face_list: List[torch.Tensor] = []
        hl_list: List[torch.Tensor] = []
        hr_list: List[torch.Tensor] = []
        miss_hl: List[int] = []
        miss_hr: List[int] = []

        zero_img = torch.zeros(3, self.img_size, self.img_size, dtype=torch.float32)

        for j in idxs:
            fp = safe_get(face_frames, j)
            lp = safe_get(hl_frames, j)
            rp = safe_get(hr_frames, j)

            face_list.append(self._read_image(fp) if fp else zero_img)

            if lp:
                hl_list.append(self._read_image(lp))
                miss_hl.append(1)
            else:
                hl_list.append(zero_img)
                miss_hl.append(0)

            if rp:
                hr_list.append(self._read_image(rp))
                miss_hr.append(1)
            else:
                hr_list.append(zero_img)
                miss_hr.append(0)

        face = torch.stack(face_list, dim=0)
        hand_l = torch.stack(hl_list, dim=0)
        hand_r = torch.stack(hr_list, dim=0)

        pose_t, pose_mask = self._sample_pose(pose)
        effective_length = self._effective_length(stream_lengths)

        pad_mask = torch.zeros(self.T, dtype=torch.bool)
        if effective_length > 0:
            pad_mask[:effective_length] = True
        length = torch.tensor(effective_length, dtype=torch.long)
        miss_mask_hl = torch.tensor(miss_hl, dtype=torch.bool)
        miss_mask_hr = torch.tensor(miss_hr, dtype=torch.bool)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

        face = (face - mean) / std
        hand_l = (hand_l - mean) / std
        hand_r = (hand_r - mean) / std

        if self.enable_flip and self.flip_prob > 0 and random.random() < self.flip_prob:
            face = torch.flip(face, dims=[3])
            new_hand_l = torch.flip(hand_r, dims=[3])
            new_hand_r = torch.flip(hand_l, dims=[3])
            hand_l, hand_r = new_hand_l, new_hand_r
            miss_mask_hl, miss_mask_hr = miss_mask_hr, miss_mask_hl
            pose_t = self._flip_pose_tensor(pose_t)
            pose_mask = self._flip_pose_mask(pose_mask)

        quality = self._build_quality_report(
            vid,
            face_frames=face_frames,
            hand_l_frames=hl_frames,
            hand_r_frames=hr_frames,
            pose_frames=pose,
            effective_length=effective_length,
        )

        return SampleItem(
            face=face,
            hand_l=hand_l,
            hand_r=hand_r,
            pose=pose_t,
            pose_conf_mask=pose_mask,
            pad_mask=pad_mask,
            length=length,
            miss_mask_hl=miss_mask_hl,
            miss_mask_hr=miss_mask_hr,
            quality=quality,
            text=text,
            video_id=vid,
        )

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------
    def _load_pose(self, pose_path: str) -> Any:
        np = self._np
        if not os.path.exists(pose_path):
            return np.zeros((1, self.lkp_count * 3), dtype="float32")

        try:
            with np.load(pose_path) as pose_npz:
                pose = pose_npz.get("pose")
                if pose is None:
                    return np.zeros((1, self.lkp_count * 3), dtype="float32")

                pose_arr = np.asarray(pose, dtype="float32")
                if pose_arr.ndim == 3:
                    frames, landmarks, dims = pose_arr.shape
                    if dims < 2:
                        return np.zeros((1, self.lkp_count * 3), dtype="float32")
                    if dims == 2:
                        conf = pose_npz.get("confidence") or pose_npz.get("pose_confidence")
                        if conf is None:
                            conf = np.ones((frames, landmarks), dtype="float32")
                        conf = np.asarray(conf, dtype="float32").reshape(frames, landmarks)
                        conf = np.clip(conf, 0.0, 1.0)
                        pose_arr = np.concatenate([pose_arr, conf[..., None]], axis=2)
                    elif dims > 3:
                        pose_arr = pose_arr[..., :3]
                elif pose_arr.ndim == 2:
                    frames = pose_arr.shape[0]
                    features = pose_arr.shape[1]
                    if features % 3 == 0:
                        landmarks = features // 3
                        pose_arr = pose_arr.reshape(frames, landmarks, 3)
                    elif features % 2 == 0:
                        landmarks = features // 2
                        coords = pose_arr.reshape(frames, landmarks, 2)
                        conf = pose_npz.get("confidence") or pose_npz.get("pose_confidence")
                        if conf is None:
                            conf = np.ones((frames, landmarks), dtype="float32")
                        conf = np.asarray(conf, dtype="float32").reshape(frames, landmarks)
                        conf = np.clip(conf, 0.0, 1.0)
                        pose_arr = np.concatenate([coords, conf[..., None]], axis=2)
                    else:
                        return np.zeros((1, self.lkp_count * 3), dtype="float32")
                else:
                    return np.zeros((1, self.lkp_count * 3), dtype="float32")

                frames = pose_arr.shape[0]
                landmarks = pose_arr.shape[1]
                out = np.zeros((frames, self.lkp_count, 3), dtype="float32")
                copy_landmarks = min(self.lkp_count, landmarks)
                out[:, :copy_landmarks, : pose_arr.shape[2]] = pose_arr[:, :copy_landmarks, : pose_arr.shape[2]]
                return out.reshape(frames, self.lkp_count * 3)
        except (OSError, ValueError):
            return np.zeros((1, self.lkp_count * 3), dtype="float32")

    def _sample_pose(self, pose: Any) -> tuple[torch.Tensor, torch.Tensor]:
        np = self._np
        pose_arr = np.asarray(pose, dtype="float32")
        T0p = pose_arr.shape[0] if pose_arr.size > 0 else 0

        if T0p <= 0:
            pose_s = np.zeros((self.T, self.lkp_count, 3), dtype="float32")
        else:
            idxs_p = self._sample_indices(T0p)
            pose_s = pose_arr[idxs_p]
            if pose_s.ndim == 2:
                expected = self.lkp_count * 3
                if pose_s.shape[1] < expected:
                    padded = np.zeros((self.T, expected), dtype="float32")
                    padded[:, : pose_s.shape[1]] = pose_s
                    pose_s = padded
                pose_s = pose_s.reshape(self.T, self.lkp_count, 3)
            elif pose_s.ndim == 3:
                if pose_s.shape[1] != self.lkp_count:
                    trimmed = np.zeros((self.T, self.lkp_count, pose_s.shape[2]), dtype="float32")
                    copy_landmarks = min(self.lkp_count, pose_s.shape[1])
                    trimmed[:, :copy_landmarks, : pose_s.shape[2]] = pose_s[:, :copy_landmarks]
                    pose_s = trimmed
                if pose_s.shape[2] < 3:
                    pad = np.zeros((self.T, self.lkp_count, 3 - pose_s.shape[2]), dtype="float32")
                    pose_s = np.concatenate([pose_s, pad], axis=2)
                elif pose_s.shape[2] > 3:
                    pose_s = pose_s[:, :, :3]
            else:
                pose_s = np.zeros((self.T, self.lkp_count, 3), dtype="float32")

        conf = pose_s[:, :, 2]
        mask = conf >= self.min_conf
        pose_s[:, :, :2] *= mask[..., None].astype("float32")
        pose_tensor = torch.from_numpy(pose_s.reshape(self.T, self.lkp_count * 3))
        mask_tensor = torch.from_numpy(mask.astype("bool"))
        return pose_tensor, mask_tensor

    def _flip_pose_tensor(self, pose: torch.Tensor) -> torch.Tensor:
        """Devuelve ``pose`` reflejada horizontalmente, intercambiando lados."""

        if pose.numel() == 0:
            return pose

        flipped = pose.clone()
        T, pose_dim = flipped.shape
        reshaped = flipped.view(T, -1, 3)
        lkp_count = reshaped.shape[1]

        reshaped[:, :, 0] = 1.0 - reshaped[:, :, 0]

        for left_idx, right_idx in self._pose_swap_pairs(lkp_count):
            if left_idx < lkp_count and right_idx < lkp_count:
                reshaped[:, [left_idx, right_idx]] = reshaped[:, [right_idx, left_idx]]

        return reshaped.view(T, pose_dim)

    def _flip_pose_mask(self, mask: torch.Tensor) -> torch.Tensor:
        """Refleja la máscara de confianza de la pose."""

        if mask.numel() == 0:
            return mask

        flipped = mask.clone()
        T, lkp_count = flipped.shape

        for left_idx, right_idx in self._pose_swap_pairs(lkp_count):
            if left_idx < lkp_count and right_idx < lkp_count:
                flipped[:, [left_idx, right_idx]] = flipped[:, [right_idx, left_idx]]

        return flipped

    @staticmethod
    def _pose_swap_pairs(lkp_count: int) -> List[tuple[int, int]]:
        """Pares de landmarks que deben intercambiarse al reflejar."""

        del lkp_count  # solo para mantener la firma uniforme
        return [
            (1, 4),
            (2, 5),
            (3, 6),
            (7, 8),
            (9, 10),
            (11, 12),
            (13, 14),
            (15, 16),
        ]

    def _effective_length(self, lengths: Dict[str, int]) -> int:
        positives = [v for v in lengths.values() if v > 0]
        if not positives:
            return 0
        if any(v == 0 for v in lengths.values()):
            face_length = lengths.get("face", 0)
            if face_length > 0:
                return min(face_length, self.T)
            return min(max(positives), self.T)
        return min(min(positives), self.T)

    def _frame_indices(self, frames: List[str]) -> List[int]:
        indices: List[int] = []
        for path in frames:
            name = os.path.basename(path)
            stem, _ = os.path.splitext(name)
            if "_f" in stem:
                try:
                    idx = int(stem.split("_f", 1)[1])
                except ValueError:
                    continue
                indices.append(idx)
        return sorted(indices)

    def _detect_missing_indices(self, indices: List[int]) -> List[int]:
        if not indices:
            return []
        expected = range(indices[0], indices[-1] + 1)
        idx_set = set(indices)
        return [i for i in expected if i not in idx_set]

    def _build_quality_report(
        self,
        vid: str,
        *,
        face_frames: List[str],
        hand_l_frames: List[str],
        hand_r_frames: List[str],
        pose_frames: Any,
        effective_length: int,
    ) -> Dict[str, Any]:
        report: Dict[str, Any] = {
            "video_id": vid,
            "effective_length": effective_length,
            "missing_frames": {},
            "fps": None,
            "issues": [],
        }

        streams = {
            "face": face_frames,
            "hand_l": hand_l_frames,
            "hand_r": hand_r_frames,
        }

        meta = self.meta.get(vid, {})
        expected_total = None
        if meta:
            expected_total = meta.get("frame_count")
        if expected_total is None:
            expected_total = max(len(face_frames), len(hand_l_frames), len(hand_r_frames), effective_length)
        expected_total = int(expected_total) if expected_total is not None else None

        for name, frames in streams.items():
            indices = self._frame_indices(frames)
            missing = self._detect_missing_indices(indices)
            if missing:
                report["missing_frames"][name] = {
                    "count": len(missing),
                    "indices": missing,
                    "available": len(indices),
                    "expected": len(indices) + len(missing),
                }
            if expected_total is not None and len(indices) < expected_total:
                deficit = expected_total - len(indices)
                entry = report["missing_frames"].setdefault(
                    name,
                    {
                        "count": 0,
                        "indices": [],
                        "available": len(indices),
                        "expected": expected_total,
                    },
                )
                entry["count"] += deficit
                entry["expected"] = expected_total

        pose_len = pose_frames.shape[0] if hasattr(pose_frames, "shape") else 0
        if pose_len <= 0:
            report["missing_frames"]["pose"] = {
                "count": 1,
                "indices": "all",
                "available": 0,
                "expected": effective_length,
            }

        expected_fps = meta.get("fps") if meta else None
        duration = meta.get("duration") if meta else None
        frame_count = meta.get("frame_count") if meta else None

        actual_frames = max([len(face_frames), len(hand_l_frames), len(hand_r_frames), pose_len, effective_length])
        actual_fps = None
        if duration and duration > 0:
            actual_fps = actual_frames / float(duration)

        if expected_fps is None and frame_count and duration:
            expected_fps = frame_count / float(duration)

        fps_info: Optional[Dict[str, Any]] = None
        if expected_fps is not None or actual_fps is not None:
            diff = None
            ok = True
            if expected_fps is not None and actual_fps is not None:
                diff = abs(actual_fps - expected_fps)
                ok = diff <= self.fps_tolerance
            fps_info = {
                "expected": expected_fps,
                "actual": actual_fps,
                "diff": diff,
                "ok": ok,
            }
        report["fps"] = fps_info

        if self.quality_checks:
            issues: List[str] = []
            if report["missing_frames"]:
                issues.append(
                    f"Frames faltantes detectados en {vid}: "
                    + ", ".join(f"{k} ({v['count']})" for k, v in report["missing_frames"].items())
                )
            if fps_info and fps_info["expected"] and fps_info["actual"] and not fps_info["ok"]:
                issues.append(
                    f"FPS fuera de tolerancia para {vid}: esperado {fps_info['expected']}, "
                    f"observado {fps_info['actual']:.2f}"
                )
            if issues:
                report["issues"].extend(issues)
                message = "; ".join(issues)
                if self.quality_strict:
                    raise ValueError(message)
                warnings.warn(message)

        return report


def collate_fn(batch: Iterable[SampleItem]) -> Dict[str, Any]:
    batch_list = list(batch)
    if not batch_list:
        raise ValueError("El batch no puede estar vacío.")

    def stack_attr(attr: str) -> torch.Tensor:
        return torch.stack([getattr(sample, attr) for sample in batch_list], dim=0)

    return {
        "face": stack_attr("face"),
        "hand_l": stack_attr("hand_l"),
        "hand_r": stack_attr("hand_r"),
        "pose": stack_attr("pose"),
        "pose_conf_mask": stack_attr("pose_conf_mask"),
        "pad_mask": stack_attr("pad_mask"),
        "lengths": stack_attr("length"),
        "miss_mask_hl": stack_attr("miss_mask_hl"),
        "miss_mask_hr": stack_attr("miss_mask_hr"),
        "quality": [sample.quality for sample in batch_list],
        "texts": [sample.text for sample in batch_list],
        "video_ids": [sample.video_id for sample in batch_list],
    }
