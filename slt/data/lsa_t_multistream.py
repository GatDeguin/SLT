"""Dataset multi-stream para LSA-T."""
from __future__ import annotations

import importlib
import math
import os
import random
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
    pad_mask: torch.Tensor
    miss_mask_hl: torch.Tensor
    miss_mask_hr: torch.Tensor
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
        if T0 <= 0:
            return [0] * self.T
        if T0 >= self.T:
            stride = math.ceil(T0 / self.T)
            offset = random.randint(0, max(0, stride - 1))
            idxs = list(range(offset, min(offset + stride * self.T, T0), stride))
            while len(idxs) < self.T:
                idxs.append(idxs[-1])
            return idxs[: self.T]

        idxs = list(range(T0))
        while len(idxs) < self.T:
            idxs.append(idxs[-1])
        return idxs[: self.T]

    # ------------------------------------------------------------------
    # Dataset API
    # ------------------------------------------------------------------
    def __getitem__(self, index: int) -> SampleItem:
        vid = self.ids[index]
        text = str(self.texts.get(vid, "")).strip()

        face_frames = self._list_frames(self.face_dir, vid)
        hl_frames = self._list_frames(self.hand_l_dir, vid)
        hr_frames = self._list_frames(self.hand_r_dir, vid)

        T0 = max(len(face_frames), len(hl_frames), len(hr_frames))
        idxs = self._sample_indices(T0)

        pose_path = os.path.join(self.pose_dir, f"{vid}.npz")
        pose = self._load_pose(pose_path)

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

        pose_t = self._sample_pose(pose)

        pad_mask = torch.ones(self.T, dtype=torch.bool)
        miss_mask_hl = torch.tensor(miss_hl, dtype=torch.bool)
        miss_mask_hr = torch.tensor(miss_hr, dtype=torch.bool)

        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(1, 3, 1, 1)

        face = (face - mean) / std
        hand_l = (hand_l - mean) / std
        hand_r = (hand_r - mean) / std

        return SampleItem(
            face=face,
            hand_l=hand_l,
            hand_r=hand_r,
            pose=pose_t,
            pad_mask=pad_mask,
            miss_mask_hl=miss_mask_hl,
            miss_mask_hr=miss_mask_hr,
            text=text,
            video_id=vid,
        )

    # ------------------------------------------------------------------
    # Pose helpers
    # ------------------------------------------------------------------
    def _load_pose(self, pose_path: str) -> Any:
        np = self._np
        if not os.path.exists(pose_path):
            return np.zeros((1, 3 * self.lkp_count), dtype="float32")

        with np.load(pose_path) as pose_npz:
            pose = pose_npz.get("pose")
            if pose is None or pose.shape[0] == 0:
                return np.zeros((1, 3 * self.lkp_count), dtype="float32")
            return pose

    def _sample_pose(self, pose: Any) -> torch.Tensor:
        np = self._np
        T0p = pose.shape[0]
        if T0p <= 0:
            pose_s = np.zeros((self.T, 3 * self.lkp_count), dtype="float32")
        else:
            idxs_p = self._sample_indices(T0p)
            pose_s = pose[idxs_p]
        return torch.from_numpy(pose_s.astype("float32"))


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
        "pad_mask": stack_attr("pad_mask"),
        "miss_mask_hl": stack_attr("miss_mask_hl"),
        "miss_mask_hr": stack_attr("miss_mask_hr"),
        "texts": [sample.text for sample in batch_list],
        "video_ids": [sample.video_id for sample in batch_list],
    }
