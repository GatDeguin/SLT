"""General utility functions for SLT experiments."""
from __future__ import annotations

import os
import random

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - NumPy is expected but optional.
    np = None  # type: ignore

try:
    import torch
except ModuleNotFoundError:  # pragma: no cover - PyTorch is expected but optional.
    torch = None  # type: ignore


def set_seed(seed: int, *, deterministic: bool = False) -> None:
    """Seed the most common pseudo-random number generators.

    Parameters
    ----------
    seed:
        Integer seed used to initialise the RNGs.
    deterministic:
        When ``True`` and PyTorch is available, configures cuDNN to run in
        deterministic mode (``benchmark`` disabled). When ``False`` the
        configuration favours performance (``benchmark`` enabled).

    The function is resilient to the absence of optional dependencies such as
    NumPy or PyTorch, which makes it convenient to reuse in lightweight
    environments (e.g. documentation builds).
    """

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)

    if np is not None:
        np.random.seed(seed)  # type: ignore[attr-defined]

    if torch is not None:
        torch.manual_seed(seed)  # type: ignore[attr-defined]
        if torch.cuda.is_available():  # type: ignore[attr-defined]
            torch.cuda.manual_seed_all(seed)  # type: ignore[attr-defined]

        try:
            torch.backends.cudnn.deterministic = bool(deterministic)  # type: ignore[attr-defined]
            torch.backends.cudnn.benchmark = not bool(deterministic)  # type: ignore[attr-defined]
        except AttributeError:
            # Some Torch builds (e.g. CPU-only) may not expose cuDNN controls.
            pass


def masked_mean(
    tensor: "torch.Tensor",
    mask: "torch.Tensor",
    dim: int,
    *,
    keepdim: bool = False,
) -> "torch.Tensor":
    """Compute the mean along ``dim`` considering only the positions where ``mask`` is ``1``.

    Parameters
    ----------
    tensor:
        Input tensor with arbitrary shape.
    mask:
        Tensor containing binary indicators (``1`` = valid, ``0`` = ignore).
        Its shape must be broadcastable to ``tensor`` except for the reduction
        dimension.
    dim:
        The dimension along which to reduce.
    keepdim:
        Mirrors :func:`torch.mean`. If ``True`` the reduced dimension is
        retained with size ``1``.
    """

    if torch is None:
        raise RuntimeError("masked_mean requires PyTorch to be installed")

    try:
        broadcast_mask = torch.broadcast_to(mask, tensor.shape)
    except RuntimeError as exc:  # pragma: no cover - defensive programming.
        raise ValueError("mask shape is not broadcastable to tensor shape") from exc

    broadcast_mask = broadcast_mask.to(dtype=tensor.dtype)
    masked_tensor = tensor * broadcast_mask

    numerator = masked_tensor.sum(dim=dim, keepdim=keepdim)
    denominator = broadcast_mask.sum(dim=dim, keepdim=keepdim)

    if torch.is_floating_point(tensor):
        eps = torch.finfo(tensor.dtype).tiny
    else:  # pragma: no cover - integer tensors are rare but supported.
        eps = 1

    denominator = denominator.clamp_min(eps)
    return numerator / denominator


__all__ = ["set_seed", "masked_mean"]
