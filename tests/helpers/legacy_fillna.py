"""Reference vector (gather) fillna for regression tests."""
from __future__ import annotations

import torch

from src.func.basic import _active_mask_torch


def legacy_fillna_torch(
    arr: torch.Tensor,
    axis: int = 0,
    *,
    force_value=None,
    backward: bool = False,
) -> torch.Tensor:
    """Full-rank ``expand`` + ``gather`` path (historical vector implementation)."""
    if axis < 0:
        axis = arr.ndim + axis
    notnan = ~torch.isnan(arr)
    if force_value is not None:
        active = _active_mask_torch(notnan, axis, backward)
        out = arr.clone()
        out[(~notnan) & active] = force_value
        return out
    n = arr.shape[axis]
    view = [1] * arr.ndim
    view[axis] = n
    ar = torch.arange(n, device=arr.device).reshape(view).expand(arr.shape)
    if backward:
        idx = torch.where(notnan, ar, torch.full_like(ar, n - 1))
        idx = torch.cummin(idx.flip(axis), dim=axis).values.flip(axis)
    else:
        idx = torch.where(notnan, ar, torch.zeros_like(ar))
        idx = torch.cummax(idx, dim=axis).values
    return torch.gather(arr, axis, idx)
