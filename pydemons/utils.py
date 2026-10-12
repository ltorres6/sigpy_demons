"""Small array utilities."""

from __future__ import annotations

from pydemons.backend import get_array_module


def resize(arr, oshape):
    """Centre-aligned zero-pad or crop of ``arr`` to ``oshape``.

    The centre voxel ``n // 2`` of each axis is kept aligned, matching
    ``sigpy.util.resize``.
    """
    xp = get_array_module(arr)
    oshape = tuple(oshape)
    if arr.shape == oshape:
        return arr
    ishift = [max(i // 2 - o // 2, 0) for i, o in zip(arr.shape, oshape)]
    oshift = [max(o // 2 - i // 2, 0) for i, o in zip(arr.shape, oshape)]
    copy = [min(i - si, o - so) for i, si, o, so in zip(arr.shape, ishift, oshape, oshift)]
    out = xp.zeros(oshape, dtype=arr.dtype)
    out[tuple(slice(s, s + c) for s, c in zip(oshift, copy))] = arr[tuple(slice(s, s + c) for s, c in zip(ishift, copy))]
    return out


def normalize(arr, vmin=0.0, vmax=1.0):
    """Linearly rescale ``arr`` so its range becomes ``[vmin, vmax]``."""
    xp = get_array_module(arr)
    lo, hi = xp.min(arr), xp.max(arr)
    return (vmax - vmin) * (arr - lo) / (hi - lo) + vmin
