"""Finite differences with periodic boundaries."""

from __future__ import annotations

from pydemons.backend import get_array_module


def _along(axis, ndim, s):
    """Index selecting slice ``s`` along ``axis``."""
    idx = [slice(None)] * ndim
    idx[axis] = s
    return tuple(idx)


def gradient(arr):
    """Central-difference gradient of ``arr``.

    Returns an array of shape ``(arr.ndim, *arr.shape)``; boundaries wrap.
    """
    xp = get_array_module(arr)
    out = xp.empty((arr.ndim,) + arr.shape, dtype=xp.result_type(arr.dtype, xp.float32))
    for i in range(arr.ndim):
        o = out[i]
        if arr.shape[i] == 1:
            o[...] = 0
            continue

        def at(s, i=i):
            return _along(i, arr.ndim, s)

        # Interior from slices, then the two wrapped edges; avoids the copies xp.roll makes.
        xp.subtract(arr[at(slice(2, None))], arr[at(slice(None, -2))], out=o[at(slice(1, -1))])
        xp.subtract(arr[at(slice(1, 2))], arr[at(slice(-1, None))], out=o[at(slice(0, 1))])
        xp.subtract(arr[at(slice(0, 1))], arr[at(slice(-2, -1))], out=o[at(slice(-1, None))])
    out *= 0.5
    return out


def laplacian(arr):
    """Sum of second differences along every axis; boundaries wrap."""
    xp = get_array_module(arr)
    return sum(xp.roll(arr, -1, axis=i) + xp.roll(arr, 1, axis=i) - 2 * arr for i in range(arr.ndim))
