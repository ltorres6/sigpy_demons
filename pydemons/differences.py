"""Finite differences with periodic boundaries."""

from __future__ import annotations

from pydemons.backend import get_array_module


def gradient(arr):
    """Central-difference gradient of ``arr``.

    Returns an array of shape ``(arr.ndim, *arr.shape)``; boundaries wrap.
    """
    xp = get_array_module(arr)
    return xp.stack([(xp.roll(arr, -1, axis=i) - xp.roll(arr, 1, axis=i)) / 2 for i in range(arr.ndim)])


def laplacian(arr):
    """Sum of second differences along every axis; boundaries wrap."""
    xp = get_array_module(arr)
    return sum(xp.roll(arr, -1, axis=i) + xp.roll(arr, 1, axis=i) - 2 * arr for i in range(arr.ndim))
