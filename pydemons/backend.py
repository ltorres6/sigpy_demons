"""CPU/GPU agnostic array helpers.

Every function in pydemons runs on whatever device its input arrays live on:
NumPy arrays run on CPU, CuPy arrays run on GPU. CuPy is optional.

Example
-------
>>> import numpy as np
>>> from pydemons.backend import get_array_module
>>> get_array_module(np.zeros(3)).__name__
'numpy'
"""

from __future__ import annotations

from types import ModuleType
from typing import Any

import numpy as np

_cupy: ModuleType | None = None
_cupy_checked = False


def _get_cupy() -> ModuleType | None:
    """Return the CuPy module if it can be imported, else None (cached)."""
    global _cupy, _cupy_checked
    if not _cupy_checked:
        try:
            import cupy

            _cupy = cupy
        except ImportError:
            _cupy = None
        _cupy_checked = True
    return _cupy


def is_gpu_available() -> bool:
    """True if CuPy is installed and at least one CUDA device is visible."""
    cp = _get_cupy()
    if cp is None:
        return False
    try:
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


def get_array_module(*arrays: Any) -> ModuleType:
    """Return ``cupy`` if any input is a CuPy array, otherwise ``numpy``."""
    for arr in arrays:
        if hasattr(arr, "__cuda_array_interface__"):
            return _get_cupy()
    return np


def to_gpu(arr: Any) -> Any:
    """Move ``arr`` to the GPU, or return it as NumPy if no GPU is available."""
    if not is_gpu_available():
        return to_numpy(arr)
    return _get_cupy().asarray(arr)


def to_numpy(arr: Any) -> np.ndarray:
    """Return ``arr`` as a NumPy array, copying from the GPU if needed."""
    if hasattr(arr, "__cuda_array_interface__"):
        return arr.get()
    return np.asarray(arr)
