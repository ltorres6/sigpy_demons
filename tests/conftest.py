"""Shared fixtures for the demons test suite.

Every numerical test runs on CPU and, when CuPy and a CUDA device are present,
again on GPU via the ``device`` fixture. Tests marked ``xfail`` document known
bugs; ``xfail_strict`` makes them fail loudly once the bug is fixed so the
marker gets removed in the same change.
"""

from __future__ import annotations

import os

os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
import pytest


def _gpu_available() -> bool:
    try:
        import cupy as cp

        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


GPU_AVAILABLE = _gpu_available()
requires_gpu = pytest.mark.skipif(not GPU_AVAILABLE, reason="CuPy/CUDA not available")


@pytest.fixture(params=[-1, pytest.param(0, marks=[pytest.mark.gpu, requires_gpu])], ids=["cpu", "gpu"])
def device(request):
    """Compute device in the legacy sigpy convention (-1 = CPU, 0 = first GPU)."""
    return request.param


@pytest.fixture
def xp(device):
    """Array module matching ``device``."""
    if device == -1:
        return np
    import cupy

    return cupy


def to_np(arr):
    """Return ``arr`` as a NumPy array regardless of where it lives."""
    if hasattr(arr, "get"):
        return arr.get()
    return np.asarray(arr)


def interior(ndim: int, margin: int = 6) -> tuple[slice, ...]:
    """Slice away ``margin`` voxels at every border to avoid boundary effects."""
    return (slice(margin, -margin),) * ndim


def grid(shape: tuple[int, ...]) -> np.ndarray:
    """Voxel coordinate grid of shape ``(ndim, *shape)``."""
    return np.mgrid[tuple(slice(0, s) for s in shape)].astype(np.float64)


def blobs(shape: tuple[int, ...], specs=None) -> np.ndarray:
    """Smooth test image: a sum of isotropic Gaussian blobs.

    ``specs`` is a list of ``(centre_fraction, sigma)`` pairs. Defaults give a
    few overlapping blobs so the image has gradients in every direction.
    """
    if specs is None:
        specs = [((0.40, 0.45, 0.50), 0.10), ((0.62, 0.55, 0.45), 0.08), ((0.47, 0.68, 0.55), 0.06)]
    g = grid(shape)
    out = np.zeros(shape)
    for centre, sigma in specs:
        r2 = sum((g[i] - centre[i] * shape[i]) ** 2 for i in range(len(shape)))
        out += np.exp(-r2 / (2 * (sigma * min(shape)) ** 2))
    return out


def smooth_field(shape: tuple[int, ...], amplitude: float, seed: int = 0) -> np.ndarray:
    """Smooth random displacement field of shape ``(ndim, *shape)``."""
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(seed)
    field = np.stack([gaussian_filter(rng.standard_normal(shape), sigma=min(shape) / 6, mode="wrap") for _ in shape])
    return amplitude * field / np.abs(field).max()
