"""Smoothing kernels and frequency-domain filters.

Functions that build arrays from scratch take ``xp`` (``numpy`` or ``cupy``)
to choose the device.
"""

from __future__ import annotations

import math

import numpy as np


def gaussian_kernel_1d(sigma=1.0, xp=np):
    """Unnormalised, odd-length 1D Gaussian with standard deviation ``sigma``."""
    n = max(math.ceil(4 * sigma) // 2 * 2 + 1, 3)
    x = xp.arange(n, dtype=xp.float64) - (n - 1) / 2
    return xp.exp(-0.5 * (x / sigma) ** 2)


def gaussian_kernel(sigma=1.0, ndim=1, xp=np):
    """Normalised, separable ``ndim``-dimensional Gaussian kernel."""
    k1d = gaussian_kernel_1d(sigma, xp=xp)
    kernel = k1d
    for _ in range(ndim - 1):
        kernel = xp.multiply.outer(kernel, k1d)
    return kernel / kernel.sum()


def normalized_frequencies(n, xp=np):
    """Frequencies in cycles/sample of a centred (``fftshift``-ed) length-``n`` FFT."""
    if n % 2 == 0:
        return xp.linspace(-0.5 + 1 / (2 * n), 0.5 - 1 / (2 * n), n)
    return xp.linspace(-0.5, 0.5 - 1 / n, n)


def butterworth(cutoff, shape, order=2, xp=np):
    """Radial Butterworth low-pass transfer function on a centred frequency grid.

    Parameters
    ----------
    cutoff : float
        Cutoff frequency in cycles/sample, where the gain is 0.5.
    shape : tuple of int
        2D or 3D grid shape.
    order : int
        Filter order; higher is sharper.
    """
    freqs = [normalized_frequencies(n, xp=xp) for n in shape]
    grids = xp.meshgrid(*freqs, indexing="ij")
    radius = xp.sqrt(sum(g**2 for g in grids))
    return 1 / (1 + (radius / cutoff) ** (2 * order))
