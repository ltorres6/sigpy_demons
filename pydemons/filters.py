"""Smoothing kernels and frequency-domain filters.

Functions that build arrays from scratch take ``xp`` (``numpy`` or ``cupy``)
to choose the device.
"""

from __future__ import annotations

import math

import numpy as np

from pydemons.backend import get_array_module
from pydemons.separable import correlate_along


def gaussian_kernel_1d(sigma=1.0, xp=np):
    """Unnormalised, odd-length 1D Gaussian with standard deviation ``sigma``, truncated at 4 sigma."""
    n = 2 * max(math.ceil(4 * sigma), 1) + 1
    x = xp.arange(n, dtype=xp.float64) - (n - 1) / 2
    return xp.exp(-0.5 * (x / sigma) ** 2)


def gaussian_kernel(sigma=1.0, ndim=1, xp=np):
    """Normalised, separable ``ndim``-dimensional Gaussian kernel."""
    k1d = gaussian_kernel_1d(sigma, xp=xp)
    kernel = k1d
    for _ in range(ndim - 1):
        kernel = xp.multiply.outer(kernel, k1d)
    return kernel / kernel.sum()


def gaussian_smooth(arr, sigma, axes=None, mode="constant"):
    """Separable Gaussian smoothing.

    Identical to convolving with ``gaussian_kernel(sigma, ndim)``, applied as
    one banded matrix multiply per axis. ``axes`` defaults to all axes; pass
    e.g. ``(1, 2, 3)`` to smooth every component of a 3D field in one call.
    ``mode`` is ``"constant"`` (zero padding) or ``"reflect"`` (half-sample
    symmetric).
    """
    xp = get_array_module(arr)
    k1d = gaussian_kernel_1d(sigma)
    k1d = k1d / k1d.sum()
    for axis in range(arr.ndim) if axes is None else axes:
        arr = correlate_along(arr, k1d, axis, mode)
    return arr


def normalized_frequencies(n, xp=np):
    """Frequencies in cycles/sample of a centred (``fftshift``-ed) length-``n`` FFT."""
    return xp.fft.fftshift(xp.fft.fftfreq(n))


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


def _extended_box_pass(arr, axis, r, c1, c2):
    """One extended box pass along ``axis`` using running sums and half-sample symmetric edges."""
    xp = get_array_module(arr)
    n = arr.shape[axis]
    pad = [(0, 0)] * arr.ndim
    pad[axis] = (r + 2, r + 1)
    csum = xp.cumsum(xp.pad(arr, pad, mode="symmetric"), axis=axis)

    def window(m):
        """Sum of the 2m+1 samples centred on each output voxel."""
        hi = [slice(None)] * arr.ndim
        lo = [slice(None)] * arr.ndim
        hi[axis] = slice(r + 2 + m, r + 2 + m + n)
        lo[axis] = slice(r + 1 - m, r + 1 - m + n)
        return csum[tuple(hi)] - csum[tuple(lo)]

    return c1 * window(r + 1) + c2 * window(r)


def extended_box_gaussian(arr, sigma, passes=4):
    """Fast Gaussian smoothing by repeated extended box filters.

    Cost is independent of ``sigma``. The impulse response has variance exactly
    ``sigma**2``, and more passes give a closer match to a true Gaussian.
    Boundaries are half-sample symmetric (``d c b a | a b c d``).

    References
    ----------
    P. Gwosdek, S. Grewenig, A. Bruhn, J. Weickert, "Theoretical foundations of
    Gaussian convolution by extended box filtering", SSVM 2011.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    var = sigma**2 / passes
    r = int(math.floor(0.5 * math.sqrt(12 * var + 1) - 0.5))
    alpha = (2 * r + 1) * (r * (r + 1) - 3 * var) / (6 * (var - (r + 1) ** 2))
    c1 = alpha / (2 * alpha + 2 * r + 1)
    c2 = (1 - alpha) / (2 * alpha + 2 * r + 1)
    out = arr
    for axis in range(arr.ndim):
        for _ in range(passes):
            out = _extended_box_pass(out, axis, r, c1, c2)
    return out


def unsharp_mask(arr, sigma=1.0, amount=1.0):
    """Sharpen by adding back ``amount`` times the detail removed by a Gaussian blur."""
    blurred = gaussian_smooth(arr, sigma, mode="reflect")
    return arr + amount * (arr - blurred)
