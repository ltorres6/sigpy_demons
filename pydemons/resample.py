"""Resampling: pyramid zoom, band-limited upsampling and image pyramids."""

from __future__ import annotations

import functools
import math

from pydemons.backend import get_array_module
from pydemons.separable import correlate_along, linear_resample_matrix, matmul_along

# Burt & Adelson's 5-tap binomial smoothing kernel (sigma ~ 1).
_BINOMIAL = (1 / 16, 4 / 16, 6 / 16, 4 / 16, 1 / 16)


def zoom(arr, factor):
    """Resample ``arr`` by ``factor`` with linear interpolation.

    Output voxel ``i`` samples input coordinate ``i / factor``, so for
    ``factor = 1/2`` voxel ``k`` of the result is voxel ``2k`` of the input.
    Downsampling first applies a Butterworth anti-aliasing filter with cutoff
    at the new Nyquist frequency.
    """
    xp = get_array_module(arr)
    if factor == 1:
        return arr
    if factor < 1:
        arr = _lowpass(arr, 0.5 * factor, order=2)
        step = 1 / factor
        if math.isclose(step, round(step)):
            return xp.ascontiguousarray(arr[(slice(None, None, round(step)),) * arr.ndim])
    # Linear interpolation on a regular grid is separable: one small matrix per axis.
    for axis, n in enumerate(arr.shape):
        resample = linear_resample_matrix(xp, n, int(round(n * factor)), factor, arr.dtype)
        arr = matmul_along(arr, resample, axis)
    return arr


def _lowpass(arr, cutoff, order):
    """Apply a squared radial Butterworth filter (see :func:`pydemons.filters.butterworth`).

    The gain depends only on |frequency|, so real FFTs give the same result as
    complex ones at half the cost.
    """
    xp = get_array_module(arr)
    axes = tuple(range(arr.ndim))
    spectrum = xp.fft.rfftn(arr, axes=axes)
    spectrum *= _squared_butterworth_rfft(xp, arr.shape, cutoff, order, arr.dtype.str)
    return xp.fft.irfftn(spectrum, s=arr.shape, axes=axes)


@functools.lru_cache(maxsize=4)
def _squared_butterworth_rfft(xp, shape, cutoff, order, dtype):
    """Squared Butterworth gain on the ``rfftn`` frequency grid (cached: the pyramid reuses it)."""
    freqs = [xp.fft.fftfreq(n) for n in shape[:-1]] + [xp.fft.rfftfreq(shape[-1])]
    grids = xp.meshgrid(*freqs, indexing="ij", sparse=True)
    h = 1 / (1 + (sum(g**2 for g in grids) / cutoff**2) ** order)
    return (h * h).astype(dtype)


def fft_upsample(arr, factor):
    """Band-limited (Fourier zero-padding) upsampling by an integer ``factor``.

    Exact for periodic signals with no energy at or above the input Nyquist
    frequency.
    """
    xp = get_array_module(arr)
    for axis, n in enumerate(arr.shape):
        m = n * factor
        spectrum = xp.fft.rfft(arr, axis=axis)
        shape = list(spectrum.shape)
        shape[axis] = m // 2 + 1
        padded = xp.zeros(shape, dtype=spectrum.dtype)
        keep = [slice(None)] * arr.ndim
        keep[axis] = slice(0, n // 2 + 1)
        padded[tuple(keep)] = spectrum[tuple(keep)]
        if n % 2 == 0:
            # The input Nyquist term stands for +n/2 and -n/2; split it between them.
            nyquist = [slice(None)] * arr.ndim
            nyquist[axis] = n // 2
            padded[tuple(nyquist)] *= 0.5
        arr = xp.fft.irfft(padded, n=m, axis=axis) * (m / n)
    return arr


def _smooth(arr, gain=1.0):
    """Separable binomial smoothing with half-sample symmetric boundaries."""
    kernel = [k * gain for k in _BINOMIAL]
    for axis in range(arr.ndim):
        arr = correlate_along(arr, kernel, axis, "reflect")
    return arr


def _expand(arr, shape):
    """Upsample by 2 to ``shape``: insert zeros, then smooth (Burt & Adelson's EXPAND)."""
    xp = get_array_module(arr)
    up = xp.zeros(shape, dtype=arr.dtype)
    up[(slice(None, None, 2),) * arr.ndim] = arr
    return _smooth(up, gain=2.0)


def gaussian_pyramid(arr, levels):
    """``levels`` images, each a smoothed and 2x-downsampled copy of the previous.

    Level 0 is ``arr``; level ``i`` has shape ``ceil(shape / 2**i)``.
    """
    pyramid = [arr]
    for _ in range(levels - 1):
        pyramid.append(_smooth(pyramid[-1])[(slice(None, None, 2),) * arr.ndim])
    return pyramid


def laplacian_pyramid(arr, levels):
    """Band-pass decomposition: each level holds the detail lost by the next coarser one.

    The last level is the coarsest Gaussian level. Invert with
    :func:`reconstruct_laplacian_pyramid`.
    """
    gauss = gaussian_pyramid(arr, levels)
    bands = [fine - _expand(coarse, fine.shape) for fine, coarse in zip(gauss, gauss[1:])]
    return bands + [gauss[-1]]


def reconstruct_laplacian_pyramid(pyramid):
    """Exactly invert :func:`laplacian_pyramid`."""
    out = pyramid[-1]
    for band in reversed(pyramid[:-1]):
        out = band + _expand(out, band.shape)
    return out
