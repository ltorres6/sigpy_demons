"""Resampling between pyramid levels."""

from __future__ import annotations

from pydemons.backend import get_array_module, get_ndimage
from pydemons.filters import butterworth


def zoom(arr, factor):
    """Resample ``arr`` by ``factor``.

    Downsampling (``factor < 1``, ``1 / factor`` an integer) applies a
    Butterworth anti-aliasing filter and keeps every ``1 / factor``-th voxel
    starting at 0. Upsampling uses linear interpolation.
    """
    xp = get_array_module(arr)
    if factor == 1:
        return arr
    if factor < 1:
        h = butterworth(0.5 * factor, arr.shape, order=2, xp=xp)
        spectrum = xp.fft.fftn(arr) * xp.fft.ifftshift(h * h)
        filtered = xp.real(xp.fft.ifftn(spectrum))
        step = int(1 // factor)
        return filtered[(slice(None, None, step),) * arr.ndim]
    return get_ndimage(xp).zoom(arr, factor, order=1)
