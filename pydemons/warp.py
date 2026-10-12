"""Image warping by a displacement field."""

from __future__ import annotations

from pydemons.backend import get_array_module, get_ndimage


def displaced_coordinates(field):
    """Sample coordinates ``x + field(x)``, shape ``field.shape``."""
    xp = get_array_module(field)
    ndim = field.shape[0]
    coords = field.copy()
    for i, n in enumerate(field.shape[1:]):
        coords[i] += xp.arange(n, dtype=field.dtype).reshape((n,) + (1,) * (ndim - 1 - i))
    return coords


def warp(image, field):
    """Return ``image(x + field(x))`` using linear interpolation.

    Parameters
    ----------
    image : ndarray, shape ``(*shape)``
    field : ndarray, shape ``(image.ndim, *shape)``
        Displacement in voxels along each axis. The image is extended past its
        border with its edge values.
    """
    xp = get_array_module(image, field)
    ndi = get_ndimage(xp)
    return ndi.map_coordinates(image, displaced_coordinates(field), order=1, mode="nearest", prefilter=False)
