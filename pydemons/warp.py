"""Image warping by a displacement field."""

from __future__ import annotations

from pydemons.backend import get_array_module, get_ndimage


def identity_grid(shape, xp, dtype=None):
    """Voxel coordinates of shape ``(len(shape), *shape)``."""
    return xp.stack(xp.meshgrid(*(xp.arange(s, dtype=dtype or xp.float64) for s in shape), indexing="ij"))


def warp(image, field):
    """Return ``image(x + field(x))`` using linear interpolation.

    Parameters
    ----------
    image : ndarray, shape ``(*shape)``
    field : ndarray, shape ``(image.ndim, *shape)``
        Displacement in voxels along each axis. Samples outside the image are 0.
    """
    xp = get_array_module(image, field)
    ndi = get_ndimage(xp)
    coords = identity_grid(image.shape, xp, field.dtype) + field
    return ndi.map_coordinates(image, coords, order=1, mode="constant", cval=0.0, prefilter=False)
