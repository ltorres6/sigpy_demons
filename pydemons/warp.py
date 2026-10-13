"""Image warping by a displacement field."""

from __future__ import annotations

from pydemons.interpolate import sample_linear


def warp(image, field, out=None):
    """Return ``image(x + field(x))`` using linear interpolation.

    Parameters
    ----------
    image : ndarray, shape ``(*shape)``
    field : ndarray, shape ``(image.ndim, *shape)``
        Displacement in voxels along each axis. The image is extended past its
        border with its edge values.
    out : ndarray, optional
        Array to write the result into (must not overlap ``image``).
    """
    return sample_linear(image[None], field, out=None if out is None else out[None])[0]
