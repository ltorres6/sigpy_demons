"""Linear interpolation at displaced grid positions, in plain array operations."""

from __future__ import annotations

import math

from pydemons.backend import get_array_module


def sample_linear(values, field):
    """Sample every component of ``values`` at ``x + field(x)``, linearly.

    Parameters
    ----------
    values : ndarray, shape ``(C, *shape)``
        Components sampled at the same positions (e.g. one image, or the
        components of a displacement field).
    field : ndarray, shape ``(ndim, *shape)``
        Displacement in voxels. Positions past the border take the edge value.

    Returns
    -------
    ndarray, shape ``(C, *shape)``
    """
    xp = get_array_module(values, field)
    shape = field.shape[1:]
    ndim = len(shape)
    size = math.prod(shape)
    # int32 indices: best on GPU, no slower overall on CPU (measured with every variant).
    itype = xp.int32 if size < 2**31 else xp.int64
    flat = values.reshape(values.shape[0], size)

    # Lower corner index (flattened) and weight per axis, computed once and
    # shared by every component and every corner. Weights stay in the field's
    # dtype (floor in float, not float minus int, which would promote to float64).
    base = None
    weights, steps = [], []
    for axis, n in enumerate(shape):
        stride = math.prod(shape[axis + 1 :])
        bshape = [1] * ndim
        bshape[axis] = n
        coord = field[axis] + xp.arange(n, dtype=field.dtype).reshape(bshape)
        xp.clip(coord, 0, n - 1, out=coord)
        lower = xp.floor(coord)
        if n > 1:
            xp.minimum(lower, n - 2, out=lower)
        coord -= lower  # now the weight of the upper neighbour
        weights.append(coord.reshape(size))
        steps.append(stride if n > 1 else 0)
        index = lower.astype(itype)
        if stride != 1:
            index *= stride
        if base is None:
            base = index
        else:
            base += index
    base = base.reshape(size)

    def blend(axis, offset):
        """Interpolate over axes ``axis..ndim-1`` from the corner at ``offset``."""
        if axis == ndim:
            # Gather from a view shifted by ``offset``: no ``base + offset`` array per corner.
            return xp.take(flat[:, offset:], base, axis=1)
        lo = blend(axis + 1, offset)
        hi = blend(axis + 1, offset + steps[axis])
        hi -= lo
        hi *= weights[axis]
        hi += lo
        return hi

    return blend(0, 0).reshape(values.shape[:1] + shape)
