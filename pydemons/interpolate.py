"""Linear interpolation at displaced grid positions, in plain array operations."""

from __future__ import annotations

import math

from pydemons.backend import get_array_module

#: Voxels per slab. Temporaries scale with this rather than with the volume.
SLAB_VOXELS = 1 << 22


def slabs(shape, voxels=None):
    """Consecutive ranges of the first axis covering roughly ``voxels`` voxels each."""
    plane = math.prod(shape[1:])
    rows = max(1, (voxels or SLAB_VOXELS) // max(plane, 1))
    return [slice(start, min(start + rows, shape[0])) for start in range(0, shape[0], rows)]


def sample_linear(values, field, out=None, rows=None):
    """Sample every component of ``values`` at ``x + field(x)``, linearly.

    Parameters
    ----------
    values : ndarray, shape ``(C, *shape)``
        Components sampled at the same positions (e.g. one image, or the
        components of a displacement field).
    field : ndarray, shape ``(ndim, *shape)``
        Displacement in voxels. Positions past the border take the edge value.
    out : ndarray, optional
        Array to write into, shaped like the result. Must not overlap ``values``.
    rows : slice, optional
        Only compute the output for this range of the first spatial axis.

    Returns
    -------
    ndarray, shape ``(C, *shape)``, or ``(C, len(rows), *shape[1:])`` with ``rows``.

    Notes
    -----
    The output is computed in slabs along the first axis, so temporary memory
    is bounded by :data:`SLAB_VOXELS` rather than by the volume size.
    """
    xp = get_array_module(values, field)
    shape = field.shape[1:]
    start, stop, _ = (rows or slice(None)).indices(shape[0])
    if out is None:
        out_shape = (values.shape[0], stop - start) + shape[1:]
        out = xp.empty(out_shape, dtype=xp.result_type(values.dtype, field.dtype))
    flat = values.reshape(values.shape[0], math.prod(shape))
    for part in slabs((stop - start,) + shape[1:]):
        first, last = start + part.start, start + part.stop
        out[:, part] = _sample_rows(xp, flat, shape, field[:, first:last], first)
    return out


def _sample_rows(xp, flat, shape, field, first_row):
    """Samples for rows ``first_row : first_row + field.shape[1]`` of the first spatial axis."""
    ndim = len(shape)
    sub_shape = field.shape[1:]
    size = math.prod(sub_shape)
    # int32 indices: best on GPU, no slower overall on CPU (measured with every variant).
    itype = xp.int32 if math.prod(shape) < 2**31 else xp.int64

    # Lower corner index (flattened) and weight per axis, computed once and
    # shared by every component and every corner. Weights stay in the field's
    # dtype (floor in float, not float minus int, which would promote to float64).
    base = None
    weights, steps = [], []
    for axis, n in enumerate(shape):
        stride = math.prod(shape[axis + 1 :])
        bshape = [1] * ndim
        bshape[axis] = sub_shape[axis]
        offset = first_row if axis == 0 else 0
        positions = xp.arange(offset, offset + sub_shape[axis], dtype=field.dtype).reshape(bshape)
        coord = field[axis] + positions
        xp.clip(coord, 0, n - 1, out=coord)
        lower = xp.floor(coord)
        if n > 1:
            xp.minimum(lower, n - 2, out=lower)
        coord -= lower  # now the weight of the upper neighbour
        weights.append(coord.reshape(size))
        steps.append(stride if n > 1 else 0)
        index = lower.astype(itype)
        del lower
        if stride != 1:
            index *= stride
        if base is None:
            base = index
        else:
            base += index
    base = base.reshape(size)

    return _blend(xp, flat, base, weights, steps, 0, 0).reshape((flat.shape[0],) + sub_shape)


def _blend(xp, flat, base, weights, steps, axis, offset):
    """Interpolate over axes ``axis..ndim-1`` from the corner at ``offset``.

    A module-level function rather than a nested closure: a recursive closure
    forms a reference cycle that keeps every slab's index and weight arrays
    alive until the cyclic garbage collector runs.
    """
    if axis == len(weights):
        # Gather from a view shifted by ``offset``: no ``base + offset`` array per corner.
        return xp.take(flat[:, offset:], base, axis=1)
    lo = _blend(xp, flat, base, weights, steps, axis + 1, offset)
    hi = _blend(xp, flat, base, weights, steps, axis + 1, offset + steps[axis])
    hi -= lo
    hi *= weights[axis]
    hi += lo
    return hi
