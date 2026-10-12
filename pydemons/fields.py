"""Operations on displacement fields.

A displacement field ``u`` has shape ``(ndim, *shape)`` and represents the
mapping ``x -> x + u(x)`` in voxel units.
"""

from __future__ import annotations

import math

from pydemons.backend import get_array_module, get_ndimage
from pydemons.differences import gradient
from pydemons.warp import identity_grid


def compose(field1, field2, diffeomorphic=False, composition="A"):
    """Compose two displacement fields.

    With ``diffeomorphic=False`` this returns ``field1(x) + field2(x + field1(x))``.
    With ``diffeomorphic=True`` the fields are treated as stationary velocity
    fields and combined with the Baker-Campbell-Hausdorff approximation:
    ``"A"`` is ``v1 + v2`` and ``"B"`` adds the Lie bracket term.
    """
    if diffeomorphic is False:
        xp = get_array_module(field1, field2)
        ndi = get_ndimage(xp)
        coords = identity_grid(field1.shape[1:], xp, field1.dtype) + field1
        out = xp.stack(
            [ndi.map_coordinates(c, coords, order=1, mode="constant", cval=0.0, prefilter=False) for c in field2]
        )
        return out + field1
    elif diffeomorphic is True:
        if composition == "A":
            out = field1 + field2
        elif composition == "B":
            out = field1 + field2 + lie_bracket(field1, field2)
        return out


def exp_field(velocity):
    """Exponential of a stationary velocity field by scaling and squaring."""
    xp = get_array_module(velocity)
    # Smallest n >= 0 with max|v| / 2**n <= 0.5 voxel.
    n = 2 * float(xp.max(xp.sqrt(xp.sum(velocity**2, axis=0))))
    n = 0 if not math.isfinite(n) or n <= 1.0 else int(math.ceil(math.log2(n)))
    out = velocity * 2.0**-n
    for _ in range(n):
        out = compose(out, out)
    return out


def jacobian(field):
    """Spatial derivatives of each component: ``J[i][k] = d field_i / d x_k``."""
    return [gradient(component) for component in field]


def jacobian_determinant(field):
    """Determinant of ``I + grad(field)`` at every voxel."""
    ndim = field.shape[0]
    a = jacobian(field)
    for i in range(ndim):
        a[i][i] = a[i][i] + 1
    if ndim == 2:
        return a[0][0] * a[1][1] - a[0][1] * a[1][0]
    if ndim == 3:
        return (
            a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1])
            - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0])
            + a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0])
        )
    raise ValueError(f"expected a 2D or 3D field, got {ndim} components")


def lie_bracket(field1, field2):
    """Half the Lie bracket of two vector fields, as used in the BCH formula."""
    xp = get_array_module(field1, field2)
    j1, j2 = jacobian(field1), jacobian(field2)
    ndim = field1.shape[0]
    a = xp.stack([sum(j2[k][i] * field1[k] for k in range(ndim)) for i in range(ndim)])
    b = xp.stack([sum(j1[k][i] * field2[k] for k in range(ndim)) for i in range(ndim)])
    return 0.5 * (a - b)
