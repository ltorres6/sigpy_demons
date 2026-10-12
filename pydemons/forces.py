"""Demons forces: the per-iteration update field for each variant.

Convention: ``moving(x + field(x))`` should match ``fixed(x)``. Variants follow
https://arxiv.org/abs/0909.0928.
"""

from __future__ import annotations

import math

import numpy as np

from pydemons.backend import get_array_module
from pydemons.differences import gradient
from pydemons.energy import regularization
from pydemons.fields import exp_field
from pydemons.interpolate import sample_linear, slabs
from pydemons.warp import warp

DENOMINATOR_THRESHOLD = 1e-9
VARIANTS = ("passive", "active", "inverse_consistent")


def _force(diff, grad, grad_sq, alpha, scaling_factor, scale=1.0, out=None):
    """``scale * diff * grad / (|grad|^2 + alpha^2 diff^2)``, zeroed where unreliable."""
    xp = get_array_module(diff, grad)
    # Work on per-voxel arrays and broadcast over the components once at the end:
    # fewer full passes than masking the (ndim, *shape) update.
    denominator = diff * diff
    denominator *= alpha**2
    denominator += grad_sq
    valid = xp.abs(diff) >= 1e-2**scaling_factor
    valid &= denominator >= DENOMINATOR_THRESHOLD
    with np.errstate(divide="ignore", invalid="ignore"):
        factor = diff / denominator
    del denominator
    factor *= scale
    valid &= xp.isfinite(factor)
    return xp.multiply(grad, xp.where(valid, factor, 0.0), out=out)


def _slab_force(fixed, warped_rows, grad_rows, field, alpha, scaling_factor, scale, compute_energy, out):
    """Fill ``out`` slab by slab and return the energy (or None).

    ``warped_rows(rows)`` gives the warped moving image for those rows and
    ``grad_rows(rows)`` the force gradient and its squared norm there, so no
    full-size difference image is ever built.
    """
    xp = get_array_module(fixed, field)
    squares = 0.0
    for rows in slabs(fixed.shape):
        diff = fixed[rows] - warped_rows(rows)
        grad, grad_sq = grad_rows(rows)
        _force(diff, grad, grad_sq, alpha, scaling_factor, scale, out=out[:, rows])
        if compute_energy:
            squares = squares + xp.sum(diff * diff)
    if not compute_energy:
        return None
    return squares / math.prod(fixed.shape) + regularization(field, alpha)


def passive_force(
    fixed, moving, field, alpha, fixed_grad, fixed_grad_sq, scaling_factor=1.0, compute_energy=True, out=None
):
    """Thirion's force using the (precomputed) fixed-image gradient."""
    xp = get_array_module(fixed, field)
    out = xp.empty_like(field) if out is None else out
    loss = _slab_force(
        fixed,
        lambda rows: sample_linear(moving[None], field, rows=rows)[0],
        lambda rows: (fixed_grad[:, rows], fixed_grad_sq[rows]),
        field,
        alpha,
        scaling_factor,
        1.0,
        compute_energy,
        out,
    )
    return out, loss


def _gradient_force(fixed, moving, field, alpha, scaling_factor, compute_energy, out, include_fixed, scale):
    """Shared body of the active and symmetric forces; gradients are taken per slab."""
    xp = get_array_module(fixed, moving)
    out = xp.empty_like(field) if out is None else out
    warped = warp(moving, field)

    def grad_rows(rows):
        grad = gradient(warped, rows)
        if include_fixed:
            grad += gradient(fixed, rows)
        return grad, xp.sum(grad * grad, axis=0)

    loss = _slab_force(
        fixed, lambda rows: warped[rows], grad_rows, field, alpha, scaling_factor, scale, compute_energy, out
    )
    return out, loss


def active_force(fixed, moving, field, alpha, scaling_factor=1.0, compute_energy=True, out=None):
    """Force driven by the gradient of the warped moving image."""
    return _gradient_force(fixed, moving, field, alpha, scaling_factor, compute_energy, out, False, 1.0)


def inverse_consistent_force(fixed, moving, field, alpha, scaling_factor=1.0, compute_energy=True, out=None):
    """Force driven by the sum of the fixed and warped-moving gradients."""
    return _gradient_force(fixed, moving, field, alpha, scaling_factor, compute_energy, out, True, 2.0)


def demons_force(
    fixed,
    moving,
    field,
    alpha,
    inverse_field=None,
    fixed_grad=None,
    fixed_grad_sq=None,
    variant="passive",
    scaling_factor=1.0,
    diffeomorphic=False,
    compute_energy=True,
    out=None,
):
    """Update field and energy for one demons iteration.

    Parameters
    ----------
    fixed, moving : ndarray
        Fixed (static) and moving images.
    field : ndarray, shape ``(ndim, *shape)``
        Current displacement (or velocity, if ``diffeomorphic``) field.
    alpha : float
        Step-size control; the update is bounded by ``1 / (2 * alpha)``.
    fixed_grad, fixed_grad_sq : ndarray
        ``gradient(fixed)`` and its squared magnitude; required for ``"passive"``.
    variant : {"passive", "active", "inverse_consistent"}
    diffeomorphic : bool
        If True, ``field`` is a stationary velocity field and the images are
        compared under its exponential.
    compute_energy : bool
        If False, skip the (relatively expensive) energy and return ``None`` for it.
    out : ndarray, optional
        Array to write the update into (must not overlap the inputs).
    """
    if diffeomorphic:
        if variant == "inverse_consistent":
            inverse_field = exp_field(-field)
        field = exp_field(field)
    if variant == "passive":
        return passive_force(
            fixed, moving, field, alpha, fixed_grad, fixed_grad_sq, scaling_factor, compute_energy, out=out
        )
    if variant == "active":
        return active_force(fixed, moving, field, alpha, scaling_factor, compute_energy, out=out)
    if variant == "inverse_consistent":
        update, loss = inverse_consistent_force(fixed, moving, field, alpha, scaling_factor, compute_energy, out=out)
        if inverse_field is None:
            inverse_field = -1 * field
        backward, loss2 = inverse_consistent_force(moving, fixed, inverse_field, alpha, scaling_factor, compute_energy)
        update -= backward
        del backward
        update *= 0.5
        return update, None if loss is None else (loss + loss2) * 0.5
    raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
