"""Demons forces: the per-iteration update field for each variant.

Convention: ``moving(x + field(x))`` should match ``fixed(x)``. Variants follow
https://arxiv.org/abs/0909.0928.
"""

from __future__ import annotations

import numpy as np

from pydemons.backend import get_array_module
from pydemons.differences import gradient
from pydemons.energy import energy
from pydemons.fields import exp_field
from pydemons.warp import warp

DENOMINATOR_THRESHOLD = 1e-9
VARIANTS = ("passive", "active", "inverse_consistent")


def _force(diff, grad, grad_sq, alpha, scaling_factor, scale=1.0):
    """``scale * diff * grad / (|grad|^2 + alpha^2 diff^2)``, zeroed where unreliable."""
    xp = get_array_module(diff, grad)
    denominator = grad_sq + diff**2 * alpha**2
    with np.errstate(divide="ignore", invalid="ignore"):
        update = scale * (diff / denominator) * grad
    invalid = (
        ~xp.isfinite(update)
        | (xp.abs(diff) < 1e-2**scaling_factor)
        | (denominator < DENOMINATOR_THRESHOLD)
    )
    return xp.where(invalid, 0.0, update)


def passive_force(fixed, moving, field, alpha, fixed_grad, fixed_grad_sq, scaling_factor=1.0):
    """Thirion's force using the (precomputed) fixed-image gradient."""
    diff = fixed - warp(moving, field)
    return _force(diff, fixed_grad, fixed_grad_sq, alpha, scaling_factor), energy(diff, field, alpha)


def active_force(fixed, moving, field, alpha, scaling_factor=1.0):
    """Force driven by the moving-image gradient."""
    xp = get_array_module(fixed, moving)
    grad = gradient(moving)
    diff = fixed - warp(moving, field)
    return _force(diff, grad, xp.sum(grad**2, axis=0), alpha, scaling_factor), energy(diff, field, alpha)


def inverse_consistent_force(fixed, moving, field, alpha, scaling_factor=1.0):
    """Force driven by the sum of fixed and moving gradients."""
    xp = get_array_module(fixed, moving)
    grad = gradient(fixed) + gradient(moving)
    diff = fixed - warp(moving, field)
    return _force(diff, grad, xp.sum(grad**2, axis=0), alpha, scaling_factor, scale=2.0), energy(diff, field, alpha)


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
    """
    if variant == "passive":
        return passive_force(fixed, moving, field, alpha, fixed_grad, fixed_grad_sq, scaling_factor)
    if variant == "active":
        return active_force(fixed, moving, field, alpha, scaling_factor)
    if variant == "inverse_consistent":
        if diffeomorphic is True:
            inverse_field = exp_field(-1 * field)
            field = exp_field(field)
        update, loss = inverse_consistent_force(fixed, moving, field, alpha, scaling_factor)
        if inverse_field is None:
            inverse_field = -1 * field
        update2, loss2 = inverse_consistent_force(moving, fixed, inverse_field, alpha, scaling_factor)
        return (update - update2) * 0.5, (loss + loss2) * 0.5
    raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
