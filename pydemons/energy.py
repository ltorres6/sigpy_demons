"""Registration cost."""

from __future__ import annotations

from pydemons.backend import get_array_module
from pydemons.fields import jacobian, jacobian_determinant


def energy(diff, field, alpha):
    """Similarity plus regularisation energy (after H. Lombaert's MATLAB demons).

    ``mean(diff**2) + alpha**2 * mean(det(I + grad(field))**2)``.
    """
    xp = get_array_module(diff, field)
    similarity = xp.mean(diff**2)
    regularization = xp.mean(jacobian_determinant(field) ** 2) * alpha**2
    return similarity + regularization


def rmse(diff):
    """Root-mean-square of a difference image."""
    xp = get_array_module(diff)
    return xp.sqrt(xp.mean(diff**2))


def harmonic_energy(field, mask=None):
    """Mean squared Frobenius norm of the field's Jacobian, ``mean(|grad u|^2)``.

    ``mask`` (a boolean array or index) restricts the mean to part of the image.
    """
    xp = get_array_module(field)
    density = sum(xp.sum(g**2, axis=0) for g in jacobian(field))
    return xp.mean(density if mask is None else density[mask])


def quantity_of_update(update, field):
    """``sum|update| / sum|field|``: relative size of an update. ``inf`` if ``field`` is 0."""
    xp = get_array_module(update, field)
    total = float(xp.sum(xp.abs(field)))
    return float("inf") if total == 0 else float(xp.sum(xp.abs(update))) / total
