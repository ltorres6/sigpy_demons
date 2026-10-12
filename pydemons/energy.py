"""Registration cost."""

from __future__ import annotations

from pydemons.backend import get_array_module
from pydemons.fields import jacobian_determinant


def energy(diff, field, alpha):
    """Similarity plus regularisation energy (after H. Lombaert's MATLAB demons).

    ``mean(diff**2) + alpha**2 * mean(det(I + grad(field))**2)``.
    """
    xp = get_array_module(diff, field)
    similarity = xp.mean(diff**2)
    regularization = xp.mean(jacobian_determinant(field) ** 2) * alpha**2
    return similarity + regularization
