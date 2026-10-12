"""Jacobian and Jacobian determinant of displacement fields.

Displacement fields ``u`` have shape ``(ndim, *shape)``; the mapping is
``x -> x + u(x)`` so its Jacobian is ``I + grad(u)``.
"""

import numpy as np
import pytest

import diffeomorphism
import energy
from conftest import grid, interior, to_np

GRADIENT_SCALE_BUG = pytest.mark.xfail(reason="CentralDifference is 2x the gradient, so grad(u) is doubled")

JACOBIAN_DETS = [
    pytest.param(energy.jacobianDet, id="energy"),
    pytest.param(diffeomorphism.jacobianDet, id="diffeomorphism"),
]


@pytest.mark.parametrize("jacobian_det", JACOBIAN_DETS)
@pytest.mark.parametrize("shape", [(24, 20), (12, 10, 8)])
def test_det_of_zero_field_is_one(xp, jacobian_det, shape):
    det = jacobian_det(xp.zeros((len(shape),) + shape))
    assert det.shape == shape
    assert np.allclose(to_np(det), 1.0)


@pytest.mark.parametrize("jacobian_det", JACOBIAN_DETS)
@pytest.mark.parametrize("shape", [(24, 20), (12, 10, 8)])
def test_det_of_translation_is_one(xp, jacobian_det, shape):
    field = np.zeros((len(shape),) + shape)
    for i in range(len(shape)):
        field[i] = 0.7 * (i + 1)
    assert np.allclose(to_np(jacobian_det(xp.asarray(field))), 1.0)


@GRADIENT_SCALE_BUG
@pytest.mark.parametrize("jacobian_det", JACOBIAN_DETS)
@pytest.mark.parametrize("shape", [(24, 20), (12, 10, 8)])
def test_det_of_uniform_scaling(xp, jacobian_det, shape):
    # u_i = a_i * x_i  ->  det = prod(1 + a_i)
    rates = [0.1, -0.2, 0.05][: len(shape)]
    g = grid(shape)
    field = np.stack([a * g[i] for i, a in enumerate(rates)])
    det = to_np(jacobian_det(xp.asarray(field)))
    assert np.allclose(det[interior(len(shape), 1)], np.prod([1 + a for a in rates]))


@GRADIENT_SCALE_BUG
@pytest.mark.parametrize("jacobian_det", JACOBIAN_DETS)
def test_det_of_rotation_is_one(xp, jacobian_det):
    shape = (24, 24)
    theta = 0.2
    g = grid(shape) - 12.0
    rotated = np.stack(
        [np.cos(theta) * g[0] - np.sin(theta) * g[1], np.sin(theta) * g[0] + np.cos(theta) * g[1]]
    )
    det = to_np(jacobian_det(xp.asarray(rotated - g)))
    assert np.allclose(det[interior(2, 1)], 1.0)


@GRADIENT_SCALE_BUG
@pytest.mark.parametrize("jacobian_det", JACOBIAN_DETS)
def test_det_detects_folding(xp, jacobian_det):
    # u_0 = -2 x_0 maps x_0 -> -x_0: a reflection with determinant -1.
    shape = (20, 20)
    field = np.zeros((2,) + shape)
    field[0] = -2.0 * grid(shape)[0]
    det = to_np(jacobian_det(xp.asarray(field)))
    assert np.allclose(det[interior(2, 1)], -1.0)


@pytest.mark.parametrize("jacobian_det", JACOBIAN_DETS)
def test_det_of_shear_is_one(xp, jacobian_det):
    shape = (20, 20)
    field = np.zeros((2,) + shape)
    field[0] = 0.3 * grid(shape)[1]
    det = to_np(jacobian_det(xp.asarray(field)))
    assert np.allclose(det[interior(2, 1)], 1.0)


@GRADIENT_SCALE_BUG
@pytest.mark.parametrize("shape", [(24, 20), (12, 10, 8)])
def test_jacobian_components(xp, shape):
    # diffeomorphism.jacobian returns one gradient per component: J[i][k] = d u_i / d x_k.
    ndim = len(shape)
    a = np.arange(1, ndim * ndim + 1, dtype=np.float64).reshape(ndim, ndim) / 10
    g = grid(shape)
    field = np.stack([sum(a[i, k] * g[k] for k in range(ndim)) for i in range(ndim)])
    jac = diffeomorphism.jacobian(xp.asarray(field))
    assert len(jac) == ndim
    for i in range(ndim):
        for k in range(ndim):
            assert np.allclose(to_np(jac[i][k])[interior(ndim, 1)], a[i, k])
