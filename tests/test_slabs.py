"""Everything computed slab by slab must equal the single-slab result."""

import numpy as np
import pytest

import pydemons.interpolate as interpolate
from conftest import blobs, smooth_field, to_np
from pydemons.differences import gradient
from pydemons.energy import energy, regularization
from pydemons.fields import jacobian_determinant
from pydemons.forces import VARIANTS, demons_force

SHAPES = [(37, 29), (13, 11, 9), (2, 12), (6, 1, 5)]


def with_slabs(monkeypatch, voxels, fn):
    monkeypatch.setattr(interpolate, "SLAB_VOXELS", voxels)
    try:
        return fn()
    finally:
        monkeypatch.undo()


@pytest.mark.parametrize("shape", [(1, 5), (2, 7), (5, 4, 3), (9, 6)])
def test_gradient_rows_matches_full(xp, shape):
    arr = xp.asarray(np.random.default_rng(0).random(shape))
    full = to_np(gradient(arr))
    for rows in (slice(0, 1), slice(0, shape[0]), slice(shape[0] - 1, shape[0]), slice(1, max(2, shape[0] - 1))):
        assert np.allclose(to_np(gradient(arr, rows)), full[:, rows])


@pytest.mark.parametrize("shape", SHAPES)
def test_jacobian_determinant_rows(xp, shape):
    field = xp.asarray(smooth_field(shape, 2.0, seed=1))
    full = to_np(jacobian_determinant(field))
    rows = slice(1, shape[0])
    assert np.allclose(to_np(jacobian_determinant(field, rows)), full[rows])


@pytest.mark.parametrize("shape", SHAPES)
def test_regularization_independent_of_slabs(xp, monkeypatch, shape):
    field = xp.asarray(smooth_field(shape, 2.0, seed=2))
    one = float(with_slabs(monkeypatch, 1 << 30, lambda: regularization(field, 2.0)))
    many = float(with_slabs(monkeypatch, 5, lambda: regularization(field, 2.0)))
    expected = 4.0 * np.mean(to_np(jacobian_determinant(field)) ** 2)
    assert np.isclose(one, expected) and np.isclose(many, expected)
    assert np.isclose(float(energy(xp.zeros(shape), field, 2.0)), expected)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("shape", SHAPES)
def test_forces_independent_of_slabs(xp, monkeypatch, shape, variant):
    fixed = xp.asarray(blobs(shape))
    moving = xp.asarray(np.roll(blobs(shape), 1, axis=0) + 0.05 * np.random.default_rng(3).random(shape))
    field = xp.asarray(smooth_field(shape, 2.0, seed=4))
    g = gradient(fixed)
    kw = dict(fixed_grad=g, fixed_grad_sq=(g**2).sum(axis=0), variant=variant, compute_energy=True)
    u1, e1 = with_slabs(monkeypatch, 1 << 30, lambda: demons_force(fixed, moving, field, 2.0, **kw))
    u2, e2 = with_slabs(monkeypatch, 7, lambda: demons_force(fixed, moving, field, 2.0, **kw))
    assert np.allclose(to_np(u1), to_np(u2), atol=1e-12)
    assert np.isclose(float(e1), float(e2))
