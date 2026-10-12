"""Demons force (update field) for each variant.

Convention: Im(x + T(x)) should match Is(x). If Im is Is shifted by +d along
axis 0 (Im(x) = Is(x - d)) the correct T is +d, so updates should push T
towards positive values along axis 0.
"""

import numpy as np
import pytest

from conftest import blobs, interior, to_np
from pydemons import forces
from pydemons.differences import gradient

VARIANTS = ["passive", "active", "inverse_consistent"]
SHAPES = [(64, 64), (32, 32, 32)]


def update(xp, Is, Im, T, variant, alpha=2.0):
    Is, Im, T = xp.asarray(Is), xp.asarray(Im), xp.asarray(T)
    dS = gradient(Is)
    V, loss = forces.demons_force(Is, Im, T, alpha, fixed_grad=dS, fixed_grad_sq=(dS**2).sum(axis=0), variant=variant)
    return to_np(V), float(loss)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("shape", SHAPES)
def test_identical_images_give_zero_update(xp, variant, shape):
    img = blobs(shape)
    V, loss = update(xp, img, img, np.zeros((len(shape),) + shape), variant)
    assert V.shape == (len(shape),) + shape
    assert np.allclose(V, 0.0)
    assert np.isfinite(loss)


@pytest.mark.parametrize("variant", VARIANTS)
def test_flat_images_give_finite_zero_update(xp, variant):
    # Zero gradient and zero difference is 0/0; it must be masked rather than become NaN.
    shape = (24, 24)
    V, loss = update(xp, np.ones(shape), np.ones(shape), np.zeros((2,) + shape), variant)
    assert np.all(np.isfinite(V)) and np.allclose(V, 0.0)
    assert np.isfinite(loss)


@pytest.mark.parametrize("variant", VARIANTS)
@pytest.mark.parametrize("shape", SHAPES)
def test_update_points_towards_misalignment(xp, variant, shape):
    img = blobs(shape)
    moving = np.roll(img, 1, axis=0)
    V, _ = update(xp, img, moving, np.zeros((len(shape),) + shape), variant)
    assert V[0].sum() > 0
    for i in range(1, len(shape)):
        assert abs(V[i].sum()) < V[0].sum()


@pytest.mark.parametrize("variant", ["passive", "active"])
def test_update_step_is_bounded(xp, variant):
    # |diff * g| / (|g|^2 + alpha^2 diff^2) <= 1 / (2 alpha) for any gradient g.
    rng = np.random.default_rng(0)
    shape, alpha = (48, 48), 2.0
    Is, Im = rng.random(shape), rng.random(shape)
    T = rng.standard_normal((2,) + shape)
    V, _ = update(xp, Is, Im, T, variant, alpha=alpha)
    assert np.sqrt((V**2).sum(axis=0)).max() <= 1 / (2 * alpha) + 1e-9


@pytest.mark.parametrize("update_fn", [forces.active_force, forces.inverse_consistent_force], ids=["active", "inverse_consistent"])
def test_update_depends_on_moving_image_only_through_warp(xp, update_fn):
    # Im6 shifted by 6 and warped by T = 5 equals Im1 shifted by 1 with T = 0, so the updates must agree.
    shape = (64, 64)
    img = blobs(shape)
    T5 = np.zeros((2,) + shape)
    T5[0] = 5.0
    a, _ = update_fn(xp.asarray(img), xp.asarray(np.roll(img, 6, axis=0)), xp.asarray(T5), 2.0)
    b, _ = update_fn(xp.asarray(img), xp.asarray(np.roll(img, 1, axis=0)), xp.zeros((2,) + shape), 2.0)
    inner = (slice(None),) + interior(2, 10)
    assert np.allclose(to_np(a)[inner], to_np(b)[inner], atol=1e-6)


def test_unknown_variant_raises():
    img = np.zeros((8, 8))
    with pytest.raises(ValueError):
        forces.demons_force(img, img, np.zeros((2, 8, 8)), 2.0, variant="nonsense")
