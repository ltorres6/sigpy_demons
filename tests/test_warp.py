"""warp(I, T)(x) = I(x + T(x)) with linear interpolation."""

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from conftest import blobs, grid, interior, smooth_field, to_np
from pydemons.warp import warp

SHAPES = [(32, 28), (16, 14, 12)]


@pytest.mark.parametrize("shape", SHAPES)
def test_zero_field_is_identity(xp, shape):
    img = blobs(shape)
    out = warp(xp.asarray(img), xp.zeros((len(shape),) + shape))
    assert out.shape == shape
    assert np.allclose(to_np(out), img)


@pytest.mark.parametrize("shape", SHAPES)
def test_integer_translation(xp, shape):
    img = blobs(shape)
    field = np.zeros((len(shape),) + shape)
    field[0] = 3.0
    out = to_np(warp(xp.asarray(img), xp.asarray(field)))
    expected = np.roll(img, -3, axis=0)
    inner = interior(len(shape), 4)
    assert np.allclose(out[inner], expected[inner])


@pytest.mark.parametrize("shape", SHAPES)
def test_subvoxel_translation_of_ramp_is_exact(xp, shape):
    ramp = grid(shape).sum(axis=0)
    field = np.full((len(shape),) + shape, 0.3)
    out = to_np(warp(xp.asarray(ramp), xp.asarray(field)))
    inner = interior(len(shape), 2)
    assert np.allclose(out[inner], ramp[inner] + 0.3 * len(shape))


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_scipy_map_coordinates(xp, shape):
    img = blobs(shape)
    field = smooth_field(shape, amplitude=2.5, seed=11)
    expected = map_coordinates(img, grid(shape) + field, order=1)
    out = to_np(warp(xp.asarray(img), xp.asarray(field)))
    inner = interior(len(shape), 4)
    assert np.allclose(out[inner], expected[inner], atol=1e-6)


@pytest.mark.parametrize("shape", SHAPES)
def test_border_is_not_darkened(xp, shape):
    # A constant image stays constant under any warp if the border is extended.
    field = smooth_field(shape, amplitude=4.0, seed=12)
    out = to_np(warp(xp.full(shape, 0.7), xp.asarray(field)))
    assert np.allclose(out, 0.7)
