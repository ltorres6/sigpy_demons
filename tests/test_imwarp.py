"""imwarp(I, T)(x) = I(x + T(x)) with linear interpolation."""

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from conftest import blobs, grid, interior, smooth_field, to_np
from imwarp import imwarp

SHAPES = [(32, 28), (16, 14, 12)]


@pytest.mark.parametrize("shape", SHAPES)
def test_zero_field_is_identity(xp, device, shape):
    img = blobs(shape)
    out = imwarp(xp.asarray(img), xp.zeros((len(shape),) + shape), device=device)
    assert out.shape == shape
    assert np.allclose(to_np(out), img)


@pytest.mark.parametrize("shape", SHAPES)
def test_integer_translation(xp, device, shape):
    img = blobs(shape)
    field = np.zeros((len(shape),) + shape)
    field[0] = 3.0
    out = to_np(imwarp(xp.asarray(img), xp.asarray(field), device=device))
    expected = np.roll(img, -3, axis=0)
    inner = interior(len(shape), 4)
    assert np.allclose(out[inner], expected[inner])


@pytest.mark.parametrize("shape", SHAPES)
def test_subvoxel_translation_of_ramp_is_exact(xp, device, shape):
    ramp = grid(shape).sum(axis=0)
    field = np.full((len(shape),) + shape, 0.3)
    out = to_np(imwarp(xp.asarray(ramp), xp.asarray(field), device=device))
    inner = interior(len(shape), 2)
    assert np.allclose(out[inner], ramp[inner] + 0.3 * len(shape))


@pytest.mark.parametrize("shape", SHAPES)
def test_matches_scipy_map_coordinates(xp, device, shape):
    img = blobs(shape)
    field = smooth_field(shape, amplitude=2.5, seed=11)
    expected = map_coordinates(img, grid(shape) + field, order=1)
    out = to_np(imwarp(xp.asarray(img), xp.asarray(field), device=device))
    inner = interior(len(shape), 4)
    assert np.allclose(out[inner], expected[inner], atol=1e-6)
