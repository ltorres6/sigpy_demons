"""Pyramid resampling. Downsampling keeps every 2nd voxel starting at 0, so
fine voxel 2k and coarse voxel k are the same physical point."""

import numpy as np
import pytest

from conftest import blobs, grid, to_np
from pydemons.resample import zoom


@pytest.mark.parametrize("shape", [(32, 32), (16, 8, 12)])
def test_factor_one_is_identity(xp, shape):
    arr = xp.asarray(blobs(shape))
    assert np.array_equal(to_np(zoom(arr, 1)), to_np(arr))


@pytest.mark.parametrize("factor", [0.5, 0.25])
@pytest.mark.parametrize("shape", [(32, 32), (16, 8, 12)])
def test_downsample_shape_and_dtype(xp, shape, factor):
    out = zoom(xp.asarray(blobs(shape)), factor)
    assert out.shape == tuple(int(s * factor) for s in shape)
    assert not np.iscomplexobj(to_np(out))


@pytest.mark.parametrize("shape", [(16, 16), (8, 4, 6)])
def test_upsample_shape(xp, shape):
    out = zoom(xp.asarray(blobs(shape)), 2)
    assert out.shape == tuple(2 * s for s in shape)


@pytest.mark.parametrize("shape", [(32, 32), (16, 8, 12)])
def test_downsample_preserves_constant(xp, shape):
    out = to_np(zoom(xp.full(shape, 2.0), 0.5))
    assert np.allclose(out, 2.0, rtol=1e-6)


@pytest.mark.parametrize("shape", [(64, 64), (64, 64, 64)])
def test_downsample_of_smooth_image_samples_even_voxels(xp, shape):
    img = blobs(shape)
    out = to_np(zoom(xp.asarray(img), 0.5))
    assert np.abs(out - img[(slice(None, None, 2),) * len(shape)]).max() < 3e-2


def test_upsample_preserves_constant(xp):
    out = to_np(zoom(xp.full((8, 8), 3.0), 2))
    assert np.allclose(out, 3.0)


@pytest.mark.xfail(reason="upsampling maps corners to corners, misaligned with the downsampling grid")
@pytest.mark.parametrize("shape", [(16, 16), (8, 8, 8)])
def test_upsample_aligns_with_downsample_grid(xp, shape):
    coarse = grid(shape).sum(axis=0)
    fine = to_np(zoom(xp.asarray(coarse), 2))
    assert np.allclose(fine[(slice(None, None, 2),) * len(shape)], coarse)
