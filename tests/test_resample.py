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


@pytest.mark.parametrize("shape", [(16, 16), (8, 8, 8)])
def test_upsample_aligns_with_downsample_grid(xp, shape):
    coarse = grid(shape).sum(axis=0)
    fine = to_np(zoom(xp.asarray(coarse), 2))
    assert np.allclose(fine[(slice(None, None, 2),) * len(shape)], coarse)


@pytest.mark.parametrize("factor", [0.75, 0.4, 1.5])
def test_zoom_arbitrary_factor_shape_and_constant(xp, factor):
    shape = (40, 30)
    out = to_np(zoom(xp.full(shape, 2.0), factor))
    assert out.shape == tuple(int(round(s * factor)) for s in shape)
    assert np.allclose(out, 2.0)


def test_zoom_arbitrary_factor_samples_smooth_image(xp):
    img = blobs((96, 96))
    out = to_np(zoom(xp.asarray(img), 0.75))
    g = grid(out.shape) / 0.75
    from scipy.ndimage import map_coordinates

    assert np.abs(out - map_coordinates(img, g, order=1, mode="nearest")).max() < 3e-2


@pytest.mark.parametrize("n", [16, 17])
def test_fft_upsample_is_exact_for_band_limited_signal(xp, n):
    from pydemons.resample import fft_upsample

    x = np.arange(n) / n
    signal = np.cos(2 * np.pi * 3 * x) + 0.5 * np.sin(2 * np.pi * 2 * x)
    out = to_np(fft_upsample(xp.asarray(np.tile(signal[:, None], (1, 4))), 2))
    xf = np.arange(2 * n) / (2 * n)
    expected = np.cos(2 * np.pi * 3 * xf) + 0.5 * np.sin(2 * np.pi * 2 * xf)
    assert out.shape == (2 * n, 8)
    assert np.allclose(out[:, 0], expected)


@pytest.mark.parametrize("shape", [(64, 48), (32, 24, 16), (37, 29)])
def test_gaussian_pyramid_shapes_and_constant(xp, shape):
    from pydemons.resample import gaussian_pyramid

    levels = gaussian_pyramid(xp.full(shape, 1.5), 3)
    assert len(levels) == 3
    assert levels[0].shape == shape
    for fine, coarse in zip(levels, levels[1:]):
        assert coarse.shape == tuple((s + 1) // 2 for s in fine.shape)
    assert all(np.allclose(to_np(lvl), 1.5) for lvl in levels)


@pytest.mark.parametrize("shape", [(64, 48), (32, 24, 16), (37, 29)])
def test_laplacian_pyramid_reconstructs_exactly(xp, shape):
    from pydemons.resample import laplacian_pyramid, reconstruct_laplacian_pyramid

    arr = np.random.default_rng(5).random(shape)
    pyr = laplacian_pyramid(xp.asarray(arr), 4)
    assert len(pyr) == 4
    assert np.allclose(to_np(reconstruct_laplacian_pyramid(pyr)), arr)
