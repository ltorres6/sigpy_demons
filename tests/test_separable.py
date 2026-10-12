"""Separable operators: blocked banded correlation and per-axis matrix application."""

import numpy as np
import pytest
from scipy.ndimage import correlate1d

from conftest import to_np
from pydemons.separable import correlate_along, linear_resample_matrix, matmul_along


@pytest.mark.parametrize("mode", ["constant", "reflect"])
@pytest.mark.parametrize("taps", [3, 9, 25])
@pytest.mark.parametrize("shape,axis", [((333,), 0), ((200, 7), 0), ((7, 200), 1), ((5, 150, 6), 1), ((4, 3, 129), 2), ((10, 12), 1)])
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_correlate_along_matches_scipy(xp, mode, taps, shape, axis, dtype):
    # Lengths above and below 2*block exercise both the blocked and the dense paths.
    rng = np.random.default_rng(taps)
    arr = rng.standard_normal(shape).astype(dtype)
    kernel = rng.random(taps)
    expected = correlate1d(arr.astype(np.float64), kernel, axis=axis, mode=mode)
    out = to_np(correlate_along(xp.asarray(arr), kernel, axis, mode))
    assert out.shape == shape and out.dtype == dtype
    assert np.allclose(out, expected, atol=1e-4 if dtype == np.float32 else 1e-11)


def test_correlate_along_reflect_on_axis_shorter_than_kernel(xp):
    arr = np.arange(3.0)
    kernel = np.arange(1.0, 10.0)
    assert np.allclose(to_np(correlate_along(xp.asarray(arr), kernel, 0, "reflect")), correlate1d(arr, kernel, mode="reflect"))


@pytest.mark.parametrize("shape,axis", [((6, 5), 0), ((6, 5), 1), ((3, 4, 5), 1)])
def test_matmul_along(xp, shape, axis):
    arr = np.random.default_rng(0).random(shape)
    mat = np.random.default_rng(1).random((7, shape[axis]))
    expected = np.moveaxis(np.tensordot(mat, arr, axes=([1], [axis])), 0, axis)
    assert np.allclose(to_np(matmul_along(xp.asarray(arr), xp.asarray(mat), axis)), expected)


@pytest.mark.parametrize("n_in,factor", [(5, 2.0), (8, 1.5), (9, 0.5), (1, 3.0)])
def test_linear_resample_matrix_rows(xp, n_in, factor):
    n_out = int(round(n_in * factor))
    mat = to_np(linear_resample_matrix(xp, n_in, n_out, factor))
    assert mat.shape == (n_out, n_in)
    assert np.allclose(mat.sum(axis=1), 1.0)  # linear interpolation preserves constants
    ramp = np.arange(n_in, dtype=float)
    assert np.allclose(mat @ ramp, np.clip(np.arange(n_out) / factor, 0, n_in - 1))
