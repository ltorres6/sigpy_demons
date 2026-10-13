"""FFT convolution as used by Demons for fluid/diffusion regularisation."""

import numpy as np
import pytest
from scipy.ndimage import convolve

from conftest import to_np
from pydemons.convolution import FFTConvolver
from pydemons.filters import gaussian_kernel


def smooth(xp, arr, sigma):
    k = gaussian_kernel(sigma, ndim=arr.ndim, xp=xp)
    return FFTConvolver(k, arr.shape)(xp.asarray(arr)), to_np(k)


@pytest.mark.parametrize("shape", [(30, 40), (12, 14, 10)])
def test_matches_zero_padded_direct_convolution(xp, shape):
    arr = np.random.default_rng(0).standard_normal(shape)
    out, k = smooth(xp, arr, 1.5)
    assert out.shape == shape
    assert np.allclose(to_np(out).real, convolve(arr, k, mode="constant"), atol=1e-5)


def test_output_is_real_for_real_input(xp):
    arr = np.random.default_rng(1).standard_normal((24, 24))
    out, _ = smooth(xp, arr, 2.0)
    out = to_np(out)
    assert np.abs(out.imag).max() < 1e-5 * np.abs(out.real).max()


def test_impulse_response_is_kernel(xp):
    arr = np.zeros((21, 21))
    arr[10, 10] = 1.0
    out, k = smooth(xp, arr, 1.0)
    r = k.shape[0] // 2
    assert np.allclose(to_np(out).real[10 - r : 11 + r, 10 - r : 11 + r], k, atol=1e-6)


def test_reusable_across_calls(xp):
    k = gaussian_kernel(1.0, ndim=2, xp=xp)
    smoother = FFTConvolver(k, (16, 16))
    a, b = (np.random.default_rng(s).standard_normal((16, 16)) for s in (2, 3))
    assert np.allclose(to_np(smoother(xp.asarray(a + b))), to_np(smoother(xp.asarray(a)) + smoother(xp.asarray(b))))


@pytest.mark.parametrize("method", ["convolve", "oa_convolve"])
@pytest.mark.parametrize("shape,kshape", [((30, 40), (5, 7)), ((12, 14, 10), (3, 3, 5)), ((500,), (31,))])
def test_direct_and_overlap_add_match_ndimage(xp, method, shape, kshape):
    from pydemons import convolution

    rng = np.random.default_rng(4)
    arr, k = rng.standard_normal(shape), rng.standard_normal(kshape)
    out = getattr(convolution, method)(xp.asarray(arr), xp.asarray(k))
    assert out.shape == shape
    assert np.allclose(to_np(out), convolve(arr, k, mode="constant"), atol=1e-8)


def test_convolve_requires_odd_kernel():
    from pydemons.convolution import convolve as pd_convolve

    with pytest.raises(ValueError):
        pd_convolve(np.zeros((8, 8)), np.ones((2, 3)))
