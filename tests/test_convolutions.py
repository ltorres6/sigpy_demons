"""FFT convolution as used by Demons for fluid/diffusion regularisation."""

import numpy as np
import pytest
import sigpy as sp
from scipy.fft import next_fast_len
from scipy.ndimage import convolve

import convolutions as conv
import filters
from conftest import to_np


def smooth(xp, device, arr, sigma):
    """Mirror how Demons.run prepares the kernel and calls FFTConvolve."""
    k = filters.gaussianKernelnd(sigma, n=arr.ndim, device=device)
    fftsize = tuple(next_fast_len(k.shape[0] + s - 1) for s in arr.shape)
    kf = sp.fft(k, oshape=fftsize, norm=None)
    return conv.FFTConvolve(kf, xp.asarray(arr), shape=fftsize, device=device), to_np(k)


@pytest.mark.parametrize("shape", [(30, 40), (12, 14, 10)])
def test_matches_zero_padded_direct_convolution(xp, device, shape):
    arr = np.random.default_rng(0).standard_normal(shape)
    out, k = smooth(xp, device, arr, 1.5)
    assert out.shape == shape
    assert np.allclose(to_np(out).real, convolve(arr, k, mode="constant"), atol=1e-5)


def test_output_is_real_for_real_input(xp, device):
    arr = np.random.default_rng(1).standard_normal((24, 24))
    out, _ = smooth(xp, device, arr, 2.0)
    out = to_np(out)
    assert np.abs(out.imag).max() < 1e-5 * np.abs(out.real).max()


def test_impulse_response_is_kernel(xp, device):
    arr = np.zeros((21, 21))
    arr[10, 10] = 1.0
    out, k = smooth(xp, device, arr, 1.0)
    r = k.shape[0] // 2
    assert np.allclose(to_np(out).real[10 - r : 11 + r, 10 - r : 11 + r], k, atol=1e-6)
