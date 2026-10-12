"""Gaussian smoothing kernels and the Butterworth anti-aliasing filter."""

import numpy as np
import pytest

import filters
from conftest import to_np


def kernel_variance(k1d):
    k1d = np.asarray(k1d) / np.sum(k1d)
    x = np.arange(len(k1d)) - len(k1d) // 2
    return float(np.sum(k1d * x**2))


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5, 3.0])
def test_gaussian_1d_is_odd_symmetric_and_centred(sigma):
    k = to_np(filters.gaussianKernel1d(sigma))
    assert len(k) % 2 == 1 and len(k) >= 3
    assert np.allclose(k, k[::-1])
    assert np.argmax(k) == len(k) // 2


@pytest.mark.xfail(reason="kernel is truncated at +/-2 sigma (comment says 4), so its variance is too small")
@pytest.mark.parametrize("sigma", [1.0, 1.5, 3.0])
def test_gaussian_1d_variance_matches_sigma(sigma):
    assert np.isclose(kernel_variance(to_np(filters.gaussianKernel1d(sigma))), sigma**2, rtol=0.02)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_gaussian_nd_is_normalised_separable_product(request, xp, device, n):
    if n == 1 and device != -1:
        request.applymarker(pytest.mark.xfail(reason="device is not passed to gaussianKernel1d, so n=1 stays on CPU"))
    k = filters.gaussianKernelnd(1.5, n=n, device=device)
    assert isinstance(k, xp.ndarray)
    k = to_np(k)
    k1d = to_np(filters.gaussianKernel1d(1.5))
    assert k.shape == (len(k1d),) * n
    assert np.isclose(k.sum(), 1.0)
    expected = k1d / k1d.sum()
    for _ in range(n - 1):
        expected = np.multiply.outer(expected, k1d / k1d.sum())
    assert np.allclose(k, expected)


@pytest.mark.xfail(reason="createNormFrequencyVector has its even/odd branches swapped")
@pytest.mark.parametrize("n", [7, 8, 31, 32])
def test_frequency_vector_matches_centred_fft(n):
    assert np.allclose(to_np(filters.createNormFrequencyVector(n)), np.fft.fftshift(np.fft.fftfreq(n)))


@pytest.mark.xfail(reason="frequency grid is offset by half a bin, so the DC sample is not exactly 0")
@pytest.mark.parametrize("oshape", [(16, 16), (15, 17), (8, 10, 12)])
def test_butterworth_unit_gain_at_dc(xp, device, oshape):
    # sigpy's centred FFT puts DC at index n // 2 along each axis.
    h = to_np(filters.butter(0.25, oshape, n=2, device=device))
    assert h.shape == oshape
    assert np.isclose(h[tuple(s // 2 for s in oshape)], 1.0)


@pytest.mark.parametrize("oshape", [(16, 16), (8, 10, 12)])
def test_butterworth_is_lowpass(xp, device, oshape):
    h = to_np(filters.butter(0.25, oshape, n=2, device=device))
    assert np.all((h > 0) & (h <= 1))
    centre = tuple(s // 2 for s in oshape)
    corner = (0,) * len(oshape)
    assert h[centre] > 0.95 and h[corner] < 0.1
