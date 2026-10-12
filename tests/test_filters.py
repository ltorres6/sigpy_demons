"""Gaussian smoothing kernels and the Butterworth anti-aliasing filter."""

import numpy as np
import pytest

from conftest import to_np
from pydemons.filters import butterworth, gaussian_kernel, gaussian_kernel_1d, normalized_frequencies


def kernel_variance(k1d):
    k1d = np.asarray(k1d) / np.sum(k1d)
    x = np.arange(len(k1d)) - len(k1d) // 2
    return float(np.sum(k1d * x**2))


@pytest.mark.parametrize("sigma", [0.5, 1.0, 1.5, 3.0])
def test_gaussian_1d_is_odd_symmetric_and_centred(sigma):
    k = to_np(gaussian_kernel_1d(sigma))
    assert len(k) % 2 == 1 and len(k) >= 3
    assert np.allclose(k, k[::-1])
    assert np.argmax(k) == len(k) // 2


@pytest.mark.parametrize("sigma", [1.0, 1.5, 3.0])
def test_gaussian_1d_variance_matches_sigma(sigma):
    assert np.isclose(kernel_variance(to_np(gaussian_kernel_1d(sigma))), sigma**2, rtol=0.02)


@pytest.mark.parametrize("n", [1, 2, 3])
def test_gaussian_nd_is_normalised_separable_product(xp, n):
    k = gaussian_kernel(1.5, ndim=n, xp=xp)
    assert isinstance(k, xp.ndarray)
    k = to_np(k)
    k1d = to_np(gaussian_kernel_1d(1.5))
    assert k.shape == (len(k1d),) * n
    assert np.isclose(k.sum(), 1.0)
    expected = k1d / k1d.sum()
    for _ in range(n - 1):
        expected = np.multiply.outer(expected, k1d / k1d.sum())
    assert np.allclose(k, expected)


@pytest.mark.parametrize("n", [7, 8, 31, 32])
def test_frequency_vector_matches_centred_fft(n):
    assert np.allclose(to_np(normalized_frequencies(n)), np.fft.fftshift(np.fft.fftfreq(n)))


@pytest.mark.parametrize("oshape", [(16, 16), (15, 17), (8, 10, 12)])
def test_butterworth_unit_gain_at_dc(xp, oshape):
    # sigpy's centred FFT puts DC at index n // 2 along each axis.
    h = to_np(butterworth(0.25, oshape, order=2, xp=xp))
    assert h.shape == oshape
    assert np.isclose(h[tuple(s // 2 for s in oshape)], 1.0)


@pytest.mark.parametrize("oshape", [(16, 16), (8, 10, 12)])
def test_butterworth_is_lowpass(xp, oshape):
    h = to_np(butterworth(0.25, oshape, order=2, xp=xp))
    assert np.all((h > 0) & (h <= 1))
    centre = tuple(s // 2 for s in oshape)
    corner = (0,) * len(oshape)
    assert h[centre] > 0.99 and h[corner] < 0.1
