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


# --------------------------------------------------------------------------- extended box Gaussian


@pytest.mark.parametrize("sigma", [1.0, 2.5, 4.0])
@pytest.mark.parametrize("passes", [3, 4])
def test_extended_box_impulse_response_has_exact_variance(xp, sigma, passes):
    # Gwosdek et al. choose the box width and edge weight so the variance is exactly sigma**2.
    from pydemons.filters import extended_box_gaussian

    impulse = np.zeros(201)
    impulse[100] = 1.0
    h = to_np(extended_box_gaussian(xp.asarray(impulse), sigma, passes=passes))
    x = np.arange(201) - 100
    assert np.isclose(h.sum(), 1.0)
    assert np.isclose((h * x**2).sum(), sigma**2, rtol=1e-9)
    assert np.allclose(h, h[::-1])


@pytest.mark.parametrize("shape", [(64, 48), (24, 20, 16)])
def test_extended_box_approximates_gaussian_filter(xp, shape):
    from scipy.ndimage import gaussian_filter

    from pydemons.filters import extended_box_gaussian

    arr = np.random.default_rng(0).random(shape)
    out = to_np(extended_box_gaussian(xp.asarray(arr), 2.0, passes=4))
    assert out.shape == shape
    assert np.abs(out - gaussian_filter(arr, 2.0, mode="reflect")).max() < 0.01


def test_extended_box_preserves_constant_and_mean(xp):
    from pydemons.filters import extended_box_gaussian

    assert np.allclose(to_np(extended_box_gaussian(xp.full((30, 30), 2.5), 3.0)), 2.5)
    arr = np.random.default_rng(1).random((40, 40))
    assert np.isclose(to_np(extended_box_gaussian(xp.asarray(arr), 3.0)).mean(), arr.mean())


def test_extended_box_rejects_bad_sigma():
    from pydemons.filters import extended_box_gaussian

    with pytest.raises(ValueError):
        extended_box_gaussian(np.zeros(10), 0.0)


# --------------------------------------------------------------------------- unsharp mask


def test_unsharp_mask_keeps_constant_and_boosts_detail(xp):
    from pydemons.filters import unsharp_mask

    assert np.allclose(to_np(unsharp_mask(xp.full((20, 20), 1.5), sigma=1.0, amount=1.0)), 1.5)
    arr = np.random.default_rng(2).random((40, 40))
    out = to_np(unsharp_mask(xp.asarray(arr), sigma=1.0, amount=1.0))
    assert np.isclose(out.mean(), arr.mean(), atol=1e-3)
    assert out.std() > arr.std()


# --------------------------------------------------------------------------- separable smoothing


@pytest.mark.parametrize("sigma", [0.5, 1.5, 3.0])
@pytest.mark.parametrize("shape", [(30, 40), (12, 14, 10)])
def test_gaussian_smooth_matches_nd_kernel_convolution(xp, sigma, shape):
    from scipy.ndimage import convolve

    from pydemons.filters import gaussian_smooth

    arr = np.random.default_rng(0).standard_normal(shape)
    out = to_np(gaussian_smooth(xp.asarray(arr), sigma))
    expected = convolve(arr, to_np(gaussian_kernel(sigma, ndim=len(shape))), mode="constant")
    assert np.allclose(out, expected, atol=1e-12)


def test_gaussian_smooth_axes_leave_batch_axis_alone(xp):
    from pydemons.filters import gaussian_smooth

    field = np.random.default_rng(1).standard_normal((3, 10, 12, 8))
    batched = to_np(gaussian_smooth(xp.asarray(field), 1.0, axes=(1, 2, 3)))
    each = np.stack([to_np(gaussian_smooth(xp.asarray(c), 1.0)) for c in field])
    assert np.allclose(batched, each, atol=1e-12)


@pytest.mark.parametrize("axes", [None, (1, 2), (1,), (0, 1, 2)])
def test_gaussian_smooth_out_and_in_place(xp, axes):
    from pydemons.filters import gaussian_smooth

    arr = xp.asarray(np.random.default_rng(3).random((3, 70, 40)))
    expected = to_np(gaussian_smooth(arr, 1.5, axes=axes))
    out = xp.full(arr.shape, np.nan)
    assert gaussian_smooth(arr, 1.5, axes=axes, out=out) is out
    assert np.allclose(to_np(out), expected)
    in_place = arr.copy()
    assert gaussian_smooth(in_place, 1.5, axes=axes, out=in_place) is in_place
    assert np.allclose(to_np(in_place), expected)
