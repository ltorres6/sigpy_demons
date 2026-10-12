"""FFT convolution with a fixed kernel."""

from __future__ import annotations

from scipy.fft import next_fast_len

from pydemons.backend import get_array_module, get_ndimage, get_signal


class FFTConvolver:
    """Zero-padded linear convolution of arrays of one shape with a fixed kernel.

    The kernel spectrum is computed once, so repeated calls only cost one
    forward and one inverse real FFT. The output has the input's shape and
    matches ``scipy.ndimage.convolve(arr, kernel, mode="constant")``.

    Parameters
    ----------
    kernel : ndarray
        Real kernel with odd length along every axis. Its device sets the device.
    shape : tuple of int
        Shape of the arrays that will be convolved.
    """

    def __init__(self, kernel, shape):
        self.xp = get_array_module(kernel)
        self.shape = tuple(shape)
        self.fft_shape = tuple(next_fast_len(k + s - 1, real=True) for k, s in zip(kernel.shape, self.shape))
        self.crop = tuple(slice(k // 2, k // 2 + s) for k, s in zip(kernel.shape, self.shape))
        self.axes = tuple(range(len(self.shape)))
        self.kernel_fft = self.xp.fft.rfftn(kernel, s=self.fft_shape, axes=self.axes)

    def __call__(self, arr):
        xp = self.xp
        spectrum = xp.fft.rfftn(arr, s=self.fft_shape, axes=self.axes) * self.kernel_fft
        out = xp.fft.irfftn(spectrum, s=self.fft_shape, axes=self.axes)
        return out[self.crop]


def _check_odd(kernel):
    if any(k % 2 == 0 for k in kernel.shape):
        raise ValueError(f"kernel must have odd length along every axis, got {kernel.shape}")


def convolve(arr, kernel):
    """Direct zero-padded convolution with output the shape of ``arr``. Best for small kernels."""
    _check_odd(kernel)
    return get_ndimage(get_array_module(arr, kernel)).convolve(arr, kernel, mode="constant", cval=0.0)


def oa_convolve(arr, kernel):
    """Overlap-add zero-padded convolution with output the shape of ``arr``.

    Faster than a single large FFT when the kernel is much smaller than ``arr``.
    """
    _check_odd(kernel)
    return get_signal(get_array_module(arr, kernel)).oaconvolve(arr, kernel, mode="same")
