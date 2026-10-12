"""Convolution with a fixed kernel: single FFT, or overlap-add for small kernels."""

from __future__ import annotations

import itertools

from pydemons.backend import get_array_module


def next_fast_len(n):
    """Smallest ``2**a * 3**b * 5**c`` that is at least ``n``: a fast FFT length."""
    best = 1
    while best < n:
        best *= 2
    for p5 in itertools.takewhile(lambda v: v < best, (5**i for i in itertools.count())):
        for p35 in itertools.takewhile(lambda v: v < best, (p5 * 3**j for j in itertools.count())):
            candidate = p35
            while candidate < n:
                candidate *= 2
            best = min(best, candidate)
    return best


class FFTConvolver:
    """Zero-padded linear convolution of arrays of one shape with a fixed kernel.

    The kernel spectrum is computed once, so repeated calls only cost one
    forward and one inverse real FFT. The output has the input's shape, with
    the kernel centred (the ``"same"`` convention of ``scipy.ndimage.convolve``
    with ``mode="constant"``).

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
        self.fft_shape = tuple(next_fast_len(k + s - 1) for k, s in zip(kernel.shape, self.shape))
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
    """Zero-padded convolution with output the shape of ``arr`` and the kernel centred."""
    _check_odd(kernel)
    return FFTConvolver(kernel, arr.shape)(arr)


def oa_convolve(arr, kernel):
    """Overlap-add zero-padded convolution with output the shape of ``arr``.

    ``arr`` is cut into blocks that are transformed together with FFTs sized for
    the kernel rather than for the whole array; cheaper than :func:`convolve`
    when the kernel is much smaller than ``arr``.
    """
    _check_odd(kernel)
    xp = get_array_module(arr, kernel)
    ndim = arr.ndim
    fft_len = tuple(next_fast_len(2 * k) for k in kernel.shape)
    block = tuple(n - k + 1 for n, k in zip(fft_len, kernel.shape))
    nblocks = tuple(-(-s // b) for s, b in zip(arr.shape, block))

    padded = xp.zeros(tuple(n * b for n, b in zip(nblocks, block)), dtype=arr.dtype)
    padded[tuple(slice(0, s) for s in arr.shape)] = arr
    interleaved = tuple(v for pair in zip(nblocks, block) for v in pair)
    order = tuple(range(0, 2 * ndim, 2)) + tuple(range(1, 2 * ndim, 2))
    blocks = padded.reshape(interleaved).transpose(order)  # (nblocks..., block...)

    axes = tuple(range(ndim, 2 * ndim))
    spectrum = xp.fft.rfftn(blocks, s=fft_len, axes=axes) * xp.fft.rfftn(kernel, s=fft_len, axes=tuple(range(ndim)))
    pieces = xp.fft.irfftn(spectrum, s=fft_len, axes=axes)

    full = xp.zeros(tuple(n * b + k - 1 for n, b, k in zip(nblocks, block, kernel.shape)), dtype=pieces.dtype)
    for index in itertools.product(*(range(n) for n in nblocks)):
        full[tuple(slice(i * b, i * b + n) for i, b, n in zip(index, block, fft_len))] += pieces[index]
    return full[tuple(slice(k // 2, k // 2 + s) for k, s in zip(kernel.shape, arr.shape))]
