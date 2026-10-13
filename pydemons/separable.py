"""Separable linear operators applied one axis at a time with matrix multiplies.

A 1D filter or 1D linear resampling along one axis is a small matrix. Applying
it with ``matmul`` lets the array library's BLAS do the work: multi-threaded on
CPU (NumPy) and cuBLAS on GPU (CuPy), from the same code.
"""

from __future__ import annotations

import functools
import math

import numpy as np

from pydemons.backend import get_array_module


def matmul_along(arr, matrix, axis):
    """Apply ``matrix`` (shape ``(m, n)``) along ``axis`` of ``arr`` (length ``n``).

    The result has length ``m`` along ``axis``; other axes are unchanged.
    """
    xp = get_array_module(arr, matrix)
    shape = arr.shape
    n, m = shape[axis], matrix.shape[0]
    before, after = math.prod(shape[:axis]), math.prod(shape[axis + 1 :])
    if after == 1:
        out = xp.matmul(arr.reshape(before, n), matrix.T)
    elif before == 1:
        out = xp.matmul(matrix, arr.reshape(n, after))
    else:
        out = xp.matmul(matrix, arr.reshape(before, n, after))
    return out.reshape(shape[:axis] + (m,) + shape[axis + 1 :])


def correlate_along(arr, kernel, axis, mode="constant", min_block=16, max_block=128):
    """1D correlation of ``arr`` with ``kernel`` (odd length) along ``axis``.

    ``mode`` is ``"constant"`` (zero padding) or ``"reflect"`` (half-sample
    symmetric). The axis is viewed, without copying, as blocks of ``b`` samples
    (``b`` the smallest divisor of the axis length in ``[min_block, max_block]``
    and at least the kernel radius). Each block is multiplied by one ``(b, b)``
    banded matrix, and small ``(r, r)`` products add the samples each block needs
    from its neighbours and, for ``"reflect"``, from the mirrored border. That is
    about ``b + len(kernel)`` multiply-adds per voxel instead of ``n`` for a dense
    ``(n, n)`` matrix. Short axes, or lengths without a suitable divisor, use the
    dense matrix.
    """
    xp = get_array_module(arr)
    n, r = arr.shape[axis], len(kernel) // 2
    b = _block_size(n, max(min_block, r, 1), max_block)
    if b is None or n < 2 * b:
        return matmul_along(arr, band_matrix(xp, n, kernel, mode, arr.dtype), axis)

    before, after = math.prod(arr.shape[:axis]), math.prod(arr.shape[axis + 1 :])
    nb = n // b
    x = xp.ascontiguousarray(arr).reshape((before, nb, b) + ((after,) if after > 1 else ()))
    main, prev, nxt, left, right = _block_matrices(xp, b, tuple(float(v) for v in kernel), np.dtype(arr.dtype).str)

    def apply(mat, v):
        if after > 1:
            return xp.matmul(mat, v)
        # Along the last axis, flatten to one large 2D product instead of many tiny batched ones.
        v = xp.ascontiguousarray(v)
        return xp.matmul(v.reshape(-1, v.shape[-1]), mat.T).reshape(v.shape[:-1] + (mat.shape[0],))

    out = apply(main, x)
    if r:
        out[:, 1:, :r] += apply(prev, x[:, :-1, b - r :])
        out[:, :-1, b - r :] += apply(nxt, x[:, 1:, :r])
        if mode == "reflect":
            out[:, 0, :r] += apply(left, x[:, 0, :r])
            out[:, -1, b - r :] += apply(right, x[:, -1, b - r :])
    return out.reshape(arr.shape)


def _block_size(n, low, high):
    """Smallest divisor of ``n`` in ``[low, high]``, or None."""
    for b in range(low, min(high, n) + 1):
        if n % b == 0:
            return b
    return None


@functools.lru_cache(maxsize=32)
def _block_matrices(xp, b, kernel, dtype):
    """Banded block plus the (r, r) neighbour and mirrored-border corrections."""
    k = len(kernel)
    r = k // 2
    main = np.zeros((b, b))
    for i in range(b):
        for s in range(max(0, i - r), min(b, i + r + 1)):
            main[i, s] = kernel[s - i + r]
    prev, nxt, left, right = (np.zeros((r, r)) for _ in range(4))
    for i in range(r):
        for q in range(r):
            if q >= i:  # output row i, previous block's sample b - r + q
                prev[i, q] = kernel[q - i]
            if q <= i:  # output row b - r + i, next block's sample q
                nxt[i, q] = kernel[q - i + 2 * r]
            if q <= r - 1 - i:  # output row i, mirrored sample -1 - q == sample q
                left[i, q] = kernel[r - 1 - q - i]
            if r - 1 - q <= i:  # output row n - r + i, mirrored sample n + m == sample n - 1 - m, m = r - 1 - q
                right[i, q] = kernel[(r - 1 - q) - i + 2 * r]
    return tuple(xp.asarray(m.astype(dtype)) for m in (main, prev, nxt, left, right))


def band_matrix(xp, n, kernel, mode="constant", dtype="float64"):
    """``(n, n)`` matrix of 1D correlation with ``kernel`` (odd length).

    ``mode="constant"`` zero-pads; ``mode="reflect"`` is half-sample symmetric
    (``d c b a | a b c d | d c b a``). Cached per device, size, kernel and dtype.
    """
    return _band_matrix(xp, n, tuple(float(k) for k in kernel), mode, np.dtype(dtype).str)


@functools.lru_cache(maxsize=32)
def _band_matrix(xp, n, kernel, mode, dtype):
    r = len(kernel) // 2
    mat = np.zeros((n, n))
    for i in range(n):
        for t, w in enumerate(kernel):
            s = i + t - r
            if mode == "reflect":
                while not 0 <= s < n:
                    s = -s - 1 if s < 0 else 2 * n - s - 1
            elif not 0 <= s < n:
                continue
            mat[i, s] += w
    return xp.asarray(mat.astype(dtype))


def linear_resample_matrix(xp, n_in, n_out, factor, dtype="float64"):
    """``(n_out, n_in)`` matrix sampling a 1D signal at ``i / factor``, linearly, edge-clamped."""
    return _linear_resample_matrix(xp, n_in, n_out, float(factor), np.dtype(dtype).str)


@functools.lru_cache(maxsize=32)
def _linear_resample_matrix(xp, n_in, n_out, factor, dtype):
    mat = np.zeros((n_out, n_in))
    for i in range(n_out):
        x = min(max(i / factor, 0.0), n_in - 1.0)
        i0 = min(int(x), max(n_in - 2, 0))
        w = x - i0
        mat[i, i0] += 1 - w
        if w:
            mat[i, i0 + 1] += w
    return xp.asarray(mat.astype(dtype))
