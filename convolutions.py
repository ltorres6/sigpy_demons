# Port from scipy for use on GPU.
import sigpy as sp
import numpy as np
from scipy import ndimage as npTools
from cupyx.scipy import ndimage as cpTools


def convolve(filt, arrIn, device=-1):
    if device == -1:
        arrOut = npTools.convolve(arrIn, filt, mode="constant", cval=0.0)
    else:
        arrOut = cpTools.convolve(arrIn, filt, mode="constant", cval=0.0)
    return arrOut


def oaconvolve(filt, arrIn, batchSize=None, device=-1):
    """
    1D Convolve arrIn with filt. Should be fast(er than other methods) for filters of very different sizes
    Filtering uses the overlap-add method converting both `arrIn` and `filt`
    into frequency domain first.  The FFT size is determined as the
    next higher power of 2 of twice the length of `filt`.
    Parameters
    ----------
    filt : one-dimensional numpy array
        The impulse response of the filter
    arrIn : one-dimensional numpy array
        Signal to be filtered

    Returns
    -------
    y : array
        The output of the digital filter.
    """
    xp = sp.Device(device).xp
    L_I = filt.shape[0]
    # Find power of 2 larger that 2*L_I (using bitshifts)
    L_F = 2 << (L_I - 1).bit_length()
    L_S = L_F - L_I + 1
    L_sig = arrIn.shape[0]
    offsets = range(0, L_sig, L_S)

    # handle complex or real input
    if xp.iscomplexobj(filt) or xp.iscomplexobj(arrIn):
        fft_func = xp.fft.fft
        ifft_func = xp.fft.ifft
        res = xp.zeros(L_sig + L_F, dtype=xp.complex128)
    else:
        fft_func = xp.fft.rfft
        ifft_func = xp.fft.irfft
        res = xp.zeros(L_sig + L_F)

    FDir = fft_func(filt, n=L_F)

    # overlap add
    for n in offsets:
        res[n : n + L_F] += ifft_func(fft_func(arrIn[n : n + L_S], n=L_F) * FDir)
    if batchSize is not None:
        res[: batchSize.shape[0]] = res[: batchSize.shape[0]] + batchSize
        return res[:L_sig], res[L_sig:]
    else:
        return res[int(L_I // 2) : L_sig + int(L_I // 2)]


def oaconvolve3(filt, arrIn, batchSize=None, device=-1):
    # Filter should be 1D filter.
    # arrIn will be flattened and indexed as contigiuous array.
    xp = sp.Device(device).xp
    ishape = arrIn.shape
    arrIn = arrIn.ravel()
    arrOut = xp.zeros(arrIn.shape)
    for z in range(ishape[2]):
        for y in range(ishape[1]):
            # arrOut[:, y, z] = oaconvolve(filt, arrIn[:, y, z], device=device)
            idxin = [0, y, z]
            idx = np.ravel_multi_index(idxin, (ishape))
            arrOut[idx] = oaconvolve(filt, arrIn[idx : idx + ishape[0]], device=device)

    arrIn = arrOut.copy()
    arrOut = xp.zeros(ishape)
    for z in range(ishape[2]):
        for x in range(ishape[0]):
            # arrOut[x, :, z] = oaconvolve(filt, arrIn[x, :, z], device=device)
            idxin = np.array([0, y, z], order="C")
            idx = np.ravel_multi_index(idxin, (ishape))
            arrOut[idx] = oaconvolve(filt, arrIn[idx : idx + ishape[0]], device=device)

    arrIn = arrOut.copy()
    arrOut = xp.zeros(ishape)
    for y in range(ishape[1]):
        for x in range(ishape[0]):
            # arrOut[x, y, :] = oaconvolve(filt, arrIn[x, y, :], device=device)
            idxin = np.array([0, y, z], order="C")
            idx = np.ravel_multi_index(idxin, (ishape))
            arrOut[idx] = oaconvolve(filt, arrIn[idx : idx + ishape[0]], device=device)

    # arrIn = arrIn.reshape(ishape)
    return arrOut


def toepelitzConvolve():
    # Toepelitz convolution here.
    print("Nope.")


def FFTConvolve(filt, arrIn, shape, device=-1):
    """
    Convolves filt (Window in fourier space) with arrIn with fft method.
    """
    oshape = arrIn.shape
    arrInFFT = sp.fft(arrIn, oshape=shape, norm=None)
    out = sp.ifft(arrInFFT * filt, norm=None)
    return sp.resize(out, oshape)


# def FFTConvolve(filt, arrIn, device=-1):
#     # Need to fix
#     xp = sp.Device(device).xp
#     ishape = arrIn.shape
#     if filt.ndim == arrIn.ndim == 0:  # scalar inputs
#         return filt * arrIn
#     elif not filt.ndim == arrIn.ndim:
#         raise ValueError("Dimensions do not match.")
#     elif filt.size == 0 or arrIn.size == 0:  # empty arrays
#         return xp.array([])
#     ishape = arrIn.shape
#     s1 = xp.asarray(filt.shape)
#     s2 = xp.asarray(arrIn.shape)

#     if xp.iscomplexobj(filt) or xp.iscomplexobj(arrIn):
#         fft_func = xp.fft.fft
#         ifft_func = xp.fft.ifft
#     else:
#         fft_func = xp.fft.rfft
#         ifft_func = xp.fft.irfft

#     shape = s1 + s2 - 1
#     fsize = int(2 ** xp.ceil(xp.log2(shape)))
#     fslice = tuple([slice(0, int(sz)) for sz in shape])
#     ret = ifft_func(fft_func(filt, fsize) * fft_func(arrIn, fsize))[fslice].copy()
#     return sp.resize(ret, ishape)


# def FFTconvolve3(filt, arrIn, device=-1):
#     # Filter should be 1D filter.
#     # arrIn will be flattened and indexed as contigiuous array.
#     xp = sp.Device(device).xp
#     ishape = arrIn.shape
#     # print(ishape)
#     arrOut = xp.zeros(ishape)
#     for z in range(ishape[2]):
#         for y in range(ishape[1]):
#             arrOut[:, y, z] = FFTConvolve(filt, arrIn[:, y, z], device=device)

#     # arrIn = arrOut.copy()
#     # arrOut = xp.zeros(ishape)
#     for z in range(ishape[2]):
#         for x in range(ishape[0]):
#             arrOut[x, :, z] = FFTConvolve(filt, arrOut[x, :, z], device=device)

#     # arrIn = arrOut.copy()
#     # arrOut = xp.zeros(ishape)
#     for y in range(ishape[1]):
#         for x in range(ishape[0]):
#             arrOut[x, y, :] = FFTConvolve(filt, arrOut[x, y, :], device=device)

#     # arrIn = arrIn.reshape(ishape)
#     return arrOut
