import sigpy as sp
from filters import butter
from cupyx.scipy import ndimage as cuTools
from scipy import ndimage as npTools


def gaussianPyramidDownsample(arrIn, n, device=-1):
    """
    Use gaussian pyramid to downsample image by n levels
    Should be faster than going straight to downsample factor
    because we keep convolution kernel small.
    Works.
    """
    xp = sp.Device(device).xp
    ishape = arrIn.shape
    if n == 0:
        return arrIn
    # loop through scales
    for ii in range(n):
        oshape = tuple(int(jj // (2**ii)) for jj in ishape)
        k1d = xp.array([1, 4, 6, 4, 1]) / 16  # sigma ~1.0
        k3dx, k3dy, k3dz = xp.ix_(k1d, k1d, k1d)
        k3d = k3dx * k3dy * k3dz
        out = sp.convolve(arrIn, k3d)
        out = sp.resize(out, oshape)
        out = sp.downsample(out, (2,) * arrIn.ndim)
        # Set arrIn for next level
        arrIn = out

    return out


def LaplacianPyramidUpsample(self, I, n):
    """
    Use gaussian pyramid to downsample image by n levels
    Should be faster than going straight to downsample factor
    because we keep convolution kernel small.
    Need to implement
    """
    print("Not implemented")
    # ishape = I.shape
    # if n == 0:
    #     return I
    # # loop through scales
    # for ii in range(n):
    #     oshape = tuple(int(jj * (2 ** (ii + 1))) for jj in ishape)
    #     print(ishape)
    #     print(oshape)

    #     k1d = self.xp.array([1, 4, 6, 4, 1]) / 16  # sigma ~1.0
    #     k3dx, k3dy, k3dz = self.xp.ix_(k1d, k1d, k1d)
    #     k3d = k3dx * k3dy * k3dz
    #     out = sp.upsample(I, oshape, (1.5,) * I.ndim)
    #     out = sp.convolve(out, k3d)
    #     out = sp.convolve(out, k3d)
    #     out = sp.resize(out, oshape) * 4
    #     # Set I for next level
    #     I = out

    # return out


def antialiasResize(arrIn, factor, device=-1):
    """
    % Implement a volumetric resize function that applies a second order
    % low-pass butterworth filter to the input image. The cutoff of the filter
    % is chosen based on the resize scale factor to limit aliasing effects.
    Works for downsampling.
    """
    # xp = sp.Device(device).xp
    if factor == 1:
        return arrIn

    if factor < 1:
        # Move to Frequency domain.
        arrInFFT = sp.fft(arrIn, norm=None)
        # Construct low-pass filter with cutoff based on scale factor.
        filt = butter(0.5 * factor, n=2, oshape=arrInFFT.shape, device=device)
        # Obtain low-pass filtered version of input spatial domain volume

        arrIn = sp.ifft(arrInFFT * filt, norm=None)

        out = sp.downsample(
            arrIn,
        )
        return
    # Apply scale transform of input volume
    # affineMatrix = xp.array([[factor, 0, 0, 0], [0, factor, 0, 0], [0, 0, factor, 0], [0, 0, 0, 1]])

    # out = rescale(arrIn, factor, device)

    return out


def FFTupsample(self, I, oshape, device):
    # Upsampling via FFT zeropadding (upsample and low pass).
    zFactor = oshape[0] / I.shape[0]
    out = sp.fft(I, norm=None)
    out = sp.resize(out, oshape)
    out = self.xp.real(sp.ifft(out, norm=None) * zFactor)
    return out


def zoom(arrIn, factor, device=-1):
    """
    % Implement a volumetric resize function that applies a second order
    % low-pass butterworth filter to the input image. The cutoff of the filter
    % is chosen based on the resize scale factor to limit aliasing effects.
    Works for downsampling.
    """
    xp = sp.Device(device).xp
    if factor == 1:
        return arrIn

    if factor < 1:
        # print("Downsampling by {}".format(1 / factor))
        # Move to Frequency domain.
        arrInFFT = sp.fft(arrIn, norm=None)
        # Construct low-pass filter with cutoff based on scale factor.
        filt = butter(0.5 * factor, n=2, oshape=arrInFFT.shape, device=device)
        # Obtain low-pass filtered version of input spatial domain volume

        arrIn = sp.ifft(arrInFFT * filt * filt, norm=None)
        factors = tuple(int(1 // factor) for ii in range(len(arrIn.shape)))
        out = sp.downsample(arrIn, factors)
        return xp.real(out)
    else:
        if device == -1:
            out = npTools.zoom(arrIn, factor, order=1)
        else:
            out = cuTools.zoom(arrIn, factor, order=1)

    return out
