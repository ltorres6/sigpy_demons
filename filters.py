import numpy as np
import sigpy as sp


def gaussianKernel1d(sigma=1.0, device=-1):
    # Generates Gaussian Kernel. Kernel truncated at 4 * sigma
    # Round to next odd integer
    xp = sp.Device(device).xp
    N = 4 * sigma
    N = xp.ceil(N) // 2 * 2 + 1
    if N < 3.0:
        N = 3.0
    # print(N)
    x = xp.linspace(0, N - 1, int(N))
    alpha = 0.5 * (N - 1) / sigma
    k1d = xp.exp(-0.5 * (alpha * (x - (N - 1) / 2) / ((N - 1) / 2)) ** 2)
    return k1d


# def gaussianKernel1dFFT(sigma=1.0, f, device=-1):
#     # Generates Gaussian Kernel with stddev and input frequency vector f.
#     # sigma needs to be reciprocal
#     # NOT WORKING YET
#     xp = sp.Device(device).xp
#     N = f.shape[0]
#     alpha = 0.5 * (N - 1) / sigma
#     k1d = xp.exp(-0.5 * (alpha * (f - (N - 1) / 2) / ((N - 1) / 2)) ** 2)
#     return k1d


def gaussianKernel3d(sigma=1.0, device=-1):
    xp = sp.Device(device).xp
    if xp.array(sigma).size == 1:
        k1d = gaussianKernel1d(sigma)
        k3dx, k3dy, k3dz = xp.ix_(k1d, k1d, k1d)
        k3d = k3dx * k3dy * k3dz
    else:
        k1d = gaussianKernel1d(sigma[0])
        k2d = gaussianKernel1d(sigma[1])
        k3d = gaussianKernel1d(sigma[2])
        k3dx, k3dy, k3dz = xp.ix_(k1d, k2d, k3d)
        k3d = k3dx * k3dy * k3dz
    # Normalize (Unless 1D already normalized)
    k3d = k3d / xp.sum(k3d)
    return k3d


# Source: A Survey of Gaussian Convolution Algorithms -- Pascal Getreuer
# Other References:
#  Wells, W.M.: Efficient synthesis of Gaussian filters by cascaded uniform filters.
#  IEEE Transactions on Pattern Analysis and Machine Intelligence 8(2), 234--239
#  (Mar 1986)

#  R. Rau,  J.H. McClellan, "Efficient approximation of Gaussian filters,"
#  IEEE Transactions on Signal Processing, Volume 45, Issue 2, February 1997,
#  Page 468-471.

#  Peter Kovesi: Fast Almost-Gaussian Filtering. DICTA 2010: 121-125

#  Pascal Gwosdek, Sven Grewenig, Andr\'es Bruhn, and Joachim Weickert,
#  "Theoretical Foundations of Gaussian Convolution by Extended Box Filtering,"
#  SSVM 2011, 447-458.

#  "Efficient and Accurate Gaussian Image Filtering Using Running Sums"
#  Elhanan Elboher, Michael Werman, arXiv:1107.4958

# from numba import njit


def symmExtension(N, n):

    """
    Half-sample symmetric boundary symmExtension(    # param N signal length
    param n requested sample, possibly outside {0,...,N-1}
    return reflected sample in {0,...,N-1}
    """
    if n < 0:
        n = -1 - n  # Reflect over n = -1/2
    elif n >= N:
        n = 2 * N - 1 - n  # Reflect over n = N - 1/2
    else:
        return n

    return n


def preComputeConstants(sigma, k):
    """
    Compute kernel size and interpolation weights for a given gaussian
    standard deviation sigma and number of filter passes.
    Sigma = gaussian stdev, k = number of passes
    (higher k == greater approximation accuracy to true gaussian)
    """
    r = np.floor(0.5 * np.sqrt((12.0 * sigma * sigma) / k + 1.0) - 0.5)
    alpha = (
        (2 * r + 1)
        * (r * (r + 1) - 3.0 * sigma * sigma / k)
        / (6.0 * (sigma * sigma / k - (r + 1) * (r + 1)))
    )
    c1 = alpha / (2.0 * (alpha + r) + 1)
    c2 = (1.0 - alpha) / (2.0 * (alpha + r) + 1)
    print("Effective Kernel Size: {}".format((2.0 * (alpha + r) + 1)))
    return int(r), alpha, c1, c2


def eboxKernel(r, alpha, c1, c2, device=-1):
    """
    return a 1D eboxKernel
    1D extended box filter (strided).
    Applies symmetric boundary conditions at edges.
    arrIn is 1D array you wish to filter.
    r1,c1,c2 are all computed using preComputeConstants()
    """
    xp = sp.Device(device).xp
    kernel = xp.ones((r,))
    kernel = xp.concatenate([[alpha], kernel, [alpha]])
    return kernel / (2.0 * (alpha + r) + 1)


def eboxFilter(
    arrIn, r, c1, c2, device=-1, stridein=1, strideout=1,
):
    """
    1D extended box filter (strided).
    Applies symmetric boundary conditions at edges.
    arrIn is 1D array you wish to filter.
    r1,c1,c2 are all computed using preComputeConstants()
    """
    xp = sp.Device(device).xp
    N = arrIn.shape[0]  # Array Sample Size

    assert N > 0 and r >= 0, "Input is empty or radius is negative"

    accum = 0
    for n in range(-r, r + 1):
        accum += arrIn[stridein * symmExtension(N, n)]
    arrOut = xp.zeros(arrIn.shape)
    arrOut[0] = accum = (
        c1
        * (arrIn[stridein * symmExtension(N, r + 1)] + arrIn[stridein * symmExtension(N, -r - 1)])
        + (c1 + c2) * accum
    )

    for n in range(1, N):
        accum += c1 * (
            arrIn[stridein * symmExtension(N, n + r + 1)]
            - arrIn[stridein * symmExtension(N, n - r - 2)]
        ) + c2 * (
            arrIn[stridein * symmExtension(N, n + r)]
            - arrIn[stridein * symmExtension(N, n - r - 1)]
        )
        arrOut[strideout * n] = accum

    # return sp.resize(arrOut, (N,))
    return arrOut


def eboxGaussianConv(arrIn, r, c1, c2, k, stride=1, device=-1):
    """
    Approximates Gaussian Filtering with multiple passes of box filter.
    """
    #  Could probably do in strides to be cache friendly.
    for ii in range(k):
        arrIn = eboxFilter(arrIn, r, c1, c2, device=device)
    return arrIn


def eboxGaussianConv2D(imIn, r, c1, c2, k, stride=1, device=-1):
    """
    Approximates Gaussian Filtering with multiple passes of box filter.
    Applies box filter to each dimension separately.
    Should be O(n) complexity (fast!)
    """
    # xp = sp.Device(device).xp

    N = imIn.shape  # Array Sample Size
    assert imIn.ndim == 2, "Input is empty or radius is negative"
    h = N[0]
    w = N[1]
    # d = N[2]
    #  Could probably do in strides to be cache friendly... a problem for another day.
    # Convolve along x, then y
    # imOut = xp.zeros(imIn.shape)
    for ii in range(h):
        imIn[ii, :] = eboxGaussianConv(imIn[ii, :], r, c1, c2, k, device=device)
    for ii in range(w):
        imIn[:, ii] = eboxGaussianConv(imIn[:, ii], r, c1, c2, k, device=device)
    return imIn


def eboxGaussianConv3D(volIn, r, c1, c2, k, stride=1, device=-1):
    """
    Approximates Gaussian Filtering with multiple passes of box filter.
    Applies box filter to each dimension separately.
    Should be O(n) complexity (fast!)
    """
    with sp.Device(device):
        N = volIn.shape  # Array Sample Size
        assert volIn.ndim == 3, "Input is empty or radius is negative"
        w = N[0]
        h = N[1]
        d = N[2]
        #  Could probably do in strides to be cache friendly... a problem for another day.
        # Convolve along 3rd dim.
        # Flatten array for speed?
        vol = volIn.ravel()
        # for ii in range(d):
        #     volIn[:, :, ii] = eboxGaussianConv2D(volIn[:, :, ii], r, c1, c2, k)
        for x in range(w):
            for y in range(h):
                idx = np.ravel_multi_index([x, y], (w, h))
                print(idx)
                # vol[idx] = eboxGaussianConv(vol[idx], r, c1, c2, k, device=device)
    return volIn


def createNormFrequencyVector(N, device=-1):
    xp = sp.Device(device).xp
    if N % 2 == 0:
        u = xp.linspace(-0.5 + 1 / (2 * N), 0.5 - 1 / (2 * N), N)
    else:
        u = xp.linspace(-0.5, 0.5 - 1 / N, N)
    return u


def butter(D0, oshape, n=2, device=-1):
    """
    D0 frequency (jw_c).
    n is number of poles in filter.
    """
    xp = sp.Device(device).xp
    x, y, z = oshape
    # Define normalized frequency mesh grid
    u = createNormFrequencyVector(y, device=device)
    v = createNormFrequencyVector(x, device=device)
    w = createNormFrequencyVector(z, device=device)
    [U, V, W] = xp.meshgrid(u, v, w)

    D = xp.sqrt(U ** 2 + V ** 2 + W ** 2)

    # Transfer function
    H = 1 / (1 + (D / D0) ** (2 * n))
    return H
