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
import numpy as np
import sigpy as sp


def symmExtension(N, n):

    """

    # brief Half-sample symmetric boundary symmExtension(    # param N signal length
    # param n requested sample, possibly outside {0,...,N-1}
    # return reflected sample in {0,...,N-1}
    #
    """
    if n < 0:
        n = -1 - n  # Reflect over n = -1/2
    elif n >= N:
        n = 2 * N - 1 - n  # Reflect over n = N - 1/2
    else:
        return n

    return n


def preComputeConstants(sigma, k):
    # Sigma = gaussian stdev, k = number of passes (higher k == greater approximation)
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


def eboxFilter(
    arrIn, r, c1, c2, device=-1, stridein=1, strideout=1,
):
    xp = sp.Device(device).xp
    with sp.Device(device):
        N = arrIn.shape[0]  # Array Sample Size
        assert N > 0 and r >= 0, "Input is empty or radius is negative"

        accum = 0
        for n in range(-r, r + 1):
            accum += arrIn[stridein * symmExtension(N, n)]
        arrOut = xp.zeros(arrIn.shape)
        arrOut[0] = accum = (
            c1
            * (
                arrIn[stridein * symmExtension(N, r + 1)]
                + arrIn[stridein * symmExtension(N, -r - 1)]
            )
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

    return arrOut
