import sigpy as sp
import sigpy.plot as plt
from tqdm.auto import tqdm

# import numpy as np
# from scipy.fft import next_fast_len


def jacobian(I):
    print("To Do")


def find_update_field(Is, Im, T, sigma_x):
    xp = sp.get_array_module(Is)
    # Warp image
    ImWarped = imwarp(Im, T)
    # Calculate directional gradients
    fd = sp.linop.FiniteDifference(Is.shape)
    # dS = fd(Is)
    # dM = fd(ImWarped)
    J = 0.5 * (fd(Is) + fd(ImWarped))
    Idiff = Is - ImWarped
    # show(Idiff ** 2, "Idiff^2", "gray")
    # show(dS, "FD of Static", "gray")
    # show(dM, "FD of Moving", "gray")
    # Symmetric Formulation (Wang 2005)
    # V = (Idiff / (dS[0] ** 2 + dS[1] ** 2 + dS[2] ** 2 + Idiff ** 2 / sigma_x ** 2)) * dS + (
    #     Idiff / (dM[0] ** 2 + dM[1] ** 2 + dM[2] ** 2 + Idiff ** 2 / sigma_x ** 2)
    # ) * dM
    V = (Idiff / (J[0] ** 2 + J[1] ** 2 + J[2] ** 2 + Idiff ** 2 / sigma_x ** 2)) * J

    # Generate mask for null values (convert to zeros)
    mask = xp.zeros(V.shape, dtype=bool)
    for j in range(3):
        mask[j] = (
            (~xp.isfinite(V[j]))
            | (xp.abs(Idiff) < 1e-2)
            # | (dS[0] ** 2 + dS[1] ** 2 + dS[2] ** 2 + Idiff ** 2 / sigma_x ** 2 < 1e-9)
            # | (dM[0] ** 2 + dM[1] ** 2 + dM[2] ** 2 + Idiff ** 2 / sigma_x ** 2 < 1e-9)
            | (J[0] ** 2 + J[1] ** 2 + J[2] ** 2 + Idiff ** 2 / sigma_x ** 2 < 1e-9)
        )
    V[mask] = 0
    return V


def imwarp(I_in, T):
    # This function warps I_in to coordinates defined by coord + T
    # Interpolate using sigpy. param = 1 is linear.
    xp = sp.get_array_module(I_in)
    nx, ny, nz = I_in.shape
    coord = xp.mgrid[0:nx, 0:ny, 0:nz].astype("float64")
    I_out = sp.interpolate(
        I_in, xp.moveaxis(coord, 0, -1) + xp.moveaxis(T, 0, -1), kernel="spline", width=2, param=1,
    )
    return I_out


def expfield(F):
    # Scaling and Squaring Algorithm iLogDemons Paper
    # Find n, scaling parameter
    xp = sp.get_array_module(F)
    n = 2 * xp.max(xp.sqrt(xp.square(F[0]) + xp.square(F[1]) + xp.square(F[2])))
    # n big enough so max(v * 2^-n) < 0.5 pixel)
    if ~xp.isfinite(n) or n == 0:
        n = 0
    else:
        n = xp.ceil(xp.log2(n))
        n = int(max(n, 0))  # avoid nulls

    # 1) Scaling Step (Scale towards 0 for efficient computation) # Compute exp(F/2^N)?
    scale = 2 ** -n
    F *= scale
    # print(n)
    # 2) square it n times to get exp(F)
    for _ in range(n):
        F = compose(F, F)

    return F, n


def compose(A, B):
    # This function composes two fields by interpolating field B to position (x + A(x), y+A(y), z+A(z)) then summing.
    xp = sp.get_array_module(A)
    _, nx, ny, nz = A.shape
    coord = xp.mgrid[0:nx, 0:ny, 0:nz].astype("float64")

    # Interpolate B at (coord + A)
    for i in range(3):
        B[i] = sp.interpolate(
            B[i],
            xp.moveaxis(coord, 0, -1) + xp.moveaxis(A, 0, -1),
            kernel="spline",
            width=2,
            param=1,
        )
    # Compose
    A += B

    return A


def show(I_in, title="Generic Title", colormap="hot", orientation=0):
    xp = sp.get_array_module(I_in)
    if orientation == 0:
        plt.ImagePlot(
            xp.flip(I_in, axis=I_in.ndim - 3),
            x=I_in.ndim - 1,
            y=I_in.ndim - 3,
            title=title,
            mode="r",
            colormap=colormap,
            interpolation=None,
        )


def gaussian_kernel_1d(sigma=1.0):
    # Generates Gaussian Kernel. Kernel truncated at 4 * sigma
    xp = sp.get_array_module(sigma)
    # Round to next odd integer
    N = 4 * sigma
    N = xp.ceil(N) // 2 * 2 + 1
    if N < 3.0:
        N = 3.0
    # print(N)
    x = xp.linspace(0, N - 1, int(N))
    alpha = 0.5 * (N - 1) / sigma
    k1d = xp.exp(-0.5 * (alpha * (x - (N - 1) / 2) / ((N - 1) / 2)) ** 2)
    return k1d


def gaussian_kernel_3d(sigma=1.0):
    xp = sp.get_array_module(sigma)
    if xp.array(sigma).size == 1:
        k1d = gaussian_kernel_1d(sigma)
        k3dx, k3dy, k3dz = xp.ix_(k1d, k1d, k1d)
        k3d = k3dx * k3dy * k3dz
    else:
        k1d = gaussian_kernel_1d(sigma[0])
        k2d = gaussian_kernel_1d(sigma[1])
        k3d = gaussian_kernel_1d(sigma[2])
        k3dx, k3dy, k3dz = xp.ix_(k1d, k2d, k3d)
        k3d = k3dx * k3dy * k3dz
    # Normalize (Unless 1D already normalized)
    k3d = k3d / xp.sum(k3d)
    return k3d


def calcPad(ishape, n, p):
    # This function pads image.
    oshape = []
    for ii in ishape:
        r = 2 ** n - (ii + p) % (2 ** n)
        oshape.append((ii + p) + r)
    return tuple(oshape)


def gaussianPyramid(I, n):
    """
    Use gaussian pyramid to downsample image by n levels
    Should be faster than going straight to downsample factor
    because we keep convolution kernel small.
    """
    xp = sp.get_array_module(I)
    ishape = I.shape
    if n == 0:
        return I
    # loop through scales
    for ii in range(n):
        oshape = tuple(int(jj // (2 ** ii)) for jj in ishape)
        k1d = xp.array([1, 4, 6, 4, 1]) / 16
        k3dx, k3dy, k3dz = xp.ix_(k1d, k1d, k1d)
        k3d = k3dx * k3dy * k3dz
        out = sp.convolve(I, k3d)
        out = sp.resize(out, oshape)
        out = sp.downsample(out, (2,) * I.ndim)
        # Set I for next level
        I = out

    return out


def gaussianBlur(I, sigma):
    print("Not Implemented Yet")
    # This function uses box blurring to approximate a gaussian blur with std dev sigma.
    # Pascal Getreuer,Gwosdek, Grewenig, Bruhn, and Weickert
    # Implementation is done as an acuumulator, should have O(n) complexity.
    # Other options: direct convolution O(N**2), FFT convolution O(NlogN)
    # simga**2 = (1/12)* K * [(2*r+1)**2 -1]
    # k = 3  # Number of passes, more passes = better approximation
    # r = 0.5 * xp.sqrt(sigma ** 2 * 12 / k + 1) - 0.5
    # alpha = (2 * r + 1) / 6 * (r * (r + 1) - 3 / k * sigma ** 2) / (sigma ** 2 / k - (r + 1) ** 2)
    # c1 = alpha / (1 * alpha + 2 * r + 1)
    # c2 = (1 - alpha) / (2 * alpha + 2 * r + 1)

    # out = I
    # for ii in range(k):
    #     s = (c1 + c2)*np.sum(out)
    #         for jj in


def convolveOverlapSave1d(I, filt):
    # I = 1d array from image, filt = Filter
    xp = sp.get_array_module(I)
    N = I.shape[0]  # Array length
    M = filt.shape[0]  # filter length
    overlap = M - 1
    S = 8 * overlap  # Stride - Not optimal. Should optimize.
    step_size = S - overlap
    F = sp.fft(filt, oshape=(S,))  # Padded filter FFT
    out = xp.zeros(N + M - 1)
    jj = 0
    while jj + S <= N:
        tout = xp.real(sp.ifft((sp.fft(I[jj + xp.arange(S)]) * F)))
        out[jj + xp.arange(step_size - 1)] = tout[M:S]  # (discard M−1 y-values)
        jj = jj + step_size
    out = sp.resize(out, (N,))
    return out


def convolveOS3d(I, filt):
    xp = sp.get_array_module(I)
    out = xp.zeros(I.shape)
    for ii in range(I.shape[0]):
        for jj in range(I.shape[1]):
            out[ii, jj, :] = convolveOverlapSave1d(I[ii, jj, :], filt)
    for ii in range(I.shape[0]):
        for jj in range(I.shape[2]):
            out[ii, :, jj] = convolveOverlapSave1d(I[ii, :, jj], filt)
    for ii in range(I.shape[1]):
        for jj in range(I.shape[2]):
            out[:, ii, jj] = convolveOverlapSave1d(I[:, ii, jj], filt)


def convolveFFT(I1, I2):
    # FFT convolution O(NlogN) Should pad with symetric image for proper periodic convolution
    xp = sp.get_array_module(I1)

    I1 = sp.fft(I1, norm=None)
    # I2 = sp.fft(I2, oshape=I1.shape, norm=None)
    I2 = sp.resize(I2, I1.shape)
    out = xp.real(sp.ifft(I1 * I2, norm=None))
    return out


def downsample(I, oshape, device):
    # Filter then decimate (can do in either frequency domain or spatial)
    xp = sp.Device(device).xp
    zFactor = I.shape[0] / oshape[0]
    # zFactor might need a factor of 2 somewhere in filterSize
    filterSize = [int(round(i / (2 * zFactor))) for i in (I.shape)]
    out = sp.fft(I, norm=None)
    W = sp.hanning(filterSize, device=device)
    out *= sp.resize(W, out.shape)
    out = sp.resize(out, oshape)
    out = xp.abs(sp.ifft(out, norm=None))
    # plt.ImagePlot(I)
    # plt.ImagePlot(out)
    # deblur?
    # filt_inv = gaussian_kernel_3d(5, 0.3, inverse="True")
    # imgOut = sp.resize(sp.conv.convolve(imgOut, filt_inv), imgOut.shape)
    return out


def upsample(I, oshape, device):
    # Upsampling via FFT zeropadding (upsample and low pass).
    xp = sp.Device(device).xp
    zFactor = oshape[0] / I.shape[0]
    out = sp.fft(I, norm=None)
    out = sp.resize(out, oshape)
    out = xp.real(sp.ifft(out, norm=None) * zFactor)
    return out


def energy(I1, I2, sigma_x):
    # E_corr s_opt
    # I1 should be Idiff, I2 should be correspondence field
    xp = sp.get_array_module(I1)
    E = xp.linalg.norm(I2) ** 2 / (sigma_x ** 2)
    #  xp.sum(xp.linalg.norm(I1) ** 2 / xp.sum(I1) +
    return E / 2


def run(
    Is,
    Im,
    iterations=[50, 50, 20],
    sigmaFluids=[1.0, 1.0, 1.0],
    sigmaDiffs=[1.0, 1.0, 1.0],
    sigmaSmooths=[1.0, 1.0, 1.0],
    stepSizes=[0.5, 0.5, 0.5],
    diffeomorphic=True,
    device=-1,
    return_warped=True,
):
    # Move to selected device
    Is = sp.to_device(Is, device)
    Im = sp.to_device(Im, device)

    # Get corresponding matrix math module
    xp = sp.Device(device).xp

    # Normalize. Important for finite differences.
    # Important for not updating field if Idiff < certain value
    Im = xp.abs(Im) / xp.max(xp.abs(Im))
    Is = xp.abs(Is) / xp.max(xp.abs(Is))

    nLevels = len(iterations)
    scales = [2 ** (nLevels - l - 1) for l in range(nLevels)]

    originalShape = Is.shape
    paddedShape = calcPad(originalShape, nLevels, 5)
    Is = sp.resize(Is, paddedShape)
    Im = sp.resize(Im, paddedShape)
    finalV = xp.zeros((3,) + paddedShape)

    # Print some info:
    if diffeomorphic:
        print("Diffeomorphic Flag is On! ...")
    print("Original Image Shape: {} ...".format(originalShape))
    print("Padded Image Shape: {} ...".format(paddedShape))
    print("Image Scaling Factors: {} ...".format(scales))
    print("Number of Levels: {} ...".format(nLevels))
    print("Iterations per Level: {} ...".format(iterations))
    print("Fluid Sigmas: {} ...".format(sigmaFluids))
    print("Diffusion Sigmas: {} ...".format(sigmaDiffs))
    print("Downsample Smooth Sigmas: {} ...".format(sigmaSmooths))
    print("Iteration Step Sizes: {} pixels ...".format([i ** 2 for i in stepSizes]))

    # Begin
    loss = []
    for l in range(nLevels):
        scale = scales[l]
        iterCurrent = iterations[l]
        sigmaFluid = sigmaFluids[l]
        sigmaDiff = sigmaDiffs[l]
        sigmaSmooth = sigmaSmooths[l]
        stepSize = stepSizes[l]
        oshapeCurrent = tuple(int((ii // scale)) for ii in paddedShape)
        if sigmaFluid != 0.0:
            filtFluid = sp.to_device(gaussian_kernel_3d(sigmaFluid), device)
            # filtFluid = sp.to_device(gaussian_kernel_1d(sigmaFluid), device)
            filtFluid = sp.fft(filtFluid, oshapeCurrent, norm=None)
        if sigmaDiff != 0.0:
            filtDiff = sp.to_device(gaussian_kernel_3d(sigmaDiff), device)
            # filtDiff = sp.to_device(gaussian_kernel_1d(sigmaDiff), device)
            filtDiff = sp.fft(filtDiff, oshapeCurrent, norm=None)
        if sigmaSmooth != 0.0:
            filtSmooth = sp.to_device(gaussian_kernel_3d(sigmaSmooth), device)
            # filtSmooth = sp.to_device(gaussian_kernel_1d(sigmaSmooth), device)
            filtSmooth = sp.fft(filtSmooth, oshapeCurrent, norm=None)
        # show(filtDiff, "filter")

        # Extra Image Smoothing
        if sigmaSmooth != 0.0:
            IsCurrent = sp.resize(convolveFFT(Is, filtSmooth), Is.shape)
            ImCurrent = sp.resize(convolveFFT(Im, filtSmooth), Is.shape)
        else:
            IsCurrent = Is
            ImCurrent = Im

        # Downsample
        # ----------------
        # show(IsCurrent, "Static")
        # show(ImCurrent, "Moving")
        # ----------------

        IsCurrent = gaussianPyramid(IsCurrent, nLevels - l - 1)
        ImCurrent = gaussianPyramid(ImCurrent, nLevels - l - 1)
        corrV = xp.zeros((3,) + oshapeCurrent)
        corrV[0] = gaussianPyramid(finalV[0], nLevels - l - 1)
        corrV[1] = gaussianPyramid(finalV[1], nLevels - l - 1)
        corrV[2] = gaussianPyramid(finalV[2], nLevels - l - 1)

        # ----------------
        # show(IsCurrent, "Static Downsampled")
        # show(ImCurrent, "Moving Downsampled")
        # ----------------

        # Iterate
        # Progress Bar
        print("Current Image Shape: {} ...".format(oshapeCurrent))
        desc = "Level {}/{}".format(l + 1, nLevels)
        disable = not True
        total = iterCurrent
        with tqdm(desc=desc, total=total, disable=disable, leave=True) as pbar:
            for ll in range(iterCurrent):
                if diffeomorphic:
                    # Map to diffeomorphism
                    # show(corrV, "Pre exponentiation")
                    T, n = expfield(corrV)
                    Tinv, ninv = expfield(-1 * corrV)
                    # show(T, "Post exponentiation")

                else:
                    T = corrV
                    Tinv = -1 * corrV
                    n = ninv = 0
                # Update Velocity Field (Symmetric version)
                V = find_update_field(IsCurrent, ImCurrent, T, stepSize)
                V -= find_update_field(ImCurrent, IsCurrent, Tinv, stepSize)

                # ----------------
                # show(V, "V (Not Smoothed)")
                # ----------------

                # Fluid Regularization
                if sigmaFluid != 0.0:
                    V[0] = sp.resize(convolveFFT(V[0], filtFluid), oshapeCurrent)
                    V[1] = sp.resize(convolveFFT(V[1], filtFluid), oshapeCurrent)
                    V[2] = sp.resize(convolveFFT(V[2], filtFluid), oshapeCurrent)
                V *= 0.5

                E_p = energy(IsCurrent - ImCurrent, T, stepSize)
                E_c = energy(IsCurrent - ImCurrent, V, stepSize)
                loss.append(1 - (E_p - E_c) / E_p)
                # ----------------
                # show(V, "V (Smoothed)")
                # ----------------

                # Compose (First Approximation, Should probably use lie bracket version (Zb))
                # Symmetric Update is V <-- 0.5 * K_diff (*) ( Z(v, K_fluid (*) U_forw) - Z(-v, K_fluid (*) U_back))
                # Za(V,U): V+U
                # Zb(V,U): V = V + U + 0.5 * [V,U]
                # [V,U] = Jac(V)*U - Jac(U)*V
                corrV += V

                # ----------------
                # show(corrV, "corrV (Not Smoothed)")
                # -----------------

                # Diff Regularization
                if sigmaDiff != 0.0:
                    corrV[0] = sp.resize(convolveFFT(corrV[0], filtDiff), oshapeCurrent)
                    corrV[1] = sp.resize(convolveFFT(corrV[1], filtDiff), oshapeCurrent)
                    corrV[2] = sp.resize(convolveFFT(corrV[2], filtDiff), oshapeCurrent)

                # ----------------
                # show(corrV, "corrV (Smoothed)")
                # -----------------

                pbar.set_postfix(n=n, ninv=ninv, loss=loss[ll])
                pbar.update()

        # Upsample corrV and save as finalV (Should I upsample all the way each time or move up the pyramid?)
        finalV[0] = upsample(corrV[0], paddedShape, device)
        finalV[1] = upsample(corrV[1], paddedShape, device)
        finalV[2] = upsample(corrV[2], paddedShape, device)

        # ----------------
        # show(corrV, "corrV")
        # show(finalV, "finalV (Upsampled)")
        # -----------------

    # crop to original dimensions
    finalV = sp.resize(finalV, ((3,) + originalShape))
    Is = sp.resize(Is, originalShape)
    Im = sp.resize(Im, originalShape)
    # np = sp.Device(-1).xp
    # plt.LinePlot(sp.to_device(xp.array(loss)))
    if return_warped:
        if diffeomorphic:
            Tinv, _ = expfield(-1 * finalV)
            finalV, _ = expfield(finalV)
            ImWarped = imwarp(Im, finalV)
            # show(Tinv * finalV, "Tinv*T")
        else:
            ImWarped = imwarp(Im, finalV)
            # show(finalV * -1 * finalV, "Tinv*T (not diffeo)")
        show(100 * convolveFFT(xp.abs((Is - Im)) / Is, filtDiff), "Not Registered")
        show(100 * xp.abs((Is - ImWarped)) / Is, "Registered")
        show(ImWarped, "Warped Image", "gray")
        return finalV, ImWarped
    else:
        return finalV
