import sigpy as sp

import sigpy.plot as plt
from tqdm.auto import tqdm
from normalize import normalize

# import numpy as np
# from scipy.fft import next_fast_len


class Demons:
    def __init__(self, Is, Im, nLevels=4, sigma=1.0, device=0):
        # Get corresponding matrix math module
        self.device = device
        self.xp = sp.Device(device).xp

        # Move to selected device
        Is = sp.to_device(Is, device)
        Im = sp.to_device(Im, device)

        # Normalize. Important for numerical stability
        Is = normalize(Is, 0, 255, device)
        Im = normalize(Im, 0, 255, device)

        self.nLevels = nLevels
        self.scales = [2 ** (self.nLevels - l - 1) for l in range(self.nLevels)]
        self.iterations = list(int(30 + 30 * ii * 3) for ii in range(nLevels))
        self.originalShape = Is.shape
        self.paddedShape = self.calcPad(self.originalShape, self.nLevels, 5)
        self.Is = sp.resize(Is, self.paddedShape)
        self.Im = sp.resize(Im, self.paddedShape)

        # Initialize final velocity field
        self.finalV = self.xp.zeros((3,) + self.paddedShape)
        # Define outshapes
        self.oshapes = []
        for scale in self.scales:
            self.oshapes.append(tuple(int((ii // scale)) for ii in self.paddedShape))

        self.fd = []
        for ii in range(nLevels):
            self.fd.append(sp.linop.FiniteDifference(self.oshapes[ii]))
        self.sigma_x = 1.0
        # Thresholds for numerical stability (only works for images with similar intensity.)
        self.IntensityDifferenceThreshold = 0.001
        self.DenominatorThreshold = 1e-9
        self.sigmaDiff = sigma

    def normalize(I_in, minv, maxv, device):
        xp = sp.Device(device)
        out = (maxv - minv) * (I_in - xp.min(I_in[:])) / (xp.max(I_in[:]) - xp.min(I_in[:])) + minv
        return out

    def calcPad(self, ishape, n, p):
        # This function pads image.
        oshape = []
        for ii in ishape:
            r = 2 ** n - (ii + p) % (2 ** n)
            oshape.append((ii + p) + r)
        return tuple(oshape)

    def findUpdateField(self):
        # Warp image
        ImWarped = self.imwarp(self.ImCurrent, self.corrV)
        Idiff = self.IsCurrent - ImWarped
        Denominator = self.dSMag + Idiff ** 2 / self.sigma_x ** 2
        cfactor = Idiff / Denominator
        V = cfactor * self.dS

        # Generate mask for null values (convert to zeros)
        mask = self.xp.zeros(V.shape, dtype=bool)
        for j in range(3):
            mask[j] = (
                (~self.xp.isfinite(V[j]))
                | (self.xp.abs(Idiff) < self.IntensityDifferenceThreshold)
                | (Denominator < self.DenominatorThreshold)
            )
        V[mask] = 0.0
        return V

    def imwarp(self, I_in, T):
        # This function warps I_in to coordinates defined by coord + T
        # Interpolate using sigpy. param = 1 is linear.
        nx, ny, nz = I_in.shape
        coord = self.xp.mgrid[0:nx, 0:ny, 0:nz].astype("f")
        I_out = sp.interpolate(
            I_in,
            self.xp.moveaxis(coord, 0, -1) + self.xp.moveaxis(T, 0, -1),
            kernel="spline",
            width=2,
            param=1,
        )
        return I_out

    def gaussianKernel1d(self, sigma=1.0):
        # Generates Gaussian Kernel. Kernel truncated at 4 * sigma
        # Round to next odd integer
        N = 4 * sigma
        N = self.xp.ceil(N) // 2 * 2 + 1
        if N < 3.0:
            N = 3.0
        # print(N)
        x = self.xp.linspace(0, N - 1, int(N))
        alpha = 0.5 * (N - 1) / sigma
        k1d = self.xp.exp(-0.5 * (alpha * (x - (N - 1) / 2) / ((N - 1) / 2)) ** 2)
        return k1d

    def gaussianKernel3d(self, sigma=1.0):
        if self.xp.array(sigma).size == 1:
            k1d = self.gaussianKernel1d(sigma)
            k3dx, k3dy, k3dz = self.xp.ix_(k1d, k1d, k1d)
            k3d = k3dx * k3dy * k3dz
        else:
            k1d = self.gaussianKernel1d(sigma[0])
            k2d = self.gaussianKernel1d(sigma[1])
            k3d = self.gaussianKernel1d(sigma[2])
            k3dx, k3dy, k3dz = self.xp.ix_(k1d, k2d, k3d)
            k3d = k3dx * k3dy * k3dz
        # Normalize (Unless 1D already normalized)
        k3d = k3d / self.xp.sum(k3d)
        return k3d

    def gaussianPyramid(self, I, n):
        """
        Use gaussian pyramid to downsample image by n levels
        Should be faster than going straight to downsample factor
        because we keep convolution kernel small.
        """
        ishape = I.shape
        if n == 0:
            return I
        # loop through scales
        for ii in range(n):
            oshape = tuple(int(jj // (2 ** ii)) for jj in ishape)
            k1d = self.xp.array([1, 4, 6, 4, 1]) / 16  # sigma ~1.0
            k3dx, k3dy, k3dz = self.xp.ix_(k1d, k1d, k1d)
            k3d = k3dx * k3dy * k3dz
            out = sp.convolve(I, k3d)
            out = sp.convolve(out, k3d)  # Convolve twice
            out = sp.resize(out, oshape)
            out = sp.downsample(out, (2,) * I.ndim)
            # Set I for next level
            I = out

        return out

    def gaussianPyramidUp(self, I, n):
        """
        Use gaussian pyramid to downsample image by n levels
        Should be faster than going straight to downsample factor
        because we keep convolution kernel small.
        """
        ishape = I.shape
        if n == 0:
            return I
        # loop through scales
        for ii in range(n):
            oshape = tuple(int(jj * (2 ** (ii + 1))) for jj in ishape)
            print(ishape)
            print(oshape)

            k1d = self.xp.array([1, 4, 6, 4, 1]) / 16  # sigma ~1.0
            k3dx, k3dy, k3dz = self.xp.ix_(k1d, k1d, k1d)
            k3d = k3dx * k3dy * k3dz
            out = sp.upsample(I, oshape, (1.5,) * I.ndim)
            out = sp.convolve(out, k3d)
            out = sp.convolve(out, k3d)
            out = sp.resize(out, oshape) * 4
            # Set I for next level
            I = out

        return out

    def convolveFFT(self, I1, I2):
        # FFT convolution O(NlogN) Should pad with symetric image for proper periodic convolution
        oshape = I1.shape
        I1 = sp.fft(I1, norm=None)
        I1 = self.xp.vstack(
            (I1, self.xp.flip(self.xp.flip(self.xp.flip(I1, axis=0), axis=1), axis=2))
        )
        I2 = sp.fft(I2, oshape=I1.shape, norm=None)
        # I2 = sp.resize(I2, I1.shape)
        out = sp.ifft(I1 * I2, oshape=oshape, norm=None)
        out = self.xp.real(out)
        return out

    # def downsample(I, oshape, device):
    #     # Filter then decimate (can do in either frequency domain or spatial)
    #     xp = sp.Device(device).xp
    #     zFactor = I.shape[0] / oshape[0]
    #     # zFactor might need a factor of 2 somewhere in filterSize
    #     filterSize = [int(round(i / (2 * zFactor))) for i in (I.shape)]
    #     out = sp.fft(I, norm=None)
    #     W = sp.hanning(filterSize, device=device)
    #     out *= sp.resize(W, out.shape)
    #     out = sp.resize(out, oshape)
    #     out = xp.abs(sp.ifft(out, norm=None))
    #     # plt.ImagePlot(I)
    #     # plt.ImagePlot(out)
    #     # deblur?
    #     # filt_inv = gaussianKernel3d(5, 0.3, inverse="True")
    #     # imgOut = sp.resize(sp.conv.convolve(imgOut, filt_inv), imgOut.shape)
    #     return out

    def upsample(self, I, oshape, device):
        # Upsampling via FFT zeropadding (upsample and low pass).
        zFactor = oshape[0] / I.shape[0]
        out = sp.fft(I, norm=None)
        out = sp.resize(out, oshape)
        out = self.xp.real(sp.ifft(out, norm=None) * zFactor)
        return out

    # def energy(I1, I2, sigma_x):
    #     # E_corr s_opt
    #     # I1 should be Idiff, I2 should be correspondence field
    #     xp = sp.get_array_module(I1)
    #     E = xp.linalg.norm(I2) ** 2 / (sigma_x ** 2)
    #     #  xp.sum(xp.linalg.norm(I1) ** 2 / xp.sum(I1) +
    #     return E / 2

    def run(self):

        print("Original Image Shape: {} ...".format(self.originalShape))
        print("Padded Image Shape: {} ...".format(self.paddedShape))
        print("Image Scaling Factors: {} ...".format(self.scales))
        print("Number of Levels: {} ...".format(self.nLevels))
        print("Iterations per Level: {} ...".format(self.iterations))
        # print("Fluid Sigmas: {} ...".format(self.sigmaFluids))
        print("Diffusion Sigmas: {} ...".format(self.sigmaDiff))

        # Begin Pyramid
        for l in range(self.nLevels):
            iterCurrent = self.iterations[l]
            sigmaDiff = self.sigmaDiff
            oshapeCurrent = self.oshapes[l]

            self.IsCurrent = self.gaussianPyramid(self.Is, self.nLevels - l - 1)
            self.ImCurrent = self.gaussianPyramid(self.Im, self.nLevels - l - 1)

            self.dS = self.fd[l] * self.IsCurrent
            self.corrV = self.xp.zeros((3,) + oshapeCurrent)
            self.corrV[0] = self.gaussianPyramid(self.finalV[0], self.nLevels - l - 1)
            self.corrV[1] = self.gaussianPyramid(self.finalV[1], self.nLevels - l - 1)
            self.corrV[2] = self.gaussianPyramid(self.finalV[2], self.nLevels - l - 1)
            self.dSMag = self.dS[0] ** 2 + self.dS[1] ** 2 + self.dS[2] ** 2
            self.filtDiff = self.gaussianKernel3d(sigmaDiff)
            # Iterate
            # Progress Bar
            print("Current Image Shape: {} ...".format(oshapeCurrent))
            desc = "Level {}/{}".format(l + 1, self.nLevels)
            disable = not True
            total = iterCurrent
            with tqdm(desc=desc, total=total, disable=disable, leave=True) as pbar:
                for ll in range(iterCurrent):
                    # Update Velocity Field
                    self.corrV += self.findUpdateField()
                    # Regularization
                    self.corrV[0] = sp.resize(
                        sp.convolve(self.corrV[0], self.filtDiff), oshapeCurrent
                    )
                    self.corrV[1] = sp.resize(
                        sp.convolve(self.corrV[1], self.filtDiff), oshapeCurrent
                    )
                    self.corrV[2] = sp.resize(
                        sp.convolve(self.corrV[2], self.filtDiff), oshapeCurrent
                    )

                    # # Regularization
                    # self.corrV[0] = sp.resize(
                    #     self.convolveFFT(self.corrV[0], self.filtDiff), oshapeCurrent
                    # )
                    # self.corrV[1] = sp.resize(
                    #     self.convolveFFT(self.corrV[1], self.filtDiff), oshapeCurrent
                    # )
                    # self.corrV[2] = sp.resize(
                    #     self.convolveFFT(self.corrV[2], self.filtDiff), oshapeCurrent
                    # )
                    pbar.update()

            # # Upsample corrV and save as finalV (Should I upsample all the way each time or move up the pyramid?)
            self.finalV[0] = self.upsample(self.corrV[0], self.paddedShape, self.device)
            self.finalV[1] = self.upsample(self.corrV[1], self.paddedShape, self.device)
            self.finalV[2] = self.upsample(self.corrV[2], self.paddedShape, self.device)
            # self.finalV[0] = self.gaussianPyramidUp(self.corrV[0], self.nLevels - l - 1)
            # self.finalV[0] = self.gaussianPyramidUp(self.corrV[1], self.nLevels - l - 1)
            # self.finalV[0] = self.gaussianPyramidUp(self.corrV[2], self.nLevels - l - 1)
            # ----------------
            plt.ImagePlot(self.corrV, title="Update")
            plt.ImagePlot(self.finalV, title="finalV (Upsampled)")
            # -----------------

        # crop to original dimensions
        self.finalV = sp.resize(self.finalV, ((3,) + self.originalShape))
        self.Is = sp.resize(self.Is, self.originalShape)
        self.Im = sp.resize(self.Im, self.originalShape)
        # np = sp.Device(-1).xp
        ImWarped = self.imwarp(self.Im, self.finalV)
        return ImWarped, self.finalV
