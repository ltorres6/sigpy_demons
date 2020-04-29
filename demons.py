import sigpy as sp

import sigpy.plot as plt
from tqdm.auto import tqdm
from normalize import normalize
import convolutions as conv
import filters
from imwarp import imwarp
from resample import zoom
from scipy.fft import next_fast_len


class Demons:
    def __init__(self, Is, Im, nLevels=4, sigmas=1.0, device=-1):
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
        # self.iterations = list(int(30 + 30 * ii * 3) for ii in range(nLevels, 0, -1))
        self.iterations = list(int(200 // 2 ** ii) for ii in range(nLevels))
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
        self.sigmaDiff = sigmas

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
        ImWarped = imwarp(self.ImCurrent, self.corrV, device=self.device)
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

    # def upsample(self, I, oshape, device):
    #     # Upsampling via FFT zeropadding (upsample and low pass).
    #     zFactor = oshape[0] / I.shape[0]
    #     out = sp.fft(I, norm=None)
    #     out = sp.resize(out, oshape)
    #     out = self.xp.real(sp.ifft(out, norm=None) * zFactor)
    #     return out

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

            # self.IsCurrent = self.gaussianPyramid(self.Is, self.nLevels - l - 1)
            # self.ImCurrent = self.gaussianPyramid(self.Im, self.nLevels - l - 1)
            # plt.ImagePlot(self.Is)
            self.IsCurrent = zoom(self.Is, 2 ** -(self.nLevels - l - 1), device=self.device)
            self.ImCurrent = zoom(self.Im, 2 ** -(self.nLevels - l - 1), device=self.device)
            # plt.ImagePlot(self.IsCurrent)
            self.dS = self.fd[l] * self.IsCurrent
            # plt.ImagePlot(self.dS)

            self.corrV = self.xp.zeros((3,) + oshapeCurrent)
            # self.corrV[0] = self.gaussianPyramid(self.finalV[0], self.nLevels - l - 1)
            # self.corrV[1] = self.gaussianPyramid(self.finalV[1], self.nLevels - l - 1)
            # self.corrV[2] = self.gaussianPyramid(self.finalV[2], self.nLevels - l - 1)
            self.corrV[0] = zoom(self.finalV[0], 2 ** -(self.nLevels - l - 1), device=self.device)
            self.corrV[1] = zoom(self.finalV[1], 2 ** -(self.nLevels - l - 1), device=self.device)
            self.corrV[2] = zoom(self.finalV[2], 2 ** -(self.nLevels - l - 1), device=self.device)
            self.dSMag = self.dS[0] ** 2 + self.dS[1] ** 2 + self.dS[2] ** 2
            # plt.ImagePlot(self.dSMag)

            # Generate Gaussian filter in Fourier Space with optimal fft sizes
            self.filtDiff = filters.gaussianKernel3d(sigmaDiff, device=self.device)
            self.fftsize = tuple(self.filtDiff.shape[0] + i - 1 for i in oshapeCurrent)
            self.fftsize = tuple(next_fast_len(i) for i in self.fftsize)
            self.filtDiff = sp.fft(self.filtDiff, oshape=self.fftsize, norm=None)
            # plt.ImagePlot(self.filtDiff)

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
                    # Convolve (Diffusion Regularize)
                    self.corrV[0] = self.xp.real(
                        conv.FFTConvolve(
                            self.filtDiff, self.corrV[0], shape=self.fftsize, device=self.device
                        )
                    )
                    self.corrV[1] = self.xp.real(
                        conv.FFTConvolve(
                            self.filtDiff, self.corrV[1], shape=self.fftsize, device=self.device
                        )
                    )
                    self.corrV[2] = self.xp.real(
                        conv.FFTConvolve(
                            self.filtDiff, self.corrV[2], shape=self.fftsize, device=self.device
                        )
                    )
                    pbar.update()

            # # Upsample corrV and save as finalV (Should I upsample all the way each time or move up the pyramid?)
            self.finalV[0] = zoom(self.corrV[0], 2 ** (self.nLevels - l - 1), device=self.device)
            self.finalV[1] = zoom(self.corrV[1], 2 ** (self.nLevels - l - 1), device=self.device)
            self.finalV[2] = zoom(self.corrV[2], 2 ** (self.nLevels - l - 1), device=self.device)
            # ----------------
            # plt.ImagePlot(self.corrV, title="Update")
            # plt.ImagePlot(self.finalV, title="finalV (Upsampled)")
            # -----------------

        # crop to original dimensions
        self.finalV = sp.resize(self.finalV, ((3,) + self.originalShape))
        self.Is = sp.resize(self.Is, self.originalShape)
        self.Im = sp.resize(self.Im, self.originalShape)
        ImWarped = imwarp(self.Im, self.finalV, device=self.device)
        # plt.ImagePlot(ImWarped, title="Final Warped Image")
        # plt.ImagePlot(self.finalV, title="Final Warp")
        # plt.ImagePlot((self.Is - self.Im) / self.Is, title="Original Difference")
        # plt.ImagePlot((self.Is - ImWarped) / self.Is, title="Registered Difference")
        return ImWarped, self.finalV
