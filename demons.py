import sigpy as sp
from tqdm.auto import tqdm
from normalize import normalize
import convolutions as conv
import filters
import finiteDifferences as fd
from imwarp import imwarp
from resample import zoom
from scipy.fft import next_fast_len
import energy
import sigpy.plot as plt
import numpy as np
from scipy.ndimage import median_filter
from demonsUpdate import demonsUpdate as update
import diffeomorphism


class Demons:
    def __init__(
        self,
        Is,
        Im,
        nLevels=4,
        diffusionSigmas=1.0,
        fluidSigmas=1.0,
        max_iter=100,
        alpha=2.5,
        cThresh=1e-6,
        variant="passive",
        diffeomorphic=False,
        compositionType="A",
        device=-1,
    ):
        # Get corresponding matrix math module
        self.device = device
        self.xp = sp.Device(device).xp

        # Move to selected device
        Is = sp.to_device(Is, device)
        Im = sp.to_device(Im, device)

        # Normalize. Important for numerical stability
        # self.maxval = 1.0
        # Is = normalize(Is, 0.0, self.maxval, device)
        # Im = normalize(Im, 0.0, self.maxval, device)
        Is = Is / Im.max()
        Im = Im / Im.max()
        # plt.ImagePlot(Is)
        self.nLevels = nLevels
        self.scales = [2 ** (self.nLevels - ii - 1) for ii in range(self.nLevels)]
        self.iterations = list(int(max_iter // 2 ** ii) for ii in range(nLevels))
        self.originalShape = Is.shape
        self.paddedShape = self.calcPad(self.originalShape, self.nLevels, 40)
        self.Is = sp.resize(Is, self.paddedShape)
        self.Im = sp.resize(Im, self.paddedShape)
        self.nD = len(self.Is.shape)
        # Initialize final velocity field
        self.finalV = self.xp.zeros((self.nD,) + self.paddedShape)
        # Define outshapes
        self.oshapes = []
        for scale in self.scales:
            self.oshapes.append(tuple(int((ii // scale)) for ii in self.paddedShape))
        self.alpha = alpha
        # Thresholds for numerical stability (only works for images with similar intensity.)
        self.IntensityDifferenceThreshold = 1e-2
        self.DenominatorThreshold = 1e-9
        self.sigmaDiff = diffusionSigmas
        self.sigmaFluid = fluidSigmas
        self.cThresh = cThresh
        self.variant = variant
        self.diffeomorphic = diffeomorphic
        self.compositionType = compositionType

    def calcPad(self, ishape, n, p):
        # This function pads image.
        oshape = []
        for ii in ishape:
            r = 2 ** n - (ii + p) % (2 ** n)
            oshape.append((ii + p) + r)
        return tuple(oshape)

    def run(self):
        try:
            print("Original Image Shape: {} ...".format(self.originalShape))
            print("Padded Image Shape: {} ...".format(self.paddedShape))
            print("Image Scaling Factors: {} ...".format(self.scales))
            print("Number of Levels: {} ...".format(self.nLevels))
            print("Iterations per Level: {} ...".format(self.iterations))
            print("Diffusion Sigmas: {} ...".format(self.sigmaDiff))

            # Begin Pyramid
            for ii in range(self.nLevels):
                iterCurrent = self.iterations[ii]
                sigmaDiff = self.sigmaDiff
                sigmaFluid = self.sigmaDiff
                # sigmaImage = self.sigmaDiff
                oshapeCurrent = self.oshapes[ii]

                self.IsCurrent = zoom(self.Is, 2 ** -(self.nLevels - ii - 1), device=self.device)
                self.ImCurrent = zoom(self.Im, 2 ** -(self.nLevels - ii - 1), device=self.device)
                D = fd.CentralDifference(oshapeCurrent)
                self.dS = D * self.IsCurrent
                # plt.ImagePlot(self.dS)
                self.corrV = self.xp.zeros((self.nD,) + oshapeCurrent)
                for jj in range(self.nD):
                    self.corrV[jj] = zoom(
                        self.finalV[jj], 2 ** -(self.nLevels - ii - 1), device=self.device
                    )
                self.dSMag = self.xp.zeros(oshapeCurrent)
                for jj in range(self.nD):
                    self.dSMag += self.dS[jj] ** 2

                # Generate Gaussian filter in Fourier Space with optimal fft sizes (not sure if needed...)
                # Diffusion
                if sigmaDiff > 0.0:
                    self.filtDiff = filters.gaussianKernelnd(
                        sigmaDiff, n=self.nD, device=self.device
                    )
                    self.fftsizeDiff = tuple(
                        self.filtDiff.shape[0] + jj - 1 for jj in oshapeCurrent
                    )
                    self.fftsizeDiff = tuple(next_fast_len(jj) for jj in self.fftsizeDiff)
                    self.filtDiff = sp.fft(self.filtDiff, oshape=self.fftsizeDiff, norm=None)

                # Fluid
                if sigmaFluid > 0.0:
                    self.filtFluid = filters.gaussianKernelnd(
                        sigmaFluid, n=self.nD, device=self.device
                    )
                    self.fftsizeFluid = tuple(
                        self.filtFluid.shape[0] + jj - 1 for jj in oshapeCurrent
                    )
                    self.fftsizeFluid = tuple(next_fast_len(jj) for jj in self.fftsizeFluid)
                    self.filtFluid = sp.fft(self.filtFluid, oshape=self.fftsizeFluid, norm=None)

                # Iterate
                # Progress Bar
                self.currentLoss = 0
                self.loss = []
                stop = 0
                nLossAverages = 4 * (self.nLevels - ii)
                print("Current Image Shape: {} ...".format(oshapeCurrent))
                desc = "Level {}/{}".format(ii + 1, self.nLevels)
                disable = not True
                total = iterCurrent
                with tqdm(desc=desc, total=total, disable=disable, leave=True) as pbar:
                    for jj in range(iterCurrent):
                        # Update Velocity Field
                        V, self.currentLoss = update(
                            self.IsCurrent,
                            self.ImCurrent,
                            self.corrV,
                            self.alpha,
                            dS=self.dS,
                            dSMag=self.dSMag,
                            variant=self.variant,
                            scalingFactor=(self.nLevels - ii),
                            diffeomorphic=self.diffeomorphic,
                            device=self.device,
                        )
                        # Convolve (Fluid Regularization (Roughly))
                        # plt.ImagePlot(V)
                        if sigmaFluid > 0.0:
                            for kk in range(self.nD):
                                V[kk] = self.xp.real(
                                    conv.FFTConvolve(
                                        self.filtFluid,
                                        V[kk],
                                        shape=self.fftsizeFluid,
                                        device=self.device,
                                    )
                                )
                        # Accumulate field
                        self.corrV = diffeomorphism.compose(
                            self.corrV,
                            V,
                            diffeomorphic=self.diffeomorphic,
                            compositionType=self.compositionType,
                            device=self.device,
                        )
                        # plt.ImagePlot(self.corrV)
                        # self.corrV += V % Additive is first order approximation in log domain for diffeomorphic
                        # Stopping criteria
                        self.loss.append(self.currentLoss)
                        if jj > 0 + nLossAverages:
                            prevLoss = self.xp.mean(
                                self.xp.array(self.loss[jj - nLossAverages - 1 : jj - 1])
                            )
                            stop = self.xp.abs((self.loss[jj] - prevLoss) / prevLoss)
                            pbar.set_postfix(loss=stop)
                            if stop < self.cThresh:
                                break

                        # Convolve (Diffusion Regularization (Roughly))
                        # plt.ImagePlot(self.corrV)
                        if sigmaDiff > 0.0:
                            for kk in range(self.nD):
                                self.corrV[kk] = self.xp.real(
                                    conv.FFTConvolve(
                                        self.filtDiff,
                                        self.corrV[kk],
                                        shape=self.fftsizeDiff,
                                        device=self.device,
                                    )
                                )
                        pbar.set_postfix(loss=stop)
                        pbar.update()
                        # plt.ImagePlot(self.corrV)

                # # Upsample corrV and save as finalV (Should I upsample all the way each time or move up the pyramid?)
                for jj in range(self.nD):
                    self.finalV[jj] = zoom(
                        self.corrV[jj], 2 ** (self.nLevels - ii - 1), device=self.device
                    )

            # crop to original dimensions
            self.finalV = sp.resize(self.finalV, ((self.nD,) + self.originalShape))
            self.Is = sp.resize(self.Is, self.originalShape)
            self.Im = sp.resize(self.Im, self.originalShape)

            # Warp Final Image
            ImWarped = imwarp(self.Im, self.finalV, device=self.device)
            # post processing
            # ImWarped = sp.to_device(imwarp(self.Im, self.finalV, device=self.device))
            # Sharpen to remove blurring by interpolation
            # ImEdges = ImWarped - median_filter(ImWarped, 5)
            # skern = filters.sharpenKernelNd(self.nD, device=self.device)
            # skern = sp.triang((3, 3, 3), device=self.device)
            # skern /= self.xp.sum(skern)
            # ImEdges = ImWarped - sp.resize(sp.convolve(ImWarped, skern), self.originalShape)
            # ImWarped += ImEdges
            # ImEdges = self.xp.sum(self.xp.square(fd.Laplacian(ImWarped.shape) * ImWarped), axis=0)
            # if self.nD == 3:
            #     ImEdges = ImEdges.reshape((ImEdges.shape[0] * ImEdges.shape[1]), ImEdges.shape[2])
            # U, s, V = np.linalg.svd(sp.to_device(ImEdges), full_matrices=False)
            # plt.LinePlot(s, mode="r")
            # ImWarped += normalize(ImEdges, 0, self.maxval, device=-1)
            # ImWarped += median_filter(normalize(ImEdges, 0, ImWarped.max(), device=-1), 5)
            # plt.ImagePlot(ImEdges, title="Laplacian Image")
            plt.ImagePlot(ImWarped, title="Warped Image")
            plt.ImagePlot(self.Is - ImWarped, title="Registered Difference")
            plt.ImagePlot(self.finalV, title="Final Warp Field")
            plt.ImagePlot(energy.jacobianDet(self.finalV) < 0, title="Jacobian Determinant < 0")
            plt.ImagePlot(energy.jacobianDet(self.finalV), title="Jacobian Determinant")
            return ImWarped, self.finalV
        except KeyboardInterrupt:
            raise
