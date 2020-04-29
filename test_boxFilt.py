import sigpy as sp
import sigpy.plot as splt
import numpy as np
import matplotlib.pyplot as plt
import eboxFilter as ebox
import oaconvolve as oac
import gaussianFilter as gFilter
import nibabel as nib
import time
import fftConvolve as fftC

sigma = 2.0
# k = 4
# r, alpha, c1, c2 = ebox.preComputeConstants(sigma, k)

device = 0
xp = sp.Device(device).xp
I1 = nib.load("/home/ltorres/data/imoco/103-005/mri/20160114/expi.nii.gz").get_fdata()
print(I1.shape)
# I1 = sp.downsample(I1, (2, 2, 2))
# This might work. Now to do a 2D gaussian slice by slice for better speed.
gkern1 = gFilter.gaussianKernel1d(sigma, device=device)
gkern1 = gkern1 / xp.sum(gkern1)
gkern3 = gFilter.gaussianKernel3d(sigma, device=device)
print("Transfering.")
# I1 = sp.to_device(I1[:, 128, :], device)
I1 = sp.to_device(I1, device)
# splt.ImagePlot(I1)
print("Filtering.")
time1 = time.time()
I2 = I1.copy()
# Ebox
# for ii in range(k):
#     I2 = sp.convolve(I2, ekern3)
# I2 = sp.to_device(ebox.eboxGaussianConv3D(I2, r, c1, c2, k, device=device))
# time_ebox = time.time() - time1
# Overlap add
I2 = oac.oaconvolve3(gkern1, I1, device=device)
time1 = time.time()
# I2 = fftC.FFTconvolve3(gkern1, I1, device=device)
time_FFT = time.time() - time1
print("FFTConv Time: {}".format(time_FFT))

sp.resize(I2, I1.shape)
# Gaussian
time1 = time.time()
I3 = sp.resize(sp.convolve(I1, gkern3), I1.shape)
time_gfilt = time.time() - time1
print("Conv Time: {}".format(time_gfilt))
print("FFTConv speedup: {}".format(time_gfilt / time_FFT))
print(xp.allclose(I2, I3))
# print("Transfering.")
# splt.ImagePlot(I2)
# splt.ImagePlot(I3)
