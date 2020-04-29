import sigpy as sp
import sigpy.plot as splt
import numpy as np
import matplotlib.pyplot as plt
import eboxFilter as ebox
import gaussianFilter as gFilter
import nibabel as nib
import time

sigma = 10.0
k = 4
r, alpha, c1, c2 = ebox.preComputeConstants(sigma, k)
# size = 600
# N = 7
# a1 = np.ones(size)
# a1 = np.random.rand(size)
# a2 = sp.to_device(ebox.eboxGaussianConv(a1, r, c1, c2, k, device=-1))
# print(a2.shape)
# a3 = np.convolve(a1, np.ones((N,)) / N, mode="same")
# # plt.plot(np.arange(size), a1, label="Original")
# plt.plot(np.arange(size), a2, label="Running")
# plt.plot(np.arange(size), a3, label="Convolve")
# plt.legend()
# plt.show()
device = 0
xp = sp.Device(device).xp
I1 = nib.load("/home/ltorres/data/imoco/103-005/mri/20160114/expi.nii.gz").get_fdata()
# I1 = sp.downsample(I1, (2, 2, 2))
# This might work. Now to do a 2D gaussian slice by slice for better speed.
ekern = sp.to_device(ebox.eboxKernel(r, alpha, c1, c2, device=-1), device)
print(ekern)
k3dx, k3dy, k3dz = xp.ix_(ekern, ekern, ekern)
ekern3 = k3dx * k3dy * k3dz
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
I2 = sp.to_device(ebox.eboxGaussianConv3D(I2, r, c1, c2, k, device=device))
time_ebox = time.time() - time1

# Gaussian
time1 = time.time()
I3 = sp.convolve(I1, gkern3)
time_gfilt = time.time() - time1
print("Ebox Time: {}".format(time_ebox))
print("gfilt Time: {}".format(time_gfilt))
print("ebox speedup Time: {}".format(time_gfilt / time_ebox))
print("Transfering.")
# splt.ImagePlot(I2)
# splt.ImagePlot(I3)
