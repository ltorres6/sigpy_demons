# import sigpy.plot as plt
import sigpy as sp
import numpy as np
import nibabel as nib
import time
from demons import Demons
from normalize import normalize
import sigpy.plot as plt
import finiteDifferences as fd

device = -1

timeI = time.time()
Is = nib.load("/home/ltorres/data/imoco/103-005/mri/20160114/expi.nii.gz").get_fdata()
Is = Is[:, 128, :]
D = fd.CentralDifference(Is.shape)
L = fd.Laplacian(Is.shape)
DI = D * Is
LI = L * Is

DM = np.zeros(Is.shape)
LM = np.zeros(Is.shape)
for ii in range(len(Is.shape)):
    DM += DI[ii] ** 2
    LM += LI[ii] ** 2
plt.ImagePlot(Is)
plt.ImagePlot(DI)
plt.ImagePlot(LI)
plt.ImagePlot(DM)
plt.ImagePlot(LM)
