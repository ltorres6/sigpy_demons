# import sigpy.plot as plt
import sigpy as sp
import numpy as np
import nibabel as nib
import time
from demons import Demons

device = 0
# variant = "passive"
# variant = "active"
variant = "inverseConsistent"
diffeomorphic = True
compositionType = "A"

cThresh = 1e-6
alpha = 3.0
diffusionSigmas = 1.5
fluidSigmas = 1.5

timeI = time.time()
Is = nib.load("/home/ltorres/data/imoco/103-005/mri/20160114/expi.nii.gz").get_fdata()
Im = nib.load("/home/ltorres/data/imoco/103-005/mri/20160114/insp.nii.gz").get_fdata()
# Is = np.squeeze(Is[:, 128, :])
# Im = np.squeeze(Im[:, 128, :])

# plt.ImagePlot(Is)
ImWarped, velocityField = Demons(
    Is,
    Im,
    nLevels=3,
    diffusionSigmas=diffusionSigmas,
    fluidSigmas=fluidSigmas,
    max_iter=4000,
    alpha=alpha,
    cThresh=cThresh,
    variant=variant,
    diffeomorphic=diffeomorphic,
    compositionType=compositionType,
    device=device,
).run()
timeF = time.time() - timeI
print("Runtime: {} seconds.".format(timeF))

ImWarped = nib.Nifti1Image(sp.to_device(ImWarped), affine=np.eye(4))
nib.save(ImWarped, "/home/ltorres/data/imoco/103-005/mri/20160114/insp2expi.nii.gz")
# Im = nib.Nifti1Image(normalize(Im), affine=np.eye(4))
# nib.save(Im, "/home/ltorres/data/imoco/103-005/mri/20160114/inspNorm.nii.gz")
# Is = nib.Nifti1Image(normalize(Is), affine=np.eye(4))
# nib.save(Is, "/home/ltorres/data/imoco/103-005/mri/20160114/expiNorm.nii.gz")
velocityField = nib.Nifti1Image(sp.to_device(velocityField), affine=np.eye(4))
nib.save(velocityField, "/home/ltorres/data/imoco/103-005/mri/20160114/velocityField.nii.gz")
