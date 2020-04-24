# import sigpy.plot as plt
import sigpy as sp
import numpy as np
import nibabel as nib
import time
import demons

device = 0

timeI = time.time()
Is = nib.load("/home/ltorres/data/imoco/103-005/mri/20160114/expi.nii.gz").get_fdata()
Im = nib.load("/home/ltorres/data/imoco/103-005/mri/20160114/insp.nii.gz").get_fdata()
# plt.ImagePlot(Is)
velocityField, ImWarped = demons.run(
    Is,
    Im,
    iterations=[100, 50, 50],
    sigmaFluids=[0.5] * 3,
    sigmaDiffs=[1.0] * 3,
    sigmaSmooths=[0.5] * 3,
    stepSizes=[1.0] * 3,
    diffeomorphic=False,
    device=device,
    return_warped=True,
)
timeF = time.time() - timeI
print("Runtime: {} seconds.".format(timeF))

ImWarped = nib.Nifti1Image(sp.to_device(ImWarped), affine=np.eye(4))
nib.save(ImWarped, "/home/ltorres/data/imoco/103-005/mri/20160114/insp2expi.nii.gz")
velocityField = nib.Nifti1Image(sp.to_device(velocityField), affine=np.eye(4))
nib.save(velocityField, "/home/ltorres/data/imoco/103-005/mri/20160114/velocityField.nii.gz")
