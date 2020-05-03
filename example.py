from demons import Demons

# Options
# Compute Device: -1=CPU or 0=GPU
device = 0

# Demons Force Variation - "passive" , "active", "inverseConsistent" https://arxiv.org/pdf/0909.0928.pdf
variant = "inverseConsistent"

# Diffeomorphic or not. Probably only works well with "inverse consistent".
# Enables "expfield" function and different composition types for log space compositions.
diffeomorphic = False

# Warp field composition method. A: V1 + V2, B: V1 + V2 + [V1, V2] (lie bracket).
# If diffeomorphic is false this does nothing and the fields are interpolated and summed.
compositionType = "A"

# Number of Multi-resolution levels.
nLevels = 3

# Convergence Threshold. Set to 0 if you want to run all iterations.
cThresh = 1e-6

# Maximum Number of iterations at lowest resolution. Each level performs half iterations of previous level.
# Set to something super large to run forever (or until convergence).
max_iter = 4000

# Step size control. 2.0 is mathematically ideal as it restricts step size to be 0.5 voxels.
# Smaller values increase step size, but may cause unrealistic warps/unstable convergence.
alpha = 2.0

# Gaussian Smoothing Sigmas.
diffusionSigmas = 1.5
fluidSigmas = 1.5

# Image Loading. Could be replaced with anything really.
# 3D
# Is = nib.load("STATIC_IMAGE.nii.gz").get_fdata()
# Im = nib.load("MOVING_IMAGE.nii.gz").get_fdata()
# 2D
# Is = np.squeeze(Is[:, 128, :])
# Im = np.squeeze(Im[:, 128, :])

# Run the algorithm
ImWarped, velocityField = Demons(
    Is,
    Im,
    nLevels=nLevels,
    diffusionSigmas=diffusionSigmas,
    fluidSigmas=fluidSigmas,
    max_iter=max_iter,
    alpha=alpha,
    cThresh=cThresh,
    variant=variant,
    diffeomorphic=diffeomorphic,
    compositionType=compositionType,
    device=device,
).run()
