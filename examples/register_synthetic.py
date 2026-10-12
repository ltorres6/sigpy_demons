"""Register a synthetic 2D image pair and save a summary figure.

    uv run --extra plot python examples/register_synthetic.py [--gpu] [--out registration.png]

To register your own volumes, replace ``make_pair`` with e.g. nibabel:

    fixed = nib.load("fixed.nii.gz").get_fdata()
    moving = nib.load("moving.nii.gz").get_fdata()
"""

from __future__ import annotations

import argparse

import numpy as np

from pydemons import Demons, is_gpu_available, to_gpu, to_numpy, warp
from pydemons.filters import gaussian_smooth


def make_pair(shape=(128, 128), max_displacement=4.0, seed=0):
    """Fixed image of smooth blobs, and the same image under a smooth random warp."""
    rng = np.random.default_rng(seed)
    fixed = gaussian_smooth(rng.random(shape), 4, mode="reflect")
    fixed = (fixed - fixed.min()) / (fixed.max() - fixed.min())
    field = gaussian_smooth(rng.standard_normal((len(shape),) + shape), 16, axes=range(1, len(shape) + 1), mode="reflect")
    field *= max_displacement / np.abs(field).max()
    # moving(x + field(x)) == fixed(x)  <=>  moving = fixed warped by the inverse field (approximated by -field)
    moving = warp(fixed, -field)
    return fixed, moving


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gpu", action="store_true", help="run on the GPU (requires the 'gpu' extra)")
    parser.add_argument("--out", default="registration.png", help="where to save the figure")
    args = parser.parse_args()

    fixed, moving = make_pair()
    if args.gpu:
        if not is_gpu_available():
            parser.error("--gpu requested but CuPy/CUDA is not available")
        fixed, moving = to_gpu(fixed), to_gpu(moving)

    warped, field = Demons(
        fixed,
        moving,
        n_levels=3,
        diffusion_sigma=1.5,
        fluid_sigma=1.5,
        max_iter=400,
        alpha=2.0,
        variant="inverse_consistent",
    ).run()

    fixed, moving, warped = to_numpy(fixed), to_numpy(moving), to_numpy(warped)
    before = np.mean((fixed - moving) ** 2)
    after = np.mean((fixed - warped) ** 2)
    print(f"MSE before: {before:.2e}  after: {after:.2e}  ({before / after:.1f}x lower)")

    from pydemons.plotting import plot_registration

    plot_registration(fixed, moving, warped, field).savefig(args.out, dpi=120)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
