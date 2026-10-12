"""Time a 3D registration (default 256x256x128) on CPU or GPU.

    uv run python benchmarks/bench_3d.py [--gpu] [--shape 256 256 128] [--iters 200]
"""

from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.ndimage import gaussian_filter

from pydemons import Demons, to_gpu, to_numpy, warp


def make_pair(shape, amplitude=4.0, seed=0):
    rng = np.random.default_rng(seed)
    moving = gaussian_filter(rng.random(shape, dtype=np.float32), 3).astype(np.float64)
    moving = (moving - moving.min()) / np.ptp(moving)
    small = tuple(max(s // 8, 4) for s in shape)
    coarse = np.stack([gaussian_filter(rng.standard_normal(small), 1.5) for _ in shape])
    from scipy.ndimage import zoom

    field = np.stack([zoom(c, [s / c_ for s, c_ in zip(shape, small)], order=1) for c in coarse])
    field *= amplitude / np.abs(field).max()
    return warp(moving, field), moving, field


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--shape", type=int, nargs="+", default=[256, 256, 128])
    parser.add_argument("--iters", type=int, default=200, help="iterations at the coarsest level")
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--variant", default="passive")
    args = parser.parse_args()

    fixed, moving, truth = make_pair(tuple(args.shape))
    if args.gpu:
        fixed, moving = to_gpu(fixed), to_gpu(moving)
        import cupy

        Demons(fixed[:64, :64, :32], moving[:64, :64, :32], n_levels=2, max_iter=4, progress=False).run()  # warm-up
        cupy.cuda.Device().synchronize()

    start = time.perf_counter()
    warped, field = Demons(
        fixed, moving, n_levels=args.levels, max_iter=args.iters, tol=0.0, variant=args.variant, progress=False
    ).run()
    if args.gpu:
        cupy.cuda.Device().synchronize()
    elapsed = time.perf_counter() - start

    fixed, moving, warped, field = (to_numpy(a) for a in (fixed, moving, warped, field))
    inner = (slice(None),) + tuple(slice(16, -16) for _ in args.shape)
    print(
        f"{'GPU' if args.gpu else 'CPU'} {tuple(args.shape)} levels={args.levels} iters={args.iters}: {elapsed:.2f} s | "
        f"MSE {np.mean((fixed - moving) ** 2):.2e} -> {np.mean((fixed - warped) ** 2):.2e} | "
        f"field error {np.abs(field - truth)[inner].mean():.3f} vox"
    )


if __name__ == "__main__":
    main()
