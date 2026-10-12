"""Peak memory of one registration and of each operation, on CPU or GPU.

    uv run python benchmarks/memory_profile.py [--gpu] [--dtype float32] [--shape 256 256 256] [--ops-only]

GPU: peak bytes held by live CuPy arrays (via a memory hook) and the most the
memory pool had cached. CPU: peak traced NumPy allocations (tracemalloc).
"""

from __future__ import annotations

import argparse
import contextlib
import tracemalloc

import numpy as np

from bench_3d import make_pair
from pydemons import Demons, to_gpu
from pydemons.differences import gradient
from pydemons.fields import compose
from pydemons.filters import gaussian_smooth
from pydemons.forces import demons_force
from pydemons.resample import zoom
from pydemons.warp import warp

MB = 1024**2


class _PeakHook:
    """CuPy memory hook tracking live and peak allocated bytes."""

    def __init__(self):
        import cupy

        class Hook(cupy.cuda.memory_hook.MemoryHook):
            name = "PeakHook"

            def __init__(hook):
                hook.live = hook.peak = 0

            def malloc_postprocess(hook, device_id, size, mem_size, mem_ptr, pmem_id):
                hook.live += mem_size
                hook.peak = max(hook.peak, hook.live)

            def free_postprocess(hook, device_id, mem_size, mem_ptr, pmem_id):
                hook.live -= mem_size

        self.hook = Hook()


@contextlib.contextmanager
def measure(gpu):
    """Yield a dict filled with ``peak`` (MB above the starting level) on exit."""
    result = {}
    if gpu:
        import cupy

        cupy.cuda.Device().synchronize()
        pool = cupy.get_default_memory_pool()
        pool.free_all_blocks()
        tracker = _PeakHook().hook
        with tracker:
            start = tracker.live
            yield result
            cupy.cuda.Device().synchronize()
        result["peak"] = (tracker.peak - start) / MB
        result["pool"] = pool.total_bytes() / MB
    else:
        tracemalloc.start()
        tracemalloc.reset_peak()
        start = tracemalloc.get_traced_memory()[0]
        yield result
        result["peak"] = (tracemalloc.get_traced_memory()[1] - start) / MB
        tracemalloc.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--shape", type=int, nargs="+", default=[256, 256, 256])
    parser.add_argument("--variant", default="passive")
    parser.add_argument("--ops-only", action="store_true", help="skip the full registration")
    args = parser.parse_args()

    fixed, moving, _ = make_pair(tuple(args.shape))
    fixed, moving = fixed.astype(args.dtype), moving.astype(args.dtype)
    if args.gpu:
        fixed, moving = to_gpu(fixed), to_gpu(moving)
    xp = Demons(fixed, moving, n_levels=1, max_iter=1, progress=False).xp
    d = Demons(fixed, moving, n_levels=3, max_iter=1, dtype=args.dtype, progress=False)
    shape = d.padded_shape
    voxels = int(np.prod(shape))
    itemsize = np.dtype(args.dtype).itemsize
    print(f"{'GPU' if args.gpu else 'CPU'} {args.dtype} {tuple(args.shape)} -> padded {shape}")
    print(f"  one image {voxels * itemsize / MB:7.0f} MB, one {len(shape)}-component field {len(shape) * voxels * itemsize / MB:7.0f} MB")

    rng = np.random.default_rng(0)
    img = xp.asarray(rng.random(shape).astype(args.dtype))
    fld = xp.asarray(((rng.random((len(shape),) + shape) - 0.5) * 4).astype(args.dtype))
    grad = gradient(img)
    grad_sq = (grad**2).sum(axis=0)
    half = xp.ascontiguousarray(img[(slice(None, None, 2),) * len(shape)])
    ops = [
        ("warp", lambda: warp(img, fld)),
        ("compose", lambda: compose(fld, fld)),
        ("demons_force passive", lambda: demons_force(img, img, fld, 2.0, fixed_grad=grad, fixed_grad_sq=grad_sq, compute_energy=False)),
        ("gaussian_smooth field", lambda: gaussian_smooth(fld, 1.0, axes=tuple(range(1, len(shape) + 1)))),
        ("gradient", lambda: gradient(img)),
        ("zoom down 1/2", lambda: zoom(img, 0.5)),
        ("zoom up x2", lambda: zoom(half, 2.0)),
    ]
    print("  peak extra memory per call (output included):")
    for name, fn in ops:
        fn()
        with measure(args.gpu) as m:
            out = fn()
            del out
        print(f"    {name:24s} {m['peak']:8.0f} MB")

    if not args.ops_only:
        del img, fld, grad, grad_sq, half
        with measure(args.gpu) as m:
            Demons(fixed, moving, n_levels=3, max_iter=20, tol=0.0, dtype=args.dtype, variant=args.variant, progress=False).run()
        extra = f", pool cached {m['pool']:.0f} MB" if args.gpu else ""
        print(f"  full registration ({args.variant}, 20/10/5 iterations): peak {m['peak']:.0f} MB{extra}")


if __name__ == "__main__":
    main()
