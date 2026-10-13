"""Per-level timing of one registration (CPU or GPU), to see where time goes.

    uv run python benchmarks/profile_levels.py [--gpu] [--dtype float32]
"""

from __future__ import annotations

import argparse
import time

import pydemons.registration as registration
from bench_3d import make_pair
from pydemons import Demons, to_gpu


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--variant", default="passive")
    args = parser.parse_args()

    fixed, moving, _ = make_pair((256, 256, 128))
    if args.gpu:
        import cupy

        fixed, moving = to_gpu(fixed), to_gpu(moving)
    sync = cupy.cuda.Device().synchronize if args.gpu else (lambda: None)
    opts = dict(tol=0.0, dtype=args.dtype, variant=args.variant, progress=False)
    Demons(fixed[:64, :64, :32], moving[:64, :64, :32], n_levels=2, max_iter=4, **opts).run()  # warm-up

    marks = []

    class Timed(registration.tqdm):
        def __init__(self, *a, **k):
            sync()
            self.t0, self.label = time.perf_counter(), (k["desc"], k["total"])
            super().__init__(*a, **k)

        def __exit__(self, *exc):
            sync()
            marks.append((*self.label, time.perf_counter() - self.t0))
            return super().__exit__(*exc)

    registration.tqdm = Timed
    sync()
    start = time.perf_counter()
    Demons(fixed, moving, n_levels=3, max_iter=200, **opts).run()
    sync()
    total = time.perf_counter() - start
    print(f"{'GPU' if args.gpu else 'CPU'} {args.dtype} {args.variant}: total {total:.2f} s")
    for desc, n, secs in marks:
        print(f"  {desc:28s} {n:4d} it {secs:6.2f} s {secs / n * 1e3:7.1f} ms/it")
    print(f"  {'outside iteration loops':38s} {total - sum(m[2] for m in marks):6.2f} s")


if __name__ == "__main__":
    main()
