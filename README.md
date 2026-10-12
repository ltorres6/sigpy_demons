# pydemons

Demons deformable image registration for 2D and 3D images, on CPU (NumPy) or
GPU (CuPy). The original idea came from [sigpy](https://github.com/mikgroup/sigpy);
the package no longer depends on it.

Every function runs on the device of the arrays you give it: NumPy arrays run
on the CPU, CuPy arrays on the GPU. There is no `device=` argument.

## Install

```bash
uv sync                      # CPU only
uv sync --extra gpu          # + CuPy for CUDA 13 GPUs
uv sync --extra plot         # + matplotlib for pydemons.plotting
```

## Usage

```python
from pydemons import Demons, to_gpu, to_numpy

warped, field = Demons(
    fixed,                   # NumPy array, or to_gpu(fixed) to run on the GPU
    moving,
    n_levels=3,              # resolution pyramid levels
    diffusion_sigma=1.5,     # smoothing of the accumulated field (voxels)
    fluid_sigma=1.5,         # smoothing of each update (voxels)
    max_iter=400,            # iterations at the coarsest level, halved per level
    alpha=2.0,               # step-size control: |update| <= 1 / (2 * alpha)
    variant="inverse_consistent",  # or "passive", "active"
    diffeomorphic=False,     # log-domain (stationary velocity field) demons
    tol=0.0,                 # 0 runs every iteration and skips the energy (fastest)
    dtype="float32",         # ~2x faster on GPU; default float64
).run()
```

`field` has shape `(ndim, *image.shape)` and maps `moving(x + field(x))` onto
`fixed(x)`.

To see the images, residuals, field and Jacobian determinant:

```python
from pydemons.plotting import plot_registration
plot_registration(fixed, moving, warped, field).savefig("registration.png")
```

A complete example on synthetic data:

```bash
uv run --extra plot python examples/register_synthetic.py [--gpu]
```

## Performance

`benchmarks/bench_3d.py` registers a 256×256×128 volume under a smooth
synthetic warp (3 levels, 200/100/50 iterations, `tol=0`) on an RTX 4090:

| variant              | float64 | float32 |
|----------------------|--------:|--------:|
| `passive`            |  2.0 s  |  1.0 s  |
| `inverse_consistent` |  3.8 s  |  2.0 s  |

Mean field error is 0.13–0.14 voxels in both precisions. With `tol > 0` each
iteration also computes the energy for the stopping rule, which costs about a
third more per iteration.

```bash
uv run python benchmarks/bench_3d.py --gpu --dtype float32 --variant passive
```

## Tests

```bash
uv run pytest                 # CPU and, if CuPy + CUDA are present, GPU
uv run pytest -m "not slow"   # skip end-to-end registrations
```

Known bugs are recorded as strict `xfail` tests. Fixing a bug makes its test
pass, which then fails the suite until the `xfail` marker is removed.
