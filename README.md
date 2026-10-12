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

pydemons is one implementation in plain NumPy/CuPy array code; nothing is
chosen per device (no SciPy or cupyx at runtime). Separable smoothing and
resampling run as matrix multiplies (multi-threaded BLAS on CPU, cuBLAS on
GPU). Interpolation and the demons force are elementwise array code, which is
single-threaded on CPU and not fused into one kernel on GPU.

`benchmarks/bench_3d.py`: smooth synthetic warp, 3 levels, 200/100/50
iterations, `tol=0`, RTX 4090 / 24-core CPU:

| volume      | variant              | GPU float32 | GPU float64 | CPU float32 |
|-------------|----------------------|------------:|------------:|------------:|
| 256×256×128 | `passive`            |  4.6 s      |  8.0 s      |  275 s      |
| 256×256×128 | `inverse_consistent` |  5.6 s      | 10.0 s      |             |
| 256×256×256 | `passive`            |  5.6 s      | 12.9 s      |             |
| 256×256×256 | `inverse_consistent` |  7.1 s      | 16.1 s      |             |

Mean field error is 0.13–0.14 voxels (256×256×128) and 0.11 voxels
(256×256×256), identical across devices and precisions. With `tol > 0` each
iteration also computes the energy for the stopping rule.

The first run at a new size or dtype on GPU also pays a one-off CuPy kernel
compilation (cached on disk afterwards).

### Memory

Interpolation, the forces and the energy work in slabs along the first axis
(`pydemons.interpolate.SLAB_VOXELS`), smoothing works one field component at a
time, and the loop updates the field in place. Peak memory for a 256×256×256
`passive` registration (padded internally to 304³):

| device | float32  | float64  |
|--------|---------:|---------:|
| GPU    | 1.28 GB  | 2.52 GB  |
| CPU    | 1.32 GB  |          |

That is about four volumes' worth: the two images, the field, the update and
one slab of temporaries. On GPU, CuPy's memory pool additionally keeps freed
blocks for reuse (up to ~2.4 GB held in float32); release them with
`cupy.get_default_memory_pool().free_all_blocks()`.

```bash
uv run python benchmarks/bench_3d.py --gpu --dtype float32 --variant passive
uv run python benchmarks/memory_profile.py --gpu --dtype float32 --shape 256 256 256
uv run python benchmarks/profile_levels.py --gpu
```

## Tests

```bash
uv run pytest                 # CPU and, if CuPy + CUDA are present, GPU
uv run pytest -m "not slow"   # skip end-to-end registrations
```

Known bugs are recorded as strict `xfail` tests. Fixing a bug makes its test
pass, which then fails the suite until the `xfail` marker is removed.
