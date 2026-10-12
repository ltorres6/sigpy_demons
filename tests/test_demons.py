"""The Demons multi-resolution driver, including end-to-end registrations."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from conftest import blobs, interior, requires_gpu, to_np
from pydemons import Demons

VARIANTS = ["passive", "active", "inverse_consistent"]


def run(Is, Im, xp=np, **kwargs):
    opts = dict(n_levels=2, max_iter=200, diffusion_sigma=1.0, fluid_sigma=1.0, tol=0.0, progress=False)
    opts.update(kwargs)
    warped, field = Demons(xp.asarray(Is), xp.asarray(Im), **opts).run()
    return to_np(warped), to_np(field)


# --------------------------------------------------------------------------- setup


@pytest.mark.parametrize("n_levels", [1, 3, 4])
@pytest.mark.parametrize("shape", [(50, 37), (20, 33, 17)])
def test_pyramid_geometry(n_levels, shape):
    d = Demons(np.ones(shape), np.ones(shape), n_levels=n_levels, max_iter=80)
    assert all(p % 2**n_levels == 0 and p >= s + 40 for p, s in zip(d.padded_shape, shape))
    assert d.scales == [2 ** (n_levels - 1 - i) for i in range(n_levels)]
    assert d.level_shapes[-1] == d.padded_shape
    assert all(o == tuple(p // s for p in d.padded_shape) for o, s in zip(d.level_shapes, d.scales))
    assert d.iterations == [80 // 2**i for i in range(n_levels)]


@pytest.mark.parametrize("option", [{"variant": "nonsense"}, {"composition": "C", "diffeomorphic": True}])
def test_invalid_options_raise_on_construction(option):
    with pytest.raises(ValueError):
        Demons(np.ones((8, 8)), np.ones((8, 8)), **option)


# --------------------------------------------------------------------------- end-to-end


@pytest.mark.slow
@pytest.mark.parametrize("variant", VARIANTS)
def test_output_shapes_and_residual_reduction(variant):
    shape = (64, 64)
    Is = blobs(shape)
    Im = np.roll(Is, 3, axis=0)
    warped, field = run(Is, Im, variant=variant)
    assert warped.shape == shape and field.shape == (2,) + shape
    assert np.all(np.isfinite(field))
    assert np.mean((Is - warped) ** 2) < 0.2 * np.mean((Is - Im) ** 2)


@pytest.mark.slow
@pytest.mark.parametrize("variant", VARIANTS)
def test_recovers_translation(variant):
    shape = (64, 64)
    Is = blobs(shape)
    Im = np.roll(Is, 3, axis=0)
    _, field = run(Is, Im, variant=variant)
    inner = interior(2, 20)
    assert abs(field[0][inner].mean() - 3.0) < 0.25
    assert abs(field[1][inner].mean()) < 0.25


@pytest.mark.slow
def test_3d_registration_runs():
    shape = (24, 24, 24)
    Is = blobs(shape)
    Im = np.roll(Is, 2, axis=1)
    warped, field = run(Is, Im, max_iter=40)
    assert warped.shape == shape and field.shape == (3,) + shape
    assert np.mean((Is - warped) ** 2) < 0.5 * np.mean((Is - Im) ** 2)


@pytest.mark.slow
def test_fluid_sigma_changes_result():
    shape = (64, 64)
    Is = blobs(shape)
    Im = np.roll(Is, 3, axis=0)
    _, a = run(Is, Im, fluid_sigma=0.0, max_iter=50)
    _, b = run(Is, Im, fluid_sigma=3.0, max_iter=50)
    assert not np.allclose(a, b)


@pytest.mark.slow
def test_run_does_not_plot():
    shape = (32, 32)
    Is = blobs(shape)
    plt.close("all")
    Demons(Is, np.roll(Is, 1, axis=0), n_levels=1, max_iter=5, progress=False).run()
    opened = len(plt.get_fignums())
    plt.close("all")
    assert opened == 0


@pytest.mark.slow
@pytest.mark.gpu
@requires_gpu
@pytest.mark.parametrize("variant", VARIANTS)
def test_gpu_matches_cpu(variant):
    shape = (64, 64)
    Is = blobs(shape)
    Im = np.roll(Is, 3, axis=0)
    w_cpu, f_cpu = run(Is, Im, variant=variant, max_iter=60)
    import cupy

    w_gpu, f_gpu = run(Is, Im, xp=cupy, variant=variant, max_iter=60)
    w_gpu, f_gpu = to_np(w_gpu), to_np(f_gpu)
    # GPU runs in single precision; hard intensity thresholds let tiny differences flip a few voxels.
    assert np.abs(f_cpu - f_gpu).mean() < 1e-3 and np.abs(f_cpu - f_gpu).max() < 0.05
    assert np.abs(w_cpu - w_gpu).mean() < 1e-3


@pytest.mark.slow
@pytest.mark.parametrize("variant", VARIANTS)
def test_diffeomorphic_run_returns_displacement(variant):
    from pydemons import warp
    from pydemons.fields import exp_field

    shape = (48, 48)
    Is = blobs(shape)
    Im = np.roll(Is, 2, axis=0)
    d = Demons(Is, Im, n_levels=2, max_iter=40, diffeomorphic=True, variant=variant, progress=False)
    warped, field = d.run()
    # run() exponentiates on the padded domain, so only the cropped border may differ.
    inner = (slice(None),) + interior(2, 6)
    assert np.allclose(field[inner], exp_field(d.velocity)[inner])
    assert np.allclose(warped, warp(d.moving, field))


@pytest.mark.slow
@pytest.mark.parametrize("xp_name", ["numpy", pytest.param("cupy", marks=[pytest.mark.gpu, requires_gpu])])
@pytest.mark.parametrize("variant", VARIANTS)
def test_float32_matches_float64(xp_name, variant):
    import importlib

    xp = importlib.import_module(xp_name)
    shape = (64, 64)
    Is = blobs(shape)
    Im = np.roll(Is, 3, axis=0)
    d32 = Demons(xp.asarray(Is), xp.asarray(Im), n_levels=2, max_iter=60, tol=0.0, variant=variant,
                 dtype="float32", progress=False)
    w32, f32 = d32.run()
    assert w32.dtype == np.float32 and f32.dtype == np.float32
    _, f64 = run(Is, Im, variant=variant, max_iter=60)
    assert np.abs(to_np(f32) - f64).mean() < 1e-3
