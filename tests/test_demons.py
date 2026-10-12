"""The Demons multi-resolution driver, including end-to-end registrations."""

import matplotlib.pyplot as plt
import numpy as np
import pytest

from conftest import blobs, interior, requires_gpu, to_np
from demons import Demons

VARIANTS = ["passive", "active", "inverseConsistent"]


def run(Is, Im, device=-1, **kwargs):
    opts = dict(nLevels=2, max_iter=200, diffusionSigmas=1.0, fluidSigmas=1.0, cThresh=0.0, device=device)
    opts.update(kwargs)
    warped, field = Demons(Is, Im, **opts).run()
    plt.close("all")
    return to_np(warped), to_np(field)


# --------------------------------------------------------------------------- setup


@pytest.mark.parametrize("n_levels", [1, 3, 4])
@pytest.mark.parametrize("shape", [(50, 37), (20, 33, 17)])
def test_pyramid_geometry(n_levels, shape):
    d = Demons(np.ones(shape), np.ones(shape), nLevels=n_levels, max_iter=80)
    assert all(p % 2**n_levels == 0 and p >= s + 40 for p, s in zip(d.paddedShape, shape))
    assert d.scales == [2 ** (n_levels - 1 - i) for i in range(n_levels)]
    assert d.oshapes[-1] == d.paddedShape
    assert all(o == tuple(p // s for p in d.paddedShape) for o, s in zip(d.oshapes, d.scales))
    assert d.iterations == [80 // 2**i for i in range(n_levels)]


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
@pytest.mark.xfail(reason="displacements are not rescaled between pyramid levels; translation is underestimated")
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
@pytest.mark.xfail(reason="fluidSigmas is ignored: run() uses the diffusion sigma for both")
def test_fluid_sigma_changes_result():
    shape = (64, 64)
    Is = blobs(shape)
    Im = np.roll(Is, 3, axis=0)
    _, a = run(Is, Im, fluidSigmas=0.0, max_iter=50)
    _, b = run(Is, Im, fluidSigmas=3.0, max_iter=50)
    assert not np.allclose(a, b)


@pytest.mark.slow
@pytest.mark.xfail(reason="run() always opens five figures")
def test_run_does_not_plot():
    shape = (32, 32)
    Is = blobs(shape)
    plt.close("all")
    Demons(Is, np.roll(Is, 1, axis=0), nLevels=1, max_iter=5).run()
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
    w_gpu, f_gpu = run(Is, Im, variant=variant, max_iter=60, device=0)
    # GPU runs in single precision; hard intensity thresholds let tiny differences flip a few voxels.
    assert np.abs(f_cpu - f_gpu).mean() < 1e-3 and np.abs(f_cpu - f_gpu).max() < 0.05
    assert np.abs(w_cpu - w_gpu).mean() < 1e-3
