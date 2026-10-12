import numpy as np

import energy
from conftest import smooth_field, to_np


def test_energy_is_scalar(xp):
    diff = xp.asarray(np.random.default_rng(0).standard_normal((16, 16)))
    e = energy.energy(diff, xp.zeros((2, 16, 16)), alpha=2.0)
    assert np.ndim(to_np(e)) == 0
    assert np.isfinite(float(e))


def test_similarity_term_is_mean_squared_difference(xp):
    # With the deformation held fixed, energy differs only by mean(diff**2).
    diff = np.random.default_rng(1).standard_normal((16, 12))
    field = xp.asarray(smooth_field((16, 12), amplitude=1.0))
    e_diff = float(energy.energy(xp.asarray(diff), field, alpha=2.0))
    e_zero = float(energy.energy(xp.zeros((16, 12)), field, alpha=2.0))
    assert np.isclose(e_diff - e_zero, np.mean(diff**2))


def test_energy_decreases_with_smaller_residual(xp):
    diff = np.random.default_rng(2).standard_normal((10, 10, 10))
    field = xp.zeros((3, 10, 10, 10))
    assert float(energy.energy(xp.asarray(0.5 * diff), field, 2.0)) < float(energy.energy(xp.asarray(diff), field, 2.0))
