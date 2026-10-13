import numpy as np

from conftest import interior, smooth_field, to_np
from pydemons.energy import energy


def test_energy_is_scalar(xp):
    diff = xp.asarray(np.random.default_rng(0).standard_normal((16, 16)))
    e = energy(diff, xp.zeros((2, 16, 16)), alpha=2.0)
    assert np.ndim(to_np(e)) == 0
    assert np.isfinite(float(e))


def test_similarity_term_is_mean_squared_difference(xp):
    # With the deformation held fixed, energy differs only by mean(diff**2).
    diff = np.random.default_rng(1).standard_normal((16, 12))
    field = xp.asarray(smooth_field((16, 12), amplitude=1.0))
    e_diff = float(energy(xp.asarray(diff), field, alpha=2.0))
    e_zero = float(energy(xp.zeros((16, 12)), field, alpha=2.0))
    assert np.isclose(e_diff - e_zero, np.mean(diff**2))


def test_energy_decreases_with_smaller_residual(xp):
    diff = np.random.default_rng(2).standard_normal((10, 10, 10))
    field = xp.zeros((3, 10, 10, 10))
    assert float(energy(xp.asarray(0.5 * diff), field, 2.0)) < float(energy(xp.asarray(diff), field, 2.0))


def test_rmse(xp):
    from pydemons.energy import rmse

    assert np.isclose(float(rmse(xp.asarray([1.0, -1.0, 1.0, -1.0]))), 1.0)
    assert np.isclose(float(rmse(xp.asarray([3.0, 4.0]))), np.sqrt(12.5))


def test_harmonic_energy(xp):
    from conftest import grid

    from pydemons.energy import harmonic_energy

    shape = (20, 20)
    translation = np.ones((2,) + shape)
    assert np.isclose(float(harmonic_energy(xp.asarray(translation))), 0.0)
    # u_0 = a x_0 + b x_1 has |grad u|^2 = a^2 + b^2 away from the wrapped border.
    g = grid(shape)
    field = np.zeros((2,) + shape)
    field[0] = 0.3 * g[0] - 0.4 * g[1]
    assert np.isclose(float(harmonic_energy(xp.asarray(field), mask=interior(2, 1))), 0.25)


def test_quantity_of_update(xp):
    from pydemons.energy import quantity_of_update

    field = xp.asarray(np.full((2, 4, 4), 2.0))
    assert np.isclose(float(quantity_of_update(0.1 * field, field)), 0.1)
    assert float(quantity_of_update(field, xp.zeros_like(field))) == np.inf
