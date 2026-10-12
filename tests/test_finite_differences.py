import numpy as np
import pytest

from conftest import grid, interior, to_np
from pydemons.differences import gradient


@pytest.mark.parametrize("shape", [(24, 20), (12, 10, 8)])
def test_gradient_shape(xp, shape):
    out = gradient(xp.zeros(shape))
    assert out.shape == (len(shape),) + shape


@pytest.mark.parametrize("shape", [(24, 20), (12, 10, 8)])
def test_gradient_of_constant_is_zero(xp, shape):
    assert np.allclose(to_np(gradient(xp.full(shape, 3.5))), 0.0)


@pytest.mark.parametrize("shape", [(24, 20), (12, 10, 8)])
def test_gradient_of_ramp_equals_slope(xp, shape):
    slopes = [0.5, -2.0, 1.25][: len(shape)]
    g = grid(shape)
    ramp = sum(s * g[i] for i, s in enumerate(slopes))
    out = to_np(gradient(xp.asarray(ramp)))
    for i, s in enumerate(slopes):
        assert np.allclose(out[i][interior(len(shape), 1)], s)


def test_gradient_is_periodic_central_difference(xp):
    # Boundaries wrap, so a periodic sinusoid has an exact closed-form central difference everywhere.
    n, k = 32, 3
    x = np.arange(n, dtype=np.float64)
    w = 2 * np.pi * k / n
    signal = np.tile(np.sin(w * x)[:, None], (1, 4))
    out = to_np(gradient(xp.asarray(signal)))
    expected = np.sin(w) * np.cos(w * x)
    assert np.allclose(out[0][:, 0], expected)
    assert np.allclose(out[1], 0.0)
