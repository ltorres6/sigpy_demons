"""Composition, exponentiation and Lie bracket of displacement fields."""

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

from conftest import grid, interior, smooth_field, to_np
from pydemons.fields import compose, exp_field, lie_bracket

SHAPES = [(32, 28), (16, 14, 12)]


def constant_field(shape, values):
    field = np.zeros((len(shape),) + shape)
    for i, v in zip(range(len(shape)), values):
        field[i] = v
    return field


# --------------------------------------------------------------------------- compose


@pytest.mark.parametrize("shape", SHAPES)
def test_compose_zero_fields(xp, shape):
    zero = xp.zeros((len(shape),) + shape)
    assert np.allclose(to_np(compose(zero, zero)), 0.0)


@pytest.mark.parametrize("shape", SHAPES)
def test_compose_with_zero_is_identity(xp, shape):
    f = xp.asarray(smooth_field(shape, amplitude=2.0))
    zero = xp.zeros_like(f)
    inner = (slice(None),) + interior(len(shape), 3)
    assert np.allclose(to_np(compose(f, zero)), to_np(f))
    assert np.allclose(to_np(compose(zero, f))[inner], to_np(f)[inner])


@pytest.mark.parametrize("shape", SHAPES)
def test_compose_translations_add(xp, shape):
    a = constant_field(shape, [1.5, -0.5, 0.25])
    b = constant_field(shape, [-0.75, 2.0, 1.0])
    out = to_np(compose(xp.asarray(a), xp.asarray(b)))
    inner = (slice(None),) + interior(len(shape), 4)
    assert np.allclose(out[inner], (a + b)[inner])


@pytest.mark.parametrize("shape", SHAPES)
def test_compose_translations_add_at_border(xp, shape):
    # Sampling past the edge must extend the field, not treat it as zero displacement.
    a = constant_field(shape, [2.5, -1.5, 1.0])
    b = constant_field(shape, [-0.75, 2.0, 1.0])
    out = to_np(compose(xp.asarray(a), xp.asarray(b)))
    assert np.allclose(out, a + b)


@pytest.mark.parametrize("shape", SHAPES)
def test_compose_samples_second_field_along_first(xp, shape):
    # compose(f1, f2)(x) = f1(x) + f2(x + f1(x)), linearly interpolated.
    f1 = smooth_field(shape, amplitude=2.0, seed=1)
    f2 = smooth_field(shape, amplitude=2.0, seed=2)
    coords = grid(shape) + f1
    expected = f1 + np.stack([map_coordinates(f2[i], coords, order=1) for i in range(len(shape))])
    out = to_np(compose(xp.asarray(f1), xp.asarray(f2)))
    inner = (slice(None),) + interior(len(shape), 4)
    assert np.allclose(out[inner], expected[inner], atol=1e-6)


@pytest.mark.parametrize("shape", SHAPES)
def test_compose_log_domain_first_order_is_sum(xp, shape):
    f1 = smooth_field(shape, 1.0, seed=3)
    f2 = smooth_field(shape, 1.0, seed=4)
    out = compose(xp.asarray(f1), xp.asarray(f2), diffeomorphic=True, composition="A")
    assert np.allclose(to_np(out), f1 + f2)


@pytest.mark.parametrize("shape", SHAPES)
def test_compose_log_domain_second_order_adds_bracket(xp, shape):
    f1 = xp.asarray(smooth_field(shape, 1.0, seed=5))
    f2 = xp.asarray(smooth_field(shape, 1.0, seed=6))
    out = compose(f1, f2, diffeomorphic=True, composition="B")
    expected = f1 + f2 + lie_bracket(f1, f2)
    assert np.allclose(to_np(out), to_np(expected))


def test_compose_rejects_unknown_type():
    f = np.zeros((2, 8, 8))
    with pytest.raises(ValueError):
        compose(f, f, diffeomorphic=True, composition="C")


# --------------------------------------------------------------------------- exp_field


@pytest.mark.parametrize("shape", SHAPES)
def test_exp_of_zero_is_zero(xp, shape):
    zero = xp.zeros((len(shape),) + shape)
    assert np.allclose(to_np(exp_field(zero)), 0.0)


@pytest.mark.parametrize("shape", SHAPES)
def test_exp_of_large_translation_is_translation(xp, shape):
    v = constant_field(shape, [2.0, -1.5, 1.0])
    out = to_np(exp_field(xp.asarray(v)))
    inner = (slice(None),) + interior(len(shape), 5)
    assert np.allclose(out[inner], v[inner])


@pytest.mark.parametrize("shape", SHAPES)
def test_exp_of_small_translation_is_translation(xp, shape):
    v = constant_field(shape, [0.1, -0.05, 0.02])
    out = to_np(exp_field(xp.asarray(v)))
    inner = (slice(None),) + interior(len(shape), 3)
    assert np.allclose(out[inner], v[inner])


@pytest.mark.parametrize("amplitude", [0.3, 3.0])
def test_exp_inverse_is_exp_of_negation(xp, amplitude):
    # exp(v) composed with exp(-v) is the identity (zero displacement).
    shape = (48, 48)
    v = smooth_field(shape, amplitude, seed=7)
    fwd = exp_field(xp.asarray(v))
    bwd = exp_field(xp.asarray(-v))
    residual = to_np(compose(fwd, bwd))
    inner = (slice(None),) + interior(2, 8)
    assert np.abs(residual[inner]).max() < 0.05 * amplitude


@pytest.mark.parametrize("a", [0.02, 0.1, 0.2])
def test_exp_of_linear_field_matches_flow(xp, a):
    # dx/dt = a (x - c) integrates to x(1) - x(0) = (x - c)(e^a - 1). Scaling and squaring
    # with steps of at most 0.5 voxel is first-order accurate: about 1% relative error here.
    shape = (40, 40)
    g = grid(shape)
    v = np.zeros((2,) + shape)
    v[0] = a * (g[0] - 20)
    out = to_np(exp_field(xp.asarray(v)))
    expected = (g[0] - 20) * np.expm1(a)
    inner = interior(2, 4)
    assert np.abs(out[0][inner] - expected[inner]).max() < 0.02 * np.abs(expected[inner]).max()
    assert np.abs(out[1][inner]).max() < 1e-6


# --------------------------------------------------------------------------- lie_bracket


@pytest.mark.parametrize("shape", SHAPES)
def test_bracket_of_field_with_itself_is_zero(xp, shape):
    v = xp.asarray(smooth_field(shape, 1.0, seed=8))
    assert np.allclose(to_np(lie_bracket(v, v)), 0.0)


@pytest.mark.parametrize("shape", SHAPES)
def test_bracket_is_antisymmetric(xp, shape):
    u = xp.asarray(smooth_field(shape, 1.0, seed=9))
    v = xp.asarray(smooth_field(shape, 1.0, seed=10))
    assert np.allclose(to_np(lie_bracket(u, v)), -to_np(lie_bracket(v, u)))


@pytest.mark.parametrize("shape", SHAPES)
def test_bracket_of_translations_is_zero(xp, shape):
    u = xp.asarray(constant_field(shape, [1.0, 2.0, 3.0]))
    v = xp.asarray(constant_field(shape, [-1.0, 0.5, 2.0]))
    assert np.allclose(to_np(lie_bracket(u, v)), 0.0)


@pytest.mark.parametrize("shape", SHAPES)
def test_bracket_matches_analytic(xp, shape):
    # lie_bracket(u, v) = 0.5 * (Jv u - Ju v). With u = e_0 constant and v_1 = x_0, Jv u = e_1.
    ndim = len(shape)
    u = constant_field(shape, [1.0])
    v = np.zeros((ndim,) + shape)
    v[1] = grid(shape)[0]
    out = to_np(lie_bracket(xp.asarray(u), xp.asarray(v)))
    expected = np.zeros_like(out)
    expected[1] = 0.5
    inner = (slice(None),) + interior(ndim, 1)
    assert np.allclose(out[inner], expected[inner])


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_bracket_improves_bch_approximation(xp, seed):
    # Applying exp(f1) then exp(f2) is exp(f1 + f2 + lie_bracket(f1, f2)) to second order,
    # so the bracket term must reduce the error of the first-order approximation.
    shape = (64, 64)
    f1 = xp.asarray(smooth_field(shape, 3.0, seed=2 * seed))
    f2 = xp.asarray(smooth_field(shape, 3.0, seed=2 * seed + 1))
    exact = to_np(compose(exp_field(f1), exp_field(f2)))
    first = to_np(exp_field(compose(f1, f2, diffeomorphic=True, composition="A")))
    second = to_np(exp_field(compose(f1, f2, diffeomorphic=True, composition="B")))
    inner = (slice(None),) + interior(2, 10)
    assert np.abs(second - exact)[inner].mean() < 0.5 * np.abs(first - exact)[inner].mean()


@pytest.mark.parametrize("diffeomorphic,composition", [(False, "A"), (True, "A"), (True, "B")])
@pytest.mark.parametrize("shape", SHAPES)
def test_compose_in_place_into_first_field(xp, monkeypatch, shape, diffeomorphic, composition):
    import pydemons.interpolate

    monkeypatch.setattr(pydemons.interpolate, "SLAB_VOXELS", 40)  # many slabs
    f1 = xp.asarray(smooth_field(shape, 3.0, seed=20))
    f2 = xp.asarray(smooth_field(shape, 1.0, seed=21))
    expected = to_np(compose(f1, f2, diffeomorphic=diffeomorphic, composition=composition))
    in_place = f1.copy()
    assert compose(in_place, f2, diffeomorphic=diffeomorphic, composition=composition, out=in_place) is in_place
    assert np.allclose(to_np(in_place), expected, atol=1e-12)
