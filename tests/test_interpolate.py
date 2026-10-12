"""Slab-wise interpolation must equal the one-shot result, for any slab size."""

import numpy as np
import pytest
from scipy.ndimage import map_coordinates

import pydemons.interpolate as interpolate
from conftest import blobs, grid, smooth_field, to_np

SHAPES = [(37, 29), (13, 11, 9), (1, 12), (6, 1, 5)]


def reference(values, field):
    coords = grid(field.shape[1:]) + field
    return np.stack([map_coordinates(v, coords, order=1, mode="nearest") for v in values])


@pytest.mark.parametrize("slab_voxels", [1, 7, 100, 1 << 22])
@pytest.mark.parametrize("shape", SHAPES)
def test_slabs_match_reference(xp, monkeypatch, shape, slab_voxels):
    monkeypatch.setattr(interpolate, "SLAB_VOXELS", slab_voxels)
    values = np.stack([blobs(shape), np.random.default_rng(0).random(shape)])
    field = smooth_field(shape, 5.0, seed=1)  # large: many samples past the border
    out = to_np(interpolate.sample_linear(xp.asarray(values), xp.asarray(field)))
    assert np.allclose(out, reference(values, field), atol=1e-6)


@pytest.mark.parametrize("shape", [(37, 29), (13, 11, 9)])
def test_rows_and_out(xp, monkeypatch, shape):
    monkeypatch.setattr(interpolate, "SLAB_VOXELS", 50)
    values = np.random.default_rng(2).random((2,) + shape)
    field = smooth_field(shape, 3.0, seed=3)
    full = reference(values, field)
    rows = slice(3, shape[0] - 2)
    part = to_np(interpolate.sample_linear(xp.asarray(values), xp.asarray(field), rows=rows))
    assert np.allclose(part, full[:, rows], atol=1e-6)
    out = xp.full((2,) + shape, np.nan)
    assert interpolate.sample_linear(xp.asarray(values), xp.asarray(field), out=out) is out
    assert np.allclose(to_np(out), full, atol=1e-6)


def test_slabs_cover_axis_once():
    for shape, voxels in [((10, 4, 4), 33), ((7, 100), 1), ((5,), 1 << 22)]:
        parts = interpolate.slabs(shape, voxels)
        assert parts[0].start == 0 and parts[-1].stop == shape[0]
        assert all(a.stop == b.start for a, b in zip(parts, parts[1:]))
