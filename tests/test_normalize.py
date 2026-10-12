import numpy as np

from conftest import blobs, to_np
from normalize import normalize


def test_maps_range_to_bounds(xp, device):
    img = 5.0 * blobs((20, 20)) - 1.0
    out = to_np(normalize(xp.asarray(img), -2.0, 3.0, device=device))
    assert np.isclose(out.min(), -2.0) and np.isclose(out.max(), 3.0)


def test_is_affine(xp, device):
    img = blobs((20, 20))
    out = to_np(normalize(xp.asarray(img), device=device))
    assert np.allclose(out, (img - img.min()) / (img.max() - img.min()))
