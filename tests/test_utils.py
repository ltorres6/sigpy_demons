import numpy as np

from conftest import blobs, to_np
from pydemons.utils import normalize


def test_maps_range_to_bounds(xp):
    img = 5.0 * blobs((20, 20)) - 1.0
    out = to_np(normalize(xp.asarray(img), -2.0, 3.0))
    assert np.isclose(out.min(), -2.0) and np.isclose(out.max(), 3.0)


def test_is_affine(xp):
    img = blobs((20, 20))
    out = to_np(normalize(xp.asarray(img)))
    assert np.allclose(out, (img - img.min()) / (img.max() - img.min()))
