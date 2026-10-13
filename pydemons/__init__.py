"""Demons image registration on CPU (NumPy) or GPU (CuPy)."""

from pydemons.backend import get_array_module, is_gpu_available, to_gpu, to_numpy
from pydemons.fields import compose, exp_field, jacobian_determinant
from pydemons.registration import Demons
from pydemons.warp import warp

__version__ = "0.2.0"

__all__ = [
    "Demons",
    "compose",
    "exp_field",
    "get_array_module",
    "is_gpu_available",
    "jacobian_determinant",
    "to_gpu",
    "to_numpy",
    "warp",
]
