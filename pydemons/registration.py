"""Multi-resolution demons registration."""

from __future__ import annotations

import logging

from tqdm.auto import tqdm

from pydemons.backend import get_array_module
from pydemons.convolution import FFTConvolver
from pydemons.differences import gradient
from pydemons.fields import compose
from pydemons.filters import gaussian_kernel
from pydemons.forces import VARIANTS, demons_force
from pydemons.resample import zoom
from pydemons.utils import resize
from pydemons.warp import warp

logger = logging.getLogger(__name__)

PADDING = 40


class Demons:
    """Demons registration of a moving image onto a fixed image.

    Runs on the device of the input arrays: pass NumPy arrays for CPU or CuPy
    arrays (see :func:`pydemons.backend.to_gpu`) for GPU.

    Parameters
    ----------
    fixed, moving : ndarray
        2D or 3D images of the same shape.
    n_levels : int
        Number of pyramid levels. Each level halves the resolution.
    diffusion_sigma : float
        Gaussian sigma (voxels) smoothing the accumulated field each iteration.
        0 disables it.
    fluid_sigma : float
        Gaussian sigma (voxels) smoothing each update. 0 disables it.
    max_iter : int
        Iterations at the coarsest level; each finer level runs half as many.
    alpha : float
        Step-size control; updates are bounded by ``1 / (2 * alpha)`` voxels.
    tol : float
        Stop a level once the relative change in energy drops below ``tol``.
        0 runs every iteration.
    variant : {"passive", "active", "inverse_consistent"}
        Which demons force to use.
    diffeomorphic : bool
        Treat the field as a stationary velocity field (log-domain demons).
    composition : {"A", "B"}
        Log-domain composition: ``"A"`` is ``v1 + v2``, ``"B"`` adds the Lie
        bracket term. Ignored unless ``diffeomorphic``.
    progress : bool
        Show a progress bar per level.

    Example
    -------
    >>> warped, field = Demons(fixed, moving, n_levels=3).run()  # doctest: +SKIP
    """

    def __init__(
        self,
        fixed,
        moving,
        n_levels=4,
        diffusion_sigma=1.0,
        fluid_sigma=1.0,
        max_iter=100,
        alpha=2.5,
        tol=1e-6,
        variant="passive",
        diffeomorphic=False,
        composition="A",
        progress=True,
    ):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
        if composition not in ("A", "B"):
            raise ValueError(f"composition must be 'A' or 'B', got {composition!r}")
        self.xp = get_array_module(fixed, moving)
        fixed = self.xp.asarray(fixed, dtype=self.xp.float64)
        moving = self.xp.asarray(moving, dtype=self.xp.float64)

        # Scale both images by the moving maximum for numerical stability.
        scale = moving.max()
        fixed = fixed / scale
        moving = moving / scale

        self.n_levels = n_levels
        self.scales = [2 ** (n_levels - i - 1) for i in range(n_levels)]
        self.iterations = [int(max_iter // 2**i) for i in range(n_levels)]
        self.original_shape = fixed.shape
        self.padded_shape = self._padded_shape(self.original_shape, n_levels, PADDING)
        self.fixed = resize(fixed, self.padded_shape)
        self.moving = resize(moving, self.padded_shape)
        self.ndim = fixed.ndim
        self.field = self.xp.zeros((self.ndim,) + self.padded_shape)
        self.level_shapes = [tuple(s // scale for s in self.padded_shape) for scale in self.scales]
        self.alpha = alpha
        self.diffusion_sigma = diffusion_sigma
        self.fluid_sigma = fluid_sigma
        self.tol = tol
        self.variant = variant
        self.diffeomorphic = diffeomorphic
        self.composition = composition
        self.progress = progress

    @staticmethod
    def _padded_shape(shape, n_levels, padding):
        """Pad each axis by at least ``padding`` and round up to a multiple of ``2**n_levels``."""
        m = 2**n_levels
        return tuple((s + padding) + m - (s + padding) % m for s in shape)

    def run(self):
        """Register and return ``(warped_moving, field)`` cropped to the input shape."""
        xp = self.xp
        logger.info("padded shape %s, scales %s, iterations %s", self.padded_shape, self.scales, self.iterations)

        for level in range(self.n_levels):
            factor = 2.0 ** -(self.n_levels - level - 1)
            shape = self.level_shapes[level]
            sigma_diffusion = self.diffusion_sigma
            sigma_fluid = self.diffusion_sigma

            fixed = zoom(self.fixed, factor)
            moving = zoom(self.moving, factor)
            fixed_grad = gradient(fixed)
            fixed_grad_sq = xp.sum(fixed_grad**2, axis=0)
            field = xp.stack([zoom(c, factor) for c in self.field])

            if sigma_diffusion > 0.0:
                smooth_diffusion = FFTConvolver(gaussian_kernel(sigma_diffusion, self.ndim, xp=xp), shape)
            if sigma_fluid > 0.0:
                smooth_fluid = FFTConvolver(gaussian_kernel(sigma_fluid, self.ndim, xp=xp), shape)

            losses = []
            window = 4 * (self.n_levels - level)
            stop = 0.0
            desc = f"Level {level + 1}/{self.n_levels} {shape}"
            with tqdm(desc=desc, total=self.iterations[level], disable=not self.progress, leave=True) as pbar:
                for it in range(self.iterations[level]):
                    update, loss = demons_force(
                        fixed,
                        moving,
                        field,
                        self.alpha,
                        fixed_grad=fixed_grad,
                        fixed_grad_sq=fixed_grad_sq,
                        variant=self.variant,
                        scaling_factor=self.n_levels - level,
                        diffeomorphic=self.diffeomorphic,
                    )
                    if sigma_fluid > 0.0:
                        update = xp.stack([smooth_fluid(c) for c in update])
                    field = compose(field, update, diffeomorphic=self.diffeomorphic, composition=self.composition)

                    losses.append(float(loss))
                    if it > window:
                        previous = sum(losses[it - window - 1 : it - 1]) / window
                        stop = abs((losses[it] - previous) / previous)
                        if stop < self.tol:
                            break

                    if sigma_diffusion > 0.0:
                        field = xp.stack([smooth_diffusion(c) for c in field])
                    pbar.set_postfix(loss=stop)
                    pbar.update()

            self.field = xp.stack([zoom(c, 1 / factor) for c in field])

        self.field = resize(self.field, (self.ndim,) + self.original_shape)
        self.fixed = resize(self.fixed, self.original_shape)
        self.moving = resize(self.moving, self.original_shape)
        warped = warp(self.moving, self.field)
        return warped, self.field
