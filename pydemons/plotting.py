"""Optional result plots (requires matplotlib: ``pip install pydemons[plot]``)."""

from __future__ import annotations

from pydemons.backend import to_numpy
from pydemons.fields import jacobian_determinant


def _middle_slice(arr):
    """Central slice along the last axis of a 3D image; 2D images pass through."""
    return arr[..., arr.shape[-1] // 2] if arr.ndim == 3 else arr


def plot_registration(fixed, moving, warped, field):
    """Plot images, residuals, field components and the Jacobian determinant.

    3D inputs are shown as their central slice along the last axis. Returns the
    matplotlib figure.
    """
    import matplotlib.pyplot as plt

    fixed, moving, warped, field = (to_numpy(a) for a in (fixed, moving, warped, field))
    det = to_numpy(jacobian_determinant(field))
    panels = [
        ("Fixed", fixed, "gray"),
        ("Moving", moving, "gray"),
        ("Warped moving", warped, "gray"),
        ("Fixed - moving", fixed - moving, "RdBu_r"),
        ("Fixed - warped", fixed - warped, "RdBu_r"),
        ("Jacobian determinant", det, "viridis"),
    ] + [(f"Displacement axis {i}", c, "RdBu_r") for i, c in enumerate(field)]

    ncols = 3
    nrows = -(-len(panels) // ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3.6 * nrows), squeeze=False)
    for ax, (title, img, cmap) in zip(axes.flat, panels):
        img = _middle_slice(img)
        if cmap == "RdBu_r":
            lim = float(abs(img).max()) or 1.0
            im = ax.imshow(img, cmap=cmap, vmin=-lim, vmax=lim)
        else:
            im = ax.imshow(img, cmap=cmap)
        if title == "Jacobian determinant" and (det <= 0).any():
            ax.contour(_middle_slice(det) <= 0, levels=[0.5], colors="red", linewidths=0.8)
            title += f" ({int((det <= 0).sum())} folded voxels)"
        ax.set_title(title, fontsize=9)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046)
    for ax in axes.flat[len(panels) :]:
        ax.axis("off")
    fig.tight_layout()
    return fig
