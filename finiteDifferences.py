import sigpy as sp


def CentralDifference(ishape, axes=None):
    """Linear operator that computes (central) finite difference gradient.

    Args:
        ishape (tuple of ints): Input shape.
        axes (tuple or list): Axes to circularly shift. All axes are used if
            None.

    """
    # Id = sp.linop.Identity(ishape)
    ndim = len(ishape)
    axes = sp.util._normalize_axes(axes, ndim)
    linops = []
    for i in axes:
        D = sp.linop.Circshift(ishape, [-1], axes=[i]) - sp.linop.Circshift(
            ishape, [1], axes=[i]
        )
        R = sp.linop.Reshape([1] + list(ishape), ishape)
        linops.append(R * D)

    G = sp.linop.Vstack(linops, axis=0)

    return G


def Laplacian(ishape, axes=None):
    """Linear operator that computes Laplacian.

    Args:
        ishape (tuple of ints): Input shape.
        axes (tuple or list): Axes to circularly shift. All axes are used if
            None.

    """
    Id = sp.linop.Identity(ishape)
    ndim = len(ishape)
    axes = sp.util._normalize_axes(axes, ndim)
    linops = []
    for i in axes:
        D = (
            sp.linop.Circshift(ishape, [-1], axes=[i])
            + sp.linop.Circshift(ishape, [1], axes=[i])
            - 2 * Id
        )
        R = sp.linop.Reshape([1] + list(ishape), ishape)
        linops.append(R * D)

    G = sp.linop.Vstack(linops, axis=0)

    return G
