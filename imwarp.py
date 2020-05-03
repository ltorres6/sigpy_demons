import sigpy as sp
import scipy.ndimage as npTools

# import cupyx.scipy.ndimage as cpTools


def imwarp(arrIn, T, device=-1):
    # This function warps I_in to coordinates defined by coord + T
    # Interpolate using sigpy. param = 1 is linear.
    xp = sp.Device(device).xp
    if len(arrIn.shape) == 2:
        nx, ny = arrIn.shape
        coord = xp.mgrid[0:nx, 0:ny].astype("float64")
    elif len(arrIn.shape) == 3:
        nx, ny, nz = arrIn.shape
        coord = xp.mgrid[0:nx, 0:ny, 0:nz].astype("float64")
        # might not have to shift axis
    if device == -1:
        # arrOut = npTools.map_coordinates(
        #     arrIn, xp.moveaxis(coord, 0, -1) + xp.moveaxis(T, 0, -1), order=1, prefilter=False
        # )
        arrOut = npTools.map_coordinates(arrIn, coord + T, order=1, prefilter=False)
    else:
        # arrOut = cpTools.map_coordinates(arrIn, coord + T, order=1, prefilter=False)
        # arrOut = cpTools.map_coordinates(
        #     arrIn, xp.moveaxis(coord, 0, -1) + xp.moveaxis(T, 0, -1), order=1, prefilter=False
        # )
        arrOut = sp.interpolate(
            arrIn,
            xp.moveaxis(coord, 0, -1) + xp.moveaxis(T, 0, -1),
            kernel="spline",
            width=2,
            param=1,
        )

    return arrOut
