import sigpy as sp
import finiteDifferences as fd


def energy(arrIn, fieldIn, alpha):
    # From Herve Lombaert (matlab exchange)
    xp = sp.get_array_module(arrIn)
    Esim = xp.sum(xp.square(arrIn)) / (xp.size(arrIn))
    Ereg = xp.sum(xp.square(jacobianDet(fieldIn))) * (alpha**2) / (xp.size(arrIn))
    return Esim + Ereg


def energy2(arrIn, fieldIn, alpha):
    xp = sp.get_array_module(arrIn)
    # Need to fix for 2d
    E = xp.sum(
        arrIn**2 + arrIn**2 * (fieldIn[0] ** 2 + fieldIn[1] ** 2 + fieldIn[2] ** 2)
    ) * (alpha**2)
    return E


def RMSE(arrIn):
    # E_corr s_opt
    # I1 should be Idiff image
    xp = sp.get_array_module(arrIn)
    E = xp.sqrt(xp.mean(arrIn**2) ** 2)
    return E


def HarmonicEnergy(fieldIn, alpha):
    # Not finished
    xp = sp.get_array_module(fieldIn)
    E = jacobianDet(fieldIn)
    E = xp.mean(xp.linalg.norm(E) ** 2) * (alpha**2)
    return -1 * E


def QU(UpdateFieldIn, TotalFieldIn):
    # Quantity of update
    xp = sp.get_array_module(UpdateFieldIn)
    E = xp.sum(xp.abs(UpdateFieldIn)) / xp.sum(xp.abs(TotalFieldIn))
    return E


def jacobianDet(fieldIn):
    D = fd.CentralDifference(fieldIn.shape[1:])
    if len(fieldIn.shape[1:]) == 2:
        # common to add identity
        Dx = D * fieldIn[0]
        Dy = D * fieldIn[1]

        # common to add identity
        Dx[0] = Dx[0] + 1
        Dy[1] = Dy[1] + 1

        J = Dx[0] * Dy[1] - Dy[0] * Dx[1]

    elif len(fieldIn.shape[1:]) == 3:
        # common to add identity
        Dx = D * fieldIn[0]
        Dy = D * fieldIn[1]
        Dz = D * fieldIn[2]

        # common to add identity
        Dx[0] = Dx[0] + 1
        Dy[1] = Dy[1] + 1
        Dz[2] = Dz[2] + 1

        J = (
            Dx[0] * Dy[1] * Dz[2]
            + Dy[0] * Dz[1] * Dx[2]
            + Dz[0] * Dx[1] * Dy[2]
            - Dz[0] * Dy[1] * Dx[2]
            - Dy[0] * Dx[1] * Dz[2]
            - Dx[0] * Dz[1] * Dy[2]
        )
    return J
