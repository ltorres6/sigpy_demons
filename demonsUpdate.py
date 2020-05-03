import sigpy as sp
from imwarp import imwarp
import finiteDifferences as fd
import energy
import diffeomorphism


def passiveUpdate(Is, Im, T, alpha, dS, dSMag, scalingFactor=1.0, device=-1):
    # Warp image
    IntensityDifferenceThreshold = 1e-2 ** scalingFactor
    DenominatorThreshold = 1e-9
    xp = sp.Device(device).xp
    nDims = len(Is.shape)
    ImWarped = imwarp(Im, T, device=device)
    Idiff = Is - ImWarped
    Denominator = dSMag + Idiff ** 2 * alpha ** 2
    cfactor = Idiff / Denominator
    V = xp.zeros(dS.shape)
    V = cfactor * dS
    # Generate mask for null values (convert to zeros)
    mask = xp.zeros(V.shape, dtype=bool)
    for ii in range(nDims):
        mask[ii] = (
            (~xp.isfinite(V[ii]))
            | (xp.abs(Idiff) < IntensityDifferenceThreshold)
            | (Denominator < DenominatorThreshold)
        )
    V[mask] = 0.0
    loss = energy.energy(Idiff, T, alpha)
    return V, loss


def activeUpdate(Is, Im, T, alpha, D, scalingFactor=1.0, device=-1):
    # Warp image
    IntensityDifferenceThreshold = 1e-2 ** scalingFactor
    # print(IntensityDifferenceThreshold)
    DenominatorThreshold = 1e-9
    xp = sp.Device(device).xp
    nDims = len(Is.shape)
    ImWarped = imwarp(Im, T, device=device)
    dM = D * Im
    dMMag = xp.sum(dM ** 2, axis=0)
    Idiff = Is - ImWarped
    Denominator = dMMag + Idiff ** 2 * alpha ** 2
    cfactor = Idiff / Denominator
    V = xp.zeros(dM.shape)
    V = cfactor * dM
    # Generate mask for null values (convert to zeros)
    mask = xp.zeros(V.shape, dtype=bool)
    for ii in range(nDims):
        mask[ii] = (
            (~xp.isfinite(V[ii]))
            | (xp.abs(Idiff) < IntensityDifferenceThreshold)
            | (Denominator < DenominatorThreshold)
        )
    V[mask] = 0.0
    loss = energy.energy(Idiff, T, alpha)
    return V, loss


def inverseConsistentUpdate(Is, Im, T, alpha, D, scalingFactor=1.0, device=-1):
    """
    Inputs
    Is - Static Image
    Im - Moving Image
    T - Warp that moves Im to Is
    D - Central Differences Operator
    """
    IntensityDifferenceThreshold = 1e-2 ** scalingFactor
    DenominatorThreshold = 1e-9
    xp = sp.Device(device).xp
    nDims = len(Is.shape)
    ImWarped = imwarp(Im, T, device=device)
    Idiff = Is - ImWarped
    J = D * Is + D * Im
    JMag = xp.sum(J ** 2, axis=0)
    Denominator = JMag + Idiff ** 2 * alpha ** 2
    cfactor = Idiff / Denominator
    V = 2 * cfactor * J
    # Generate mask for null values (convert to zeros)
    mask = xp.zeros(V.shape, dtype=bool)
    for ii in range(nDims):
        mask[ii] = (
            ~xp.isfinite(V[ii])
            | (xp.abs(Idiff) < IntensityDifferenceThreshold)
            | (Denominator < DenominatorThreshold)
        )
    V[mask] = 0.0
    loss = energy.energy(Idiff, T, alpha)
    return V, loss


def demonsUpdate(
    Is,
    Im,
    T,
    alpha,
    Tinv=None,
    dS=None,
    dSMag=None,
    variant="passive",
    scalingFactor=1.0,
    diffeomorphic=False,
    device=-1,
):
    """
    Computes the update field using the demons approach.
    Always Required Inputs:
    Is - Static Image
    Im - Moving Image
    T - Warp that moves Im to Is

    Variation Specific Inputs:
    - "passive"
    dS - Static Image Directional Gradients (precomputed for efficiency)
    dSMag - Magnitude of dS (elementwise sum of squares for each dimension)
    - "inverseConsistent"
    D = Finite Differences Operator. Used to compute gradients each iteration.

    """
    if variant == "passive":
        V, loss = passiveUpdate(Is, Im, T, alpha, dS, dSMag, scalingFactor, device=device)
        return V, loss
    elif variant == "active":
        D = fd.CentralDifference(Is.shape)
        V, loss = activeUpdate(Is, Im, T, alpha, D, scalingFactor, device=device)
        return V, loss
    elif variant == "inverseConsistent":
        D = fd.CentralDifference(Is.shape)
        if diffeomorphic is True:
            Tinv = diffeomorphism.expfield(-1 * T, device=device)
            T = diffeomorphism.expfield(T, device=device)
        V, loss = inverseConsistentUpdate(Is, Im, T, alpha, D, scalingFactor, device=device)
        if Tinv is None:
            Tinv = -1 * T
        # Swap Inputs and use inverse
        V2, loss2 = inverseConsistentUpdate(Im, Is, Tinv, alpha, D, scalingFactor, device=device)
        V = (V - V2) * 0.5  # This might have to be done in the log domain to be correct...
        loss = (loss + loss2) * 0.5
        return V, loss
    else:
        print("variant not implemented yet.")
        return
