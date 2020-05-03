import sigpy as sp


def normalize(I_in, minv=0.0, maxv=1.0, device=-1):
    xp = sp.Device(device).xp
    out = (maxv - minv) * (I_in - xp.min(I_in[:])) / (xp.max(I_in[:]) - xp.min(I_in[:])) + minv
    return out
