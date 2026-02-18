import numpy as np
from numba import njit


@njit(cache=True)
def exozippy_keplereq(m, ecc, thresh=1e-10):
    if ecc < 0.0 or ecc >= 1.0:
        raise ValueError('Eccentricity must be 0 <= ecc < 1')

    mx = np.copy(m)

    zz = np.where(mx > np.pi)[0]
    if len(zz) > 0:
        mx[zz] = np.mod(mx[zz], 2 * np.pi)
        zz = np.where(mx > np.pi)[0]
        if len(zz) > 0:
            mx[zz] -= 2.0 * np.pi

    zz = np.where(mx <= -np.pi)[0]
    if len(zz) > 0:
        mx[zz] = np.mod(mx[zz], 2 * np.pi)
        zz = np.where(mx <= -np.pi)[0]
        if len(zz) > 0:
            mx[zz] += 2.0 * np.pi

    if ecc == 0.0:
        return mx

    aux = 4.0 * ecc + 0.5
    alpha = (1.0 - ecc) / aux
    beta = mx / (2.0 * aux)

    aux = np.sqrt(beta ** 2 + alpha ** 3)
    z = beta + aux
    zz = np.where(z <= 0.0)[0]
    if len(zz) > 0:
        z[zz] -= 2 * aux[zz]
    z = z ** (1.0 / 3.0)

    s0 = z - alpha / z
    s1 = s0 - (0.078 * s0 ** 5) / (1.0 + ecc)
    e0 = mx + ecc * (3.0 * s1 - 4.0 * s1 ** 3)

    se0 = np.sin(e0)
    ce0 = np.cos(e0)

    f = e0 - ecc * se0 - mx
    f1 = 1.0 - ecc * ce0
    f2 = ecc * se0
    f3 = ecc * ce0
    u1 = -f / f1
    u2 = -f / (f1 + 0.5 * f2 * u1)
    u3 = -f / (f1 + 0.5 * f2 * u2 + 0.16666666666666667 * f3 * u2 ** 2)
    u4 = -f / (f1 + 0.5 * f2 * u3 + 0.16666666666666667 * f3 * u3 ** 2 - 0.041666666666666667 * f2 * u3 ** 3)

    eccanom = e0 + u4

    zz = np.where(eccanom >= 2.0 * np.pi)[0]
    if len(zz) > 0:
        eccanom[zz] -= 2.0 * np.pi
    zz = np.where(eccanom < 0.0)[0]
    if len(zz) > 0:
        eccanom[zz] += 2.0 * np.pi

    ndx = np.where(mx < 0.0)[0]
    if len(ndx) != 0:
        mx[ndx] += 2.0 * np.pi

    diff = np.abs(eccanom - ecc * np.sin(eccanom) - mx)
    bad = np.where(diff > (2 * np.pi))[0]
    diff[bad] -= (2 * np.pi)
    bad = np.where(diff < (-2 * np.pi))[0]
    diff[bad] += (2 * np.pi)
    ndx = np.where(diff > thresh)[0]

    while len(ndx) > 0:
        fe = np.mod(eccanom[ndx] - ecc * np.sin(eccanom[ndx]) - mx[ndx], 2 * np.pi)
        fs = np.mod(1.0 - ecc * np.cos(eccanom[ndx]), 2 * np.pi)
        oldval = eccanom[ndx]
        eccanom[ndx] = oldval - fe / fs

        tmp = np.where(np.abs(oldval - eccanom[ndx]) > thresh)[0]
        ndx = ndx[tmp]

    toohigh = np.where(eccanom >= 2 * np.pi)[0]
    if len(toohigh) > 0:
        eccanom[toohigh] = np.mod(eccanom[toohigh], 2 * np.pi)
    toolow = np.where(eccanom < 0)[0]
    if len(toolow) > 0:
        eccanom[toolow] = np.mod(eccanom[toolow], 2 * np.pi) + 2 * np.pi

    return eccanom

