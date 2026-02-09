import numpy as np


def exozippy_blackbody(temp, x, freq=False, wave=False, cgs=False, mks=False):
    """
    Port of EXOFASTv2 exofast_blackbody.pro.

    Computes Planck function I_nu or I_lambda.

    Parameters
    ----------
    temp : float or ndarray
        Temperature in K. Scalar or same size as x.
    x : float or ndarray
        By default wavelength in mks units. If cgs=True, in cgs units.
        If freq=True, x is frequency.
    freq, wave : bool
        Interpret x as frequency (default) or wavelength.
    cgs, mks : bool
        Use CGS or MKS units (default MKS).
    """
    temp = np.asarray(temp, dtype=float)
    x = np.asarray(x, dtype=float)

    ntemp = temp.size
    nx = x.size
    if (ntemp != nx) and (ntemp != 1):
        raise ValueError("temp must be scalar or same size as x")

    do_cgs = bool(cgs)
    if do_cgs and mks:
        raise ValueError("do not set both cgs and mks")

    do_wave = bool(wave)
    if do_wave and freq:
        raise ValueError("do not set both freq and wave")

    # constants
    if do_cgs:
        h = 6.626e-27
        k = 1.38065e-16
        c = 2.99792e10
    else:
        h = 6.626e-34
        k = 1.38065e-23
        c = 2.99792e8

    nu = c / x if do_wave else x

    result = 2.0 * h * nu**3 * c**(-2.0) / (np.exp(h * nu / (k * temp)) - 1.0)

    if do_wave:
        result = result * nu**2 / c

    return result

