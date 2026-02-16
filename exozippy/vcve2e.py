"""
Convert Vc/Ve parameterization to eccentricity.

Vc/Ve = sqrt(1 - e^2) / (1 + e*sin(omega))

This is the transit duration scaling factor: the ratio of the circular
velocity to the eccentric velocity at the time of transit.  When only
transit data are available (no RV), the light curve directly constrains
vcve rather than e itself, making it the natural stepping parameter.

Ported from EXOFASTv2's vcve2e.pro.
"""

import numpy as np


def vcve2e(vcve, omega=None, lsinw=None, lcosw=None, sign=None):
    """
    Convert vcve (+ omega or lsinw/lcosw) to eccentricity by solving
    the quadratic:

        a*e^2 + b*e + c = 0

    where:
        a = vcve^2 * sin^2(omega) + 1
        b = 2 * vcve^2 * sin(omega)
        c = vcve^2 - 1

    Parameters
    ----------
    vcve : float
        Vc/Ve = sqrt(1 - e^2) / (1 + e*sin(omega)).  Range: (0, 1].
    omega : float, optional
        Argument of periastron [radians].
    lsinw : float, optional
        L * sin(omega), used to derive omega = atan2(lsinw, lcosw).
    lcosw : float, optional
        L * cos(omega).
    sign : float, optional
        Root selector.  If floor(sign) is odd, use the negative root;
        if even, use the positive root.  When not provided, the code
        picks the physical solution (0 <= e < 1) with lower e preferred.

    Returns
    -------
    e : float
        Eccentricity.
    """
    if omega is None:
        if lsinw is None or lcosw is None:
            raise ValueError('must specify either omega or lsinw and lcosw')
        omega = np.arctan2(lsinw, lcosw)

    sinw = np.sin(omega)

    a = vcve**2 * sinw**2 + 1.0
    b = 2.0 * vcve**2 * sinw
    c = vcve**2 - 1.0

    disc = b**2 - 4.0 * a * c
    if disc < 0:
        return 0.0  # no real solution → circular

    sqrt_disc = np.sqrt(disc)
    epos = (-b + sqrt_disc) / (2.0 * a)
    eneg = (-b - sqrt_disc) / (2.0 * a)

    if sign is not None:
        # Use sign to select root (EXOFASTv2 convention: floor(sign) odd → neg)
        if int(np.floor(sign)) % 2 == 1:
            return eneg
        else:
            return epos
    elif lsinw is not None and lcosw is not None:
        # Use L = sqrt(lsinw^2 + lcosw^2) to select root
        # L^2 >= 0.5 → negative root (reduces gaps in parameter space)
        L2 = lsinw**2 + lcosw**2
        if L2 >= 0.5:
            return eneg
        else:
            return epos
    else:
        # No sign specified: pick the physical solution (lower e preferred)
        pos_ok = np.isfinite(epos) and 0.0 <= epos < 1.0
        neg_ok = np.isfinite(eneg) and 0.0 <= eneg < 1.0
        if neg_ok and (not pos_ok or eneg < epos):
            return eneg
        if pos_ok:
            return epos
        return 0.0
