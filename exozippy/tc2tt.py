import numpy as np

from .utils import exozippy_getphase


def tc2tt(time, e, inc, omega, period, tol=1e-15,
          tt2tc=False, ts2te=False, te2ts=False):
    """
    Port of EXOFASTv2 tc2tt.pro.

    Parameters
    ----------
    time : array-like
        Input time(s): Tc, Tt, Ts, or Te depending on flags.
    e, inc, omega, period : array-like
        Eccentricity, inclination (radians), argument of periastron (radians),
        and period (days).
    tol : float
        Convergence tolerance for theta iteration.
    tt2tc, ts2te, te2ts : bool
        Select conversion direction (default Tc -> Tt).

    Returns
    -------
    ndarray
        Converted times, same shape as input.
    """
    time = np.asarray(time, dtype=float)
    e = np.asarray(e, dtype=float)
    inc = np.asarray(inc, dtype=float)
    omega = np.asarray(omega, dtype=float)
    period = np.asarray(period, dtype=float)

    thetanew = 0.0
    maxiter = 100
    for _ in range(maxiter):
        thetaold = thetanew
        if ts2te or te2ts:
            thetanew = -np.arctan(
                -e * np.cos(omega) * np.cos(inc)**2 /
                (np.cos(thetaold) * np.sin(inc)**2 - e * np.sin(omega))
            )
        else:
            thetanew = np.arctan(
                -e * np.cos(omega) * np.cos(inc)**2 /
                (np.cos(thetaold) * np.sin(inc)**2 + e * np.sin(omega))
            )
        if np.max(np.abs(thetanew - thetaold)) < tol:
            break
    else:
        raise RuntimeError("tc2tt: maximum iterations exceeded")

    if ts2te or te2ts:
        phase = exozippy_getphase(e, omega, secondary=True)
        phase0 = exozippy_getphase(e, omega, trueanom=1.5 * np.pi - omega + thetanew)
        if te2ts:
            tfinal = time - period * (phase0 - phase)
        else:
            tfinal = time + period * (phase0 - phase)
    else:
        phase = exozippy_getphase(e, omega, primary=True)
        phase0 = exozippy_getphase(e, omega, trueanom=0.5 * np.pi - omega + thetanew)
        if tt2tc:
            tfinal = time - period * (phase0 - phase)
        else:
            tfinal = time + period * (phase0 - phase)

    toohigh = tfinal > time + period / 2.0
    if np.any(toohigh):
        tfinal[toohigh] -= period[toohigh] if period.size > 1 else period
    toolow = tfinal < time - period / 2.0
    if np.any(toolow):
        tfinal[toolow] += period[toolow] if period.size > 1 else period

    return tfinal
