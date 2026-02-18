import numpy as np
from numba import njit

from .kepler import exozippy_keplereq

DEFAULT_LIGHT_SPEED_AU_PER_DAY = 173.144483


@njit(cache=True)
def _target2bjd_scalar(
    bjd_target,
    inclination,
    a,
    tp,
    period,
    e,
    omega,
    q=np.inf,
    primary=False,
    c=DEFAULT_LIGHT_SPEED_AU_PER_DAY,
):
    bjd_target = np.asarray(bjd_target, dtype=np.float64)

    if (not np.isfinite(q)) and primary:
        return bjd_target

    mean_anom = 2.0 * np.pi * ((bjd_target - tp) / period)
    mean_anom = np.mod(mean_anom, 2.0 * np.pi)

    ecc_anom = exozippy_keplereq(mean_anom, e)
    true_anom = 2.0 * np.arctan(np.sqrt((1.0 + e) / (1.0 - e)) * np.tan(ecc_anom / 2.0))

    if np.isfinite(q):
        factor = (1.0 / (1.0 + q)) if primary else (q / (1.0 + q))
    else:
        factor = 1.0

    r = a * (1.0 - e * e) / (1.0 + e * np.cos(true_anom)) * factor
    om = omega if primary else (omega + np.pi)
    z = r * np.sin(true_anom + om) * np.sin(inclination)
    return bjd_target - z / c


def target2bjd(
    bjd_target,
    inclination,
    a,
    tp,
    period,
    e,
    omega,
    q=np.inf,
    primary=False,
    c=DEFAULT_LIGHT_SPEED_AU_PER_DAY,
):
    bjd_target = np.asarray(bjd_target, dtype=np.float64)
    c_use = DEFAULT_LIGHT_SPEED_AU_PER_DAY if c is None else c

    # Fast path used by bjd2target and most transit/RV use cases.
    if np.ndim(e) == 0:
        return _target2bjd_scalar(
            bjd_target,
            inclination,
            a,
            tp,
            period,
            float(e),
            omega,
            q=q,
            primary=primary,
            c=c_use,
        )

    # Solve Kepler's Equation
    mean_anom = 2.0 * np.pi * ((bjd_target - tp) / period)
    mean_anom = np.mod(mean_anom, 2.0 * np.pi)
    e_arr = np.asarray(e, dtype=np.float64)
    if e_arr.shape != bjd_target.shape:
        raise ValueError("e must be scalar or same shape as bjd_target")
    ecc_anom = np.empty_like(mean_anom)
    for i in range(mean_anom.size):
        ecc_anom[i] = exozippy_keplereq(np.array([mean_anom[i]]), e_arr[i])[0]

    # True anomaly
    true_anom = 2.0 * np.arctan(np.sqrt((1.0 + e_arr) / (1.0 - e_arr)) * np.tan(ecc_anom / 2.0))

    # Displacement from barycenter
    if np.isfinite(q):
        if primary:
            factor = 1.0 / (1.0 + q)  # a1 = a * factor
        else:
            factor = q / (1.0 + q)   # a2 = a * factor
    else:
        factor = 1.0

    # Distance from barycenter to target
    r = a * (1.0 - e_arr * e_arr) / (1.0 + e_arr * np.cos(true_anom)) * factor

    # Rotate orbit by omega
    if not primary:
        om = omega + np.pi
    else:
        om = omega

    # Line-of-sight component
    z = r * np.sin(true_anom + om) * np.sin(inclination)

    return bjd_target - z / c_use

@njit(cache=True)
def bjd2target(
    bjd_tdb,
    inclination,
    a,
    tp,
    period,
    e,
    omega,
    q=None,
    tol=1e-8,
    primary=False,
    pars=None,
    c=None
):
    bjd_target = np.asarray(bjd_tdb, dtype=np.float64).copy()
    c_use = DEFAULT_LIGHT_SPEED_AU_PER_DAY if c is None else c
    niter = 0

    while True:
        # Iterative process to find BJD_TARGET corresponding to BJD_TDB
        # Typically completes in ~3 iterations
        target_new = _target2bjd_scalar(
            bjd_target,
            inclination,
            a,
            tp,
            period,
            e,
            omega,
            q=q,
            primary=primary,
            c=c_use,
        )

        diff = bjd_tdb - target_new
        bjd_target += diff
        niter += 1

        if niter > 100:
            raise RuntimeError(
                "Not converging. This is a rare bug usually associated with poorly constrained parameters. "
                "Try again or consider imposing priors on poorly constrained parameters. Especially if you "
                "have parallel tempering enabled, you should have loose, uniform priors on Period and Tc."
            )

        if np.max(np.abs(diff)) < tol:
            break

    return bjd_target

def exozippy_getb2(bjd, inc, a, tperiastron, period, e=None, omega=None,
                   lonascnode=None, q=None):    
    bjd = np.asarray(bjd, dtype=np.float64)
    inc = np.atleast_1d(inc).astype(np.float64)
    a = np.atleast_1d(a).astype(np.float64)
    tperiastron = np.atleast_1d(tperiastron).astype(np.float64)
    period = np.atleast_1d(period).astype(np.float64)
    
    # shape = bjd.shape
    if bjd.ndim == 0:
        # ntimes, ninterp = 1, 1
        bjd = bjd[None]
    elif bjd.ndim == 1:
        # ntimes, ninterp = len(bjd), 1
        bjd = bjd[:, None]
    elif bjd.ndim == 2:
        # ntimes, ninterp = bjd.shape
        pass
    else:
        raise ValueError("Incompatible dimensions for BJD")
    nplanets = len(inc)
    e = np.atleast_1d(e if e is not None else np.zeros(nplanets))
    omega = np.atleast_1d(omega if omega is not None else np.ones(nplanets) * (np.pi / 2))
    q = np.atleast_1d(q if q is not None else np.full(nplanets, np.inf))
    
    b, z0 = exozippy_getb2_(bjd, inc, a, tperiastron, period, e, omega, lonascnode, q)
    return b.squeeze(), z0.squeeze()

@njit
def exozippy_getb2_(
    bjd, inc, a, tperiastron, period, e=None, omega=None,
    lonascnode=None, q=None):
    nplanets = len(inc)
    ntimes, ninterp = bjd.shape

    # Allocate arrays
    x1 = np.zeros((ntimes, ninterp))
    y1 = np.zeros((ntimes, ninterp))
    z1 = np.zeros((ntimes, ninterp))

    x2 = np.zeros((nplanets, ntimes, ninterp))
    y2 = np.zeros((nplanets, ntimes, ninterp))
    z2 = np.zeros((nplanets, ntimes, ninterp))

    x0 = np.zeros((nplanets, ntimes, ninterp))
    y0 = np.zeros((nplanets, ntimes, ninterp))
    z0 = np.zeros((nplanets, ntimes, ninterp))

    x1tmp = np.zeros((nplanets, ntimes, ninterp))
    y1tmp = np.zeros((nplanets, ntimes, ninterp))
    z1tmp = np.zeros((nplanets, ntimes, ninterp))

    isinf = ~np.isfinite(q)
    isfinite = np.isfinite(q)
    na = a.shape
    a1 = np.zeros(na)
    a2 = np.zeros(na)

    a2[isinf] = a[isinf]
    a1[isinf] = 0.0
    a2[isfinite] = a[isfinite] * q[isfinite] / (1.0 + q[isfinite])
    a1[isfinite] = a2[isfinite] / q[isfinite]


    sqrt_fac = np.zeros(nplanets)
    for i in range(nplanets):
        if e[i] != 0.0:
            sqrt_fac[i] = np.sqrt((1 + e[i]) / (1 - e[i]))


    for i in range(nplanets):
        # Mean anomaly
        meananom = 2.0 * np.pi * ((bjd - tperiastron[i]) / period[i])
        meananom = np.mod(meananom, 2.0 * np.pi)

        if e[i] != 0.0:
            eccanom = exozippy_keplereq(meananom, e[i])
            # trueanom = 2.0 * np.arctan(np.sqrt((1 + e[i]) / (1 - e[i])) * np.tan(eccanom / 2.0))
            trueanom = 2.0 * np.arctan(sqrt_fac[i] * np.tan(eccanom / 2.0))

        else:
            trueanom = meananom
            
            
        theta = trueanom + omega[i]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        # Distance and coordinates (planet in barycentric frame)
        r2 = -a2[i] * (1 - e[i] ** 2) / (1 + e[i] * np.cos(trueanom))

        # x2[i] = (r2 * np.cos(trueanom + omega[i])).reshape(ntimes, ninterp)
        # tmp = r2 * np.sin(trueanom + omega[i])
        # y2[i] = (tmp * np.cos(inc[i])).reshape(ntimes, ninterp)
        # z2[i] = (tmp * np.sin(inc[i])).reshape(ntimes, ninterp)
        x2[i, :, :] = r2 * cos_theta
        tmp = r2 * sin_theta
        y2[i, :, :] = tmp * np.cos(inc[i])
        z2[i, :, :] = tmp * np.sin(inc[i])

        # Rotate by longitude of ascending node if provided
        if lonascnode is not None and len(lonascnode) == nplanets:
            lon = lonascnode[i]
            x_old = x2[i].copy()
            y_old = y2[i].copy()
            x2[i] = x_old * np.cos(lon) - y_old * np.sin(lon)
            y2[i] = x_old * np.sin(lon) + y_old * np.cos(lon)

        # Star position in barycentric frame
        r1 = a1[i] * (1 - e[i] ** 2) / (1 + e[i] * np.cos(trueanom))
        # x1tmp = (r1 * np.cos(trueanom + omega[i])).reshape(ntimes, ninterp)
        x1tmp[i, :, :] = r1 * cos_theta
        tmp = r1 * sin_theta
        y1tmp[i, :, :] = (tmp * np.cos(inc[i])).reshape(ntimes, ninterp)
        z1tmp[i, :, :] = (tmp * np.sin(inc[i])).reshape(ntimes, ninterp)
        # tmp = r1 * np.sin(trueanom + omega[i])
        # y1tmp = (tmp * np.cos(inc[i])).reshape(ntimes, ninterp)
        # z1tmp = (tmp * np.sin(inc[i])).reshape(ntimes, ninterp)

        if lonascnode is not None and len(lonascnode) == nplanets:
            lon = lonascnode[i]
            x1 += x1tmp * np.cos(lon) - y1tmp * np.sin(lon)
            y1 += x1tmp * np.sin(lon) + y1tmp * np.cos(lon)
        else:
            x1 += x1tmp
            y1 += y1tmp
        z1 += z1tmp

    # Convert to stellar frame (planet position relative to star)
    for i in range(nplanets):
        x0[i] = x2[i] - x1
        y0[i] = y2[i] - y1
        z0[i] = z2[i] - z1

    # Impact parameter = projected sky-plane separation
    b = np.sqrt(x0**2 + y0**2)
    return b, z0

@njit
def _exozippy_getphase_scalar(
    eccen,
    omega,
    trueanom=None,
    primary=False,
    secondary=False,
    l4=False,
    l5=False,
    periastron=False,
    ascendingnode=False,
    descendingnode=False
):
    eccen = np.float64(eccen)
    omega = np.float64(omega)

    # Handle common special-case phase positions
    if periastron:
        trueanom = 0.0
    elif l5:
        trueanom = (5.0 * np.pi / 6.0) - omega
    elif l4:
        trueanom = (1.0 * np.pi / 6.0) - omega
    elif secondary:
        trueanom = (3.0 * np.pi / 2.0) - omega
    elif primary:
        trueanom = (1.0 * np.pi / 2.0) - omega
    elif ascendingnode:
        trueanom = -omega
    elif descendingnode:
        trueanom = np.pi - omega

    if trueanom is None:
        raise ValueError("Must specify trueanom or one of the special-case keywords")

    # Convert true anomaly to eccentric anomaly
    eccanom = 2.0 * np.arctan(np.sqrt((1.0 - eccen) / (1.0 + eccen)) * np.tan(trueanom / 2.0))

    # Mean anomaly
    M = eccanom - eccen * np.sin(eccanom)

    # Phase = M / 2π
    phase = M / (2.0 * np.pi)

    # Normalize to [0, 1]
    phase = np.mod(phase, 1.0)

    return phase

def exozippy_getphase(
    eccen,
    omega,
    trueanom=None,
    primary=False,
    secondary=False,
    l4=False,
    l5=False,
    periastron=False,
    ascendingnode=False,
    descendingnode=False
):
    """
    Wrapper to support scalar or vector eccen/omega inputs.
    """
    eccen_arr = np.asarray(eccen, dtype=np.float64)
    omega_arr = np.asarray(omega, dtype=np.float64)

    # Check if trueanom is array — if so, must use the vector path even
    # when eccen/omega are scalar (broadcast them to match trueanom shape)
    trueanom_is_array = trueanom is not None and np.ndim(trueanom) > 0
    if trueanom_is_array:
        trueanom_arr = np.asarray(trueanom, dtype=np.float64)
        eccen_arr = np.broadcast_to(eccen_arr, trueanom_arr.shape).copy()
        omega_arr = np.broadcast_to(omega_arr, trueanom_arr.shape).copy()

    if eccen_arr.ndim == 0 and omega_arr.ndim == 0:
        ta = float(trueanom) if trueanom is not None else None
        return _exozippy_getphase_scalar(
            float(eccen_arr.reshape(1)[0]),
            float(omega_arr.reshape(1)[0]),
            trueanom=ta,
            primary=primary,
            secondary=secondary,
            l4=l4,
            l5=l5,
            periastron=periastron,
            ascendingnode=ascendingnode,
            descendingnode=descendingnode
        )

    if eccen_arr.shape != omega_arr.shape:
        raise ValueError("eccen and omega must have the same shape")

    n = eccen_arr.size
    phase = np.empty_like(eccen_arr)
    eccen_flat = eccen_arr.reshape(-1)
    omega_flat = omega_arr.reshape(-1)
    phase_flat = phase.reshape(-1)

    if trueanom_is_array:
        trueanom_flat = trueanom_arr.reshape(-1)
        for i in range(n):
            phase_flat[i] = _exozippy_getphase_scalar(
                float(eccen_flat[i]),
                float(omega_flat[i]),
                trueanom=float(trueanom_flat[i]),
                primary=primary,
                secondary=secondary,
                l4=l4,
                l5=l5,
                periastron=periastron,
                ascendingnode=ascendingnode,
                descendingnode=descendingnode
            )
    else:
        ta = float(trueanom) if trueanom is not None else None
        for i in range(n):
            phase_flat[i] = _exozippy_getphase_scalar(
                float(eccen_flat[i]),
                float(omega_flat[i]),
                trueanom=ta,
                primary=primary,
                secondary=secondary,
                l4=l4,
                l5=l5,
                periastron=periastron,
                ascendingnode=ascendingnode,
                descendingnode=descendingnode
            )

    return phase


def tc2tt(time, e, inc, omega, period, tol=1e-15,
          tt2tc=False, ts2te=False, te2ts=False):
    """Convert between conjunction and transit/eclipse timing conventions."""
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
                -e * np.cos(omega) * np.cos(inc) ** 2 /
                (np.cos(thetaold) * np.sin(inc) ** 2 - e * np.sin(omega))
            )
        else:
            thetanew = np.arctan(
                -e * np.cos(omega) * np.cos(inc) ** 2 /
                (np.cos(thetaold) * np.sin(inc) ** 2 + e * np.sin(omega))
            )
        if np.max(np.abs(thetanew - thetaold)) < tol:
            break
    else:
        raise RuntimeError("tc2tt: maximum iterations exceeded")

    if ts2te or te2ts:
        phase = exozippy_getphase(e, omega, secondary=True)
        phase0 = exozippy_getphase(e, omega, trueanom=1.5 * np.pi - omega + thetanew)
        tfinal = time - period * (phase0 - phase) if te2ts else time + period * (phase0 - phase)
    else:
        phase = exozippy_getphase(e, omega, primary=True)
        phase0 = exozippy_getphase(e, omega, trueanom=0.5 * np.pi - omega + thetanew)
        tfinal = time - period * (phase0 - phase) if tt2tc else time + period * (phase0 - phase)

    toohigh = tfinal > time + period / 2.0
    if np.any(toohigh):
        tfinal[toohigh] -= period[toohigh] if period.size > 1 else period
    toolow = tfinal < time - period / 2.0
    if np.any(toolow):
        tfinal[toolow] += period[toolow] if period.size > 1 else period

    return tfinal
