import numpy as np

from .mkconstants import mkconstants
from .utils import exozippy_getphase, exozippy_occultquad_cel
from .tc2tt import tc2tt
from .blackbody import exozippy_blackbody


def _blackbody_wave(temp, wavelength_m):
    """Helper: blackbody with wavelength input (MKS units)."""
    return exozippy_blackbody(temp, wavelength_m, wave=True)


def exofast_recenter(par, period):
    """
    Port of exofast_recenter.pro.
    Recenters a periodic distribution to (mode - period/2, mode + period/2].
    """
    par = np.asarray(par, dtype=float).copy()
    period = np.asarray(period, dtype=float)

    hist, x = np.histogram(par, bins=100)
    mode = x[np.argmax(hist)]

    if period.size == 1:
        per = np.full_like(par, period.item(), dtype=float)
    elif period.size == par.size:
        per = period
    else:
        raise ValueError("period must have 1 or npar elements")

    if (mode - period / 2.0 - mode) == 0.0:
        return par

    nper = np.rint((mode - par) / per).astype(np.int64)
    par -= nper * per

    for _ in range(10):
        toohigh = par > (mode + period / 2.0)
        if not np.any(toohigh):
            break
        par[toohigh] -= per[toohigh]
    for _ in range(10):
        toolow = par <= (mode - period / 2.0)
        if not np.any(toolow):
            break
        par[toolow] += per[toolow]

    return par


    # tc2tt is now imported from exozippy.tc2tt


def _solve_mpsun_from_k(K_ms, period_days, e, mstar_msun, sini, constants):
    """
    Solve for planet mass in Msun given RV semi-amplitude K (m/s).
    Uses bisection on mpsun.
    """
    if K_ms <= 0 or sini <= 0:
        return 0.0

    G = constants['G']
    Msun = constants['GMsun'] / constants['G']
    day = constants['day']

    K_cms = K_ms * 100.0
    P = period_days * day
    fac = (2.0 * np.pi * G / P)**(1.0 / 3.0) / np.sqrt(1.0 - e**2)

    def k_of_mpsun(mpsun):
        mpsun = max(mpsun, 0.0)
        mstar = mstar_msun
        m_tot = (mstar + mpsun) * Msun
        return fac * (mpsun * Msun * sini) / (m_tot**(2.0 / 3.0))

    lo = 1e-12
    hi = 1.0
    while k_of_mpsun(hi) < K_cms and hi < 100:
        hi *= 2.0

    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if k_of_mpsun(mid) < K_cms:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def derivepars(samples, param_names, priors, e=0.0, omega=np.pi / 2.0,
               band_name="Sloani"):
    """
    Compute derived parameters (subset matching EXOFAST CSV list) from samples.

    Parameters
    ----------
    samples : ndarray, shape (nsamples, ndim)
    param_names : list[str]
    priors : dict
    e, omega : float
    band_name : str (used for depth_<band>_0 key)
    """
    const = mkconstants()
    names = list(param_names)

    def _get(name, default=None):
        if name in names:
            return samples[:, names.index(name)]
        if default is not None:
            return np.full(samples.shape[0], default, dtype=float)
        return None

    mstar = _get('mstar', priors.get('mstar', {}).get('value', 1.0))
    rstar = _get('rstar', priors.get('rstar', {}).get('value', 1.0))
    teff = _get('teff', priors.get('teff', {}).get('value', 5500.0))
    feh = _get('feh', priors.get('feh', {}).get('value', 0.0))
    distance = _get('distance', priors.get('distance', {}).get('value', 100.0))
    period = _get('period', priors.get('period', {}).get('value', 1.0))
    p = _get('p', priors.get('p', {}).get('value', 0.1))
    cosi = _get('cosi', priors.get('cosi', {}).get('value', 0.0))
    K = _get('K', priors.get('k_0', {}).get('value', 0.0))
    tc = _get('tc', priors.get('tc', {}).get('value', 0.0))
    u1 = _get('u1', priors.get('u1', {}).get('value', 0.3))
    u2 = _get('u2', priors.get('u2', {}).get('value', 0.2))
    f0 = _get('f0', priors.get('f0', {}).get('value', 1.0))
    gamma = _get('gamma', priors.get('gamma_0', {}).get('value', 0.0))

    logmstar = np.log10(mstar)
    rhostar = mstar / rstar**3 * const['RhoSun']
    logg = np.log10(mstar / rstar**2 * const['GravitySun'])
    lstar = (4.0 * np.pi * rstar**2 * teff**4 * const['sigmab'] / const['LSun'] * const['RSun']**2)

    sini = np.sqrt(np.maximum(0.0, 1.0 - cosi**2))
    mpsun = np.array([
        _solve_mpsun_from_k(K[i], period[i], e, mstar[i], sini[i], const)
        for i in range(samples.shape[0])
    ])
    mp = mpsun / (const['GMjupiter'] / const['GMsun'])
    q = mpsun / mstar

    rpsun = p * rstar
    rp = rpsun / (const['RJupiter'] / const['RSun'])

    logp = np.log10(period)

    G = const['GMsun'] / const['RSun']**3 * const['day']**2
    arsun = (G * (mstar + mpsun) * period**2 / (4.0 * np.pi**2))**(1.0 / 3.0)
    ar = arsun / rstar
    arp = ar / p
    a = arsun * const['RSun'] / const['AU']

    n = np.sqrt(const['GMsun'] * (mstar + mpsun) / (a * const['AU'])**3)
    omegagr = (3.0 * const['GMsun'] * mstar * n /
               (a * const['AU'] * const['c']**2 * (1.0 - e**2)) *
               180.0 / np.pi * 36525.0 * 86400.0)

    b = ar * cosi * (1.0 - e**2) / (1.0 + e * np.sin(omega))
    inc = np.arccos(cosi)
    ideg = np.degrees(inc)

    teq = teff * np.sqrt(1.0 / (2.0 * ar))
    dr = ar * (1.0 - e**2) / (1.0 + e * np.sin(omega))

    Qp = 1e6
    tcirc = (4.0 * Qp / 63.0 / (const['day'] * 365.25e9) *
             ((a * const['AU'])**3 / (const['GMsun'] * mstar))**0.5 *
             (mpsun / mstar) * (ar / p)**5 *
             (1.0 - e**2)**(13.0 / 2.0) / (1.0 + 6.0 * e**2))

    mconvratio = np.interp(
        mstar,
        [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6,
         1.7, 1.8, 1.9, 2.0, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8,
         2.9, 3.0],
        [0.3508, 0.1842, 0.0991, 0.0667, 0.0437, 0.0257, 0.0107, 0.0031,
         0.0003, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    )
    tefficiency = mconvratio * q**2.0 * ar**(-6.0)
    tce = (1.0 / 10.0 * q**2.0 * (ar / 40.0)**(-6.0))**(-1.0)
    tra = (1.0 / 1.25 * q**2.0 * (1 + q)**(5.0 / 6.0) * (ar / 6.0)**(-8.5))**(-1.0)

    fave = const['sigmab'] * teff**4 / (ar * (1.0 + e**2 / 2.0))**2 / 1e9

    t14 = period / np.pi * np.arcsin(
        np.sqrt((1.0 + np.abs(p))**2 - b**2) / (sini * ar)
    ) * np.sqrt(1.0 - e**2) / (1.0 + e * np.sin(omega))
    t23 = period / np.pi * np.arcsin(
        np.sqrt((1.0 - np.abs(p))**2 - b**2) / (sini * ar)
    ) * np.sqrt(1.0 - e**2) / (1.0 + e * np.sin(omega))
    notransit = np.abs(b) > (1.0 + np.abs(p))
    t14[notransit] = 0.0
    grazing = np.abs(b) > (1.0 - np.abs(p))
    t23[grazing] = 0.0
    tau = (t14 - t23) / 2.0
    tfwhm = t14 - tau

    ptg = (rstar + rpsun) / arsun * (1.0 + e * np.sin(omega)) / (1.0 - e**2)
    pt = (rstar - rpsun) / arsun * (1.0 + e * np.sin(omega)) / (1.0 - e**2)

    rhop = mpsun / (rpsun**3) * const['RhoSun']
    loggp = np.log10(mpsun / (rpsun**2) * const['GravitySun'])
    safronov = ar * q / p
    delta = p**2

    # Depth in the observed band (limb-darkening)
    mu1, _, _ = exozippy_occultquad_cel(np.abs(b), u1, u2, p)
    depth_band = 1.0 - mu1

    # Time conversions
    tt = tc2tt(tc, e, inc, omega, period, tt2tc=False)
    tp = tc - period * exozippy_getphase(e, omega, primary=True)
    ts = tc - period * (exozippy_getphase(e, omega, primary=True) -
                        exozippy_getphase(e, omega, secondary=True))
    ta = tc - period * (exozippy_getphase(e, omega, primary=True) -
                        exozippy_getphase(e, omega, ascendingnode=True))
    td = tc - period * (exozippy_getphase(e, omega, primary=True) -
                        exozippy_getphase(e, omega, descendingnode=True))

    tp = exofast_recenter(tp, np.median(period))
    ts = exofast_recenter(ts, np.median(period))
    ta = exofast_recenter(ta, np.median(period))
    td = exofast_recenter(td, np.median(period))

    # Observed-time placeholders (full EXOFAST uses transit data)
    t0 = tt.copy()
    tco = tc.copy()
    te = tc2tt(ts, e, inc, omega, period, ts2te=True)
    te0 = te.copy()
    tso = ts.copy()

    # Blackbody eclipse depths (ppm)
    starbb25 = _blackbody_wave(teff, np.full_like(teff, 2500e-9))
    starbb50 = _blackbody_wave(teff, np.full_like(teff, 5000e-9))
    starbb75 = _blackbody_wave(teff, np.full_like(teff, 7500e-9))
    planetbb25 = _blackbody_wave(teq, np.full_like(teq, 2500e-9))
    planetbb50 = _blackbody_wave(teq, np.full_like(teq, 5000e-9))
    planetbb75 = _blackbody_wave(teq, np.full_like(teq, 7500e-9))
    x25 = p**2 * planetbb25 / starbb25
    x50 = p**2 * planetbb50 / starbb50
    x75 = p**2 * planetbb75 / starbb75
    eclipsedepth25 = x25 / (1.0 + x25) * 1e6
    eclipsedepth50 = x50 / (1.0 + x50) * 1e6
    eclipsedepth75 = x75 / (1.0 + x75) * 1e6

    jittervar = np.full(samples.shape[0], priors.get('jittervar', {}).get('value', 0.0))
    jitter = np.sqrt(np.maximum(jittervar, 0.0))
    variance = np.full(samples.shape[0], priors.get('variance', {}).get('value', 0.0))

    return {
        'mstar_0': mstar,
        'rstar_0': rstar,
        'lstar_0': lstar,
        'rhostar_0': rhostar,
        'logg_0': logg,
        'teff_0': teff,
        'feh_0': feh,
        'logmstar_0': logmstar,
        'Period_0': period,
        'rp_0': rp,
        'mp_0': mp,
        'mpsun_0': mpsun,
        'tco_0': tco,
        'tc_0': tc,
        'tt_0': tt,
        't0_0': t0,
        'a_0': a,
        'ideg_0': ideg,
        'omegagr_0': omegagr,
        'teq_0': teq,
        'tcirc_0': tcirc,
        'tefficiency_0': tefficiency,
        'tce_0': tce,
        'tra_0': tra,
        'k_0': K,
        'p_0': p,
        'ar_0': ar,
        'arp_0': arp,
        'delta_0': delta,
        f'depth_{band_name}_0': depth_band,
        'tau_0': tau,
        't14_0': t14,
        'tfwhm_0': tfwhm,
        'b_0': b,
        'cosi_0': cosi,
        'eclipsedepth25_0': eclipsedepth25,
        'eclipsedepth50_0': eclipsedepth50,
        'eclipsedepth75_0': eclipsedepth75,
        'rhop_0': rhop,
        'logp_0': logp,
        'loggp_0': loggp,
        'safronov_0': safronov,
        'fave_0': fave,
        'tso_0': tso,
        'ts_0': ts,
        'te_0': te,
        'te0_0': te0,
        'tp_0': tp,
        'ta_0': ta,
        'td_0': td,
        'vcve_0': np.sqrt(1.0 - e**2) / (1.0 + e * np.sin(omega)),
        'msini_0': mp * sini,
        'q_0': q,
        'dr_0': dr,
        'pt_0': pt,
        'ptg_0': ptg,
        'u1_0': u1,
        'u2_0': u2,
        'gamma_0': gamma,
        'jitter_0': jitter,
        'jittervar_0': jittervar,
        'variance_0': variance,
        'f0_0': f0,
    }
