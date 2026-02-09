"""
Chi-squared computation for EXOZIPPy joint fitting.

Modular design following EXOFASTv2's exofast_chi2v2.pro:
  1. Parameter unpacking and bounds enforcement
  2. Derived quantities (logg, Lstar, a/R*, etc.)
  3. MIST evolutionary chi2
  4. SED chi2
  5. Transit chi2
  6. RV chi2
  7. Prior chi2

Each component returns a chi2 contribution (float) or np.inf for
invalid parameters, so the caller can accumulate:
    total_chi2 = bounds_chi2 + mist_chi2 + sed_chi2 + tran_chi2 + rv_chi2 + prior_chi2
"""

import numpy as np

from exozippy.sed.utils import mistmultised
from exozippy.massradius_mist import massradius_mist
from exozippy.exozippy_tran import exozippy_tran
from exozippy.exozippy_rv import exozippy_rv
from exozippy.utils import exozippy_getphase
from exozippy.mkconstants import mkconstants

CONSTANTS = mkconstants()

# ---------------------------------------------------------------------------
# Parameter ordering
# ---------------------------------------------------------------------------
BASE_PARAM_NAMES = [
    'teff', 'rstar', 'feh', 'av', 'distance',   # stellar (SED)
    'tc', 'period', 'p', 'cosi',                  # orbital (transit)
    'u1', 'u2', 'f0',                             # transit nuisance
    'K', 'gamma',                                  # RV
]

BASE_SCALES = np.array([
    30.0, 0.02, 0.05, 0.005, 1.0,     # stellar
    0.0003, 0.00005, 0.003, 0.01,      # orbital
    0.02, 0.02, 0.0002,               # transit nuisance
    2.0, 1.0,                          # RV
])

INF_CHI2 = 1e10  # sentinel for impossible parameters


def param_names(use_mist: bool):
    """Return the ordered parameter name list."""
    if use_mist:
        return ['mstar', 'age'] + BASE_PARAM_NAMES
    return BASE_PARAM_NAMES.copy()


# ---------------------------------------------------------------------------
# 1. Parameter unpacking
# ---------------------------------------------------------------------------
def unpack_params(params, use_mist=False, mstar_fixed=None, age_prior=None,
                  priors=None):
    """
    Unpack the flat parameter vector into a named dict.

    Returns
    -------
    d : dict  or  None (if out-of-bounds)
        Keys include all fitting parameters plus derived quantities.
    """
    idx = 0

    if use_mist:
        mstar = params[idx]; idx += 1
        age   = params[idx]; idx += 1
        if age <= 0 or age > 14.5:
            return None
    else:
        mstar = (mstar_fixed if mstar_fixed is not None
                 else (priors or {}).get('mstar', {}).get('value', 1.0))
        age   = (age_prior if age_prior is not None
                 else (priors or {}).get('age', {}).get('value', 1.0))

    teff, rstar, feh, av, distance = params[idx:idx+5]; idx += 5
    tc, period, p, cosi            = params[idx:idx+4]; idx += 4
    u1, u2, f0                     = params[idx:idx+3]; idx += 3
    K, gamma                       = params[idx:idx+2]

    return dict(
        mstar=mstar, age=age,
        teff=teff, rstar=rstar, feh=feh, av=av, distance=distance,
        tc=tc, period=period, p=p, cosi=cosi,
        u1=u1, u2=u2, f0=f0,
        K=K, gamma=gamma,
    )


# ---------------------------------------------------------------------------
# 2. Bounds enforcement
# ---------------------------------------------------------------------------
def check_bounds(d):
    """
    Return INF_CHI2 if any parameter is out of physical bounds, else 0.0.

    Follows the boundary checks in exofast_chi2v2.pro.
    """
    if d['mstar'] <= 0:
        return INF_CHI2
    if d['teff'] < 2500 or d['teff'] > 50000:
        return INF_CHI2
    if d['rstar'] < 0.05 or d['rstar'] > 100:
        return INF_CHI2
    if d['av'] < 0:
        return INF_CHI2
    if d['distance'] < 1:
        return INF_CHI2
    if d['period'] <= 0:
        return INF_CHI2
    if d['p'] <= 0 or d['p'] >= 0.5:
        return INF_CHI2
    if d['cosi'] < 0 or d['cosi'] >= 1:
        return INF_CHI2
    if d['f0'] <= 0:
        return INF_CHI2
    # Kipping (2013) limb darkening constraints
    if d['u1'] < 0:
        return INF_CHI2
    if d['u1'] + d['u2'] > 1:
        return INF_CHI2
    if d['u1'] + 2 * d['u2'] < 0:
        return INF_CHI2
    return 0.0


# ---------------------------------------------------------------------------
# 3. Derived quantities
# ---------------------------------------------------------------------------
def derive_logg(mstar, rstar):
    """log(g) in cgs from Mstar (Msun) and Rstar (Rsun)."""
    g = CONSTANTS['GravitySun'] * mstar / rstar**2
    return np.log10(g)


def derive_lstar(teff, rstar):
    """Stellar luminosity in Lsun from Teff and Rstar."""
    return (4.0 * np.pi * (rstar * CONSTANTS['RSun'])**2
            * CONSTANTS['sigmab'] * teff**4 / CONSTANTS['LSun'])


def derive_ar(period, mstar, rstar):
    """a/Rstar from Kepler's third law."""
    period_yr = period / 365.25
    a_au = (period_yr**2 * mstar)**(1.0 / 3.0)
    a_rsun = a_au * 215.094177   # AU in solar radii
    return a_rsun / rstar


def tc_to_tp(tc, period, e, omega):
    """Convert time of conjunction to time of periastron."""
    phase = exozippy_getphase(e, omega, primary=True)
    return tc - phase * period


def compute_derived(d, e, omega):
    """
    Add derived quantities to *d* in-place:
    logg, lstar, ar, inc, tp.
    """
    d['logg']  = derive_logg(d['mstar'], d['rstar'])
    d['lstar'] = derive_lstar(d['teff'], d['rstar'])
    d['ar']    = derive_ar(d['period'], d['mstar'], d['rstar'])
    d['inc']   = np.arccos(d['cosi'])
    d['tp']    = tc_to_tp(d['tc'], d['period'], e, omega)


# ---------------------------------------------------------------------------
# 4. MIST evolutionary chi2
# ---------------------------------------------------------------------------
def chi2_mist(d):
    """
    MIST isochrone chi2 penalty.

    Parameters
    ----------
    d : dict   (must contain mstar, feh, age, teff, rstar)

    Returns
    -------
    float   chi2 contribution (0 if not applicable, INF_CHI2 on failure).
    """
    try:
        val = massradius_mist(d['mstar'], d['feh'], d['age'],
                              d['teff'], d['rstar'])
    except Exception:
        return INF_CHI2
    if not np.isfinite(val):
        return INF_CHI2
    return val


# ---------------------------------------------------------------------------
# 5. SED chi2
# ---------------------------------------------------------------------------
def chi2_sed(d, sedfile, sed_data=None):
    """
    Broadband SED chi2 using MIST bolometric corrections.

    Parameters
    ----------
    d : dict       (must contain teff, logg, feh, av, distance, lstar)
    sedfile : str

    Returns
    -------
    float   chi2 contribution.
    """
    try:
        val, _, _, _ = mistmultised(
            d['teff'], d['logg'], d['feh'], d['av'],
            d['distance'], d['lstar'], 1.0, sedfile,
            sed_data=sed_data,
        )
    except Exception:
        return INF_CHI2
    if not np.isfinite(val):
        return INF_CHI2
    return val


# ---------------------------------------------------------------------------
# 6. Transit chi2
# ---------------------------------------------------------------------------
def chi2_transit(d, tran_data, e, omega, tran_addvar=0.0):
    """
    Transit light-curve chi2.

    Parameters
    ----------
    d : dict            (inc, ar, tp, period, p, u1, u2, f0)
    tran_data : dict    (bjd, flux, err)
    e, omega : float
    tran_addvar : float (extra variance to add in quadrature)

    Returns
    -------
    float   chi2 contribution.
    """
    try:
        model_flux = exozippy_tran(
            tran_data['bjd'], d['inc'], d['ar'], d['tp'], d['period'],
            e, omega, d['p'], d['u1'], d['u2'], d['f0'],
        )
    except Exception:
        return INF_CHI2
    resid = tran_data['flux'] - model_flux
    err2 = tran_data['err']**2 + tran_addvar
    val = np.sum(resid**2 / err2)
    if not np.isfinite(val):
        return INF_CHI2
    return val


# ---------------------------------------------------------------------------
# 7. RV chi2
# ---------------------------------------------------------------------------
def chi2_rv(d, rv_data, e, omega, rv_jittervar=0.0):
    """
    Radial-velocity chi2.

    Parameters
    ----------
    d : dict          (tp, period, gamma, K)
    rv_data : dict    (bjd, vel, err)
    e, omega : float
    rv_jittervar : float

    Returns
    -------
    float   chi2 contribution.
    """
    try:
        model_rv = exozippy_rv(
            rv_data['bjd'], d['tp'], d['period'],
            d['gamma'], d['K'], e=e, omega=omega,
        )
    except Exception:
        return INF_CHI2
    resid = rv_data['vel'] - model_rv
    err2 = rv_data['err']**2 + rv_jittervar
    val = np.sum(resid**2 / err2)
    if not np.isfinite(val):
        return INF_CHI2
    return val


# ---------------------------------------------------------------------------
# 8. Prior chi2
# ---------------------------------------------------------------------------
def chi2_priors(d, priors, use_mist=False):
    """
    Gaussian prior chi2 penalties + hard bounds.

    Parameters
    ----------
    d : dict       current parameter values
    priors : dict  from parse_priors()
    use_mist : bool

    Returns
    -------
    float   chi2 contribution.
    """
    chi2 = 0.0

    # Map fitting params → prior keys
    prior_map = {
        'teff': d['teff'], 'feh': d['feh'],
        'period_0': d['period'], 'tc': d['tc'],
        'p': d['p'], 'cosi': d['cosi'],
        'k_0': d['K'], 'gamma_0': d['gamma'],
        'f0': d['f0'],
    }
    if use_mist:
        prior_map['mstar'] = d['mstar']
        prior_map['age']   = d['age']

    for pname, pval in prior_map.items():
        if pname in priors and priors[pname]['sigma'] > 0:
            chi2 += ((pval - priors[pname]['value'])
                     / priors[pname]['sigma'])**2

    # Parallax prior (parallax = 1000 / distance)
    if 'parallax' in priors and priors['parallax']['sigma'] > 0:
        plx_model = 1000.0 / d['distance']
        chi2 += ((plx_model - priors['parallax']['value'])
                 / priors['parallax']['sigma'])**2

    # Av upper bound
    if 'av' in priors and np.isfinite(priors['av'].get('upper', np.nan)):
        if d['av'] > priors['av']['upper']:
            chi2 += ((d['av'] - priors['av']['upper']) / 0.001)**2

    return chi2


# ---------------------------------------------------------------------------
# 9. Combined chi2  (the main entry point, analogous to exofast_chi2v2)
# ---------------------------------------------------------------------------
def joint_chi2(params, tran_data, rv_data, sedfile, priors,
               e=0.0, omega=np.pi / 2,
               rv_jittervar=0.0, tran_addvar=0.0,
               use_mist=False, mstar_fixed=None, age_prior=None,
               sed_data=None):
    """
    Total chi2 for the joint SED (+ optional MIST) + Transit + RV fit.

    This is the top-level function called by the optimizer and MCMC.
    It mirrors the structure of EXOFASTv2's ``exofast_chi2v2.pro``.

    Parameters
    ----------
    params : ndarray
        Parameter vector in the order given by ``param_names(use_mist)``.
    tran_data, rv_data : dict
    sedfile : str
    priors : dict
    e, omega : float
    rv_jittervar, tran_addvar : float
    use_mist : bool
    mstar_fixed, age_prior : float or None

    Returns
    -------
    float
        Total chi2 (= -2 ln L).  Returns INF_CHI2 for impossible models.
    """
    # 1. Unpack
    d = unpack_params(params, use_mist=use_mist,
                      mstar_fixed=mstar_fixed, age_prior=age_prior,
                      priors=priors)
    if d is None:
        return INF_CHI2

    # 2. Bounds
    bnd = check_bounds(d)
    if bnd > 0:
        return bnd

    # 3. Derived quantities
    compute_derived(d, e, omega)

    total = 0.0

    # 4. MIST evolutionary penalty
    if use_mist:
        val = chi2_mist(d)
        if val >= INF_CHI2:
            return INF_CHI2
        total += val

    # 5. SED
    val = chi2_sed(d, sedfile, sed_data=sed_data)
    if val >= INF_CHI2:
        return INF_CHI2
    total += val

    # 6. Transit
    val = chi2_transit(d, tran_data, e, omega, tran_addvar)
    if val >= INF_CHI2:
        return INF_CHI2
    total += val

    # 7. RV
    val = chi2_rv(d, rv_data, e, omega, rv_jittervar)
    if val >= INF_CHI2:
        return INF_CHI2
    total += val

    # 8. Priors
    total += chi2_priors(d, priors, use_mist)

    return total
