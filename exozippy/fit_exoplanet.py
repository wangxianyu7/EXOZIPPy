"""
Joint SED + Transit + RV fitting for EXOZIPPy.

Simultaneously fits stellar parameters (Teff, Rstar, [Fe/H], Av, distance)
and planetary parameters (tc, period, p, cosi, u1, u2, f0, K, gamma) to
broadband photometry, transit light curve, and radial velocity data.
"""
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
import multiprocessing as mp
import sys
import functools

from exozippy.sed.utils import mistmultised
from exozippy.massradius_mist import massradius_mist
from exozippy.exozippy_tran import exozippy_tran
from exozippy.exozippy_rv import exozippy_rv
from exozippy.utils import exozippy_getphase
from exozippy.mkconstants import mkconstants
from exozippy.amoeba import amoeba
from exozippy.exozippy_demcpt import exozippy_demcpt

CONSTANTS = mkconstants()


def _mcmc_log_prob(params, args):
    (tran_data, rv_data, sedfile, priors, e, omega,
     rv_jittervar, tran_addvar, use_mist,
     mstar_prior, age_prior) = args
    chi2 = joint_negloglike(params, tran_data, rv_data, sedfile,
                            priors, e, omega,
                            rv_jittervar, tran_addvar,
                            use_mist=use_mist,
                            mstar_fixed=mstar_prior,
                            age_prior=age_prior)
    if not np.isfinite(chi2) or chi2 > 1e9:
        return -np.inf
    return -0.5 * chi2

# Parameter ordering when fitting without MIST (14 parameters)
BASE_PARAM_NAMES = [
    'teff', 'rstar', 'feh', 'av', 'distance',  # stellar (SED)
    'tc', 'period', 'p', 'cosi',                 # orbital (transit)
    'u1', 'u2', 'f0',                            # transit nuisance
    'K', 'gamma',                                 # RV
]

BASE_SCALES = np.array([
    30.0, 0.02, 0.05, 0.005, 1.0,    # stellar
    0.0003, 0.00005, 0.003, 0.01,     # orbital
    0.02, 0.02, 0.0002,               # transit nuisance
    2.0, 1.0,                          # RV
])


def _log(msg: str, verbose: bool):
    if verbose:
        print(msg)


def _log_section(title: str, verbose: bool):
    if verbose:
        bar = '=' * len(title)
        print(f'\n{bar}\n{title}\n{bar}')


def _param_names(use_mist: bool):
    if use_mist:
        return ['mstar', 'age'] + BASE_PARAM_NAMES
    return BASE_PARAM_NAMES.copy()


def parse_priors(priorfile):
    """
    Parse an EXOFASTv2-style prior file.

    Returns
    -------
    priors : dict
        Keys are parameter names; values are dicts with fields:
        'value', 'sigma' (0 if none), 'lower', 'upper' (NaN if none).
    """
    priors = {}
    with open(priorfile) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            parts = line.split()
            name = parts[0]
            vals = [float(x) for x in parts[1:]]
            entry = {'value': vals[0], 'sigma': 0.0,
                     'lower': np.nan, 'upper': np.nan}
            if len(vals) >= 2:
                entry['sigma'] = vals[1]
            if len(vals) >= 3:
                entry['lower'] = vals[2]
            if len(vals) >= 4:
                entry['upper'] = vals[3]
            priors[name] = entry
    return priors


def read_transit_data(tranfile):
    """Read a transit light curve file (BJD flux err)."""
    data = np.loadtxt(tranfile, comments='#')
    return {
        'bjd': data[:, 0],
        'flux': data[:, 1],
        'err': data[:, 2],
    }


def read_rv_data(rvfile):
    """Read an RV data file (BJD vel err_vel)."""
    data = np.loadtxt(rvfile, comments='#')
    return {
        'bjd': data[:, 0],
        'vel': data[:, 1],
        'err': data[:, 2],
    }


def _derive_logg(mstar, rstar):
    """Compute log(g) in cgs from Mstar (Msun) and Rstar (Rsun)."""
    g = CONSTANTS['GravitySun'] * mstar / rstar**2
    return np.log10(g)


def _derive_lstar(teff, rstar):
    """Compute Lstar in solar luminosities from Teff and Rstar."""
    return (4.0 * np.pi * (rstar * CONSTANTS['RSun'])**2
            * CONSTANTS['sigmab'] * teff**4 / CONSTANTS['LSun'])


def _derive_ar(period, mstar, rstar):
    """Compute a/Rstar from Kepler's third law."""
    period_yr = period / 365.25
    a_au = (period_yr**2 * mstar)**(1./3.)
    a_rsun = a_au * 215.094177  # AU in solar radii
    return a_rsun / rstar


def tc_to_tp(tc, period, e, omega):
    """Convert time of conjunction to time of periastron."""
    phase = exozippy_getphase(e, omega, primary=True)
    return tc - phase * period


def joint_negloglike(params, tran_data, rv_data, sedfile, priors,
                     e=0.0, omega=np.pi/2,
                     rv_jittervar=0.0, tran_addvar=0.0,
                     use_mist=False, mstar_fixed=None, age_prior=None):
    """
    Total negative-log-likelihood for the joint SED (+ optional MIST) + Transit + RV fit.

    Parameters
    ----------
    params : ndarray
        Parameter vector in the order defined by `_param_names(use_mist)`.
    tran_data : dict
        From read_transit_data.
    rv_data : dict
        From read_rv_data.
    sedfile : str
    priors : dict
    mstar_fixed : float (Msun, optional)
        Reference stellar mass when not fitting MIST evolutionary constraints.
    e : float
    omega : float
    use_mist : bool
        If True, first two entries of `params` are (mstar, age) and a MIST chi2
        penalty is applied.
    mstar_fixed : float
        Value to use for mstar when not fitting it directly.
    age_prior : float
        Value to use for age when not fitting it directly.

    Returns
    -------
    float
        -2 * ln(likelihood) = sed_chi2 + transit_chi2 + rv_chi2 + prior_chi2
    """
    idx = 0
    if use_mist:
        mstar = params[idx]
        idx += 1
        age = params[idx]
        idx += 1
        if age <= 0 or age > 14.5:
            return 1e10
    else:
        mstar = mstar_fixed if mstar_fixed is not None else priors.get('mstar', {}).get('value', 1.0)
        age = age_prior if age_prior is not None else priors.get('age', {}).get('value', 1.0)

    teff, rstar, feh, av, distance = params[idx:idx+5]
    idx += 5
    tc, period, p, cosi = params[idx:idx+4]
    idx += 4
    u1, u2, f0 = params[idx:idx+3]
    idx += 3
    K, gamma = params[idx:idx+2]

    # --- Bounds enforcement ---
    if mstar <= 0:
        return 1e10
    if teff < 2500 or teff > 50000:
        return 1e10
    if rstar < 0.05 or rstar > 100:
        return 1e10
    if av < 0:
        return 1e10
    if distance < 1:
        return 1e10
    if period <= 0:
        return 1e10
    if p <= 0 or p >= 0.5:
        return 1e10
    if cosi < 0 or cosi >= 1:
        return 1e10
    if f0 <= 0:
        return 1e10
    # Kipping (2013) limb darkening bounds
    if u1 < 0:
        return 1e10
    if u1 + u2 > 1:
        return 1e10
    if u1 + 2 * u2 < 0:
        return 1e10

    # --- Derived quantities ---
    logg = _derive_logg(mstar, rstar)
    lstar = _derive_lstar(teff, rstar)
    ar = _derive_ar(period, mstar, rstar)
    inc = np.arccos(cosi)
    tp = tc_to_tp(tc, period, e, omega)

    total_chi2 = 0.0

    # --- MIST evolutionary chi2 ---
    if use_mist:
        try:
            mist_chi2 = massradius_mist(mstar, feh, age, teff, rstar)
        except Exception:
            return 1e10
        if not np.isfinite(mist_chi2):
            return 1e10
        total_chi2 += mist_chi2

    # --- SED chi2 ---
    try:
        sedchi2, _, _, _ = mistmultised(
            teff, logg, feh, av, distance, lstar, 1.0, sedfile
        )
    except Exception:
        return 1e10
    if not np.isfinite(sedchi2):
        return 1e10
    total_chi2 += sedchi2

    # --- Transit chi2 ---
    try:
        model_flux = exozippy_tran(
            tran_data['bjd'], inc, ar, tp, period, e, omega,
            p, u1, u2, f0
        )
    except Exception:
        return 1e10
    tran_resid = tran_data['flux'] - model_flux
    tran_err2 = tran_data['err']**2 + tran_addvar
    tran_chi2 = np.sum(tran_resid**2 / tran_err2)
    if not np.isfinite(tran_chi2):
        return 1e10
    total_chi2 += tran_chi2

    # --- RV chi2 ---
    try:
        model_rv = exozippy_rv(
            rv_data['bjd'], tp, period, gamma, K, e=e, omega=omega
        )
    except Exception:
        return 1e10
    rv_resid = rv_data['vel'] - model_rv
    rv_err2 = rv_data['err']**2 + rv_jittervar
    rv_chi2 = np.sum(rv_resid**2 / rv_err2)
    if not np.isfinite(rv_chi2):
        return 1e10
    total_chi2 += rv_chi2

    # --- Prior chi2 ---
    prior_chi2 = 0.0

    # Map fitting params to prior names
    prior_map = {
        'teff': teff, 'feh': feh,
        'period_0': period, 'tc': tc,
        'p': p, 'cosi': cosi,
        'k_0': K, 'gamma_0': gamma,
        'f0': f0,
    }
    if use_mist:
        prior_map['mstar'] = mstar
        prior_map['age'] = age
    for pname, pval in prior_map.items():
        if pname in priors and priors[pname]['sigma'] > 0:
            prior_chi2 += ((pval - priors[pname]['value'])
                           / priors[pname]['sigma'])**2

    # Parallax prior (parallax = 1000/distance)
    if 'parallax' in priors and priors['parallax']['sigma'] > 0:
        plx_model = 1000.0 / distance
        prior_chi2 += ((plx_model - priors['parallax']['value'])
                       / priors['parallax']['sigma'])**2

    # Av upper bound
    if 'av' in priors and np.isfinite(priors['av'].get('upper', np.nan)):
        if av > priors['av']['upper']:
            prior_chi2 += ((av - priors['av']['upper']) / 0.001)**2

    total_chi2 += prior_chi2

    return total_chi2


def fit_exoplanet(priorfile, tranfile, rvfile, sedfile, e=0.0,
                  omega=np.pi/2, verbose=True, use_mist=False):
    """
    Joint fit of SED (+ optional MIST evolutionary prior) + Transit + RV data.

    Returns
    -------
    result : dict
        Best-fit parameters, chi2, derived quantities.
    """
    start_time = datetime.utcnow()
    _log_section('Joint SED + Transit + RV Fit', verbose)
    _log(f'Start time (UTC): {start_time:%Y-%m-%d %H:%M:%S}', verbose)
    _log(f'Prior file       : {priorfile}', verbose)
    _log(f'Transit file     : {tranfile}', verbose)
    _log(f'RV file          : {rvfile}', verbose)
    _log(f'SED file         : {sedfile}', verbose)
    _log(f'Orbital params   : e={e}, omega={omega:.4f}', verbose)
    _log(f'Use MIST         : {use_mist}', verbose)

    priors = parse_priors(priorfile)
    tran_data = read_transit_data(tranfile)
    rv_data = read_rv_data(rvfile)
    mstar_prior = priors.get('mstar', {}).get('value', 1.0)
    age_prior = priors.get('age', {}).get('value', 1.0)
    param_names = _param_names(use_mist)

    # Fixed jitter/variance from priors
    rv_jittervar = priors.get('jittervar', {}).get('value', 0.0)
    tran_addvar = priors.get('variance', {}).get('value', 0.0)

    # Starting values
    teff0 = priors.get('teff', {}).get('value', 5500.0)
    rstar0 = priors.get('rstar', {}).get('value', 1.0)
    feh0 = priors.get('feh', {}).get('value', 0.0)
    av0 = 0.01
    if 'parallax' in priors:
        dist0 = 1000.0 / priors['parallax']['value']
    else:
        dist0 = 135.0
    tc0 = priors.get('tc', {}).get('value', np.median(tran_data['bjd']))
    period0 = priors.get('period_0', {}).get('value', 3.0)
    p0 = priors.get('p', {}).get('value', 0.1)
    cosi0 = priors.get('cosi', {}).get('value', 0.05)
    u1_0 = 0.4
    u2_0 = 0.2
    f0_0 = priors.get('f0', {}).get('value', np.median(tran_data['flux']))
    K0 = priors.get('k_0', {}).get('value', 50.0)
    gamma0 = priors.get('gamma_0', {}).get('value', 0.0)

    x0_list = []
    if use_mist:
        x0_list.extend([mstar_prior, max(age_prior, 0.5)])
    x0_list.extend([
        teff0, rstar0, feh0, av0, dist0,
        tc0, period0, p0, cosi0,
        u1_0, u2_0, f0_0,
        K0, gamma0
    ])
    x0 = np.array(x0_list)

    if verbose:
        if use_mist:
            _log(f"Initial mstar/age: {mstar_prior:.3f} Msun, {age_prior:.2f} Gyr", verbose)
        else:
            _log(f"Fixed mstar = {mstar_prior:.3f} Msun", verbose)
        _log(f"Transit data     : {len(tran_data['bjd'])} points", verbose)
        _log(f"RV data          : {len(rv_data['bjd'])} points", verbose)
        _log("Starting parameters:", verbose)
        _log(f"  Teff={teff0:.0f}K  Rstar={rstar0:.3f}  [Fe/H]={feh0:.3f}  Av={av0:.3f}  dist={dist0:.1f}", verbose)
        _log(f"  tc={tc0:.5f}  P={period0:.6f}  p={p0:.4f}  cosi={cosi0:.4f}", verbose)
        _log(f"  u1={u1_0:.2f}  u2={u2_0:.2f}  f0={f0_0:.5f}", verbose)
        _log(f"  K={K0:.1f}  gamma={gamma0:.1f}", verbose)
        chi2_init = joint_negloglike(
            x0, tran_data, rv_data, sedfile, priors,
            e, omega, rv_jittervar, tran_addvar,
            use_mist=use_mist,
            mstar_fixed=mstar_prior,
            age_prior=age_prior)
        _log(f"Initial chi2     : {chi2_init:.2f}", verbose)

    # Av upper bound
    av_upper = priors.get('av', {}).get('upper', 1.0)
    if not np.isfinite(av_upper):
        av_upper = 1.0

    # Phase 1: Nelder-Mead
    scale_vec = BASE_SCALES.copy()
    if use_mist:
        scale_vec = np.concatenate(([0.05, 0.5], scale_vec))

    if verbose:
        _log('Starting Amoeba (Nelder-Mead) search...', verbose)
    amoeba_sol, amoeba_chi2, amoeba_info = amoeba(
        lambda params: joint_negloglike(
            params, tran_data, rv_data, sedfile, priors,
            e, omega, rv_jittervar, tran_addvar,
            use_mist, mstar_prior, age_prior),
        x0=x0, scale=scale_vec,
        ftol=1e-6, maxiter=50000, verbose=verbose)
    if verbose:
        if amoeba_info['success']:
            _log(f'Amoeba converged: chi2 ~ {amoeba_chi2:.2f}', verbose)
        else:
            _log('Amoeba reached max iterations; proceeding to L-BFGS-B.', verbose)

    # Phase 2: L-BFGS-B with bounds
    bounds = []
    if use_mist:
        bounds.extend([
            (0.1, 2.0),         # mstar
            (0.01, 14.0),       # age (Gyr)
        ])
    bounds.extend([
        (3000, 10000),          # teff
        (0.1, 10.0),            # rstar
        (-2.0, 0.75),           # feh
        (0.0, av_upper),        # av
        (1.0, 10000.0),         # distance
        (tc0 - 0.5, tc0 + 0.5), # tc
        (period0 * 0.99, period0 * 1.01),  # period
        (0.001, 0.5),           # p
        (0.0, 0.99),            # cosi
        (0.0, 2.0),             # u1
        (-1.0, 1.0),            # u2
        (0.5, 1.5),             # f0
        (0.1, 500.0),           # K
        (-1000.0, 1000.0),      # gamma
    ])

    if verbose:
        _log('Starting L-BFGS-B refinement...', verbose)
        class ProgressCallback:
            def __init__(self):
                self.count = 0

            def __call__(self, xk):
                self.count += 1
                if self.count % 20 == 0:
                    _log(f'  Iteration {self.count}: chi2 ~ {joint_negloglike(xk, tran_data, rv_data, sedfile, priors, e, omega, rv_jittervar, tran_addvar, use_mist, mstar_prior, age_prior):.2f}', verbose)
        callback = ProgressCallback()
    else:
        callback = None

    res = minimize(joint_negloglike, amoeba_sol,
                   args=(tran_data, rv_data, sedfile, priors,
                         e, omega, rv_jittervar, tran_addvar,
                         use_mist, mstar_prior, age_prior),
                   method='L-BFGS-B', bounds=bounds, callback=callback,
                   options={'maxiter': 10000, 'ftol': 1e-12})

    bf = res.x
    idx = 0
    if use_mist:
        mstar_f = bf[idx]
        idx += 1
        age_f = bf[idx]
        idx += 1
    else:
        mstar_f = mstar_prior
        age_f = age_prior
    teff_f, rstar_f, feh_f, av_f, dist_f = bf[idx:idx+5]
    idx += 5
    tc_f, period_f, p_f, cosi_f = bf[idx:idx+4]
    idx += 4
    u1_f, u2_f, f0_f = bf[idx:idx+3]
    idx += 3
    K_f, gamma_f = bf[idx:idx+2]

    logg_f = _derive_logg(mstar_f, rstar_f)
    lstar_f = _derive_lstar(teff_f, rstar_f)
    ar_f = _derive_ar(period_f, mstar_f, rstar_f)
    inc_f = np.arccos(cosi_f)
    b_f = ar_f * cosi_f

    ndata = len(tran_data['bjd']) + len(rv_data['bjd']) + 9  # 9 SED bands
    ndof = ndata - len(param_names)
    nfit = ndata - ndof
    bic = res.fun + nfit * np.log(ndata)
    aic = res.fun + 2 * nfit

    result = {
        'teff': teff_f, 'rstar': rstar_f, 'feh': feh_f,
        'av': av_f, 'distance': dist_f,
        'tc': tc_f, 'period': period_f, 'p': p_f, 'cosi': cosi_f,
        'u1': u1_f, 'u2': u2_f, 'f0': f0_f,
        'K': K_f, 'gamma': gamma_f,
        'logg': logg_f, 'lstar': lstar_f, 'ar': ar_f,
        'inc_rad': inc_f, 'ideg': np.degrees(inc_f),
        'b': b_f, 'delta': p_f**2,
        'parallax': 1000.0 / dist_f, 'mstar': mstar_f,
        'age': age_f,
        'e': e, 'omega': omega,
        'rv_jittervar': rv_jittervar, 'tran_addvar': tran_addvar,
        'chi2': res.fun, 'ndata': ndata, 'ndof': ndof,
        'chi2_red': res.fun / ndof if ndof > 0 else np.nan,
        'bic': bic, 'aic': aic,
        'success': res.success, 'message': res.message,
        'param_names': param_names, 'use_mist': use_mist,
    }

    if verbose:
        _log_section('Optimizer Summary', verbose)
        _log(f"Mstar    = {mstar_f:.4f} Msun", verbose)
        if use_mist:
            _log(f"Age      = {age_f:.3f} Gyr", verbose)
        _log(f"Teff     = {teff_f:.1f} K", verbose)
        _log(f"Rstar    = {rstar_f:.4f} Rsun", verbose)
        _log(f"[Fe/H]   = {feh_f:.4f}", verbose)
        _log(f"Av       = {av_f:.4f}", verbose)
        _log(f"Distance = {dist_f:.2f} pc (plx = {1000./dist_f:.4f} mas)", verbose)
        _log(f"logg     = {logg_f:.4f}", verbose)
        _log(f"Lstar    = {lstar_f:.4f} Lsun", verbose)
        _log('---', verbose)
        _log(f"Tc       = {tc_f:.5f}", verbose)
        _log(f"Period   = {period_f:.6f} d", verbose)
        _log(f"Rp/Rs    = {p_f:.4f}", verbose)
        _log(f"cosi     = {cosi_f:.4f} (i = {np.degrees(inc_f):.2f} deg)", verbose)
        _log(f"a/Rs     = {ar_f:.2f}", verbose)
        _log(f"b        = {b_f:.4f}", verbose)
        _log(f"u1       = {u1_f:.4f}", verbose)
        _log(f"u2       = {u2_f:.4f}", verbose)
        _log(f"f0       = {f0_f:.5f}", verbose)
        _log('---', verbose)
        _log(f"K        = {K_f:.2f} m/s", verbose)
        _log(f"gamma    = {gamma_f:.2f} m/s", verbose)
        _log('---', verbose)
        _log(f"Chi2     = {res.fun:.4f}", verbose)
        _log(f"Chi2/dof = {result['chi2_red']:.4f}", verbose)
        _log(f"NDATA    = {ndata}", verbose)
        _log(f"NFIT     = {nfit}", verbose)
        _log(f"BIC      = {bic:.4f}", verbose)
        _log(f"AIC      = {aic:.4f}", verbose)
        _log(f"Success  = {res.success}", verbose)
        end_time = datetime.utcnow()
        elapsed = end_time - start_time
        _log_section('Run Complete', verbose)
        _log(f'End time (UTC): {end_time:%Y-%m-%d %H:%M:%S}', verbose)
        _log(f'Elapsed       : {elapsed}', verbose)

    return result


def run_mcmc(priorfile, tranfile, rvfile, sedfile, bestfit=None,
             e=0.0, omega=np.pi/2, nwalkers=32, nsteps=2000,
             nburn=500, verbose=True, use_mist=False, nthreads=None,
             backend='emcee'):
    """Run MCMC sampling around the best-fit joint solution."""
    backend = (backend or 'emcee').lower()

    priors = parse_priors(priorfile)
    tran_data = read_transit_data(tranfile)
    rv_data = read_rv_data(rvfile)
    mstar_prior = priors.get('mstar', {}).get('value', 1.0)
    age_prior = priors.get('age', {}).get('value', 1.0)
    rv_jittervar = priors.get('jittervar', {}).get('value', 0.0)
    tran_addvar = priors.get('variance', {}).get('value', 0.0)

    if bestfit is None:
        bestfit = fit_exoplanet(priorfile, tranfile, rvfile, sedfile,
                                e=e, omega=omega, verbose=verbose,
                                use_mist=use_mist)

    param_names = bestfit.get('param_names') or _param_names(use_mist)
    name_to_idx = {name: i for i, name in enumerate(param_names)}
    x_best = np.array([bestfit[n] for n in param_names])
    ndim = len(x_best)

    # Initialization scatter
    if use_mist:
        scales = np.concatenate(([0.02, 0.5], BASE_SCALES))
    else:
        scales = BASE_SCALES

    if backend == 'demcpt':
        if verbose:
            print(f"=== DEMC-PT: {nwalkers} chains, {nsteps} steps ===")
        bestpars = {name: bestfit[name] for name in param_names}
        scale_dict = {name: max(abs(scales[i]), 1e-3) for i, name in enumerate(param_names)}

        def chi2_dict(pars_dict):
            params = np.array([pars_dict[name] for name in param_names])
            return joint_negloglike(params, tran_data, rv_data, sedfile,
                                    priors, e, omega,
                                    rv_jittervar, tran_addvar,
                                    use_mist=use_mist,
                                    mstar_fixed=mstar_prior,
                                    age_prior=age_prior)

        demcpt_res = exozippy_demcpt(
            chi2_dict,
            bestpars,
            scale=scale_dict,
            nchains=nwalkers,
            maxsteps=nsteps,
            debug=verbose,
        )
        burn = demcpt_res.get('burnndx', max(nsteps // 4, 1))
        cold_chains = demcpt_res['pars'][0, :, :, burn:]
        if cold_chains.shape[-1] == 0:
            cold_chains = demcpt_res['pars'][0, :, :, :]
        samples = cold_chains.reshape(-1, ndim)
    else:
        import emcee

        pos = x_best + scales * np.random.randn(nwalkers, ndim)
        # Enforce bounds
        if use_mist:
            age_idx = name_to_idx['age']
            pos[:, age_idx] = np.clip(pos[:, age_idx], 0.01, 14.0)

        av_idx = name_to_idx['av']
        p_idx = name_to_idx['p']
        cosi_idx = name_to_idx['cosi']
        pos[:, av_idx] = np.abs(pos[:, av_idx])          # av >= 0
        pos[:, p_idx] = np.abs(pos[:, p_idx])            # p > 0
        pos[:, cosi_idx] = np.clip(pos[:, cosi_idx], 0, 0.99)  # 0 <= cosi < 1

        log_prob_args = (tran_data, rv_data, sedfile, priors, e, omega,
                         rv_jittervar, tran_addvar, use_mist, mstar_prior, age_prior)
        log_prob = functools.partial(_mcmc_log_prob, args=log_prob_args)

        if verbose:
            print(f"=== MCMC: {nwalkers} walkers, {nsteps} steps ===")

        pool = None
        if nthreads and nthreads > 1:
            ctx = mp.get_context('fork' if sys.platform != 'win32' else 'spawn')
            pool = ctx.Pool(nthreads)

        try:
            sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob, pool=pool)
            if verbose:
                report_every = max(1, nsteps // 20)
                for i, _ in enumerate(sampler.sample(pos, iterations=nsteps, progress=False), 1):
                    if i == 1 or i % report_every == 0 or i == nsteps:
                        frac = 100.0 * i / nsteps
                        print(f"MCMC progress: {i}/{nsteps} steps ({frac:5.1f}%)")
            else:
                sampler.run_mcmc(pos, nsteps, progress=False)
        finally:
            if pool is not None:
                pool.close()
                pool.join()

        samples = sampler.get_chain(discard=nburn, flat=True)

    if samples.size == 0:
        raise RuntimeError("No MCMC samples generated.")

    def _arr(name):
        return samples[:, name_to_idx[name]]

    mstar_arr = _arr('mstar') if 'mstar' in name_to_idx else np.full(samples.shape[0], mstar_prior)
    rstar_arr = _arr('rstar')
    teff_arr = _arr('teff')
    period_arr = _arr('period')
    cosi_arr = _arr('cosi')
    distance_arr = _arr('distance')

    # Derived parameter arrays
    logg_s = _derive_logg(mstar_arr, rstar_arr)
    lstar_s = _derive_lstar(teff_arr, rstar_arr)
    ar_s = _derive_ar(period_arr, mstar_arr, rstar_arr)
    ideg_s = np.degrees(np.arccos(cosi_arr))
    b_s = ar_s * cosi_arr
    plx_s = 1000.0 / distance_arr

    summary = {}
    # Free parameters
    for i, name in enumerate(param_names):
        med = np.median(samples[:, i])
        lo = np.percentile(samples[:, i], 15.87)
        hi = np.percentile(samples[:, i], 84.13)
        summary[name] = (med, med - lo, hi - med)
        if verbose:
            print(f"{name:10s} = {med:.4f}  -{med-lo:.4f}  +{hi-med:.4f}")

    # Derived parameters
    for name, arr in [('logg', logg_s), ('lstar', lstar_s), ('ar', ar_s),
                      ('ideg', ideg_s), ('b', b_s), ('parallax', plx_s)]:
        med = np.median(arr)
        lo = np.percentile(arr, 15.87)
        hi = np.percentile(arr, 84.13)
        summary[name] = (med, med - lo, hi - med)
        if verbose:
            print(f"{name:10s} = {med:.4f}  -{med-lo:.4f}  +{hi-med:.4f}")

    summary['backend'] = backend
    return samples, param_names, summary
