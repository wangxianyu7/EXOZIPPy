"""
Joint SED + Transit + RV fitting for EXOZIPPy.

Simultaneously fits stellar parameters (Teff, Rstar, [Fe/H], Av, distance)
and planetary parameters (tc, period, p, cosi, u1, u2, f0, K, gamma) to
broadband photometry, transit light curve, and radial velocity data.
"""
import numpy as np
import os
from scipy.optimize import minimize
from datetime import datetime
import functools

from exozippy.amoeba import amoeba
from exozippy.exozippy_demcpt import DEMCPTSampler
from exozippy.exozippy_chi2 import (
    joint_chi2,
    param_names as _param_names,
    derive_logg as _derive_logg,
    derive_lstar as _derive_lstar,
    derive_ar as _derive_ar,
    BASE_PARAM_NAMES, BASE_SCALES,
)

# Keep backward-compatible alias
joint_negloglike = joint_chi2


def _mcmc_log_posterior(theta, tran_data, rv_data, sedfile, priors, sed_data,
                        e, omega, rv_jittervar, tran_addvar, use_mist,
                        mstar_fixed, age_prior):
    """Module-level log-posterior for picklability with multiprocessing."""
    chi2 = joint_chi2(theta, tran_data, rv_data, sedfile, priors,
                      sed_data=sed_data,
                      e=e, omega=omega,
                      rv_jittervar=rv_jittervar, tran_addvar=tran_addvar,
                      use_mist=use_mist, mstar_fixed=mstar_fixed,
                      age_prior=age_prior)
    if not np.isfinite(chi2) or chi2 > 1e9:
        return -np.inf
    return -0.5 * chi2


def _log(msg: str, verbose: bool):
    if verbose:
        print(msg)


def _log_section(title: str, verbose: bool):
    if verbose:
        bar = '=' * len(title)
        print(f'\n{bar}\n{title}\n{bar}')


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


def build_initial_guess(priorfile, tranfile, rvfile, e=0.0,
                        omega=np.pi/2, use_mist=False):
    """
    Construct a bestfit-like dict from priors (before optimization).

    Returns the same dict structure as fit_exoplanet so it can be
    passed directly to plottran / plotrv / plotsed.
    """
    priors = parse_priors(priorfile)
    tran_data = read_transit_data(tranfile)
    mstar = priors.get('mstar', {}).get('value', 1.0)
    age = priors.get('age', {}).get('value', 1.0)

    teff = priors.get('teff', {}).get('value', 5500.0)
    rstar = priors.get('rstar', {}).get('value', 1.0)
    feh = priors.get('feh', {}).get('value', 0.0)
    av = 0.01
    if 'parallax' in priors:
        distance = 1000.0 / priors['parallax']['value']
    else:
        distance = 135.0
    tc = priors.get('tc', {}).get('value', np.median(tran_data['bjd']))
    period = priors.get('period_0', {}).get('value', 3.0)
    p = priors.get('p', {}).get('value', 0.1)
    cosi = priors.get('cosi', {}).get('value', 0.05)
    u1 = 0.4
    u2 = 0.2
    f0 = priors.get('f0', {}).get('value', np.median(tran_data['flux']))
    K = priors.get('k_0', {}).get('value', 50.0)
    gamma = priors.get('gamma_0', {}).get('value', 0.0)

    logg = _derive_logg(mstar, rstar)
    lstar = _derive_lstar(teff, rstar)
    ar = _derive_ar(period, mstar, rstar)
    inc = np.arccos(cosi)

    return {
        'teff': teff, 'rstar': rstar, 'feh': feh,
        'av': av, 'distance': distance,
        'tc': tc, 'period': period, 'p': p, 'cosi': cosi,
        'u1': u1, 'u2': u2, 'f0': f0,
        'K': K, 'gamma': gamma,
        'logg': logg, 'lstar': lstar, 'ar': ar,
        'inc_rad': inc, 'ideg': np.degrees(inc),
        'b': ar * cosi, 'delta': p**2,
        'parallax': 1000.0 / distance, 'mstar': mstar,
        'age': age, 'e': e, 'omega': omega,
        'param_names': _param_names(use_mist), 'use_mist': use_mist,
    }


def _bestfit_from_params(params, param_names, e, omega, mstar_prior, age_prior,
                         use_mist):
    """Build a bestfit-like dict from a parameter vector."""
    d = {name: params[i] for i, name in enumerate(param_names)}

    mstar = d.get('mstar', mstar_prior)
    age = d.get('age', age_prior)

    teff = d['teff']
    rstar = d['rstar']
    feh = d['feh']
    av = d['av']
    distance = d['distance']
    tc = d['tc']
    period = d['period']
    p = d['p']
    cosi = d['cosi']
    u1 = d['u1']
    u2 = d['u2']
    f0 = d['f0']
    K = d['K']
    gamma = d['gamma']

    logg = _derive_logg(mstar, rstar)
    lstar = _derive_lstar(teff, rstar)
    ar = _derive_ar(period, mstar, rstar)
    inc = np.arccos(cosi)

    return {
        'teff': teff, 'rstar': rstar, 'feh': feh,
        'av': av, 'distance': distance,
        'tc': tc, 'period': period, 'p': p, 'cosi': cosi,
        'u1': u1, 'u2': u2, 'f0': f0,
        'K': K, 'gamma': gamma,
        'logg': logg, 'lstar': lstar, 'ar': ar,
        'inc_rad': inc, 'ideg': np.degrees(inc),
        'b': ar * cosi, 'delta': p**2,
        'parallax': 1000.0 / distance, 'mstar': mstar,
        'age': age, 'e': e, 'omega': omega,
        'param_names': list(param_names), 'use_mist': use_mist,
    }


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

    from exozippy.sed.utils import read_sed_file

    priors = parse_priors(priorfile)
    tran_data = read_transit_data(tranfile)
    rv_data = read_rv_data(rvfile)
    sed_data = read_sed_file(sedfile, 1) if sedfile is not None else None
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
            age_prior=age_prior,
            sed_data=sed_data)
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
            use_mist, mstar_prior, age_prior,
            sed_data=sed_data),
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
                    _log(f'  Iteration {self.count}: chi2 ~ {joint_negloglike(xk, tran_data, rv_data, sedfile, priors, e, omega, rv_jittervar, tran_addvar, use_mist, mstar_prior, age_prior, sed_data=sed_data):.2f}', verbose)
        callback = ProgressCallback()
    else:
        callback = None

    res = minimize(joint_negloglike, amoeba_sol,
                   args=(tran_data, rv_data, sedfile, priors,
                         e, omega, rv_jittervar, tran_addvar,
                         use_mist, mstar_prior, age_prior, sed_data),
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
             e=0.0, omega=np.pi/2, nchains=None, nsteps=2000,
             ntemps=1, verbose=True, use_mist=False, nthreads=None,
             checkpoint=None, checkpoint_every=0, resume=True):
    """Run DEMC-PT MCMC sampling around the best-fit joint solution."""
    from exozippy.sed.utils import read_sed_file

    priors = parse_priors(priorfile)
    tran_data = read_transit_data(tranfile)
    rv_data = read_rv_data(rvfile)
    sed_data = read_sed_file(sedfile, 1) if sedfile is not None else None
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

    # Default nchains = 2 * ndim (EXOFASTv2 convention), minimum 3
    if nchains is None:
        nchains = max(2 * ndim, 3)

    # Initialization scatter
    if use_mist:
        scales = np.concatenate(([0.02, 0.5], BASE_SCALES))
    else:
        scales = BASE_SCALES

    if verbose:
        print(f"=== DEMC-PT: {nchains} chains, {ntemps} temps, {nsteps} steps ===")

    log_posterior = functools.partial(
        _mcmc_log_posterior,
        tran_data=tran_data, rv_data=rv_data, sedfile=sedfile,
        priors=priors, sed_data=sed_data, e=e, omega=omega,
        rv_jittervar=rv_jittervar, tran_addvar=tran_addvar,
        use_mist=use_mist, mstar_fixed=mstar_prior,
        age_prior=age_prior,
    )

    sampler = None
    save_every = checkpoint_every if checkpoint else 0

    if checkpoint and resume and os.path.exists(checkpoint):
        sampler = DEMCPTSampler.load(checkpoint, log_posterior)
        if verbose:
            print(f"Resuming from checkpoint: {checkpoint}")
        if sampler._pos_full is None or sampler._logp_full is None:
            if verbose:
                print("Checkpoint lacks temperature state; starting fresh.")
            sampler = None
        elif sampler.nchains != nchains:
            if verbose:
                print(f"WARNING: nchains changed ({sampler.nchains} -> {nchains}); "
                      f"discarding checkpoint, starting fresh.")
            sampler = None
        elif sampler.ntemps != ntemps:
            if verbose:
                print(f"WARNING: ntemps changed ({sampler.ntemps} -> {ntemps}); "
                      f"discarding checkpoint, starting fresh.")
            sampler = None
        else:
            old_steps = sampler.chain.shape[0]
            if nsteps <= old_steps:
                if verbose:
                    print(f"Checkpoint already has {old_steps} steps; "
                          f"requested nsteps={nsteps}. Skipping extra run.")
            else:
                extra = nsteps - old_steps
                if verbose:
                    print(f"Continuing for +{extra} steps (to {nsteps}).")
                sampler._run_continue(nsteps=extra, scale=scales,
                                       progress=verbose,
                                       nworkers=nthreads or 1,
                                       save_every=save_every,
                                       save_file=checkpoint)

    if sampler is None:
        sampler = DEMCPTSampler(log_posterior, ndim=ndim,
                                nchains=nchains, ntemps=ntemps)
        sampler.run(x_best, nsteps=nsteps, scale=scales,
                    progress=verbose,
                    nworkers=nthreads or 1,
                    save_every=save_every,
                    save_file=checkpoint)
    samples = sampler.flatchain
    flatlog = sampler.flatlog_prob
    if samples is None or len(samples) == 0:
        # fallback: use full chain with manual burn-in
        chain = sampler.chain
        burn = max(chain.shape[0] // 4, 1)
        samples = chain[burn:].reshape(-1, ndim)
        flatlog = sampler.log_prob[burn:].reshape(-1)

    if samples.size == 0:
        raise RuntimeError("No MCMC samples generated.")

    # MCMC best-fit (MAP) from max log_prob
    bestfit_mcmc = None
    if flatlog is not None and len(flatlog) == len(samples):
        imax = int(np.nanargmax(flatlog))
        map_params = samples[imax]
        bestfit_mcmc = _bestfit_from_params(
            map_params, param_names, e, omega, mstar_prior, age_prior, use_mist
        )
        bestfit_mcmc['chi2'] = -2.0 * flatlog[imax]

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

    return samples, param_names, summary, bestfit_mcmc
