"""MCMC entry points for EXOZIPPy fitting."""

import functools
import os

import numpy as np

from exozippy.jointfit.chi2 import (
    derive_ar as _derive_ar,
    derive_logg as _derive_logg,
    derive_lstar as _derive_lstar,
    param_names as _param_names,
    param_scales as _param_scales,
)
from exozippy.jointfit.demcpt import DEMCPTSampler
from exozippy.jointfit.mkss import mkss, _get_band_info
from exozippy.jointfit.ss import SS

from .inputs import _build_detrend_info, parse_priors, read_all_dt_data, read_all_rv_data, read_all_transit_data
from .optimize import fit_exoplanet
from .pipeline import _mcmc_log_posterior, _update_ss_from_params

def run_mcmc(priorfile, tranfile, rvfile, sedfile, bestfit=None,
             e=0.0, omega=np.pi/2, circular=True, usevcve=False,
             nchains=None, nsteps=2000,
             ntemps=1, verbose=True, use_mist=False, nthreads=None,
             checkpoint=None, checkpoint_every=0, resume=True, nstars=1,
             fitjittervar=False, fitvariance=False,
             fitdilute=False, fitttv=False,
             fitslope=False, fitquad=False,
             fitthermal=False, fitreflect=False,
             fitbeam=False, fitellip=False,
             rossiter=False, rmbands=None,
             dtpath=None, fitdt=False, fiterrscale=False):
    """Run DEMC-PT MCMC sampling around the best-fit joint solution."""
    from exozippy.sed.utils import read_sed_file

    has_sed = sedfile is not None
    priors = parse_priors(priorfile)
    tran_data_list, tranfiles = read_all_transit_data(tranfile)
    rv_data_list, rvfiles = read_all_rv_data(rvfile)
    sed_data = read_sed_file(sedfile, nstars) if has_sed else None
    mstar_prior = priors.get('mstar', {}).get('value', 1.0)
    age_prior = priors.get('age', {}).get('value', 1.0)

    dt_data_list, _ = (read_all_dt_data(dtpath) if dtpath else ([], []))
    _fitdt = fitdt and bool(dt_data_list)
    ndt = len(dt_data_list)
    dtbandndx_list = [0] * ndt

    ntran = len(tran_data_list)
    ntel = len(rv_data_list)

    if tranfiles:
        _, tran_bandndx = _get_band_info(tranfiles)
        for td, bi in zip(tran_data_list, tran_bandndx):
            td['bandndx'] = bi
        nbands = len(set(tran_bandndx))
    else:
        nbands = 1

    # Compute detrend_info from data files
    detrend_info = _build_detrend_info(tran_data_list, rv_data_list)

    rv_jittervar_list = [priors.get(f'jittervar_{j}', {}).get('value', 0.0)
                         for j in range(ntel)]
    tran_addvar_list = [priors.get(f'variance_{j}', {}).get('value', 0.0)
                        for j in range(ntran)]

    # Compute epoch list for TTV
    _fitttv = fitttv and ntran >= 3
    epoch_list = None
    if _fitttv:
        tc_prior = priors.get('tc', {}).get('value', None)
        period_prior = priors.get('period_0', priors.get('period', {}))
        period_prior = period_prior.get('value', 3.0) if isinstance(period_prior, dict) else 3.0
        if tc_prior is not None and period_prior > 0:
            epoch_list = [int(round((float(np.median(td['bjd'])) - tc_prior) / period_prior))
                          for td in tran_data_list]
        else:
            _fitttv = False

    if bestfit is None:
        bestfit = fit_exoplanet(priorfile, tranfile, rvfile, sedfile,
                                e=e, omega=omega, circular=circular,
                                usevcve=usevcve, verbose=verbose,
                                use_mist=use_mist, nstars=nstars,
                                fitjittervar=fitjittervar, fitvariance=fitvariance,
                                fitdilute=fitdilute, fitttv=_fitttv,
                                fitslope=fitslope, fitquad=fitquad,
                                fitthermal=fitthermal, fitreflect=fitreflect,
                                fitbeam=fitbeam, fitellip=fitellip,
                                rossiter=rossiter, rmbands=rmbands,
                                dtpath=dtpath, fitdt=_fitdt,
                                fiterrscale=fiterrscale)

    # Get rvepoch from bestfit SS
    rvepoch = bestfit.rvepoch if isinstance(bestfit, SS) else 0.0

    # Build rmbandndx_list from bestfit SS
    rmbandndx_list = None
    if rossiter and isinstance(bestfit, SS):
        rmbandndx_list = [tel.rmbandndx for tel in bestfit.telescope]

    pc_kwargs = dict(fitjittervar=fitjittervar, fitvariance=fitvariance,
                     fitdilute=fitdilute, fitttv=_fitttv,
                     fitslope=fitslope, fitquad=fitquad,
                     fitthermal=fitthermal, fitreflect=fitreflect,
                     fitbeam=fitbeam, fitellip=fitellip,
                     rossiter=rossiter,
                     fitdt=_fitdt, fiterrscale=fiterrscale,
                     usevcve=usevcve)
    pnames = bestfit.param_names if isinstance(bestfit, SS) else (bestfit.get('param_names') or _param_names(use_mist, nstars=nstars, has_sed=has_sed, ntran=ntran, ntel=ntel, nbands=nbands, circular=circular, ndt=ndt, detrend_info=detrend_info, **pc_kwargs))
    name_to_idx = {name: i for i, name in enumerate(pnames)}
    x_best = np.array([bestfit[n] for n in pnames])
    ndim = len(x_best)

    if nchains is None:
        nchains = max(2 * ndim, 3)

    ar_init = bestfit['ar'] if isinstance(bestfit, SS) else bestfit.get('ar', 10.0)
    scales = _param_scales(use_mist, nstars=nstars, has_sed=has_sed,
                           ntran=ntran, ntel=ntel, nbands=nbands,
                           circular=circular, ar_init=ar_init,
                           ndt=ndt, detrend_info=detrend_info, **pc_kwargs)

    if verbose:
        print(f"=== DEMC-PT: {nchains} chains, {ntemps} temps, {nsteps} steps ===")

    log_posterior = functools.partial(
        _mcmc_log_posterior,
        tran_data_list=tran_data_list, rv_data_list=rv_data_list,
        sedfile=sedfile, priors=priors, sed_data=sed_data,
        e=e, omega=omega,
        rv_jittervar_list=rv_jittervar_list,
        tran_addvar_list=tran_addvar_list,
        use_mist=use_mist, mstar_fixed=mstar_prior,
        age_prior=age_prior, nstars=nstars,
        ntran=ntran, ntel=ntel, nbands=nbands,
        circular=circular, epoch_list=epoch_list,
        rvepoch=rvepoch, rmbandndx_list=rmbandndx_list,
        dt_data_list=dt_data_list, dtbandndx_list=dtbandndx_list,
        detrend_info=detrend_info,
        **pc_kwargs,
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
    # Raw chain: (nsteps, nchains, ndim) — for arviz trace/corner plots
    raw_chain = sampler.chain
    raw_log_prob = sampler.log_prob

    samples = sampler.flatchain
    flatlog = sampler.flatlog_prob
    if samples is None or len(samples) == 0:
        burn = max(raw_chain.shape[0] // 4, 1)
        samples = raw_chain[burn:].reshape(-1, ndim)
        flatlog = raw_log_prob[burn:].reshape(-1)

    if samples.size == 0:
        raise RuntimeError("No MCMC samples generated.")

    # MCMC best-fit (MAP) from max log_prob
    bestfit_mcmc = None
    if flatlog is not None and len(flatlog) == len(samples):
        imax = int(np.nanargmax(flatlog))
        map_params = samples[imax]

        bestfit_mcmc = mkss(
            parfile=priorfile,
            tranpath=tranfile,
            rvpath=rvfile,
            sedfile=sedfile,
            use_mist=use_mist,
            nstars=nstars,
            circular=circular,
            rmbands=rmbands,
            **pc_kwargs,
        )
        _update_ss_from_params(bestfit_mcmc, map_params, pnames, e, omega)
        bestfit_mcmc.chi2 = -2.0 * flatlog[imax]

    def _arr(name):
        return samples[:, name_to_idx[name]]

    # Stellar mass: stored as logmstar in param vector when use_mist, convert to linear
    if 'logmstar' in name_to_idx:
        mstar_arr = 10.0**_arr('logmstar')
    elif 'mstar' in name_to_idx:
        mstar_arr = _arr('mstar')
    else:
        mstar_arr = np.full(samples.shape[0], mstar_prior)
    rstar_arr = _arr('rstar')
    teff_arr = _arr('teff')
    # Period: stored as logP in param vector, convert to linear
    if 'logP' in name_to_idx:
        period_arr = 10.0**_arr('logP')
    elif 'period' in name_to_idx:
        period_arr = _arr('period')
    else:
        period_arr = np.full(samples.shape[0], bestfit['period'])
    cosi_arr = _arr('cosi')

    # Derived parameter arrays
    logg_s = _derive_logg(mstar_arr, rstar_arr)
    lstar_s = _derive_lstar(teff_arr, rstar_arr)
    ar_s = _derive_ar(period_arr, mstar_arr, rstar_arr)
    ideg_s = np.degrees(np.arccos(cosi_arr))
    b_s = ar_s * cosi_arr

    derived_list = [('logg', logg_s), ('lstar', lstar_s), ('ar', ar_s),
                    ('ideg', ideg_s), ('b', b_s)]
    if 'distance' in name_to_idx:
        distance_arr = _arr('distance')
        plx_s = 1000.0 / distance_arr
        derived_list.append(('parallax', plx_s))

    summary = {}
    for i, name in enumerate(pnames):
        med = np.median(samples[:, i])
        lo = np.percentile(samples[:, i], 15.87)
        hi = np.percentile(samples[:, i], 84.13)
        summary[name] = (med, med - lo, hi - med)
        if verbose:
            print(f"{name:10s} = {med:.4f}  -{med-lo:.4f}  +{hi-med:.4f}")

    for name, arr in derived_list:
        med = np.median(arr)
        lo = np.percentile(arr, 15.87)
        hi = np.percentile(arr, 84.13)
        summary[name] = (med, med - lo, hi - med)
        if verbose:
            print(f"{name:10s} = {med:.4f}  -{med-lo:.4f}  +{hi-med:.4f}")

    # Pack chain info for arviz
    chain_info = {
        'chain': raw_chain,           # (nsteps, nchains, ndim)
        'log_prob': raw_log_prob,     # (nsteps, nchains)
        'param_names': pnames,
        'nchains': sampler.nchains,
    }

    return samples, pnames, summary, bestfit_mcmc, chain_info
