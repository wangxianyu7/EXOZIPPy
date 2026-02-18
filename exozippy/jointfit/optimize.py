"""Deterministic optimization entry points for EXOZIPPy fitting."""

import os
from datetime import datetime

import numpy as np
from scipy.optimize import minimize

from exozippy.optim import amoeba
from exozippy.jointfit.chi2 import (
    joint_chi2,
    param_names as _param_names,
    param_scales as _param_scales,
)
from exozippy.jointfit.mkss import mkss, _get_band_info

from .inputs import (
    _build_detrend_info,
    build_initial_guess,
    parse_priors,
    read_all_dt_data,
    read_all_rv_data,
    read_all_transit_data,
)
from .pipeline import _log, _log_section, _update_ss_from_params

# Keep backward-compatible alias
joint_negloglike = joint_chi2

def fit_exoplanet(priorfile, tranfile, rvfile, sedfile, e=0.0,
                  omega=np.pi/2, circular=True, usevcve=False,
                  verbose=True, use_mist=False, nstars=1,
                  fitjittervar=False, fitvariance=False,
                  fitdilute=False, fitttv=False,
                  fitslope=False, fitquad=False,
                  fitthermal=False, fitreflect=False,
                  fitbeam=False, fitellip=False,
                  rossiter=False, rmbands=None,
                  dtpath=None, fitdt=False, fiterrscale=False,
                  rmbands_dt=None):
    """
    Joint fit of SED (+ optional MIST evolutionary prior) + Transit + RV data.

    Supports multiple transit light curves and multiple RV telescopes.

    Returns
    -------
    result : SS
        Best-fit stellar system with parameter values, chi2, derived quantities.
    """
    start_time = datetime.utcnow()
    _log_section('Joint SED + Transit + RV Fit', verbose)
    _log(f'Start time (UTC): {start_time:%Y-%m-%d %H:%M:%S}', verbose)
    _log(f'Prior file       : {priorfile}', verbose)
    _log(f'Transit file     : {tranfile}', verbose)
    _log(f'RV file          : {rvfile}', verbose)
    _log(f'SED file         : {sedfile or "(none)"}', verbose)
    _log(f'Orbital params   : e={e}, omega={omega:.4f}', verbose)
    _log(f'Use MIST         : {use_mist}', verbose)

    from exozippy.sed.utils import read_sed_file

    priors = parse_priors(priorfile)
    tran_data_list, tranfiles = read_all_transit_data(tranfile)
    rv_data_list, rvfiles = read_all_rv_data(rvfile)
    has_sed = sedfile is not None
    sed_data = read_sed_file(sedfile, nstars) if has_sed else None
    mstar_prior = priors.get('mstar', {}).get('value', 1.0)
    age_prior = priors.get('age', {}).get('value', 1.0)

    # Doppler Tomography data
    dt_data_list, dtfiles = (read_all_dt_data(dtpath) if dtpath else ([], []))
    _fitdt = fitdt and bool(dt_data_list)
    ndt = len(dt_data_list)
    # dtbandndx_list: per-DT-file band index for limb darkening (default all → 0)
    dtbandndx_list = [0] * ndt

    ntran = len(tran_data_list)
    ntel = len(rv_data_list)

    # Assign per-transit band index from filenames (EXOFASTv2 convention)
    if tranfiles:
        _, tran_bandndx = _get_band_info(tranfiles)
        for td, bi in zip(tran_data_list, tran_bandndx):
            td['bandndx'] = bi
        nbands = len(set(tran_bandndx))
    else:
        nbands = 1

    # Compute detrend_info from data files (auto-detected extra columns)
    detrend_info = _build_detrend_info(tran_data_list, rv_data_list)

    # Compute epoch list for TTV (needs tc/period from priors)
    _fitttv = fitttv and ntran >= 3
    epoch_list = None
    if _fitttv:
        tc_prior = priors.get('tc', {}).get('value', None)
        period_prior = priors.get('period_0', priors.get('period', {})).get('value', 3.0) if isinstance(priors.get('period_0', priors.get('period', {})), dict) else 3.0
        if tc_prior is not None and period_prior > 0:
            epoch_list = [int(round((float(np.median(td['bjd'])) - tc_prior) / period_prior))
                          for td in tran_data_list]
        else:
            _fitttv = False  # can't compute epochs without tc/period

    pnames = _param_names(use_mist, nstars=nstars, has_sed=has_sed,
                          ntran=ntran, ntel=ntel, nbands=nbands,
                          circular=circular, usevcve=usevcve,
                          fitjittervar=fitjittervar, fitvariance=fitvariance,
                          fitdilute=fitdilute, fitttv=_fitttv,
                          fitslope=fitslope, fitquad=fitquad,
                          fitthermal=fitthermal, fitreflect=fitreflect,
                          fitbeam=fitbeam, fitellip=fitellip,
                          rossiter=rossiter,
                          fitdt=_fitdt, ndt=ndt, fiterrscale=fiterrscale,
                          detrend_info=detrend_info)

    # Per-instrument jittervar/variance from priors
    rv_jittervar_list = [priors.get(f'jittervar_{j}', {}).get('value', 0.0)
                         for j in range(ntel)]
    tran_addvar_list = [priors.get(f'variance_{j}', {}).get('value', 0.0)
                        for j in range(ntran)]

    # Build SS for initial values
    ss = mkss(
        parfile=priorfile,
        tranpath=tranfile,
        rvpath=rvfile,
        sedfile=sedfile,
        use_mist=use_mist,
        nstars=nstars,
        circular=circular,
        usevcve=usevcve,
        fitdilute=fitdilute, fitttv=_fitttv,
        fitslope=fitslope, fitquad=fitquad,
        fitthermal=fitthermal, fitreflect=fitreflect,
        fitbeam=fitbeam, fitellip=fitellip,
        rossiter=rossiter, rmbands=rmbands,
        dtpath=dtpath, fitdt=_fitdt, fiterrscale=fiterrscale,
    )
    rvepoch = ss.rvepoch

    # Build rmbandndx_list for chi2_rv
    rmbandndx_list = None
    if rossiter:
        rmbandndx_list = [tel.rmbandndx for tel in ss.telescope]
    ss.planet[0].e.value = e
    ss.planet[0].omega.value = omega
    sqrte = np.sqrt(e)
    ss.planet[0].sesinw.value = sqrte * np.sin(omega)
    ss.planet[0].secosw.value = sqrte * np.cos(omega)

    if 'parallax' in priors:
        ss.star[0].distance.value = 1000.0 / priors['parallax']['value']
    if 'tc' not in priors and tran_data_list:
        all_bjd = np.concatenate([td['bjd'] for td in tran_data_list])
        ss.planet[0].tc.value = float(np.median(all_bjd))

    ss.compute_derived()
    x0 = ss.to_vector(pnames)

    if verbose:
        if use_mist:
            _log(f"Initial mstar/age: {ss['mstar']:.3f} Msun, {ss['age']:.2f} Gyr", verbose)
        else:
            _log(f"Fixed mstar = {mstar_prior:.3f} Msun", verbose)
        for j, tf in enumerate(tranfiles):
            _log(f"Transit {j}       : {os.path.basename(tf)} ({len(tran_data_list[j]['bjd'])} pts)", verbose)
        for j, rf in enumerate(rvfiles):
            _log(f"Telescope {j}     : {os.path.basename(rf)} ({len(rv_data_list[j]['bjd'])} pts)", verbose)
        _log("Starting parameters:", verbose)
        _log(f"  Teff={ss['teff']:.0f}K  Rstar={ss['rstar']:.3f}  [Fe/H]={ss['feh']:.3f}  Av={ss['av']:.3f}  dist={ss['distance']:.1f}", verbose)
        _log(f"  tc={ss['tc']:.5f}  P={ss['period']:.6f}  p={ss['p']:.4f}  cosi={ss['cosi']:.4f}", verbose)
        for _bi, _bnd in enumerate(ss.band):
            _log(f"  u1_{_bi}={ss[f'u1_{_bi}']:.2f}  u2_{_bi}={ss[f'u2_{_bi}']:.2f}  [{_bnd.name}]", verbose)
        for j in range(ntran):
            _log(f"  f0_{j}={ss[f'f0_{j}']:.5f}", verbose)
        _log(f"  K={ss['K']:.1f}", verbose)
        for j in range(ntel):
            _log(f"  gamma_{j}={ss[f'gamma_{j}']:.1f}", verbose)
        chi2_init = joint_chi2(
            x0, tran_data_list, rv_data_list, sedfile, priors,
            e, omega, rv_jittervar_list=rv_jittervar_list,
            tran_addvar_list=tran_addvar_list,
            use_mist=use_mist,
            mstar_fixed=mstar_prior,
            age_prior=age_prior,
            sed_data=sed_data, nstars=nstars,
            ntran=ntran, ntel=ntel, nbands=nbands,
            circular=circular, usevcve=usevcve,
            fitjittervar=fitjittervar, fitvariance=fitvariance,
            fitdilute=fitdilute, fitttv=_fitttv, epoch_list=epoch_list,
            fitslope=fitslope, fitquad=fitquad, rvepoch=rvepoch,
            fitthermal=fitthermal, fitreflect=fitreflect,
            fitbeam=fitbeam, fitellip=fitellip,
            rossiter=rossiter, rmbandndx_list=rmbandndx_list,
            fitdt=_fitdt, fiterrscale=fiterrscale,
            dt_data_list=dt_data_list, dtbandndx_list=dtbandndx_list,
            detrend_info=detrend_info)
        _log(f"Initial chi2     : {chi2_init:.2f}", verbose)

    # Av upper bound
    av_upper = priors.get('av', {}).get('upper', 1.0)
    if not np.isfinite(av_upper):
        av_upper = 1.0

    # Phase 1: Nelder-Mead
    ar_init = ss['ar'] if ss['ar'] > 0 else 10.0
    scale_vec = _param_scales(use_mist, nstars=nstars, has_sed=has_sed,
                              ntran=ntran, ntel=ntel, nbands=nbands,
                              circular=circular, usevcve=usevcve, ar_init=ar_init,
                              fitjittervar=fitjittervar, fitvariance=fitvariance,
                              fitdilute=fitdilute, fitttv=_fitttv,
                              fitslope=fitslope, fitquad=fitquad,
                              fitthermal=fitthermal, fitreflect=fitreflect,
                              fitbeam=fitbeam, fitellip=fitellip,
                              rossiter=rossiter,
                              fitdt=_fitdt, ndt=ndt, fiterrscale=fiterrscale,
                              detrend_info=detrend_info)

    def _chi2_func(params):
        return joint_chi2(
            params, tran_data_list, rv_data_list, sedfile, priors,
            e, omega, rv_jittervar_list=rv_jittervar_list,
            tran_addvar_list=tran_addvar_list,
            use_mist=use_mist, mstar_fixed=mstar_prior,
            age_prior=age_prior,
            sed_data=sed_data, nstars=nstars,
            ntran=ntran, ntel=ntel, nbands=nbands,
            circular=circular, usevcve=usevcve,
            fitjittervar=fitjittervar, fitvariance=fitvariance,
            fitdilute=fitdilute, fitttv=_fitttv, epoch_list=epoch_list,
            fitslope=fitslope, fitquad=fitquad, rvepoch=rvepoch,
            fitthermal=fitthermal, fitreflect=fitreflect,
            fitbeam=fitbeam, fitellip=fitellip,
            rossiter=rossiter, rmbandndx_list=rmbandndx_list,
            fitdt=_fitdt, fiterrscale=fiterrscale,
            dt_data_list=dt_data_list, dtbandndx_list=dtbandndx_list,
            detrend_info=detrend_info)

    if verbose:
        _log('Starting Amoeba (Nelder-Mead) search...', verbose)

    amoeba_sol, amoeba_chi2, amoeba_info = amoeba(
        _chi2_func, x0=x0, scale=scale_vec,
        ftol=1e-6, maxiter=50000, verbose=verbose)
    if verbose:
        if amoeba_info['success']:
            _log(f'Amoeba converged: chi2 ~ {amoeba_chi2:.2f}', verbose)
        else:
            _log('Amoeba reached max iterations; proceeding to L-BFGS-B.', verbose)

    # Phase 2: L-BFGS-B with bounds
    tc0 = ss['tc']
    period0 = ss['period']
    bounds = []
    # Per-star bounds
    for i in range(nstars):
        if use_mist:
            bounds.extend([
                (-1.0, 0.7),        # logmstar (0.1 to 5 Msun)
                (0.01, 14.0),       # age (Gyr)
            ])
        bounds.extend([
            (3000, 10000),          # teff
            (0.1, 10.0),            # rstar
            (-2.0, 0.75),           # feh
        ])
        if has_sed:
            bounds.extend([
                (0.0, av_upper),        # av
                (1.0, 10000.0),         # distance
            ])
    # Shared orbital bounds
    K0 = ss['K']
    logP0 = np.log10(period0)
    bounds.extend([
        (tc0 - 0.5, tc0 + 0.5),             # tc
        (logP0 - 0.005, logP0 + 0.005),     # logP (±1.2% in period)
        (-0.5, 0.5),                         # p (negative allowed per EXOFASTv2)
        (0.0, 0.99),                         # cosi
        (0.1, max(K0 * 5, 500.0)),           # K
    ])
    # Eccentricity bounds (when non-circular)
    if not circular:
        if usevcve:
            bounds.extend([
                (1e-6, 1.0),                        # vcve (0, 1]
                (-1.0, 1.0),                        # lsinw
                (-1.0, 1.0),                        # lcosw
                (-1.0, 2.0),                        # sign
            ])
        else:
            bounds.extend([
                (-1.0, 1.0),                        # sesinw
                (-1.0, 1.0),                        # secosw
            ])
    # Per-band LD bounds
    for j in range(nbands):
        bounds.extend([
            (0.0, 2.0),             # u1
            (-1.0, 1.0),            # u2
        ])
    # Per-band phase curve bounds (ppm)
    if fitthermal:
        for j in range(nbands):
            bounds.append((0.0, 5000.0))       # thermal >= 0
    if fitreflect:
        for j in range(nbands):
            bounds.append((0.0, 5000.0))       # reflect >= 0
    # Per-transit f0 bounds
    for j in range(ntran):
        bounds.append((0.5, 1.5))
    # Per-transit variance bounds (allow negative per EXOFASTv2)
    if fitvariance:
        for j in range(ntran):
            max_flux_err = max(td['err'].max() for td in tran_data_list) if tran_data_list else 0.01
            bounds.append((-max_flux_err**2, 10 * max_flux_err**2))
    # Per-transit dilution bounds
    if fitdilute:
        for j in range(ntran):
            bounds.append((-1.0, 1.0))
    # Per-transit TTV bounds: |ttv| < period/2
    if _fitttv:
        half_period = period0 / 2.0
        for j in range(ntran):
            bounds.append((-half_period, half_period))
    # Per-transit detrend bounds (unbounded per EXOFASTv2)
    if detrend_info is not None:
        for j in range(ntran):
            nadd = detrend_info['tran_nadd'][j] if j < len(detrend_info['tran_nadd']) else 0
            nmult = detrend_info['tran_nmult'][j] if j < len(detrend_info['tran_nmult']) else 0
            for _ in range(nadd + nmult):
                bounds.append((-1e4, 1e4))
    # Per-telescope gamma bounds
    for j in range(ntel):
        gamma_j = ss[f'gamma_{j}']
        bounds.append((gamma_j - 5000.0, gamma_j + 5000.0))
    # Per-telescope jittervar bounds (allow negative per EXOFASTv2)
    if fitjittervar:
        for j in range(ntel):
            max_rv_err = max(rd['err'].max() for rd in rv_data_list) if rv_data_list else 10.0
            bounds.append((-max_rv_err**2, 10 * max_rv_err**2))
    # Per-telescope detrend bounds (unbounded per EXOFASTv2)
    if detrend_info is not None:
        for j in range(ntel):
            nadd = detrend_info['rv_nadd'][j] if j < len(detrend_info['rv_nadd']) else 0
            nmult = detrend_info['rv_nmult'][j] if j < len(detrend_info['rv_nmult']) else 0
            for _ in range(nadd + nmult):
                bounds.append((-1e6, 1e6))
    # Global RV trend bounds
    if fitslope or fitquad:
        bounds.append((-1e5, 1e5))         # slope (m/s/day)
    if fitquad:
        bounds.append((-1e5, 1e5))         # quad (m/s/day^2)
    # Per-planet phase curve bounds (ppm)
    if fitbeam:
        bounds.append((-500.0, 500.0))         # beam can be positive or negative
    if fitellip:
        bounds.append((0.0, 500.0))            # ellipsoidal >= 0
    # Rossiter-McLaughlin bounds
    if rossiter:
        bounds.extend([
            (-1e4, 1e4),                           # svsinicoslam
            (-1e4, 1e4),                           # svsinisinlam
            (0.0, 1e5),                            # vgamma
            (0.0, 1e5),                            # vzeta
            (0.0, 1e5),                            # vxi
            (0.0, 1e5),                            # valpha
        ])
    # Doppler Tomography bounds
    if _fitdt:
        if not rossiter:
            bounds.extend([
                (-1e4, 1e4),                       # svsinicoslam
                (-1e4, 1e4),                       # svsinisinlam
            ])
        bounds.append((1.0, 1e5))                  # vline (m/s, must be > 0)
        if fiterrscale:
            for _ in range(ndt):
                bounds.append((0.01, 100.0))       # errscale_j

    if verbose:
        _log('Starting L-BFGS-B refinement...', verbose)
        class ProgressCallback:
            def __init__(self):
                self.count = 0
            def __call__(self, xk):
                self.count += 1
                if self.count % 20 == 0:
                    _log(f'  Iteration {self.count}: chi2 ~ {_chi2_func(xk):.2f}', verbose)
        callback = ProgressCallback()
    else:
        callback = None

    res = minimize(_chi2_func, amoeba_sol,
                   method='L-BFGS-B', bounds=bounds, callback=callback,
                   options={'maxiter': 10000, 'ftol': 1e-12})

    # Update SS from optimized parameters
    _update_ss_from_params(ss, res.x, pnames, e, omega)
    ss.param_names = pnames          # full list including detrend params
    ss.opt_params  = res.x.copy()   # full optimizer solution vector

    # Fit diagnostics
    nsed = len(sed_data['mag']) if sed_data is not None else 0
    ndata_tran = sum(len(td['bjd']) for td in tran_data_list)
    ndata_rv = sum(len(rd['bjd']) for rd in rv_data_list)
    ndata = ndata_tran + ndata_rv + nsed
    ndof = ndata - len(pnames)
    nfit = len(pnames)
    ss.chi2 = res.fun
    ss.ndata = ndata
    ss.ndof = ndof
    ss.chi2_red = res.fun / ndof if ndof > 0 else np.nan
    ss.bic = res.fun + nfit * np.log(ndata)
    ss.aic = res.fun + 2 * nfit
    ss.success = res.success
    ss.message = str(res.message)

    if verbose:
        _log_section('Optimizer Summary', verbose)
        _log(f"Mstar    = {ss['mstar']:.4f} Msun", verbose)
        if use_mist:
            _log(f"Age      = {ss['age']:.3f} Gyr", verbose)
        _log(f"Teff     = {ss['teff']:.1f} K", verbose)
        _log(f"Rstar    = {ss['rstar']:.4f} Rsun", verbose)
        _log(f"[Fe/H]   = {ss['feh']:.4f}", verbose)
        _log(f"Av       = {ss['av']:.4f}", verbose)
        _log(f"Distance = {ss['distance']:.2f} pc (plx = {ss['parallax']:.4f} mas)", verbose)
        _log(f"logg     = {ss['logg']:.4f}", verbose)
        _log(f"Lstar    = {ss['lstar']:.4f} Lsun", verbose)
        _log('---', verbose)
        _log(f"Tc       = {ss['tc']:.5f}", verbose)
        _log(f"Period   = {ss['period']:.6f} d", verbose)
        _log(f"Rp/Rs    = {ss['p']:.4f}", verbose)
        _log(f"cosi     = {ss['cosi']:.4f} (i = {ss['ideg']:.2f} deg)", verbose)
        _log(f"a/Rs     = {ss['ar']:.2f}", verbose)
        _log(f"b        = {ss['b']:.4f}", verbose)
        for j, _bnd in enumerate(ss.band):
            _log(f"u1_{j} [{_bnd.name:8s}] = {ss[f'u1_{j}']:.4f}", verbose)
            _log(f"u2_{j} [{_bnd.name:8s}] = {ss[f'u2_{j}']:.4f}", verbose)
        for j in range(ntran):
            _log(f"f0_{j}     = {ss[f'f0_{j}']:.5f}", verbose)
        _log('---', verbose)
        _log(f"K        = {ss['K']:.2f} m/s", verbose)
        if not circular:
            _log(f"e        = {ss['e']:.4f}", verbose)
            _log(f"omega    = {ss['omega']:.4f} rad ({np.degrees(ss['omega']):.2f} deg)", verbose)
            _log(f"sesinw   = {ss['sesinw']:.4f}", verbose)
            _log(f"secosw   = {ss['secosw']:.4f}", verbose)
        for j in range(ntel):
            _log(f"gamma_{j}  = {ss[f'gamma_{j}']:.2f} m/s", verbose)
        if fitslope or fitquad:
            _log(f"slope    = {ss['slope']:.4f} m/s/day", verbose)
        if fitquad:
            _log(f"quad     = {ss['quad']:.6f} m/s/day^2", verbose)
        if fitthermal or fitreflect or fitbeam or fitellip:
            _log('---', verbose)
            if fitthermal:
                for j in range(nbands):
                    _log(f"thermal_{j} = {ss[f'thermal_{j}']:.1f} ppm", verbose)
            if fitreflect:
                for j in range(nbands):
                    _log(f"reflect_{j} = {ss[f'reflect_{j}']:.1f} ppm", verbose)
            if fitbeam:
                _log(f"beam     = {ss['beam']:.1f} ppm", verbose)
            if fitellip:
                _log(f"ellip    = {ss['ellipsoidal']:.1f} ppm", verbose)
        _log('---', verbose)
        _log(f"Chi2     = {ss.chi2:.4f}", verbose)
        _log(f"Chi2/dof = {ss.chi2_red:.4f}", verbose)
        _log(f"NDATA    = {ndata} (tran={ndata_tran}, rv={ndata_rv}, sed={nsed})", verbose)
        _log(f"NFIT     = {nfit}", verbose)
        _log(f"BIC      = {ss.bic:.4f}", verbose)
        _log(f"AIC      = {ss.aic:.4f}", verbose)
        _log(f"Success  = {ss.success}", verbose)
        end_time = datetime.utcnow()
        elapsed = end_time - start_time
        _log_section('Run Complete', verbose)
        _log(f'End time (UTC): {end_time:%Y-%m-%d %H:%M:%S}', verbose)
        _log(f'Elapsed       : {elapsed}', verbose)

    return ss
