"""
Joint SED + Transit + RV fitting for EXOZIPPy.

Simultaneously fits stellar parameters (Teff, Rstar, [Fe/H], Av, distance)
and planetary parameters (tc, period, p, cosi, K) plus per-band limb darkening
(u1, u2), per-transit normalization (f0), and per-telescope gamma to
broadband photometry, transit light curves, and radial velocity data.

Supports multiple transit light curves and multiple RV telescopes.
"""
import glob as _glob
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
    param_scales as _param_scales,
    derive_logg as _derive_logg,
    derive_lstar as _derive_lstar,
    derive_ar as _derive_ar,
    BASE_PARAM_NAMES, BASE_SCALES,
)
from exozippy.mkss import mkss, _parse_priors, _read_data_with_detrend, _get_band_info
from exozippy.ss import SS

# Keep backward-compatible alias
joint_negloglike = joint_chi2


def _mcmc_log_posterior(theta, tran_data_list, rv_data_list, sedfile, priors,
                        sed_data, e, omega, rv_jittervar_list, tran_addvar_list,
                        use_mist, mstar_fixed, age_prior, nstars=1,
                        ntran=1, ntel=1, nbands=1,
                        circular=True, usevcve=False,
                        fitjittervar=False, fitslope=False, fitquad=False,
                        fitvariance=False,
                        fitdilute=False, fitttv=False, epoch_list=None,
                        rvepoch=0.0,
                        fitthermal=False, fitreflect=False,
                        fitbeam=False, fitellip=False,
                        rossiter=False, rmbandndx_list=None,
                        fitdt=False, fiterrscale=False,
                        dt_data_list=None, dtbandndx_list=None,
                        detrend_info=None):
    """Module-level log-posterior for picklability with multiprocessing."""
    chi2 = joint_chi2(theta, tran_data_list, rv_data_list, sedfile, priors,
                      sed_data=sed_data,
                      e=e, omega=omega,
                      rv_jittervar_list=rv_jittervar_list,
                      tran_addvar_list=tran_addvar_list,
                      use_mist=use_mist, mstar_fixed=mstar_fixed,
                      age_prior=age_prior, nstars=nstars,
                      ntran=ntran, ntel=ntel, nbands=nbands,
                      circular=circular, usevcve=usevcve,
                      fitjittervar=fitjittervar, fitslope=fitslope, fitquad=fitquad,
                      fitvariance=fitvariance,
                      fitdilute=fitdilute, fitttv=fitttv, epoch_list=epoch_list,
                      rvepoch=rvepoch,
                      fitthermal=fitthermal, fitreflect=fitreflect,
                      fitbeam=fitbeam, fitellip=fitellip,
                      rossiter=rossiter, rmbandndx_list=rmbandndx_list,
                      fitdt=fitdt, fiterrscale=fiterrscale,
                      dt_data_list=dt_data_list, dtbandndx_list=dtbandndx_list,
                      detrend_info=detrend_info)
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
    """Parse an EXOFASTv2-style prior file."""
    return _parse_priors(priorfile)


def _resolve_glob(pattern):
    """Expand a glob pattern to the first matching file, or return as-is."""
    if '*' in str(pattern) or '?' in str(pattern):
        matches = sorted(_glob.glob(str(pattern)))
        if matches:
            return matches[0]
    return str(pattern)


def read_transit_data(tranfile):
    """Read a transit light curve file (BJD flux err [detrend_cols...])."""
    tranfile = _resolve_glob(tranfile)
    bjd, flux, err, detrendadd, detrendmult = _read_data_with_detrend(tranfile)
    d = {'bjd': bjd, 'flux': flux, 'err': err}
    if detrendadd is not None:
        d['detrendadd'] = detrendadd
    if detrendmult is not None:
        d['detrendmult'] = detrendmult
    return d


def read_rv_data(rvfile):
    """Read an RV data file (BJD vel err_vel [detrend_cols...])."""
    rvfile = _resolve_glob(rvfile)
    bjd, vel, err, detrendadd, detrendmult = _read_data_with_detrend(rvfile)
    d = {'bjd': bjd, 'vel': vel, 'err': err}
    if detrendadd is not None:
        d['detrendadd'] = detrendadd
    if detrendmult is not None:
        d['detrendmult'] = detrendmult
    return d


def read_all_transit_data(tranpath):
    """Read all transit files matching a glob pattern.

    Returns (list[dict], list[str]) — data dicts and file paths.
    """
    files = sorted(_glob.glob(str(tranpath)))
    return [read_transit_data(f) for f in files], files


def read_all_rv_data(rvpath):
    """Read all RV files matching a glob pattern.

    Returns (list[dict], list[str]) — data dicts and file paths.
    """
    files = sorted(_glob.glob(str(rvpath)))
    return [read_rv_data(f) for f in files], files


def read_all_dt_data(dtpath):
    """Read all Doppler Tomography FITS files matching a glob pattern.

    Returns (list[dict], list[str]) — data dicts and file paths.
    """
    from exozippy.exozippy_dopptom import read_dt_fits
    files = sorted(_glob.glob(str(dtpath)))
    return [read_dt_fits(f) for f in files], files


def _build_detrend_info(tran_data_list, rv_data_list):
    """Build detrend_info dict from data lists. Returns None if no detrending."""
    tran_nadd = [td.get('detrendadd', np.empty((0, 0))).shape[0] if td.get('detrendadd') is not None else 0
                 for td in tran_data_list]
    tran_nmult = [td.get('detrendmult', np.empty((0, 0))).shape[0] if td.get('detrendmult') is not None else 0
                  for td in tran_data_list]
    rv_nadd = [rd.get('detrendadd', np.empty((0, 0))).shape[0] if rd.get('detrendadd') is not None else 0
               for rd in rv_data_list]
    rv_nmult = [rd.get('detrendmult', np.empty((0, 0))).shape[0] if rd.get('detrendmult') is not None else 0
                for rd in rv_data_list]
    if sum(tran_nadd) + sum(tran_nmult) + sum(rv_nadd) + sum(rv_nmult) == 0:
        return None
    return dict(tran_nadd=tran_nadd, tran_nmult=tran_nmult,
                rv_nadd=rv_nadd, rv_nmult=rv_nmult)


def build_initial_guess(priorfile, tranfile, rvfile, e=0.0,
                        omega=np.pi/2, circular=True, usevcve=False,
                        use_mist=False, nstars=1,
                        fitjittervar=False, fitvariance=False,
                        fitdilute=False, fitttv=False,
                        fitslope=False, fitquad=False,
                        fitthermal=False, fitreflect=False,
                        fitbeam=False, fitellip=False,
                        rossiter=False, rmbands=None,
                        dtpath=None, fitdt=False, fiterrscale=False):
    """
    Construct an SS object from priors (before optimization).
    """
    ss = mkss(
        parfile=priorfile,
        tranpath=tranfile,
        rvpath=rvfile,
        use_mist=use_mist,
        nstars=nstars,
        circular=circular,
        usevcve=usevcve,
        fitjittervar=fitjittervar, fitvariance=fitvariance,
        fitdilute=fitdilute, fitttv=fitttv,
        fitslope=fitslope, fitquad=fitquad,
        fitthermal=fitthermal, fitreflect=fitreflect,
        fitbeam=fitbeam, fitellip=fitellip,
        rossiter=rossiter, rmbands=rmbands,
    )
    ss.planet[0].e.value = e
    ss.planet[0].omega.value = omega
    # Initialize sesinw/secosw from e/omega
    sqrte = np.sqrt(e)
    ss.planet[0].sesinw.value = sqrte * np.sin(omega)
    ss.planet[0].secosw.value = sqrte * np.cos(omega)

    priors = _parse_priors(priorfile)
    if 'tc' not in priors:
        tran_data_list, _ = read_all_transit_data(tranfile)
        if tran_data_list:
            all_bjd = np.concatenate([td['bjd'] for td in tran_data_list])
            ss.planet[0].tc.value = float(np.median(all_bjd))

    if 'parallax' in priors:
        ss.star[0].distance.value = 1000.0 / priors['parallax']['value']

    ss.compute_derived()
    return ss


def _update_ss_from_params(ss, params, param_names, e=0.0, omega=np.pi/2):
    """Update an SS object from a flat parameter vector."""
    ss.from_vector(params, param_names)
    # Derive e/omega from eccentricity parameterization
    if 'vcve' in param_names:
        # vcve parameterization — compute_derived handles it via ss.py
        pass
    elif 'sesinw' in param_names:
        sesinw = ss['sesinw']
        secosw = ss['secosw']
        ss.planet[0].e.value = sesinw**2 + secosw**2
        ss.planet[0].omega.value = np.arctan2(sesinw, secosw)
    else:
        ss.planet[0].e.value = e
        ss.planet[0].omega.value = omega
    return ss


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
