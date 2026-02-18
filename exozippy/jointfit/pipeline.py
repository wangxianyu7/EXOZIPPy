"""Shared orchestration helpers for optimization and MCMC."""

import numpy as np

from exozippy.jointfit.chi2 import joint_chi2

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
