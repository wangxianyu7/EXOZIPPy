"""
Chi-squared computation for EXOZIPPy joint fitting.

Modular design following EXOFASTv2's exofast_chi2v2.pro:
  1. Parameter unpacking and bounds enforcement
  2. Derived quantities (logg, Lstar, a/R*, etc.)
  3. MIST evolutionary chi2 (per star)
  4. SED chi2 (multi-star blend)
  5. Transit chi2 (per light curve)
  6. RV chi2 (per telescope)
  7. Prior chi2

Each component returns a chi2 contribution (float) or np.inf for
invalid parameters, so the caller can accumulate:
    total_chi2 = bounds_chi2 + mist_chi2 + sed_chi2 + tran_chi2 + rv_chi2 + prior_chi2
"""

import numpy as np

from exozippy.sed.utils import mistmultised
from exozippy.physics.massradius_mist import massradius_mist
from exozippy.physics.exozippy_tran import exozippy_tran
from exozippy.physics.exozippy_rv import exozippy_rv
from exozippy.utils import exozippy_getphase
from exozippy.utils.mkconstants import mkconstants

CONSTANTS = mkconstants()

# ---------------------------------------------------------------------------
# Parameter ordering
# ---------------------------------------------------------------------------
BASE_STELLAR_PARAMS = ['teff', 'rstar', 'feh', 'av', 'distance']
NOSED_STELLAR_PARAMS = ['teff', 'rstar', 'feh']  # no av/distance when no SED
MIST_STELLAR_PARAMS = ['logmstar', 'age']
SHARED_ORBITAL_PARAMS = ['tc', 'logP', 'p', 'cosi', 'K']

# Per-stellar-parameter scales
_STELLAR_SCALES = np.array([30.0, 0.02, 0.05, 0.005, 1.0])
_NOSED_STELLAR_SCALES = np.array([30.0, 0.02, 0.05])
_MIST_SCALES = np.array([0.1, 0.5])   # logmstar, age
_ORBITAL_SCALES = np.array([0.0003, 0.01, 0.003, 0.01, 2.0])  # tc, logP, p, cosi (placeholder), K
_LD_SCALES = np.array([0.02, 0.02])   # per band: u1, u2
_F0_SCALE = 0.0002                     # per transit
_GAMMA_SCALE = 1.0                     # per telescope
_THERMAL_SCALE = 50.0                  # per band, ppm
_REFLECT_SCALE = 20.0                  # per band, ppm
_BEAM_SCALE = 5.0                      # per planet, ppm
_ELLIP_SCALE = 5.0                     # per planet, ppm
_ECC_SCALE = 0.1                       # sesinw, secosw
_VCVE_SCALE = 0.05                     # vcve
_LSINW_SCALE = 0.1                     # lsinw
_LCOSW_SCALE = 0.1                     # lcosw
_SIGN_SCALE = 0.1                      # sign (root selector)
_JITTERVAR_SCALE = 1.0                 # per telescope, m^2/s^2
_SLOPE_SCALE = 1.0                     # global, m/s/day
_QUAD_SCALE = 1.0                      # global, m/s/day^2
_VARIANCE_SCALE = 1e-8                 # per transit, flux^2
_DILUTE_SCALE = 0.01                   # per transit, dilution fraction
_TTV_SCALE = 0.02                      # per transit, days (~30 min)
_DETREND_TRAN_SCALE = 0.1              # per transit detrend coeff
_DETREND_RV_SCALE = 1.0                # per telescope detrend coeff
_SVSINI_SCALE = 100.0                  # svsinicoslam, svsinisinlam (m/s^0.5)
_VLINE_SCALE = 1000.0                  # vgamma, vzeta, vxi, valpha (m/s)

# Legacy single-instrument constants (for backward compat imports)
BASE_PLANET_PARAMS = [
    'tc', 'period', 'p', 'cosi',
    'u1', 'u2', 'f0',
    'K', 'gamma',
]
BASE_PARAM_NAMES = BASE_STELLAR_PARAMS + BASE_PLANET_PARAMS
BASE_SCALES = np.array([
    30.0, 0.02, 0.05, 0.005, 1.0,     # stellar
    0.0003, 0.00005, 0.003, 0.01,      # orbital
    0.02, 0.02, 0.0002,               # transit nuisance
    2.0, 1.0,                          # RV
])

INF_CHI2 = 1e10  # sentinel for impossible parameters


def param_names(use_mist: bool, nstars: int = 1, has_sed: bool = True,
                ntran: int = 1, ntel: int = 1, nbands: int = 1,
                circular: bool = True, usevcve: bool = False,
                fitjittervar: bool = False, fitslope: bool = False,
                fitquad: bool = False, fitvariance: bool = False,
                fitdilute: bool = False, fitttv: bool = False,
                fitthermal: bool = False, fitreflect: bool = False,
                fitbeam: bool = False, fitellip: bool = False,
                rossiter: bool = False,
                fitdt: bool = False, ndt: int = 0, fiterrscale: bool = False,
                detrend_info: dict = None):
    """Return the ordered parameter name list.

    Layout: [stellar...] [shared orbital] [sesinw secosw? | vcve lsinw lcosw sign?]
            [per-band LD]
            [per-band thermal?] [per-band reflect?]
            [per-transit f0] [per-transit variance?] [per-transit dilute?]
            [per-transit ttv?] [per-transit detrend?]
            [per-telescope gamma] [per-telescope jittervar?]
            [per-telescope detrend?]
            [slope?] [quad?] [beam?] [ellipsoidal?]
            [RM: svsinicoslam svsinisinlam vgamma vzeta vxi valpha?]
            [DT: svsinicoslam? svsinisinlam? vline errscale_j?]
    """
    stellar = BASE_STELLAR_PARAMS if has_sed else NOSED_STELLAR_PARAMS
    names = []
    # Stellar params (per star)
    for i in range(nstars):
        suffix = f'_{i}' if nstars > 1 else ''
        if use_mist:
            names.extend([f'{p}{suffix}' for p in MIST_STELLAR_PARAMS])
        names.extend([f'{p}{suffix}' for p in stellar])
    # Shared orbital
    names.extend(SHARED_ORBITAL_PARAMS)
    # Eccentricity (when not circular)
    if not circular:
        if usevcve:
            names.extend(['vcve', 'lsinw', 'lcosw', 'sign'])
        else:
            names.extend(['sesinw', 'secosw'])
    # Per-band limb darkening
    for j in range(nbands):
        names.extend([f'u1_{j}', f'u2_{j}'])
    # Per-band phase curve params
    if fitthermal:
        for j in range(nbands):
            names.append(f'thermal_{j}')
    if fitreflect:
        for j in range(nbands):
            names.append(f'reflect_{j}')
    # Per-transit normalization
    for j in range(ntran):
        names.append(f'f0_{j}')
    # Per-transit variance (jitter)
    if fitvariance:
        for j in range(ntran):
            names.append(f'variance_{j}')
    # Per-transit dilution
    if fitdilute:
        for j in range(ntran):
            names.append(f'dilute_{j}')
    # Per-transit TTV
    if fitttv:
        for j in range(ntran):
            names.append(f'ttv_{j}')
    # Per-transit detrend coefficients
    if detrend_info is not None:
        for j in range(ntran):
            for k in range(detrend_info.get('tran_nadd', [0] * ntran)[j] if j < len(detrend_info.get('tran_nadd', [])) else 0):
                names.append(f'C{k}_{j}')
            for k in range(detrend_info.get('tran_nmult', [0] * ntran)[j] if j < len(detrend_info.get('tran_nmult', [])) else 0):
                names.append(f'M{k}_{j}')
    # Per-telescope gamma
    for j in range(ntel):
        names.append(f'gamma_{j}')
    # Per-telescope jittervar
    if fitjittervar:
        for j in range(ntel):
            names.append(f'jittervar_{j}')
    # Per-telescope detrend coefficients
    if detrend_info is not None:
        for j in range(ntel):
            for k in range(detrend_info.get('rv_nadd', [0] * ntel)[j] if j < len(detrend_info.get('rv_nadd', [])) else 0):
                names.append(f'RVC{k}_{j}')
            for k in range(detrend_info.get('rv_nmult', [0] * ntel)[j] if j < len(detrend_info.get('rv_nmult', [])) else 0):
                names.append(f'RVM{k}_{j}')
    # Global RV trend (fitquad implies fitslope)
    if fitslope or fitquad:
        names.append('slope')
    if fitquad:
        names.append('quad')
    # Per-planet phase curve params
    if fitbeam:
        names.append('beam')
    if fitellip:
        names.append('ellipsoidal')
    if rossiter:
        names.extend(['svsinicoslam', 'svsinisinlam',
                       'vgamma', 'vzeta', 'vxi', 'valpha'])
    if fitdt:
        # svsinicoslam/svsinisinlam shared with RM if rossiter=True
        if not rossiter:
            names.extend(['svsinicoslam', 'svsinisinlam'])
        names.append('vline')
        if fiterrscale:
            for j in range(ndt):
                names.append(f'errscale_{j}')
    return names


def param_scales(use_mist: bool, nstars: int = 1, has_sed: bool = True,
                 ntran: int = 1, ntel: int = 1, nbands: int = 1,
                 circular: bool = True, usevcve: bool = False,
                 ar_init: float = None,
                 fitjittervar: bool = False, fitslope: bool = False,
                 fitquad: bool = False, fitvariance: bool = False,
                 fitdilute: bool = False, fitttv: bool = False,
                 fitthermal: bool = False, fitreflect: bool = False,
                 fitbeam: bool = False, fitellip: bool = False,
                 rossiter: bool = False,
                 fitdt: bool = False, ndt: int = 0, fiterrscale: bool = False,
                 detrend_info: dict = None):
    """Return scales matching param_names ordering.

    Parameters
    ----------
    ar_init : float, optional
        Initial a/R* value. If provided, the cosi scale is set to 1/ar_init
        (EXOFASTv2 convention: transit probability ~ 1/ar).
    """
    stellar_scales = _STELLAR_SCALES if has_sed else _NOSED_STELLAR_SCALES
    scales = []
    for _ in range(nstars):
        if use_mist:
            scales.append(_MIST_SCALES)
        scales.append(stellar_scales)
    orbital = _ORBITAL_SCALES.copy()
    if ar_init is not None and ar_init > 0:
        orbital[3] = 1.0 / ar_init  # cosi scale = 1/ar
    scales.append(orbital)
    if not circular:
        if usevcve:
            scales.append(np.array([_VCVE_SCALE, _LSINW_SCALE, _LCOSW_SCALE, _SIGN_SCALE]))
        else:
            scales.append(np.array([_ECC_SCALE, _ECC_SCALE]))
    for _ in range(nbands):
        scales.append(_LD_SCALES)
    if fitthermal:
        scales.append(np.full(nbands, _THERMAL_SCALE))
    if fitreflect:
        scales.append(np.full(nbands, _REFLECT_SCALE))
    scales.append(np.full(ntran, _F0_SCALE))
    if fitvariance:
        scales.append(np.full(ntran, _VARIANCE_SCALE))
    if fitdilute:
        scales.append(np.full(ntran, _DILUTE_SCALE))
    if fitttv:
        scales.append(np.full(ntran, _TTV_SCALE))
    # Per-transit detrend scales
    if detrend_info is not None:
        for j in range(ntran):
            nadd = detrend_info.get('tran_nadd', [0] * ntran)[j] if j < len(detrend_info.get('tran_nadd', [])) else 0
            nmult = detrend_info.get('tran_nmult', [0] * ntran)[j] if j < len(detrend_info.get('tran_nmult', [])) else 0
            if nadd > 0:
                scales.append(np.full(nadd, _DETREND_TRAN_SCALE))
            if nmult > 0:
                scales.append(np.full(nmult, _DETREND_TRAN_SCALE))
    scales.append(np.full(ntel, _GAMMA_SCALE))
    if fitjittervar:
        scales.append(np.full(ntel, _JITTERVAR_SCALE))
    # Per-telescope detrend scales
    if detrend_info is not None:
        for j in range(ntel):
            nadd = detrend_info.get('rv_nadd', [0] * ntel)[j] if j < len(detrend_info.get('rv_nadd', [])) else 0
            nmult = detrend_info.get('rv_nmult', [0] * ntel)[j] if j < len(detrend_info.get('rv_nmult', [])) else 0
            if nadd > 0:
                scales.append(np.full(nadd, _DETREND_RV_SCALE))
            if nmult > 0:
                scales.append(np.full(nmult, _DETREND_RV_SCALE))
    if fitslope or fitquad:
        scales.append(np.array([_SLOPE_SCALE]))
    if fitquad:
        scales.append(np.array([_QUAD_SCALE]))
    if fitbeam:
        scales.append(np.array([_BEAM_SCALE]))
    if fitellip:
        scales.append(np.array([_ELLIP_SCALE]))
    if rossiter:
        scales.append(np.array([_SVSINI_SCALE, _SVSINI_SCALE,
                                 _VLINE_SCALE, _VLINE_SCALE, _VLINE_SCALE, _VLINE_SCALE]))
    if fitdt:
        if not rossiter:
            scales.append(np.array([_SVSINI_SCALE, _SVSINI_SCALE]))
        scales.append(np.array([_VLINE_SCALE]))
        if fiterrscale:
            scales.append(np.full(ndt, 0.1))    # errscale scale ~ 0.1
    return np.concatenate(scales)


# ---------------------------------------------------------------------------
# 1. Parameter unpacking
# ---------------------------------------------------------------------------
def unpack_params(params, use_mist=False, mstar_fixed=None, age_prior=None,
                  priors=None, nstars=1, has_sed=True,
                  ntran=1, ntel=1, nbands=1,
                  circular=True, usevcve=False,
                  fitjittervar=False, fitslope=False, fitquad=False,
                  fitvariance=False,
                  fitdilute=False, fitttv=False, epoch_list=None,
                  rvepoch=0.0,
                  fitthermal=False, fitreflect=False,
                  fitbeam=False, fitellip=False,
                  rossiter=False,
                  fitdt=False, ndt=0, fiterrscale=False,
                  detrend_info=None):
    """
    Unpack the flat parameter vector into a named dict.

    Returns
    -------
    d : dict  or  None (if out-of-bounds)
    """
    idx = 0

    # Stellar parameters (per star)
    mstar_arr = np.empty(nstars)
    age_arr = np.empty(nstars)
    teff_arr = np.empty(nstars)
    rstar_arr = np.empty(nstars)
    feh_arr = np.empty(nstars)
    av_arr = np.empty(nstars)
    distance_arr = np.empty(nstars)

    for i in range(nstars):
        if use_mist:
            logmstar_i = params[idx]; idx += 1
            mstar_arr[i] = 10.0**logmstar_i
            age_arr[i] = params[idx]; idx += 1
            if age_arr[i] <= 0 or age_arr[i] > 14.5:
                return None
        else:
            mstar_arr[i] = (mstar_fixed if mstar_fixed is not None
                           else (priors or {}).get('mstar', {}).get('value', 1.0))
            age_arr[i] = (age_prior if age_prior is not None
                         else (priors or {}).get('age', {}).get('value', 1.0))

        teff_arr[i] = params[idx]; idx += 1
        rstar_arr[i] = params[idx]; idx += 1
        feh_arr[i] = params[idx]; idx += 1
        if has_sed:
            av_arr[i] = params[idx]; idx += 1
            distance_arr[i] = params[idx]; idx += 1
        else:
            av_arr[i] = 0.0
            distance_arr[i] = (priors or {}).get('distance', {}).get('value', 10.0)

    # Shared orbital params
    tc, logP, p, cosi, K = params[idx:idx+5]; idx += 5
    period = 10.0**logP

    # Eccentricity
    vcve_val = 1.0
    lsinw_val = 0.0
    lcosw_val = 1.0
    sign_val = 0.0
    if not circular and usevcve:
        from .utils import vcve2e as _vcve2e
        vcve_val = params[idx]; idx += 1
        lsinw_val = params[idx]; idx += 1
        lcosw_val = params[idx]; idx += 1
        sign_val = params[idx]; idx += 1
        omega = np.arctan2(lsinw_val, lcosw_val)
        e = _vcve2e(vcve_val, lsinw=lsinw_val, lcosw=lcosw_val, sign=sign_val)
        sqrte = np.sqrt(max(e, 0.0))
        sesinw = sqrte * np.sin(omega)
        secosw = sqrte * np.cos(omega)
    elif not circular:
        sesinw = params[idx]; idx += 1
        secosw = params[idx]; idx += 1
        e = sesinw**2 + secosw**2
        omega = np.arctan2(sesinw, secosw)
    else:
        sesinw = 0.0
        secosw = 0.0
        e = 0.0
        omega = np.pi / 2

    # Per-band limb darkening
    u1_list = []
    u2_list = []
    for j in range(nbands):
        u1_list.append(params[idx]); idx += 1
        u2_list.append(params[idx]); idx += 1

    # Per-band phase curve params
    thermal_list = []
    if fitthermal:
        for j in range(nbands):
            thermal_list.append(params[idx]); idx += 1
    reflect_list = []
    if fitreflect:
        for j in range(nbands):
            reflect_list.append(params[idx]); idx += 1

    # Per-transit f0
    f0_list = []
    for j in range(ntran):
        f0_list.append(params[idx]); idx += 1

    # Per-transit variance (jitter)
    variance_list = []
    if fitvariance:
        for j in range(ntran):
            variance_list.append(params[idx]); idx += 1

    # Per-transit dilution
    dilute_list = []
    if fitdilute:
        for j in range(ntran):
            dilute_list.append(params[idx]); idx += 1

    # Per-transit TTV
    ttv_list = []
    if fitttv:
        for j in range(ntran):
            ttv_list.append(params[idx]); idx += 1
        # Linear ephemeris override (EXOFASTv2 convention):
        # transit_time_j = tc + epoch_j * period + ttv_j
        # Fit a line → derive tc, period from the TTVs
        if epoch_list is not None and len(epoch_list) == ntran and ntran >= 3:
            epochs = np.array(epoch_list, dtype=float)
            transit_times = np.array([tc + epochs[j] * period + ttv_list[j]
                                      for j in range(ntran)])
            coeffs = np.polyfit(epochs, transit_times, 1)
            period = coeffs[0]   # slope
            tc = coeffs[1]       # intercept
            logP = np.log10(period) if period > 0 else logP

    # Per-transit detrend coefficients
    tran_detrendadd = []   # list of lists
    tran_detrendmult = []
    if detrend_info is not None:
        for j in range(ntran):
            nadd = detrend_info.get('tran_nadd', [0] * ntran)[j] if j < len(detrend_info.get('tran_nadd', [])) else 0
            coeffs_add = []
            for _ in range(nadd):
                coeffs_add.append(params[idx]); idx += 1
            tran_detrendadd.append(coeffs_add)
            nmult = detrend_info.get('tran_nmult', [0] * ntran)[j] if j < len(detrend_info.get('tran_nmult', [])) else 0
            coeffs_mult = []
            for _ in range(nmult):
                coeffs_mult.append(params[idx]); idx += 1
            tran_detrendmult.append(coeffs_mult)

    # Per-telescope gamma
    gamma_list = []
    for j in range(ntel):
        gamma_list.append(params[idx]); idx += 1

    # Per-telescope jittervar
    jittervar_list = []
    if fitjittervar:
        for j in range(ntel):
            jittervar_list.append(params[idx]); idx += 1

    # Per-telescope detrend coefficients
    rv_detrendadd = []
    rv_detrendmult = []
    if detrend_info is not None:
        for j in range(ntel):
            nadd = detrend_info.get('rv_nadd', [0] * ntel)[j] if j < len(detrend_info.get('rv_nadd', [])) else 0
            coeffs_add = []
            for _ in range(nadd):
                coeffs_add.append(params[idx]); idx += 1
            rv_detrendadd.append(coeffs_add)
            nmult = detrend_info.get('rv_nmult', [0] * ntel)[j] if j < len(detrend_info.get('rv_nmult', [])) else 0
            coeffs_mult = []
            for _ in range(nmult):
                coeffs_mult.append(params[idx]); idx += 1
            rv_detrendmult.append(coeffs_mult)

    # Global RV trend
    slope_val = 0.0
    quad_val = 0.0
    if fitslope or fitquad:
        slope_val = params[idx]; idx += 1
    if fitquad:
        quad_val = params[idx]; idx += 1

    # Per-planet phase curve params
    beam_val = params[idx] if fitbeam else 0.0
    if fitbeam:
        idx += 1
    ellip_val = params[idx] if fitellip else 0.0
    if fitellip:
        idx += 1

    # Rossiter-McLaughlin params
    svsinicoslam_val = 0.0
    svsinisinlam_val = 0.0
    vgamma_val = 1000.0
    vzeta_val = 4000.0
    vxi_val = 1000.0
    valpha_val = 0.0
    vsini_val = 0.0
    lam_val = 0.0
    if rossiter:
        svsinicoslam_val = params[idx]; idx += 1
        svsinisinlam_val = params[idx]; idx += 1
        vgamma_val = params[idx]; idx += 1
        vzeta_val = params[idx]; idx += 1
        vxi_val = params[idx]; idx += 1
        valpha_val = params[idx]; idx += 1
        vsini_val = svsinicoslam_val**2 + svsinisinlam_val**2
        lam_val = np.arctan2(svsinisinlam_val, svsinicoslam_val)

    # Doppler Tomography params
    vline_val = 5000.0   # default intrinsic line broadening (m/s)
    errscale_list = []
    if fitdt:
        if not rossiter:
            svsinicoslam_val = params[idx]; idx += 1
            svsinisinlam_val = params[idx]; idx += 1
            vsini_val = svsinicoslam_val**2 + svsinisinlam_val**2
            lam_val   = np.arctan2(svsinisinlam_val, svsinicoslam_val)
        vline_val = params[idx]; idx += 1
        if fiterrscale:
            for j in range(ndt):
                errscale_list.append(params[idx]); idx += 1

    d = dict(
        mstar=mstar_arr, age=age_arr,
        teff=teff_arr, rstar=rstar_arr, feh=feh_arr,
        av=av_arr, distance=distance_arr,
        tc=tc, logP=logP, period=period, p=p, cosi=cosi, K=K,
        e=e, omega=omega, sesinw=sesinw, secosw=secosw,
        vcve=vcve_val, lsinw=lsinw_val, lcosw=lcosw_val, sign=sign_val,
        circular=circular, usevcve=usevcve,
        u1=u1_list, u2=u2_list,
        thermal=thermal_list, reflect=reflect_list,
        beam=beam_val, ellipsoidal=ellip_val,
        f0=f0_list, gamma=gamma_list,
        variance=variance_list, dilute=dilute_list, jittervar=jittervar_list,
        slope=slope_val, quad=quad_val, rvepoch=rvepoch,
        fitslope=fitslope, fitquad=fitquad,
        ttv=ttv_list, fitttv=fitttv, epoch_list=epoch_list,
        nstars=nstars, ntran=ntran, ntel=ntel, nbands=nbands,
        fitjittervar=fitjittervar, fitvariance=fitvariance, fitdilute=fitdilute,
        fitthermal=fitthermal, fitreflect=fitreflect,
        fitbeam=fitbeam, fitellip=fitellip,
        tran_detrendadd=tran_detrendadd, tran_detrendmult=tran_detrendmult,
        rv_detrendadd=rv_detrendadd, rv_detrendmult=rv_detrendmult,
        rossiter=rossiter,
        svsinicoslam=svsinicoslam_val, svsinisinlam=svsinisinlam_val,
        vsini=vsini_val, lam=lam_val,
        vgamma=vgamma_val, vzeta=vzeta_val, vxi=vxi_val, valpha=valpha_val,
        fitdt=fitdt, fiterrscale=fiterrscale,
        vline=vline_val, errscale=errscale_list,
    )
    return d


# ---------------------------------------------------------------------------
# 2. Bounds enforcement
# ---------------------------------------------------------------------------
def check_bounds(d):
    """
    Return INF_CHI2 if any parameter is out of physical bounds, else 0.0.
    """
    nstars = d.get('nstars', 1)
    for i in range(nstars):
        mstar = d['mstar'][i] if isinstance(d['mstar'], np.ndarray) else d['mstar']
        teff = d['teff'][i] if isinstance(d['teff'], np.ndarray) else d['teff']
        rstar = d['rstar'][i] if isinstance(d['rstar'], np.ndarray) else d['rstar']
        av = d['av'][i] if isinstance(d['av'], np.ndarray) else d['av']
        dist = d['distance'][i] if isinstance(d['distance'], np.ndarray) else d['distance']

        if mstar <= 0:
            return INF_CHI2
        if teff < 2500 or teff > 50000:
            return INF_CHI2
        if rstar < 0.05 or rstar > 100:
            return INF_CHI2
        if av < 0:
            return INF_CHI2
        if dist < 1:
            return INF_CHI2

    if d['period'] <= 0 or d['logP'] < -1 or d['logP'] > 13:
        return INF_CHI2
    if d['e'] < 0 or d['e'] >= 1:
        return INF_CHI2
    if d.get('usevcve') and (d['vcve'] <= 0 or d['vcve'] > 1):
        return INF_CHI2
    if abs(d['p']) >= 0.5:
        return INF_CHI2
    if d['cosi'] < 0 or d['cosi'] >= 1:
        return INF_CHI2

    # Per-transit bounds
    for f0 in d['f0']:
        if f0 <= 0:
            return INF_CHI2

    # Global RV trend bounds
    if d.get('fitslope') or d.get('fitquad'):
        if abs(d.get('slope', 0.0)) > 1e5:
            return INF_CHI2
    if d.get('fitquad'):
        if abs(d.get('quad', 0.0)) > 1e5:
            return INF_CHI2

    # Per-transit dilution bounds: (-1, 1)
    if d.get('fitdilute') and d.get('dilute'):
        for dilute_j in d['dilute']:
            if dilute_j <= -1.0 or dilute_j >= 1.0:
                return INF_CHI2

    # TTV bounds: |ttv_j| < period/2
    if d.get('fitttv') and d.get('ttv'):
        half_period = d['period'] / 2.0
        for ttv_j in d['ttv']:
            if abs(ttv_j) >= half_period:
                return INF_CHI2

    # RM bounds
    if d.get('rossiter'):
        if d['vsini'] < 0:
            return INF_CHI2
        if d['vgamma'] < 0 or d['vzeta'] < 0 or d['vxi'] < 0 or d['valpha'] < 0:
            return INF_CHI2

    # DT bounds
    if d.get('fitdt'):
        if d.get('vsini', 1.0) < 0:
            return INF_CHI2
        if d.get('vline', 1.0) <= 0:
            return INF_CHI2
        if d.get('fiterrscale') and d.get('errscale'):
            for es in d['errscale']:
                if es <= 0:
                    return INF_CHI2

    # Per-band limb darkening (Kipping 2013)
    for j in range(len(d['u1'])):
        u1 = d['u1'][j]
        u2 = d['u2'][j]
        if u1 < 0:
            return INF_CHI2
        if u1 + u2 > 1:
            return INF_CHI2
        if u1 + 2 * u2 < 0:
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


def compute_derived(d, e=None, omega=None):
    """
    Add derived quantities to *d* in-place:
    logg, lstar, ar, inc, tp.

    If e/omega are None, uses d['e'] and d['omega'] (from sesinw/secosw).
    """
    if e is None:
        e = d.get('e', 0.0)
    if omega is None:
        omega = d.get('omega', np.pi / 2)
    d['logg']  = derive_logg(d['mstar'], d['rstar'])
    d['lstar'] = derive_lstar(d['teff'], d['rstar'])
    mstar0 = d['mstar'][0] if isinstance(d['mstar'], np.ndarray) else d['mstar']
    rstar0 = d['rstar'][0] if isinstance(d['rstar'], np.ndarray) else d['rstar']
    d['ar']    = derive_ar(d['period'], mstar0, rstar0)
    d['inc']   = np.arccos(d['cosi'])
    d['tp']    = tc_to_tp(d['tc'], d['period'], e, omega)


# ---------------------------------------------------------------------------
# 4. MIST evolutionary chi2
# ---------------------------------------------------------------------------
def chi2_mist(d, nstars=1):
    """MIST isochrone chi2 penalty — summed over all stars."""
    total = 0.0
    for i in range(nstars):
        mstar_i = d['mstar'][i] if isinstance(d['mstar'], np.ndarray) else d['mstar']
        feh_i = d['feh'][i] if isinstance(d['feh'], np.ndarray) else d['feh']
        age_i = d['age'][i] if isinstance(d['age'], np.ndarray) else d['age']
        teff_i = d['teff'][i] if isinstance(d['teff'], np.ndarray) else d['teff']
        rstar_i = d['rstar'][i] if isinstance(d['rstar'], np.ndarray) else d['rstar']
        try:
            val = massradius_mist(mstar_i, feh_i, age_i, teff_i, rstar_i)
        except Exception:
            return INF_CHI2
        if not np.isfinite(val):
            return INF_CHI2
        total += val
    return total


# ---------------------------------------------------------------------------
# 5. SED chi2
# ---------------------------------------------------------------------------
def chi2_sed(d, sedfile, sed_data=None, nstars=1):
    """Broadband SED chi2 using MIST bolometric corrections."""
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
# 6. Transit chi2 (loops over all light curves)
# ---------------------------------------------------------------------------
def chi2_transit(d, tran_data_list, e, omega, tran_addvar_list):
    """
    Transit light-curve chi2, summed over all light curves.

    Parameters
    ----------
    d : dict
    tran_data_list : list[dict]
        Each dict has keys: bjd, flux, err, bandndx (optional, default 0).
    tran_addvar_list : list[float]
        Per-transit added variance.
    """
    total = 0.0
    fitthermal = d.get('fitthermal', False)
    fitreflect = d.get('fitreflect', False)
    fitbeam = d.get('fitbeam', False)
    fitellip = d.get('fitellip', False)
    fitdilute = d.get('fitdilute', False)
    dilute_list = d.get('dilute', [])
    fitttv = d.get('fitttv', False)
    ttv_list = d.get('ttv', [])
    epoch_list = d.get('epoch_list') or []
    for j, tdata in enumerate(tran_data_list):
        bandndx = tdata.get('bandndx', 0)
        u1_j = d['u1'][bandndx]
        u2_j = d['u2'][bandndx]
        f0_j = d['f0'][j]
        # Use fitted variance if available, otherwise use fixed list
        if d.get('fitvariance') and d.get('variance'):
            addvar_j = d['variance'][j]
        else:
            addvar_j = tran_addvar_list[j] if j < len(tran_addvar_list) else 0.0
        # Phase curve params
        thermal_j = d['thermal'][bandndx] if fitthermal and d['thermal'] else 0.0
        reflect_j = d['reflect'][bandndx] if fitreflect and d['reflect'] else 0.0
        beam_j = d['beam'] if fitbeam else 0.0
        ellip_j = d['ellipsoidal'] if fitellip else 0.0
        dilute_j = dilute_list[j] if fitdilute and dilute_list else 0.0
        # Per-transit tp: apply TTV offset if active
        if fitttv and ttv_list and j < len(ttv_list) and j < len(epoch_list):
            tc_j = d['tc'] + epoch_list[j] * d['period'] + ttv_list[j]
            tp_j = tc_to_tp(tc_j, d['period'], e, omega)
        else:
            tp_j = d['tp']
        # Check if this transit has detrend covariates
        tran_add_coeffs = d.get('tran_detrendadd', [])
        tran_mult_coeffs = d.get('tran_detrendmult', [])
        has_tran_detrend = (
            (tran_add_coeffs and j < len(tran_add_coeffs) and tran_add_coeffs[j])
            or (tran_mult_coeffs and j < len(tran_mult_coeffs) and tran_mult_coeffs[j])
        )
        # When detrending: pass f0=1.0 to get raw transit, apply F0 externally
        tran_f0 = 1.0 if has_tran_detrend else f0_j
        try:
            model_flux = exozippy_tran(
                tdata['bjd'], d['inc'], d['ar'], tp_j, d['period'],
                e, omega, d['p'], u1_j, u2_j, tran_f0,
                thermal=thermal_j, reflect=reflect_j,
                beam=beam_j, ellipsoidal=ellip_j,
                dilute=dilute_j, tc=d['tc'],
            )
        except Exception:
            return INF_CHI2
        # Apply detrending: model = (transit + C*x) * (F0 + M*z)
        if has_tran_detrend:
            if tran_add_coeffs and j < len(tran_add_coeffs) and tran_add_coeffs[j]:
                detrendadd = tdata.get('detrendadd')  # (nadd, npts)
                if detrendadd is not None:
                    coeffs = np.array(tran_add_coeffs[j])
                    model_flux = model_flux + coeffs @ detrendadd
            mult_term = f0_j
            if tran_mult_coeffs and j < len(tran_mult_coeffs) and tran_mult_coeffs[j]:
                detrendmult = tdata.get('detrendmult')  # (nmult, npts)
                if detrendmult is not None:
                    coeffs = np.array(tran_mult_coeffs[j])
                    mult_term = mult_term + coeffs @ detrendmult
            model_flux = model_flux * mult_term
        resid = tdata['flux'] - model_flux
        err2 = tdata['err']**2 + addvar_j
        val = np.sum(resid**2 / err2)
        if not np.isfinite(val):
            return INF_CHI2
        total += val
    return total


# ---------------------------------------------------------------------------
# 7. RV chi2 (loops over all telescopes)
# ---------------------------------------------------------------------------
def chi2_rv(d, rv_data_list, e, omega, rv_jittervar_list, rmbandndx_list=None):
    """
    Radial-velocity chi2, summed over all telescopes.

    Parameters
    ----------
    d : dict
    rv_data_list : list[dict]
        Each dict has keys: bjd, vel, err.
    rv_jittervar_list : list[float]
        Per-telescope jitter variance.
    rmbandndx_list : list[int] or None
        Per-telescope RM band index (-1 = no RM).
    """
    total = 0.0
    # Global RV trend params
    slope_val = d.get('slope', 0.0) if (d.get('fitslope') or d.get('fitquad')) else None
    quad_val = d.get('quad', 0.0) if d.get('fitquad') else None
    rvepoch = d.get('rvepoch', 0.0)
    for j, rvdata in enumerate(rv_data_list):
        gamma_j = d['gamma'][j]
        # Use fitted jittervar if available, otherwise use fixed list
        if d.get('fitjittervar') and d.get('jittervar'):
            jittervar_j = d['jittervar'][j]
        else:
            jittervar_j = rv_jittervar_list[j] if j < len(rv_jittervar_list) else 0.0
        # Check if this telescope has RM
        has_rm = (d.get('rossiter') and rmbandndx_list is not None
                  and j < len(rmbandndx_list) and rmbandndx_list[j] >= 0)
        try:
            if has_rm:
                bandndx = rmbandndx_list[j]
                u1_rm = d['u1'][bandndx]
                u2_rm = d['u2'][bandndx]
                model_rv = exozippy_rv(
                    rvdata['bjd'], d['tp'], d['period'],
                    gamma_j, d['K'], e=e, omega=omega,
                    slope=slope_val, quad=quad_val, t0=rvepoch,
                    rossiter=True, i=d['inc'], a=d['ar'],
                    u1=u1_rm, u2=u2_rm, p=d['p'],
                    vsini=d['vsini'], _lambda=d['lam'],
                    vgamma=d['vgamma'], vzeta=d['vzeta'],
                    vxi=d['vxi'], valpha=d['valpha'],
                )
            else:
                model_rv = exozippy_rv(
                    rvdata['bjd'], d['tp'], d['period'],
                    gamma_j, d['K'], e=e, omega=omega,
                    slope=slope_val, quad=quad_val, t0=rvepoch,
                )
        except Exception:
            return INF_CHI2
        # Apply RV detrending: model = (rv + RVC*x) * (1 + RVM*z)
        rv_add_coeffs = d.get('rv_detrendadd', [])
        rv_mult_coeffs = d.get('rv_detrendmult', [])
        if rv_add_coeffs and j < len(rv_add_coeffs) and rv_add_coeffs[j]:
            detrendadd = rvdata.get('detrendadd')  # (nadd, npts)
            if detrendadd is not None:
                coeffs = np.array(rv_add_coeffs[j])
                model_rv = model_rv + coeffs @ detrendadd
        if rv_mult_coeffs and j < len(rv_mult_coeffs) and rv_mult_coeffs[j]:
            detrendmult = rvdata.get('detrendmult')  # (nmult, npts)
            if detrendmult is not None:
                coeffs = np.array(rv_mult_coeffs[j])
                model_rv = model_rv * (1.0 + coeffs @ detrendmult)
        resid = rvdata['vel'] - model_rv
        err2 = rvdata['err']**2 + jittervar_j
        val = np.sum(resid**2 / err2)
        if not np.isfinite(val):
            return INF_CHI2
        total += val
    return total


# ---------------------------------------------------------------------------
# 8. Doppler Tomography chi2
# ---------------------------------------------------------------------------
def chi2_dt(d, dt_data_list, dtbandndx_list=None):
    """
    Doppler Tomography chi2, summed over all DT files.

    Parameters
    ----------
    d : dict
        Unpacked parameter dict from ``unpack_params``.
    dt_data_list : list[dict]
        Each dict is the output of ``read_dt_fits()``.
    dtbandndx_list : list[int] or None
        Per-DT-file LD band index (default: 0 for all).
    """
    from .physics.exozippy_dopptom import chi2_dopptom
    total = 0.0
    errscale_list = d.get('errscale', [])
    vsini_kms = d['vsini'] / 1000.0          # m/s → km/s
    vline_kms = d.get('vline', 5000.0) / 1000.0  # m/s → km/s
    for j, dt_data in enumerate(dt_data_list):
        bandndx = (dtbandndx_list[j]
                   if dtbandndx_list is not None and j < len(dtbandndx_list)
                   else 0)
        u1_j = d['u1'][bandndx] if d['u1'] else 0.4
        u2_j = d['u2'][bandndx] if d['u2'] else 0.2
        errscale_j = errscale_list[j] if j < len(errscale_list) else 1.0
        try:
            val = chi2_dopptom(
                dt_data, d['tp'], d['period'], d['e'], d['omega'],
                d['inc'], d['ar'], d['p'], d['lam'],
                vsini_kms, vline_kms, u1_j, u2_j,
                errscale=errscale_j,
            )
        except Exception:
            return INF_CHI2
        if not np.isfinite(val):
            return INF_CHI2
        total += val
    return total


# ---------------------------------------------------------------------------
# 9. Prior chi2
# ---------------------------------------------------------------------------
def chi2_priors(d, priors, use_mist=False, nstars=1):
    """Gaussian prior chi2 penalties + hard bounds."""
    chi2 = 0.0

    # Per-star priors
    for i in range(nstars):
        suffix = f'_{i}' if nstars > 1 else ''
        teff_i = d['teff'][i] if isinstance(d['teff'], np.ndarray) else d['teff']
        feh_i = d['feh'][i] if isinstance(d['feh'], np.ndarray) else d['feh']

        for pname_base, val in [('teff', teff_i), ('feh', feh_i)]:
            pname = f'{pname_base}{suffix}'
            for key in [pname, pname_base]:
                if key in priors and priors[key]['sigma'] > 0:
                    chi2 += ((val - priors[key]['value'])
                             / priors[key]['sigma'])**2
                    break

        if use_mist:
            mstar_i = d['mstar'][i] if isinstance(d['mstar'], np.ndarray) else d['mstar']
            age_i = d['age'][i] if isinstance(d['age'], np.ndarray) else d['age']
            for pname_base, val in [('mstar', mstar_i), ('age', age_i)]:
                pname = f'{pname_base}{suffix}'
                for key in [pname, pname_base]:
                    if key in priors and priors[key]['sigma'] > 0:
                        chi2 += ((val - priors[key]['value'])
                                 / priors[key]['sigma'])**2
                        break

        # Parallax prior (per star)
        for plx_key in [f'parallax{suffix}', 'parallax']:
            if plx_key in priors and priors[plx_key]['sigma'] > 0:
                dist_i = d['distance'][i] if isinstance(d['distance'], np.ndarray) else d['distance']
                plx_model = 1000.0 / dist_i
                chi2 += ((plx_model - priors[plx_key]['value'])
                         / priors[plx_key]['sigma'])**2
                break

        # Av upper bound (per star)
        av_i = d['av'][i] if isinstance(d['av'], np.ndarray) else d['av']
        for av_key in [f'av{suffix}', 'av']:
            if av_key in priors and np.isfinite(priors[av_key].get('upper', np.nan)):
                if av_i > priors[av_key]['upper']:
                    chi2 += ((av_i - priors[av_key]['upper']) / 0.001)**2
                break

    # Per-telescope priors (gamma_0, gamma_1, ...)
    ntel = d.get('ntel', 1)
    for j in range(ntel):
        gamma_key = f'gamma_{j}'
        if gamma_key in priors and priors[gamma_key]['sigma'] > 0:
            chi2 += ((d['gamma'][j] - priors[gamma_key]['value'])
                     / priors[gamma_key]['sigma'])**2

    # Per-transit priors (f0_0, f0_1, ...)
    ntran = d.get('ntran', 1)
    for j in range(ntran):
        f0_key = f'f0_{j}'
        if f0_key in priors and priors[f0_key]['sigma'] > 0:
            chi2 += ((d['f0'][j] - priors[f0_key]['value'])
                     / priors[f0_key]['sigma'])**2

    # Shared orbital priors
    orbital_map = {
        'period_0': d['period'], 'tc': d['tc'],
        'p': d['p'], 'cosi': d['cosi'],
        'k_0': d['K'],
    }
    for pname, pval in orbital_map.items():
        if pname in priors and priors[pname]['sigma'] > 0:
            chi2 += ((pval - priors[pname]['value'])
                     / priors[pname]['sigma'])**2

    return chi2


# ---------------------------------------------------------------------------
# 9. Combined chi2  (the main entry point, analogous to exofast_chi2v2)
# ---------------------------------------------------------------------------
def joint_chi2(params, tran_data_list, rv_data_list, sedfile, priors,
               e=0.0, omega=np.pi / 2,
               rv_jittervar_list=None, tran_addvar_list=None,
               use_mist=False, mstar_fixed=None, age_prior=None,
               sed_data=None, nstars=1,
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
    """
    Total chi2 for the joint SED (+ optional MIST) + Transit + RV (+ optional DT) fit.

    Parameters
    ----------
    params : ndarray
        Parameter vector in the order given by ``param_names(...)``.
    tran_data_list : list[dict]
        Per-transit data dicts with keys: bjd, flux, err.
    rv_data_list : list[dict]
        Per-telescope data dicts with keys: bjd, vel, err.
    sedfile : str or None
    priors : dict
    rv_jittervar_list : list[float] or None
        Per-telescope jitter variance.
    tran_addvar_list : list[float] or None
        Per-transit added variance.
    ntran, ntel, nbands : int
    fitdt : bool
        Include Doppler Tomography chi2.
    fiterrscale : bool
        Fit per-DT-file error scale.
    dt_data_list : list[dict] or None
        Per-DT-file dicts from ``read_dt_fits()``.
    dtbandndx_list : list[int] or None
        Per-DT-file LD band index.
    """
    if rv_jittervar_list is None:
        rv_jittervar_list = [0.0] * ntel
    if tran_addvar_list is None:
        tran_addvar_list = [0.0] * ntran

    has_sed = sedfile is not None
    ndt = len(dt_data_list) if dt_data_list else 0
    # 1. Unpack
    d = unpack_params(params, use_mist=use_mist,
                      mstar_fixed=mstar_fixed, age_prior=age_prior,
                      priors=priors, nstars=nstars, has_sed=has_sed,
                      ntran=ntran, ntel=ntel, nbands=nbands,
                      circular=circular, usevcve=usevcve,
                      fitjittervar=fitjittervar, fitslope=fitslope, fitquad=fitquad,
                      fitvariance=fitvariance,
                      fitdilute=fitdilute, fitttv=fitttv, epoch_list=epoch_list,
                      rvepoch=rvepoch,
                      fitthermal=fitthermal, fitreflect=fitreflect,
                      fitbeam=fitbeam, fitellip=fitellip,
                      rossiter=rossiter,
                      fitdt=fitdt, ndt=ndt, fiterrscale=fiterrscale,
                      detrend_info=detrend_info)
    if d is None:
        return INF_CHI2

    # 2. Bounds
    bnd = check_bounds(d)
    if bnd > 0:
        return bnd

    # Use e/omega from dict (derived from sesinw/secosw if non-circular)
    e_val = d['e']
    omega_val = d['omega']

    # 3. Derived quantities
    compute_derived(d, e_val, omega_val)

    total = 0.0

    # 4. MIST evolutionary penalty (per star)
    if use_mist:
        val = chi2_mist(d, nstars=nstars)
        if val >= INF_CHI2:
            return INF_CHI2
        total += val

    # 5. SED (multi-star blending handled by mistmultised)
    if sedfile is not None:
        val = chi2_sed(d, sedfile, sed_data=sed_data, nstars=nstars)
        if val >= INF_CHI2:
            return INF_CHI2
        total += val

    # 6. Transit (all light curves)
    if tran_data_list:
        val = chi2_transit(d, tran_data_list, e_val, omega_val, tran_addvar_list)
        if val >= INF_CHI2:
            return INF_CHI2
        total += val

    # 7. RV (all telescopes)
    if rv_data_list:
        val = chi2_rv(d, rv_data_list, e_val, omega_val, rv_jittervar_list,
                      rmbandndx_list=rmbandndx_list)
        if val >= INF_CHI2:
            return INF_CHI2
        total += val

    # 8. Doppler Tomography (all DT files)
    if fitdt and dt_data_list:
        val = chi2_dt(d, dt_data_list, dtbandndx_list=dtbandndx_list)
        if val >= INF_CHI2:
            return INF_CHI2
        total += val

    # 9. Priors
    total += chi2_priors(d, priors, use_mist, nstars=nstars)

    return total
