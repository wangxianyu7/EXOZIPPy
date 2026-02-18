"""
KELT-17b Doppler Tomography fit using EXOZIPPy.

Reference: Beatty et al. 2017
Published values:
  Mstar  = 1.635 Msun
  Teff   = 7454 K
  vsini  = 44.2 km/s
  lambda = -115.9 deg
  P      = 3.0801735 d
  p      = Rp/Rstar = 0.0957

Data in this directory:
  Transits : n20??????.*.dat       (12 light curves, multi-band)
  RVs      : KELT-17b.*.rv        (TRES + TRESRM)
  DT       : n201?????.KELT-17b.TRES.44000.fits  (2 nights)
  SED      : kelt17.sed
  Priors   : kelt17.priors

Usage
-----
  cd /Volumes/SSDDISK/MacMiniM4/Github/EXOZIPPy
  python exozippy/data/exofastv2/examples/kelt17/DT/fit_kelt17_dt.py
"""

import os
import sys
import glob
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
PRIORFILE = os.path.join(HERE, 'kelt17.priors')
SEDFILE   = os.path.join(HERE, 'kelt17.sed')
TRANPATH  = os.path.join(HERE, 'n20??????.*.dat')
RVPATH    = os.path.join(HERE, 'KELT-17b.*.rv')
DTPATH    = os.path.join(HERE, 'n201?????.KELT-17b.TRES.44000.fits')
OUTDIR    = os.path.join(HERE, 'fitresults')
os.makedirs(OUTDIR, exist_ok=True)

# ── Quick data summary ──────────────────────────────────────────────────
tran_files = sorted(glob.glob(TRANPATH))
rv_files   = sorted(glob.glob(RVPATH))
dt_files   = sorted(glob.glob(DTPATH))
print(f'Transits : {len(tran_files)} files')
for f in tran_files:
    print(f'  {os.path.basename(f)}')
print(f'RVs      : {len(rv_files)} files')
for f in rv_files:
    print(f'  {os.path.basename(f)}')
print(f'DT       : {len(dt_files)} files')
for f in dt_files:
    print(f'  {os.path.basename(f)}')
print()

# ── Run fit ────────────────────────────────────────────────────────────
from exozippy.jointfit import fit_exoplanet

print('=' * 60)
print('Starting KELT-17b DT fit (optimizer only, no MCMC)')
print('=' * 60)

bestfit = fit_exoplanet(
    priorfile  = PRIORFILE,
    tranfile   = TRANPATH,
    rvfile     = RVPATH,
    sedfile    = SEDFILE,
    # Orbit
    circular   = True,
    # Stellar physics
    use_mist   = True,
    # Doppler Tomography
    dtpath     = DTPATH,
    fitdt      = True,
    fiterrscale= False,    # set True to fit per-DT-file error scales
    # Rossiter–McLaughlin (separate from DT)
    rossiter   = False,
    verbose    = True,
)

# ── Print key results ──────────────────────────────────────────────────
print()
print('=' * 60)
print('Best-fit results')
print('=' * 60)

# Helper: get scalar from SS or dict
def _val(ss, name, default=np.nan):
    try:
        v = ss[name]
        if hasattr(v, 'value'):
            v = v.value
        return float(v)
    except Exception:
        return default

mstar    = _val(bestfit, 'mstar')
teff     = _val(bestfit, 'teff')
rstar    = _val(bestfit, 'rstar')
feh      = _val(bestfit, 'feh')
tc       = _val(bestfit, 'tc')
period   = _val(bestfit, 'period')
p_rp     = _val(bestfit, 'p')
cosi_val = _val(bestfit, 'cosi')
inc      = np.degrees(np.arccos(np.clip(cosi_val, -1.0, 1.0)))
ar       = _val(bestfit, 'ar')
K        = _val(bestfit, 'K')

# Spin–orbit params (stored in m^0.5/s^0.5 internally)
svsinicoslam = _val(bestfit, 'svsinicoslam')
svsinisinlam = _val(bestfit, 'svsinisinlam')
vsini_ms     = svsinicoslam**2 + svsinisinlam**2   # m/s
lam_rad      = np.arctan2(svsinisinlam, svsinicoslam)
vsini_kms    = vsini_ms / 1000.0
lam_deg      = np.degrees(lam_rad)

vline_ms     = _val(bestfit, 'vline')              # m/s
vline_kms    = vline_ms / 1000.0

print(f'Stellar:')
print(f'  Mstar  = {mstar:.4f} Msun     (published: 1.635)')
print(f'  Teff   = {teff:.1f} K       (published: 7454)')
print(f'  Rstar  = {rstar:.4f} Rsun     (published: 1.645)')
print(f'  [Fe/H] = {feh:.3f} dex')
print()
print(f'Orbital:')
print(f'  Tc     = {tc:.5f} BJD')
print(f'  Period = {period:.7f} d        (published: 3.0801735)')
print(f'  Rp/Rs  = {p_rp:.4f}            (published: 0.0957)')
print(f'  inc    = {inc:.3f} deg         (published: 83.0)')
print(f'  a/Rs   = {ar:.4f}')
print(f'  K      = {K:.2f} m/s')
print()
print(f'Spin-orbit (DT):')
print(f'  vsini         = {vsini_kms:.2f} km/s     (published: 44.2 km/s)')
print(f'  lambda        = {lam_deg:.2f} deg      (published: -115.9 deg)')
print(f'  vline         = {vline_kms:.2f} km/s     (published: 5.49 km/s)')
print(f'  svsinicoslam  = {svsinicoslam:.4f} m^0.5/s^0.5')
print(f'  svsinisinlam  = {svsinisinlam:.4f} m^0.5/s^0.5')
print()

# ── Optional: compute chi2 breakdown ───────────────────────────────────
try:
    from exozippy.jointfit import (
        read_all_transit_data, read_all_rv_data, read_all_dt_data,
        _build_detrend_info,
    )
    from exozippy.jointfit.mkss import _parse_priors
    from exozippy.jointfit.chi2 import joint_chi2
    from exozippy.jointfit.chi2 import chi2_dt as _chi2_dt
    from exozippy.jointfit.chi2 import unpack_params as _unpack
    from exozippy.jointfit.chi2 import compute_derived as _compute_derived

    tran_data_list, _ = read_all_transit_data(TRANPATH)
    rv_data_list,   _ = read_all_rv_data(RVPATH)
    dt_data_list,   _ = read_all_dt_data(DTPATH)
    priors_dict = _parse_priors(PRIORFILE)
    detrend_info = _build_detrend_info(tran_data_list, rv_data_list)

    ntran  = len(tran_data_list)
    ntel   = len(rv_data_list)
    ndt    = len(dt_data_list)
    nbands = 1

    # Use the exact param names and solution vector from the optimizer
    pnames = bestfit.param_names
    x0     = bestfit.opt_params     # full optimizer solution (includes detrend)
    mstar_fixed = priors_dict.get('mstar', {}).get('value', 1.0)
    age_prior   = priors_dict.get('age',   {}).get('value', 1.0)

    chi2_total = joint_chi2(
        x0, tran_data_list, rv_data_list,
        SEDFILE, priors_dict,
        use_mist=True, mstar_fixed=mstar_fixed, age_prior=age_prior,
        ntran=ntran, ntel=ntel, nbands=nbands,
        circular=True, rossiter=False,
        fitdt=True, fiterrscale=False,
        dt_data_list=dt_data_list, dtbandndx_list=[0]*ndt,
        detrend_info=detrend_info,
    )

    # DT chi2 alone
    d = _unpack(x0, use_mist=True, mstar_fixed=mstar_fixed, age_prior=age_prior,
                priors=priors_dict, nstars=1, has_sed=True,
                ntran=ntran, ntel=ntel, nbands=nbands,
                circular=True, fitdt=True, ndt=ndt, fiterrscale=False,
                detrend_info=detrend_info)
    _compute_derived(d)   # add ar, inc, tp to d
    chi2_dt_val = _chi2_dt(d, dt_data_list, [0]*ndt)
    print(f'  [debug] inc={np.degrees(d["inc"]):.2f} deg, ar={d["ar"]:.3f}, p={d["p"]:.4f}, vsini={d["vsini"]/1000:.2f} km/s, lam={np.degrees(d["lam"]):.2f} deg')

    # Effective DT data points
    from exozippy.physics.exozippy_dopptom import C_LIGHT, FWHM2SIGMA
    neff_dt = 0.0
    for dtd in dt_data_list:
        vel = dtd['vel']
        dv  = float(np.mean(np.diff(vel)))  # km/s per pixel
        rvel = C_LIGHT / dtd['Rspec']       # km/s FWHM
        IndepVels = (rvel / FWHM2SIGMA) / (dv / vsini_kms)
        neff_dt += dtd['ccf2d'].shape[0] * IndepVels  # ntime * IndepVels

    n_data_nodt = (sum(len(t['bjd']) for t in tran_data_list) +
                   sum(len(r['bjd']) for r in rv_data_list))
    print(f'Chi2 breakdown:')
    print(f'  Total chi2          = {chi2_total:.1f}')
    print(f'  DT chi2             = {chi2_dt_val:.1f}')
    print(f'  DT eff. data pts    = {neff_dt:.1f}')
    print(f'  DT chi2/eff.dof     = {chi2_dt_val / max(neff_dt, 1):.3f}')
    n_data = n_data_nodt + neff_dt
    ndof   = max(n_data - len(x0), 1)
    print(f'  n_data (incl. DT)   = {n_data:.0f}')
    print(f'  n_params            = {len(x0)}')
    print(f'  chi2/dof            = {chi2_total / ndof:.3f}')
except Exception as exc:
    print(f'Chi2 breakdown failed: {exc}')

print()
print(f'Output directory: {OUTDIR}')
print('Done.')
