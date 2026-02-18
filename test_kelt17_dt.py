"""
Quick functional test for the KELT-17 Doppler Tomography implementation.

Tests:
  1. read_dt_fits() correctly parses a DT FITS file (shape, Rspec, rms)
  2. compute_dt_model() returns a non-trivial 2-D model (Doppler shadow present)
  3. chi2_dopptom() returns a finite value
  4. joint_chi2() with fitdt=True returns a finite value
  5. param_names() ordering includes vline (and svsinicoslam if !rossiter)
"""
import os
import sys
import glob
import numpy as np

BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'exozippy', 'data', 'exofastv2', 'examples', 'kelt17', 'DT')

PRIORFILE = os.path.join(BASE, 'kelt17.priors')
SEDFILE   = os.path.join(BASE, 'kelt17.sed')
TRANPATH  = os.path.join(BASE, 'n20??????.*.dat')
RVPATH    = os.path.join(BASE, 'KELT-17b.*.rv')
DTPATH    = os.path.join(BASE, 'n20??????.KELT-17b.TRES.44000.fits')

PASS = '\033[32mPASS\033[0m'
FAIL = '\033[31mFAIL\033[0m'


def section(title):
    print(f'\n{"─"*60}')
    print(f'  {title}')
    print(f'{"─"*60}')


# ══════════════════════════════════════════════════════════════════════
# Test 1 — read_dt_fits parses FITS correctly
# ══════════════════════════════════════════════════════════════════════

section('Test 1 — read_dt_fits()')

from exozippy.exozippy_dopptom import read_dt_fits

dtfiles = sorted(glob.glob(DTPATH))
print(f'  DT files found: {len(dtfiles)}')
for f in dtfiles:
    print(f'    {os.path.basename(f)}')

ok1 = len(dtfiles) >= 1
dt_data = None
if ok1:
    dt_data = read_dt_fits(dtfiles[0])
    print(f'  ccf2d shape : {dt_data["ccf2d"].shape}')
    print(f'  bjd range   : {dt_data["bjd"].min():.2f} … {dt_data["bjd"].max():.2f}')
    print(f'  vel range   : {dt_data["vel"].min():.1f} … {dt_data["vel"].max():.1f} km/s')
    print(f'  Rspec       : {dt_data["Rspec"]}')
    print(f'  rms         : {dt_data["rms"]:.6f}')
    ok1 = (dt_data['ccf2d'].ndim == 2
            and dt_data['Rspec'] == 44000
            and dt_data['rms'] > 0
            and len(dt_data['bjd']) == dt_data['ccf2d'].shape[0])

print(f'  Result: {PASS if ok1 else FAIL}')


# ══════════════════════════════════════════════════════════════════════
# Test 2 — compute_dt_model returns non-trivial Doppler shadow
# ══════════════════════════════════════════════════════════════════════

section('Test 2 — compute_dt_model()')

from exozippy.exozippy_dopptom import compute_dt_model
from exozippy.utils import exozippy_getphase

# KELT-17b parameters (Beatty 2017)
tc_0   = 2457330.86812
period = 3.0801735
e      = 0.0
omega  = np.pi / 2

# Compute transit epoch for night 1 (n20160222)
# BJD of DT observations ~2457440.8 (n20160222)
epoch  = round((2457440.8 - tc_0) / period)
tc     = tc_0 + epoch * period
tp     = tc - exozippy_getphase(e, omega, primary=True) * period

inc    = np.radians(83.0)
ar     = 7.0
p      = 0.096
u1, u2 = 0.4, 0.2
vsini_kms = 44.2          # km/s
lam     = np.radians(-115.9)
vline_kms = 5.0           # km/s (intrinsic line broadening)

print(f'  Transit tc (epoch {epoch}): BJD {tc:.5f}')
print(f'  tp (circular, omega=pi/2): {tp:.5f}')

ok2 = False
if dt_data is not None:
    model = compute_dt_model(dt_data, tp, period, e, omega,
                              inc, ar, p, lam, vsini_kms, vline_kms, u1, u2)
    baseline = np.median(dt_data['ccf2d'])
    deviation = np.max(np.abs(model - baseline))
    print(f'  model shape     : {model.shape}')
    print(f'  baseline (median): {baseline:.6f}')
    print(f'  max |model-base|: {deviation:.6f}')
    ok2 = (model.shape == dt_data['ccf2d'].shape) and (deviation > 0)

print(f'  Result: {PASS if ok2 else FAIL}'
      f'  (Doppler shadow present if deviation > 0)')


# ══════════════════════════════════════════════════════════════════════
# Test 3 — chi2_dopptom returns finite value
# ══════════════════════════════════════════════════════════════════════

section('Test 3 — chi2_dopptom()')

from exozippy.exozippy_dopptom import chi2_dopptom

ok3 = False
if dt_data is not None:
    chi2_val = chi2_dopptom(
        dt_data, tp, period, e, omega,
        inc, ar, p, lam, vsini_kms, vline_kms, u1, u2,
        errscale=1.0,
    )
    print(f'  chi2_dopptom = {chi2_val:.3f}')
    ok3 = np.isfinite(chi2_val) and chi2_val > 0

print(f'  Result: {PASS if ok3 else FAIL}')


# ══════════════════════════════════════════════════════════════════════
# Test 4 — param_names ordering with fitdt
# ══════════════════════════════════════════════════════════════════════

section('Test 4 — param_names with fitdt=True')

from exozippy.exozippy_chi2 import param_names

pnames_dt_only = param_names(
    use_mist=True, has_sed=True,
    ntran=1, ntel=1, nbands=1,
    circular=True, rossiter=False,
    fitdt=True, ndt=2, fiterrscale=True,
)
print(f'  fitdt=True, rossiter=False, ndt=2, fiterrscale=True:')
print(f'    last 6 params: {pnames_dt_only[-6:]}')
has_svsinicoslam = 'svsinicoslam' in pnames_dt_only
has_svsinisinlam = 'svsinisinlam' in pnames_dt_only
has_vline        = 'vline'        in pnames_dt_only
has_errscale0    = 'errscale_0'   in pnames_dt_only
has_errscale1    = 'errscale_1'   in pnames_dt_only
ok4a = has_svsinicoslam and has_svsinisinlam and has_vline and has_errscale0 and has_errscale1
print(f'  svsinicoslam: {has_svsinicoslam}, svsinisinlam: {has_svsinisinlam}')
print(f'  vline: {has_vline}, errscale_0: {has_errscale0}, errscale_1: {has_errscale1}')

# When rossiter=True, svsinicoslam/svsinisinlam should NOT be duplicated
pnames_rm_dt = param_names(
    use_mist=True, has_sed=True,
    ntran=1, ntel=1, nbands=1,
    circular=True, rossiter=True,
    fitdt=True, ndt=1, fiterrscale=False,
)
count_svsinicoslam = pnames_rm_dt.count('svsinicoslam')
ok4b = count_svsinicoslam == 1   # shared, not duplicated
print(f'  rossiter=True + fitdt=True: svsinicoslam count = {count_svsinicoslam} (expected 1)')
ok4 = ok4a and ok4b
print(f'  Result: {PASS if ok4 else FAIL}')


# ══════════════════════════════════════════════════════════════════════
# Test 5 — joint_chi2 with fitdt=True returns finite value
# ══════════════════════════════════════════════════════════════════════

section('Test 5 — joint_chi2 with fitdt=True')

from exozippy.exozippy_chi2 import joint_chi2, param_names, INF_CHI2
from exozippy.fit_exoplanet import read_all_transit_data, read_all_rv_data, read_all_dt_data
from exozippy.mkss import mkss, _parse_priors

tran_data_list, _ = read_all_transit_data(TRANPATH)
rv_data_list, _   = read_all_rv_data(RVPATH)
dt_data_list, _   = read_all_dt_data(DTPATH)

ntran  = len(tran_data_list)
ntel   = len(rv_data_list)
ndt    = len(dt_data_list)
nbands = 1

print(f'  ntran={ntran}, ntel={ntel}, ndt={ndt}')

pnames = param_names(
    use_mist=True, has_sed=True,
    ntran=ntran, ntel=ntel, nbands=nbands,
    circular=True, rossiter=False,
    fitdt=True, ndt=ndt, fiterrscale=False,
)
print(f'  n_params = {len(pnames)}')
print(f'  last 4  : {pnames[-4:]}')

priors_dict = _parse_priors(PRIORFILE)

ss = mkss(
    parfile=PRIORFILE,
    tranpath=TRANPATH,
    rvpath=RVPATH,
    sedfile=SEDFILE,
    use_mist=True,
    circular=True,
    dtpath=DTPATH,
    fitdt=True,
    fiterrscale=False,
)
ss.compute_derived()
x0 = ss.to_vector(pnames)

dtbandndx_list = [0] * ndt
chi2_val = joint_chi2(
    x0, tran_data_list, rv_data_list,
    SEDFILE, priors_dict,
    use_mist=True,
    ntran=ntran, ntel=ntel, nbands=nbands,
    circular=True, rossiter=False,
    fitdt=True, fiterrscale=False,
    dt_data_list=dt_data_list, dtbandndx_list=dtbandndx_list,
)

n_data = (sum(len(t['bjd']) for t in tran_data_list) +
          sum(len(r['bjd']) for r in rv_data_list) +
          sum(d['ccf2d'].size for d in dt_data_list))
ndof = max(n_data - len(x0), 1)

print(f'\n  chi2     = {chi2_val:.2f}')
print(f'  n_data   = {n_data}')
print(f'  n_params = {len(x0)}')
print(f'  ndof     = {ndof}')
print(f'  chi2/dof = {chi2_val/ndof:.4f}')

ok5 = np.isfinite(chi2_val) and 0 < chi2_val < INF_CHI2
print(f'  Result: {PASS if ok5 else FAIL}')


# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

section('Summary')

results = [ok1, ok2, ok3, ok4, ok5]
labels  = ['read_dt_fits', 'compute_dt_model', 'chi2_dopptom',
           'param_names (DT)', 'joint_chi2 (DT)']
for lbl, ok in zip(labels, results):
    print(f'  {lbl:25s}: {PASS if ok else FAIL}')

all_pass = all(results)
print(f'\n  Overall: {PASS if all_pass else FAIL}')
sys.exit(0 if all_pass else 1)
