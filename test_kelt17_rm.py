"""
Quick functional test for the KELT-17 Rossiter-McLaughlin implementation.

Tests:
  1. exozippy_rossiter() returns a non-trivial RM signal
  2. mkss() builds the SS structure with rossiter=True
  3. Prior name aliases (svsinicoslambda→svsinicoslam and vsini+lambda) work
  4. chi2 evaluation with RM telescopes returns a finite, non-INF value
"""
import os
import sys
import glob
import numpy as np

# ── paths ─────────────────────────────────────────────────────────────
BASE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                    'exozippy', 'data', 'exofastv2', 'examples', 'kelt17', 'RM')

PRIORFILE = os.path.join(BASE, 'kelt17.priors')
SEDFILE   = os.path.join(BASE, 'kelt17.sed')
TRANPATH  = os.path.join(BASE, 'n20??????.*.dat')
RVPATH    = os.path.join(BASE, 'KELT-17b.*.rv')

PASS = '\033[32mPASS\033[0m'
FAIL = '\033[31mFAIL\033[0m'


def section(title):
    print(f'\n{"─"*60}')
    print(f'  {title}')
    print(f'{"─"*60}')


# ══════════════════════════════════════════════════════════════════════
# Test 1 — exozippy_rossiter() gives a non-trivial RM signal
# ══════════════════════════════════════════════════════════════════════

section('Test 1 — exozippy_rossiter() RM signal')

from exozippy.exozippy_rossiter import exozippy_rossiter
from exozippy.utils import exozippy_getphase

# KELT-17b: tc_0=2457330.86812, P=3.0801735 d
# RM night 1 observed: BJD 2457441.63 … 2457441.88  (TRES_RM0.rv)
# Transit epoch: round((2457441.7 - 2457330.86812) / 3.0801735) = 36
tc_0   = 2457330.86812
period = 3.0801735
epoch  = round((2457441.7 - tc_0) / period)
tc     = tc_0 + epoch * period      # ≈ 2457441.754

e       = 0.0
omega   = np.pi / 2
# For circular orbit: phase = exozippy_getphase(0, pi/2) = 0 → tp = tc
tp      = tc - exozippy_getphase(e, omega, primary=True) * period

inc     = np.radians(83.0)   # KELT-17b published inclination
ar      = 7.0                # a/R*
p       = 0.096              # Rp/R*
u1, u2  = 0.4, 0.2

# KELT-17b: vsini ≈ 44.2 km/s, lambda ≈ -115.9°
vsini   = 44200.0
lam     = np.radians(-115.9)
vgamma  = 1000.0
vzeta   = 4000.0
vxi     = 1000.0
valpha  = 0.0

print(f'  Transit center (epoch {epoch}): BJD {tc:.5f}')
print(f'  tp (for circular e=0, omega=pi/2): {tp:.5f}')

# Sample densely around the transit
bjd_rm = np.linspace(tc - 0.15, tc + 0.15, 100)

delta_rv = exozippy_rossiter(bjd_rm, tp, period, e, omega,
                              inc, ar, p, u1, u2,
                              vsini, lam, vgamma, vzeta, vxi, valpha)

in_transit = np.sum(np.abs(delta_rv) > 0)
peak       = np.max(np.abs(delta_rv))
print(f'  In-transit points: {in_transit}/{len(bjd_rm)}')
print(f'  Peak |ΔRV|:        {peak:.1f} m/s')
print(f'  Min ΔRV:           {np.min(delta_rv):.1f} m/s')
print(f'  Max ΔRV:           {np.max(delta_rv):.1f} m/s')

ok1 = (peak > 100) and (in_transit > 0)
print(f'  Result: {PASS if ok1 else FAIL}'
      f'  (expected peak > 100 m/s and some in-transit points)')


# ══════════════════════════════════════════════════════════════════════
# Test 2 — prior alias  svsinicoslambda → svsinicoslam
# ══════════════════════════════════════════════════════════════════════

section('Test 2 — prior aliases (svsinicoslambda→svsinicoslam)')

from exozippy.mkss import _parse_priors

priors = _parse_priors(PRIORFILE)
print('  RM-related priors after parsing:')
for k, v in sorted(priors.items()):
    if any(sub in k.lower() for sub in ['vsini', 'lambda', 'vgamma',
                                          'vzeta', 'vxi', 'valpha']):
        print(f'    {k:30s}: val={v["value"]:.4f}')

has_svsinicoslam  = 'svsinicoslam'  in priors
has_svsinisinlam  = 'svsinisinlam'  in priors

print(f'\n  svsinicoslam  present: {has_svsinicoslam}')
print(f'  svsinisinlam  present: {has_svsinisinlam}')

if has_svsinicoslam and has_svsinisinlam:
    c = priors['svsinicoslam']['value']
    s = priors['svsinisinlam']['value']
    vsini_d = c**2 + s**2
    lam_d   = np.degrees(np.arctan2(s, c))
    print(f'  derived vsini = {vsini_d:.1f} m/s')
    print(f'  derived lam   = {lam_d:.2f}°')

ok2 = has_svsinicoslam and has_svsinisinlam
print(f'  Result: {PASS if ok2 else FAIL}')


# ══════════════════════════════════════════════════════════════════════
# Test 3 — mkss builds SS with rossiter=True; telescope RM assignment
# ══════════════════════════════════════════════════════════════════════

section('Test 3 — mkss builds SS with rossiter=True')

from exozippy.mkss import mkss

ss = mkss(
    parfile   = PRIORFILE,
    tranpath  = TRANPATH,
    rvpath    = RVPATH,
    sedfile   = SEDFILE,
    rossiter  = True,
    rmbands   = ['notrm', 'V', 'V'],   # TRES=no RM; TRES_RM0, TRES_RM1 = V band
    use_mist  = True,
    circular  = True,
)

print(f'  ntransits  : {len(ss.transit)}')
print(f'  ntelescopes: {len(ss.telescope)}')
print(f'  nbands     : {len(ss.band)}')
print(f'  pl.rossiter: {ss.planet[0].rossiter}')

print('  Telescopes:')
rv_sorted = sorted(glob.glob(RVPATH))
for j, tel in enumerate(ss.telescope):
    fname = os.path.basename(rv_sorted[j]) if j < len(rv_sorted) else '?'
    print(f'    [{j}] {fname:30s}  rmband={tel.rmband!r:10s}  rmbandndx={tel.rmbandndx}')

ok3 = (ss.planet[0].rossiter
       and ss.telescope[0].rmbandndx == -1
       and ss.telescope[1].rmbandndx >= 0
       and ss.telescope[2].rmbandndx >= 0)
print(f'  Result: {PASS if ok3 else FAIL}'
      f'  (rossiter=True, tel[0] notrm, tel[1/2] have RM band)')

# Check svsinicoslam was initialized from prior (not zero)
sv_c = ss.planet[0].svsinicoslam.value
sv_s = ss.planet[0].svsinisinlam.value
print(f'\n  ss.planet[0].svsinicoslam = {sv_c:.4f}')
print(f'  ss.planet[0].svsinisinlam = {sv_s:.4f}')
ok3b = abs(sv_c) > 1.0 or abs(sv_s) > 1.0
print(f'  svsinicoslam/svsinisinlam non-zero: {PASS if ok3b else FAIL}')
ok3 = ok3 and ok3b


# ══════════════════════════════════════════════════════════════════════
# Test 4 — chi2 evaluation with RM returns finite value
# ══════════════════════════════════════════════════════════════════════

section('Test 4 — chi2 evaluation with RM')

from exozippy.exozippy_chi2 import joint_chi2, param_names, INF_CHI2
from exozippy.fit_exoplanet import read_all_transit_data, read_all_rv_data

tran_data_list, _ = read_all_transit_data(TRANPATH)
rv_data_list, _   = read_all_rv_data(RVPATH)

ntran  = len(tran_data_list)
ntel   = len(rv_data_list)
nbands = len(ss.band)

pnames = param_names(
    use_mist = True,
    has_sed  = True,
    ntran    = ntran,
    ntel     = ntel,
    nbands   = nbands,
    circular = True,
    rossiter = True,
)
print(f'  Parameter names ({len(pnames)}): {pnames}')

ss.compute_derived()
x0 = ss.to_vector(pnames)
print(f'  x0 (len={len(x0)}): ...svsinicoslam={x0[-4]:.3f}  svsinisinlam={x0[-3]:.3f}')

rmbandndx_list    = [tel.rmbandndx for tel in ss.telescope]
rv_jittervar_list = [0.0] * ntel
tran_addvar_list  = [0.0] * ntran
priors_dict       = _parse_priors(PRIORFILE)

chi2_val = joint_chi2(
    x0, tran_data_list, rv_data_list,
    SEDFILE, priors_dict,
    rv_jittervar_list = rv_jittervar_list,
    tran_addvar_list  = tran_addvar_list,
    use_mist          = True,
    ntran             = ntran,
    ntel              = ntel,
    nbands            = nbands,
    circular          = True,
    rossiter          = True,
    rmbandndx_list    = rmbandndx_list,
)

n_data = (sum(len(t['bjd']) for t in tran_data_list) +
          sum(len(r['bjd']) for r in rv_data_list))
ndof = max(n_data - len(x0), 1)

print(f'\n  chi2     = {chi2_val:.2f}')
print(f'  n_data   = {n_data}')
print(f'  n_params = {len(x0)}')
print(f'  ndof     = {ndof}')
if ndof > 0:
    print(f'  chi2/dof = {chi2_val/ndof:.3f}')

ok4 = np.isfinite(chi2_val) and 0 < chi2_val < INF_CHI2
print(f'  Result: {PASS if ok4 else FAIL}'
      f'  (chi2 must be finite and < {INF_CHI2})')


# ══════════════════════════════════════════════════════════════════════
# Summary
# ══════════════════════════════════════════════════════════════════════

section('Summary')

results = [ok1, ok2, ok3, ok4]
labels  = ['RM signal', 'Prior alias', 'mkss+rossiter', 'chi2 evaluation']
for lbl, ok in zip(labels, results):
    print(f'  {lbl:20s}: {PASS if ok else FAIL}')

all_pass = all(results)
print(f'\n  Overall: {PASS if all_pass else FAIL}')
sys.exit(0 if all_pass else 1)
