"""
Profile chi2 computation — measure time spent in each module.

Usage:
    python profile_chi2.py
"""
import time
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

# ── paths ──
base = os.path.join(os.path.dirname(__file__),
                    'exozippy', 'data', 'exofastv2', 'examples', 'hat3') + '/'
priorfile = base + 'HAT-3.priors'
tranfile  = base + 'n20070428.Sloani.KepCam.dat'
rvfile    = base + 'HAT-3b.HIRES.rv'
sedfile   = base + 'HAT-3.sed'

# ── load data (one-time) ──
from exozippy.fit_exoplanet import parse_priors, read_transit_data, read_rv_data
from exozippy.exozippy_chi2 import (
    unpack_params, check_bounds, compute_derived,
    chi2_mist, chi2_sed, chi2_transit, chi2_rv, chi2_priors,
    param_names as _param_names, joint_chi2,
)
from exozippy.sed.utils import read_sed_file

priors    = parse_priors(priorfile)
tran_data = read_transit_data(tranfile)
rv_data   = read_rv_data(rvfile)
sed_data  = read_sed_file(sedfile, 1)
use_mist  = False
e, omega  = 0.0, np.pi / 2
mstar_prior = priors.get('mstar', {}).get('value', 1.0)
age_prior   = priors.get('age', {}).get('value', 1.0)

print(f"Transit data: {len(tran_data['bjd'])} points")
print(f"RV data:      {len(rv_data['bjd'])} points")

# ── build a reasonable parameter vector ──
param_names = _param_names(use_mist)
teff0    = priors.get('teff', {}).get('value', 5500.0)
rstar0   = priors.get('rstar', {}).get('value', 1.0)
feh0     = priors.get('feh', {}).get('value', 0.0)
av0      = 0.01
dist0    = 1000.0 / priors['parallax']['value'] if 'parallax' in priors else 135.0
tc0      = priors.get('tc', {}).get('value', np.median(tran_data['bjd']))
period0  = priors.get('period_0', {}).get('value', 3.0)
p0       = priors.get('p', {}).get('value', 0.1)
cosi0    = priors.get('cosi', {}).get('value', 0.05)
u1_0, u2_0 = 0.4, 0.2
f0_0     = priors.get('f0', {}).get('value', np.median(tran_data['flux']))
K0       = priors.get('k_0', {}).get('value', 50.0)
gamma0   = priors.get('gamma_0', {}).get('value', 0.0)

x0 = np.array([teff0, rstar0, feh0, av0, dist0,
               tc0, period0, p0, cosi0,
               u1_0, u2_0, f0_0,
               K0, gamma0])

print(f"\nParameter vector ({len(x0)} params): {param_names}")
print(f"x0 = {x0}\n")

# ── warm-up call (trigger all caches / JIT) ──
print("=" * 60)
print("Warm-up call (first call, includes file I/O / cache init)...")
print("=" * 60)
t_total_start = time.perf_counter()
chi2_val = joint_chi2(x0, tran_data, rv_data, sedfile, priors,
                      e, omega, use_mist=use_mist,
                      mstar_fixed=mstar_prior, age_prior=age_prior,
                      sed_data=sed_data)
t_total_end = time.perf_counter()
print(f"Warm-up total: {1000*(t_total_end - t_total_start):.2f} ms   chi2 = {chi2_val:.2f}\n")

# ── detailed profiling (caches warm) ──
print("=" * 60)
print("Profiling each component (caches warm)")
print("=" * 60)

N_REPEAT = 20

def timeit(label, func, n=N_REPEAT):
    """Time a function over n calls, return mean/min/max in ms."""
    times = []
    result = None
    for _ in range(n):
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    arr = np.array(times)
    print(f"  {label:30s}  mean={arr.mean():8.3f} ms  "
          f"min={arr.min():8.3f}  max={arr.max():8.3f}  "
          f"total={arr.sum():8.1f} ms ({n} calls)")
    return result, arr.mean()

# 1. Unpack params
d_result, t_unpack = timeit("unpack_params",
    lambda: unpack_params(x0, use_mist=use_mist,
                          mstar_fixed=mstar_prior, age_prior=age_prior,
                          priors=priors))
d = d_result

# 2. Check bounds
_, t_bounds = timeit("check_bounds",
    lambda: check_bounds(d))

# 3. Derived quantities
def _derive():
    dd = d.copy()
    compute_derived(dd, e, omega)
    return dd
d_derived, t_derived = timeit("compute_derived", _derive)
d = d_derived  # use this from now on

# 4. MIST (skip if use_mist=False)
if use_mist:
    _, t_mist = timeit("chi2_mist", lambda: chi2_mist(d))
else:
    t_mist = 0.0
    print(f"  {'chi2_mist (skipped)':30s}  use_mist=False")

# 5. SED
_, t_sed = timeit("chi2_sed",
    lambda: chi2_sed(d, sedfile, sed_data=sed_data))

# 6. Transit
_, t_transit = timeit("chi2_transit",
    lambda: chi2_transit(d, tran_data, e, omega))

# 7. RV
_, t_rv = timeit("chi2_rv",
    lambda: chi2_rv(d, rv_data, e, omega))

# 8. Priors
_, t_priors = timeit("chi2_priors",
    lambda: chi2_priors(d, priors, use_mist))

# 9. Full joint_chi2 (end-to-end)
_, t_joint = timeit("joint_chi2 (end-to-end)",
    lambda: joint_chi2(x0, tran_data, rv_data, sedfile, priors,
                       e, omega, use_mist=use_mist,
                       mstar_fixed=mstar_prior, age_prior=age_prior,
                       sed_data=sed_data))

# ── Summary ──
print("\n" + "=" * 60)
print("SUMMARY (mean per call, caches warm)")
print("=" * 60)
components = [
    ("unpack_params",    t_unpack),
    ("check_bounds",     t_bounds),
    ("compute_derived",  t_derived),
    ("chi2_mist",        t_mist),
    ("chi2_sed",         t_sed),
    ("chi2_transit",     t_transit),
    ("chi2_rv",          t_rv),
    ("chi2_priors",      t_priors),
]
total_parts = sum(t for _, t in components)

for name, t in components:
    pct = 100.0 * t / total_parts if total_parts > 0 else 0
    bar = '#' * int(pct / 2)
    print(f"  {name:20s} {t:8.3f} ms  ({pct:5.1f}%)  {bar}")

print(f"  {'':20s} {'─'*8}")
print(f"  {'SUM of parts':20s} {total_parts:8.3f} ms")
print(f"  {'joint_chi2 measured':20s} {t_joint:8.3f} ms")
print()

# ── Estimate MCMC time ──
nsteps = 2000
ndim = len(x0)
nchains = max(2 * ndim, 3)
calls_per_step = nchains  # one chi2 eval per chain per step
total_calls = nsteps * calls_per_step
est_time_sec = total_calls * t_joint / 1000
print(f"Estimated MCMC time ({nsteps} steps, {nchains} chains):")
print(f"  {total_calls} chi2 evaluations")
print(f"  {est_time_sec:.1f} sec = {est_time_sec/60:.1f} min = {est_time_sec/3600:.2f} hr")
