"""
Detailed profiling of chi2_sed / mistmultised internals.
"""
import time
import numpy as np
import os, sys, pathlib, functools

sys.path.insert(0, os.path.dirname(__file__))

from scipy.io import readsav

# ── paths ──
base = os.path.join(os.path.dirname(__file__),
                    'exozippy', 'data', 'exofastv2', 'examples', 'hat3') + '/'
sedfile = base + 'HAT-3.sed'

import exozippy
from exozippy.sed.utils import (
    read_sed_file, _load_mist_grid, _load_bc_cube,
    get_grid_point, ninterpolate, mistmultised,
)

# Parameters for SED evaluation
teff, logg, feh, av, distance, lstar = 5224.0, 4.56, 0.41, 0.01, 134.5, 0.56

N = 20

def timeit(label, func, n=N):
    times = []
    result = None
    for _ in range(n):
        t0 = time.perf_counter()
        result = func()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
    arr = np.array(times)
    print(f"  {label:40s}  mean={arr.mean():8.3f} ms  "
          f"min={arr.min():8.3f}  max={arr.max():8.3f}")
    return result, arr.mean()

print("=" * 70)
print("SED chi2 internals profiling")
print("=" * 70)

# 1. read_sed_file (one-time I/O cost)
nstars = 1
sed_result, t_read = timeit("read_sed_file (file I/O)",
    lambda: read_sed_file(sedfile, nstars))
sed_data = sed_result
sedbands = sed_data['sedbands']
nbands = len(sedbands)
print(f"    -> {nbands} bands: {list(sedbands)}")

# 1b. mistmultised with cached sed_data
# Warm-up (numba compile, cache build)
mistmultised(teff, logg, feh, av, distance, lstar, 1.0, sedfile, sed_data=sed_data)
_, t_mist_cached_sed = timeit("mistmultised (sed_data cached)",
    lambda: mistmultised(
        teff, logg, feh, av, distance, lstar, 1.0, sedfile,
        sed_data=sed_data
    ))

# 2. Load MIST grid
root = pathlib.Path(exozippy.MODULE_PATH) / 'EXOZIPPy' / 'exozippy' / 'sed' / 'mist'
gridfile = root / 'mist.sed.grid.idl'

# Clear cache, time fresh load
_load_mist_grid.cache_clear()
t0 = time.perf_counter()
teffgrid, logggrid, fehgrid, avgrid = _load_mist_grid(str(gridfile))
t1 = time.perf_counter()
print(f"  {'_load_mist_grid (cold)':40s}  {1000*(t1-t0):.3f} ms")

# Cached load
_, t_grid_cached = timeit("_load_mist_grid (cached)",
    lambda: _load_mist_grid(str(gridfile)))

# 3. filternames2.txt load
kname_result, t_filter_names = timeit("np.loadtxt filternames2.txt",
    lambda: np.loadtxt(root / 'filternames2.txt', dtype=str, comments="#", unpack=True))
kname, mname, cname, svoname = kname_result

# 4. Load BC cubes
_load_bc_cube.cache_clear()
print(f"\n  Loading {nbands} BC cubes (cold):")
bc_cubes_cold = []
for band in sedbands:
    candidates = [band]
    if band in kname:
        candidates.append(mname[np.where(kname == band)[0][0]])
    if band in svoname:
        candidates.append(mname[np.where(svoname == band)[0][0]])
    for cand in candidates:
        bc_path = root / f"{cand}.idl"
        if bc_path.exists():
            t0 = time.perf_counter()
            bc, props = _load_bc_cube(str(bc_path))
            t1 = time.perf_counter()
            bc = np.transpose(bc, (3, 2, 1, 0))
            bc_cubes_cold.append(bc)
            print(f"    {band:15s} -> {cand}.idl   {1000*(t1-t0):.1f} ms  shape={bc.shape}")
            break

# Cached
bc_cubes = []
t_bc_total = 0
for band in sedbands:
    candidates = [band]
    if band in kname:
        candidates.append(mname[np.where(kname == band)[0][0]])
    if band in svoname:
        candidates.append(mname[np.where(svoname == band)[0][0]])
    for cand in candidates:
        bc_path = root / f"{cand}.idl"
        if bc_path.exists():
            t0 = time.perf_counter()
            bc, props = _load_bc_cube(str(bc_path))
            t1 = time.perf_counter()
            bc = np.transpose(bc, (3, 2, 1, 0))
            bc_cubes.append(bc)
            t_bc_total += (t1 - t0) * 1000
            break
print(f"  {'_load_bc_cube all bands (cached)':40s}  total={t_bc_total:.3f} ms")

bcarrays = np.stack(bc_cubes, axis=-1)

# 5. Grid point interpolation
def _do_interpolation():
    bcs = np.empty((nbands, nstars))
    coord = [get_grid_point(g, v) for g, v in
             ((teffgrid, teff), (logggrid, logg),
              (fehgrid, feh), (avgrid, av))]
    for i in range(nbands):
        bcs[i, 0] = ninterpolate(bcarrays[..., i], coord)
    return bcs

_, t_interp = timeit("Grid interpolation (all bands)", _do_interpolation)

# 5a. get_grid_point alone
_, t_gp = timeit("get_grid_point (4 calls)",
    lambda: [get_grid_point(g, v) for g, v in
             ((teffgrid, teff), (logggrid, logg),
              (fehgrid, feh), (avgrid, av))])

# 5b. ninterpolate alone
coord = [get_grid_point(g, v) for g, v in
         ((teffgrid, teff), (logggrid, logg),
          (fehgrid, feh), (avgrid, av))]
_, t_ninterp = timeit(f"ninterpolate ({nbands} bands)",
    lambda: [ninterpolate(bcarrays[..., i], coord) for i in range(nbands)])

# 6. Model magnitudes & chi2 (trivial)
mags = sed_data['mag']
errs = sed_data['errmag']
blend = sed_data['blend']

def _model_and_chi2():
    bcs = _do_interpolation()
    mu = 5.0 * np.log10(distance) - 5.0
    logL = -2.5 * np.log10(lstar)
    modelmag = logL + 4.74 - bcs[:, 0] + mu
    magresid = mags - modelmag
    sigma = errs * 1.0
    return np.sum(magresid**2 / sigma**2 + np.log(2.0 * np.pi * sigma**2))

_, t_model_chi2 = timeit("Model mags + chi2 (with interp)", _model_and_chi2)

# ── Summary ──
print("\n" + "=" * 70)
print("SUMMARY: SED chi2 per-call breakdown")
print("=" * 70)
components = [
    ("read_sed_file",           t_read),
    ("mistmultised (sed_data)", t_mist_cached_sed),
    ("filternames2.txt",        t_filter_names),
    ("_load_mist_grid (cached)",t_grid_cached),
    ("_load_bc_cube (cached)",  t_bc_total / N),  # normalized
    ("get_grid_point (4D)",     t_gp),
    ("ninterpolate (N bands)",  t_ninterp),
    ("model mags + chi2 arith", t_model_chi2 - t_interp),  # subtract interp
]
total = sum(t for _, t in components)
for name, t in components:
    pct = 100 * t / total if total > 0 else 0
    bar = '#' * int(pct / 2)
    print(f"  {name:35s} {t:8.3f} ms  ({pct:5.1f}%)  {bar}")
print(f"  {'SUM':35s} {total:8.3f} ms")
