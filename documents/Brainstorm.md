# Brainstorming Ideas for EXOZIPPy

## Short-term Goal: SED + Transit + RV Global Fitting

Priority is to implement a working end-to-end pipeline for simultaneous
SED + Transit + RV fitting. Microlensing is out of scope for this effort.

### Key milestones:
1. SED model: MIST isochrone-based stellar characterization (Teff, logg, [Fe/H], R*, M*, distance, Av)
2. Transit model: light curve fitting with limb darkening, per-band detrending
3. RV model: Keplerian orbit fitting (K, e, omega, period, gamma, jitter)
4. Joint likelihood: combine SED + Transit + RV into a single chi-squared / log-likelihood
5. MCMC sampling: run DEMC-PT or PyMC NUTS on the combined model
6. Output: posterior summaries, LaTeX table, diagnostic plots

### Reference IDL source

EXOFASTv2 (IDL) local path: `/Users/wangxianyu/Applications/NV5/idl90/lib/EXOFASTv2`

### Testing strategy

Use IDL EXOFASTv2 as ground truth: run IDL with known inputs, capture intermediate
and final outputs, then verify the Python implementation produces matching results.
This applies to each module independently (SED, Transit, RV) and the combined fit.

### Current status / blockers:
- Three build_model variants exist; need to converge on one working path
- Several modules are stubs (rossiter, event, model classes)
- Typos in star.py (e.g. `self.mself.value`)
- Limb darkening Claret priors not yet wired in
- Need a working end-to-end test case (e.g. HAT-3b data in data/exofastv2/)

---

## Microlensing Events w/RV Data

JDE is focused on getting RV working. JCY is focused on getting MM working. 
Eventually, the goal is to integrate them into the same, universal planet-fitter.
The intersection of these efforts is microlensing events with RV data. There are
two. JCY published one and JP Beaulieu published the other.

## Microlensing Events w/Eclipses/Transits?

Low-priority because we can't think of an actual science case. It's just a fun
edge-case.

There are microlensing lightcurves that also show eclipses (for an eclipsing 
binary). JCY thinks the example she knows of (KB23-2027) has an eclipsing 
binary source.

