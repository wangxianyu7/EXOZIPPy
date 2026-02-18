"""
Doppler Tomography (DT) model
==============================
Implements the Beatty (2015) / EXOFASTv2 ``dopptom_chi2`` algorithm.

The model builds a 2-D CCF (time × velocity) by computing, at each
observed epoch, the semi-circular rotation profile of the stellar-disk
patch occulted by the planet.  That profile is convolved with a Gaussian
(instrument resolution + intrinsic line broadening) and normalized so that
its integral equals the fractional depth ``β = 1 − flux``.

Reference: Beatty et al. 2017 (EXOFASTv2 documentation) / Thomas G. Beatty
Ported from EXOFASTv2 IDL ``dopptom_chi2.pro``.
"""

import os
import numpy as np
from scipy.ndimage import gaussian_filter1d
from astropy.io import fits

from .exozippy_keplereq import exozippy_keplereq
from .exozippy_tran import exozippy_tran

C_LIGHT    = 299792.458                    # km/s
FWHM2SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


# ── Data I/O ──────────────────────────────────────────────────────────

def read_dt_fits(filename):
    """Read a Doppler Tomography FITS file (EXOFASTv2 format).

    Expected structure
    ------------------
    HDU 0 — primary : float64 array, shape ``(ntime, nvel)``
        Residual 2-D CCF (median-baseline-subtracted).
    HDU 1 — image   : float64 array, shape ``(ntime,)``
        BJD mid-exposure times.
    HDU 2 — image   : float64 array, shape ``(nvel,)``
        Velocity axis (km/s).

    The spectrograph resolving power ``Rspec`` is encoded in the filename
    as the 4th dot-separated field, e.g.
    ``n20160226.KELT-17b.TRES.44000.fits`` → ``Rspec = 44000``.

    Returns
    -------
    dict with keys:

    * ``ccf2d``    — (ntime, nvel) float64 array
    * ``bjd``      — (ntime,) float64 array
    * ``vel``      — (nvel,) float64 array in km/s
    * ``rms``      — scalar, per-pixel noise estimate
    * ``Rspec``    — scalar, spectrograph resolving power
    * ``label``    — human-readable label string
    * ``filename`` — original path
    """
    basename = os.path.basename(filename)
    with fits.open(filename) as f:
        ccf2d = np.asarray(f[0].data, dtype=float)   # (ntime, nvel)
        bjd   = np.asarray(f[1].data, dtype=float)   # (ntime,)
        vel   = np.asarray(f[2].data, dtype=float)   # (nvel,) km/s

    # Parse Rspec from filename (4th dot-separated token)
    parts = basename.split('.')
    try:
        Rspec = float(parts[3])
        if Rspec == 0:
            raise ValueError
    except (IndexError, ValueError):
        raise ValueError(
            f"DT filename '{basename}' does not encode instrument resolution. "
            "Expected format: 'nYYYYMMDD.PLANET.INSTRUMENT.RSPEC.fits'")

    # Per-pixel RMS noise (same approach as IDL exofast_readdt.pro)
    rms0      = float(np.std(ccf2d))
    N         = ccf2d.size
    med_ccf   = float(np.median(ccf2d))
    errfactor = float(np.sqrt(np.sum(((ccf2d - med_ccf) / rms0)**2) / max(N - 3, 1)))
    rms       = rms0 * errfactor

    # Human-readable label
    night = f'UT {basename[1:5]}-{basename[5:7]}-{basename[7:9]}'
    try:
        telescope = parts[2]
    except IndexError:
        telescope = 'unknown'
    label = f'{night} {telescope}'

    return dict(
        ccf2d    = ccf2d,
        bjd      = bjd,
        vel      = vel,
        rms      = rms,
        Rspec    = Rspec,
        label    = label,
        filename = filename,
    )


# ── Planet position ───────────────────────────────────────────────────

def _planet_up(bjd, tp, period, e, omega, ar, inc, lam):
    """Sub-planet projected velocity in units of vsini (= x in R* units).

    Replicates the geometry in ``exozippy_rossiter._planet_xy`` but
    returns only ``(up, z)`` needed by the DT model.

    Returns
    -------
    up : ndarray
        Planet position along the stellar rotation axis, in R* units.
        ``v_sub = vsini * up`` (km/s).
    z : ndarray
        Line-of-sight coordinate (z > 0: planet in front of star).
    """
    bjd = np.atleast_1d(bjd)
    meananom = 2.0 * np.pi * (1.0 + np.mod((bjd - tp) / period, 1.0))
    if e > 0:
        eccanom  = exozippy_keplereq(meananom, e)
        trueanom = 2.0 * np.arctan(
            np.sqrt((1.0 + e) / (1.0 - e)) * np.tan(eccanom / 2.0))
    else:
        trueanom = meananom

    r     = ar * (1.0 - e**2) / (1.0 + e * np.cos(trueanom))
    x_old = -r * np.cos(trueanom + omega)
    y_old = -r * np.sin(trueanom + omega) * np.cos(inc)
    z     =  r * np.sin(trueanom + omega) * np.sin(inc)

    cos_lam = np.cos(lam)
    sin_lam = np.sin(lam)
    up = x_old * cos_lam - y_old * sin_lam
    return up, z


# ── Core model ────────────────────────────────────────────────────────

def compute_dt_model(dt_data, tp, period, e, omega, inc, ar, p,
                     lam, vsini_kms, vline_kms, u1, u2):
    """Compute the 2-D DT model CCF.

    Parameters
    ----------
    dt_data : dict
        Output of :func:`read_dt_fits`.
    tp : float
        Time of periastron (BJD).
    period : float
        Orbital period (days).
    e, omega : float
        Eccentricity and argument of periastron (rad).
    inc : float
        Orbital inclination (rad).
    ar : float
        Scaled semi-major axis *a/R★*.
    p : float
        Planet-to-star radius ratio *Rp/R★*.
    lam : float
        Projected spin-orbit angle λ (rad).
    vsini_kms : float
        Projected stellar rotation velocity (km/s).
    vline_kms : float
        Intrinsic spectral line broadening sigma (km/s).
    u1, u2 : float
        Quadratic limb-darkening coefficients.

    Returns
    -------
    model : ndarray
        Model 2-D CCF array, shape ``(ntime, nvel)``.
    """
    ccf2d = dt_data['ccf2d']   # (ntime, nvel)
    bjd   = dt_data['bjd']     # (ntime,)
    vel   = dt_data['vel']     # (nvel,) km/s
    Rspec = dt_data['Rspec']

    ntime, nvel = ccf2d.shape

    # Velocities and step sizes in vsini units
    velsini = vel / vsini_kms
    if nvel > 1:
        dv_arr = np.diff(vel)
        step_kms = np.append(dv_arr, dv_arr[-1])
    else:
        step_kms = np.array([0.1])
    step_sini   = step_kms / vsini_kms
    meanstep    = float(np.mean(step_sini[:-1]))   # mean over interior steps

    # Instrument resolution → velocity FWHM
    rvel      = C_LIGHT / Rspec            # km/s FWHM
    # Total broadening Gaussian sigma (instrument + intrinsic line)
    GaussTerm  = np.sqrt(vline_kms**2 + rvel**2) / FWHM2SIGMA  # sigma in km/s
    GaussRel   = GaussTerm / vsini_kms     # sigma in vsini units
    sigma_pix  = GaussRel / meanstep       # sigma in pixel units (for gaussian_filter1d)

    # Only compute the shadow at relevant velocities (|v/vsini| ≤ 2000)
    rel = np.where(np.abs(velsini) <= 2000.0)[0]
    velsini_rel = velsini[rel]
    step_rel    = step_sini[rel]

    # Sub-planet velocity (vsini units) and z-coordinate
    up, z_arr = _planet_up(bjd, tp, period, e, omega, ar, inc, lam)

    # Transit depth: beta[i] = fraction of light blocked
    flux = exozippy_tran(bjd, inc, ar, tp, period, e, omega, abs(p), u1, u2, 1.0)
    beta = 1.0 - flux

    # Rotation profile half-width = p = Rp/R★ (in vsini units)
    velwidth = abs(p)
    c1 = 2.0 / (np.pi * velwidth)

    # Build model: start from median CCF (baseline)
    model = np.full_like(ccf2d, float(np.median(ccf2d)))

    nrel = len(rel)
    for i in range(ntime):
        if beta[i] <= 0.0 or z_arr[i] <= 0.0:
            continue
        up_i = float(up[i])
        c2   = ((velsini_rel - up_i) / velwidth)**2
        valid = np.where(c2 < 1.0)[0]
        if len(valid) == 0:
            continue

        rotprofile = np.zeros(nrel)
        rotprofile[valid] = c1 * np.sqrt(1.0 - c2[valid])

        # Gaussian convolution (instrument + line broadening)
        if sigma_pix > 0.1:
            unnorm = gaussian_filter1d(rotprofile, sigma=sigma_pix)
        else:
            unnorm = rotprofile

        total = float(np.sum(unnorm * step_rel))
        if total <= 0.0:
            continue

        model[i, rel] += beta[i] / total * unnorm

    return model


# ── Chi2 ─────────────────────────────────────────────────────────────

def chi2_dopptom(dt_data, tp, period, e, omega, inc, ar, p,
                 lam, vsini_kms, vline_kms, u1, u2, errscale=1.0):
    """Doppler Tomography chi2 (scalar).

    chi2 = Σ (resid / (rms * errscale))² / IndepVels

    where ``IndepVels`` corrects for spectral oversampling:
    ``IndepVels = (rvel/FWHM2SIGMA) / (meanstep * vsini_kms)``.

    Parameters
    ----------
    dt_data : dict
        Output of :func:`read_dt_fits`.
    errscale : float, optional
        Multiplicative error scale (default 1.0).  Set > 1 if the
        per-pixel RMS underestimates the true noise.

    Returns
    -------
    float
        Chi2 value (or ``np.inf`` for invalid parameters).
    """
    if vsini_kms <= 0 or vline_kms <= 0 or errscale <= 0:
        return np.inf

    vel   = dt_data['vel']
    Rspec = dt_data['Rspec']
    rms   = dt_data['rms']

    nvel = len(vel)
    if nvel > 1:
        dv_arr = np.diff(vel)
        step_kms  = np.append(dv_arr, dv_arr[-1])
    else:
        step_kms = np.array([0.1])
    meanstep = float(np.mean((step_kms / vsini_kms)[:-1]))

    rvel      = C_LIGHT / Rspec
    IndepVels = (rvel / FWHM2SIGMA) / (meanstep * vsini_kms)

    model = compute_dt_model(dt_data, tp, period, e, omega, inc, ar, p,
                             lam, vsini_kms, vline_kms, u1, u2)
    resid = dt_data['ccf2d'] - model
    chi2  = float(np.sum((resid / (rms * errscale))**2)) / IndepVels

    # Log-prior on errscale (Jeffrey's prior: uniform in log)
    # Prevents errscale → ∞ when it is fitted.
    if errscale != 1.0:
        Neff = dt_data['ccf2d'].size / IndepVels
        chi2 += 2.0 * Neff * np.log(errscale)

    return chi2
