import numpy as np
from pathlib import Path
from exozippy.mist.utils import readeep

# Persistent track storage (mass x feh x vvcrit x alpha)
tracks = None
ALLOWED_MASS = np.array([
    0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
    0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 1.00, 1.02, 1.04,
    1.06, 1.08, 1.10, 1.12, 1.14, 1.16, 1.18, 1.20, 1.22, 1.24, 1.26, 1.28,
    1.30, 1.32, 1.34, 1.36, 1.38, 1.40, 1.42, 1.44, 1.46, 1.48, 1.50, 1.52,
    1.54, 1.56, 1.58, 1.60, 1.62, 1.64, 1.66, 1.68, 1.70, 1.72, 1.74, 1.76,
    1.78, 1.80, 1.82, 1.84, 1.86, 1.88, 1.90, 1.92, 1.94, 1.96, 1.98, 2.00,
    2.02, 2.04, 2.06, 2.08, 2.10, 2.12, 2.14, 2.16, 2.18, 2.20, 2.22, 2.24,
    2.26, 2.28, 2.30, 2.32, 2.34, 2.36, 2.38, 2.40, 2.42, 2.44, 2.46, 2.48,
    2.50, 2.52, 2.54, 2.56, 2.58, 2.60, 2.62, 2.64, 2.66, 2.68, 2.70, 2.72,
    2.74, 2.76, 2.78, 2.80, 3.00, 3.20, 3.40, 3.60, 3.80, 4.00, 4.20, 4.40,
    4.60, 4.80, 5.00, 5.20, 5.40, 5.60, 5.80, 6.00, 6.20, 6.40, 6.60, 6.80,
    7.00, 7.20, 7.40, 7.60, 7.80, 8.00, 9.00, 10.00, 11.00, 12.00, 13.00,
    14.00, 15.00, 16.00, 17.00, 18.00, 19.00, 20.00, 22.00, 24.00, 26.00,
    28.00, 30.00, 32.00, 34.00, 36.00, 38.00, 40.00, 45.00, 50.00, 55.00,
    60.00, 65.00, 70.00, 75.00, 80.00, 85.00, 90.00, 95.00, 100.00, 105.00,
    110.00, 115.00, 120.00, 125.00, 130.00, 135.00, 140.00, 145.00, 150.00,
    175.00, 200.00, 225.00, 250.00, 275.00, 300.00
])
ALLOWED_INITFEH = np.array([
    -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.25, -1.0, -0.75, -0.5,
    -0.25, 0.0, 0.25, 0.5
])
ALLOWED_VVCRIT = np.array([0.0, 0.4])
ALLOWED_ALPHA = np.array([0.0])


def _init_tracks():
    """Allocate the cache array on first use."""
    global tracks
    if tracks is None:
        tracks = np.empty(
            (len(ALLOWED_MASS), len(ALLOWED_INITFEH),
             len(ALLOWED_VVCRIT), len(ALLOWED_ALPHA)),
            dtype=object,
        )
        tracks.fill(None)


def _mass_index(mstar):
    return int(np.argmin(np.abs(ALLOWED_MASS - mstar)))


def _feh_index(feh):
    return int(np.argmin(np.abs(ALLOWED_INITFEH - feh)))


def _vvcrit_index(vvcrit):
    matches = np.where(np.isclose(ALLOWED_VVCRIT, vvcrit))[0]
    if len(matches) == 0:
        raise ValueError(f"vvcrit ({vvcrit}) not allowed")
    return int(matches[0])


def _alpha_index(alpha):
    matches = np.where(np.isclose(ALLOWED_ALPHA, alpha))[0]
    if len(matches) == 0:
        raise ValueError(f"alpha ({alpha}) not allowed")
    return int(matches[0])


def _get_track_tuple_cached(mass_idx, feh_idx, vvcrit_idx, alpha_idx):
    """Return (ages, rstars, teffs, fehs, ageweights) for a cached grid point."""
    _init_tracks()
    global tracks
    if tracks[mass_idx, feh_idx, vvcrit_idx, alpha_idx] is None:
        tracks[mass_idx, feh_idx, vvcrit_idx, alpha_idx] = _load_track_tuple(
            ALLOWED_MASS[mass_idx],
            ALLOWED_INITFEH[feh_idx],
            ALLOWED_VVCRIT[vvcrit_idx],
            ALLOWED_ALPHA[alpha_idx],
        )
    return tracks[mass_idx, feh_idx, vvcrit_idx, alpha_idx]


def _load_track_tuple(mstar, feh, vvcrit, alpha):
    """Return (ages, rstars, teffs, fehs, ageweights) arrays for a grid point."""
    track = readeep(mstar, feh, vvcrit=vvcrit, alpha=alpha)
    if isinstance(track, dict):
        track_arr = track['track']
    else:
        track_arr = np.asarray(track)
    if track_arr.ndim != 2 or (5 not in track_arr.shape):
        raise ValueError("Unexpected track array shape")
    if track_arr.shape[0] != 5:
        track_arr = track_arr.T
    if track_arr.shape[0] != 5:
        raise ValueError("Track array has invalid orientation")
    ages, rstars, teffs, fehs, ageweights = track_arr
    return ages.astype(float), rstars.astype(float), teffs.astype(float), fehs.astype(float), ageweights.astype(float)

def massradius_mist(mstar, feh, age, teff, rstar, vvcrit=None, alpha=None, span=1, epsname=None, debug=False,
                    gravitysun=27420.011, fitage=False, ageweight=None, verbose=False, logname=None, 
                    trackfile=None, allowold=False, tefffloor=None, fehfloor=None, rstarfloor=None, 
                    agefloor=None, pngname=None, range=None):
    '''
    ;+
; NAME:
;   massradius_mist
;
; PURPOSE: 
;   Interpolate the MIST stellar evolutionary models to derive Teff
;   and Rstar from mass, metallicity, and age. Intended to be a drop in
;   replacement for the Yonsie Yale model interpolation
;   (massradius_yy3.pro).
;
; CALLING SEQUENCE:
;   chi2 = massradius_mist(mstar, feh, age, teff, rstar, $
;                          VVCRIT=vvcrit, ALPHA=alpha, SPAN=span,$
;                          MISTRSTAR=mistrstar, MISTTEFF=mistteff)
; INPUTS:
;
;    MSTAR  - The mass of the star, in m_sun
;    FEH    - The metallicity of the star [Fe/H]
;    AGE    - The age of the star, in Gyr
;    RSTAR  - The radius you expect; used to calculate a chi^2
;    TEFF   - The Teff you expect; used to calculate a chi^2
;    
; OPTIONAL INPUTS:
;   VVCRIT    - The rotational velocity normalized by the critical
;               rotation speed. Must be 0.0d0 or 0.4d0 (default 0.0d0).
;   ALPHA     - The alpha abundance. Must be 0.0 (default 0.0). A
;               placeholder for future improvements to MIST models.
;   SPAN      - The interpolation is done at the closest value +/-
;               SPAN grid points in the evolutionary tracks in mass,
;               age, metallicity. The larger this number, the longer it
;               takes. Default=1. Change with care.
;   EPSNAME   - A string specifying the name of postscript file to plot
;               the evolutionary track. If not specified, no plot is
;               generated.
;
; OPTIONAL KEYWORDS:
;   DEBUG     - If set, will plot the teff and rstar over the MIST
;               Isochrone.
;
; OPTIONAL OUTPUTS:
;   MISTRSTAR - The rstar interpolated from the MIST models.
;   MISTTEFF  - The Teff interpolated from the MIST models.
;
; RESULT:
;   The chi^2 penalty due to the departure from the MIST models,
;   assuming 3% errors in the MIST model values.
;
; COMMON BLOCKS:
;   MIST_BLOCK:
;     Loading EEPs (model tracks) is very slow. This common block
;     allows us to store the tracks in memory between calls. The first
;     call will take ~3 seconds. Subsequent calls that use the same
;     EEP files take 1 ms.
;
; EXAMPLE: 
;   ;; penalize a model for straying from the MIST models 
;   chi2 += massradius_mist(mstar, feh, age, rstar=rstar, teff=teff)
;
; MODIFICATION HISTORY
; 
;  2018/01 -- Written, JDE
;-
    '''
    global tracks

    if tefffloor is None:
        tefffloor = -1
    if fehfloor is None:
        fehfloor = -1
    if rstarfloor is None:
        rstarfloor = -1
    if agefloor is None:
        agefloor = -1

    if not (ALLOWED_MASS.min() <= mstar <= ALLOWED_MASS.max()):
        if verbose:
            print(f"Mstar ({mstar}) is out of range [0.1, 300]", file=logname or None)
        return np.inf

    if not (ALLOWED_INITFEH.min() <= feh <= ALLOWED_INITFEH.max()):
        if verbose:
            print(f"initfeh ({feh}) is out of range [-4, 0.5]", file=logname or None)
        return np.inf

    try:
        massndx = _mass_index(mstar)
        fehndx = _feh_index(feh)
        vvcritndx = 0 if vvcrit is None else _vvcrit_index(vvcrit)
        alphandx = 0 if alpha is None else _alpha_index(alpha)
    except ValueError as exc:
        if verbose:
            print(str(exc), file=logname or None)
        return np.inf

    ages, rstars, teffs, fehs, ageweights_data = _get_track_tuple_cached(
        massndx, fehndx, vvcritndx, alphandx
    )

    eep = np.searchsorted(ages, age)
    if eep == len(ages):
        eep -= 1
    if eep < 1:
        if verbose:
            print(f"EEP ({eep}) is out of range [1, ∞]", file=logname or None)
        return np.inf

    neep = len(ages)
    if eep >= neep:
        if verbose:
            print(f"EEP ({eep}) is out of bounds for track with {neep} points", file=logname or None)
        return np.inf

    # Interpolation using two closest ages
    x_eep = (age - ages[eep - 1]) / (ages[eep] - ages[eep - 1]) if ages[eep] != ages[eep - 1] else 0.0
    mistage = (1 - x_eep) * ages[eep - 1] + x_eep * ages[eep]
    mistrstar = (1 - x_eep) * rstars[eep - 1] + x_eep * rstars[eep]
    mistteff = (1 - x_eep) * teffs[eep - 1] + x_eep * teffs[eep]
    mistfeh = (1 - x_eep) * fehs[eep - 1] + x_eep * fehs[eep]
    ageweight_interp = (1 - x_eep) * ageweights_data[eep - 1] + x_eep * ageweights_data[eep]

    if mistage < 0 or (not allowold and mistage > 13.82):
        if verbose:
            print(f"Age ({mistage}) is out of range", file=logname or None)
        return np.inf

    percenterror = 0.03 - 0.025 * np.log10(mstar) + 0.045 * (np.log10(mstar))**2

    chi2_rstar = ((mistrstar - rstar) / (rstarfloor * mistrstar if rstarfloor > 0 else percenterror * mistrstar))**2
    chi2_teff = ((mistteff - teff) / (tefffloor * mistteff if tefffloor > 0 else percenterror * mistteff))**2
    chi2_feh = ((mistfeh - feh) / (fehfloor if fehfloor > 0 else percenterror))**2
    chi2_age = ((mistage - age) / (agefloor * mistage if agefloor > 0 else percenterror * mistage))**2

    chi2 = chi2_rstar + chi2_teff + chi2_feh + chi2_age

    if trackfile:
        _write_track_file(trackfile, teffs, rstars, ages)

    plot_target = pngname or epsname
    if debug and not plot_target:
        plot_target = Path.cwd() / "mist_track_debug.png"
    if plot_target:
        _plot_mist_track(
            teffs,
            rstars,
            ages,
            mstar,
            feh,
            age,
            teff,
            rstar,
            gravitysun,
            eep_index=eep,
            outfile=str(plot_target),
            range_vals=range,
        )
    return chi2


def _write_track_file(path, teffs, rstars, ages):
    path = Path(path)
    data = np.column_stack((teffs, rstars, ages))
    header = "teff[K] rstar[Rsun] age[Gyr]"
    np.savetxt(path, data, header=header)


def _plot_mist_track(teffs, rstars, ages, mstar, feh, age, teff, rstar,
                     gravitysun, eep_index, outfile=None, range_vals=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    teffs = np.asarray(teffs, dtype=float)
    rstars = np.asarray(rstars, dtype=float)
    ages = np.asarray(ages, dtype=float)

    safe_rstars = np.clip(rstars, 1e-6, None)
    track_logg = np.log10((mstar / (safe_rstars**2)) * gravitysun)
    finite = np.isfinite(teffs) & np.isfinite(track_logg)
    if not np.any(finite):
        return

    # Filter track: logg in [3, 5] and age <= 14 Gyr (age of universe)
    # Matches IDL: excludes PMS (logg<3), post-AGB/WD (age>14 Gyr), and extreme stages
    use = finite & (track_logg > 3) & (track_logg < 5) & (ages <= 14.0)
    if not np.any(use):
        use = finite  # fallback to full track

    teff_vals = teffs[use]
    logg_vals = track_logg[use]

    logg_best = np.log10((mstar / (max(rstar, 1e-6)**2)) * gravitysun)
    eep_index = int(np.clip(eep_index, 0, len(teffs) - 1))

    fig, ax = plt.subplots(figsize=(6, 5.5))
    ax.plot(teff_vals, logg_vals, color="black", lw=1.5, label="MIST track")
    ax.scatter([teff], [logg_best], color="tab:red", s=35, zorder=5, label="Best fit")
    ax.scatter(
        [teffs[eep_index]],
        [track_logg[eep_index]],
        marker="s",
        color="tab:blue",
        s=40,
        zorder=6,
        label="Nearest grid point",
    )

    if range_vals and len(range_vals) >= 4:
        xmin, xmax, ymin, ymax = range_vals[:4]
    else:
        # IDL convention: include best-fit point with ±10% padding and track range
        xmin = max(np.max(teff_vals), teff * 1.1, teff * 0.9)
        xmax = min(np.min(teff_vals), teff * 0.9, teff * 1.1)
        # Round to 100 K boundaries
        xmin = int(np.ceil(xmin / 100)) * 100
        xmax = int(np.floor(xmax / 100)) * 100
        # IDL convention: logg range includes best-fit, clamp to [3, 5]
        ymin = max(logg_best, np.max(logg_vals), 3.0)
        ymax = min(logg_best, np.min(logg_vals), 5.0)
        # Add padding
        dy = ymin - ymax
        ypad = max(dy * 0.1, 0.1)
        ymin += ypad
        ymax -= ypad

    # IDL HR diagram convention: both axes inverted
    # xrange = [xmin, xmax] with xmin > xmax (hot on left)
    # yrange = [ymin, ymax] with ymin > ymax (high logg at bottom, low logg at top)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)

    ax.set_xlabel(r"$T_{\mathrm{eff}}$ (K)")
    ax.set_ylabel(r"$\log g_\star$ (cgs)")
    ax.set_title(f"MIST track: M={mstar:.2f} M$_\odot$, [Fe/H]={feh:+.2f}, age={age:.2f} Gyr")
    ax.legend(loc="best")
    ax.grid(alpha=0.3, ls="--")

    outfile = outfile or "mist_track.png"
    fig.tight_layout()
    fig.savefig(outfile, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_mist_track(bestfit, outfile, vvcrit=None, alpha=None, range_vals=None):
    """
    Convenience wrapper to render a Teff-logg plot for a best-fit solution.
    """
    mstar = bestfit['mstar']
    feh = bestfit['feh']
    age = bestfit['age']
    teff = bestfit['teff']
    rstar = bestfit['rstar']

    if not (ALLOWED_MASS.min() <= mstar <= ALLOWED_MASS.max()):
        return
    if not (ALLOWED_INITFEH.min() <= feh <= ALLOWED_INITFEH.max()):
        return

    try:
        massndx = _mass_index(mstar)
        fehndx = _feh_index(feh)
        vvcritndx = 0 if vvcrit is None else _vvcrit_index(vvcrit)
        alphandx = 0 if alpha is None else _alpha_index(alpha)
    except ValueError:
        return

    ages, rstars, teffs, _, _ = _get_track_tuple_cached(massndx, fehndx, vvcritndx, alphandx)
    eep = np.searchsorted(ages, age)
    if eep == len(ages):
        eep -= 1
    _plot_mist_track(
        teffs,
        rstars,
        ages,
        mstar,
        feh,
        age,
        teff,
        rstar,
        27420.011,
        eep_index=eep,
        outfile=outfile,
        range_vals=range_vals,
    )
