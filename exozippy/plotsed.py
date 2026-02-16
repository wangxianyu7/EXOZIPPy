"""
SED plotting for EXOZIPPy.

Mirrors EXOFASTv2 exofast_multised.pro plotting — produces a single figure:
  Top:    log(lambda F_lambda) vs lambda (log scale)
          - Model atmosphere (black line)
          - Model band fluxes (blue circles)
          - Observed fluxes with error bars (red)
  Bottom: O-C residuals in sigma units

Usage:
    from exozippy.plotsed import plotsed
    plotsed(sedfile, bestfit, outfile='sed.png')
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pathlib
import functools

from scipy.io import readsav

import exozippy
from exozippy.sed.utils import (
    read_sed_file, _load_mist_grid, _load_bc_cube,
    get_grid_point, ninterpolate,
)
from exozippy.mkconstants import mkconstants

CONSTANTS = mkconstants()

# Wavelength grid for NextGen models (same as IDL)
WAVELENGTH = np.arange(24000) / 1000.0 + 0.1  # 0.1 to 24 um


# ---------- NextGen model interpolation ----------

# Grid definitions (from exofast_interp_model3d.pro)
ALLOWED_TEFF = np.array([
    8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37,
    38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52,
    53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67,
    68, 69, 70, 72, 74, 76, 78, 80, 82, 84, 86, 88, 90, 92, 94,
    96, 98, 100, 102, 104, 106, 108, 110, 112, 114, 116, 118, 120,
    125, 130, 135, 140, 145, 150, 155, 160, 165, 170, 175, 180, 185,
    190, 195, 200, 210, 220, 230, 240, 250, 260, 270, 280, 290, 300,
    310, 320, 330, 340, 350, 360, 370, 380, 390, 400, 410, 420, 430,
    440, 450, 460, 470, 480, 490, 500, 510, 520, 530, 540, 550, 560,
    570, 580, 590, 600, 610, 620, 630, 640, 650, 660, 670, 680, 690,
    700
]) * 100.0  # Convert to Kelvin

ALLOWED_LOGG = np.array([-0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0])

ALLOWED_FEH = np.array([-4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.3, 0.5])

TEFF_STR = [str(int(t // 100)) for t in ALLOWED_TEFF]
LOGG_STR = ['-0.5', '+0.0', '+0.5', '+1.0', '+1.5', '+2.0', '+2.5', '+3.0', '+3.5', '+4.0', '+4.5', '+5.0', '+5.5', '+6.0']
FEH_STR = ['-4.0', '-3.5', '-3.0', '-2.5', '-2.0', '-1.5', '-1.0', '-0.5', '+0.0', '+0.3', '+0.5']
ALPHA_STR = ['+0.0', '+0.2', '-0.2', '+0.4', '+0.6']


@functools.lru_cache(maxsize=64)
def _load_nextgen_model(filepath):
    """Load and cache a NextGen model spectrum."""
    s = readsav(filepath, python_dict=True)
    return s['lamflam1']


def _interp_atmosphere(teff, logg, feh):
    """
    Interpolate NextGen model atmosphere.

    Returns
    -------
    lamflam : ndarray, shape (24000,)
        lambda * F_lambda on the standard wavelength grid.
        Returns None if interpolation fails.
    """
    # Check bounds
    if teff < 800 or teff > 70000:
        return None
    if logg < ALLOWED_LOGG[0] or logg > ALLOWED_LOGG[-1]:
        return None
    if feh < ALLOWED_FEH[0] or feh > ALLOWED_FEH[-1]:
        return None

    # Find grid indices
    teff_ndx = np.searchsorted(ALLOWED_TEFF, teff) - 1
    teff_ndx = np.clip(teff_ndx, 0, len(ALLOWED_TEFF) - 2)

    logg_ndx = np.searchsorted(ALLOWED_LOGG, logg) - 1
    logg_ndx = np.clip(logg_ndx, 0, len(ALLOWED_LOGG) - 2)

    feh_ndx = np.searchsorted(ALLOWED_FEH, feh) - 1
    feh_ndx = np.clip(feh_ndx, 0, len(ALLOWED_FEH) - 2)

    # Path to NextGen models
    nextgen_path = pathlib.Path(exozippy.NEXTGENFIN_PATH)

    # Load 8 corner models for trilinear interpolation
    lamflams = np.zeros((24000, 2, 2, 2))
    for i in range(2):
        for j in range(2):
            for k in range(2):
                # Try different alpha values
                found = False
                for alpha_str in ALPHA_STR:
                    filename = f"lte{TEFF_STR[teff_ndx + i]}{LOGG_STR[logg_ndx + j]}{FEH_STR[feh_ndx + k]}{alpha_str}.NextGen.spec.idl"
                    filepath = nextgen_path / filename
                    if filepath.exists():
                        lamflams[:, i, j, k] = _load_nextgen_model(str(filepath))
                        found = True
                        break
                if not found:
                    return None

    # Trilinear interpolation
    x_teff = (teff - ALLOWED_TEFF[teff_ndx]) / (ALLOWED_TEFF[teff_ndx + 1] - ALLOWED_TEFF[teff_ndx])
    y_logg = (logg - ALLOWED_LOGG[logg_ndx]) / (ALLOWED_LOGG[logg_ndx + 1] - ALLOWED_LOGG[logg_ndx])
    z_feh = (feh - ALLOWED_FEH[feh_ndx]) / (ALLOWED_FEH[feh_ndx + 1] - ALLOWED_FEH[feh_ndx])

    # Interpolate each wavelength point
    lamflam = np.zeros(24000)
    for w in range(24000):
        cube = lamflams[w, :, :, :]
        # Trilinear interpolation
        c00 = cube[0, 0, 0] * (1 - x_teff) + cube[1, 0, 0] * x_teff
        c01 = cube[0, 0, 1] * (1 - x_teff) + cube[1, 0, 1] * x_teff
        c10 = cube[0, 1, 0] * (1 - x_teff) + cube[1, 1, 0] * x_teff
        c11 = cube[0, 1, 1] * (1 - x_teff) + cube[1, 1, 1] * x_teff
        c0 = c00 * (1 - y_logg) + c10 * y_logg
        c1 = c01 * (1 - y_logg) + c11 * y_logg
        lamflam[w] = c0 * (1 - z_feh) + c1 * z_feh

    return lamflam


def _apply_extinction(lamflam, av):
    """Apply extinction to model atmosphere."""
    # Load extinction law
    ext_file = pathlib.Path(exozippy.MODULE_PATH) / 'sed' / 'extinction_law.ascii'
    if ext_file.exists():
        klam, kkap = np.loadtxt(ext_file, unpack=True)
        kapv = np.interp(0.55, klam, kkap)
        kapp = np.interp(WAVELENGTH, klam, kkap)
        tau = kapp / kapv / 1.086 * av
        extinct = np.exp(-tau)
        return lamflam * extinct
    return lamflam


def _scale_atmosphere(lamflam, rstar, distance):
    """
    Scale model atmosphere to observed flux level using geometric scaling.

    Matches IDL EXOFASTv2 (exofast_chi2v2.pro):
        lamflam_obs = lamflam_surface * (Rstar * Rsun)^2 / (distance * pc)^2

    Parameters
    ----------
    lamflam : ndarray
        Model lambda*F_lambda at the stellar surface.
    rstar : float
        Stellar radius in solar radii.
    distance : float
        Distance in parsecs.
    """
    pc_cm = CONSTANTS['pc']      # cm
    rsun_cm = CONSTANTS['RSun']  # cm

    scale = (rstar * rsun_cm)**2 / (distance * pc_cm)**2
    return lamflam * scale


# ---------- SED model computation ----------

def _compute_sed_model(teff, logg, feh, av, distance, lstar, rstar, sedfile):
    """
    Compute model SED and compare with observations.

    Returns
    -------
    dict with keys:
        weff, widtheff : effective wavelength and bandwidth (um)
        obs_flux, obs_err : observed flux and error (erg/s/cm^2)
        model_flux : model flux at each band (erg/s/cm^2)
        residuals : (obs - model) / err in sigma units
        sedbands : band names
        atmosphere : (wavelength, lamflam) for continuous spectrum
    """
    # Read SED file
    sed_data = read_sed_file(sedfile, nstars=1)
    sedbands = sed_data['sedbands']
    mags = sed_data['mag']
    errmag = sed_data['errmag']
    weff = sed_data['weff']
    widtheff = sed_data['widtheff']
    zero_point = sed_data['zero_point']
    nbands = len(sedbands)

    # Convert observed mag to flux
    obs_flux = zero_point * 10**(-0.4 * mags)
    obs_err = obs_flux * np.log(10) / 2.5 * errmag

    filter_curves = np.asarray(sed_data['filter_curves'], dtype=float)
    filter_curve_sum = np.asarray(sed_data['filter_curve_sum'], dtype=float)

    # Load MIST BC grid for synthetic magnitudes (matches fit_exoplanet)
    root = pathlib.Path(exozippy.MODULE_PATH) / 'sed' / 'mist'
    gridfile = root / 'mist.sed.grid.idl'
    teffgrid, logggrid, fehgrid, avgrid = _load_mist_grid(str(gridfile))

    kname, mname, cname, svoname = np.loadtxt(
        root / 'filternames2.txt', dtype=str, comments="#", unpack=True
    )

    bc_cubes = []
    for band in sedbands:
        candidates = [band]
        if band in kname:
            candidates.append(mname[np.where(kname == band)[0][0]])
        if band in svoname:
            candidates.append(mname[np.where(svoname == band)[0][0]])

        for cand in candidates:
            bc_path = root / f"{cand}.idl"
            if bc_path.exists():
                bc, _ = _load_bc_cube(str(bc_path))
                bc = np.transpose(bc, (3, 2, 1, 0))
                bc_cubes.append(bc)
                break
        else:
            raise FileNotFoundError(f"{band} BC not found")

    bcarrays = np.stack(bc_cubes, axis=-1)
    coord = [get_grid_point(g, v) for g, v in
             ((teffgrid, teff), (logggrid, logg), (fehgrid, feh), (avgrid, av))]
    bcs = np.array([ninterpolate(bcarrays[..., i], coord) for i in range(nbands)])
    mu = 5.0 * np.log10(distance) - 5.0
    logL_term = -2.5 * np.log10(lstar)
    modelmag = logL_term + 4.74 - bcs + mu
    model_flux = zero_point * 10**(-0.4 * modelmag)
    residuals = (obs_flux - model_flux) / obs_err

    # Interpolate model atmosphere for continuous spectrum
    # Matches IDL (exofast_chi2v2.pro):
    #   lamflam = lamflam_surface * (Rstar*Rsun)^2 / (d*pc)^2  (geometric scaling)
    #   then apply extinction
    atmosphere = None
    lamflam = _interp_atmosphere(teff, logg, feh)
    if lamflam is not None:
        lamflam = _scale_atmosphere(lamflam, rstar, distance)
        lamflam = _apply_extinction(lamflam, av)
        atmosphere = (WAVELENGTH, lamflam)

    return {
        'weff': weff,
        'widtheff': widtheff,
        'obs_flux': obs_flux,
        'obs_err': obs_err,
        'model_flux': model_flux,
        'residuals': residuals,
        'sedbands': sedbands,
        'atmosphere': atmosphere,
    }


def _oc_ylim_sed(residuals):
    """Symmetric O-C y-limits rounded to 0.5 sigma."""
    ymax = np.max(np.abs(residuals)) * 1.1
    if ymax == 0:
        return 1.0
    return np.ceil(ymax / 0.5) * 0.5


def plotsed(sedfile, bestfit, outfile=None):
    """
    SED plot — single figure with GridSpec.

    Layout:
        Top (height 3):    log(lambda F_lambda) vs lambda
        Bottom (height 1): O-C residuals (sigma)

    Supports multi-star: overlays each star's atmosphere with different colors.

    Parameters
    ----------
    sedfile : str
        Path to SED data file.
    bestfit : dict or SS
        Best-fit parameter dictionary from fit_exoplanet.
    outfile : str, optional
        Output filename (.png or .pdf).
        If None, displays interactively.

    Returns
    -------
    fig : Figure
    """
    from .ss import SS

    # Determine nstars
    nstars = bestfit.nstars if isinstance(bestfit, SS) else 1

    # Extract primary star parameters (for model band fluxes / residuals)
    teff = bestfit['teff']
    logg = bestfit['logg']
    feh = bestfit['feh']
    av = bestfit['av']
    distance = bestfit['distance']
    lstar = bestfit['lstar']
    rstar = bestfit['rstar']

    # Compute model (using primary star for band fluxes)
    sed = _compute_sed_model(teff, logg, feh, av, distance, lstar, rstar, sedfile)
    weff = sed['weff']
    widtheff = sed['widtheff']
    obs_flux = sed['obs_flux']
    obs_err = sed['obs_err']
    model_flux = sed['model_flux']
    residuals = sed['residuals']
    atmosphere = sed['atmosphere']

    # zero_point * 10^(-0.4*mag) already gives lambda*F_lambda
    # (EXOFASTv2 convention — see IDL exofast_multised.pro)
    obs_lamflam = obs_flux
    obs_lamflam_err = obs_err
    model_lamflam = model_flux

    fig = plt.figure(figsize=(10, 8))

    # Nested GridSpec
    outer = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=(3, 1),
        left=0.15, right=0.95, top=0.95, bottom=0.10, hspace=0.0,
    )

    ax_data = fig.add_subplot(outer[0])
    ax_oc = fig.add_subplot(outer[1], sharex=ax_data)

    # --- Top panel: SED ---
    # Model atmosphere continuous spectrum
    from scipy.ndimage import uniform_filter1d
    _atm_colors = ['black', 'gray', 'steelblue', 'darkgreen', 'purple']

    if nstars > 1:
        # Multi-star: overlay each star's atmosphere
        for i in range(nstars):
            suffix = f'_{i}' if nstars > 1 else ''
            teff_i = bestfit[f'teff{suffix}']
            logg_i = bestfit[f'logg{suffix}']
            feh_i = bestfit[f'feh{suffix}']
            av_i = bestfit[f'av{suffix}']
            dist_i = bestfit[f'distance{suffix}']
            rstar_i = bestfit[f'rstar{suffix}']
            lamflam_i = _interp_atmosphere(teff_i, logg_i, feh_i)
            if lamflam_i is not None:
                lamflam_i = _scale_atmosphere(lamflam_i, rstar_i, dist_i)
                lamflam_i = _apply_extinction(lamflam_i, av_i)
                lamflam_smooth = uniform_filter1d(lamflam_i, size=10)
                mask = lamflam_smooth > 0
                color = _atm_colors[i % len(_atm_colors)]
                label = f'Star {chr(65+i)}' if nstars > 1 else 'Model atmosphere'
                ax_data.plot(WAVELENGTH[mask], np.log10(lamflam_smooth[mask]), '-',
                             color=color, lw=1, zorder=1, label=label)
    elif atmosphere is not None:
        wav, lamflam = atmosphere
        lamflam_smooth = uniform_filter1d(lamflam, size=10)
        mask = lamflam_smooth > 0
        ax_data.plot(wav[mask], np.log10(lamflam_smooth[mask]), '-', color='black',
                     lw=1, zorder=1, label='Model atmosphere')

    # Model band fluxes (blue circles)
    safe_model = np.where(model_lamflam > 0, model_lamflam, np.nan)
    ax_data.plot(weff, np.log10(safe_model), 'o', color='blue',
                 ms=8, mfc='blue', mec='blue', zorder=3, label='Model bands')

    # Observed fluxes (red points with error bars)
    safe_obs = np.where(obs_lamflam > 0, obs_lamflam, np.nan)
    for i in range(len(weff)):
        if obs_lamflam[i] <= 0:
            continue
        # y error bar
        y_lo = np.log10(obs_lamflam[i] - obs_lamflam_err[i]) if obs_lamflam[i] > obs_lamflam_err[i] else np.log10(obs_lamflam[i]) - 0.5
        y_hi = np.log10(obs_lamflam[i] + obs_lamflam_err[i])
        ax_data.plot([weff[i], weff[i]], [y_lo, y_hi], '-', color='red', lw=1.5, zorder=2)
        # x error bar (bandwidth)
        ax_data.plot([weff[i] - widtheff[i]/2, weff[i] + widtheff[i]/2],
                     [np.log10(obs_lamflam[i]), np.log10(obs_lamflam[i])],
                     '-', color='red', lw=1.5, zorder=2)

    ax_data.plot(weff, np.log10(safe_obs), 'o', color='red',
                 ms=6, mfc='red', mec='red', zorder=4, label='Observed')

    ax_data.set_xscale('log')
    ax_data.set_ylabel(r'log $\lambda F_\lambda$ (erg s$^{-1}$ cm$^{-2}$)')
    ax_data.set_xlim(0.3, 30)

    # Set y limits based on data (guard against zero/negative fluxes)
    all_vals = np.concatenate([obs_lamflam, model_lamflam])
    pos_mask = all_vals > 0
    if np.any(pos_mask):
        all_log = np.log10(all_vals[pos_mask])
        ymin = np.min(all_log) - 0.3
        ymax = np.max(all_log) + 0.3
    else:
        ymin, ymax = -12, -8
    ax_data.set_ylim(ymin, ymax)

    ax_data.legend(loc='upper right', frameon=False)
    plt.setp(ax_data.get_xticklabels(), visible=False)

    # --- Bottom panel: O-C residuals ---
    for i in range(len(weff)):
        # x error bar
        ax_oc.plot([weff[i] - widtheff[i]/2, weff[i] + widtheff[i]/2],
                   [residuals[i], residuals[i]], '-', color='red', lw=1.5)
        # y error bar (1 sigma)
        ax_oc.plot([weff[i], weff[i]], [residuals[i] - 1, residuals[i] + 1],
                   '-', color='red', lw=1.5)

    ax_oc.plot(weff, residuals, 'o', color='red', ms=6, mfc='red', mec='red')

    ymax_oc = _oc_ylim_sed(residuals)
    ax_oc.set_ylim(-ymax_oc / 0.7, ymax_oc / 0.7)
    ax_oc.set_yticks([-ymax_oc, 0, ymax_oc])
    ax_oc.axhline(0, ls='--', color='red', lw=0.8)
    ax_oc.set_xlabel(r'$\lambda$ ($\mu$m)')
    ax_oc.set_ylabel(r'Res ($\sigma$)')

    # Save or show
    if outfile is not None:
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'Saved SED plot to {outfile}')
    else:
        plt.show()

    return fig
