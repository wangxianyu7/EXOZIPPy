"""
Transit plotting for EXOZIPPy.

Mirrors EXOFASTv2 plottran.pro — produces a single combined figure:
  Top:    Phase-folded primary transit with O-C residuals
  Bottom: Unphased transit (Norm flux vs BJD_TDB)

Supports multiple transit light curves stacked vertically.

Usage:
    from exozippy.plottran import plottran
    plottran(tranfiles, bestfit, outfile='transit.png')
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from exozippy.exozippy_tran import exozippy_tran
from exozippy.fit_exoplanet import read_transit_data
from exozippy.exozippy_chi2 import tc_to_tp, derive_ar as _derive_ar

# Color cycling (matches plotrv.py)
_COLORS = ['black', 'red', 'blue', 'darkgreen', 'orange', 'purple', 'cyan', 'magenta']


# ---------- shared helpers ----------

def _exofast_mod(x, period):
    """Fold x into [-period/2, +period/2)  (like exofast_mod,/negative)."""
    return np.mod(x + period / 2, period) - period / 2


def _make_model(bjd, bestfit, e, omega, mstar, tran_idx=0):
    """Evaluate transit model at arbitrary times."""
    tp = tc_to_tp(bestfit['tc'], bestfit['period'], e, omega)
    u1 = bestfit.get(f'u1_{tran_idx}', bestfit.get('u1_0', bestfit.get('u1', 0.4)))
    u2 = bestfit.get(f'u2_{tran_idx}', bestfit.get('u2_0', bestfit.get('u2', 0.2)))
    f0 = bestfit.get(f'f0_{tran_idx}', bestfit.get('f0_0', bestfit.get('f0', 1.0)))
    return exozippy_tran(
        bjd, bestfit['inc_rad'], bestfit['ar'], tp,
        bestfit['period'], e, omega,
        bestfit['p'], u1, u2, f0,
    )


def _oc_ylim(residuals):
    """Compute symmetric O-C y-limits rounded to 2 sig-figs (IDL convention)."""
    ymax = np.max(np.abs(residuals)) * 1.1
    if ymax == 0:
        return 0.001
    ndigits = np.floor(np.log10(ymax)) - 1
    return np.round(ymax / 10**ndigits) * 10**ndigits


def _transit_label(filepath):
    """Extract a short label from a transit file path."""
    return os.path.basename(filepath)


def _compute_t14(bestfit, e, omega):
    """Compute transit duration T14 in days."""
    p = bestfit['p']
    ar = bestfit['ar']
    cosi = bestfit['cosi']
    inc = bestfit['inc_rad']
    period = bestfit['period']
    b = ar * cosi
    sini = np.sin(inc)
    esinw = e * np.sin(omega)
    arg = (1 + p)**2 - b**2
    if arg > 0 and sini * ar > 0:
        t14 = (period / np.pi) * np.arcsin(
            np.sqrt(arg) / (sini * ar)
        ) * np.sqrt(1 - e**2) / (1 + esinw)
    else:
        t14 = period * 0.02
    return t14


# ---------- axes-level drawing ----------

def _draw_phased_multi(ax_data, ax_oc, data_list, bestfit, e, omega, mstar,
                       labels=None, samples=None):
    """Phase-folded transit with vertical stacking for multiple light curves."""
    tc = bestfit['tc']
    period = bestfit['period']
    tp = tc_to_tp(tc, period, e, omega)
    t14_days = _compute_t14(bestfit, e, omega)
    t14_hrs = t14_days * 24.0

    nlc = len(data_list)

    # Compute spacing between stacked light curves (IDL plottran.pro convention)
    depth = bestfit['p'] ** 2
    max_noise = 0.0
    for data in data_list:
        model_j = _make_model(data['bjd'], bestfit, e, omega, mstar, tran_idx=0)
        res_j = data['flux'] - model_j
        max_noise = max(max_noise, np.std(res_j))
    spacing = 3 * (depth + max_noise) if nlc > 1 else 0.0

    # Fine model grid
    inc = bestfit['inc_rad']
    ar = bestfit['ar']
    p_val = bestfit['p']
    u1_0 = bestfit.get('u1_0', bestfit.get('u1', 0.4))
    u2_0 = bestfit.get('u2_0', bestfit.get('u2', 0.2))
    npretty = max(int(np.ceil(2 * t14_days * 1440 * 2)), 500)
    t_fine_rel = np.linspace(-t14_days, t14_days, npretty)
    t_fine = tc + t_fine_rel
    dt_fine_hrs = t_fine_rel * 24.0

    all_residuals = []

    for j, data in enumerate(data_list):
        color = _COLORS[j % len(_COLORS)]
        label = labels[j] if labels else f'Transit {j}'
        offset = (nlc - 1 - j) * spacing

        f0_j = bestfit.get(f'f0_{j}', bestfit.get('f0_0', bestfit.get('f0', 1.0)))
        # Use band 0 LD for all transits in single-band mode
        model_fine_j = exozippy_tran(
            t_fine, inc, ar, tp, period, e, omega,
            p_val, u1_0, u2_0, f0_j,
        )

        dt_hrs = _exofast_mod(data['bjd'] - tc, period) * 24.0
        model_data = _make_model(data['bjd'], bestfit, e, omega, mstar, tran_idx=j)
        residuals = data['flux'] - model_data
        all_residuals.append(residuals)

        ax_data.plot(dt_hrs, data['flux'] + offset, '.', color=color, ms=3,
                     zorder=2, label=label)
        ax_data.plot(dt_fine_hrs, model_fine_j + offset, '-', color='red',
                     lw=2, zorder=3)

        # Label for stacked curves
        if nlc > 1:
            ax_data.text(t14_hrs * 0.95, f0_j + offset - depth / 2,
                         label, fontsize=7, ha='left', va='center',
                         color=color)

        ax_oc.plot(dt_hrs, residuals, '.', color=color, ms=3)

    if nlc > 1:
        ax_data.legend(fontsize=7, loc='lower left', ncol=min(nlc, 4))

    ax_data.set_ylabel('Norm flux')
    ax_data.set_xlim(-t14_hrs, t14_hrs)
    plt.setp(ax_data.get_xticklabels(), visible=False)

    all_res = np.concatenate(all_residuals)
    ymax_oc = _oc_ylim(all_res)
    ax_oc.set_ylim(-ymax_oc / 0.7, ymax_oc / 0.7)
    ax_oc.set_yticks([-ymax_oc, 0, ymax_oc])
    ax_oc.axhline(0, ls='--', color='red', lw=0.8)
    ax_oc.set_xlabel(r'Time $-$ T$_C$ (Hrs)')
    ax_oc.set_ylabel('O-C')


def _draw_unphased_multi(ax, data_list, bestfit, e, omega, mstar,
                         labels=None, samples=None):
    """Unphased transit — Norm flux vs BJD_TDB, all light curves overlaid."""
    all_bjd = np.concatenate([d['bjd'] for d in data_list])
    roundto = 10 ** len(str(int(all_bjd.max() - all_bjd.min())))
    t0 = np.floor(all_bjd.min() / roundto) * roundto

    depth = bestfit['p'] ** 2
    max_noise = 0.0
    nlc = len(data_list)

    for j, data in enumerate(data_list):
        color = _COLORS[j % len(_COLORS)]
        label = labels[j] if labels else f'Transit {j}'

        bjd = data['bjd']
        flux = data['flux']
        model = _make_model(bjd, bestfit, e, omega, mstar, tran_idx=j)
        residuals = flux - model
        noise = np.std(residuals)
        max_noise = max(max_noise, noise)

        # Fine model
        npretty = max(int(np.ceil((bjd.max() - bjd.min()) * 1440)), 500)
        t_fine = np.linspace(bjd.min(), bjd.max(), npretty)
        model_fine = _make_model(t_fine, bestfit, e, omega, mstar, tran_idx=j)

        ax.plot(bjd - t0, flux, '.', color=color, ms=2, zorder=2, label=label)
        ax.plot(t_fine - t0, model_fine, '-', color='red', lw=1.5, zorder=3)

    if nlc > 1:
        ax.legend(fontsize=7, loc='best', ncol=min(nlc, 4))

    ax.set_xlabel(r'BJD$_{\mathrm{TDB}}$' + f' $-$ {int(t0)}')
    ax.set_ylabel('Norm flux')
    ax.set_ylim(1.0 - depth - 3 * max_noise, 1.0 + 3 * max_noise)
    ax.set_xlim(all_bjd.min() - t0, all_bjd.max() - t0)


# ---------- public API ----------

def plottran(tranfiles, bestfit, samples=None, mstar=None,
             e=0.0, omega=np.pi / 2, outfile=None):
    """
    Combined transit plot — single figure with GridSpec.

    Supports multiple transit light curves:
      - Phase-folded panel: stacked vertically with spacing
      - Unphased panel: all overlaid with color coding

    Layout (nested GridSpec):
        Row 0 (height 2): Phase-folded transit + O-C residuals
        Row 1 (height 1): Unphased transit

    Parameters
    ----------
    tranfiles : str or list[str]
        Path(s) to transit data file(s) (BJD flux err).
    bestfit : dict or SS
        Best-fit parameter dictionary from fit_exoplanet.
    samples : ndarray, optional
        MCMC samples for posterior draws.
    mstar : float, optional
        Stellar mass in Msun.  Defaults to bestfit['mstar'].
    e, omega : float
        Eccentricity and argument of periastron.
    outfile : str, optional
        Output filename (.png or .pdf).

    Returns
    -------
    fig : Figure
    """
    if mstar is None:
        mstar = bestfit.get('mstar', 0.904)

    if isinstance(tranfiles, str):
        tranfiles = [tranfiles]

    data_list = [read_transit_data(f) for f in tranfiles]
    labels = [_transit_label(f) for f in tranfiles]

    fig = plt.figure(figsize=(12, 10))

    outer = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=(2, 1),
        left=0.12, right=0.95, top=0.95, bottom=0.08, hspace=0.35,
    )

    gs_phased = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0], height_ratios=(3, 1), hspace=0.0,
    )
    ax_phased = fig.add_subplot(gs_phased[0])
    ax_phased_oc = fig.add_subplot(gs_phased[1], sharex=ax_phased)
    _draw_phased_multi(ax_phased, ax_phased_oc, data_list, bestfit,
                       e, omega, mstar, labels=labels, samples=samples)

    ax_unphased = fig.add_subplot(outer[1])
    _draw_unphased_multi(ax_unphased, data_list, bestfit, e, omega, mstar,
                         labels=labels, samples=samples)

    if outfile is not None:
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'Saved transit plot to {outfile}')
    else:
        plt.show()

    return fig
