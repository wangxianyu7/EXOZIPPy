"""
Transit plotting for EXOZIPPy.

Mirrors EXOFASTv2 plottran.pro — produces a single combined figure:
  Top:    Phase-folded primary transit with O-C residuals
  Bottom: Unphased transit (Norm flux vs BJD_TDB)

Usage:
    from exozippy.plottran import plottran
    plottran(tranfile, bestfit, outfile='transit.png')
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from exozippy.exozippy_tran import exozippy_tran
from exozippy.fit_exoplanet import read_transit_data
from exozippy.exozippy_chi2 import tc_to_tp, derive_ar as _derive_ar


# ---------- shared helpers ----------

def _exofast_mod(x, period):
    """Fold x into [-period/2, +period/2)  (like exofast_mod,/negative)."""
    return np.mod(x + period / 2, period) - period / 2


def _make_model(bjd, bestfit, e, omega, mstar):
    """Evaluate transit model at arbitrary times."""
    tp = tc_to_tp(bestfit['tc'], bestfit['period'], e, omega)
    return exozippy_tran(
        bjd, bestfit['inc_rad'], bestfit['ar'], tp,
        bestfit['period'], e, omega,
        bestfit['p'], bestfit['u1'], bestfit['u2'], bestfit['f0'],
    )


def _oc_ylim(residuals):
    """Compute symmetric O-C y-limits rounded to 2 sig-figs (IDL convention)."""
    ymax = np.max(np.abs(residuals)) * 1.1
    if ymax == 0:
        return 0.001
    ndigits = np.floor(np.log10(ymax)) - 1
    return np.round(ymax / 10**ndigits) * 10**ndigits


def _posterior_transit_models(t_fine, samples, mstar, e, omega, ndraws=100):
    """Draw transit models from MCMC posterior samples."""
    ndraws = min(ndraws, len(samples))
    idx = np.random.choice(len(samples), ndraws, replace=False)
    models = []
    for ii in idx:
        s = samples[ii]
        s_inc = np.arccos(s[8])
        s_ar = _derive_ar(s[6], mstar, s[1])
        s_tp = tc_to_tp(s[5], s[6], e, omega)
        try:
            m = exozippy_tran(t_fine, s_inc, s_ar, s_tp, s[6],
                              e, omega, s[7], s[9], s[10], s[11])
            models.append(m)
        except Exception:
            continue
    return models


# ---------- axes-level drawing ----------

def _draw_phased(ax_data, ax_oc, data, bestfit, e, omega, mstar,
                 samples=None):
    """Phase-folded primary transit with O-C residuals."""
    tc = bestfit['tc']
    period = bestfit['period']
    tp = tc_to_tp(tc, period, e, omega)

    bjd = data['bjd']
    flux = data['flux']

    dt_hrs = _exofast_mod(bjd - tc, period) * 24.0

    model_data = _make_model(bjd, bestfit, e, omega, mstar)
    residuals = flux - model_data

    # Transit duration for x-range
    p = bestfit['p']
    ar = bestfit['ar']
    cosi = bestfit['cosi']
    inc = bestfit['inc_rad']
    b = ar * cosi
    sini = np.sin(inc)
    esinw = e * np.sin(omega)
    t14_days = (period / np.pi) * np.arcsin(
        np.sqrt((1 + p)**2 - b**2) / (sini * ar)
    ) * np.sqrt(1 - e**2) / (1 + esinw)
    t14_hrs = t14_days * 24.0

    # Fine grid spanning +/- duration
    npretty = max(int(np.ceil(2 * t14_days * 1440 * 2)), 500)
    t_fine_rel = np.linspace(-t14_days, t14_days, npretty)
    t_fine = tc + t_fine_rel
    dt_fine_hrs = t_fine_rel * 24.0
    model_fine = exozippy_tran(
        t_fine, inc, ar, tp, period, e, omega,
        p, bestfit['u1'], bestfit['u2'], bestfit['f0'],
    )

    # Posterior draws
    # if samples is not None:
    #     posterior = _posterior_transit_models(
    #         t_fine, samples, mstar, e, omega
    #     )
    #     for m in posterior:
    #         ax_data.plot(dt_fine_hrs, m, color='lightskyblue',
    #                      alpha=0.1, lw=0.5, zorder=1)

    ax_data.plot(dt_hrs, flux, 'k.', ms=3, zorder=2)
    ax_data.plot(dt_fine_hrs, model_fine, '-', color='red', lw=2, zorder=3)
    ax_data.set_ylabel('Norm flux')
    ax_data.set_xlim(-t14_hrs, t14_hrs)
    plt.setp(ax_data.get_xticklabels(), visible=False)

    ax_oc.plot(dt_hrs, residuals, 'k.', ms=3)
    ymax_oc = _oc_ylim(residuals)
    ax_oc.set_ylim(-ymax_oc / 0.7, ymax_oc / 0.7)
    ax_oc.set_yticks([-ymax_oc, 0, ymax_oc])
    ax_oc.axhline(0, ls='--', color='red', lw=0.8)
    ax_oc.set_xlabel(r'Time $-$ T$_C$ (Hrs)')
    ax_oc.set_ylabel('O-C')


def _draw_unphased(ax, data, bestfit, e, omega, mstar, samples=None):
    """Unphased transit — Norm flux vs BJD_TDB."""
    bjd = data['bjd']
    flux = data['flux']

    roundto = 10 ** len(str(int(bjd.max() - bjd.min())))
    t0 = np.floor(bjd.min() / roundto) * roundto

    model = _make_model(bjd, bestfit, e, omega, mstar)
    residuals = flux - model

    npretty = max(int(np.ceil((bjd.max() - bjd.min()) * 1440)), 500)
    t_fine = np.linspace(bjd.min(), bjd.max(), npretty)
    model_fine = _make_model(t_fine, bestfit, e, omega, mstar)

    noise = np.std(residuals)
    depth = bestfit['p'] ** 2

    ax.plot(bjd - t0, flux, 'k.', ms=2, zorder=2)
    ax.plot(t_fine - t0, model_fine, '-', color='red', lw=1.5, zorder=3)
    ax.set_xlabel(r'BJD$_{\mathrm{TDB}}$' + f' $-$ {int(t0)}')
    ax.set_ylabel('Norm flux')
    ax.set_ylim(1.0 - depth - 3 * noise, 1.0 + 3 * noise)
    ax.set_xlim(bjd.min() - t0, bjd.max() - t0)


# ---------- public API ----------

def plottran(tranfile, bestfit, samples=None, mstar=None,
             e=0.0, omega=np.pi / 2, outfile=None):
    """
    Combined transit plot — single figure with GridSpec.

    Layout (nested GridSpec):
        Row 0 (height 2): Phase-folded transit + O-C residuals
        Row 1 (height 1): Unphased transit

    Parameters
    ----------
    tranfile : str
        Path to transit data file (BJD flux err).
    bestfit : dict
        Best-fit parameter dictionary from fit_exoplanet.
    samples : ndarray, optional
        MCMC samples for posterior draws.
    mstar : float, optional
        Stellar mass in Msun.  Defaults to bestfit['mstar'].
    e, omega : float
        Eccentricity and argument of periastron.
    outfile : str, optional
        Output filename (.png or .pdf).
        If None, displays interactively.

    Returns
    -------
    fig : Figure
    """
    if mstar is None:
        mstar = bestfit.get('mstar', 0.904)

    data = read_transit_data(tranfile)

    fig = plt.figure(figsize=(12, 10))

    # Outer: 2 rows — phased section (bigger) + unphased (smaller)
    outer = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=(2, 1),
        left=0.12, right=0.95, top=0.95, bottom=0.08, hspace=0.35,
    )

    # Row 0: Phase-folded transit with O-C (nested 2 rows, hspace=0)
    gs_phased = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0], height_ratios=(3, 1), hspace=0.0,
    )
    ax_phased = fig.add_subplot(gs_phased[0])
    ax_phased_oc = fig.add_subplot(gs_phased[1], sharex=ax_phased)

    _draw_phased(ax_phased, ax_phased_oc, data, bestfit,
                 e, omega, mstar, samples)

    # Row 1: Unphased transit (single panel)
    ax_unphased = fig.add_subplot(outer[1])
    _draw_unphased(ax_unphased, data, bestfit, e, omega, mstar, samples)

    # Save or show
    if outfile is not None:
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'Saved transit plot to {outfile}')
    else:
        plt.show()

    return fig