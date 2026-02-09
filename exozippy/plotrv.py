"""
RV plotting for EXOZIPPy.

Mirrors EXOFASTv2 plotrv.pro — produces a single combined figure:
  Top:    Phase-folded RV with O-C residuals
  Bottom: Unphased RV vs BJD_TDB with O-C residuals

Usage:
    from exozippy.plotrv import plotrv
    plotrv(rvfile, bestfit, outfile='rv.png')
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from exozippy.exozippy_rv import exozippy_rv
from exozippy.fit_exoplanet import read_rv_data
from exozippy.exozippy_chi2 import tc_to_tp


# ---------- shared helpers ----------

def _oc_ylim_rv(residuals, err_total):
    """Symmetric O-C y-limits including error bars, rounded to 2 sig-figs."""
    ymax = np.max(np.abs(residuals) + err_total) * 1.1
    if ymax == 0:
        return 1.0
    ndigits = np.floor(np.log10(ymax)) - 1
    return np.round(ymax / 10**ndigits) * 10**ndigits


def _posterior_rv_models(t_fine, samples, e, omega, ndraws=100):
    """Draw RV models (gamma=0) from MCMC posterior samples."""
    ndraws = min(ndraws, len(samples))
    idx = np.random.choice(len(samples), ndraws, replace=False)
    models = []
    for ii in idx:
        s = samples[ii]
        s_tp = tc_to_tp(s[5], s[6], e, omega)
        try:
            m = exozippy_rv(t_fine, s_tp, s[6], 0.0, s[12],
                            e=e, omega=omega)
            models.append(m)
        except Exception:
            continue
    return models


# ---------- axes-level drawing ----------

def _draw_phased(ax_data, ax_oc, data, bestfit, e, omega, samples=None):
    """Phase-folded RV with O-C residuals."""
    tc = bestfit['tc']
    period = bestfit['period']
    K = bestfit['K']
    gamma = bestfit['gamma']
    rv_jittervar = bestfit.get('rv_jittervar', 0.0)
    tp = tc_to_tp(tc, period, e, omega)

    phase = np.mod((data['bjd'] - tc) / period + 1.25, 1.0)
    err_total = np.sqrt(data['err']**2 + rv_jittervar)

    model_nogamma = exozippy_rv(data['bjd'], tp, period, 0.0, K,
                                e=e, omega=omega)
    residuals = data['vel'] - gamma - model_nogamma

    phase_fine = np.linspace(0, 1, 500)
    t_fine = tc + (phase_fine - 0.25) * period
    model_fine = exozippy_rv(t_fine, tp, period, 0.0, K,
                             e=e, omega=omega)

    # Posterior draws
    if samples is not None:
        posterior = _posterior_rv_models(t_fine, samples, e, omega)
        for m in posterior:
            ax_data.plot(phase_fine, m, color='lightskyblue',
                         alpha=0.1, lw=0.5, zorder=1)

    ax_data.errorbar(phase, residuals + model_nogamma, yerr=err_total,
                     fmt='ko', ms=4, capsize=0, zorder=3)
    ax_data.plot(phase_fine, model_fine, '-', color='red', lw=2, zorder=2)
    ax_data.set_ylabel('RV (m/s)')
    ax_data.set_xlim(0, 1)
    plt.setp(ax_data.get_xticklabels(), visible=False)

    ax_oc.errorbar(phase, residuals, yerr=err_total,
                   fmt='ko', ms=4, capsize=0)
    ymax_oc = _oc_ylim_rv(residuals, err_total)
    ax_oc.set_ylim(-ymax_oc / 0.7, ymax_oc / 0.7)
    ax_oc.set_yticks([-ymax_oc, 0, ymax_oc])
    ax_oc.axhline(0, ls='--', color='red', lw=0.8)
    ax_oc.set_xlabel(r'Phase + (T$_P$ $-$ T$_C$)/P + 0.25')
    ax_oc.set_ylabel('O-C (m/s)')


def _draw_unphased(ax_data, ax_oc, data, bestfit, e, omega, samples=None):
    """Unphased RV vs BJD_TDB with O-C residuals."""
    tc = bestfit['tc']
    period = bestfit['period']
    K = bestfit['K']
    gamma = bestfit['gamma']
    rv_jittervar = bestfit.get('rv_jittervar', 0.0)
    tp = tc_to_tp(tc, period, e, omega)

    bjd = data['bjd']
    vel = data['vel']
    err_total = np.sqrt(data['err']**2 + rv_jittervar)

    roundto = 10 ** len(str(int(bjd.max() - bjd.min())))
    bjd0 = np.floor(bjd.min() / roundto) * roundto

    model_data = exozippy_rv(bjd, tp, period, 0.0, K, e=e, omega=omega)
    residuals = vel - gamma - model_data

    cadence = period / 100.0
    nsteps = max(int((bjd.max() - bjd.min()) / cadence), 500)
    t_fine = np.linspace(bjd.min(), bjd.max(), nsteps)
    model_fine = exozippy_rv(t_fine, tp, period, 0.0, K,
                             e=e, omega=omega)

    # Posterior draws
    if samples is not None:
        posterior = _posterior_rv_models(t_fine, samples, e, omega)
        for m in posterior:
            ax_data.plot(t_fine - bjd0, m, color='lightskyblue',
                         alpha=0.1, lw=0.5, zorder=1)

    ax_data.errorbar(bjd - bjd0, vel - gamma, yerr=err_total,
                     fmt='ko', ms=4, capsize=0, zorder=3)
    ax_data.plot(t_fine - bjd0, model_fine, '-', color='red', lw=1.5,
                 zorder=2)
    ax_data.set_ylabel('RV (m/s)')
    plt.setp(ax_data.get_xticklabels(), visible=False)

    ax_oc.errorbar(bjd - bjd0, residuals, yerr=err_total,
                   fmt='ko', ms=4, capsize=0)
    ymax_oc = _oc_ylim_rv(residuals, err_total)
    ax_oc.set_ylim(-ymax_oc / 0.7, ymax_oc / 0.7)
    ax_oc.set_yticks([-ymax_oc, 0, ymax_oc])
    ax_oc.axhline(0, ls='--', color='red', lw=0.8)
    ax_oc.set_xlabel(r'BJD$_{\mathrm{TDB}}$' + f' $-$ {int(bjd0)}')
    ax_oc.set_ylabel('O-C (m/s)')


# ---------- public API ----------

def plotrv(rvfile, bestfit, samples=None, e=0.0, omega=np.pi / 2,
           outfile=None):
    """
    Combined RV plot — single figure with GridSpec.

    Layout (nested GridSpec):
        Row 0 (height 1): Phase-folded RV + O-C residuals
        Row 1 (height 1): Unphased RV + O-C residuals

    Parameters
    ----------
    rvfile : str
        Path to RV data file (BJD vel err).
    bestfit : dict
        Best-fit parameter dictionary from fit_exoplanet.
    samples : ndarray, optional
        MCMC samples for posterior draws.
    e, omega : float
        Eccentricity and argument of periastron.
    outfile : str, optional
        Output filename (.png or .pdf).
        If None, displays interactively.

    Returns
    -------
    fig : Figure
    """
    data = read_rv_data(rvfile)

    fig = plt.figure(figsize=(12, 10))

    # Outer: 2 rows — phased section + unphased section (equal height)
    outer = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=(1, 1),
        left=0.12, right=0.95, top=0.95, bottom=0.08, hspace=0.35,
    )

    # Row 0: Phase-folded RV with O-C (nested 2 rows, hspace=0)
    gs_phased = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0], height_ratios=(3, 1), hspace=0.0,
    )
    ax_phased = fig.add_subplot(gs_phased[0])
    ax_phased_oc = fig.add_subplot(gs_phased[1], sharex=ax_phased)

    _draw_phased(ax_phased, ax_phased_oc, data, bestfit, e, omega, samples)

    # Row 1: Unphased RV with O-C (nested 2 rows, hspace=0)
    gs_unphased = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1], height_ratios=(3, 1), hspace=0.0,
    )
    ax_unphased = fig.add_subplot(gs_unphased[0])
    ax_unphased_oc = fig.add_subplot(gs_unphased[1], sharex=ax_unphased)

    _draw_unphased(ax_unphased, ax_unphased_oc, data, bestfit,
                   e, omega, samples)

    # Save or show
    if outfile is not None:
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'Saved RV plot to {outfile}')
    else:
        plt.show()

    return fig