"""
RV plotting for EXOZIPPy.

Mirrors EXOFASTv2 plotrv.pro — produces a single combined figure:
  Top:    Phase-folded RV with O-C residuals
  Bottom: Unphased RV vs BJD_TDB with O-C residuals

Supports multiple RV telescopes with color/symbol cycling.

Usage:
    from exozippy.plotrv import plotrv
    plotrv(rvfiles, bestfit, outfile='rv.png')
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from exozippy.exozippy_rv import exozippy_rv
from exozippy.fit_exoplanet import read_rv_data
from exozippy.exozippy_chi2 import tc_to_tp

# IDL-style color/symbol cycling (matches EXOFASTv2 plotrv.pro)
_COLORS = ['black', 'red', 'blue', 'darkgreen', 'orange', 'purple', 'cyan', 'magenta']
_SYMBOLS = ['o', 's', 'D', '^', 'v', '<', '>', 'p']


# ---------- shared helpers ----------

def _oc_ylim_rv(residuals, err_total):
    """Symmetric O-C y-limits including error bars, rounded to 2 sig-figs."""
    ymax = np.max(np.abs(residuals) + err_total) * 1.1
    if ymax == 0:
        return 1.0
    ndigits = np.floor(np.log10(ymax)) - 1
    return np.round(ymax / 10**ndigits) * 10**ndigits


def _telescope_label(filepath):
    """Extract a short label from an RV file path."""
    basename = os.path.basename(filepath)
    # Try to extract instrument name from patterns like 'KELT-14b.AAT.rv'
    parts = os.path.splitext(basename)[0].split('.')
    if len(parts) >= 2:
        return parts[-1]
    return basename


# ---------- axes-level drawing ----------

def _draw_phased(ax_data, ax_oc, data_list, bestfit, e, omega,
                 labels=None, samples=None):
    """Phase-folded RV with O-C residuals — all telescopes overlaid."""
    tc = bestfit['tc']
    period = bestfit['period']
    K = bestfit['K']
    tp = tc_to_tp(tc, period, e, omega)

    # Model curve (gamma=0)
    phase_fine = np.linspace(0, 1, 500)
    t_fine = tc + (phase_fine - 0.25) * period
    model_fine = exozippy_rv(t_fine, tp, period, 0.0, K, e=e, omega=omega)
    ax_data.plot(phase_fine, model_fine, '-', color='red', lw=2, zorder=2)

    all_residuals = []
    all_err_total = []

    for j, data in enumerate(data_list):
        gamma_j = bestfit.get(f'gamma_{j}', bestfit.get('gamma', 0.0))
        jittervar_j = bestfit.get(f'jittervar_{j}', bestfit.get('rv_jittervar', 0.0))
        color = _COLORS[j % len(_COLORS)]
        marker = _SYMBOLS[j % len(_SYMBOLS)]
        label = labels[j] if labels else f'Tel {j}'

        phase = np.mod((data['bjd'] - tc) / period + 1.25, 1.0)
        err_total = np.sqrt(data['err']**2 + jittervar_j)
        model_nogamma = exozippy_rv(data['bjd'], tp, period, 0.0, K,
                                    e=e, omega=omega)
        residuals = data['vel'] - gamma_j - model_nogamma
        all_residuals.append(residuals)
        all_err_total.append(err_total)

        ax_data.errorbar(phase, residuals + model_nogamma, yerr=err_total,
                         fmt=marker, color=color, ms=4, capsize=0,
                         zorder=3, label=label)
        ax_oc.errorbar(phase, residuals, yerr=err_total,
                       fmt=marker, color=color, ms=4, capsize=0)

    if len(data_list) > 1:
        ax_data.legend(fontsize=8, loc='best')

    ax_data.set_ylabel('RV (m/s)')
    ax_data.set_xlim(0, 1)
    plt.setp(ax_data.get_xticklabels(), visible=False)

    all_res = np.concatenate(all_residuals)
    all_err = np.concatenate(all_err_total)
    ymax_oc = _oc_ylim_rv(all_res, all_err)
    ax_oc.set_ylim(-ymax_oc / 0.7, ymax_oc / 0.7)
    ax_oc.set_yticks([-ymax_oc, 0, ymax_oc])
    ax_oc.axhline(0, ls='--', color='red', lw=0.8)
    ax_oc.set_xlabel(r'Phase + (T$_P$ $-$ T$_C$)/P + 0.25')
    ax_oc.set_ylabel('O-C (m/s)')


def _draw_unphased(ax_data, ax_oc, data_list, bestfit, e, omega,
                   labels=None, samples=None):
    """Unphased RV vs BJD_TDB with O-C residuals — all telescopes overlaid."""
    tc = bestfit['tc']
    period = bestfit['period']
    K = bestfit['K']
    tp = tc_to_tp(tc, period, e, omega)

    # Gather all BJDs for axis range
    all_bjd = np.concatenate([d['bjd'] for d in data_list])
    roundto = 10 ** len(str(int(all_bjd.max() - all_bjd.min())))
    bjd0 = np.floor(all_bjd.min() / roundto) * roundto

    # Model curve (gamma=0)
    cadence = period / 100.0
    nsteps = max(int((all_bjd.max() - all_bjd.min()) / cadence), 500)
    t_fine = np.linspace(all_bjd.min(), all_bjd.max(), nsteps)
    model_fine = exozippy_rv(t_fine, tp, period, 0.0, K, e=e, omega=omega)
    ax_data.plot(t_fine - bjd0, model_fine, '-', color='red', lw=1.5, zorder=2)

    all_residuals = []
    all_err_total = []

    for j, data in enumerate(data_list):
        gamma_j = bestfit.get(f'gamma_{j}', bestfit.get('gamma', 0.0))
        jittervar_j = bestfit.get(f'jittervar_{j}', bestfit.get('rv_jittervar', 0.0))
        color = _COLORS[j % len(_COLORS)]
        marker = _SYMBOLS[j % len(_SYMBOLS)]
        label = labels[j] if labels else f'Tel {j}'

        bjd = data['bjd']
        vel = data['vel']
        err_total = np.sqrt(data['err']**2 + jittervar_j)
        model_data = exozippy_rv(bjd, tp, period, 0.0, K, e=e, omega=omega)
        residuals = vel - gamma_j - model_data
        all_residuals.append(residuals)
        all_err_total.append(err_total)

        ax_data.errorbar(bjd - bjd0, vel - gamma_j, yerr=err_total,
                         fmt=marker, color=color, ms=4, capsize=0,
                         zorder=3, label=label)
        ax_oc.errorbar(bjd - bjd0, residuals, yerr=err_total,
                       fmt=marker, color=color, ms=4, capsize=0)

    if len(data_list) > 1:
        ax_data.legend(fontsize=8, loc='best')

    ax_data.set_ylabel('RV (m/s)')
    plt.setp(ax_data.get_xticklabels(), visible=False)

    all_res = np.concatenate(all_residuals)
    all_err = np.concatenate(all_err_total)
    ymax_oc = _oc_ylim_rv(all_res, all_err)
    ax_oc.set_ylim(-ymax_oc / 0.7, ymax_oc / 0.7)
    ax_oc.set_yticks([-ymax_oc, 0, ymax_oc])
    ax_oc.axhline(0, ls='--', color='red', lw=0.8)
    ax_oc.set_xlabel(r'BJD$_{\mathrm{TDB}}$' + f' $-$ {int(bjd0)}')
    ax_oc.set_ylabel('O-C (m/s)')


# ---------- public API ----------

def plotrv(rvfiles, bestfit, samples=None, e=0.0, omega=np.pi / 2,
           outfile=None):
    """
    Combined RV plot — single figure with GridSpec.

    Supports multiple RV telescopes overlaid with color/symbol cycling.

    Layout (nested GridSpec):
        Row 0 (height 1): Phase-folded RV + O-C residuals
        Row 1 (height 1): Unphased RV + O-C residuals

    Parameters
    ----------
    rvfiles : str or list[str]
        Path(s) to RV data file(s) (BJD vel err).
    bestfit : dict or SS
        Best-fit parameter dictionary from fit_exoplanet.
    samples : ndarray, optional
        MCMC samples for posterior draws.
    e, omega : float
        Eccentricity and argument of periastron.
    outfile : str, optional
        Output filename (.png or .pdf).

    Returns
    -------
    fig : Figure
    """
    if isinstance(rvfiles, str):
        rvfiles = [rvfiles]

    data_list = [read_rv_data(f) for f in rvfiles]
    labels = [_telescope_label(f) for f in rvfiles]

    fig = plt.figure(figsize=(12, 10))

    outer = gridspec.GridSpec(
        2, 1, figure=fig, height_ratios=(1, 1),
        left=0.12, right=0.95, top=0.95, bottom=0.08, hspace=0.35,
    )

    gs_phased = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[0], height_ratios=(3, 1), hspace=0.0,
    )
    ax_phased = fig.add_subplot(gs_phased[0])
    ax_phased_oc = fig.add_subplot(gs_phased[1], sharex=ax_phased)
    _draw_phased(ax_phased, ax_phased_oc, data_list, bestfit, e, omega,
                 labels=labels, samples=samples)

    gs_unphased = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1], height_ratios=(3, 1), hspace=0.0,
    )
    ax_unphased = fig.add_subplot(gs_unphased[0])
    ax_unphased_oc = fig.add_subplot(gs_unphased[1], sharex=ax_unphased)
    _draw_unphased(ax_unphased, ax_unphased_oc, data_list, bestfit,
                   e, omega, labels=labels, samples=samples)

    if outfile is not None:
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'Saved RV plot to {outfile}')
    else:
        plt.show()

    return fig
