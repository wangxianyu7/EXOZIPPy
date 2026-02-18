"""
RV plotting for EXOZIPPy.

Mirrors EXOFASTv2 plotrv.pro — produces a single combined figure:
  Top:    Phase-folded RV with O-C residuals  (model includes RM)
  Middle: Unphased RV vs BJD_TDB with O-C residuals
  Bottom: RM close-up panel (only when rossiter=True)
           x-axis: Time − T_C (hrs), range = [−t14, +t14]
           model = RM-only component
           data  = RM signal reconstructed from residuals

Supports multiple RV telescopes with color/symbol cycling.

Usage:
    from exozippy.plotting.rv import plotrv
    plotrv(rvfiles, bestfit, outfile='rv.png')
"""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from exozippy.physics.exozippy_rv import exozippy_rv
from exozippy.jointfit import read_rv_data
from exozippy.jointfit.chi2 import tc_to_tp
from .common import (
    DEFAULT_COLORS,
    DEFAULT_SYMBOLS,
    rounded_symmetric_limit_with_errors,
    save_or_show,
)

# IDL-style color/symbol cycling (matches EXOFASTv2 plotrv.pro)
_COLORS = DEFAULT_COLORS
_SYMBOLS = DEFAULT_SYMBOLS


# ---------- shared helpers ----------

def _oc_ylim_rv(residuals, err_total):
    """Symmetric O-C y-limits including error bars, rounded to 2 sig-figs."""
    return rounded_symmetric_limit_with_errors(residuals, err_total, fallback=1.0)


def _telescope_label(filepath):
    """Extract a short label from an RV file path."""
    basename = os.path.basename(filepath)
    parts = os.path.splitext(basename)[0].split('.')
    if len(parts) >= 2:
        return parts[-1]
    return basename


def _rm_kwargs(bestfit, rmbandndx_list, j):
    """Return RM keyword dict for exozippy_rv(); empty dict if no RM for telescope j."""
    bi = (rmbandndx_list[j]
          if rmbandndx_list is not None and j < len(rmbandndx_list)
          else -1)
    if bi < 0:
        return {}
    if not (hasattr(bestfit, 'planet') and bestfit.planet
            and bestfit.planet[0].rossiter):
        return {}
    svc = bestfit.get('svsinicoslam') or 0.0
    svs = bestfit.get('svsinisinlam') or 0.0
    vsini = svc**2 + svs**2
    lam   = np.arctan2(svs, svc)
    u1 = bestfit.get(f'u1_{bi}') or bestfit.get('u1_0', 0.4) or 0.4
    u2 = bestfit.get(f'u2_{bi}') or bestfit.get('u2_0', 0.2) or 0.2
    return dict(
        rossiter=True,
        i=bestfit.get('inc_rad') or np.arccos(np.clip(bestfit.get('cosi', 0.0) or 0.0, -1, 1)),
        a=bestfit.get('ar') or 10.0,
        p=bestfit.get('p') or 0.1,
        u1=float(u1), u2=float(u2),
        vsini=vsini, _lambda=lam,
        vgamma=bestfit.get('vgamma') or 1000.0,
        vzeta =bestfit.get('vzeta')  or 4000.0,
        vxi   =bestfit.get('vxi')    or 1000.0,
        valpha=bestfit.get('valpha') or 0.0,
    )


def _first_rm_kwargs(bestfit, rmbandndx_list):
    """Return RM kwargs for the first RM telescope, or empty dict."""
    if rmbandndx_list is None:
        return {}
    for j, bi in enumerate(rmbandndx_list):
        if bi >= 0:
            return _rm_kwargs(bestfit, rmbandndx_list, j)
    return {}


def _extract_rmbandndx_list(bestfit):
    """Extract per-telescope rmbandndx list from an SS bestfit object."""
    if hasattr(bestfit, 'telescope') and bestfit.telescope:
        return [tel.rmbandndx for tel in bestfit.telescope]
    return None


def _has_rossiter(bestfit, rmbandndx_list):
    """Return True if any telescope has RM enabled."""
    if rmbandndx_list is None:
        return False
    if not (hasattr(bestfit, 'planet') and bestfit.planet
            and bestfit.planet[0].rossiter):
        return False
    return any(bi >= 0 for bi in rmbandndx_list)


def _t14(bestfit, e, omega):
    """Full transit duration in days (Seager & Mallén-Ornelas 2003 + eccentricity factor)."""
    ar   = bestfit.get('ar') or 10.0
    p    = abs(bestfit.get('p') or 0.1)
    b    = bestfit.get('b') or 0.0
    period = bestfit.get('period') or 1.0
    sini = np.sin(bestfit.get('inc_rad')
                  or np.arccos(np.clip(bestfit.get('cosi', 0.0) or 0.0, -1, 1)))
    ecc_factor = np.sqrt(max(1.0 - e**2, 0.0)) / (1.0 + e * np.sin(omega))
    arg = np.sqrt(max((1.0 + p)**2 - b**2, 0.0)) / max(ar * sini, 1e-6)
    arg = np.clip(arg, 0.0, 1.0)
    return period / np.pi * np.arcsin(arg) * ecc_factor


# ---------- axes-level drawing ----------

def _draw_phased(ax_data, ax_oc, data_list, bestfit, e, omega,
                 labels=None, samples=None, rmbandndx_list=None):
    """Phase-folded RV with O-C residuals.

    Model curve includes RM (evaluated at the representative transit tc).
    Data shows gamma-subtracted velocities (vel − gamma), same as EXOFASTv2.
    O-C residuals use the full Keplerian + RM model.
    """
    tc     = bestfit['tc']
    period = bestfit['period']
    K      = bestfit['K']
    tp     = tc_to_tp(tc, period, e, omega)

    # Fine model: Keplerian + RM at canonical transit (tp = tc for circular)
    phase_fine = np.linspace(0, 1, 500)
    t_fine     = tc + (phase_fine - 0.25) * period
    rm_kw_fine = _first_rm_kwargs(bestfit, rmbandndx_list)
    model_fine = exozippy_rv(t_fine, tp, period, 0.0, K,
                             e=e, omega=omega, **rm_kw_fine)
    ax_data.plot(phase_fine, model_fine, '-', color='red', lw=2, zorder=2)

    all_residuals = []
    all_err_total = []

    for j, data in enumerate(data_list):
        gamma_j     = bestfit.get(f'gamma_{j}', bestfit.get('gamma', 0.0))
        jittervar_j = bestfit.get(f'jittervar_{j}', bestfit.get('rv_jittervar', 0.0))
        color  = _COLORS[j % len(_COLORS)]
        marker = _SYMBOLS[j % len(_SYMBOLS)]
        label  = labels[j] if labels else f'Tel {j}'

        phase     = np.mod((data['bjd'] - tc) / period + 1.25, 1.0)
        err_total = np.sqrt(data['err']**2 + (jittervar_j or 0.0))

        # Full model (Keplerian + RM) for residuals
        rm_kw_j = _rm_kwargs(bestfit, rmbandndx_list, j)
        model_j = exozippy_rv(data['bjd'], tp, period, 0.0, K,
                              e=e, omega=omega, **rm_kw_j)
        residuals  = data['vel'] - gamma_j - model_j
        plot_vel_j = data['vel'] - gamma_j          # matches EXOFASTv2: residuals + model_j

        all_residuals.append(residuals)
        all_err_total.append(err_total)

        ax_data.errorbar(phase, plot_vel_j, yerr=err_total,
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
                   labels=None, samples=None, rmbandndx_list=None):
    """Unphased RV vs BJD_TDB with O-C residuals.

    Per-telescope model curves include RM where applicable.
    """
    tc     = bestfit['tc']
    period = bestfit['period']
    K      = bestfit['K']
    tp     = tc_to_tp(tc, period, e, omega)

    all_bjd = np.concatenate([d['bjd'] for d in data_list])
    roundto = 10 ** len(str(int(all_bjd.max() - all_bjd.min())))
    bjd0    = np.floor(all_bjd.min() / roundto) * roundto

    # Global Keplerian baseline (thin gray)
    nsteps = max(int((all_bjd.max() - all_bjd.min()) / (period / 100.0)), 500)
    t_fine = np.linspace(all_bjd.min(), all_bjd.max(), nsteps)
    ax_data.plot(t_fine - bjd0,
                 exozippy_rv(t_fine, tp, period, 0.0, K, e=e, omega=omega),
                 '-', color='gray', lw=1.0, zorder=1, alpha=0.5)

    all_residuals = []
    all_err_total = []

    for j, data in enumerate(data_list):
        gamma_j     = bestfit.get(f'gamma_{j}', bestfit.get('gamma', 0.0))
        jittervar_j = bestfit.get(f'jittervar_{j}', bestfit.get('rv_jittervar', 0.0))
        color  = _COLORS[j % len(_COLORS)]
        marker = _SYMBOLS[j % len(_SYMBOLS)]
        label  = labels[j] if labels else f'Tel {j}'

        bjd       = data['bjd']
        vel       = data['vel']
        err_total = np.sqrt(data['err']**2 + (jittervar_j or 0.0))

        rm_kw_j   = _rm_kwargs(bestfit, rmbandndx_list, j)
        model_j   = exozippy_rv(bjd, tp, period, 0.0, K, e=e, omega=omega, **rm_kw_j)
        residuals = vel - gamma_j - model_j

        all_residuals.append(residuals)
        all_err_total.append(err_total)

        ax_data.errorbar(bjd - bjd0, vel - gamma_j, yerr=err_total,
                         fmt=marker, color=color, ms=4, capsize=0,
                         zorder=3, label=label)
        ax_oc.errorbar(bjd - bjd0, residuals, yerr=err_total,
                       fmt=marker, color=color, ms=4, capsize=0)

        # Per-telescope model curve (includes RM for RM telescopes)
        pad = period * 0.1
        t_j = np.linspace(bjd.min() - pad, bjd.max() + pad,
                          max(int((bjd.max() - bjd.min() + 2*pad) / (period / 200.0)), 300))
        ax_data.plot(t_j - bjd0,
                     exozippy_rv(t_j, tp, period, 0.0, K, e=e, omega=omega, **rm_kw_j),
                     '-', color=color, lw=1.5, zorder=2)

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


def _draw_rm_closeup(ax_data, ax_oc, data_list, bestfit, e, omega,
                     labels=None, rmbandndx_list=None):
    """RM close-up panel (mirrors EXOFASTv2 RM panel).

    X-axis : Time − T_C (hrs), range = [−t14, +t14]
    Model  : RM-only delta_rv (prettydeltarv in EXOFASTv2)
    Data   : delta_rv + residuals  (vel − gamma − keplerian, RM reconstructed)

    All RM telescopes are overlaid on the same panel; each night is
    shifted so that its nearest transit center maps to t=0.
    """
    tc     = bestfit['tc']
    period = bestfit['period']
    K      = bestfit['K']
    tp     = tc_to_tp(tc, period, e, omega)
    dur    = _t14(bestfit, e, omega)          # full transit duration (days)
    xrange_hrs = dur * 24.0 * 1.5            # ±1.5 × t14

    # Fine RM model centered on transit
    t_hrs_fine  = np.linspace(-xrange_hrs, xrange_hrs, 500)
    t_fine      = tc + t_hrs_fine / 24.0
    rm_kw_fine  = _first_rm_kwargs(bestfit, rmbandndx_list)
    if not rm_kw_fine:
        return  # nothing to draw

    model_full_fine = exozippy_rv(t_fine, tp, period, 0.0, K,
                                  e=e, omega=omega, **rm_kw_fine)
    model_kep_fine  = exozippy_rv(t_fine, tp, period, 0.0, K, e=e, omega=omega)
    delta_rv_fine   = model_full_fine - model_kep_fine
    ax_data.plot(t_hrs_fine, delta_rv_fine, '-', color='red', lw=2, zorder=2,
                 label='RM model')

    all_residuals = []
    all_err_total = []

    for j, data in enumerate(data_list):
        rm_kw_j = _rm_kwargs(bestfit, rmbandndx_list, j)
        if not rm_kw_j:
            continue  # non-RM telescope — skip

        gamma_j     = bestfit.get(f'gamma_{j}', bestfit.get('gamma', 0.0))
        jittervar_j = bestfit.get(f'jittervar_{j}', bestfit.get('rv_jittervar', 0.0))
        color  = _COLORS[j % len(_COLORS)]
        marker = _SYMBOLS[j % len(_SYMBOLS)]
        label  = labels[j] if labels else f'Tel {j}'

        bjd       = data['bjd']
        err_total = np.sqrt(data['err']**2 + (jittervar_j or 0.0))

        # Nearest transit center for this telescope's data
        epoch_j = round((np.mean(bjd) - tc) / period)
        tc_j    = tc + epoch_j * period
        tp_j    = tc_j   # valid for circular; use tc_to_tp(tc_j,...) if eccentric

        # Per-data-point model
        model_full_j = exozippy_rv(bjd, tp, period, 0.0, K,
                                   e=e, omega=omega, **rm_kw_j)
        model_kep_j  = exozippy_rv(bjd, tp, period, 0.0, K, e=e, omega=omega)
        delta_rv_j   = model_full_j - model_kep_j
        residuals_j  = data['vel'] - gamma_j - model_full_j   # full O-C

        # Data: delta_rv + residuals  (= vel − gamma − keplerian)
        plot_rv_j = delta_rv_j + residuals_j
        t_hrs_j   = (bjd - tc_j) * 24.0

        # Only plot points within the x-range
        mask = np.abs(t_hrs_j) <= xrange_hrs
        if not np.any(mask):
            continue

        ax_data.errorbar(t_hrs_j[mask], plot_rv_j[mask], yerr=err_total[mask],
                         fmt=marker, color=color, ms=4, capsize=0,
                         zorder=3, label=label)
        ax_oc.errorbar(t_hrs_j[mask], residuals_j[mask], yerr=err_total[mask],
                       fmt=marker, color=color, ms=4, capsize=0)

        all_residuals.append(residuals_j[mask])
        all_err_total.append(err_total[mask])

    ax_data.axvline(0, ls=':', color='gray', lw=0.8)   # transit center
    # ingress/egress markers
    ax_data.axvline(-dur * 12.0, ls='--', color='gray', lw=0.8)
    ax_data.axvline( dur * 12.0, ls='--', color='gray', lw=0.8)
    ax_data.set_xlim(-xrange_hrs, xrange_hrs)
    ax_data.legend(fontsize=8, loc='best')
    ax_data.set_ylabel('RM anomaly (m/s)')
    plt.setp(ax_data.get_xticklabels(), visible=False)

    ax_oc.axhline(0, ls='--', color='red', lw=0.8)
    ax_oc.set_xlabel(r'Time $-$ T$_C$ (hrs)')
    ax_oc.set_ylabel('O-C (m/s)')
    ax_oc.set_xlim(-xrange_hrs, xrange_hrs)

    if all_residuals:
        all_res = np.concatenate(all_residuals)
        all_err = np.concatenate(all_err_total)
        ymax_oc = _oc_ylim_rv(all_res, all_err)
        ax_oc.set_ylim(-ymax_oc / 0.7, ymax_oc / 0.7)
        ax_oc.set_yticks([-ymax_oc, 0, ymax_oc])


# ---------- public API ----------

def plotrv(rvfiles, bestfit, samples=None, e=0.0, omega=np.pi / 2,
           outfile=None):
    """
    Combined RV plot — single figure with GridSpec.

    Rows:
        0: Phase-folded RV + O-C  (model includes RM when rossiter=True)
        1: Unphased RV + O-C      (per-telescope model with RM)
        2: RM close-up            (only when rossiter=True)
               x = Time − T_C (hrs), model = RM-only, data = vel−gamma−Kep

    Parameters
    ----------
    rvfiles : str or list[str]
    bestfit : SS or dict-like
    samples : ndarray, optional
    e, omega : float
    outfile : str, optional
    """
    if isinstance(rvfiles, str):
        rvfiles = [rvfiles]

    data_list      = [read_rv_data(f) for f in rvfiles]
    labels         = [_telescope_label(f) for f in rvfiles]
    rmbandndx_list = _extract_rmbandndx_list(bestfit)
    rossiter       = _has_rossiter(bestfit, rmbandndx_list)

    # Layout: 2 main rows + optional RM row
    nrows  = 3 if rossiter else 2
    hrats  = [1, 1, 0.7] if rossiter else [1, 1]
    fig    = plt.figure(figsize=(12, 5 * nrows))
    outer  = gridspec.GridSpec(nrows, 1, figure=fig,
                               height_ratios=hrats,
                               left=0.12, right=0.95,
                               top=0.95, bottom=0.06,
                               hspace=0.40)

    # Row 0: phase-folded
    gs0 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[0],
                                           height_ratios=(3, 1), hspace=0.0)
    ax0  = fig.add_subplot(gs0[0])
    ax0r = fig.add_subplot(gs0[1], sharex=ax0)
    _draw_phased(ax0, ax0r, data_list, bestfit, e, omega,
                 labels=labels, samples=samples,
                 rmbandndx_list=rmbandndx_list)

    # Row 1: unphased
    gs1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[1],
                                           height_ratios=(3, 1), hspace=0.0)
    ax1  = fig.add_subplot(gs1[0])
    ax1r = fig.add_subplot(gs1[1], sharex=ax1)
    _draw_unphased(ax1, ax1r, data_list, bestfit, e, omega,
                   labels=labels, samples=samples,
                   rmbandndx_list=rmbandndx_list)

    # Row 2: RM close-up (only when rossiter)
    if rossiter:
        gs2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[2],
                                               height_ratios=(3, 1), hspace=0.0)
        ax2  = fig.add_subplot(gs2[0])
        ax2r = fig.add_subplot(gs2[1], sharex=ax2)
        _draw_rm_closeup(ax2, ax2r, data_list, bestfit, e, omega,
                         labels=labels, rmbandndx_list=rmbandndx_list)

    save_or_show(fig, outfile, label='RV')

    return fig
