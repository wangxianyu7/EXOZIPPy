"""
MCMC diagnostic plots for EXOZIPPy using arviz.

Generates:
  - Corner plot (pair plot with marginals + 2D contours)
  - Trace plot (parameter traces + marginal posteriors per chain)

Usage:
    from exozippy.plotting.mcmc import plot_corner, plot_trace, chain_to_arviz
    idata = chain_to_arviz(chain_info)
    plot_corner(idata, outfile='corner.png')
    plot_trace(idata, outfile='trace.png')
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .common import save_or_show

try:
    import arviz as az
    HAS_ARVIZ = True
except ImportError:
    HAS_ARVIZ = False


# LaTeX-friendly labels for common parameters
_LATEX_LABELS = {
    'teff': r'$T_{\rm eff}$ [K]',
    'rstar': r'$R_*$ [$R_\odot$]',
    'feh': r'[Fe/H]',
    'av': r'$A_V$',
    'distance': r'$d$ [pc]',
    'mstar': r'$M_*$ [$M_\odot$]',
    'age': r'Age [Gyr]',
    'tc': r'$T_C$ [BJD]',
    'period': r'$P$ [d]',
    'p': r'$R_p/R_*$',
    'cosi': r'$\cos i$',
    'K': r'$K$ [m/s]',
    'logg': r'$\log g$',
    'lstar': r'$L_*$ [$L_\odot$]',
    'ar': r'$a/R_*$',
    'ideg': r'$i$ [deg]',
    'b': r'$b$',
    'parallax': r'$\varpi$ [mas]',
}


def _get_label(name):
    """Return a nice label for a parameter name."""
    # Check exact match
    if name in _LATEX_LABELS:
        return _LATEX_LABELS[name]
    # Check base name for indexed params like u1_0, f0_1, gamma_0
    base = name.rsplit('_', 1)[0] if '_' in name else name
    idx = name.rsplit('_', 1)[1] if '_' in name else ''
    if base == 'u1':
        return rf'$u_1^{{{idx}}}$'
    if base == 'u2':
        return rf'$u_2^{{{idx}}}$'
    if base == 'f0':
        return rf'$F_0^{{{idx}}}$'
    if base == 'gamma':
        return rf'$\gamma_{{{idx}}}$ [m/s]'
    if base == 'jittervar':
        return rf'$\sigma_J^{{{idx},2}}$'
    return name


def chain_to_arviz(chain_info, burn_frac=0.25):
    """
    Convert DEMC-PT chain output to an arviz InferenceData object.

    Parameters
    ----------
    chain_info : dict
        From run_mcmc: keys 'chain' (nsteps, nchains, ndim),
        'log_prob' (nsteps, nchains), 'param_names' (list[str]).
    burn_frac : float
        Fraction of steps to discard as burn-in (default 0.25).

    Returns
    -------
    arviz.InferenceData
    """
    if not HAS_ARVIZ:
        raise ImportError("arviz is required for MCMC diagnostic plots. "
                          "Install with: pip install arviz")

    chain = chain_info['chain']       # (nsteps, nchains, ndim)
    log_prob = chain_info['log_prob'] # (nsteps, nchains)
    param_names = chain_info['param_names']

    nsteps, nchains, ndim = chain.shape
    burn = max(int(nsteps * burn_frac), 1)

    # Post-burn-in: (draw, chain, ndim)
    post_chain = chain[burn:]  # (nsteps-burn, nchains, ndim)

    # arviz expects dict of {param_name: array(chain, draw)}
    posterior_dict = {}
    for i, name in enumerate(param_names):
        # Shape: (nchains, ndraws) — arviz convention
        posterior_dict[name] = post_chain[:, :, i].T

    # Sample stats (log_likelihood proxy)
    sample_stats = {
        'lp': log_prob[burn:].T,  # (nchains, ndraws)
    }

    idata = az.from_dict(
        posterior=posterior_dict,
        sample_stats=sample_stats,
    )
    return idata


def plot_corner(chain_info_or_idata, var_names=None, outfile=None,
                burn_frac=0.25, figsize=None, **kwargs):
    """
    Corner plot (pair plot with marginals + 2D KDE contours).

    Parameters
    ----------
    chain_info_or_idata : dict or arviz.InferenceData
        Either chain_info dict from run_mcmc, or pre-built InferenceData.
    var_names : list[str], optional
        Parameter names to include. Default: all fitted parameters.
    outfile : str, optional
        Output file path (.png or .pdf).
    burn_frac : float
        Burn-in fraction (only used if chain_info dict is passed).
    """
    if not HAS_ARVIZ:
        raise ImportError("arviz is required. Install with: pip install arviz")

    if isinstance(chain_info_or_idata, dict):
        idata = chain_to_arviz(chain_info_or_idata, burn_frac=burn_frac)
        param_names = chain_info_or_idata['param_names']
    else:
        idata = chain_info_or_idata
        param_names = list(idata.posterior.data_vars)

    if var_names is None:
        var_names = param_names

    # Build labeller
    labeller_map = {name: _get_label(name) for name in var_names}
    labeller = az.labels.MapLabeller(var_name_map=labeller_map)

    nvar = len(var_names)
    if figsize is None:
        size = max(2 * nvar, 8)
        figsize = (size, size)

    axes = az.plot_pair(
        idata,
        var_names=var_names,
        kind='kde',
        kde_kwargs={'contourf_kwargs': {'cmap': 'Blues'},
                    'contour_kwargs': {'colors': 'k', 'linewidths': 0.5}},
        marginals=True,
        marginal_kwargs={'color': 'steelblue'},
        figsize=figsize,
        labeller=labeller,
        **kwargs,
    )

    fig = axes.ravel()[0].get_figure() if hasattr(axes, 'ravel') else plt.gcf()
    fig.tight_layout()

    save_or_show(fig, outfile, label='corner')

    return fig


def plot_trace(chain_info_or_idata, var_names=None, outfile=None,
               burn_frac=0.25, figsize=None, **kwargs):
    """
    Trace plot: parameter value vs step number + marginal posterior per chain.

    Parameters
    ----------
    chain_info_or_idata : dict or arviz.InferenceData
        Either chain_info dict from run_mcmc, or pre-built InferenceData.
    var_names : list[str], optional
        Parameter names to include. Default: all fitted parameters.
    outfile : str, optional
        Output file path (.png or .pdf).
    burn_frac : float
        Burn-in fraction (only used if chain_info dict is passed).
    """
    if not HAS_ARVIZ:
        raise ImportError("arviz is required. Install with: pip install arviz")

    if isinstance(chain_info_or_idata, dict):
        idata = chain_to_arviz(chain_info_or_idata, burn_frac=burn_frac)
        param_names = chain_info_or_idata['param_names']
    else:
        idata = chain_info_or_idata
        param_names = list(idata.posterior.data_vars)

    if var_names is None:
        var_names = param_names

    labeller_map = {name: _get_label(name) for name in var_names}
    labeller = az.labels.MapLabeller(var_name_map=labeller_map)

    nvar = len(var_names)
    if figsize is None:
        figsize = (14, max(2.5 * nvar, 6))

    axes = az.plot_trace(
        idata,
        var_names=var_names,
        figsize=figsize,
        labeller=labeller,
        compact=True,
        **kwargs,
    )

    fig = axes.ravel()[0].get_figure() if hasattr(axes, 'ravel') else plt.gcf()
    fig.tight_layout()

    save_or_show(fig, outfile, label='trace')

    return fig
