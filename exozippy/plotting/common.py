"""Shared helpers for plotting modules."""

import os

import matplotlib.pyplot as plt
import numpy as np

DEFAULT_COLORS = ['black', 'red', 'blue', 'darkgreen', 'orange', 'purple', 'cyan', 'magenta']
DEFAULT_SYMBOLS = ['o', 's', 'D', '^', 'v', '<', '>', 'p']


def short_label(filepath):
    """Return basename as a compact legend label."""
    return os.path.basename(filepath)


def rounded_symmetric_limit(values, scale=1.1, sig_digits=2, fallback=1.0):
    """Symmetric y-limit rounded to the requested significant digits."""
    arr = np.asarray(values, dtype=float)
    finite = np.abs(arr[np.isfinite(arr)])
    if finite.size == 0:
        return fallback
    ymax = finite.max() * scale
    if ymax == 0:
        return fallback
    ndigits = np.floor(np.log10(ymax)) - (sig_digits - 1)
    return np.round(ymax / 10 ** ndigits) * 10 ** ndigits


def rounded_symmetric_limit_with_errors(residuals, errors, scale=1.1, sig_digits=2, fallback=1.0):
    """Symmetric y-limit using |residual| + error envelope."""
    residuals = np.asarray(residuals, dtype=float)
    errors = np.asarray(errors, dtype=float)
    finite = np.isfinite(residuals) & np.isfinite(errors)
    if not np.any(finite):
        return fallback
    ymax = np.max(np.abs(residuals[finite]) + errors[finite]) * scale
    if ymax == 0:
        return fallback
    ndigits = np.floor(np.log10(ymax)) - (sig_digits - 1)
    return np.round(ymax / 10 ** ndigits) * 10 ** ndigits


def ceil_step_limit(values, step=0.5, scale=1.1, fallback=1.0):
    """Symmetric y-limit rounded up to a fixed step size."""
    arr = np.asarray(values, dtype=float)
    finite = np.abs(arr[np.isfinite(arr)])
    if finite.size == 0:
        return fallback
    ymax = finite.max() * scale
    if ymax == 0:
        return fallback
    return np.ceil(ymax / step) * step


def save_or_show(fig, outfile=None, label='plot'):
    """Save figure when outfile is set, else show interactively."""
    if outfile is not None:
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'Saved {label} plot to {outfile}')
    else:
        plt.show()

