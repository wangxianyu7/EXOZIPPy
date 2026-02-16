"""
Generate an updated prior file after fitting/MCMC.

Mirrors EXOFASTv2's ``mkprior2.pro``: for each parameter line in the
original prior file, replace the starting value with the best-fit value
while preserving gaussian widths, bounds, and comments.

The output filename increments a numeric suffix:
    wasp18.priors   → wasp18.priors.2
    wasp18.priors.2 → wasp18.priors.3
"""

import re
from pathlib import Path

import numpy as np


def _next_priorfile(parfile):
    """Compute the next prior filename by incrementing a numeric suffix.

    wasp18.priors     → wasp18.priors.2
    wasp18.priors.2   → wasp18.priors.3
    """
    p = Path(parfile)
    suffix = p.suffix.lstrip('.')
    if suffix.isdigit():
        new_suffix = str(int(suffix) + 1)
        return str(p.with_suffix(f'.{new_suffix}'))
    else:
        return str(p) + '.2'


def _get_bestval(ss, name):
    """Try to retrieve the best-fit value for *name* from the SS object.

    Returns (value, found) tuple.
    """
    try:
        val = ss[name]
        if val is not None and np.isfinite(val):
            return val, True
    except (KeyError, IndexError, AttributeError):
        pass
    return None, False


def _format_val(val, name):
    """Format a numeric value with appropriate precision.

    High-precision for time-like params (tc, period); moderate for others.
    """
    if name.startswith('tc') or name.startswith('period'):
        return f'{val:.10f}'
    elif name.startswith('gamma') or name.startswith('jittervar'):
        return f'{val:.8f}'
    elif name.startswith('parallax'):
        return f'{val:.5f}'
    elif name.startswith('f0'):
        return f'{val:.7f}'
    elif abs(val) > 100:
        return f'{val:.5f}'
    elif abs(val) > 1:
        return f'{val:.6f}'
    else:
        return f'{val:.8f}'


def mkprior2(parfile, ss, outfile=None, verbose=True):
    """Write an updated prior file using best-fit values from *ss*.

    Parameters
    ----------
    parfile : str or Path
        Path to the original prior file.
    ss : SS
        The stellar system object containing best-fit parameter values.
    outfile : str or Path, optional
        Output path. If None, auto-increments the suffix.
    verbose : bool
        Print the output path.

    Returns
    -------
    str
        Path to the written prior file.
    """
    parfile = str(parfile)
    if outfile is None:
        outfile = _next_priorfile(parfile)

    lines_out = []
    with open(parfile) as f:
        for line in f:
            raw = line.rstrip('\n')

            # Preserve blank lines and pure-comment lines
            stripped = raw.lstrip()
            if not stripped or stripped.startswith('#'):
                lines_out.append(raw)
                continue

            # Split off inline comment
            if '#' in raw:
                code_part, comment_part = raw.split('#', 1)
                comment_part = '  #' + comment_part
            else:
                code_part = raw
                comment_part = ''

            parts = code_part.split()
            if len(parts) < 2:
                lines_out.append(raw)
                continue

            name = parts[0]
            vals = []
            for v in parts[1:]:
                try:
                    vals.append(float(v))
                except ValueError:
                    break

            if not vals:
                lines_out.append(raw)
                continue

            # Try to get best-fit value from SS
            bestval, found = _get_bestval(ss, name)

            if found:
                # Build updated line: name bestval [sigma [lower upper]]
                new_parts = [name, _format_val(bestval, name)]

                if len(vals) >= 2:
                    # Preserve sigma
                    sigma = vals[1]
                    new_parts.append(f'{sigma}')
                if len(vals) >= 3:
                    # Preserve lower bound
                    new_parts.append(f'{vals[2]}')
                if len(vals) >= 4:
                    # Preserve upper bound
                    new_parts.append(f'{vals[3]}')

                lines_out.append(' '.join(new_parts) + comment_part)
            else:
                # Parameter not in SS (e.g. variance, jittervar) — keep original
                lines_out.append(raw)

    with open(outfile, 'w') as f:
        f.write('\n'.join(lines_out) + '\n')

    if verbose:
        print(f'Updated priors: {outfile}')

    return outfile
