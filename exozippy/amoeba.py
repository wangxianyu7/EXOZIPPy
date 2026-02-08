"""Python implementation of EXOFAST's amoeba (Nelder-Mead) optimizer."""

from __future__ import annotations

import numpy as np


def _amotry(simplex, values, psum, func, ihi, fac):
    ndim = simplex.shape[0]
    fac1 = (1.0 - fac) / ndim
    fac2 = fac1 - fac
    ptry = psum * fac1 - simplex[:, ihi] * fac2
    ytry = func(ptry)
    if ytry < values[ihi]:
        values[ihi] = ytry
        psum[:] = psum + ptry - simplex[:, ihi]
        simplex[:, ihi] = ptry
    return ytry


def amoeba(func, x0=None, scale=None, simplex=None, ftol=1e-4, maxiter=5000,
           verbose=False):
    """Run the Nelder–Mead simplex algorithm.

    Parameters
    ----------
    func : callable
        Function to minimize. Receives a 1-D numpy array and returns a scalar.
    x0 : array-like, optional
        Starting point. Used with ``scale`` to build the initial simplex.
    scale : float or array-like, optional
        Characteristic step size for each dimension.
    simplex : array-like, shape (ndim, ndim+1), optional
        Explicit simplex. Overrides ``x0``/``scale`` if provided.
    ftol : float
        Fractional tolerance for convergence.
    maxiter : int
        Maximum number of function evaluations.
    verbose : bool
        If True, prints periodic progress updates.

    Returns
    -------
    best_x : ndarray
        Parameters at the minimum.
    best_y : float
        Minimum function value.
    data : dict
        Diagnostic information (simplex, values, nfev, success).
    """

    if simplex is None:
        if x0 is None or scale is None:
            raise ValueError("Provide either simplex or (x0, scale)")
        x0 = np.asarray(x0, dtype=float)
        scale = np.broadcast_to(np.asarray(scale, dtype=float), x0.shape)
        ndim = x0.size
        simplex = np.tile(x0[:, None], ndim + 1)
        for i in range(ndim):
            simplex[i, i + 1] = x0[i] + scale[i]
    else:
        simplex = np.asarray(simplex, dtype=float)
        if simplex.ndim != 2:
            raise ValueError("Simplex must be 2-D (ndim, ndim+1)")
        ndim = simplex.shape[0]

    mpts = ndim + 1
    values = np.empty(mpts)
    for i in range(mpts):
        values[i] = func(simplex[:, i])
    psum = np.sum(simplex, axis=1)
    nfev = mpts

    while True:
        idx = np.argsort(values)
        ilo = idx[0]
        ihi = idx[-1]
        inhi = idx[-2]
        y_lo = values[ilo]
        y_hi = values[ihi]
        denom = abs(y_hi) + abs(y_lo)
        if denom == 0:
            rtol = ftol / 2.0
        else:
            rtol = 2.0 * abs(y_hi - y_lo) / denom
        if verbose and (nfev % 100 == 0 or rtol < ftol):
            print(f"Amoeba: rtol={rtol:.3e}, nfev={nfev}, chi2={y_lo:.3f}")
        if rtol < ftol or nfev >= maxiter:
            simplex[:, [0, ilo]] = simplex[:, [ilo, 0]]
            values[[0, ilo]] = values[[ilo, 0]]
            success = rtol < ftol
            break

        nfev += 2
        ytry = _amotry(simplex, values, psum, func, ihi, -1.0)
        if ytry <= values[ilo]:
            _amotry(simplex, values, psum, func, ihi, 2.0)
        elif ytry >= values[inhi]:
            ysave = values[ihi]
            ytry = _amotry(simplex, values, psum, func, ihi, 0.5)
            if ytry >= ysave:
                for i in range(mpts):
                    if i == ilo:
                        continue
                    simplex[:, i] = 0.5 * (simplex[:, i] + simplex[:, ilo])
                    values[i] = func(simplex[:, i])
                nfev += ndim
                psum = np.sum(simplex, axis=1)
        psum = np.sum(simplex, axis=1)

    result = {
        'simplex': simplex,
        'values': values,
        'nfev': nfev,
        'success': success,
        'rtol': rtol,
    }
    return simplex[:, 0], values[0], result
