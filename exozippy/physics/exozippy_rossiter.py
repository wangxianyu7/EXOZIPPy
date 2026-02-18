"""
Rossiter-McLaughlin effect — Hirano et al. 2011
================================================

Computes the RM anomalous radial velocity shift using the full analytic
formalism of Hirano et al. (2011, ApJ, 742, 69).  The expensive M̃(σ)
disk integral uses Gauss-Legendre quadrature (GL-32) with adaptive
σ_max, which keeps the oscillation count bounded for any vsini.

Ported from pyRM/src/hirano2011.py + filon_prototype.py.
Author: Xian-Yu Wang
"""

import functools
import numpy as np
from scipy.special import j0 as _scipy_j0
from scipy.integrate import simpson as _simpson

from ..utils.kepler import exozippy_keplereq
from .exozippy_tran import exozippy_tran


# ── GL-32 nodes/weights (cached) ───────────────────────────────────────

@functools.lru_cache(maxsize=16)
def _gl_nodes_weights(N=32):
    """Gauss-Legendre nodes/weights on [0, 1], cached."""
    nodes, weights = np.polynomial.legendre.leggauss(N)
    return 0.5 * (nodes + 1.0), 0.5 * weights


# ── Smooth envelope g(t) ───────────────────────────────────────────────

def _g_func(t, sigma, vsini_kms, u1, u2, zeta_kms):
    """Smooth part of M̃ integrand: limb × macro-turb × t  (no J₀)."""
    sqrt_1mt2 = np.sqrt(np.maximum(0.0, 1.0 - t * t))
    limb = (1.0 - u1 * (1.0 - sqrt_1mt2)
            - u2 * (1.0 - sqrt_1mt2)**2) / (1.0 - u1 / 3.0 - u2 / 6.0)
    s2 = np.pi**2 * zeta_kms**2 * sigma**2
    exp_macro = np.exp(-s2 * (1.0 - t * t)) + np.exp(-s2 * t * t)
    return limb * exp_macro * t


# ── M̃(σ) via Gauss-Legendre ───────────────────────────────────────────

def _M_sigma_gl(sigma, vsini_kms, u1, u2, zeta_kms, N=32):
    """Compute M̃(σ) = ∫₀¹ g(t) J₀(2πσ·vsini·t) dt via GL quadrature."""
    omega = 2.0 * np.pi * sigma * vsini_kms
    t, w = _gl_nodes_weights(N)
    g = _g_func(t, sigma, vsini_kms, u1, u2, zeta_kms)
    j0_vals = _scipy_j0(omega * t)
    return np.dot(w, g * j0_vals)


def _compute_M_array(sigma_array, vsini_kms, u1, u2, zeta_kms):
    """M̃(σ) for an array of σ values.  Adaptive GL order."""
    n_sigma = len(sigma_array)
    M = np.empty(n_sigma)
    for k in range(n_sigma):
        omega = 2.0 * np.pi * sigma_array[k] * vsini_kms
        n_osc = omega / (2.0 * np.pi)
        N = max(32, int(4 * n_osc + 16))
        N = min(N, 512)
        M[k] = _M_sigma_gl(sigma_array[k], vsini_kms, u1, u2, zeta_kms, N=N)
    return M


# ── Planet sky-plane position with λ rotation ──────────────────────────

def _planet_xy(trueanom, e, omega, ar, inc, lam):
    """Planet (x, y) in units of R★, rotated by spin-orbit angle λ.

    x is along projected stellar spin axis (v_sub = vsini × x).
    """
    r = ar * (1.0 - e**2) / (1.0 + e * np.cos(trueanom))
    x_old = -r * np.cos(trueanom + omega)
    y_old = -r * np.sin(trueanom + omega) * np.cos(inc)
    cos_lam = np.cos(lam)
    sin_lam = np.sin(lam)
    x = x_old * cos_lam - y_old * sin_lam
    y = x_old * sin_lam + y_old * cos_lam
    z = r * np.sin(trueanom + omega) * np.sin(inc)
    return x, y, z


# ── Outer σ integral (vectorised over time) ────────────────────────────

def _rm_delta_v(flux, v_sub, cos_thetas, sin_thetas,
                M_array, sigma_array, zeta_kms, beta_kms, gamma_kms):
    """Compute Δv for all time points (vectorised, pure NumPy).

    Returns delta_v in km/s (caller multiplies by 1e3 for m/s).
    """
    n_time = len(flux)
    n_sigma = len(sigma_array)
    delta_v = np.zeros(n_time)

    # Pre-compute σ-dependent terms (shared across time)
    sigma2 = sigma_array**2
    exp_broad = np.exp(-2.0 * np.pi**2 * beta_kms**2 * sigma2
                       - 4.0 * np.pi * gamma_kms * sigma_array)
    eM = exp_broad * M_array  # (n_sigma,)

    for i in range(n_time):
        f_ = 1.0 - flux[i]
        if f_ <= 0.0:
            continue
        vp = v_sub[i]
        ct = cos_thetas[i]
        st = sin_thetas[i]

        # Θ̃ kernel
        Theta = 0.5 * (np.exp(-(np.pi * zeta_kms * ct)**2 * sigma2)
                       + np.exp(-(np.pi * zeta_kms * st)**2 * sigma2))

        sin_term = np.sin(2.0 * np.pi * vp * sigma_array)
        cos_term = np.cos(2.0 * np.pi * vp * sigma_array)

        numer = eM * Theta * sin_term * sigma_array
        denom = eM * (M_array - f_ * Theta * cos_term) * sigma2

        num_int = _simpson(numer, x=sigma_array)
        den_int = _simpson(denom, x=sigma_array)

        if abs(den_int) > 1e-30:
            delta_v[i] = f_ / (2.0 * np.pi) * num_int / den_int

    return delta_v


# ── Public entry point ─────────────────────────────────────────────────

def exozippy_rossiter(bjd, tp, period, e, omega, inc, ar, p, u1, u2,
                      vsini, lam, vgamma, vzeta, vxi, valpha):
    """Hirano 2011 RM velocity anomaly.

    Parameters
    ----------
    bjd : array
        Observation times (BJD).
    tp : float
        Time of periastron (BJD).
    period : float
        Orbital period (days).
    e, omega : float
        Eccentricity and argument of periastron (radians).
    inc : float
        Orbital inclination (radians).
    ar : float
        Scaled semi-major axis a/R★.
    p : float
        Planet-to-star radius ratio Rp/R★.
    u1, u2 : float
        Quadratic limb-darkening coefficients for the RM band.
    vsini : float
        Projected stellar rotation velocity (m/s).
    lam : float
        Projected spin-orbit angle λ (radians).
    vgamma : float
        Lorentzian (natural/pressure) line broadening (m/s).
    vzeta : float
        Macroturbulent velocity (m/s).
    vxi : float
        Microturbulent velocity (m/s).
    valpha : float
        Extra broadening (m/s).

    Returns
    -------
    delta_rv : array
        RM velocity anomaly in m/s at each bjd.
    """
    bjd = np.atleast_1d(np.asarray(bjd, dtype=float))

    # Convert all velocities from m/s → km/s for Hirano formulae
    vsini_kms = vsini / 1e3
    zeta_kms = vzeta / 1e3
    # β combines thermal broadening + microturbulence + extra broadening
    beta_kms = np.sqrt((valpha / 1e3)**2 + (vxi / 1e3)**2)
    gamma_kms = vgamma / 1e3

    # Kepler equation → true anomaly
    meananom = 2.0 * np.pi * (1.0 + np.mod((bjd - tp) / period, 1.0))
    if e > 0:
        eccanom = exozippy_keplereq(meananom, e)
        trueanom = 2.0 * np.arctan(
            np.sqrt((1.0 + e) / (1.0 - e)) * np.tan(eccanom / 2.0))
    else:
        trueanom = meananom

    # Planet sky-plane position with λ rotation
    x, y, z = _planet_xy(trueanom, e, omega, ar, inc, lam)

    # Sub-planet velocity (km/s)
    v_sub = vsini_kms * x

    # Transit flux via EXOZIPPy transit model (f0=1)
    flux = exozippy_tran(bjd, inc, ar, tp, period, e, omega, abs(p),
                         u1, u2, 1.0)

    # cos(θ), sin(θ) at sub-planet point on stellar disk
    r2 = x**2 + y**2
    cos_thetas = np.sqrt(np.maximum(0.0, 1.0 - r2))
    sin_thetas = np.sqrt(np.maximum(0.0, r2))

    # σ grid — adaptive σ_max keeps ω_max bounded
    sigma_max = max(5.0 / (vsini_kms + zeta_kms + 0.1), 0.01)
    n_sigma = 101  # odd for Simpson
    sigma_array = np.logspace(-6, np.log10(sigma_max), n_sigma)

    # M̃(σ) via GL quadrature
    M_array = _compute_M_array(sigma_array, vsini_kms, u1, u2, zeta_kms)

    # Outer σ integral → Δv (km/s)
    delta_v = _rm_delta_v(flux, v_sub, cos_thetas, sin_thetas,
                          M_array, sigma_array, zeta_kms, beta_kms, gamma_kms)

    # Zero out-of-transit points (planet behind star: z < 0)
    delta_v[z < 0] = 0.0

    # Convert km/s → m/s
    return delta_v * 1e3
