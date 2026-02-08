"""
SED-only fitting for EXOZIPPy.

Fits stellar parameters (Teff, Rstar, [Fe/H], Av, distance) to broadband
photometry using MIST bolometric correction tables via mistmultised.
"""
import numpy as np
from scipy.optimize import minimize

from exozippy.sed.utils import mistmultised
from exozippy.mkconstants import mkconstants

CONSTANTS = mkconstants()


def parse_priors(priorfile):
    """
    Parse an EXOFASTv2-style prior file.

    Returns
    -------
    priors : dict
        Keys are parameter names; values are dicts with fields:
        'value', 'sigma' (0 if none), 'lower', 'upper' (NaN if none).
    """
    priors = {}
    with open(priorfile) as f:
        for line in f:
            line = line.split('#')[0].strip()
            if not line:
                continue
            parts = line.split()
            name = parts[0]
            vals = [float(x) for x in parts[1:]]
            entry = {'value': vals[0], 'sigma': 0.0,
                     'lower': np.nan, 'upper': np.nan}
            if len(vals) >= 2:
                entry['sigma'] = vals[1]
            if len(vals) >= 3:
                entry['lower'] = vals[2]
            if len(vals) >= 4:
                entry['upper'] = vals[3]
            priors[name] = entry
    return priors


def _derive_logg(mstar, rstar):
    """Compute log(g) in cgs from Mstar (Msun) and Rstar (Rsun)."""
    g = CONSTANTS['GravitySun'] * mstar / rstar**2   # cm/s^2
    return np.log10(g)


def _derive_lstar(teff, rstar):
    """Compute Lstar in solar luminosities from Teff and Rstar."""
    return (4.0 * np.pi * (rstar * CONSTANTS['RSun'])**2
            * CONSTANTS['sigmab'] * teff**4 / CONSTANTS['LSun'])


def sed_negloglike(params, sedfile, priors, mstar):
    """
    Total negative-log-likelihood for the SED fit.

    Parameters
    ----------
    params : array, shape (5,)
        [teff, rstar, feh, av, distance]
    sedfile : str
    priors : dict from parse_priors
    mstar : float (Msun, fixed)

    Returns
    -------
    float
        -2 * ln(likelihood) = sed_chi2 + prior_chi2
    """
    teff, rstar, feh, av, distance = params

    # Bounds enforcement (return large penalty if out of bounds)
    if teff < 2500 or teff > 50000:
        return 1e10
    if rstar < 0.05 or rstar > 100:
        return 1e10
    if av < 0:
        return 1e10
    if distance < 1:
        return 1e10

    # Derived quantities
    logg = _derive_logg(mstar, rstar)
    lstar = _derive_lstar(teff, rstar)
    errscale = 1.0

    # SED chi2 (matches IDL exofast_like /chi2)
    try:
        sedchi2, blendmag, modelflux, magresiduals = mistmultised(
            teff, logg, feh, av, distance, lstar, errscale, sedfile
        )
    except Exception:
        return 1e10

    if not np.isfinite(sedchi2):
        return 1e10

    # Prior chi2 terms
    prior_chi2 = 0.0

    # [Fe/H] Gaussian prior
    if 'feh' in priors and priors['feh']['sigma'] > 0:
        prior_chi2 += ((feh - priors['feh']['value'])
                       / priors['feh']['sigma'])**2

    # Parallax Gaussian prior (parallax = 1000/distance in mas)
    if 'parallax' in priors and priors['parallax']['sigma'] > 0:
        plx_model = 1000.0 / distance
        prior_chi2 += ((plx_model - priors['parallax']['value'])
                       / priors['parallax']['sigma'])**2

    # Av upper bound (soft wall)
    if 'av' in priors and np.isfinite(priors['av']['upper']):
        if av > priors['av']['upper']:
            prior_chi2 += ((av - priors['av']['upper']) / 0.001)**2

    return sedchi2 + prior_chi2


def fit_sed(priorfile, sedfile, verbose=True):
    """
    Fit SED photometry to determine stellar parameters.

    Parameters
    ----------
    priorfile : str
        Path to EXOFASTv2-style prior file.
    sedfile : str
        Path to SED data file.
    verbose : bool

    Returns
    -------
    result : dict
        Best-fit parameters, chi2, derived quantities.
    """
    priors = parse_priors(priorfile)

    # Fixed mass
    mstar = priors.get('mstar', {}).get('value', 1.0)

    # Starting values from priors
    teff0 = priors.get('teff', {}).get('value', 5500.0)
    rstar0 = priors.get('rstar', {}).get('value', 1.0)
    feh0 = priors.get('feh', {}).get('value', 0.0)
    av0 = 0.01  # small positive start
    if 'parallax' in priors:
        dist0 = 1000.0 / priors['parallax']['value']
    else:
        dist0 = 100.0

    x0 = np.array([teff0, rstar0, feh0, av0, dist0])

    if verbose:
        print("=== SED Fitting ===")
        print(f"Fixed: mstar = {mstar:.3f} Msun")
        print(f"Starting: Teff={teff0:.0f} K, Rstar={rstar0:.3f} Rsun, "
              f"[Fe/H]={feh0:.3f}, Av={av0:.3f}, dist={dist0:.2f} pc")
        chi2_init = sed_negloglike(x0, sedfile, priors, mstar)
        print(f"Initial chi2 = {chi2_init:.2f}")
        print()

    # Av upper bound from priors
    av_upper = priors.get('av', {}).get('upper', 1.0)
    if not np.isfinite(av_upper):
        av_upper = 1.0

    # Bounds for L-BFGS-B
    bounds = [
        (3000, 10000),      # teff
        (0.1, 10.0),        # rstar
        (-2.0, 0.75),       # feh
        (0.0, av_upper),    # av
        (1.0, 10000.0),     # distance
    ]

    # Phase 1: Nelder-Mead (no gradients needed, robust)
    res_nm = minimize(sed_negloglike, x0,
                      args=(sedfile, priors, mstar),
                      method='Nelder-Mead',
                      options={'maxiter': 10000, 'xatol': 1e-6,
                               'fatol': 1e-6, 'adaptive': True})

    # Phase 2: Polish with L-BFGS-B (respects bounds)
    res = minimize(sed_negloglike, res_nm.x,
                   args=(sedfile, priors, mstar),
                   method='L-BFGS-B', bounds=bounds,
                   options={'maxiter': 5000, 'ftol': 1e-12})

    teff_f, rstar_f, feh_f, av_f, dist_f = res.x
    logg_f = _derive_logg(mstar, rstar_f)
    lstar_f = _derive_lstar(teff_f, rstar_f)

    result = {
        'teff': teff_f,
        'rstar': rstar_f,
        'feh': feh_f,
        'av': av_f,
        'distance': dist_f,
        'parallax': 1000.0 / dist_f,
        'logg': logg_f,
        'lstar': lstar_f,
        'mstar': mstar,
        'chi2': res.fun,
        'success': res.success,
        'message': res.message,
    }

    if verbose:
        print("=== Best-Fit Results ===")
        print(f"Teff     = {teff_f:.1f} K")
        print(f"Rstar    = {rstar_f:.4f} Rsun")
        print(f"[Fe/H]   = {feh_f:.4f}")
        print(f"Av       = {av_f:.4f}")
        print(f"Distance = {dist_f:.2f} pc")
        print(f"Parallax = {1000.0/dist_f:.4f} mas")
        print(f"logg     = {logg_f:.4f}")
        print(f"Lstar    = {lstar_f:.4f} Lsun")
        print(f"Chi2     = {res.fun:.4f}")
        print(f"Success  = {res.success}")
        print()

    return result


def run_mcmc(priorfile, sedfile, bestfit=None, nwalkers=32, nsteps=5000,
             nburn=1000, verbose=True):
    """
    Run MCMC sampling around the best-fit SED solution.

    Parameters
    ----------
    priorfile, sedfile : str
    bestfit : dict or None
        If None, runs fit_sed first.
    nwalkers, nsteps, nburn : int
    verbose : bool

    Returns
    -------
    samples : ndarray, shape (nwalkers*(nsteps-nburn), 5)
    labels : list of str
    summary : dict
    """
    import emcee

    priors = parse_priors(priorfile)
    mstar = priors.get('mstar', {}).get('value', 1.0)

    if bestfit is None:
        bestfit = fit_sed(priorfile, sedfile, verbose=verbose)

    x_best = np.array([bestfit['teff'], bestfit['rstar'], bestfit['feh'],
                        bestfit['av'], bestfit['distance']])
    labels = ['Teff', 'Rstar', 'feh', 'Av', 'distance']

    ndim = len(x_best)

    # Initialize walkers in a small ball around best-fit
    scales = np.array([50.0, 0.02, 0.05, 0.005, 1.0])
    pos = x_best + scales * np.random.randn(nwalkers, ndim)

    # Enforce positive Av
    pos[:, 3] = np.abs(pos[:, 3])

    def log_prob(params):
        chi2 = sed_negloglike(params, sedfile, priors, mstar)
        if not np.isfinite(chi2) or chi2 > 1e9:
            return -np.inf
        return -0.5 * chi2  # chi2 is already -2*lnL, so lnL = -chi2/2

    if verbose:
        print(f"=== MCMC: {nwalkers} walkers, {nsteps} steps ===")

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    sampler.run_mcmc(pos, nsteps, progress=verbose)

    # Discard burn-in and flatten
    samples = sampler.get_chain(discard=nburn, flat=True)

    # Derived parameters
    logg_samples = np.array([_derive_logg(mstar, s[1]) for s in samples])
    lstar_samples = np.array([_derive_lstar(s[0], s[1]) for s in samples])
    plx_samples = 1000.0 / samples[:, 4]

    summary = {}
    for i, name in enumerate(labels):
        med = np.median(samples[:, i])
        lo = np.percentile(samples[:, i], 15.87)
        hi = np.percentile(samples[:, i], 84.13)
        summary[name] = (med, med - lo, hi - med)
        if verbose:
            print(f"{name:10s} = {med:.4f}  -{med-lo:.4f}  +{hi-med:.4f}")

    # Derived
    for name, arr in [('logg', logg_samples), ('Lstar', lstar_samples),
                      ('parallax', plx_samples)]:
        med = np.median(arr)
        lo = np.percentile(arr, 15.87)
        hi = np.percentile(arr, 84.13)
        summary[name] = (med, med - lo, hi - med)
        if verbose:
            print(f"{name:10s} = {med:.4f}  -{med-lo:.4f}  +{hi-med:.4f}")

    return samples, labels, summary


def plot_sed(priorfile, sedfile, bestfit=None, samples=None, outfile=None):
    """
    Plot SED model vs observed data with residuals.

    Parameters
    ----------
    priorfile, sedfile : str
    bestfit : dict or None
        If None, runs fit_sed first.
    samples : ndarray or None
        MCMC samples, shape (nsamples, 5). If provided, draws
        posterior realizations on the SED plot.
    outfile : str or None
        Save figure to this path. If None, calls plt.show().
    """
    import matplotlib.pyplot as plt

    if bestfit is None:
        bestfit = fit_sed(priorfile, sedfile, verbose=False)

    priors = parse_priors(priorfile)
    mstar = priors.get('mstar', {}).get('value', 1.0)

    teff = bestfit['teff']
    logg = bestfit['logg']
    feh = bestfit['feh']
    av = bestfit['av']
    dist = bestfit['distance']
    lstar = bestfit['lstar']

    # Compute best-fit model
    sedchi2, blendmag, modelflux, magresiduals = mistmultised(
        teff, logg, feh, av, dist, lstar, 1.0, sedfile
    )

    # Read observed SED for wavelengths and errors
    from exozippy.sed.utils import read_sed_file
    sed_data = read_sed_file(sedfile, 1)
    bands = sed_data['sedbands']
    obs_mag = sed_data['mag']
    obs_err = sed_data['errmag']
    weff = sed_data['weff']  # effective wavelength in Angstrom

    # weff is already in microns from the filter files
    weff_um = weff

    # Convert magnitudes to fluxes (Fλ) for SED plot
    obs_flux = sed_data['flux']
    obs_errflux = sed_data['errflux']
    zp = sed_data['zero_point']
    model_flux = zp * 10**(-0.4 * blendmag)

    # Sort by wavelength
    order = np.argsort(weff_um)
    weff_um = weff_um[order]
    obs_flux = obs_flux[order]
    obs_errflux = obs_errflux[order]
    model_flux = model_flux[order]
    obs_mag = obs_mag[order]
    obs_err = obs_err[order]
    blendmag_sorted = blendmag[order]
    bands = bands[order]

    mag_resid = obs_mag - blendmag_sorted
    resid_sigma = mag_resid / obs_err

    # --- Draw posterior realizations if MCMC samples provided ---
    posterior_fluxes = None
    if samples is not None:
        ndraws = min(100, len(samples))
        idx = np.random.choice(len(samples), ndraws, replace=False)
        posterior_fluxes = np.empty((ndraws, len(weff_um)))
        for k, ii in enumerate(idx):
            s_teff, s_rstar, s_feh, s_av, s_dist = samples[ii]
            s_logg = _derive_logg(mstar, s_rstar)
            s_lstar = _derive_lstar(s_teff, s_rstar)
            try:
                _, s_bmag, _, _ = mistmultised(
                    s_teff, s_logg, s_feh, s_av, s_dist, s_lstar, 1.0, sedfile
                )
                posterior_fluxes[k, :] = (zp * 10**(-0.4 * s_bmag))[order]
            except Exception:
                posterior_fluxes[k, :] = np.nan

    # --- Figure ---
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1],
                                                 'hspace': 0.05})

    # Top panel: SED
    if posterior_fluxes is not None:
        for k in range(len(posterior_fluxes)):
            ax1.plot(weff_um, posterior_fluxes[k], color='lightskyblue',
                     alpha=0.15, lw=0.5, zorder=1)

    ax1.errorbar(weff_um, obs_flux, yerr=obs_errflux, fmt='ko',
                 ms=6, capsize=3, label='Observed', zorder=3)
    ax1.plot(weff_um, model_flux, 'rs', ms=8, mfc='none', mew=1.5,
             label='Best-fit model', zorder=2)

    ax1.set_ylabel(r'$F_\lambda$ (erg s$^{-1}$ cm$^{-2}$ Hz$^{-1}$)')
    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_title(f'SED Fit: Teff={teff:.0f} K, R*={bestfit["rstar"]:.3f} '
                  rf'R$_\odot$, [Fe/H]={feh:.2f}, Av={av:.3f}')

    # Bottom panel: residuals in sigma
    ax2.axhline(0, color='gray', ls='--', lw=0.8)
    ax2.errorbar(weff_um, resid_sigma, yerr=1.0, fmt='ko', ms=6, capsize=3)

    # Label each point with band name
    for i, b in enumerate(bands):
        ax2.annotate(b, (weff_um[i], resid_sigma[i]),
                     textcoords='offset points', xytext=(0, 8),
                     fontsize=6, ha='center', rotation=45)

    ax2.set_xlabel(r'Wavelength ($\mu$m)')
    ax2.set_ylabel(r'Residual ($\sigma$)')
    ax2.set_ylim(-3, 3)

    fig.tight_layout()

    if outfile is not None:
        fig.savefig(outfile, dpi=150, bbox_inches='tight')
        print(f'Saved plot to {outfile}')
    else:
        plt.show()

    return fig


if __name__ == '__main__':
    import sys

    datadir = 'data/exofastv2/examples/hat3_staronly/'
    priorfile = datadir + 'HAT-3.priors'
    sedfile = datadir + 'HAT-3.sed'

    # Phase 1: Optimizer
    bestfit = fit_sed(priorfile, sedfile)

    # Phase 2: MCMC (if emcee is available)
    if '--mcmc' in sys.argv:
        samples, labels, summary = run_mcmc(
            priorfile, sedfile, bestfit=bestfit,
            nwalkers=32, nsteps=5000, nburn=1000
        )
