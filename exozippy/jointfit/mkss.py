"""
mkss.py — Build a Stellar System (SS) structure.

Analogous to EXOFASTv2's mkss.pro: constructs a nested structure
(SS -> Star, Planet, Band, Transit, Telescope) that describes an
arbitrary number of stars, planets, observed bands, RV telescopes,
and observed transits.

This structure is the single source of truth for parameter definitions,
initial values, priors, bounds, and latex labels.
"""

import glob
import math
import os
import numpy as np

from ..utils.mkconstants import mkconstants
from .parameter import Parameter
from .ss import SS, Star, Planet, Band, Transit, Telescope, DopplerTomography


# ── Band name parsing (EXOFASTv2 convention) ─────────────────────────

# Single-letter Sloan aliases → canonical names (same as EXOFASTv2 readtran.pro)
_SLOAN_ALIASES = {
    'u': 'Sloanu', 'g': 'Sloang', 'r': 'Sloanr',
    'i': 'Sloani', 'z': 'Sloanz',
}


def _parse_band_name(filename):
    """
    Extract and normalize the filter/band name from a transit filename.

    EXOFASTv2 convention: nYYYYMMDD.FILTER.TELESCOPE.whateveryouwant
    The band name is the second dot-separated field of the basename.
    Single-letter Sloan aliases (g, r, i, z, u) are expanded to their
    canonical names (Sloang, Sloanr, etc.).
    """
    parts = os.path.basename(filename).split('.')
    name = parts[1] if len(parts) > 1 else 'default'
    return _SLOAN_ALIASES.get(name, name)


def _get_band_info(tranfiles):
    """
    Parse band names from a list of transit filenames.

    Returns
    -------
    unique_bands : list[str]
        Sorted unique band names (alphabetical, same as EXOFASTv2).
    bandndx : list[int]
        Per-transit index into ``unique_bands``.
    """
    names = [_parse_band_name(f) for f in tranfiles]
    unique_bands = sorted(set(names))
    bandndx = [unique_bands.index(n) for n in names]
    return unique_bands, bandndx


# ── Helper: parse a prior file ────────────────────────────────────────

def _parse_priors(parfile):
    """
    Parse an EXOFASTv2-style prior file.

    Returns dict of dicts with keys: value, sigma, lower, upper.

    Handles EXOFASTv2 log-space parameters:
      logmstar → mstar = 10^logmstar
      logk_N   → k_N   = 10^logk_N
      logp_N   → period_N = 10^logp_N
    Also maps planet-indexed params (tc_0→tc, p_0→p, cosi_0→cosi)
    as aliases for backward compatibility.
    """
    if parfile is None:
        return {}
    priors = {}
    with open(parfile) as f:
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

    # Convert log-space parameters to linear
    _log_conversions = {
        'logmstar': 'mstar',
    }
    # logk_N → k_N, logp_N → period_N (with index)
    for key in list(priors.keys()):
        if key.startswith('logk'):
            suffix = key[4:]  # e.g. '_0'
            linear_key = f'k{suffix}'
            if linear_key not in priors:
                entry = dict(priors[key])
                entry['value'] = 10.0 ** entry['value']
                priors[linear_key] = entry
        elif key.startswith('logp'):
            suffix = key[4:]  # e.g. '_0'
            linear_key = f'period{suffix}'
            if linear_key not in priors:
                entry = dict(priors[key])
                entry['value'] = 10.0 ** entry['value']
                priors[linear_key] = entry
        elif key in _log_conversions:
            linear_key = _log_conversions[key]
            if linear_key not in priors:
                entry = dict(priors[key])
                entry['value'] = 10.0 ** entry['value']
                priors[linear_key] = entry

    # Convert e/omega to sesinw/secosw if not already present
    if 'e' in priors and 'sesinw' not in priors:
        import math as _math
        e_val = priors['e']['value']
        omega_val = priors.get('omega', {}).get('value', _math.pi / 2)
        sqrte = _math.sqrt(max(e_val, 0.0))
        priors['sesinw'] = {'value': sqrte * _math.sin(omega_val), 'sigma': 0.0,
                            'lower': np.nan, 'upper': np.nan}
        priors['secosw'] = {'value': sqrte * _math.cos(omega_val), 'sigma': 0.0,
                            'lower': np.nan, 'upper': np.nan}

    # RM prior name aliases: EXOFASTv2 uses the long form "lambda" suffix
    # but our parameter vector uses the short form without "lambda".
    # Also convert vsini + lambda → svsinicoslam + svsinisinlam if needed.
    _rm_aliases = [('svsinicoslambda', 'svsinicoslam'),
                   ('svsinisinlambda', 'svsinisinlam')]
    for long_key, short_key in _rm_aliases:
        if long_key in priors and short_key not in priors:
            priors[short_key] = priors[long_key]

    # If only (vsini, lambda) are supplied but no svsinicoslam*, derive them.
    if ('vsini' in priors and 'lambda' in priors
            and 'svsinicoslam' not in priors):
        import math as _math
        vsini_v = priors['vsini']['value']
        lam_v   = priors['lambda']['value']
        sv = _math.sqrt(max(vsini_v, 0.0))
        priors['svsinicoslam'] = {'value': sv * _math.cos(lam_v),
                                  'sigma': 0.0, 'lower': np.nan, 'upper': np.nan}
        priors['svsinisinlam'] = {'value': sv * _math.sin(lam_v),
                                  'sigma': 0.0, 'lower': np.nan, 'upper': np.nan}

    # Map planet-indexed params to unsuffixed aliases (for single-planet compat)
    # e.g. tc_0 → tc, p_0 → p, cosi_0 → cosi (and reverse: tc → tc_0)
    _planet_aliases = ['tc', 'p', 'cosi', 'period']
    for base in _planet_aliases:
        key0 = f'{base}_0'
        if key0 in priors and base not in priors:
            priors[base] = priors[key0]
        if base in priors and key0 not in priors:
            priors[key0] = priors[base]

    return priors


def _prior_val(priors, key, default):
    """Get prior value for a key, or default."""
    return priors.get(key, {}).get('value', default)


# ── Helper: make a Parameter with prior overrides ─────────────────────

def _mkpar(label, priors, *, initval, lower=-np.inf, upper=np.inf,
           latex='', description='', unit='', scale=0.0,
           fit=False, derive=True):
    """Create a Parameter, applying prior overrides if present."""
    p = priors.get(label, {})
    val = p.get('value', initval)
    sigma = p.get('sigma', None)
    plower = p.get('lower', np.nan)
    pupper = p.get('upper', np.nan)

    if sigma is not None and sigma == 0 and not np.isnan(val):
        fit = False

    # Tighten bounds if user specifies them
    if np.isfinite(plower):
        lower = max(lower, plower)
    if np.isfinite(pupper):
        upper = min(upper, pupper)

    par = Parameter.__new__(Parameter)
    par.value = val
    par.unit = unit
    par.label = label
    par.latex = latex
    par.latex_unit = unit
    par.latex_value = None
    par.description = description
    par.latex_prefix = 'ez'
    par.table_note = None
    par.posterior = None
    par.prior = val if sigma else None
    par.gaussian_width = sigma if sigma and sigma > 0 else np.inf
    par.lowerbound = lower
    par.upperbound = upper
    par.link = None
    par.amoeba_scale = scale
    par.userchanged = label in priors
    par.fit = fit
    par.derive = derive
    par.medvalue = None
    par.upper = None
    par.lower = None
    par.best = None
    par.get_latex_var()
    return par


# ── Build Star ────────────────────────────────────────────────────────

def _make_star(idx, priors, constants):
    """Build a Star with IDL-matching parameter definitions."""
    suffix = f'_{idx}' if idx > 0 else ''

    mstar_init = _prior_val(priors, 'mstar', 1.0)
    rstar_init = _prior_val(priors, 'rstar', 1.0)
    teff_init = _prior_val(priors, 'teff', 5778.0)
    feh_init = _prior_val(priors, 'feh', 0.0)
    age_init = _prior_val(priors, 'age', 4.603)

    if 'parallax' in priors:
        dist_init = 1000.0 / priors['parallax']['value']
    else:
        dist_init = _prior_val(priors, 'distance', 10.0)

    logg_init = math.log10(mstar_init / rstar_init**2 * constants['GravitySun'])
    lstar_init = (4.0 * math.pi * rstar_init**2 * teff_init**4
                  * constants['sigmab'] / constants['LSun'] * constants['RSun']**2)
    rhostar_init = mstar_init / rstar_init**3 * constants['RhoSun']

    return Star(
        mstar=_mkpar(f'mstar{suffix}', priors, initval=mstar_init,
                      lower=0.1, upper=250, scale=0.5,
                      latex=r'M_*', description='Mass', unit=r'\msun', fit=True),
        rstar=_mkpar(f'rstar{suffix}', priors, initval=rstar_init,
                      lower=0.1, upper=2000, scale=0.5,
                      latex=r'R_*', description='Radius', unit=r'\rsun', fit=True),
        teff=_mkpar(f'teff{suffix}', priors, initval=teff_init,
                     lower=1.0, upper=50000, scale=500,
                     latex=r'T_{\rm eff}', description='Effective Temperature', unit='K', fit=True),
        feh=_mkpar(f'feh{suffix}', priors, initval=feh_init,
                    lower=-5.0, upper=5.0, scale=0.5,
                    latex=r'[{\rm Fe/H}]', description='Metallicity', unit='dex', fit=True),
        logg=_mkpar(f'logg{suffix}', priors, initval=logg_init,
                     scale=0.3,
                     latex=r'\log{g}', description='Surface gravity', unit='cgs'),
        lstar=_mkpar(f'lstar{suffix}', priors, initval=lstar_init,
                      latex=r'L_*', description='Luminosity', unit=r'\lsun'),
        rhostar=_mkpar(f'rhostar{suffix}', priors, initval=rhostar_init,
                        latex=r'\rho_*', description='Density', unit='cgs'),
        age=_mkpar(f'age{suffix}', priors, initval=age_init,
                    lower=0.0, upper=15.0, scale=3.0,
                    latex='Age', description='Age', unit='Gyr'),
        eep=_mkpar(f'eep{suffix}', priors, initval=354.17,
                    scale=50,
                    latex='EEP', description='Equal Evolutionary Phase'),
        av=_mkpar('av', priors, initval=0.01,
                   lower=0.0, upper=10.0, scale=0.3,
                   latex=r'A_V', description='V-band extinction', unit='mag'),
        distance=_mkpar(f'distance{suffix}', priors, initval=dist_init,
                         lower=1.0, upper=1e6, scale=100,
                         latex='d', description='Distance', unit='pc'),
        parallax=_mkpar(f'parallax{suffix}', priors, initval=1000.0 / dist_init,
                         scale=100,
                         latex=r'\varpi', description='Parallax', unit='mas'),
        slope=_mkpar('slope', priors, initval=0.0,
                      scale=1.0,
                      latex=r'\dot{\gamma}', description='RV slope',
                      unit='m/s/day'),
        quad=_mkpar('quad', priors, initval=0.0,
                     scale=1.0,
                     latex=r'\ddot{\gamma}', description='RV quadratic term',
                     unit='m/s/day^2'),
        vgamma=_mkpar('vgamma', priors, initval=_prior_val(priors, 'vgamma', 1000.0),
                        lower=0.0, upper=1e5, scale=1000.0,
                        latex=r'v_{\gamma}', description='Lorentzian line width',
                        unit='m/s'),
        vzeta=_mkpar('vzeta', priors, initval=_prior_val(priors, 'vzeta', 4000.0),
                       lower=0.0, upper=1e5, scale=1000.0,
                       latex=r'v_{\zeta}', description='Macroturbulent velocity',
                       unit='m/s'),
        vxi=_mkpar('vxi', priors, initval=_prior_val(priors, 'vxi', 1000.0),
                     lower=0.0, upper=1e5, scale=1000.0,
                     latex=r'v_{\xi}', description='Microturbulent velocity',
                     unit='m/s'),
        valpha=_mkpar('valpha', priors, initval=_prior_val(priors, 'valpha', 0.0),
                        lower=0.0, upper=1e5, scale=1000.0,
                        latex=r'v_{\alpha}', description='Extra broadening',
                        unit='m/s'),
        vline=_mkpar('vline', priors, initval=_prior_val(priors, 'vline', 5000.0),
                      lower=0.0, upper=1e6, scale=1000.0,
                      latex=r'v_{\rm line}',
                      description='Intrinsic spectral line broadening sigma',
                      unit='m/s'),
        label=chr(65 + idx),  # 'A', 'B', 'C', ...
    )


# ── Build Planet ──────────────────────────────────────────────────────

def _make_planet(idx, priors, constants, circular=True, fittran=True, fitrv=True,
                 usevcve=False):
    """Build a Planet with IDL-matching parameter definitions."""
    suffix = f'_{idx}'

    period_init = _prior_val(priors, f'period{suffix}',
                             _prior_val(priors, 'period_0', 3.0))
    tc_init = _prior_val(priors, 'tc', 0.0)
    p_init = _prior_val(priors, 'p', 0.1)
    cosi_init = _prior_val(priors, 'cosi', 0.05)
    K_init = _prior_val(priors, f'k{suffix}',
                         _prior_val(priors, 'k_0', 50.0))

    e_val = 0.0 if circular else _prior_val(priors, 'e', 0.0)
    omega_val = math.pi / 2 if circular else _prior_val(priors, 'omega', math.pi / 2)
    sqrte = math.sqrt(e_val)
    sesinw_val = sqrte * math.sin(omega_val)
    secosw_val = sqrte * math.cos(omega_val)

    # Vc/Ve parameterization: compute initial vcve from e/omega
    if e_val > 0 and e_val < 1:
        vcve_val = math.sqrt(1.0 - e_val**2) / (1.0 + e_val * math.sin(omega_val))
    else:
        vcve_val = 1.0  # circular
    lsinw_val = _prior_val(priors, 'lsinw', 0.5 * math.sin(omega_val))
    lcosw_val = _prior_val(priors, 'lcosw', 0.5 * math.cos(omega_val))
    sign_val = _prior_val(priors, 'sign', 0.0)

    # Eccentricity fit flags:
    # - circular → nothing fitted
    # - usevcve → fit vcve/lsinw/lcosw/sign (transit-only)
    # - else → fit sesinw/secosw (RV available)
    fit_sesinw = (not circular) and (not usevcve)
    fit_vcve = (not circular) and usevcve

    return Planet(
        period=_mkpar(f'period{suffix}', priors, initval=period_init,
                       lower=1e-6, upper=1e6, scale=0.01,
                       latex='P', description='Period', unit='days', fit=True),
        tc=_mkpar('tc', priors, initval=tc_init,
                   scale=0.1,
                   latex=r'T_C', description='Time of conjunction', unit=r'\bjdtdb', fit=True),
        p=_mkpar('p', priors, initval=p_init,
                  lower=-0.5, upper=1.0, scale=0.1,
                  latex=r'R_P/R_*', description='Radius of planet in stellar radii', fit=True),
        cosi=_mkpar('cosi', priors, initval=cosi_init,
                     lower=0.0, upper=1.0, scale=0.1,
                     latex=r'\cos{i}', description='Cos of inclination', fit=True),
        K=_mkpar(f'k{suffix}', priors, initval=K_init,
                  lower=0.0, upper=1e4, scale=5000,
                  latex='K', description='RV semi-amplitude', unit='m/s', fit=True),
        e=_mkpar('e', priors, initval=e_val,
                  lower=0.0, upper=1.0,
                  latex='e', description='Eccentricity'),
        omega=_mkpar('omega', priors, initval=omega_val,
                      latex=r'\omega_*', description='Argument of periastron', unit='Radians'),
        sesinw=_mkpar('sesinw', priors, initval=sesinw_val,
                       lower=-1.0, upper=1.0, scale=0.1,
                       latex=r'\sqrt{e}\sin{\omega_*}', description='',
                       fit=fit_sesinw),
        secosw=_mkpar('secosw', priors, initval=secosw_val,
                       lower=-1.0, upper=1.0, scale=0.1,
                       latex=r'\sqrt{e}\cos{\omega_*}', description='',
                       fit=fit_sesinw),
        vcve=_mkpar('vcve', priors, initval=vcve_val,
                     lower=0.0, upper=1.0, scale=0.05,
                     latex=r'V_c/V_e', description='Circ/ecc velocity ratio',
                     fit=fit_vcve),
        lsinw=_mkpar('lsinw', priors, initval=lsinw_val,
                      lower=-1.0, upper=1.0, scale=0.1,
                      latex=r'L\sin{\omega_*}', description='',
                      fit=fit_vcve),
        lcosw=_mkpar('lcosw', priors, initval=lcosw_val,
                      lower=-1.0, upper=1.0, scale=0.1,
                      latex=r'L\cos{\omega_*}', description='',
                      fit=fit_vcve),
        sign=_mkpar('sign', priors, initval=sign_val,
                     lower=-1.0, upper=2.0, scale=0.1,
                     latex='sign', description='vcve quadratic root selector',
                     fit=fit_vcve),
        # Derived (initial placeholders, recomputed by ss.compute_derived())
        ar=_mkpar(f'ar{suffix}', priors, initval=10.0,
                   latex=r'a/R_*', description='Semi-major axis in stellar radii'),
        b=_mkpar(f'b{suffix}', priors, initval=0.5,
                  latex='b', description='Transit impact parameter'),
        inc_rad=_mkpar(f'inc{suffix}', priors, initval=math.acos(cosi_init),
                        latex='i', description='Inclination', unit='Radians'),
        ideg=_mkpar(f'ideg{suffix}', priors, initval=math.degrees(math.acos(cosi_init)),
                     latex='i', description='Inclination', unit='Degrees'),
        delta=_mkpar(f'delta{suffix}', priors, initval=p_init**2,
                      latex=r'\delta', description='Transit depth', unit='frac'),
        mp=_mkpar(f'mp{suffix}', priors, initval=0.0,
                   latex=r'M_P', description='Mass', unit=r'\mj'),
        rp=_mkpar(f'rp{suffix}', priors, initval=0.0,
                   latex=r'R_P', description='Radius', unit=r'\rj'),
        a=_mkpar(f'a{suffix}', priors, initval=0.0,
                  latex='a', description='Semi-major axis', unit='AU'),
        teq=_mkpar(f'teq{suffix}', priors, initval=0.0,
                    latex=r'T_{\rm eq}', description='Equilibrium temperature', unit='K'),
        beam=_mkpar('beam', priors, initval=_prior_val(priors, 'beam', 0.0),
                     lower=-500, upper=500, scale=5.0,
                     latex=r'A_B', description='Doppler beaming amplitude', unit='ppm'),
        ellipsoidal=_mkpar('ellipsoidal', priors, initval=_prior_val(priors, 'ellipsoidal', 0.0),
                            lower=0.0, upper=500, scale=5.0,
                            latex=r'A_{\rm ellip}', description='Ellipsoidal variation amplitude', unit='ppm'),
        svsinicoslam=_mkpar('svsinicoslam', priors,
                             initval=_prior_val(priors, 'svsinicoslam', 0.0),
                             lower=-1e4, upper=1e4, scale=100.0,
                             latex=r'\sqrt{v\sin{i}}\cos{\lambda}',
                             description='sqrt(vsini)*cos(lambda)', unit='m^{0.5}/s^{0.5}'),
        svsinisinlam=_mkpar('svsinisinlam', priors,
                              initval=_prior_val(priors, 'svsinisinlam', 0.0),
                              lower=-1e4, upper=1e4, scale=100.0,
                              latex=r'\sqrt{v\sin{i}}\sin{\lambda}',
                              description='sqrt(vsini)*sin(lambda)', unit='m^{0.5}/s^{0.5}'),
        fittran=fittran,
        fitrv=fitrv,
        circular=circular,
        label='b' if idx == 0 else chr(99 + idx),  # b, c, d, ...
    )


# ── Build Band ────────────────────────────────────────────────────────

def _make_band(name, idx, priors):
    """Build a Band (per-wavelength limb darkening)."""
    suffix = f'_{idx}'
    return Band(
        u1=_mkpar(f'u1{suffix}', priors, initval=0.4,
                   lower=0.0, upper=2.0,
                   latex=r'u_1', description='Linear limb-darkening coeff', fit=True),
        u2=_mkpar(f'u2{suffix}', priors, initval=0.2,
                   lower=-1.0, upper=1.0,
                   latex=r'u_2', description='Quadratic limb-darkening coeff', fit=True),
        thermal=_mkpar(f'thermal{suffix}', priors,
                        initval=_prior_val(priors, f'thermal{suffix}', 0.0),
                        lower=0.0, upper=5000, scale=50.0,
                        latex=r'A_{\rm therm}', description='Thermal emission', unit='ppm'),
        reflect=_mkpar(f'reflect{suffix}', priors,
                        initval=_prior_val(priors, f'reflect{suffix}', 0.0),
                        lower=0.0, upper=5000, scale=20.0,
                        latex=r'A_{\rm refl}', description='Reflected light', unit='ppm'),
        name=name,
        label=name,
    )


# ── Build DopplerTomography ──────────────────────────────────────────

def _make_dopptom(filename, priors, idx):
    """Build a DopplerTomography object from a FITS file.

    Reads the 2-D CCF, BJD, velocity arrays and computes the per-pixel
    RMS noise.  The ``errscale`` parameter defaults to 1.0 (fixed).
    """
    from .physics.exozippy_dopptom import read_dt_fits
    data = read_dt_fits(filename)
    errscale_init = _prior_val(priors, f'errscale_{idx}',
                               _prior_val(priors, 'errscale', 1.0))
    return DopplerTomography(
        ccf2d      = data['ccf2d'],
        bjd        = data['bjd'],
        vel        = data['vel'],
        rms        = data['rms'],
        Rspec      = data['Rspec'],
        errscale   = _mkpar(f'errscale_{idx}', priors,
                            initval=errscale_init,
                            lower=0.0, upper=1e3, scale=0.1,
                            latex=r'\sigma_{\rm err}',
                            description='DT error scale factor'),
        label      = data['label'],
        filename   = filename,
    )


# ── Detrending helpers ───────────────────────────────────────────────

def _parse_detrend_header(filename):
    """Parse the first line of a data file for detrend column classification.

    Returns (add_indices, mult_indices) — lists of column indices (0-based
    from col 3 onward) that are additive vs multiplicative.

    Convention (EXOFASTv2): header line starts with '#'; column names
    prefixed with 'M' are multiplicative, others are additive.
    If no header, all extra columns are additive.
    """
    add_indices = []
    mult_indices = []
    with open(filename, 'r') as f:
        first_line = f.readline().strip()
    if not first_line.startswith('#'):
        return None, None  # no header — caller counts columns
    names = first_line.lstrip('#').split()
    # First 3 columns are BJD, FLUX/RV, ERR — skip them
    for i, name in enumerate(names[3:]):
        if name.upper().startswith('M'):
            mult_indices.append(i)
        else:
            add_indices.append(i)
    return add_indices, mult_indices


def _normalize_covariates(array):
    """Mean-subtract and scale to [-1, 1] (EXOFASTv2 convention).

    Parameters
    ----------
    array : np.ndarray, shape (ncov, npts)

    Returns
    -------
    np.ndarray, same shape, normalized.
    """
    if array.size == 0:
        return array
    arr = array.copy()
    for i in range(arr.shape[0]):
        arr[i] -= np.mean(arr[i])
        mx = np.max(np.abs(arr[i]))
        if mx > 0:
            arr[i] /= mx
    return arr


def _read_data_with_detrend(filename):
    """Read a data file and extract detrend covariates.

    Returns
    -------
    bjd, col1, err : 1D arrays (columns 0, 1, 2)
    detrendadd : (nadd, npts) normalized additive covariates or None
    detrendmult : (nmult, npts) normalized multiplicative covariates or None
    """
    # Count header lines to skip for np.loadtxt
    nheader = 0
    with open(filename, 'r') as f:
        for line in f:
            if line.strip().startswith('#'):
                nheader += 1
            else:
                break
    data = np.loadtxt(filename, comments='#')
    if data.ndim == 1:
        data = data.reshape(1, -1)
    bjd = data[:, 0]
    col1 = data[:, 1]
    err = data[:, 2]

    ncol = data.shape[1]
    if ncol <= 3:
        return bjd, col1, err, None, None

    # Extra columns present → parse header for add/mult classification
    extras = data[:, 3:]  # (npts, nextra)
    add_indices, mult_indices = _parse_detrend_header(filename)

    if add_indices is None:
        # No header → all extra columns are additive
        detrendadd = _normalize_covariates(extras.T)  # (nextra, npts)
        return bjd, col1, err, detrendadd, None

    # Header present — split by classification
    detrendadd = None
    detrendmult = None
    if add_indices:
        detrendadd = _normalize_covariates(extras[:, add_indices].T)
    if mult_indices:
        detrendmult = _normalize_covariates(extras[:, mult_indices].T)
    return bjd, col1, err, detrendadd, detrendmult


# ── Build Transit ────────────────────────────────────────────────────

def _make_transit(tranfile, idx, priors, tc=None, period=None, fitttv=False, bandndx=0):
    """Build a Transit from a data file."""
    bjd, flux, err, detrendadd, detrendmult = _read_data_with_detrend(tranfile)

    basename = os.path.basename(tranfile)

    # Compute integer epoch from linear ephemeris
    epoch = 0
    if tc is not None and period is not None and period > 0:
        epoch = int(round((float(np.median(bjd)) - tc) / period))

    suffix = f'_{idx}'

    # Create detrend parameters
    detrendaddpars = []
    if detrendadd is not None:
        for k in range(detrendadd.shape[0]):
            par = _mkpar(f'C{k}{suffix}', priors, initval=0.0, scale=0.1,
                         latex=f'C_{{{k}}}', description=f'Additive detrend {k}',
                         fit=True)
            detrendaddpars.append(par)

    detrendmultpars = []
    if detrendmult is not None:
        for k in range(detrendmult.shape[0]):
            par = _mkpar(f'M{k}{suffix}', priors, initval=0.0, scale=0.1,
                         latex=f'M_{{{k}}}', description=f'Multiplicative detrend {k}',
                         fit=True)
            detrendmultpars.append(par)

    return Transit(
        f0=_mkpar(f'f0{suffix}', priors, initval=float(np.median(flux)),
                   lower=0.0, upper=2.0, scale=0.001,
                   latex=r'F_0', description='Baseline flux', fit=True),
        variance=_mkpar(f'variance{suffix}', priors, initval=0.0,
                         lower=0.0,
                         latex=r'\sigma_j^2', description='Added variance'),
        dilute=_mkpar(f'dilute{suffix}', priors, initval=0.0,
                       lower=-1.0, upper=1.0, scale=0.01,
                       latex=r'A_D', description='Dilution', unit=''),
        ttv=_mkpar(f'ttv{suffix}', priors, initval=0.0,
                    scale=0.02,
                    latex=r'TTV', description='Transit timing variation',
                    unit='days', fit=fitttv),
        bjd=bjd,
        flux=flux,
        err=err,
        detrendadd=detrendadd,
        detrendmult=detrendmult,
        detrendaddpars=detrendaddpars,
        detrendmultpars=detrendmultpars,
        epoch=epoch,
        bandndx=bandndx,
        name=basename,
        label=basename,
    )


# ── Build Telescope ──────────────────────────────────────────────────

def _make_telescope(rvfile, idx, priors):
    """Build a Telescope from an RV data file."""
    bjd, vel, err, detrendadd, detrendmult = _read_data_with_detrend(rvfile)

    basename = os.path.basename(rvfile)
    label = (os.path.splitext(basename)[0].split('.')[1]
             if '.' in basename else f'RV{idx}')

    suffix = f'_{idx}'
    gamma_init = _prior_val(priors, f'gamma{suffix}',
                             _prior_val(priors, 'gamma_0', 0.0))
    jittervar_init = _prior_val(priors, 'jittervar', 0.0)

    # Create RV detrend parameters
    detrendaddpars = []
    if detrendadd is not None:
        for k in range(detrendadd.shape[0]):
            par = _mkpar(f'RVC{k}{suffix}', priors, initval=0.0, scale=1.0,
                         latex=f'RVC_{{{k}}}', description=f'RV additive detrend {k}',
                         unit='m/s', fit=True)
            detrendaddpars.append(par)

    detrendmultpars = []
    if detrendmult is not None:
        for k in range(detrendmult.shape[0]):
            par = _mkpar(f'RVM{k}{suffix}', priors, initval=0.0, scale=1.0,
                         latex=f'RVM_{{{k}}}', description=f'RV multiplicative detrend {k}',
                         unit='', fit=True)
            detrendmultpars.append(par)

    return Telescope(
        gamma=_mkpar(f'gamma{suffix}', priors, initval=gamma_init,
                      lower=-1e5, upper=1e5, scale=5000,
                      latex=r'\gamma_{\rm rel}', description='Relative RV Offset',
                      unit='m/s', fit=True),
        jittervar=_mkpar(f'jittervar{suffix}', priors, initval=jittervar_init,
                          lower=0.0,
                          latex=r'\sigma_J^2', description='RV Jitter Variance',
                          unit='m^2/s^2'),
        jitter=_mkpar(f'jitter{suffix}', priors,
                       initval=math.sqrt(max(jittervar_init, 0.0)),
                       latex=r'\sigma_J', description='RV Jitter', unit='m/s'),
        bjd=bjd,
        vel=vel,
        err=err,
        detrendadd=detrendadd,
        detrendmult=detrendmult,
        detrendaddpars=detrendaddpars,
        detrendmultpars=detrendmultpars,
        name=basename,
        label=label,
    )


# ══════════════════════════════════════════════════════════════════════
# Main entry point
# ══════════════════════════════════════════════════════════════════════

def mkss(
    parfile=None,
    tranpath=None,
    rvpath=None,
    sedfile=None,
    nstars=1,
    nplanets=1,
    fittran=True,
    fitrv=True,
    circular=True,
    use_mist=False,
    fitjittervar=False,
    fitslope=False,
    fitquad=False,
    fitvariance=False,
    fitdilute=False,
    fitthermal=False,
    fitreflect=False,
    fitbeam=False,
    fitellip=False,
    usevcve=False,
    fitttv=False,
    rossiter=False,
    rmbands=None,
    dtpath=None,
    fitdt=False,
    fiterrscale=False,
):
    """
    Construct a Stellar System (SS) structure.

    Analogous to EXOFASTv2's mkss.pro.

    Parameters
    ----------
    parfile : str
        Path to the prior file.
    tranpath : str
        Glob pattern for transit light curve files.
    rvpath : str
        Glob pattern for RV data files.
    sedfile : str
        Path to the SED definition file.
    nstars, nplanets : int
        Number of stars / planets.
    fittran, fitrv : bool or list[bool]
        Whether to fit transit / RV for each planet.
    circular : bool or list[bool]
        Whether each planet's orbit is circular.
    use_mist : bool
        Use MIST evolutionary models.
    usevcve : bool
        Use Vc/Ve eccentricity parameterization (transit-only).

    Returns
    -------
    SS
        The stellar system structure.
    """
    const = mkconstants()
    priors = _parse_priors(parfile)

    # Data file discovery
    tranfiles = sorted(glob.glob(tranpath)) if tranpath else []
    rvfiles = sorted(glob.glob(rvpath)) if rvpath else []

    # Broadcast scalar flags to per-planet arrays
    if np.isscalar(fittran):
        fittran = [bool(fittran)] * nplanets
    if np.isscalar(fitrv):
        fitrv = [bool(fitrv)] * nplanets
    if np.isscalar(circular):
        circular_list = [bool(circular)] * nplanets
    else:
        circular_list = list(circular)

    # ── Build sub-structures ──

    stars = [_make_star(i, priors, const) for i in range(nstars)]
    planets = [_make_planet(i, priors, const,
                             circular=circular_list[i],
                             fittran=fittran[i],
                             fitrv=fitrv[i],
                             usevcve=usevcve)
               for i in range(nplanets)]

    # Bands: one per unique filter name (EXOFASTv2 convention)
    # Band names come from the second dot-separated field of each filename.
    if tranfiles:
        unique_bands, tran_bandndx = _get_band_info(tranfiles)
    else:
        unique_bands, tran_bandndx = ['default'], []
    bands = [_make_band(name, i, priors) for i, name in enumerate(unique_bands)]

    # Get tc/period for epoch computation
    tc_init = _prior_val(priors, 'tc', 0.0)
    period_init = _prior_val(priors, 'period_0', _prior_val(priors, 'period', 3.0))
    # Need at least 3 transits for TTV (2-param linear fit needs ≥3 points)
    _fitttv = fitttv and len(tranfiles) >= 3
    transits = [_make_transit(f, i, priors, tc=tc_init, period=period_init,
                              fitttv=_fitttv,
                              bandndx=tran_bandndx[i] if tran_bandndx else 0)
                for i, f in enumerate(tranfiles)]
    telescopes = [_make_telescope(f, i, priors) for i, f in enumerate(rvfiles)]

    # ── Build param_names (matches exozippy_chi2.py ordering) ──

    has_sed = sedfile is not None and str(sedfile) != ''
    param_names = []
    # Stellar parameters (per star)
    stellar_fit = ['teff', 'rstar', 'feh', 'av', 'distance'] if has_sed else ['teff', 'rstar', 'feh']
    stellar_mist = ['logmstar', 'age']
    for i in range(nstars):
        suffix = f'_{i}' if nstars > 1 else ''
        if use_mist:
            param_names.extend([f'{p}{suffix}' for p in stellar_mist])
        param_names.extend([f'{p}{suffix}' for p in stellar_fit])
    # Shared orbital parameters
    param_names.extend(['tc', 'logP', 'p', 'cosi', 'K'])
    # Eccentricity (when non-circular)
    if not all(circular_list):
        if usevcve:
            param_names.extend(['vcve', 'lsinw', 'lcosw', 'sign'])
        else:
            param_names.extend(['sesinw', 'secosw'])
    # Per-band limb darkening
    for j in range(len(bands)):
        param_names.extend([f'u1_{j}', f'u2_{j}'])
    # Per-band phase curve params
    if fitthermal:
        for j in range(len(bands)):
            param_names.append(f'thermal_{j}')
    if fitreflect:
        for j in range(len(bands)):
            param_names.append(f'reflect_{j}')
    # Per-transit normalization
    for j in range(len(transits)):
        param_names.append(f'f0_{j}')
    # Per-transit variance (jitter)
    if fitvariance:
        for j in range(len(transits)):
            param_names.append(f'variance_{j}')
    # Per-transit dilution
    if fitdilute:
        for j in range(len(transits)):
            param_names.append(f'dilute_{j}')
    # Per-transit TTV
    if _fitttv:
        for j in range(len(transits)):
            param_names.append(f'ttv_{j}')
    # Per-transit detrending coefficients (auto-detected from extra columns)
    for j, tr in enumerate(transits):
        for k in range(len(tr.detrendaddpars)):
            param_names.append(f'C{k}_{j}')
        for k in range(len(tr.detrendmultpars)):
            param_names.append(f'M{k}_{j}')
    # Per-telescope gamma
    for j in range(len(telescopes)):
        param_names.append(f'gamma_{j}')
    # Per-telescope jittervar
    if fitjittervar:
        for j in range(len(telescopes)):
            param_names.append(f'jittervar_{j}')
    # Per-telescope detrending coefficients (auto-detected from extra columns)
    for j, tel in enumerate(telescopes):
        for k in range(len(tel.detrendaddpars)):
            param_names.append(f'RVC{k}_{j}')
        for k in range(len(tel.detrendmultpars)):
            param_names.append(f'RVM{k}_{j}')
    # Global RV trend (fitquad implies fitslope)
    _fitslope = fitslope or fitquad
    if _fitslope:
        param_names.append('slope')
    if fitquad:
        param_names.append('quad')
    # Per-planet phase curve params
    if fitbeam:
        param_names.append('beam')
    if fitellip:
        param_names.append('ellipsoidal')
    # Rossiter-McLaughlin params (per-planet + per-star)
    if rossiter:
        param_names.extend(['svsinicoslam', 'svsinisinlam'])
        param_names.extend(['vgamma', 'vzeta', 'vxi', 'valpha'])
        for pl in planets:
            pl.rossiter = True
            pl.svsinicoslam.fit = True
            pl.svsinisinlam.fit = True
        for s in stars:
            s.vgamma.fit = True
            s.vzeta.fit = True
            s.vxi.fit = True
            s.valpha.fit = True
    # Assign rmbands to telescopes
    if rmbands is not None:
        for j, tel in enumerate(telescopes):
            if j < len(rmbands):
                tel.rmband = rmbands[j]
                if rmbands[j] != 'notrm':
                    # Find band index by name
                    for bi, b in enumerate(bands):
                        if b.name == rmbands[j]:
                            tel.rmbandndx = bi
                            break
                    else:
                        # If not found, use band 0 (default)
                        tel.rmbandndx = 0

    # ── Doppler Tomography ──
    dtfiles = sorted(glob.glob(dtpath)) if dtpath else []
    dopptoms = [_make_dopptom(f, priors, j) for j, f in enumerate(dtfiles)]

    if fitdt and dopptoms:
        # svsinicoslam/svsinisinlam (shared with RM if rossiter=True)
        if not rossiter:
            param_names.extend(['svsinicoslam', 'svsinisinlam'])
            for pl in planets:
                pl.svsinicoslam.fit = True
                pl.svsinisinlam.fit = True
        # vline — intrinsic line broadening
        param_names.append('vline')
        for s in stars:
            s.vline.fit = True
        # per-DT errscale (optional)
        if fiterrscale:
            for j in range(len(dopptoms)):
                param_names.append(f'errscale_{j}')
                dopptoms[j].errscale.fit = True

    # ── Assemble SS ──

    # Compute rvepoch and seed slope/quad from RV data
    rvepoch = 0.0
    if telescopes and (_fitslope or fitquad):
        alltime = np.concatenate([t.bjd for t in telescopes])
        allrv = np.concatenate([t.vel for t in telescopes])
        rvepoch = (float(np.min(alltime)) + float(np.max(alltime))) / 2.0
        if fitquad:
            coeffs = np.polyfit(alltime - rvepoch, allrv, 2)
            stars[0].quad.value = coeffs[0]
            stars[0].slope.value = coeffs[1]
            stars[0].quad.fit = True
            stars[0].slope.fit = True
            # Absorb constant into gammas
            for tel in telescopes:
                tel.gamma.value += coeffs[2] / len(telescopes)
        elif _fitslope:
            coeffs = np.polyfit(alltime - rvepoch, allrv, 1)
            stars[0].slope.value = coeffs[0]
            stars[0].slope.fit = True
            for tel in telescopes:
                tel.gamma.value += coeffs[1] / len(telescopes)

    ss = SS(
        star=stars,
        planet=planets,
        band=bands,
        transit=transits,
        telescope=telescopes,
        dopptom=dopptoms,
        constants=const,
        nstars=nstars,
        nplanets=nplanets,
        use_mist=use_mist,
        param_names=param_names,
        rvepoch=rvepoch,
        sedfile=str(sedfile) if sedfile else '',
        tranpath=str(tranpath) if tranpath else '',
        rvpath=str(rvpath) if rvpath else '',
    )

    # Compute derived quantities from initial values
    ss.compute_derived()

    return ss
