"""
Nested dataclass system for EXOZIPPy (analogous to IDL mkss.pro structures).

Provides:
    SS (Stellar System) — top-level container
    Star, Planet, Band, Transit, Telescope — sub-structures

IDL-style access:
    ss.star[0].teff.value
    ss.planet[0].period.value

Backward-compatible dict access (for existing plotting functions):
    ss['teff']  →  ss.star[0].teff.value
    ss['period'] → ss.planet[0].period.value

Multi-star indexed access:
    ss['teff_0']  →  ss.star[0].teff.value
    ss['teff_1']  →  ss.star[1].teff.value
"""

from dataclasses import dataclass, field
from typing import List, Optional
import re
import numpy as np

from .parameter import Parameter


# ── Star ──────────────────────────────────────────────────────────────

@dataclass
class Star:
    """Stellar parameters (mirrors IDL ss.star[i])."""
    mstar: Parameter = None
    rstar: Parameter = None
    teff: Parameter = None
    feh: Parameter = None
    logg: Parameter = None          # derived
    lstar: Parameter = None         # derived
    rhostar: Parameter = None       # derived
    age: Parameter = None
    eep: Parameter = None
    av: Parameter = None
    distance: Parameter = None
    parallax: Parameter = None      # derived
    label: str = ''
    rootlabel: str = 'Stellar Parameters:'


# ── Planet ────────────────────────────────────────────────────────────

@dataclass
class Planet:
    """Planetary parameters (mirrors IDL ss.planet[i])."""
    period: Parameter = None
    tc: Parameter = None
    p: Parameter = None             # Rp/Rs
    cosi: Parameter = None
    K: Parameter = None             # RV semi-amplitude
    e: Parameter = None
    omega: Parameter = None
    sesinw: Parameter = None        # sqrt(e)*sin(omega)
    secosw: Parameter = None        # sqrt(e)*cos(omega)
    # Vc/Ve parameterization (transit-only eccentricity)
    vcve: Parameter = None          # Vc/Ve = sqrt(1-e^2)/(1+e*sin(omega))
    lsinw: Parameter = None         # L*sin(omega)
    lcosw: Parameter = None         # L*cos(omega)
    sign: Parameter = None          # root selector for vcve2e quadratic
    # Derived
    ar: Parameter = None            # a/Rs
    b: Parameter = None             # impact parameter
    inc_rad: Parameter = None       # inclination [rad]  (named inc_rad for backward compat)
    ideg: Parameter = None          # inclination [deg]
    delta: Parameter = None         # transit depth = p^2
    mp: Parameter = None            # planet mass [Mjup]
    rp: Parameter = None            # planet radius [Rjup]
    a: Parameter = None             # semi-major axis [AU]
    teq: Parameter = None           # equilibrium temp [K]
    # Phase curve (per planet)
    beam: Parameter = None              # Doppler beaming amplitude [ppm]
    ellipsoidal: Parameter = None       # Ellipsoidal variation amplitude [ppm]
    # Flags
    fittran: bool = True
    fitrv: bool = True
    circular: bool = True
    starndx: int = 0
    label: str = ''
    rootlabel: str = 'Planetary Parameters:'


# ── Band (per wavelength) ────────────────────────────────────────────

@dataclass
class Band:
    """Per-wavelength limb-darkening parameters (mirrors IDL ss.band[i])."""
    u1: Parameter = None
    u2: Parameter = None
    thermal: Parameter = None           # Thermal emission [ppm]
    reflect: Parameter = None           # Reflected light [ppm]
    name: str = ''
    label: str = ''
    rootlabel: str = 'Wavelength Parameters:'


# ── Transit (per light curve) ────────────────────────────────────────

@dataclass
class Transit:
    """Per-transit parameters (mirrors IDL ss.transit[i])."""
    f0: Parameter = None            # normalization
    variance: Parameter = None      # added variance (jitter)
    ttv: Parameter = None           # transit timing variation (days)
    # Data arrays
    bjd: Optional[np.ndarray] = None
    flux: Optional[np.ndarray] = None
    err: Optional[np.ndarray] = None
    # Flags / indices
    epoch: int = 0                  # integer epoch from linear ephemeris
    bandndx: int = 0
    pndx: int = 0                   # which planet
    name: str = ''
    label: str = ''
    rootlabel: str = 'Transit Parameters:'


# ── Telescope (per RV instrument) ────────────────────────────────────

@dataclass
class Telescope:
    """Per-telescope RV parameters (mirrors IDL ss.telescope[i])."""
    gamma: Parameter = None
    jittervar: Parameter = None
    jitter: Parameter = None        # derived = sqrt(jittervar)
    # Data arrays
    bjd: Optional[np.ndarray] = None
    vel: Optional[np.ndarray] = None
    err: Optional[np.ndarray] = None
    name: str = ''
    label: str = ''
    rootlabel: str = 'Telescope Parameters:'


# ── SS (Stellar System — top level) ──────────────────────────────────

# Parameter names that live on each sub-structure.
# Used by __getitem__ for backward-compatible dict access.
_STAR_PARAMS = frozenset([
    'mstar', 'rstar', 'teff', 'feh', 'logg', 'lstar', 'rhostar',
    'age', 'eep', 'av', 'distance', 'parallax',
])
_PLANET_PARAMS = frozenset([
    'period', 'tc', 'p', 'cosi', 'K', 'e', 'omega',
    'sesinw', 'secosw',
    'vcve', 'lsinw', 'lcosw', 'sign',
    'ar', 'b', 'inc_rad', 'ideg', 'delta',
    'mp', 'rp', 'a', 'teq',
    'beam', 'ellipsoidal',
])
_BAND_PARAMS = frozenset(['u1', 'u2', 'thermal', 'reflect'])
_TRANSIT_PARAMS = frozenset(['f0', 'variance', 'tran_addvar', 'ttv'])
_TELESCOPE_PARAMS = frozenset(['gamma', 'jittervar', 'jitter', 'rv_jittervar'])

# Aliases for backward compatibility with the old bestfit dict
_ALIASES = {
    'rv_jittervar': ('telescope', 'jittervar'),
    'tran_addvar': ('transit', 'variance'),
}

# Regex for indexed parameter names like 'teff_1', 'mstar_0'
_INDEXED_RE = re.compile(r'^(.+?)_(\d+)$')


@dataclass
class SS:
    """
    Stellar System — top-level structure (mirrors IDL ``ss = mkss(...)``).

    Usage::

        # IDL-style
        ss.star[0].teff.value
        ss.planet[0].period.value

        # Backward-compatible dict-style (single star/planet shortcut)
        ss['teff']    # → ss.star[0].teff.value
        ss['period']  # → ss.planet[0].period.value

        # Multi-star indexed access
        ss['teff_0']  # → ss.star[0].teff.value
        ss['teff_1']  # → ss.star[1].teff.value
    """
    star: List[Star] = field(default_factory=list)
    planet: List[Planet] = field(default_factory=list)
    band: List[Band] = field(default_factory=list)
    transit: List[Transit] = field(default_factory=list)
    telescope: List[Telescope] = field(default_factory=list)
    constants: dict = field(default_factory=dict)

    # Configuration
    nstars: int = 1
    nplanets: int = 1
    use_mist: bool = False
    param_names: list = field(default_factory=list)

    # Data paths
    sedfile: str = ''
    tranpath: str = ''
    rvpath: str = ''

    # Fit diagnostics
    chi2: float = np.nan
    ndata: int = 0
    ndof: int = 0
    chi2_red: float = np.nan
    bic: float = np.nan
    aic: float = np.nan
    success: bool = False
    message: str = ''

    # ── Dict-compatible interface ──

    def _resolve(self, key):
        """
        Resolve a key to (container_obj, attr_name) or raise KeyError.

        Supports:
          - Plain keys: 'teff' → star[0].teff
          - Indexed keys: 'teff_1' → star[1].teff
          - Aliases: 'rv_jittervar' → telescope[0].jittervar
        """
        # Handle aliases first
        if key in _ALIASES:
            container_name, real_attr = _ALIASES[key]
            container_list = getattr(self, container_name)
            if container_list:
                return container_list[0], real_attr
            raise KeyError(key)

        # Try indexed pattern: 'teff_1' → base='teff', idx=1
        m = _INDEXED_RE.match(key)
        if m:
            base, idx = m.group(1), int(m.group(2))
            if base in _STAR_PARAMS and idx < len(self.star):
                return self.star[idx], base
            if base in _PLANET_PARAMS and idx < len(self.planet):
                return self.planet[idx], base
            if base in _BAND_PARAMS and idx < len(self.band):
                return self.band[idx], base
            if base in _TRANSIT_PARAMS and idx < len(self.transit):
                return self.transit[idx], base
            if base in _TELESCOPE_PARAMS and idx < len(self.telescope):
                return self.telescope[idx], base
            # Check alias with index (e.g. 'rv_jittervar_0')
            if base in _ALIASES:
                container_name, real_attr = _ALIASES[base]
                container_list = getattr(self, container_name)
                if idx < len(container_list):
                    return container_list[idx], real_attr

        # Plain keys → default to index 0
        if key in _STAR_PARAMS and self.star:
            return self.star[0], key
        if key in _PLANET_PARAMS and self.planet:
            return self.planet[0], key
        if key in _BAND_PARAMS and self.band:
            return self.band[0], key
        if key in _TRANSIT_PARAMS and self.transit:
            return self.transit[0], key
        if key in _TELESCOPE_PARAMS and self.telescope:
            return self.telescope[0], key
        # SS-level attributes
        if hasattr(self, key):
            return self, key
        raise KeyError(key)

    # Log-space parameter mappings: logX ↔ X = 10**logX
    _LOG_PARAMS = {
        'logP': 'period',
        'logmstar': 'mstar',
    }

    def __getitem__(self, key):
        # Handle log-space params: logP → log10(period), logmstar → log10(mstar)
        if key in self._LOG_PARAMS:
            return np.log10(self[self._LOG_PARAMS[key]])
        obj, attr = self._resolve(key)
        val = getattr(obj, attr)
        if isinstance(val, Parameter):
            return val.value
        return val

    def __setitem__(self, key, value):
        # Handle log-space params: setting logP sets period = 10**logP
        if key in self._LOG_PARAMS:
            self[self._LOG_PARAMS[key]] = 10.0**value
            return
        obj, attr = self._resolve(key)
        current = getattr(obj, attr)
        if isinstance(current, Parameter):
            current.value = value
        else:
            setattr(obj, attr, value)

    def __contains__(self, key):
        if key in self._LOG_PARAMS:
            return True
        try:
            self._resolve(key)
            return True
        except KeyError:
            return False

    def get(self, key, default=None):
        try:
            return self[key]
        except KeyError:
            return default

    # ── Vector packing / unpacking ──

    def to_vector(self, param_names=None):
        """Pack current parameter values into a flat array for optimizer/MCMC."""
        names = param_names or self.param_names
        return np.array([self[name] for name in names])

    def from_vector(self, params, param_names=None):
        """Update parameter values from a flat array, then recompute derived."""
        names = param_names or self.param_names
        for name, val in zip(names, params):
            self[name] = float(val)
        self.compute_derived()

    # ── Derived quantities ──

    def compute_derived(self):
        """Recompute all derived stellar and planetary quantities for all stars/planets."""
        from .exozippy_chi2 import derive_logg, derive_lstar, derive_ar

        if not self.star:
            return

        # Loop over all stars
        for s in self.star:
            mstar = s.mstar.value
            rstar = s.rstar.value
            teff = s.teff.value
            distance = s.distance.value

            s.logg.value = derive_logg(mstar, rstar)
            s.lstar.value = derive_lstar(teff, rstar)
            s.rhostar.value = mstar / rstar**3  # in solar units
            s.parallax.value = 1000.0 / distance

        if not self.planet:
            return

        # Loop over all planets
        for pl in self.planet:
            # Use the star this planet orbits
            sidx = pl.starndx
            s = self.star[sidx] if sidx < len(self.star) else self.star[0]
            mstar = s.mstar.value
            rstar = s.rstar.value

            period = pl.period.value
            cosi = pl.cosi.value

            pl.ar.value = derive_ar(period, mstar, rstar)
            pl.inc_rad.value = np.arccos(cosi)
            pl.ideg.value = np.degrees(pl.inc_rad.value)
            pl.b.value = pl.ar.value * cosi
            pl.delta.value = pl.p.value ** 2  # always positive regardless of sign

            # Derive e/omega from vcve or sesinw/secosw
            if (pl.vcve is not None and pl.vcve.fit
                    and pl.lsinw is not None and pl.lcosw is not None):
                from .vcve2e import vcve2e
                sign_val = pl.sign.value if pl.sign is not None else None
                e_derived = vcve2e(pl.vcve.value,
                                   lsinw=pl.lsinw.value,
                                   lcosw=pl.lcosw.value,
                                   sign=sign_val)
                pl.e.value = e_derived
                pl.omega.value = np.arctan2(pl.lsinw.value, pl.lcosw.value)
                # Update sesinw/secosw for consistency
                if pl.sesinw is not None:
                    sqrte = np.sqrt(max(e_derived, 0.0))
                    pl.sesinw.value = sqrte * np.sin(pl.omega.value)
                    pl.secosw.value = sqrte * np.cos(pl.omega.value)
            elif pl.sesinw is not None and pl.secosw is not None:
                sesinw = pl.sesinw.value
                secosw = pl.secosw.value
                pl.e.value = sesinw**2 + secosw**2
                pl.omega.value = np.arctan2(sesinw, secosw)
