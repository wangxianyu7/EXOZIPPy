"""
Lightweight mkss.py (non-PyMC).

Builds a plain-Python SS-like structure with stars/planets/transits/telescopes.
Intended as a minimal, extensible analog of EXOFASTv2 mkss.pro.
"""
import glob
import math
import numpy as np

from .mkconstants import mkconstants
from .parameter import Parameter
from .read_par import read_par
from .read_tran import read_tran
from .read_rv import read_rv


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
    ttvs=False,
    tivs=False,
    tdeltavs=False,
):
    """
    Construct a minimal SS-like dict for EXOZIPPy (no PyMC dependency).
    """
    const = mkconstants()
    user_params = read_par(parfile) if parfile else {}

    # Data inputs
    tranfiles = glob.glob(tranpath) if tranpath else []
    rvfiles = glob.glob(rvpath) if rvpath else []

    # Handle scalar/array flags
    if np.isscalar(fittran):
        fittran = np.zeros((nplanets,), dtype=bool) + bool(fittran)
    if np.isscalar(fitrv):
        fitrv = np.zeros((nplanets,), dtype=bool) + bool(fitrv)
    if np.isscalar(ttvs):
        ttvs = np.zeros((len(tranfiles),), dtype=bool) + bool(ttvs)
    if np.isscalar(tivs):
        tivs = np.zeros((len(tranfiles),), dtype=bool) + bool(tivs)
    if np.isscalar(tdeltavs):
        tdeltavs = np.zeros((len(tranfiles),), dtype=bool) + bool(tdeltavs)
    if np.isscalar(circular):
        circular = np.zeros((nplanets,), dtype=bool) + bool(circular)

    starnames = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    ss = {
        "constants": const,
        "nstars": nstars,
        "nplanets": nplanets,
        "sedfile": sedfile,
        "star": [],
        "planet": [],
        "transit": [],
        "telescope": [],
        "fittran": fittran,
        "fitrv": fitrv,
        "circular": circular,
    }

    # --- Stars ---
    for i in range(nstars):
        star = {
            "rootlabel": "Stellar Parameters:",
            "label": starnames[i],
            "mstar": Parameter(f"mstar_{i}", lower=1e-1, upper=250, initval=1.0,
                               latex="M_*", description="Mass", latex_unit="\\msun",
                               user_params=user_params),
            "rstar": Parameter(f"rstar_{i}", lower=1e-1, upper=2000, initval=1.0,
                               latex="R_*", description="Radius", latex_unit="\\rsun",
                               user_params=user_params),
            "teff": Parameter(f"teff_{i}", lower=1.0, upper=5e5, initval=5778,
                              latex="T_{\\rm eff}", description="Effective Temperature",
                              latex_unit="K", user_params=user_params),
            "feh": Parameter(f"feh_{i}", lower=-5.0, upper=5.0, initval=0.0,
                             latex="[{\\rm Fe/H}]", description="Metallicity",
                             latex_unit="dex", user_params=user_params),
            "distance": Parameter(f"distance_{i}", lower=1.0, upper=1e6, initval=100.0,
                                  latex="d", description="Distance", latex_unit="pc",
                                  user_params=user_params),
        }

        # Derived values (numeric)
        mstar = float(star["mstar"].value)
        rstar = float(star["rstar"].value)
        teff = float(star["teff"].value)
        star["logg"] = Parameter(
            f"logg_{i}",
            initval=math.log10(mstar / rstar**2 * const["GravitySun"]),
            latex="\\log{g}", description="Surface gravity",
            latex_unit="cgs", user_params=user_params,
        )
        star["rhostar"] = Parameter(
            f"rhostar_{i}",
            initval=mstar / (rstar**3) * const["RhoSun"],
            latex="\\rho_*", description="Density",
            latex_unit="cgs", user_params=user_params,
        )
        star["lstar"] = Parameter(
            f"lstar_{i}",
            initval=4.0 * math.pi * rstar**2 * teff**4 * const["sigmab"] / const["LSun"] * const["RSun"]**2,
            latex="L_*", description="Luminosity",
            latex_unit="\\lsun", user_params=user_params,
        )

        ss["star"].append(star)

    # --- Planets (minimal set) ---
    for i in range(nplanets):
        planet = {
            "rootlabel": "Planetary Parameters:",
            "label": f"b{i}",
            "period": Parameter(f"period_{i}", lower=1e-6, upper=1e6, initval=3.0,
                                latex="P", description="Period", latex_unit="days",
                                user_params=user_params),
            "tc": Parameter(f"tc_{i}", lower=-np.inf, upper=np.inf, initval=0.0,
                            latex="T_C", description="Transit time", latex_unit="\\bjdtdb",
                            user_params=user_params),
            "p": Parameter(f"p_{i}", lower=1e-4, upper=1.0, initval=0.1,
                           latex="R_P/R_*", description="Radius ratio",
                           latex_unit="", user_params=user_params),
            "cosi": Parameter(f"cosi_{i}", lower=0.0, upper=1.0, initval=0.05,
                              latex="\\cos i", description="Cosine inclination",
                              latex_unit="", user_params=user_params),
            "K": Parameter(f"k_{i}", lower=0.0, upper=1e4, initval=50.0,
                           latex="K", description="RV semi-amplitude",
                           latex_unit="m/s", user_params=user_params),
            "gamma": Parameter(f"gamma_{i}", lower=-1e4, upper=1e4, initval=0.0,
                               latex="\\gamma", description="RV offset",
                               latex_unit="m/s", user_params=user_params),
        }
        ss["planet"].append(planet)

    # --- Transits ---
    for i, fname in enumerate(tranfiles):
        transit = read_tran(fname, ndx=i, tiv=tivs[i], ttv=ttvs[i], tdeltav=tdeltavs[i],
                            user_params=user_params)
        ss["transit"].append(transit)

    # --- Telescopes (RV) ---
    for i, fname in enumerate(rvfiles):
        rv = read_rv(fname)
        rv["rootlabel"] = "Telescope Parameters:"
        rv["label"] = rv.get("label", f"RV{i}")
        ss["telescope"].append(rv)

    return ss

