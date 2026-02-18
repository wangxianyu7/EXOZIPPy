import numpy as np


def _round_err(err):
    if err == 0 or not np.isfinite(err):
        return err, 0
    exp = int(np.floor(np.log10(abs(err))))
    ndigits = 1 - exp
    rounded = np.round(err, ndigits)
    return rounded, ndigits


def _format_value(med, up, low):
    if not np.isfinite(med):
        return "$\\mathrm{NaN}$"
    up_r, nd = _round_err(up)
    low_r, _ = _round_err(low)
    med_r = np.round(med, nd) if nd > 0 else np.round(med, 0)
    if np.isclose(up_r, low_r, rtol=0, atol=10**(-nd) if nd > 0 else 0):
        return f"${med_r:g}\\pm{up_r:g}$"
    return f"${med_r:g}^{{+{up_r:g}}}_{{-{low_r:g}}}$"


def summarize_samples(data):
    """
    data: dict name -> array
    return dict name -> (median, plus, minus)
    """
    summary = {}
    for name, arr in data.items():
        arr = np.asarray(arr, dtype=float)
        med = np.nanmedian(arr)
        lo = np.nanpercentile(arr, 15.87)
        hi = np.nanpercentile(arr, 84.13)
        summary[name] = (med, hi - med, med - lo)
    return summary


def write_csv(summary, path, order=None):
    with open(path, "w") as f:
        f.write("#parname, median value, upper errorbar, lower errorbar\n")
        keys = order if order is not None else list(summary.keys())
        for name in keys:
            if name not in summary:
                continue
            med, up, low = summary[name]
            f.write(f"{name},{med:.6g},{up:.6g},{low:.6g}\n")


def exozippy_latextab(summary, path, caption=None, label=None,
                      section_labels=None, order=None):
    section_labels = section_labels or {}

    def _section(title, tag):
        tag = "" if tag is None else str(tag)
        lines.append(rf"\smallskip\\\multicolumn{{2}}{{l}}{{{title}}}&{tag}\smallskip\\")

    def _row(key, latex, desc):
        if key not in summary:
            return
        med, up, low = summary[key]
        val = _format_value(med, up, low)
        lines.append(rf"~~~~${latex}$\dotfill &{desc}\dotfill &{val}\\")

    lines = []
    lines.append(r"\documentclass{aastex62}")
    lines.append(r"\providecommand{\bjdtdb}{\ensuremath{\rm {BJD_{TDB}}}}")
    lines.append(r"\providecommand{\tjdtdb}{\ensuremath{\rm {TJD_{TDB}}}}")
    lines.append(r"\providecommand{\feh}{\ensuremath{\left[{\rm Fe}/{\rm H}\right]}}")
    lines.append(r"\providecommand{\teff}{\ensuremath{T_{\rm eff}}}")
    lines.append(r"\providecommand{\teq}{\ensuremath{T_{\rm eq}}}")
    lines.append(r"\providecommand{\ecosw}{\ensuremath{e\cos{\omega_*}}}")
    lines.append(r"\providecommand{\esinw}{\ensuremath{e\sin{\omega_*}}}")
    lines.append(r"\providecommand{\msun}{\ensuremath{\,M_\Sun}}")
    lines.append(r"\providecommand{\rsun}{\ensuremath{\,R_\Sun}}")
    lines.append(r"\providecommand{\lsun}{\ensuremath{\,L_\Sun}}")
    lines.append(r"\providecommand{\mj}{\ensuremath{\,M_{\rm J}}}")
    lines.append(r"\providecommand{\rj}{\ensuremath{\,R_{\rm J}}}")
    lines.append(r"\providecommand{\me}{\ensuremath{\,M_{\rm E}}}")
    lines.append(r"\providecommand{\re}{\ensuremath{\,R_{\rm E}}}")
    lines.append(r"\providecommand{\fave}{\langle F \rangle}")
    lines.append(r"\providecommand{\fluxcgs}{10$^9$ erg s$^{-1}$ cm$^{-2}$}")
    # lines.append(r"\usepackage{apjfonts}")
    lines.append(r"\begin{document}")
    lines.append(r"\startlongtable")
    lines.append(r"\begin{deluxetable*}{lcc}")
    if caption:
        lines.append(rf"\tablecaption{{{caption}}}")
    lines.append(r"\tablehead{\colhead{~~~Parameter} & \colhead{Description} & \colhead{Values}}")
    lines.append(r"\startdata")

    # EXOFAST-style ordering and labels (single-star/single-planet default)
    order_map = [
        ("Stellar Parameters:", section_labels.get("stellar", "0"), [
            ("mstar_0", r"M_*", r"Mass (\msun)"),
            ("rstar_0", r"R_*", r"Radius (\rsun)"),
            ("lstar_0", r"L_*", r"Luminosity (\lsun)"),
            ("rhostar_0", r"\rho_*", r"Density (cgs)"),
            ("logg_0", r"\log{g}", r"Surface gravity (cgs)"),
            ("teff_0", r"T_{\rm eff}", r"Effective temperature (K)"),
            ("feh_0", r"[{\rm Fe/H}]", r"Metallicity (dex)"),
        ]),
        ("Planetary Parameters:", section_labels.get("planet", "b"), [
            ("Period_0", r"P", r"Period (days)"),
            ("rp_0", r"R_P", r"Radius (\rj)"),
            ("mp_0", r"M_P", r"Mass (\mj)"),
            ("tco_0", r"T_C", r"Observed Time of conjunction$^{1}$ (\bjdtdb)"),
            ("tc_0", r"T_C", r"Model Time of conjunction$^{1,2}$ (\tjdtdb)"),
            ("tt_0", r"T_T", r"Model time of min proj sep$^{2,3,4}$ (\tjdtdb)"),
            ("t0_0", r"T_0", r"Obs time of min proj sep$^{3,5,6}$ (\bjdtdb)"),
            ("a_0", r"a", r"Semi-major axis (AU)"),
            ("ideg_0", r"i", r"Inclination (Degrees)"),
            ("omegagr_0", r"\dot{\omega}_{\rm GR}", r"Computed GR precession ($^\circ$/century)"),
            ("teq_0", r"T_{\rm eq}", r"Equilibrium temp$^{7}$ (K)"),
            ("tcirc_0", r"\tau_{\rm circ}", r"Tidal circ timescale (Gyr)"),
            ("tefficiency_0", r"\tau_{\rm tidalefficiency}", r"Tidal efficiency factor"),
            ("tce_0", r"\tau_{\rm CE}", r"Convective tidal realignment timescale (Gyr)"),
            ("tra_0", r"\tau_{\rm RA}", r"radiative realignment timescale (Gyr)"),
            ("k_0", r"K", r"RV semi-amplitude (m/s)"),
            ("p_0", r"R_P/R_*", r"Radius of planet in stellar radii"),
            ("ar_0", r"a/R_*", r"Semi-major axis in stellar radii"),
            ("arp_0", r"a/R_p", r"Semi-major axis in planetary radii"),
            ("delta_0", r"\delta", r"$\left(R_P/R_*\right)^2$"),
            ("depth_Sloani_0", r"\delta_{i'}", r"Transit depth in i' (frac)"),
            ("tau_0", r"\tau", r"In/egress transit duration (days)"),
            ("t14_0", r"T_{14}", r"Total transit duration (days)"),
            ("tfwhm_0", r"T_{FWHM}", r"FWHM transit duration (days)"),
            ("b_0", r"b", r"Transit impact parameter"),
            ("eclipsedepth25_0", r"\delta_{S,2.5\mu m}", r"BB eclipse depth at 2.5$\mu$m (ppm)"),
            ("eclipsedepth50_0", r"\delta_{S,5.0\mu m}", r"BB eclipse depth at 5.0$\mu$m (ppm)"),
            ("eclipsedepth75_0", r"\delta_{S,7.5\mu m}", r"BB eclipse depth at 7.5$\mu$m (ppm)"),
            ("rhop_0", r"\rho_P", r"Density (cgs)"),
            ("loggp_0", r"logg_P", r"Surface gravity (cgs)"),
            ("safronov_0", r"\Theta", r"Safronov Number"),
            ("fave_0", r"\fave", r"Incident Flux (\fluxcgs)"),
            ("tso_0", r"T_S", r"Observed Time of eclipse$^{1}$ (\bjdtdb)"),
            ("ts_0", r"T_S", r"Model Time of eclipse$^{1,2}$ (\tjdtdb)"),
            ("te_0", r"T_E", r"Model time of sec min proj sep$^{2,3,4}$ (\tjdtdb)"),
            ("te0_0", r"T_{E,0}", r"Obs time of sec min proj sep$^{3,5,6}$ (\bjdtdb)"),
            ("tp_0", r"T_P", r"Time of Periastron (\tjdtdb)"),
            ("ta_0", r"T_A", r"Time of asc node (\tjdtdb)"),
            ("td_0", r"T_D", r"Time of desc node (\tjdtdb)"),
            ("vcve_0", r"V_c/V_e", r"Scaled velocity"),
            ("msini_0", r"M_P\sin i", r"Minimum mass (\mj)"),
            ("q_0", r"M_P/M_*", r"Mass ratio"),
            ("dr_0", r"d/R_*", r"Separation at mid transit"),
            ("pt_0", r"P_T", r"A priori non-grazing transit prob"),
            ("ptg_0", r"P_{T,G}", r"A priori transit prob"),
        ]),
        ("Wavelength Parameters:", section_labels.get("band", "i'"), [
            ("u1_0", r"u_{1}", r"Linear limb-darkening coeff"),
            ("u2_0", r"u_{2}", r"Quadratic limb-darkening coeff"),
        ]),
        ("Telescope Parameters:", section_labels.get("telescope", "HIRES"), [
            ("gamma_0", r"\gamma_{\rm rel}", r"Relative RV Offset (m/s)"),
            ("jitter_0", r"\sigma_J", r"RV Jitter (m/s)"),
            ("jittervar_0", r"\sigma_J^2", r"RV Jitter Variance"),
        ]),
        ("Transit Parameters:", section_labels.get("transit", "KepCam UT 2007-04-28 (i')"), [
            ("variance_0", r"\sigma^{2}", r"Added Variance"),
            ("f0_0", r"F_0", r"Baseline flux"),
        ]),
    ]

    if order is None:
        order = []
        for _, _, items in order_map:
            order.extend([k for k, _, _ in items])

    for title, tag, items in order_map:
        _section(title, tag)
        for key, latex, desc in items:
            _row(key, latex, desc)

    lines.append(r"\enddata")
    if label:
        lines.append(rf"\label{{{label}}}")
    lines.append(r"\end{deluxetable*}")
    lines.append(r"\bibliographystyle{apj}")
    lines.append(r"\bibliography{References}")
    lines.append(r"\end{document}")

    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# Backward-compatible name
write_latex = exozippy_latextab
