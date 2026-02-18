"""I/O and setup helpers for exoplanet fitting workflows."""

import glob as _glob

import numpy as np

from exozippy.jointfit.mkss import mkss, _parse_priors, _read_data_with_detrend

def parse_priors(priorfile):
    """Parse an EXOFASTv2-style prior file."""
    return _parse_priors(priorfile)

def _resolve_glob(pattern):
    """Expand a glob pattern to the first matching file, or return as-is."""
    if '*' in str(pattern) or '?' in str(pattern):
        matches = sorted(_glob.glob(str(pattern)))
        if matches:
            return matches[0]
    return str(pattern)

def read_transit_data(tranfile):
    """Read a transit light curve file (BJD flux err [detrend_cols...])."""
    tranfile = _resolve_glob(tranfile)
    bjd, flux, err, detrendadd, detrendmult = _read_data_with_detrend(tranfile)
    d = {'bjd': bjd, 'flux': flux, 'err': err}
    if detrendadd is not None:
        d['detrendadd'] = detrendadd
    if detrendmult is not None:
        d['detrendmult'] = detrendmult
    return d

def read_rv_data(rvfile):
    """Read an RV data file (BJD vel err_vel [detrend_cols...])."""
    rvfile = _resolve_glob(rvfile)
    bjd, vel, err, detrendadd, detrendmult = _read_data_with_detrend(rvfile)
    d = {'bjd': bjd, 'vel': vel, 'err': err}
    if detrendadd is not None:
        d['detrendadd'] = detrendadd
    if detrendmult is not None:
        d['detrendmult'] = detrendmult
    return d

def read_all_transit_data(tranpath):
    """Read all transit files matching a glob pattern.

    Returns (list[dict], list[str]) — data dicts and file paths.
    """
    files = sorted(_glob.glob(str(tranpath)))
    return [read_transit_data(f) for f in files], files

def read_all_rv_data(rvpath):
    """Read all RV files matching a glob pattern.

    Returns (list[dict], list[str]) — data dicts and file paths.
    """
    files = sorted(_glob.glob(str(rvpath)))
    return [read_rv_data(f) for f in files], files

def read_all_dt_data(dtpath):
    """Read all Doppler Tomography FITS files matching a glob pattern.

    Returns (list[dict], list[str]) — data dicts and file paths.
    """
    from exozippy.physics.exozippy_dopptom import read_dt_fits
    files = sorted(_glob.glob(str(dtpath)))
    return [read_dt_fits(f) for f in files], files

def _build_detrend_info(tran_data_list, rv_data_list):
    """Build detrend_info dict from data lists. Returns None if no detrending."""
    tran_nadd = [td.get('detrendadd', np.empty((0, 0))).shape[0] if td.get('detrendadd') is not None else 0
                 for td in tran_data_list]
    tran_nmult = [td.get('detrendmult', np.empty((0, 0))).shape[0] if td.get('detrendmult') is not None else 0
                  for td in tran_data_list]
    rv_nadd = [rd.get('detrendadd', np.empty((0, 0))).shape[0] if rd.get('detrendadd') is not None else 0
               for rd in rv_data_list]
    rv_nmult = [rd.get('detrendmult', np.empty((0, 0))).shape[0] if rd.get('detrendmult') is not None else 0
                for rd in rv_data_list]
    if sum(tran_nadd) + sum(tran_nmult) + sum(rv_nadd) + sum(rv_nmult) == 0:
        return None
    return dict(tran_nadd=tran_nadd, tran_nmult=tran_nmult,
                rv_nadd=rv_nadd, rv_nmult=rv_nmult)

def build_initial_guess(priorfile, tranfile, rvfile, e=0.0,
                        omega=np.pi/2, circular=True, usevcve=False,
                        use_mist=False, nstars=1,
                        fitjittervar=False, fitvariance=False,
                        fitdilute=False, fitttv=False,
                        fitslope=False, fitquad=False,
                        fitthermal=False, fitreflect=False,
                        fitbeam=False, fitellip=False,
                        rossiter=False, rmbands=None,
                        dtpath=None, fitdt=False, fiterrscale=False):
    """
    Construct an SS object from priors (before optimization).
    """
    ss = mkss(
        parfile=priorfile,
        tranpath=tranfile,
        rvpath=rvfile,
        use_mist=use_mist,
        nstars=nstars,
        circular=circular,
        usevcve=usevcve,
        fitjittervar=fitjittervar, fitvariance=fitvariance,
        fitdilute=fitdilute, fitttv=fitttv,
        fitslope=fitslope, fitquad=fitquad,
        fitthermal=fitthermal, fitreflect=fitreflect,
        fitbeam=fitbeam, fitellip=fitellip,
        rossiter=rossiter, rmbands=rmbands,
    )
    ss.planet[0].e.value = e
    ss.planet[0].omega.value = omega
    # Initialize sesinw/secosw from e/omega
    sqrte = np.sqrt(e)
    ss.planet[0].sesinw.value = sqrte * np.sin(omega)
    ss.planet[0].secosw.value = sqrte * np.cos(omega)

    priors = _parse_priors(priorfile)
    if 'tc' not in priors:
        tran_data_list, _ = read_all_transit_data(tranfile)
        if tran_data_list:
            all_bjd = np.concatenate([td['bjd'] for td in tran_data_list])
            ss.planet[0].tc.value = float(np.median(all_bjd))

    if 'parallax' in priors:
        ss.star[0].distance.value = 1000.0 / priors['parallax']['value']

    ss.compute_derived()
    return ss
