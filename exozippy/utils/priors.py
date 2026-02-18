import re

import numpy as np
import requests

def angsep(ra1, dec1, ra2, dec2):
    arg = np.sin(dec1) * np.sin(dec2) + np.cos(dec1) * np.cos(dec2) * np.cos(ra1 - ra2)
    arg = np.clip(arg, -1.0, 1.0)
    return np.arccos(arg)



# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def get_av_prior(ra=None, dec=None, object_name=None):
    """
    Returns the maximum V-band extinction (Av) from Schlafly and Finkbeiner (2011).

    Parameters
    ----------
    ra : float, optional
        J2000 right ascension in decimal degrees.
    dec : float, optional
        J2000 declination in decimal degrees.
    object_name : str, optional
        Object name to resolve via SIMBAD or IRSA.

    Returns
    -------
    maxav : float
        Maximum Av value.
    line : str
        String of the form 'av 0 -1 0 <maxav>' for EXOFASTv2.
    """

    if object_name:
        url = f"https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr={object_name}"
    elif ra is not None and dec is not None:
        url = f"https://irsa.ipac.caltech.edu/cgi-bin/DUST/nph-dust?locstr={ra}+{dec}+equ+j2000"
    else:
        raise ValueError("Must specify either object_name or both ra and dec.")

    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        content = response.text.splitlines()
    except Exception as e:
        print(f"[WARNING] Failed to query dust map: {e}")
        return 99.0, "av 0 -1 0 99"

    if len(content) == 0 or "Invalid object name" in content[2]:
        print("[WARNING] Invalid object name or empty response.")
        return 99.0, "av 0 -1 0 99"

    try:
        # Find the last line before the closing tag
        match_index = next(i for i, line in enumerate(content) if '</maxValueSandF>' in line)
        raw_value = content[match_index - 1]
        ebv = float(re.split(r'\(|\s', raw_value.strip())[0])
        maxav = ebv * 3.1 * 1.5 * 0.87
    except Exception as e:
        print(f"[WARNING] Parsing failed: {e}")
        return 99.0, "av 0 -1 0 99"

    line = f"av 0 -1 0 {maxav:.6f}"
    return maxav, line

