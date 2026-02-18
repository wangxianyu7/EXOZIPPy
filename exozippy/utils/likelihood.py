import numpy as np

def vcve2e(vcve0, omega=None, lsinw=None, lcosw=None, sign=None):
    """
    Convert vcve to eccentricity e using omega or sin(omega), cos(omega).
    
    Parameters
    ----------
    vcve0 : float or array_like
        The quantity sqrt((1 - e^2) / (1 + e sin(omega))) to be inverted.
    omega : float or array_like, optional
        Argument of periastron in radians.
    lsinw, lcosw : float or array_like, optional
        Sine and cosine of omega multiplied by sqrt(e), used to recover omega if not given.
    sign : int or array_like, optional
        Sign to select between two solutions (0 = positive root, 1 = negative root).

    Returns
    -------
    e : float or np.ndarray
        Eccentricity.
    """
    vcve = np.atleast_1d(vcve0).astype(float)

    if omega is None:
        if lsinw is None or lcosw is None:
            raise ValueError("Must specify either omega or lsinw and lcosw")
        omega = np.arctan2(lsinw, lcosw)
    else:
        omega = np.atleast_1d(omega).astype(float)

    # Broadcast shapes
    if vcve.size == 1 and omega.size > 1:
        vcve = np.full_like(omega, vcve[0])
    elif omega.size == 1 and vcve.size > 1:
        omega = np.full_like(vcve, omega[0])
    
    if sign is None:
        if lsinw is not None and lcosw is not None:
            L2 = lsinw**2 + lcosw**2
            useneg = L2 >= 0.5
        else:
            a = vcve**2 * np.sin(omega)**2 + 1
            b = 2 * vcve**2 * np.sin(omega)
            c = vcve**2 - 1
            disc = b**2 - 4 * a * c
            epos = (-b + np.sqrt(disc)) / (2 * a)
            eneg = (-b - np.sqrt(disc)) / (2 * a)

            good_neg = np.isfinite(eneg) & (eneg >= 0) & (eneg < 1) & (eneg < epos)
            sign = np.zeros_like(vcve, dtype=int)
            sign[good_neg] = 1
            if vcve.size == 1:
                sign = int(sign[0])
            useneg = sign == 1
    else:
        sign = np.atleast_1d(sign)
        if sign.size == 1 and vcve.size > 1:
            sign = np.full_like(vcve, sign[0])
        useneg = np.floor(sign).astype(bool)

    # Solve quadratic
    a = vcve**2 * np.sin(omega)**2 + 1
    b = 2 * vcve**2 * np.sin(omega)
    c = vcve**2 - 1
    disc = b**2 - 4 * a * c

    epos = (-b + np.sqrt(disc)) / (2 * a)
    eneg = (-b - np.sqrt(disc)) / (2 * a)

    e = np.zeros_like(vcve)
    usepos = ~useneg
    e[usepos] = epos[usepos]
    e[useneg] = eneg[useneg]

    return e[0] if e.size == 1 else e


# ⚠️ Initial auto-translation from IDL (ChatGPT). Review required.
def exozippy_like(residuals, var_r, sigma_w0, chi2=False, truechi2=False):
    """
    Compute the log likelihood for a given residual and noise model. 
    A simpler (and faster) alternative to the wavelet analysis of Carter & Winn 2009.

    Parameters
    ----------
    residuals : array_like
        The residuals of a fit (data - model).
    var_r : float
        Red noise amplitude variance (sigma_r^2).
    sigma_w0 : float or array_like
        White noise amplitude (or error array).
    chi2 : bool, optional
        If True, returns the "effective chi2" used for EXOFAST_DEMC.
    truechi2 : bool, optional
        If True, returns the true chi2 value.

    Returns
    -------
    float
        Log-likelihood, effective chi2, or true chi2 depending on keywords.
    """
    residuals = np.asarray(residuals)
    if np.isscalar(sigma_w0):
        sigma_w = np.full_like(residuals, sigma_w0)
    else:
        sigma_w = np.asarray(sigma_w0)

    good = np.isfinite(sigma_w)
    if not np.any(good):
        return np.inf if chi2 else -np.inf

    denom = sigma_w[good]**2 + var_r
    chisq = np.sum(residuals[good]**2 / denom)

    if truechi2:
        return chisq

    loglike = -0.5 * (np.sum(np.log(2.0 * np.pi * denom)) + chisq)

    if not np.isfinite(loglike):
        return np.inf if chi2 else -np.inf

    return -2.0 * loglike if chi2 else loglike

