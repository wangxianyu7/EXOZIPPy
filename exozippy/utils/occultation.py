import numpy as np
from numba import njit

@njit
def cel_bulirsch_vec(k2, kc, p, a1, a2, a3, b1, b2, b3, f1, f2, f3):
    """
    Vectorized version of the Bulirsch-Stoer integration for computing
    limb darkening parameters during occultations with eccentric orbits.
    Implements the efficient recursive steps as per Mandel & Agol (2002).
    """

    ca = np.sqrt(k2 * 2.2e-16)

    # Avoid undefined k2 = 1 or kc = 0 cases
    mask = (k2 == 1.0) | (kc == 0.0)
    kc = np.where(mask, 2.22e-16, kc)

    ee = kc.copy()
    m = np.ones_like(kc)

    pos = np.where(p >= 0.0)[0]
    neg = np.where(p < 0.0)[0]

    pinv = np.zeros_like(k2)
    if pos.size:
        p[pos] = np.sqrt(p[pos])
        pinv[pos] = 1.0 / p[pos]
        b1[pos] *= pinv[pos]

    if neg.size:
        q = k2[neg].copy()
        g = 1.0 - p[neg]
        f = g - k2[neg]
        q *= (b1[neg] - a1[neg] * p[neg])
        ginv = 1.0 / g
        p[neg] = np.sqrt(f * ginv)
        a1[neg] = (a1[neg] - b1[neg]) * ginv
        pinv[neg] = 1.0 / p[neg]
        b1[neg] = -q * ginv**2 * pinv[neg] + a1[neg] * p[neg]

    # Compute recursion
    f1 = a1.copy()
    a1 += b1 * pinv
    g = ee * pinv
    b1 += f1 * g
    b1 *= 2
    p += g

    # Compute remainder with p = 1
    p1 = np.ones_like(p)
    g1 = ee.copy()

    f2 = a2.copy()
    f3 = a3.copy()

    a2 += b2
    b2 += f2 * g1
    b2 *= 2

    a3 += b3
    b3 += f3 * g1
    b3 *= 2

    p1 += g1

    g = m.copy()
    m += kc

    iter = 0
    itmax = 50

    while np.max(np.abs(g - kc) > g * ca) and (iter < itmax):
        kc = np.sqrt(ee)
        kc *= 2
        ee = kc * m

        f1 = a1.copy()
        f2 = a2.copy()
        f3 = a3.copy()

        pinv = 1.0 / p
        pinv1 = 1.0 / p1

        a1 += b1 * pinv
        a2 += b2 * pinv1
        a3 += b3 * pinv1

        g = ee * pinv
        g1 = ee * pinv1

        b1 += f1 * g
        b2 += f2 * g1
        b3 += f3 * g1

        b1 *= 2
        b2 *= 2
        b3 *= 2

        p += g
        p1 += g1

        g = m.copy()
        m += kc

        iter += 1

    dpi = np.pi
    f1 = 0.5 * dpi * (a1 * m + b1) / (m * (m + p))
    f2 = 0.5 * dpi * (a2 * m + b2) / (m * (m + p1))
    f3 = 0.5 * dpi * (a3 * m + b3) / (m * (m + p1))

    return f1, f2, f3

@njit
def ellke(k):
    """
    Computes the complete elliptic integrals of the first (kk) and
    second (ek) kind using Hastings' polynomial approximation.

    Parameters
    ----------
    k : float or array-like
        The elliptic modulus.

    Returns
    -------
    ek : float or ndarray
        Elliptic integral of the second kind.
    kk : float or ndarray
        Elliptic integral of the first kind.
    
    References
    ----------
    - Jason Eastman (2009), EXOFAST/IDL implementation
    - Hastings (1955) polynomial approximation
    """
    k = np.asarray(k, dtype=np.float64)
    m1 = 1.0 - k**2
    logm1 = np.log(m1)

    # Elliptic integral of the second kind
    a1 = 0.44325141463
    a2 = 0.06260601220
    a3 = 0.04757383546
    a4 = 0.01736506451
    b1 = 0.24998368310
    b2 = 0.09200180037
    b3 = 0.04069697526
    b4 = 0.00526449639

    ee1 = 1.0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ee2 = m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4))) * (-logm1)
    ek = ee1 + ee2

    # Elliptic integral of the first kind
    a0 = 1.38629436112
    a1 = 0.09666344259
    a2 = 0.03590092383
    a3 = 0.03742563713
    a4 = 0.01451196212
    b0 = 0.5
    b1 = 0.12498593597
    b2 = 0.06880248576
    b3 = 0.03328355346
    b4 = 0.00441787012

    ek1 = a0 + m1 * (a1 + m1 * (a2 + m1 * (a3 + m1 * a4)))
    ek2 = (b0 + m1 * (b1 + m1 * (b2 + m1 * (b3 + m1 * b4)))) * logm1
    kk = ek1 - ek2

    return ek, kk

@njit
def sqarea_triangle(z0, p0):
    """
    Computes sixteen times the square of the area of a triangle
    with sides 1, z0, and p0 using the Kahan method (Goldberg 1991).

    Parameters:
        z0 (array-like): Lengths of side z0.
        p0 (array-like): Lengths of side p0.

    Returns:
        numpy.ndarray: Array containing 16 times the squared areas.
    """
    z0 = np.asarray(z0, dtype=np.float64)
    sqarea = np.zeros_like(z0)

    # Six cases to consider
    pz1 = np.where((p0 <= z0) & (z0 <= 1))[0]
    if pz1.size:
        sqarea[pz1] = (p0 + (z0[pz1] + 1)) * (1 - (p0 - z0[pz1])) * \
                      (1 + (p0 - z0[pz1])) * (p0 + (z0[pz1] - 1))

    zp1 = np.where((z0 <= p0) & (p0 <= 1))[0]
    if zp1.size:
        sqarea[zp1] = (z0[zp1] + (p0 + 1)) * (1 - (z0[zp1] - p0)) * \
                      (1 + (z0[zp1] - p0)) * (z0[zp1] + (p0 - 1))

    p1z = np.where((p0 <= 1) & (1 <= z0))[0]
    if p1z.size:
        sqarea[p1z] = (p0 + (1 + z0[p1z])) * (z0[p1z] - (p0 - 1)) * \
                      (z0[p1z] + (p0 - 1)) * (p0 + (1 - z0[p1z]))

    z1p = np.where((z0 <= 1) & (1 <= p0))[0]
    if z1p.size:
        sqarea[z1p] = (z0[z1p] + (1 + p0)) * (p0 - (z0[z1p] - 1)) * \
                      (p0 + (z0[z1p] - 1)) * (z0[z1p] + (1 - p0))

    onepz = np.where((1 <= p0) & (p0 <= z0))[0]
    if onepz.size:
        sqarea[onepz] = (1 + (p0 + z0[onepz])) * (z0[onepz] - (1 - p0)) * \
                        (z0[onepz] + (1 - p0)) * (1 + (p0 - z0[onepz]))

    onezp = np.where((1 <= z0) & (z0 <= p0))[0]
    if onezp.size:
        sqarea[onezp] = (1 + (z0[onezp] + p0)) * (p0 - (1 - z0[onezp])) * \
                        (p0 + (1 - z0[onezp])) * (1 + (z0[onezp] - p0))

    return sqarea

@njit
def _exozippy_occultquad_cel_scalar(z0, u1, u2, p0, return_coeffs=False):
    """
    Full translation of exozippy_OCCULTQUAD_CEL from IDL.
    Computes flux for quadratically limb-darkened occultation.
    """
    # Ensure inputs are numpy arrays with double precision
    z = np.asarray(z0, dtype=np.float64) # Checked, good
    p = np.abs(np.float64(p0)) # Checked, good

    nz = len(z)  # Checked, good
    lambdad = np.zeros(nz, dtype=np.float64) # Checked, good
    etad = np.zeros(nz, dtype=np.float64) # Checked, good
    lambdae = np.zeros(nz, dtype=np.float64) # Checked, good

    # Intermediate terms (not used yet in logic, but included for completeness)
    # x1 = (p - z) ** 2
    # x2 = (p + z) ** 2
    # x3 = p ** 2 - z ** 2

    # Case 1: star is unocculted — only consider z < 1 + p and p > 0
    notusedyet = np.where((z < (1.0 + p)) & (p > 0.0))[0]
    if notusedyet.size == 0:
        # goto final (in Python, you can just skip further computation or wrap the rest in an if block)
        pass
    else:
        # Case 11: source completely occulted — if p >= 1 and z <= p - 1
        if p >= 1.0:
            z_notused = z[notusedyet]
            occulted = np.where(z_notused <= (p - 1.0))[0]
            mask = np.ones(z_notused.shape, dtype=np.bool_)
            mask[occulted] = False
            notused2 = np.where(mask)[0]


            if occulted.size > 0:
                ndxuse = notusedyet[occulted]
                etad[ndxuse] = 0.5  # corrected typo in paper
                lambdae[ndxuse] = 1.0
                # lambdad stays 0

            if len(notused2) == 0:
                # goto final
                pass
            else:
                notusedyet = notusedyet[notused2]
                
    # Case 2, 7, 8 - ingress/egress (uniform disk only)
    # Ingress/egress region: abs(1 - p) ≤ z < 1 + p
    z_notused = z[notusedyet]
    inegressuni = np.where((z_notused >= np.abs(1.0 - p)) & (z_notused < 1.0 + p))[0]

    if inegressuni.size > 0:
        ndxuse = notusedyet[inegressuni]
        z_ndx = z[ndxuse]

        # Compute triangle area
        sqarea = sqarea_triangle(z_ndx, p)
        kite_area2 = np.sqrt(sqarea)

        # Compute angles
        kap1 = np.arctan2(kite_area2, (1.0 - p) * (1.0 + p) + z_ndx ** 2)
        kap0 = np.arctan2(kite_area2, (p - 1.0) * (1.0 + p) + z_ndx ** 2)

        # lambda_e: uniform disk flux
        lambdae[ndxuse] = (p ** 2 * kap0 + kap1 - 0.5 * kite_area2) / np.pi

        # eta_d
        etad[ndxuse] = (
            1.0 / (2.0 * np.pi) * (
                kap1 +
                p ** 2 * (p ** 2 + 2.0 * z_ndx ** 2) * kap0 -
                0.25 * (1.0 + 5.0 * p ** 2 + z_ndx ** 2) * kite_area2
            )
        )
        
        
    # Case 5, 6, 7 — z == p (edge of planet at origin of star)
    z_notused = z[notusedyet]
    ocltor = np.where(z_notused == p)[0]  # indices where z == p
    # notused3 = np.setdiff1d(np.arange(len(z_notused)), ocltor)
    # notused5 = np.delete(np.arange(len(z_notused)), inside)
    notused3 = np.delete(np.arange(len(z_notused)), ocltor)

    if ocltor.size > 0:
        ndxuse = notusedyet[ocltor]
        z_ndx = z[ndxuse]

        if p < 0.5:
            # Case 5
            q = 2.0 * p
            Ek, Kk = ellke(q)  # complete elliptic integral of the first kind
            # lambda_4
            lambdad[ndxuse] = (1.0 / 3.0 +
                2.0 / (9.0 * np.pi) *
                (4.0 * (2.0 * p**2 - 1.0) * Ek + (1.0 - 4.0 * p**2) * Kk)
            )

            # eta_2
            etad[ndxuse] = 1.5 * p**4  # = 3*p^4/2

            # Uniform disk
            lambdae[ndxuse] = p**2
        elif p > 0.5:
            # Case 7
            q = 0.5 / p
            Ek, Kk = ellke(q)  # complete elliptic integral of the first kind


            # lambda_3
            lambdad[ndxuse] = (
                1.0 / 3.0 +
                (16.0 * p / (9.0 * np.pi)) * (2.0 * p**2 - 1.0) * Ek -
                ((32.0 * p**4 - 20.0 * p**2 + 3.0) / (9.0 * np.pi * p)) * Kk
            )

            # etad = already computed elsewhere (eta_1), so nothing to do

        else:
            # Case 6: p == 0.5 exactly
            lambdad[ndxuse] = 1.0 / 3.0 - (4.0 / (9.0 * np.pi))
            etad[ndxuse] = 3.0 / 32.0
    # Update notusedyet by removing indices we just processed
    if notused3.size == 0:
        # goto final
        pass
    else:
        notusedyet = notusedyet[notused3]
        
        
    # ;; Case 3, 4, 9, 10 - planet completely inside star
    z_notused = z[notusedyet]
    inside = np.where((p < 1.0) & (z_notused <= (1.0 - p)))[0]
    # notused5 = np.setdiff1d(np.arange(len(z_notused)), inside)
    notused5 = np.delete(np.arange(len(z_notused)), inside)

    if inside.size > 0:
        ndxuse = notusedyet[inside]
        z_ndx = z[ndxuse]

        # eta_2
        etad[ndxuse] = 0.5 * p ** 2 * (p ** 2 + 2 * z_ndx ** 2)
        # uniform disk
        lambdae[ndxuse] = p ** 2
        # Case 4: edge of planet hits edge of star
        edge = np.where(z_ndx == (1.0 - p))[0]
        # notused6 = np.setdiff1d(np.arange(len(z_ndx)), edge)
        notused6 = np.delete(np.arange(len(z_ndx)), edge)

        if edge.size > 0:
            term1 = (2.0 / (3.0 * np.pi)) * np.arccos(1.0 - 2.0 * p)
            term2 = (4.0 / (9.0 * np.pi)) * np.sqrt(p * (1.0 - p)) * (3.0 + 2.0 * p - 8.0 * p ** 2)
            term3 = (2.0 / 3.0) if p > 0.5 else 0.0
            lambdad[ndxuse[edge]] = term1 - term2 - term3
            if notused6.size == 0:
                pass
            else:
                ndxuse = ndxuse[notused6]
                z_ndx = z[ndxuse]
        # Case 10: center of planet hits center of star
        origin = np.where(z_ndx == 0.0)[0]
        # notused7 = np.setdiff1d(np.arange(len(z_ndx)), origin)
        notused7 = np.delete(np.arange(len(z_ndx)), origin)
        
        if origin.size > 0:
            lambdad[ndxuse[origin]] = - (2.0 / 3.0) * (1.0 - p ** 2) ** 1.5

            if notused7.size == 0:
                pass
            else:
                ndxuse = ndxuse[notused7]
                z_ndx = z[ndxuse]

        # Prepare arguments for cel_bulirsch_vec
        onembpr2 = (1 - z_ndx - p) * (1 + z_ndx + p)
        onembmr2 = (p - z_ndx + 1) * (1 - p + z_ndx)
        fourbr = 4 * z_ndx * p
        fourbrinv = 1.0 / fourbr
        k2 = onembpr2 * fourbrinv + 1.0
        k2inv = 1.0 / k2
        kc2 = onembpr2 / onembmr2
        kc = np.sqrt(np.clip(kc2, 0, None))

        bmrdbpr = (z_ndx - p) / (z_ndx + p)
        mu = 3 * bmrdbpr / onembmr2
        p_bulirsch = bmrdbpr ** 2 * onembpr2 / onembmr2

        # Run Bulirsch-style elliptic integral vector routine
        Piofk, Eofk, Em1mKdm = cel_bulirsch_vec(
            k2=k2inv,
            kc=kc.copy(),
            p=p_bulirsch.copy(),
            a1=1 + mu,
            a2=np.ones_like(k2),
            a3=np.ones_like(k2),
            b1=p_bulirsch + mu,
            b2=kc2,
            b3=np.zeros_like(k2),
            f1=None,
            f2=None,
            f3=None
        )

        # Final expression for lambdad
        lambdad[ndxuse] = (
            2 * np.sqrt(onembmr2) *
            (onembpr2 * Piofk - (4 - 7 * p ** 2 - z_ndx ** 2) * Eofk)
            / (9.0 * np.pi)
        )

    # Case 2, 8 – Ingress/Egress with limb darkening
    inegress = notused5
    if inegress.size > 0:
        ndxuse = notusedyet[inegress]
        z_ndx = z[ndxuse]

        # Geometric terms
        onembpr2 = (1 - z_ndx - p) * (1 + z_ndx + p)
        onembmr2 = (p - z_ndx + 1) * (1 - p + z_ndx)
        fourbr = 4 * z_ndx * p
        fourbrinv = 1.0 / fourbr

        k2 = onembpr2 * fourbrinv + 1
        kc2 = -onembpr2 * fourbrinv
        kc = np.sqrt(np.clip(kc2, 0, None))  # clamp to prevent NaN

        # Arguments for cel_bulirsch_vec
        a1 = np.zeros_like(k2)         # 0.0
        a2 = np.ones_like(k2)          # 1.0
        a3 = np.ones_like(k2)          # 1.0
        b1 = 3 * kc2 * (z_ndx - p) * (z_ndx + p)
        b2 = kc2
        b3 = np.zeros_like(k2)
        p_bulirsch = (z_ndx - p) ** 2 * kc2

        # Call vectorized Bulirsch integrator
        Piofk, Eofk, Em1mKdm = cel_bulirsch_vec(
            k2=k2,
            kc=kc.copy(),
            p=p_bulirsch.copy(),
            a1=a1,
            a2=a2,
            a3=a3,
            b1=b1,
            b2=b2,
            b3=b3,
            f1=None,
            f2=None,
            f3=None
        )
        # Final lambdad for ingress/egress + limb darkening
        lambdad[ndxuse] = (
            onembmr2 * (
                Piofk +
                (-3 + 6 * p ** 2 + 2 * z_ndx * p) * Em1mKdm -
                fourbr * Eofk
            ) / (9 * np.pi * np.sqrt(z_ndx * p))
        )
    # === Final Light Curve Computation ===
    omega = 1.0 - u1 / 3.0 - u2 / 6.0
    z_mask = p > z  # for condition (p > z)
    if p0 > 0:
        # Limb-darkened flux
        muo1 = 1.0 - (
            (1.0 - u1 - 2.0 * u2) * lambdae +
            (u1 + 2.0 * u2) * (lambdad + (2.0 / 3.0) * z_mask) +
            u2 * etad
        ) / omega

        # Uniform disk
        mu0 = 1.0 - lambdae

        # Optional limb darkening coefficient output
        if return_coeffs:
            N = lambdae.shape[0]
            d = np.empty((3, N), dtype=np.float64)
            d[0, :] = 1.0 - lambdae
            d[1, :] = (2.0 / 3.0) * (lambdae - z_mask) - lambdad
            d[2, :] = lambdae / 2.0 - etad
            return muo1, mu0, d
        else:
            d = np.empty((3, 1), dtype=np.float64)  # dummy placeholder
            return muo1, mu0, d

    else:
        # Negative p0 — treat as anti-transit (e.g., for symmetry or edge cases)
        muo1 = 1.0 + (
            (1.0 - u1 - 2.0 * u2) * lambdae +
            (u1 + 2.0 * u2) * (lambdad + (2.0 / 3.0) * z_mask) +
            u2 * etad
        ) / omega

        mu0 = 1.0 + lambdae
        if return_coeffs:
            N = lambdae.shape[0]
            d = np.empty((3, N), dtype=np.float64)
            d[0, :] = 1.0 + lambdae
            d[1, :] = (2.0 / 3.0) * (z_mask - lambdae) + lambdad
            d[2, :] = etad - lambdae / 2.0
            return muo1, mu0, d
        else:
            d = np.empty((3, 1), dtype=np.float64)  # dummy placeholder
            return muo1, mu0, d

def exozippy_occultquad_cel(z0, u1, u2, p0, return_coeffs=False):
    """
    Wrapper that supports scalar or vector p0/u1/u2 by looping over samples
    when needed, while keeping the scalar core JIT-compiled.
    """
    z = np.asarray(z0, dtype=np.float64)
    p = np.asarray(p0, dtype=np.float64)
    u1_arr = np.asarray(u1, dtype=np.float64)
    u2_arr = np.asarray(u2, dtype=np.float64)

    if p.ndim == 0 or p.size == 1:
        p_scalar = float(p.reshape(-1)[0])
        return _exozippy_occultquad_cel_scalar(z, u1_arr, u2_arr, p_scalar, return_coeffs)

    if z.size != p.size:
        raise ValueError("p0 array must match z0 size for exozippy_occultquad_cel")

    n = z.size
    muo1 = np.empty(n, dtype=np.float64)
    mu0 = np.empty(n, dtype=np.float64)
    if return_coeffs:
        d = np.empty((3, n), dtype=np.float64)
    else:
        d = np.empty((3, 1), dtype=np.float64)

    u1_is_scalar = (u1_arr.ndim == 0 or u1_arr.size == 1)
    u2_is_scalar = (u2_arr.ndim == 0 or u2_arr.size == 1)
    u1_scalar = float(u1_arr.reshape(-1)[0]) if u1_is_scalar else 0.0
    u2_scalar = float(u2_arr.reshape(-1)[0]) if u2_is_scalar else 0.0

    for i in range(n):
        u1_i = u1_scalar if u1_is_scalar else float(u1_arr[i])
        u2_i = u2_scalar if u2_is_scalar else float(u2_arr[i])
        muo1_i, mu0_i, d_i = _exozippy_occultquad_cel_scalar(
            np.array([z[i]], dtype=np.float64),
            u1_i,
            u2_i,
            float(p[i]),
            return_coeffs
        )
        muo1[i] = muo1_i[0]
        mu0[i] = mu0_i[0]
        if return_coeffs:
            d[:, i] = d_i[:, 0]

    return muo1, mu0, d

