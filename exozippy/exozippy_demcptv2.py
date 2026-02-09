"""
demcpt.py — Differential Evolution MCMC with Parallel Tempering

Python implementation of EXOFASTv2's exofast_demcpt_multi.pro
(Eastman et al. 2013, 2019; ter Braak 2006; Ford 2006).

Usage
-----
    from demcpt import DEMCPTSampler

    def log_posterior(theta):
        return -0.5 * np.sum(theta**2)

    sampler = DEMCPTSampler(log_posterior, ndim=5, nchains=20)
    converged = sampler.run(p0=np.zeros(5), nsteps=50000, scale=np.ones(5)*0.1)

    print(sampler.summary())
    flat = sampler.flatchain          # burn-in removed, bad chains discarded
    logp = sampler.flatlog_prob
"""

import numpy as np
from numba import njit


# ---------------------------------------------------------------------------
#  JIT-compiled diagnostic functions
# ---------------------------------------------------------------------------

@njit(cache=True)
def _gelman_rubin(chains):
    """
    Gelman-Rubin statistic (Rhat) and independent draws (Tz)
    following Ford 2006, equations 21-26.

    Parameters
    ----------
    chains : ndarray, shape (nsteps, nchains, ndim)

    Returns
    -------
    Rhat : ndarray (ndim,)   — eq 25: sqrt(V̂⁺ / W)
    Tz   : ndarray (ndim,)   — eq 26: m*n * min(V̂⁺ / B, 1)
    """
    nsteps, nchains, ndim = chains.shape
    Rhat = np.empty(ndim)
    Tz = np.empty(ndim)

    for d in range(ndim):
        # eq 20-21: per-chain mean and W(z)
        chain_means = np.empty(nchains)
        chain_vars = np.empty(nchains)
        for c in range(nchains):
            s = 0.0
            for t in range(nsteps):
                s += chains[t, c, d]
            mu = s / nsteps
            chain_means[c] = mu
            v = 0.0
            for t in range(nsteps):
                diff = chains[t, c, d] - mu
                v += diff * diff
            chain_vars[c] = v / (nsteps - 1)

        # W = mean within-chain variance (eq 21)
        W = 0.0
        for c in range(nchains):
            W += chain_vars[c]
        W /= nchains

        # grand mean (eq 22)
        grand = 0.0
        for c in range(nchains):
            grand += chain_means[c]
        grand /= nchains

        # variance of chain means
        var_of_means = 0.0
        for c in range(nchains):
            diff = chain_means[c] - grand
            var_of_means += diff * diff
        var_of_means /= (nchains - 1)

        # B(z) = n * var(chain_means)  (eq 23)
        B = nsteps * var_of_means

        # V̂⁺(z) = (n-1)/n * W + var_of_means  (eq 24)
        Vplus = (nsteps - 1.0) / nsteps * W + var_of_means

        # eq 25: Rhat
        Rhat[d] = np.sqrt(Vplus / W) if W > 0 else np.inf

        # eq 26: Tz = m * n * min(V̂⁺ / B, 1)
        if B > 0:
            ratio = Vplus / B
            if ratio > 1.0:
                ratio = 1.0
            Tz[d] = nchains * nsteps * ratio
        else:
            Tz[d] = 0.0

    return Rhat, Tz


@njit(cache=True)
def _find_burnin(neg2logp):
    """
    Burn-in index: the latest step at which any chain first crosses
    below the median -2*logpost.

    Parameters
    ----------
    neg2logp : ndarray (nsteps, nchains)

    Returns
    -------
    burnin : int
    """
    nsteps, nchains = neg2logp.shape

    # median via sort
    flat = neg2logp.ravel().copy()
    flat.sort()
    n = len(flat)
    median_val = (flat[n // 2 - 1] + flat[n // 2]) / 2.0 if n % 2 == 0 else flat[n // 2]

    burn = 0
    for c in range(nchains):
        for t in range(nsteps):
            if neg2logp[t, c] <= median_val:
                if t > burn:
                    burn = t
                break
    return burn


@njit(cache=True)
def _identify_good_chains(neg2logp):
    """
    Discard chains stuck in local minima.
    A chain is "bad" if its median -2*logpost exceeds
    (overall median of chain medians) + 5 * 1.4826 * MAD.

    Parameters
    ----------
    neg2logp : ndarray (nsteps, nchains)

    Returns
    -------
    good : int64 array of good chain indices
    """
    nsteps, nchains = neg2logp.shape

    medians = np.empty(nchains)
    for c in range(nchains):
        col = neg2logp[:, c].copy()
        col.sort()
        n = len(col)
        medians[c] = (col[n // 2 - 1] + col[n // 2]) / 2.0 if n % 2 == 0 else col[n // 2]

    sm = medians.copy()
    sm.sort()
    n = len(sm)
    overall = (sm[n // 2 - 1] + sm[n // 2]) / 2.0 if n % 2 == 0 else sm[n // 2]

    devs = np.abs(medians - overall)
    ds = devs.copy()
    ds.sort()
    mad = (ds[n // 2 - 1] + ds[n // 2]) / 2.0 if n % 2 == 0 else ds[n // 2]

    threshold = overall + 5.0 * 1.4826 * mad

    good = []
    for c in range(nchains):
        if medians[c] <= threshold:
            good.append(c)
    return np.array(good, dtype=np.int64)


# ---------------------------------------------------------------------------
#  Sampler
# ---------------------------------------------------------------------------

class DEMCPTSampler:
    """
    Differential Evolution MCMC with optional Parallel Tempering.

    Parameters
    ----------
    log_posterior : callable
        Function  theta(ndim,) -> float  returning log-posterior.
    ndim : int
    nchains : int, optional
        Default max(2*ndim, 10).
    ntemps : int, optional
        Number of temperature rungs. 1 = no tempering.
    Tf : float, optional
        Temperature factor for the hottest rung (default 200).
    stretch : bool, optional
        Use stretch move instead of DE (default False).
    maxgr : float, optional
        Convergence threshold for Gelman-Rubin (default 1.01).
    mintz : float, optional
        Convergence threshold for independent draws (default 1000).
    seed : int or None
    """

    def __init__(self, log_posterior, ndim, nchains=None, ntemps=1, Tf=200.0,
                 stretch=False, maxgr=1.01, mintz=1000, seed=None):
        self.logpost_func = log_posterior
        self.ndim = ndim
        self.nchains = nchains or max(2 * ndim, 10)
        self.ntemps = ntemps
        self.stretch = stretch
        self.maxgr = maxgr
        self.mintz = mintz
        self.rng = np.random.default_rng(seed)

        # temperature ladder  (betas[0]=1 cold, betas[-1]=1/Tf hot)
        if ntemps > 1:
            self.betas = (1.0 / Tf) ** (np.arange(ntemps) / (ntemps - 1))
        else:
            self.betas = np.array([1.0])

        self.gamma = 2.38 / np.sqrt(2.0 * ndim)
        self.a_stretch = 2.0

        # results (populated by run)
        self._chain = None
        self._log_prob = None

    # ----- proposal helpers --------------------------------------------------

    def _de_proposal(self, j, m, current_pos, scale):
        """Differential-Evolution proposal for walker j at temperature m."""
        nc = self.nchains
        ndim = self.ndim
        rng = self.rng

        # pick two distinct chains != j
        pool = np.delete(np.arange(nc), j)
        r1, r2 = rng.choice(pool, 2, replace=False)

        jitter = (rng.random(ndim) - 0.5) * scale / 10.0
        proposal = (current_pos[j, m]
                     + self.gamma * (current_pos[r1, m] - current_pos[r2, m] + jitter))
        return proposal, 0.0  # log_fac = 0

    def _stretch_proposal(self, j, m, current_pos):
        """Affine-invariant stretch-move proposal."""
        nc = self.nchains
        rng = self.rng
        a = self.a_stretch

        r1 = rng.integers(0, nc - 1)
        if r1 >= j:
            r1 += 1

        z = ((a - 1.0) * rng.random() + 1.0) ** 2 / a
        proposal = current_pos[r1, m] + z * (current_pos[j, m] - current_pos[r1, m])
        log_fac = (self.ndim - 1) * np.log(z)
        return proposal, log_fac

    # ----- main loop ---------------------------------------------------------

    def run(self, p0, nsteps, nthin=1, scale=None, progress=True,
            check_every=None, npass_required=6):
        """
        Run the sampler.

        Parameters
        ----------
        p0 : ndarray (ndim,)
            Best-fit starting point.
        nsteps : int
            Number of stored steps per chain.
        nthin : int
            Keep every nthin-th sample (default 1).
        scale : ndarray (ndim,) or None
            Per-parameter step scale.  If None, uses 1% of |p0|.
        progress : bool
            Print progress bar.
        check_every : int or None
            Steps between convergence checks.  Default nsteps//20.
        npass_required : int
            Consecutive passes needed (default 6).

        Returns
        -------
        converged : bool
        """
        ndim = self.ndim
        nchains = self.nchains
        ntemps = self.ntemps
        betas = self.betas
        rng = self.rng
        logpost = self.logpost_func

        p0 = np.asarray(p0, dtype=np.float64)
        if scale is None:
            scale = np.abs(p0) * 0.01
            scale[scale == 0] = 1e-5

        if check_every is None:
            check_every = max(100, nsteps // 20)

        # ---- initialise walkers (nchains, ntemps, ndim) ---------------------
        pos = np.empty((nchains, ntemps, ndim))
        logp = np.full((nchains, ntemps), -np.inf)

        if progress:
            print(f"Initialising {nchains} chains x {ntemps} temps ...")

        for j in range(nchains):
            for m in range(ntemps):
                niter = 0
                while True:
                    if j == 0:
                        trial = p0.copy()
                    else:
                        factor = min(np.sqrt(500.0 / ndim), 3.0)
                        trial = p0 + (factor / np.exp(niter / 1000.0)
                                      * scale * rng.standard_normal(ndim))
                    lp = logpost(trial)
                    if np.isfinite(lp):
                        pos[j, m] = trial
                        logp[j, m] = lp
                        break
                    niter += 1
                    if niter > 10000:
                        raise RuntimeError(
                            f"Cannot find finite logpost near p0 "
                            f"(chain {j}, temp {m})")

        # ---- storage (cold chain only) -------------------------------------
        chain = np.empty((nsteps, nchains, ndim))
        log_prob = np.empty((nsteps, nchains))
        chain[0] = pos[:, 0]
        log_prob[0] = logp[:, 0]

        naccept = 0
        nattempt = 0
        nswap = 0
        nswap_attempt = 0
        npass = 0
        converged = False
        final_step = nsteps

        if progress:
            print("Running MCMC ...")

        # ---- MCMC loop ------------------------------------------------------
        for i in range(1, nsteps):
            for _thin in range(nthin):
                for j in range(nchains):
                    for m in range(ntemps):

                        # --- parallel-tempering swap attempt -----------------
                        if m < ntemps - 1 and rng.random() < 0.5:
                            nswap_attempt += 1
                            log_alpha = ((betas[m] - betas[m + 1])
                                         * (logp[j, m + 1] - logp[j, m]))
                            if np.log(rng.random()) < log_alpha:
                                nswap += 1
                                pos[j, m], pos[j, m + 1] = (
                                    pos[j, m + 1].copy(), pos[j, m].copy())
                                logp[j, m], logp[j, m + 1] = (
                                    logp[j, m + 1], logp[j, m])
                        else:
                            # --- DE or stretch proposal ----------------------
                            nattempt += 1
                            if self.stretch:
                                proposal, log_fac = self._stretch_proposal(
                                    j, m, pos)
                            else:
                                proposal, log_fac = self._de_proposal(
                                    j, m, pos, scale)

                            new_lp = logpost(proposal)

                            if np.isfinite(new_lp):
                                log_alpha = (betas[m] * (new_lp - logp[j, m])
                                             + log_fac)
                                if np.log(rng.random()) < log_alpha:
                                    naccept += 1
                                    pos[j, m] = proposal
                                    logp[j, m] = new_lp

                # store cold chain
                chain[i] = pos[:, 0]
                log_prob[i] = logp[:, 0]

            # --- convergence check -------------------------------------------
            if (i + 1) % check_every == 0 and i > 2 * nchains:
                chi2 = -2.0 * log_prob[:i + 1]
                burnndx = _find_burnin(chi2)
                post_burn = chain[burnndx:i + 1]

                if post_burn.shape[0] > 10:
                    Rhat, Tz = _gelman_rubin(post_burn)
                    max_gr = float(np.max(Rhat))
                    min_tz = float(np.min(Tz))

                    acc = naccept / max(nattempt, 1) * 100
                    swap_s = ""
                    if ntemps > 1 and nswap_attempt > 0:
                        swap_s = f"; swap={nswap / nswap_attempt * 100:.1f}%"

                    if progress:
                        print(
                            f"\r  {100 * (i + 1) / nsteps:5.1f}% | "
                            f"accept={acc:.1f}%{swap_s} | "
                            f"GR={max_gr:.4f} (<{self.maxgr}) | "
                            f"Tz={min_tz:.0f} (>{self.mintz})   ",
                            end="", flush=True)

                    if max_gr < self.maxgr and min_tz > self.mintz:
                        npass += 1
                        if npass >= npass_required:
                            converged = True
                            final_step = i + 1
                            break
                    else:
                        npass = 0

        # trim
        chain = chain[:final_step]
        log_prob = log_prob[:final_step]

        if progress:
            if converged:
                print(f"\n  Converged at step {final_step}/{nsteps}.")
            else:
                print(f"\n  Reached max steps ({nsteps}). NOT converged.")

        self._chain = chain
        self._log_prob = log_prob
        return converged

    # ----- properties --------------------------------------------------------

    @property
    def chain(self):
        """Raw chain array (nsteps, nchains, ndim)."""
        return self._chain

    @property
    def log_prob(self):
        """Log-posterior array (nsteps, nchains)."""
        return self._log_prob

    @property
    def flatchain(self):
        """Flat chain with burn-in removed and bad chains discarded."""
        if self._chain is None:
            return None
        chi2 = -2.0 * self._log_prob
        burnndx = _find_burnin(chi2)
        good = _identify_good_chains(chi2[burnndx:])
        return self._chain[burnndx:, good].reshape(-1, self.ndim)

    @property
    def flatlog_prob(self):
        """Flat log-posterior with burn-in removed and bad chains discarded."""
        if self._log_prob is None:
            return None
        chi2 = -2.0 * self._log_prob
        burnndx = _find_burnin(chi2)
        good = _identify_good_chains(chi2[burnndx:])
        return self._log_prob[burnndx:, good].ravel()

    # ----- diagnostics -------------------------------------------------------

    def summary(self, param_names=None):
        """
        Print convergence summary.

        Parameters
        ----------
        param_names : list of str, optional
            Names for each parameter dimension.

        Returns
        -------
        dict with 'Rhat', 'Tz', 'burnin', 'good_chains', 'n_bad'
        """
        if self._chain is None:
            print("No chain. Call run() first.")
            return None

        chi2 = -2.0 * self._log_prob
        burnndx = _find_burnin(chi2)
        good = _identify_good_chains(chi2[burnndx:])
        n_bad = self.nchains - len(good)

        post_burn = self._chain[burnndx:, good]
        Rhat, Tz = _gelman_rubin(post_burn)

        nsteps = self._chain.shape[0]
        print(f"Steps: {nsteps}  Burn-in: {burnndx}  "
              f"Good chains: {len(good)}/{self.nchains}")
        if n_bad:
            print(f"  WARNING: {n_bad} bad chain(s) discarded")
        print()
        print(f"{'#':>4s}  {'Parameter':<16s} {'Rhat':>8s} {'Tz':>10s}  Status")
        print("-" * 52)
        for d in range(self.ndim):
            name = param_names[d] if param_names and d < len(param_names) else str(d)
            ok = Rhat[d] < self.maxgr and Tz[d] > self.mintz
            mark = "OK" if ok else "**BAD**"
            print(f"{d:4d}  {name:<16s} {Rhat[d]:8.4f} {Tz[d]:10.1f}  {mark}")

        return {
            "Rhat": Rhat,
            "Tz": Tz,
            "burnin": burnndx,
            "good_chains": good,
            "n_bad": n_bad,
        }
