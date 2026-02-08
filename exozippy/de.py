"""Simple differential evolution optimizer inspired by EXOFAST's exofast_de."""

from __future__ import annotations

import numpy as np


def differential_evolution(
    func,
    bounds,
    x0=None,
    pop_size=40,
    max_gen=500,
    tol=1e-3,
    verbose=False,
    seed=None,
):
    """
    Parameters
    ----------
    func : callable
        Objective function.
    bounds : array-like, shape (ndim, 2)
        Lower/upper bounds for each parameter.
    x0 : array-like, optional
        Initial guess to inject into the population.
    pop_size : int
        Number of individuals in the population.
    max_gen : int
        Number of generations.
    tol : float
        Convergence threshold on (max-min) fitness.
    verbose : bool
        Print progress information.
    seed : int, optional
        Random seed.
    """
    bounds = np.asarray(bounds, dtype=float)
    ndim = bounds.shape[0]
    rng = np.random.default_rng(seed)

    pop = rng.uniform(bounds[:, 0], bounds[:, 1], size=(pop_size, ndim))
    if x0 is not None:
        pop[0] = np.clip(np.asarray(x0, dtype=float), bounds[:, 0], bounds[:, 1])

    fitness = np.array([func(ind) for ind in pop])

    for gen in range(max_gen):
        for i in range(pop_size):
            idxs = np.delete(np.arange(pop_size), i)
            r1, r2, r3 = rng.choice(idxs, 3, replace=False)
            F = rng.uniform(0.25, 0.75)
            mutant = pop[r1] + F * (pop[r2] - pop[r3])
            mutant = np.clip(mutant, bounds[:, 0], bounds[:, 1])

            CR = rng.uniform(0.25, 1.0)
            cross_points = rng.random(ndim) < CR
            if not np.any(cross_points):
                cross_points[rng.integers(0, ndim)] = True
            trial = np.where(cross_points, mutant, pop[i])
            trial = np.clip(trial, bounds[:, 0], bounds[:, 1])

            fval = func(trial)
            if fval < fitness[i]:
                pop[i] = trial
                fitness[i] = fval

        spread = np.max(fitness) - np.min(fitness)
        if verbose:
            print(f"DE: generation {gen+1}, Δchi2={spread:.3e}, best={fitness.min():.3f}")
        if spread < tol:
            break

    best_idx = np.argmin(fitness)
    return pop[best_idx], fitness[best_idx], {"population": pop, "fitness": fitness}
