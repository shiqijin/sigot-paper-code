from __future__ import annotations

from math import sqrt

from scipy.special import ndtr
from typing import Iterator

import numpy as np

from .configs import BrownianReserveConfig


Array = np.ndarray


def Phi(x: float | Array) -> float | Array:
    return ndtr(np.asarray(x, dtype=float))


def pareto_moments(alpha: float = 2.2) -> tuple[float, float]:
    m1 = alpha / (alpha - 1.0)
    m2 = alpha / (alpha - 2.0)
    return float(m1), float(m2)


def psi_brownian_closed_form(u: float | Array, cfg: BrownianReserveConfig) -> float | Array:
    """Closed-form Brownian approximation to the finite-horizon ruin probability.

    This matches the formula used in the student's notebook and BM19 Example 1.
    """

    m1, m2 = pareto_moments(cfg.alpha)
    sigma = sqrt(cfg.nu * m2)
    mu = -cfg.eta * cfg.nu * m1
    T = cfg.T
    u = np.asarray(u, dtype=float)
    a = (u - mu * T) / (sigma * sqrt(T))
    b = (-u - mu * T) / (sigma * sqrt(T))
    return (1.0 - Phi(a)) + np.exp(2.0 * mu * u / (sigma**2)) * Phi(b)


def reserve_drift_and_diffusion(cfg: BrownianReserveConfig) -> tuple[float, float]:
    m1, m2 = pareto_moments(cfg.alpha)
    drift = cfg.eta * cfg.nu * m1
    diffusion = -sqrt(cfg.nu * m2)
    return float(drift), float(diffusion)


def simulate_brownian_reserve_paths(
    cfg: BrownianReserveConfig,
    initial_capital: float,
    batch_size: int | None = None,
    seed: int | None = None,
) -> Iterator[tuple[Array, Array]]:
    """Yield reserve paths in batches.

    Returns tuples (times, paths), where times has shape (steps + 1,) and paths has shape (batch, steps + 1).
    The generator is deterministic for a fixed seed.
    """

    batch_size = cfg.batch_size if batch_size is None else int(batch_size)
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    dt = cfg.T / cfg.steps
    drift, diffusion = reserve_drift_and_diffusion(cfg)
    times = np.linspace(0.0, cfg.T, cfg.steps + 1, dtype=float)

    remaining = cfg.paths
    while remaining > 0:
        current = min(batch_size, remaining)
        dB = rng.normal(0.0, np.sqrt(dt), size=(current, cfg.steps))
        increments = drift * dt + diffusion * dB
        paths = np.empty((current, cfg.steps + 1), dtype=float)
        paths[:, 0] = initial_capital
        paths[:, 1:] = initial_capital + np.cumsum(increments, axis=1)
        yield times, paths
        remaining -= current


from .configs import CramerLundbergConfig


def _sample_pareto_type1(rng: np.random.Generator, alpha: float, size: int) -> Array:
    u = rng.random(size=size)
    return (1.0 - u) ** (-1.0 / alpha)


def simulate_cramer_lundberg_paths(
    cfg: CramerLundbergConfig,
    initial_capital: float,
    batch_size: int | None = None,
    seed: int | None = None,
) -> Iterator[tuple[Array, Array]]:
    """Yield batched compound Poisson reserve paths on a regular grid.

    Reserve model: R_t = u + c t - sum_{i<=N_t} Y_i, with
    c = (1 + eta) * nu * E[Y] and Y following a Pareto Type-I(alpha) law with xm=1.
    Paths are returned on a uniform grid of length ``steps + 1``.
    """

    batch_size = cfg.batch_size if batch_size is None else int(batch_size)
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    times = np.linspace(0.0, cfg.T, cfg.steps + 1, dtype=float)
    m1, _ = pareto_moments(cfg.alpha)
    premium_rate = (1.0 + cfg.eta) * cfg.nu * m1

    remaining = cfg.paths
    while remaining > 0:
        current = min(batch_size, remaining)
        paths = np.empty((current, cfg.steps + 1), dtype=float)
        base_line = initial_capital + premium_rate * times
        paths[:] = base_line[None, :]

        claim_counts = rng.poisson(cfg.nu * cfg.T, size=current)
        for i, n_claims in enumerate(claim_counts):
            if n_claims <= 0:
                continue
            arrival_times = np.sort(rng.uniform(0.0, cfg.T, size=int(n_claims)))
            claim_sizes = _sample_pareto_type1(rng=rng, alpha=cfg.alpha, size=int(n_claims))
            grid_idx = np.searchsorted(times, arrival_times, side='left')
            jump_vector = np.zeros(cfg.steps + 1, dtype=float)
            np.add.at(jump_vector, grid_idx, claim_sizes)
            cumulative_claims = np.cumsum(jump_vector)
            paths[i] -= cumulative_claims
        yield times, paths
        remaining -= current
