from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .blocks import coordinate_block_spec
from .discrepancy import joint_ot_budget_vector

Array = np.ndarray


@dataclass(frozen=True)
class LogPathModel:
    """Specification for a one-dimensional log-price path model."""

    name: str
    kind: str
    mu: float = 0.05
    sigma: float = 0.20
    # Merton jump parameters.
    jump_intensity: float = 0.0
    jump_mean: float = -0.12
    jump_sd: float = 0.08
    # Mixture parameters.
    mixture_prob_crash: float = 0.20
    normal_mu: float = 0.05
    normal_sigma: float = 0.20
    crash_mu: float = -0.30
    crash_sigma: float = 0.45
    # Heston stochastic-volatility parameters.
    v0: float = 0.04
    theta_v: float = 0.04
    kappa_v: float = 3.0
    xi_v: float = 0.35
    rho: float = -0.5
    # CEV / local-volatility parameter.
    beta: float = 0.75
    # Numerical floor for positive price / variance schemes.
    floor: float = 1e-10


def _validate_common(paths: int, steps: int, T: float) -> tuple[int, int, float]:
    paths = int(paths)
    steps = int(steps)
    T = float(T)
    if paths <= 0:
        raise ValueError("paths must be positive.")
    if steps <= 0:
        raise ValueError("steps must be positive.")
    if T <= 0.0:
        raise ValueError("T must be positive.")
    return paths, steps, T


def simulate_gbm_log_paths(
    paths: int,
    steps: int,
    T: float,
    mu: float,
    sigma: float,
    seed: int = 0,
) -> tuple[Array, Array]:
    """Simulate log-price paths under GBM.

    X_t = log(S_t/S_0), dX_t = (mu - 0.5 sigma^2) dt + sigma dW_t.
    Returns (times, paths) with paths shape (paths, steps + 1) and X_0 = 0.
    """

    paths, steps, T = _validate_common(paths, steps, T)
    rng = np.random.default_rng(int(seed))
    dt = T / steps
    times = np.linspace(0.0, T, steps + 1, dtype=float)
    drift = (float(mu) - 0.5 * float(sigma) ** 2) * dt
    vol = float(sigma) * np.sqrt(dt)
    increments = drift + vol * rng.normal(size=(paths, steps))
    out = np.empty((paths, steps + 1), dtype=float)
    out[:, 0] = 0.0
    out[:, 1:] = np.cumsum(increments, axis=1)
    return times, out


def simulate_merton_log_paths(
    paths: int,
    steps: int,
    T: float,
    mu: float,
    sigma: float,
    jump_intensity: float,
    jump_mean: float,
    jump_sd: float,
    seed: int = 0,
) -> tuple[Array, Array]:
    """Simulate compensated Merton jump-diffusion log paths.

    The compensation uses kappa_J = E[e^J - 1] so that the asset drift is mu.
    The log increment is
        (mu - 0.5 sigma^2 - lambda*kappa_J) dt + sigma dW + sum_{i=1}^{N_dt} J_i.
    """

    paths, steps, T = _validate_common(paths, steps, T)
    rng = np.random.default_rng(int(seed))
    dt = T / steps
    times = np.linspace(0.0, T, steps + 1, dtype=float)
    lam = float(jump_intensity)
    jm = float(jump_mean)
    js = float(jump_sd)
    kappa_j = float(np.exp(jm + 0.5 * js * js) - 1.0)
    drift = (float(mu) - 0.5 * float(sigma) ** 2 - lam * kappa_j) * dt
    diffusion = float(sigma) * np.sqrt(dt) * rng.normal(size=(paths, steps))
    counts = rng.poisson(lam * dt, size=(paths, steps))
    jump_noise = rng.normal(size=(paths, steps))
    jump_sums = counts * jm + np.sqrt(np.maximum(counts, 0.0)) * js * jump_noise
    increments = drift + diffusion + jump_sums
    out = np.empty((paths, steps + 1), dtype=float)
    out[:, 0] = 0.0
    out[:, 1:] = np.cumsum(increments, axis=1)
    return times, out


def simulate_mixture_gbm_log_paths(
    paths: int,
    steps: int,
    T: float,
    prob_crash: float,
    normal_mu: float,
    normal_sigma: float,
    crash_mu: float,
    crash_sigma: float,
    seed: int = 0,
) -> tuple[Array, Array]:
    """Simulate a pathwise mixture of normal-regime and crash-regime GBMs."""

    paths, steps, T = _validate_common(paths, steps, T)
    rng = np.random.default_rng(int(seed))
    times = np.linspace(0.0, T, steps + 1, dtype=float)
    crash = rng.random(paths) < float(prob_crash)
    mu = np.where(crash, float(crash_mu), float(normal_mu))
    sigma = np.where(crash, float(crash_sigma), float(normal_sigma))
    dt = T / steps
    z = rng.normal(size=(paths, steps))
    increments = (mu[:, None] - 0.5 * sigma[:, None] ** 2) * dt + sigma[:, None] * np.sqrt(dt) * z
    out = np.empty((paths, steps + 1), dtype=float)
    out[:, 0] = 0.0
    out[:, 1:] = np.cumsum(increments, axis=1)
    return times, out


def simulate_heston_log_paths(
    paths: int,
    steps: int,
    T: float,
    mu: float,
    v0: float,
    theta_v: float,
    kappa_v: float,
    xi_v: float,
    rho: float,
    seed: int = 0,
) -> tuple[Array, Array]:
    """Simulate log-price paths under a Heston stochastic-volatility model.

    Uses full-truncation Euler for the variance process and outputs
    X_t = log(S_t/S_0), with X_0 = 0. The drift parameter ``mu`` is the
    physical/asset drift used consistently with the GBM examples.
    """

    paths, steps, T = _validate_common(paths, steps, T)
    rng = np.random.default_rng(int(seed))
    dt = T / steps
    sqrt_dt = np.sqrt(dt)
    times = np.linspace(0.0, T, steps + 1, dtype=float)
    out = np.empty((paths, steps + 1), dtype=float)
    out[:, 0] = 0.0
    v = np.full(paths, float(v0), dtype=float)
    rho = float(rho)
    if not (-1.0 <= rho <= 1.0):
        raise ValueError("rho must lie in [-1, 1].")
    rho_perp = float(np.sqrt(max(1.0 - rho * rho, 0.0)))
    for t in range(steps):
        z_s = rng.normal(size=paths)
        z_ind = rng.normal(size=paths)
        z_v = rho * z_s + rho_perp * z_ind
        v_pos = np.maximum(v, 0.0)
        sqrt_v = np.sqrt(v_pos)
        out[:, t + 1] = out[:, t] + (float(mu) - 0.5 * v_pos) * dt + sqrt_v * sqrt_dt * z_s
        v = v + float(kappa_v) * (float(theta_v) - v_pos) * dt + float(xi_v) * sqrt_v * sqrt_dt * z_v
        v = np.maximum(v, 0.0)
    return times, out


def simulate_cev_log_paths(
    paths: int,
    steps: int,
    T: float,
    mu: float,
    sigma: float,
    beta: float,
    seed: int = 0,
    floor: float = 1e-10,
) -> tuple[Array, Array]:
    """Simulate log-price paths under a CEV/local-volatility model.

    Price dynamics: dS_t = mu S_t dt + sigma S_t^beta dW_t, with S_0=1.
    The output is X_t = log(S_t/S_0). A small positive floor prevents
    numerical crossing of zero under Euler discretization.
    """

    paths, steps, T = _validate_common(paths, steps, T)
    rng = np.random.default_rng(int(seed))
    dt = T / steps
    sqrt_dt = np.sqrt(dt)
    times = np.linspace(0.0, T, steps + 1, dtype=float)
    S = np.ones(paths, dtype=float)
    out = np.empty((paths, steps + 1), dtype=float)
    out[:, 0] = 0.0
    beta = float(beta)
    floor = float(floor)
    if floor <= 0.0:
        raise ValueError("floor must be positive for CEV simulation.")
    for t in range(steps):
        z = rng.normal(size=paths)
        S_pos = np.maximum(S, floor)
        S = S_pos + float(mu) * S_pos * dt + float(sigma) * (S_pos ** beta) * sqrt_dt * z
        S = np.maximum(S, floor)
        out[:, t + 1] = np.log(S)
    return times, out

def simulate_log_path_model(model: LogPathModel, paths: int, steps: int, T: float, seed: int) -> tuple[Array, Array]:
    kind = model.kind.lower()
    if kind == "gbm":
        return simulate_gbm_log_paths(paths=paths, steps=steps, T=T, mu=model.mu, sigma=model.sigma, seed=seed)
    if kind == "merton":
        return simulate_merton_log_paths(
            paths=paths,
            steps=steps,
            T=T,
            mu=model.mu,
            sigma=model.sigma,
            jump_intensity=model.jump_intensity,
            jump_mean=model.jump_mean,
            jump_sd=model.jump_sd,
            seed=seed,
        )
    if kind in {"mixture", "mixture_gbm", "crisis_mixture"}:
        return simulate_mixture_gbm_log_paths(
            paths=paths,
            steps=steps,
            T=T,
            prob_crash=model.mixture_prob_crash,
            normal_mu=model.normal_mu,
            normal_sigma=model.normal_sigma,
            crash_mu=model.crash_mu,
            crash_sigma=model.crash_sigma,
            seed=seed,
        )
    if kind == "heston":
        return simulate_heston_log_paths(
            paths=paths,
            steps=steps,
            T=T,
            mu=model.mu,
            v0=model.v0,
            theta_v=model.theta_v,
            kappa_v=model.kappa_v,
            xi_v=model.xi_v,
            rho=model.rho,
            seed=seed,
        )
    if kind in {"cev", "localvol", "local_vol"}:
        return simulate_cev_log_paths(
            paths=paths,
            steps=steps,
            T=T,
            mu=model.mu,
            sigma=model.sigma,
            beta=model.beta,
            seed=seed,
            floor=model.floor,
        )
    raise ValueError(f"Unsupported model kind {model.kind!r}.")


def simulate_log_path_model_batches(
    model: LogPathModel,
    *,
    paths: int,
    steps: int,
    T: float,
    seed: int,
    batch_size: int = 256,
) -> Iterable[tuple[Array, Array]]:
    """Yield reproducible path batches without materializing the full sample.

    Batch-level child seeds are deterministically spawned from ``seed``.  This keeps
    the streaming runner reproducible while allowing large Monte Carlo jobs to avoid
    holding the full ``paths x (steps+1)`` matrix in memory at once.
    """

    paths, steps, T = _validate_common(paths, steps, T)
    batch_size = int(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    n_batches = (paths + batch_size - 1) // batch_size
    children = np.random.SeedSequence(int(seed)).spawn(n_batches)
    remaining = int(paths)
    for child in children:
        n_batch = min(batch_size, remaining)
        child_seed = int(child.generate_state(1, dtype=np.uint64)[0])
        yield simulate_log_path_model(model, paths=n_batch, steps=steps, T=T, seed=child_seed)
        remaining -= n_batch
    if remaining != 0:  # pragma: no cover - invariant check
        raise RuntimeError("streaming path generator did not exhaust the requested path count.")


def default_stress_models() -> list[LogPathModel]:
    return [
        LogPathModel(name="gbm_negative_drift", kind="gbm", mu=-0.10, sigma=0.20),
        LogPathModel(name="gbm_high_vol", kind="gbm", mu=0.05, sigma=0.40),
        LogPathModel(
            name="merton_negative_jump",
            kind="merton",
            mu=0.05,
            sigma=0.18,
            jump_intensity=1.0,
            jump_mean=-0.12,
            jump_sd=0.08,
        ),
        LogPathModel(
            name="crisis_mixture",
            kind="mixture",
            mixture_prob_crash=0.10,
            normal_mu=0.05,
            normal_sigma=0.20,
            crash_mu=-0.20,
            crash_sigma=0.40,
        ),
    ]


def parse_theta_spec(theta_spec: str) -> dict[str, float]:
    """Parse a coordinate coefficient string such as 'X:-0.35,tX:0.15,Xt:-0.45,XX:0.20'."""

    out: dict[str, float] = {}
    for token in str(theta_spec).split(","):
        token = token.strip()
        if not token:
            continue
        if ":" not in token:
            raise ValueError(f"Invalid theta token {token!r}; expected name:value.")
        name, value = token.split(":", 1)
        name = name.strip()
        if not name:
            raise ValueError("Empty coordinate name in theta spec.")
        out[name] = float(value)
    if not out:
        raise ValueError("theta spec produced no active coordinates.")
    return out


def _empirical_w1_1d_equal_weights(x: Array, y: Array) -> float:
    """Exact W1 distance between two 1D empirical laws with equal weights within each sample."""

    a = np.sort(np.asarray(x, dtype=float).reshape(-1))
    b = np.sort(np.asarray(y, dtype=float).reshape(-1))
    n = a.size
    m = b.size
    if n == 0 or m == 0:
        raise ValueError("W1 samples must be nonempty.")
    if n == m:
        return float(np.mean(np.abs(a - b)))

    i = j = 0
    wa = 1.0 / n
    wb = 1.0 / m
    rem_a = wa
    rem_b = wb
    cost = 0.0
    # Transport quantile mass from sorted a to sorted b.
    while i < n and j < m:
        mass = rem_a if rem_a < rem_b else rem_b
        cost += mass * abs(a[i] - b[j])
        rem_a -= mass
        rem_b -= mass
        if rem_a <= 1e-15:
            i += 1
            if i < n:
                rem_a = wa
        if rem_b <= 1e-15:
            j += 1
            if j < m:
                rem_b = wb
    return float(cost)


def decoupled_coordinate_w1_budget(baseline_active: Array, stress_active: Array, theta: Array) -> tuple[Array, Array, float]:
    """Coordinatewise marginal W1 diagnostic for an affine payoff.

    This is intentionally decoupled: each active coordinate uses its own optimal 1D
    Wasserstein coupling. It is a marginal diagnostic relaxation, not a common-coupling
    pathwise budget.
    """

    Z0 = np.asarray(baseline_active, dtype=float)
    Zq = np.asarray(stress_active, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if Z0.ndim != 2 or Zq.ndim != 2:
        raise ValueError("active features must be 2D.")
    if Z0.shape[1] != theta.size or Zq.shape[1] != theta.size:
        raise ValueError("theta dimension inconsistent with active features.")
    deltas = np.array([_empirical_w1_1d_equal_weights(Z0[:, j], Zq[:, j]) for j in range(theta.size)], dtype=float)
    contributions = np.abs(theta) * deltas
    return deltas, contributions, float(np.sum(contributions))


@dataclass(frozen=True)
class AffinePayoffResult:
    stress_model: str
    baseline_mean: float
    stress_mean: float
    mean_shift: float
    baseline_sd: float
    effect_size: float
    kappa_joint: float
    kappa_decoupled: float
    joint_over_decoupled: float
    upper_bound: float
    lower_bound: float
    covered_upper: bool
    covered_lower: bool
    covered_abs: bool
    signed_tightness_ratio: float
    abs_tightness_ratio: float
    upper_bound_decoupled: float
    lower_bound_decoupled: float
    covered_upper_decoupled: bool
    covered_lower_decoupled: bool
    covered_abs_decoupled: bool
    signed_tightness_ratio_decoupled: float
    abs_tightness_ratio_decoupled: float
    ot_objective: float
    paths: int
    steps: int
    T: float
    signature_level: int
    scaling_method: str
    signature_backend: str
    signature_type: str


def evaluate_affine_payoff_against_stress(
    baseline_active: Array,
    stress_active: Array,
    theta: Array,
    stress_name: str,
    ot_batch_size: int = 512,
    ot_batches: int = 8,
    ot_epsilon: float = 1e-2,
    ot_max_iter: int = 400,
    ot_tol: float = 1e-9,
    ot_seed: int = 0,
    ot_mode: str = "decoupled",
) -> tuple[AffinePayoffResult, Array, Array, Array, Array]:
    """Compute mean shifts and affine-score robust premiums.

    In ``decoupled`` mode, each active coordinate uses its own empirical
    one-dimensional Wasserstein coupling. In ``subsampled`` mode, the joint
    coupling is estimated with subsampled Sinkhorn solves.
    """

    Z0 = np.asarray(baseline_active, dtype=float)
    Zq = np.asarray(stress_active, dtype=float)
    theta = np.asarray(theta, dtype=float).reshape(-1)
    if Z0.ndim != 2 or Zq.ndim != 2:
        raise ValueError("active features must be 2D.")
    if Z0.shape[1] != theta.size or Zq.shape[1] != theta.size:
        raise ValueError("theta dimension inconsistent with active features.")
    if ot_mode not in {"decoupled", "subsampled"}:
        raise ValueError("ot_mode must be 'decoupled' or 'subsampled'.")

    L0 = Z0 @ theta
    Lq = Zq @ theta
    baseline_mean = float(np.mean(L0))
    stress_mean = float(np.mean(Lq))
    mean_shift = float(stress_mean - baseline_mean)
    baseline_sd = float(np.std(L0, ddof=1)) if L0.size > 1 else 0.0
    effect_size = float(mean_shift / baseline_sd) if baseline_sd > 0.0 else float("nan")

    deltas_dec, contributions_dec, kappa_dec = decoupled_coordinate_w1_budget(Z0, Zq, theta)

    if ot_mode == "decoupled":
        # The decoupled mode reports coordinatewise budgets only.
        deltas = np.full_like(deltas_dec, np.nan, dtype=float)
        contributions = np.full_like(contributions_dec, np.nan, dtype=float)
        kappa_joint = float("nan")
        ot_objective = float("nan")
    else:
        block_spec = coordinate_block_spec(feature_dim=theta.size, norm="l2")
        lambda_weights = np.abs(theta)
        ot_res = joint_ot_budget_vector(
            baseline_features=Z0,
            true_features=Zq,
            block_spec=block_spec,
            lambda_weights=lambda_weights,
            batch_size=int(ot_batch_size),
            n_batches=int(ot_batches),
            epsilon=float(ot_epsilon),
            max_iter=int(ot_max_iter),
            tol=float(ot_tol),
            seed=int(ot_seed),
            mode=ot_mode,
        )
        deltas = np.asarray(ot_res.delta_blocks, dtype=float)
        contributions = lambda_weights * deltas
        kappa_joint = float(np.sum(contributions))
        ot_objective = float(ot_res.objective_value)

    joint_over_decoupled = float(kappa_joint / kappa_dec) if np.isfinite(kappa_joint) and kappa_dec > 0 else float("nan")
    kappa_primary = kappa_dec if ot_mode == "decoupled" else kappa_joint

    upper_bound = float(baseline_mean + kappa_primary)
    lower_bound = float(baseline_mean - kappa_primary)
    covered_upper = bool(stress_mean <= upper_bound + 1e-12)
    covered_lower = bool(stress_mean >= lower_bound - 1e-12)
    covered_abs = bool(abs(mean_shift) <= kappa_primary + 1e-12)
    signed_tightness = float(mean_shift / kappa_primary) if kappa_primary > 0 else float("nan")
    abs_tightness = float(abs(mean_shift) / kappa_primary) if kappa_primary > 0 else float("nan")

    upper_bound_dec = float(baseline_mean + kappa_dec)
    lower_bound_dec = float(baseline_mean - kappa_dec)
    covered_upper_dec = bool(stress_mean <= upper_bound_dec + 1e-12)
    covered_lower_dec = bool(stress_mean >= lower_bound_dec - 1e-12)
    covered_abs_dec = bool(abs(mean_shift) <= kappa_dec + 1e-12)
    signed_tightness_dec = float(mean_shift / kappa_dec) if kappa_dec > 0 else float("nan")
    abs_tightness_dec = float(abs(mean_shift) / kappa_dec) if kappa_dec > 0 else float("nan")

    result = AffinePayoffResult(
        stress_model=stress_name,
        baseline_mean=baseline_mean,
        stress_mean=stress_mean,
        mean_shift=mean_shift,
        baseline_sd=baseline_sd,
        effect_size=effect_size,
        kappa_joint=kappa_joint,
        kappa_decoupled=kappa_dec,
        joint_over_decoupled=joint_over_decoupled,
        upper_bound=upper_bound,
        lower_bound=lower_bound,
        covered_upper=covered_upper,
        covered_lower=covered_lower,
        covered_abs=covered_abs,
        signed_tightness_ratio=signed_tightness,
        abs_tightness_ratio=abs_tightness,
        upper_bound_decoupled=upper_bound_dec,
        lower_bound_decoupled=lower_bound_dec,
        covered_upper_decoupled=covered_upper_dec,
        covered_lower_decoupled=covered_lower_dec,
        covered_abs_decoupled=covered_abs_dec,
        signed_tightness_ratio_decoupled=signed_tightness_dec,
        abs_tightness_ratio_decoupled=abs_tightness_dec,
        ot_objective=ot_objective,
        paths=0,
        steps=0,
        T=0.0,
        signature_level=0,
        scaling_method="",
        signature_backend="",
        signature_type="",
    )
    return result, deltas, contributions, deltas_dec, contributions_dec

# --- Linear halfspace probability utilities ----------------------------------------------
@dataclass(frozen=True)
class HalfspaceShiftResult:
    threshold: float
    kappa: float
    u_star: float
    robust_probability: float
    degenerate: bool
    mean_shortfall: float
    shifted_threshold: float
    robust_tail_count: int


@dataclass(frozen=True)
class PreparedHalfspaceShift:
    threshold: float
    scores_sorted: Array
    D_sorted: Array
    truncated_moment_grid: Array
    mean_shortfall: float
    baseline_tail_count: int

    @property
    def n(self) -> int:
        return int(self.scores_sorted.size)


def prepare_shifted_threshold_probability(
    baseline_scores: Array,
    threshold: float,
) -> PreparedHalfspaceShift:
    """Pre-sort the empirical shortfall distribution for repeated kappa queries."""

    scores = np.asarray(baseline_scores, dtype=float).reshape(-1)
    if scores.size == 0:
        raise ValueError("baseline_scores must be nonempty.")
    b = float(threshold)
    scores_sorted = np.sort(scores)
    baseline_tail_count = int(scores.size - np.searchsorted(scores_sorted, b, side="left"))
    D_sorted = np.sort(np.maximum(b - scores, 0.0))
    truncated_moment_grid = np.cumsum(D_sorted) / float(D_sorted.size)
    return PreparedHalfspaceShift(
        threshold=b,
        scores_sorted=scores_sorted,
        D_sorted=D_sorted,
        truncated_moment_grid=truncated_moment_grid,
        mean_shortfall=float(truncated_moment_grid[-1]) if truncated_moment_grid.size else 0.0,
        baseline_tail_count=baseline_tail_count,
    )


def shifted_threshold_probability_from_prepared(
    prepared: PreparedHalfspaceShift,
    kappa: float,
    atol: float = 1e-14,
) -> HalfspaceShiftResult:
    """Evaluate the shifted-threshold formula using one pre-sorted baseline sample."""

    k = float(kappa)
    if k < -atol:
        raise ValueError("kappa must be nonnegative.")
    k = max(k, 0.0)
    b = float(prepared.threshold)
    n = int(prepared.n)
    if n <= 0:
        raise ValueError("prepared baseline sample must be nonempty.")
    if k <= atol:
        return HalfspaceShiftResult(
            threshold=b,
            kappa=k,
            u_star=0.0,
            robust_probability=float(prepared.baseline_tail_count / n),
            degenerate=False,
            mean_shortfall=float(prepared.mean_shortfall),
            shifted_threshold=b,
            robust_tail_count=int(prepared.baseline_tail_count),
        )
    if k >= float(prepared.mean_shortfall) - atol:
        return HalfspaceShiftResult(
            threshold=b,
            kappa=k,
            u_star=float("inf"),
            robust_probability=1.0,
            degenerate=True,
            mean_shortfall=float(prepared.mean_shortfall),
            shifted_threshold=float("-inf"),
            robust_tail_count=n,
        )
    idx = int(np.searchsorted(prepared.truncated_moment_grid, k, side="left"))
    idx = min(max(idx, 0), prepared.D_sorted.size - 1)
    u_star = float(prepared.D_sorted[idx])
    shifted_threshold = float(b - u_star)
    robust_tail_count = int(n - np.searchsorted(prepared.scores_sorted, shifted_threshold, side="left"))
    return HalfspaceShiftResult(
        threshold=b,
        kappa=k,
        u_star=u_star,
        robust_probability=float(robust_tail_count / n),
        degenerate=False,
        mean_shortfall=float(prepared.mean_shortfall),
        shifted_threshold=shifted_threshold,
        robust_tail_count=robust_tail_count,
    )
