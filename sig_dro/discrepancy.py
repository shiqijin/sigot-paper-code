from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp

from .blocks import BlockSpec, FeatureBlock

Array = np.ndarray


def empirical_w1_1d(x: Array, y: Array) -> float:
    x = np.sort(np.asarray(x, dtype=float).reshape(-1))
    y = np.sort(np.asarray(y, dtype=float).reshape(-1))
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if x.size == 0 or y.size == 0:
        return 0.0
    if x.size == y.size:
        return float(np.mean(np.abs(x - y)))
    probs = (np.arange(max(x.size, y.size), dtype=float) + 0.5) / max(x.size, y.size)
    qx = np.quantile(x, probs, method="linear")
    qy = np.quantile(y, probs, method="linear")
    return float(np.mean(np.abs(qx - qy)))


def component_ot_vector(baseline_features: Array, true_features: Array) -> Array:
    """Componentwise empirical W1 distances.

    The equal-sample-size case is the dominant path in the engineering runners.
    It admits an exact vectorised formula based on columnwise order statistics,
    which is substantially faster than a Python loop over coordinates.  The
    Unequal sample sizes and non-finite values use the scalar quantile path.
    """
    baseline_features = np.asarray(baseline_features, dtype=float)
    true_features = np.asarray(true_features, dtype=float)
    if baseline_features.ndim != 2 or true_features.ndim != 2:
        raise ValueError("baseline_features and true_features must be 2D arrays.")
    if baseline_features.shape[1] != true_features.shape[1]:
        raise ValueError("Feature dimensions do not match.")
    if (
        baseline_features.shape[0] == true_features.shape[0]
        and np.all(np.isfinite(baseline_features))
        and np.all(np.isfinite(true_features))
    ):
        x_sort = np.sort(baseline_features, axis=0)
        y_sort = np.sort(true_features, axis=0)
        return np.mean(np.abs(x_sort - y_sort), axis=0, dtype=float)
    return np.array([
        empirical_w1_1d(baseline_features[:, j], true_features[:, j])
        for j in range(baseline_features.shape[1])
    ], dtype=float)


@dataclass(frozen=True)
class JointOTResult:
    delta_blocks: Array
    lambda_weights: Array
    objective_value: float
    batch_estimates: Array | None = None
    n_source: int = 0
    n_target: int = 0
    coupling_mode: str = "subsampled_sinkhorn"


def block_cost_matrix(X: Array, Y: Array, block: FeatureBlock) -> Array:
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    ZX = X[:, block.slc]
    ZY = Y[:, block.slc]
    diff = ZX[:, None, :] - ZY[None, :, :]
    if block.norm == "l2":
        return np.sqrt(np.sum(diff * diff, axis=-1))
    raise ValueError("The paper implementation uses the l2 block norm.")


def sinkhorn_uniform(
    cost: Array,
    epsilon: float,
    max_iter: int = 400,
    tol: float = 1e-9,
):
    cost = np.asarray(cost, dtype=float)
    if cost.ndim != 2:
        raise ValueError("cost must be a matrix.")
    if cost.shape[0] == 0 or cost.shape[1] == 0:
        raise ValueError("cost matrix must be non-empty.")
    eps = float(epsilon)
    if eps <= 0.0:
        raise ValueError("epsilon must be positive.")
    n, m = cost.shape
    log_a = np.full(n, -np.log(float(n)), dtype=float)
    log_b = np.full(m, -np.log(float(m)), dtype=float)
    f = np.zeros(n, dtype=float)
    g = np.zeros(m, dtype=float)
    for _ in range(int(max_iter)):
        f_new = eps * (log_a - logsumexp((g[None, :] - cost) / eps, axis=1))
        g_new = eps * (log_b - logsumexp((f_new[:, None] - cost) / eps, axis=0))
        err = max(float(np.max(np.abs(f_new - f))), float(np.max(np.abs(g_new - g))))
        f, g = f_new, g_new
        if err <= float(tol):
            break
    log_pi = (f[:, None] + g[None, :] - cost) / eps
    pi = np.exp(log_pi)
    total = float(np.sum(pi))
    if not np.isfinite(total) or total <= 0.0:
        raise FloatingPointError("Sinkhorn coupling underflow/overflow encountered.")
    pi = pi / total
    return pi


def _resolve_lambda_weights(block_spec: BlockSpec, lambda_weights: Array | None) -> Array:
    if lambda_weights is None:
        return np.ones(block_spec.n_blocks, dtype=float)
    weights = np.asarray(lambda_weights, dtype=float).reshape(-1)
    if weights.size < block_spec.n_blocks:
        raise ValueError("lambda_weights shorter than number of blocks.")
    weights = weights[: block_spec.n_blocks]
    if np.any(weights < 0.0):
        raise ValueError("lambda_weights must be non-negative.")
    return weights


def _joint_ot_budget_vector_subsampled(
    baseline_features: Array,
    true_features: Array,
    block_spec: BlockSpec,
    lambda_weights: Array | None = None,
    batch_size: int = 512,
    n_batches: int = 8,
    epsilon: float = 1e-2,
    max_iter: int = 400,
    tol: float = 1e-9,
    seed: int = 0,
) -> JointOTResult:
    baseline_features = np.asarray(baseline_features, dtype=float)
    true_features = np.asarray(true_features, dtype=float)
    if baseline_features.ndim != 2 or true_features.ndim != 2:
        raise ValueError("baseline_features and true_features must be 2D arrays.")
    if baseline_features.shape[1] != true_features.shape[1]:
        raise ValueError("Feature dimensions do not match.")
    if baseline_features.shape[1] != block_spec.feature_dim:
        raise ValueError("block_spec.feature_dim inconsistent with feature matrices.")
    if int(n_batches) <= 0:
        raise ValueError("n_batches must be positive.")
    lam = _resolve_lambda_weights(block_spec, lambda_weights)
    rng = np.random.default_rng(int(seed))
    deltas = []
    objectives = []
    n_src = baseline_features.shape[0]
    n_tgt = true_features.shape[0]
    src_batch = min(int(batch_size), n_src)
    tgt_batch = min(int(batch_size), n_tgt)
    replace_src = n_src < int(batch_size)
    replace_tgt = n_tgt < int(batch_size)
    for _ in range(int(n_batches)):
        src_idx = rng.choice(n_src, size=src_batch, replace=replace_src)
        tgt_idx = rng.choice(n_tgt, size=tgt_batch, replace=replace_tgt)
        X = baseline_features[src_idx]
        Y = true_features[tgt_idx]
        block_costs = [block_cost_matrix(X, Y, block) for block in block_spec.blocks]
        total_cost = np.zeros((src_batch, tgt_batch), dtype=float)
        for weight, cost in zip(lam, block_costs):
            if weight == 0.0:
                continue
            total_cost += float(weight) * cost
        pi = sinkhorn_uniform(total_cost, epsilon=epsilon, max_iter=max_iter, tol=tol)
        delta_batch = np.array([float(np.sum(pi * cost)) for cost in block_costs], dtype=float)
        deltas.append(delta_batch)
        objectives.append(float(np.dot(lam, delta_batch)))
    batch_estimates = np.vstack(deltas)
    return JointOTResult(
        delta_blocks=np.mean(batch_estimates, axis=0),
        lambda_weights=lam,
        objective_value=float(np.mean(objectives)),
        batch_estimates=batch_estimates,
        n_source=int(n_src),
        n_target=int(n_tgt),
        coupling_mode="subsampled_sinkhorn",
    )


def joint_ot_budget_vector(
    baseline_features: Array,
    true_features: Array,
    block_spec: BlockSpec,
    lambda_weights: Array | None = None,
    batch_size: int = 512,
    n_batches: int = 8,
    epsilon: float = 1e-2,
    max_iter: int = 400,
    tol: float = 1e-9,
    seed: int = 0,
    mode: str = "subsampled",
) -> JointOTResult:
    if mode != "subsampled":
        raise ValueError("The paper implementation uses subsampled Sinkhorn OT.")
    return _joint_ot_budget_vector_subsampled(
        baseline_features=baseline_features,
        true_features=true_features,
        block_spec=block_spec,
        lambda_weights=lambda_weights,
        batch_size=batch_size,
        n_batches=n_batches,
        epsilon=epsilon,
        max_iter=max_iter,
        tol=tol,
        seed=seed,
    )
