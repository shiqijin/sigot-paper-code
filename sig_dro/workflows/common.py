"""Shared fitting, OT, and path-diagnostic helpers for paper workflows."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..blocks import block_dual_norm, expand_block_values
from ..configs import ExperimentConfig
from ..discrepancy import joint_ot_budget_vector
from ..regression import fit_surrogate
from ..signatures import resolve_signature_backend, resolve_signature_type

Array = np.ndarray


def _max_level(cfg: ExperimentConfig) -> int:
    return int(max(cfg.signature_levels))


def _resolved_backend(signature_backend: str, signature_type: str) -> str:
    resolve_signature_type(signature_type)
    return resolve_signature_backend(signature_backend)


def _safe_calibration_train_size(paths: int, requested: int) -> int:
    paths = int(paths)
    if paths <= 1:
        raise ValueError("calibration baseline paths must exceed 1.")
    return int(min(max(1, int(requested)), paths - 1))


def _mc_se(p: float, n: int) -> float:
    n = int(n)
    if n <= 0:
        return float("nan")
    return float(math.sqrt(max(float(p) * (1.0 - float(p)), 0.0) / n))


def _score_model(model, features: Array) -> Array:
    if hasattr(model, "link") and getattr(model, "link_type", "identity") != "identity":
        return np.asarray(model.score(features), dtype=float)
    return np.asarray(model.predict(features), dtype=float)


def _fit_surrogate_with_cfg(features: Array, targets: Array, cfg: ExperimentConfig, delta_vec: Array | None):
    return fit_surrogate(
        features=features,
        targets=targets,
        train_size=int(cfg.train_size),
        alpha=float(cfg.regression_alpha),
        delta_vec=None if delta_vec is None else np.asarray(delta_vec, dtype=float),
        max_iter=int(cfg.optimizer_max_iter),
        link_type=cfg.link_type,
        link_fit_mode=getattr(cfg, "link_fit_mode", "joint"),
        link_lipschitz=float(cfg.link_lipschitz),
        link_n_knots=int(cfg.link_n_knots),
        link_knot_quantile_low=float(cfg.link_knot_quantile_low),
        link_knot_quantile_high=float(cfg.link_knot_quantile_high),
        surrogate_type=cfg.surrogate_type,
    )


def _is_preliminary_score_joint(cfg: ExperimentConfig) -> bool:
    return str(cfg.ot_mode) == "joint" and str(cfg.ot_lambda_mode) == "preliminary_score"


def _fit_preliminary_score_model(features: Array, targets: Array, cfg: ExperimentConfig):
    """Fit the unweighted sparse affine score used to weight common-coupling OT."""

    return fit_surrogate(
        features=np.asarray(features, dtype=float),
        targets=np.asarray(targets, dtype=float),
        train_size=int(cfg.train_size),
        alpha=float(cfg.regression_alpha),
        delta_vec=None,
        max_iter=int(cfg.optimizer_max_iter),
        surrogate_type="affine",
        link_type="identity",
        link_fit_mode="posthoc",
    )


def _block_alpha_from_model(model: Any, block_spec) -> Array:
    link_lip = float(getattr(model, "link_lip_const", 1.0))
    theta = np.asarray(getattr(model, "theta", np.zeros(block_spec.feature_dim)), dtype=float).reshape(-1)
    if theta.shape[0] < block_spec.feature_dim:
        raise ValueError("preliminary model has fewer coefficients than the block feature dimension.")
    return link_lip * np.array([block_dual_norm(theta[block.slc], block.norm) for block in block_spec.blocks], dtype=float)


def _lambda_weights_from_preliminary_model(model: Any, block_spec, cfg: ExperimentConfig) -> Array:
    weights = np.asarray(_block_alpha_from_model(model, block_spec), dtype=float).reshape(-1)
    if weights.shape[0] < block_spec.n_blocks:
        raise ValueError("preliminary weight vector shorter than number of OT blocks.")
    weights = weights[: block_spec.n_blocks]
    weights = np.maximum(weights, float(cfg.ot_lambda_floor))
    if float(cfg.ot_lambda_power) != 1.0:
        weights = weights ** float(cfg.ot_lambda_power)
    if bool(cfg.ot_lambda_normalize):
        total = float(weights.sum())
        if total > 0.0:
            weights = weights / total
    return weights


def _preliminary_score_joint_delta(
    *,
    fit_features: Array,
    targets: Array,
    ot_baseline_features: Array,
    ot_stress_features: Array,
    block_spec,
    cfg: ExperimentConfig,
    ot_seed_offset: int = 0,
) -> dict[str, Any]:
    preliminary_model = _fit_preliminary_score_model(fit_features, targets, cfg)
    lambda_weights = _lambda_weights_from_preliminary_model(preliminary_model, block_spec, cfg)
    ot_res = joint_ot_budget_vector(
        baseline_features=np.asarray(ot_baseline_features, dtype=float),
        true_features=np.asarray(ot_stress_features, dtype=float),
        block_spec=block_spec,
        lambda_weights=lambda_weights,
        batch_size=cfg.ot_batch_size,
        n_batches=cfg.ot_batches,
        epsilon=cfg.ot_sinkhorn_epsilon,
        max_iter=cfg.ot_sinkhorn_max_iter,
        tol=cfg.ot_sinkhorn_tol,
        seed=int(cfg.ot_seed) + int(ot_seed_offset),
        mode="subsampled",
    )
    delta = np.asarray(ot_res.delta_blocks, dtype=float)
    return {
        "delta": delta,
        "regression_delta_vec": expand_block_values(delta, block_spec, feature_dim=np.asarray(fit_features).shape[1]),
        "ot_diag": {
            "objective_value": float(ot_res.objective_value),
            "coupling_mode": str(ot_res.coupling_mode),
            "n_source": int(ot_res.n_source),
            "n_target": int(ot_res.n_target),
            "lambda_weights": np.asarray(ot_res.lambda_weights, dtype=float),
            "lambda_mode": "preliminary_score",
            "preliminary_theta_l1": float(getattr(preliminary_model, "theta_l1", float("nan"))),
            "preliminary_n_active": int(getattr(preliminary_model, "n_active", -1)),
            "preliminary_rmse_train": float(getattr(preliminary_model, "rmse_train", float("nan"))),
            "preliminary_rmse_test": float(getattr(preliminary_model, "rmse_test", float("nan"))),
            "preliminary_r2_test": float(getattr(preliminary_model, "r2_test", float("nan"))),
        },
        "preliminary_model": preliminary_model,
    }


def _log_path_diag_summary(terminal_parts: list[Array], rv_parts: list[Array]) -> dict[str, float]:
    terminal = np.concatenate(terminal_parts) if terminal_parts else np.empty(0, dtype=float)
    rv = np.concatenate(rv_parts) if rv_parts else np.empty(0, dtype=float)
    if terminal.size == 0 or rv.size == 0:
        return {
            "terminal_mean": float("nan"),
            "terminal_sd": float("nan"),
            "terminal_q05": float("nan"),
            "terminal_q50": float("nan"),
            "terminal_q95": float("nan"),
            "rv_mean": float("nan"),
            "rv_sd": float("nan"),
            "rv_sqrt_mean": float("nan"),
        }
    return {
        "terminal_mean": float(np.mean(terminal)),
        "terminal_sd": float(np.std(terminal, ddof=1)) if terminal.size > 1 else 0.0,
        "terminal_q05": float(np.quantile(terminal, 0.05)),
        "terminal_q50": float(np.quantile(terminal, 0.50)),
        "terminal_q95": float(np.quantile(terminal, 0.95)),
        "rv_mean": float(np.mean(rv)),
        "rv_sd": float(np.std(rv, ddof=1)) if rv.size > 1 else 0.0,
        "rv_sqrt_mean": float(np.sqrt(max(float(np.mean(rv)), 0.0))),
    }


def _append_log_path_diag_parts(paths: Array, terminal_parts: list[Array], rv_parts: list[Array]) -> None:
    x = np.asarray(paths, dtype=float)
    terminal_parts.append(np.asarray(x[:, -1], dtype=float).copy())
    dx = np.diff(x, axis=1)
    rv_parts.append(np.asarray(np.sum(dx * dx, axis=1), dtype=float))
