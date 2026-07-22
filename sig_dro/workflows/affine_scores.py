"""Fixed affine-signature expectation and tail-probability experiments."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..preprocess import fit_component_scaler
from ..signatures import (
    resolve_signature_type,
    signature_coordinate_names,
    time_augmented_signatures,
)
from .common import _append_log_path_diag_parts, _log_path_diag_summary, _resolved_backend

Array = np.ndarray


def _active_coordinates(
    theta_by_name: dict[str, float], signature_level: int
) -> tuple[list[str], list[int], Array]:
    names = signature_coordinate_names(
        path_dim=2, max_level=int(signature_level), channel_names=("t", "X")
    )
    index = {name: i for i, name in enumerate(names)}
    missing = sorted(set(theta_by_name) - set(index))
    if missing:
        raise ValueError(f"Unknown signature coordinates: {missing}")
    active = list(theta_by_name)
    return active, [index[name] for name in active], np.asarray(
        [theta_by_name[name] for name in active], dtype=float
    )


def _simulate_active_signatures(
    model: Any,
    *,
    paths: int,
    steps: int,
    T: float,
    seed: int,
    batch_size: int,
    signature_level: int,
    signature_backend: str,
    signature_type: str,
    active_indices: list[int],
    diagnostics: bool = False,
) -> tuple[Array, dict[str, float], str]:
    from ..affine_payoff import simulate_log_path_model_batches

    feature_parts: list[Array] = []
    terminal_parts: list[Array] = []
    rv_parts: list[Array] = []
    backend_used: str | None = None
    for times, log_paths in simulate_log_path_model_batches(
        model,
        paths=int(paths),
        steps=int(steps),
        T=float(T),
        seed=int(seed),
        batch_size=int(batch_size),
    ):
        signatures, backend_used = time_augmented_signatures(
            times=times,
            paths=log_paths,
            max_level=int(signature_level),
            backend=signature_backend,
            return_backend=True,
            signature_type=resolve_signature_type(signature_type),
        )
        feature_parts.append(
            np.asarray(signatures[int(signature_level)], dtype=float)[:, active_indices]
        )
        if diagnostics:
            _append_log_path_diag_parts(log_paths, terminal_parts, rv_parts)
    features = (
        np.vstack(feature_parts)
        if feature_parts
        else np.empty((0, len(active_indices)), dtype=float)
    )
    diag = _log_path_diag_summary(terminal_parts, rv_parts) if diagnostics else {}
    backend = backend_used or _resolved_backend(signature_backend, signature_type)
    return features, diag, str(backend)


def calibrate_fixed_score(
    *,
    baseline_model: Any,
    stress_model: Any,
    theta_by_name: dict[str, float],
    delta_baseline_paths: int,
    delta_stress_paths: int,
    delta_steps: int,
    delta_baseline_seed: int,
    delta_stress_seed: int,
    T: float,
    path_batch_size: int,
    signature_level: int,
    signature_backend: str,
    signature_type: str,
    scaling_method: str,
    scaling_floor: float,
    ot_mode: str,
    ot_batch_size: int,
    ot_batches: int,
    ot_epsilon: float,
    ot_max_iter: int,
    ot_tol: float,
    ot_seed: int,
) -> dict[str, Any]:
    """Fit baseline scaling and calibrate joint/decoupled score budgets."""
    from ..affine_payoff import evaluate_affine_payoff_against_stress

    active_names, active_indices, theta = _active_coordinates(
        theta_by_name, signature_level
    )
    raw0, _, backend_used = _simulate_active_signatures(
        baseline_model,
        paths=delta_baseline_paths,
        steps=delta_steps,
        T=T,
        seed=delta_baseline_seed,
        batch_size=path_batch_size,
        signature_level=signature_level,
        signature_backend=signature_backend,
        signature_type=signature_type,
        active_indices=active_indices,
    )
    scaler = fit_component_scaler(
        raw0,
        train_size=raw0.shape[0],
        method=scaling_method,
        floor=scaling_floor,
    )
    raw1, _, _ = _simulate_active_signatures(
        stress_model,
        paths=delta_stress_paths,
        steps=delta_steps,
        T=T,
        seed=delta_stress_seed,
        batch_size=path_batch_size,
        signature_level=signature_level,
        signature_backend=signature_backend,
        signature_type=signature_type,
        active_indices=active_indices,
    )
    result, delta_joint, contribution_joint, delta_decoupled, contribution_decoupled = (
        evaluate_affine_payoff_against_stress(
            baseline_active=scaler.transform(raw0),
            stress_active=scaler.transform(raw1),
            theta=theta,
            stress_name=str(getattr(stress_model, "name", "stress")),
            ot_batch_size=ot_batch_size,
            ot_batches=ot_batches,
            ot_epsilon=ot_epsilon,
            ot_max_iter=ot_max_iter,
            ot_tol=ot_tol,
            ot_seed=ot_seed,
            ot_mode=ot_mode,
        )
    )
    return {
        "scaler": scaler,
        "active_names": active_names,
        "active_indices": active_indices,
        "theta": theta,
        "signature_backend_used": backend_used,
        "delta_joint": np.asarray(delta_joint, dtype=float),
        "delta_decoupled": np.asarray(delta_decoupled, dtype=float),
        "contribution_joint": np.asarray(contribution_joint, dtype=float),
        "contribution_decoupled": np.asarray(contribution_decoupled, dtype=float),
        "kappa_joint": float(result.kappa_joint),
        "kappa_decoupled": float(result.kappa_decoupled),
        "joint_over_decoupled": float(result.joint_over_decoupled),
        "ot_objective": float(result.ot_objective),
        "budget_mode": str(ot_mode),
    }


def _evaluate_scores(
    baseline_model: Any,
    stress_model: Any,
    calibration: dict[str, Any],
    *,
    eval_baseline_paths: int,
    eval_stress_paths: int,
    eval_steps: int,
    eval_baseline_seed: int,
    eval_stress_seed: int,
    T: float,
    path_batch_size: int,
    signature_level: int,
    signature_backend: str,
    signature_type: str,
) -> tuple[Array, Array, dict[str, float], dict[str, float]]:
    raw0, diag0, _ = _simulate_active_signatures(
        baseline_model,
        paths=eval_baseline_paths,
        steps=eval_steps,
        T=T,
        seed=eval_baseline_seed,
        batch_size=path_batch_size,
        signature_level=signature_level,
        signature_backend=signature_backend,
        signature_type=signature_type,
        active_indices=calibration["active_indices"],
        diagnostics=True,
    )
    raw1, diag1, _ = _simulate_active_signatures(
        stress_model,
        paths=eval_stress_paths,
        steps=eval_steps,
        T=T,
        seed=eval_stress_seed,
        batch_size=path_batch_size,
        signature_level=signature_level,
        signature_backend=signature_backend,
        signature_type=signature_type,
        active_indices=calibration["active_indices"],
        diagnostics=True,
    )
    scaler, theta = calibration["scaler"], calibration["theta"]
    return (
        scaler.transform(raw0) @ theta,
        scaler.transform(raw1) @ theta,
        diag0,
        diag1,
    )


def _calibration_for_stress(
    stress_index: int,
    stress_model: Any,
    *,
    baseline_model: Any,
    theta_by_name: dict[str, float],
    delta_baseline_paths: int,
    delta_stress_paths: int,
    delta_steps: int,
    delta_baseline_seed: int,
    delta_stress_seed: int,
    T: float,
    path_batch_size: int,
    signature_level: int,
    signature_backend: str,
    signature_type: str,
    scaling_method: str,
    scaling_floor: float,
    ot_mode: str,
    ot_batch_size: int,
    ot_batches: int,
    ot_epsilon: float,
    ot_max_iter: int,
    ot_tol: float,
    ot_seed: int,
) -> dict[str, Any]:
    return calibrate_fixed_score(
        baseline_model=baseline_model,
        stress_model=stress_model,
        theta_by_name=theta_by_name,
        delta_baseline_paths=delta_baseline_paths,
        delta_stress_paths=delta_stress_paths,
        delta_steps=delta_steps,
        delta_baseline_seed=delta_baseline_seed,
        delta_stress_seed=delta_stress_seed + 1000 * stress_index,
        T=T,
        path_batch_size=path_batch_size,
        signature_level=signature_level,
        signature_backend=signature_backend,
        signature_type=signature_type,
        scaling_method=scaling_method,
        scaling_floor=scaling_floor,
        ot_mode=ot_mode,
        ot_batch_size=ot_batch_size,
        ot_batches=ot_batches,
        ot_epsilon=ot_epsilon,
        ot_max_iter=ot_max_iter,
        ot_tol=ot_tol,
        ot_seed=ot_seed + 2000 * stress_index,
    )


def run_affine_signature_payoff(
    *,
    baseline_model: Any,
    stress_models: Iterable[Any],
    theta_by_name: dict[str, float],
    delta_baseline_paths: int,
    delta_stress_paths: int,
    delta_steps: int,
    delta_baseline_seed: int,
    delta_stress_seed: int,
    eval_baseline_paths: int,
    eval_stress_paths: int,
    eval_steps: int,
    eval_baseline_seed: int,
    eval_stress_seed: int,
    T: float,
    path_batch_size: int,
    signature_level: int,
    signature_backend: str,
    signature_type: str,
    scaling_method: str,
    scaling_floor: float,
    ot_mode: str,
    ot_batch_size: int,
    ot_batches: int,
    ot_epsilon: float,
    ot_max_iter: int,
    ot_tol: float,
    ot_seed: int,
) -> pd.DataFrame:
    """Evaluate the fixed affine score under each stress law (Table 2)."""
    rows: list[dict[str, Any]] = []
    calibration_args = dict(
        baseline_model=baseline_model,
        theta_by_name=theta_by_name,
        delta_baseline_paths=delta_baseline_paths,
        delta_stress_paths=delta_stress_paths,
        delta_steps=delta_steps,
        delta_baseline_seed=delta_baseline_seed,
        delta_stress_seed=delta_stress_seed,
        T=T,
        path_batch_size=path_batch_size,
        signature_level=signature_level,
        signature_backend=signature_backend,
        signature_type=signature_type,
        scaling_method=scaling_method,
        scaling_floor=scaling_floor,
        ot_mode=ot_mode,
        ot_batch_size=ot_batch_size,
        ot_batches=ot_batches,
        ot_epsilon=ot_epsilon,
        ot_max_iter=ot_max_iter,
        ot_tol=ot_tol,
        ot_seed=ot_seed,
    )
    for index, stress_model in enumerate(stress_models, start=1):
        calibration = _calibration_for_stress(
            index, stress_model, **calibration_args
        )
        score0, score1, diag0, diag1 = _evaluate_scores(
            baseline_model,
            stress_model,
            calibration,
            eval_baseline_paths=eval_baseline_paths,
            eval_stress_paths=eval_stress_paths,
            eval_steps=eval_steps,
            eval_baseline_seed=eval_baseline_seed,
            eval_stress_seed=eval_stress_seed + 1000 * index,
            T=T,
            path_batch_size=path_batch_size,
            signature_level=signature_level,
            signature_backend=signature_backend,
            signature_type=signature_type,
        )
        baseline_mean, stress_mean = float(np.mean(score0)), float(np.mean(score1))
        shift = stress_mean - baseline_mean
        kappa = calibration["kappa_joint"]
        rows.append({
            "stress_model": str(getattr(stress_model, "name", "stress")),
            "baseline_mean": baseline_mean,
            "stress_mean": stress_mean,
            "mean_shift": shift,
            "kappa_joint": kappa,
            "upper_bound": baseline_mean + kappa,
            "lower_bound": baseline_mean - kappa,
            "covered_upper": stress_mean <= baseline_mean + kappa + 1e-12,
            "covered_lower": stress_mean >= baseline_mean - kappa - 1e-12,
            "covered_abs": abs(shift) <= kappa + 1e-12,
            "signed_tightness_ratio": shift / kappa if kappa > 0 else np.nan,
            "abs_tightness_ratio": abs(shift) / kappa if kappa > 0 else np.nan,
            "terminal_sd_ratio": diag1["terminal_sd"] / diag0["terminal_sd"],
            "rv_mean_ratio": diag1["rv_mean"] / diag0["rv_mean"],
        })
    return pd.DataFrame(rows).sort_values("stress_model").reset_index(drop=True)


def run_linear_halfspace_probability(
    *,
    baseline_model: Any,
    stress_models: Iterable[Any],
    theta_by_name: dict[str, float],
    thresholds: Iterable[float],
    delta_baseline_paths: int,
    delta_stress_paths: int,
    delta_steps: int,
    delta_baseline_seed: int,
    delta_stress_seed: int,
    eval_baseline_paths: int,
    eval_stress_paths: int,
    eval_steps: int,
    eval_baseline_seed: int,
    eval_stress_seed: int,
    T: float,
    path_batch_size: int,
    signature_level: int,
    signature_backend: str,
    signature_type: str,
    scaling_method: str,
    scaling_floor: float,
    ot_mode: str,
    ot_batch_size: int,
    ot_batches: int,
    ot_epsilon: float,
    ot_max_iter: int,
    ot_tol: float,
    ot_seed: int,
) -> pd.DataFrame:
    """Evaluate shifted-threshold robust probabilities (Table 3)."""
    from ..affine_payoff import (
        prepare_shifted_threshold_probability,
        shifted_threshold_probability_from_prepared,
    )

    rows: list[dict[str, Any]] = []
    calibration_args = dict(
        baseline_model=baseline_model,
        theta_by_name=theta_by_name,
        delta_baseline_paths=delta_baseline_paths,
        delta_stress_paths=delta_stress_paths,
        delta_steps=delta_steps,
        delta_baseline_seed=delta_baseline_seed,
        delta_stress_seed=delta_stress_seed,
        T=T,
        path_batch_size=path_batch_size,
        signature_level=signature_level,
        signature_backend=signature_backend,
        signature_type=signature_type,
        scaling_method=scaling_method,
        scaling_floor=scaling_floor,
        ot_mode=ot_mode,
        ot_batch_size=ot_batch_size,
        ot_batches=ot_batches,
        ot_epsilon=ot_epsilon,
        ot_max_iter=ot_max_iter,
        ot_tol=ot_tol,
        ot_seed=ot_seed,
    )
    for index, stress_model in enumerate(stress_models, start=1):
        calibration = _calibration_for_stress(
            index, stress_model, **calibration_args
        )
        score0, score1, _, _ = _evaluate_scores(
            baseline_model,
            stress_model,
            calibration,
            eval_baseline_paths=eval_baseline_paths,
            eval_stress_paths=eval_stress_paths,
            eval_steps=eval_steps,
            eval_baseline_seed=eval_baseline_seed,
            eval_stress_seed=eval_stress_seed + 1000 * index,
            T=T,
            path_batch_size=path_batch_size,
            signature_level=signature_level,
            signature_backend=signature_backend,
            signature_type=signature_type,
        )
        kappa = calibration["kappa_joint"]
        for threshold in map(float, thresholds):
            p0 = float(np.mean(score0 >= threshold))
            p1 = float(np.mean(score1 >= threshold))
            shift = shifted_threshold_probability_from_prepared(
                prepare_shifted_threshold_probability(score0, threshold), kappa
            )
            rows.append({
                "stress_model": str(getattr(stress_model, "name", "stress")),
                "threshold": threshold,
                "baseline_prob": p0,
                "stress_prob": p1,
                "baseline_tail_count": int(np.sum(score0 >= threshold)),
                "stress_tail_count": int(np.sum(score1 >= threshold)),
                "kappa_joint": kappa,
                "u_star_joint": float(shift.u_star),
                "shifted_threshold_joint": float(shift.shifted_threshold),
                "robust_prob_joint": float(shift.robust_probability),
                "robust_tail_count_joint": int(shift.robust_tail_count),
                "covered_joint": p1 <= shift.robust_probability + 1e-12,
                "degenerate_joint": bool(shift.degenerate),
            })
    return pd.DataFrame(rows).sort_values(["stress_model", "threshold"]).reset_index(drop=True)
