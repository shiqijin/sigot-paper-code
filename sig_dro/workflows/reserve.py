"""Shared Sig-OT experiment for the reserve-path functionals."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..configs import BrownianReserveConfig, CramerLundbergConfig, ExperimentConfig
from ..experiments import _block_spec_for_level, _budget_payload, _empirical_event_probability
from ..functionals import PathFunctional
from ..models import (
    psi_brownian_closed_form,
    simulate_brownian_reserve_paths,
    simulate_cramer_lundberg_paths,
)
from ..preprocess import fit_component_scaler
from ..references import BM19_EXAMPLE1_TABLE
from ..robust import conservative_shift_from_scores, robust_probability_from_scores
from ..signatures import (
    resolve_signature_representation,
    resolve_signature_type,
    time_augmented_signatures,
)
from .common import (
    _fit_surrogate_with_cfg,
    _is_preliminary_score_joint,
    _max_level,
    _mc_se,
    _preliminary_score_joint_delta,
    _resolved_backend,
    _safe_calibration_train_size,
    _score_model,
)

Array = np.ndarray


def _features_from_generator(
    generator: Iterable[tuple[Array, Array]],
    *,
    eps: float,
    cfg: ExperimentConfig,
    max_level: int,
    functional: PathFunctional | None = None,
) -> tuple[dict[int, Array], Array | None, str]:
    buckets: dict[int, list[Array]] = {
        level: [] for level in range(1, max_level + 1)
    }
    targets: list[Array] = []
    backend_used: str | None = None
    for times, paths in generator:
        signatures, backend_used = time_augmented_signatures(
            times=times,
            paths=paths,
            max_level=max_level,
            eps=eps,
            backend=cfg.signature_backend,
            return_backend=True,
            add_running_max_channel=cfg.add_running_max_channel,
            add_running_min_channel=cfg.add_running_min_channel,
            signature_type=resolve_signature_type(cfg.signature_type),
            signature_representation=resolve_signature_representation(
                cfg.signature_representation
            ),
        )
        for level in buckets:
            buckets[level].append(np.asarray(signatures[level], dtype=float))
        if functional is not None:
            targets.append(np.asarray(functional(times, paths), dtype=float))
    features = {
        level: np.vstack(parts) if parts else np.empty((0, 0), dtype=float)
        for level, parts in buckets.items()
    }
    target = None if functional is None else (
        np.concatenate(targets) if targets else np.empty(0, dtype=float)
    )
    backend = backend_used or _resolved_backend(cfg.signature_backend, cfg.signature_type)
    return features, target, str(backend)


def _brownian_features(
    cfg: BrownianReserveConfig,
    *,
    initial_capital: float,
    experiment_cfg: ExperimentConfig,
    max_level: int,
    functional: PathFunctional | None = None,
) -> tuple[dict[int, Array], Array | None, str]:
    return _features_from_generator(
        simulate_brownian_reserve_paths(cfg, initial_capital=initial_capital),
        eps=cfg.normalization_eps,
        cfg=experiment_cfg,
        max_level=max_level,
        functional=functional,
    )


def _cramer_lundberg_features(
    cfg: CramerLundbergConfig,
    *,
    initial_capital: float,
    experiment_cfg: ExperimentConfig,
    max_level: int,
) -> dict[int, Array]:
    features, _, _ = _features_from_generator(
        simulate_cramer_lundberg_paths(cfg, initial_capital=initial_capital),
        eps=cfg.normalization_eps,
        cfg=experiment_cfg,
        max_level=max_level,
    )
    return features


def _true_targets(
    cfg: CramerLundbergConfig,
    *,
    initial_capital: float,
    functional: PathFunctional,
) -> Array:
    parts = [
        np.asarray(functional(times, paths), dtype=float)
        for times, paths in simulate_cramer_lundberg_paths(
            cfg, initial_capital=initial_capital
        )
    ]
    return np.concatenate(parts) if parts else np.empty(0, dtype=float)


def _fit_models(
    *,
    functional: PathFunctional,
    delta_reserve_cfg: BrownianReserveConfig,
    delta_true_cfg: CramerLundbergConfig,
    fit_reserve_cfg: BrownianReserveConfig,
    experiment_cfg: ExperimentConfig,
    initial_capital: float,
) -> dict[int, dict[str, Any]]:
    if not _is_preliminary_score_joint(experiment_cfg):
        raise ValueError("Paper reserve experiments require preliminary-score joint OT")
    max_level = _max_level(experiment_cfg)
    fit_features, targets, backend_used = _brownian_features(
        fit_reserve_cfg,
        initial_capital=initial_capital,
        experiment_cfg=experiment_cfg,
        max_level=max_level,
        functional=functional,
    )
    delta_base, _, _ = _brownian_features(
        delta_reserve_cfg,
        initial_capital=initial_capital,
        experiment_cfg=experiment_cfg,
        max_level=max_level,
    )
    delta_true = _cramer_lundberg_features(
        delta_true_cfg,
        initial_capital=initial_capital,
        experiment_cfg=experiment_cfg,
        max_level=max_level,
    )
    train_size = _safe_calibration_train_size(
        delta_reserve_cfg.paths, experiment_cfg.train_size
    )
    fitted: dict[int, dict[str, Any]] = {}
    for level_value in experiment_cfg.signature_levels:
        level = int(level_value)
        raw_base = np.asarray(delta_base[level], dtype=float)
        scaler = fit_component_scaler(
            raw_base,
            train_size=train_size,
            method=experiment_cfg.scaling_method,
            floor=experiment_cfg.scaling_floor,
        )
        block_spec = _block_spec_for_level(delta_base, level, experiment_cfg)
        fit_scaled = scaler.transform(np.asarray(fit_features[level], dtype=float))
        preliminary = _preliminary_score_joint_delta(
            fit_features=fit_scaled,
            targets=np.asarray(targets, dtype=float),
            ot_baseline_features=scaler.transform(raw_base)[train_size:],
            ot_stress_features=scaler.transform(
                np.asarray(delta_true[level], dtype=float)
            ),
            block_spec=block_spec,
            cfg=experiment_cfg,
            ot_seed_offset=level,
        )
        model = _fit_surrogate_with_cfg(
            fit_scaled,
            np.asarray(targets, dtype=float),
            experiment_cfg,
            np.asarray(preliminary["regression_delta_vec"], dtype=float),
        )
        budget = _budget_payload(
            model=model,
            config=experiment_cfg,
            block_spec=block_spec,
            delta_override=np.asarray(preliminary["delta"], dtype=float),
        )
        ot_diag = preliminary["ot_diag"]
        fitted[level] = {
            "model": model,
            "scaler": scaler,
            "budget": budget,
            "backend_used": backend_used,
            "ot_objective": float(ot_diag.get("objective_value", np.nan)),
            "preliminary_theta_l1": float(
                ot_diag.get("preliminary_theta_l1", np.nan)
            ),
            "preliminary_n_active": int(
                ot_diag.get("preliminary_n_active", -1)
            ),
            "preliminary_rmse_train": float(
                ot_diag.get("preliminary_rmse_train", np.nan)
            ),
            "preliminary_rmse_test": float(
                ot_diag.get("preliminary_rmse_test", np.nan)
            ),
            "preliminary_r2_test": float(
                ot_diag.get("preliminary_r2_test", np.nan)
            ),
        }
    return fitted


def _evaluation_arrays(
    *,
    baseline_cfg: BrownianReserveConfig,
    true_cfg: CramerLundbergConfig,
    functional: PathFunctional,
    experiment_cfg: ExperimentConfig,
    initial_capital: float,
    fitted: dict[int, dict[str, Any]],
) -> tuple[Array, Array, dict[int, Array]]:
    max_level = _max_level(experiment_cfg)
    score_parts = {level: [] for level in fitted}
    base_target_parts: list[Array] = []
    for times, paths in simulate_brownian_reserve_paths(
        baseline_cfg, initial_capital=initial_capital
    ):
        signatures = time_augmented_signatures(
            times=times,
            paths=paths,
            max_level=max_level,
            eps=baseline_cfg.normalization_eps,
            backend=experiment_cfg.signature_backend,
            add_running_max_channel=experiment_cfg.add_running_max_channel,
            add_running_min_channel=experiment_cfg.add_running_min_channel,
            signature_type=resolve_signature_type(experiment_cfg.signature_type),
            signature_representation=resolve_signature_representation(
                experiment_cfg.signature_representation
            ),
        )
        base_target_parts.append(np.asarray(functional(times, paths), dtype=float))
        for level, item in fitted.items():
            features = item["scaler"].transform(
                np.asarray(signatures[level], dtype=float)
            )
            score_parts[level].append(_score_model(item["model"], features))
    base_targets = (
        np.concatenate(base_target_parts)
        if base_target_parts
        else np.empty(0, dtype=float)
    )
    true_targets = _true_targets(
        true_cfg, initial_capital=initial_capital, functional=functional
    )
    scores = {
        level: np.concatenate(parts) if parts else np.empty(0, dtype=float)
        for level, parts in score_parts.items()
    }
    return base_targets, true_targets, scores


def run_reserve_benchmark(
    *,
    functional: PathFunctional,
    delta_reserve_cfg: BrownianReserveConfig,
    delta_true_cfg: CramerLundbergConfig,
    fit_reserve_cfg: BrownianReserveConfig,
    eval_reserve_cfg: BrownianReserveConfig,
    eval_true_cfg: CramerLundbergConfig,
    experiment_cfg: ExperimentConfig,
    reference_initial_capital: float = 0.0,
    constant_boundary_baseline_closed_form: bool = False,
    attach_bm19_reference: bool = False,
) -> pd.DataFrame:
    """Fit one reserve score and evaluate all requested capital thresholds."""
    fitted = _fit_models(
        functional=functional,
        delta_reserve_cfg=delta_reserve_cfg,
        delta_true_cfg=delta_true_cfg,
        fit_reserve_cfg=fit_reserve_cfg,
        experiment_cfg=experiment_cfg,
        initial_capital=reference_initial_capital,
    )
    base_targets, true_targets, scores_by_level = _evaluation_arrays(
        baseline_cfg=eval_reserve_cfg,
        true_cfg=eval_true_cfg,
        functional=functional,
        experiment_cfg=experiment_cfg,
        initial_capital=reference_initial_capital,
        fitted=fitted,
    )
    rows: list[dict[str, Any]] = []
    for level, item in fitted.items():
        model, budget = item["model"], item["budget"]
        scores = scores_by_level[level]
        for threshold in map(float, experiment_cfg.thresholds):
            shift = conservative_shift_from_scores(
                model=model,
                scores=scores,
                threshold=threshold,
                event_sense=experiment_cfg.event_sense,
                kappa=float(budget["kappa"]),
                theta_l1=float(getattr(model, "theta_l1", 0.0)),
                budget_dim=int(budget["budget_dim"]),
            )
            p_base_mc = _empirical_event_probability(
                base_targets, threshold, experiment_cfg.event_sense
            )
            p_true = _empirical_event_probability(
                true_targets, threshold, experiment_cfg.event_sense
            )
            p_base = (
                psi_brownian_closed_form(threshold, eval_reserve_cfg)
                if constant_boundary_baseline_closed_form
                else p_base_mc
            )
            p_rob = robust_probability_from_scores(
                scores, shift.shifted_score_threshold, experiment_cfg.event_sense
            )
            closed_form_robust = (
                constant_boundary_baseline_closed_form
                and getattr(shift, "threshold_space", "prediction") == "prediction"
            )
            if closed_form_robust:
                p_rob = psi_brownian_closed_form(
                    shift.shifted_threshold, eval_reserve_cfg
                )
            rows.append({
                "k": level,
                "u": threshold,
                "kappa": float(budget["kappa"]),
                "theta_l1": float(getattr(model, "theta_l1", 0.0)),
                "shift_star": float(shift.shift_star),
                "shifted_threshold": float(shift.shifted_threshold),
                "budget_dim": int(budget["budget_dim"]),
                "psi_B": float(p_base),
                "psi_emp_base": float(p_base_mc),
                "psi_true": float(p_true),
                "psi_sig_rob": float(p_rob),
                "se_emp_base": _mc_se(p_base_mc, base_targets.size),
                "se_true": _mc_se(p_true, true_targets.size),
                "se_sig_rob_mc": 0.0 if closed_form_robust else _mc_se(
                    p_rob, scores.size
                ),
                "rmse_train": float(model.rmse_train),
                "rmse_test": float(model.rmse_test),
                "r2_test": float(model.r2_test),
                "n_active": int(model.n_active),
                "functional": functional.name,
                "preliminary_theta_l1": item["preliminary_theta_l1"],
                "preliminary_n_active": item["preliminary_n_active"],
                "preliminary_rmse_train": item["preliminary_rmse_train"],
                "preliminary_rmse_test": item["preliminary_rmse_test"],
                "preliminary_r2_test": item["preliminary_r2_test"],
            })
    result = pd.DataFrame(rows).sort_values(["k", "u"]).reset_index(drop=True)
    if attach_bm19_reference:
        reference = BM19_EXAMPLE1_TABLE.copy()
        reference["u"] = reference["u"].astype(float)
        result = result.merge(reference, on="u", how="left", suffixes=("", "_BM19"))
    return result
