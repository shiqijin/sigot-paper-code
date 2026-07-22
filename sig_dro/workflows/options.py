"""Signature-surrogate experiment for Asian and lookback options."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..configs import ExperimentConfig
from ..experiments import _block_spec_for_level, _budget_payload
from ..preprocess import fit_component_scaler
from ..signatures import (
    resolve_signature_representation,
    resolve_signature_type,
    time_augmented_signatures,
)
from .common import (
    _fit_surrogate_with_cfg,
    _is_preliminary_score_joint,
    _max_level,
    _preliminary_score_joint_delta,
    _resolved_backend,
    _safe_calibration_train_size,
)

Array = np.ndarray


def _path_option_payoff(
    log_paths: Array,
    name: str,
    *,
    S0: float,
    strike: float,
    r: float,
    T: float,
    asian_include_initial: bool,
) -> Array:
    from ..option_payoffs import (
        discounted_arithmetic_asian_call_payoff,
        discounted_fixed_strike_lookback_call_payoff,
    )

    if name == "asian_call":
        return discounted_arithmetic_asian_call_payoff(
            log_paths,
            S0=S0,
            K=strike,
            r=r,
            T=T,
            include_initial=asian_include_initial,
        )
    if name == "lookback_call":
        return discounted_fixed_strike_lookback_call_payoff(
            log_paths, S0=S0, K=strike, r=r, T=T
        )
    raise ValueError(f"Unsupported payoff {name!r}")


def _stream_dataset(
    model: Any,
    *,
    paths: int,
    steps: int,
    T: float,
    seed: int,
    cfg: ExperimentConfig,
    batch_size: int,
    payoff_names: Iterable[str] = (),
    S0: float = 1.0,
    strike: float = 1.0,
    r: float = 0.0,
    asian_include_initial: bool = True,
) -> tuple[dict[int, Array], dict[str, Array], str]:
    from ..affine_payoff import simulate_log_path_model_batches

    payoff_names = tuple(payoff_names)
    max_level = _max_level(cfg)
    feature_parts: dict[int, list[Array]] = {
        level: [] for level in range(1, max_level + 1)
    }
    payoff_parts = {name: [] for name in payoff_names}
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
            max_level=max_level,
            backend=cfg.signature_backend,
            return_backend=True,
            add_running_max_channel=cfg.add_running_max_channel,
            add_running_min_channel=cfg.add_running_min_channel,
            signature_type=resolve_signature_type(cfg.signature_type),
            signature_representation=resolve_signature_representation(
                cfg.signature_representation
            ),
        )
        for level in feature_parts:
            feature_parts[level].append(np.asarray(signatures[level], dtype=float))
        for name in payoff_names:
            payoff_parts[name].append(
                _path_option_payoff(
                    log_paths,
                    name,
                    S0=S0,
                    strike=strike,
                    r=r,
                    T=T,
                    asian_include_initial=asian_include_initial,
                )
            )
    features = {
        level: np.vstack(parts) if parts else np.empty((0, 0), dtype=float)
        for level, parts in feature_parts.items()
    }
    payoffs = {
        name: np.concatenate(parts) if parts else np.empty(0, dtype=float)
        for name, parts in payoff_parts.items()
    }
    backend = backend_used or _resolved_backend(cfg.signature_backend, cfg.signature_type)
    return features, payoffs, str(backend)


def _mean_and_se(n: int, total: float, total_sq: float) -> tuple[float, float]:
    if n <= 0:
        return np.nan, np.nan
    mean = total / n
    if n == 1:
        return float(mean), 0.0
    variance = max(total_sq - total * total / n, 0.0) / (n - 1)
    return float(mean), float(np.sqrt(variance / n))


def _prediction_summary(
    model_spec: Any,
    *,
    fitted_model: Any,
    scaler: Any,
    level: int,
    payoff_name: str,
    paths: int,
    steps: int,
    T: float,
    seed: int,
    cfg: ExperimentConfig,
    batch_size: int,
    S0: float,
    strike: float,
    r: float,
    asian_include_initial: bool,
    residual_buffers: bool,
) -> dict[str, float]:
    from ..affine_payoff import simulate_log_path_model_batches

    n = 0
    hard_sum = hard_sq_sum = pred_sum = 0.0
    eps_upper_sum = eps_lower_sum = 0.0
    for times, log_paths in simulate_log_path_model_batches(
        model_spec,
        paths=int(paths),
        steps=int(steps),
        T=float(T),
        seed=int(seed),
        batch_size=int(batch_size),
    ):
        signatures = time_augmented_signatures(
            times=times,
            paths=log_paths,
            max_level=_max_level(cfg),
            backend=cfg.signature_backend,
            add_running_max_channel=cfg.add_running_max_channel,
            add_running_min_channel=cfg.add_running_min_channel,
            signature_type=resolve_signature_type(cfg.signature_type),
            signature_representation=resolve_signature_representation(
                cfg.signature_representation
            ),
        )
        features = scaler.transform(np.asarray(signatures[level], dtype=float))
        pred = np.asarray(fitted_model.predict(features), dtype=float).reshape(-1)
        hard = np.asarray(
            _path_option_payoff(
                log_paths,
                payoff_name,
                S0=S0,
                strike=strike,
                r=r,
                T=T,
                asian_include_initial=asian_include_initial,
            ),
            dtype=float,
        ).reshape(-1)
        n += hard.size
        hard_sum += float(hard.sum())
        hard_sq_sum += float(hard @ hard)
        pred_sum += float(pred.sum())
        if residual_buffers:
            eps_upper_sum += float(np.maximum(hard - pred, 0.0).sum())
            eps_lower_sum += float(np.maximum(pred - hard, 0.0).sum())
    hard_mean, hard_se = _mean_and_se(n, hard_sum, hard_sq_sum)
    return {
        "hard_mean": hard_mean,
        "hard_se": hard_se,
        "pred_mean": pred_sum / n if n else np.nan,
        "eps_upper": eps_upper_sum / n if n and residual_buffers else np.nan,
        "eps_lower": eps_lower_sum / n if n and residual_buffers else np.nan,
    }


def run_path_option_expectation(
    *,
    baseline_model: Any,
    stress_models: Iterable[Any],
    experiment_cfg: ExperimentConfig,
    payoff_names: Iterable[str],
    S0: float,
    strike: float,
    T: float,
    r: float,
    q: float,
    asian_include_initial: bool,
    delta_baseline_paths: int,
    delta_true_paths: int,
    delta_steps: int,
    delta_baseline_seed: int,
    delta_true_seed: int,
    fit_baseline_paths: int,
    fit_steps: int,
    fit_baseline_seed: int,
    eval_baseline_paths: int,
    eval_true_paths: int,
    eval_steps: int,
    eval_baseline_seed: int,
    eval_true_seed: int,
    path_batch_size: int = 256,
) -> pd.DataFrame:
    """Run the staged option experiment used by Table 4 and Appendix E.3."""
    del q  # The supplied log-price models already encode the risk-neutral drift.
    payoff_names = tuple(str(name).lower() for name in payoff_names)
    stress_models = list(stress_models)
    if not payoff_names:
        raise ValueError("payoff_names must be non-empty")
    if not _is_preliminary_score_joint(experiment_cfg):
        raise ValueError("Paper option experiments require preliminary-score joint OT")

    fit_features, fit_payoffs, backend_used = _stream_dataset(
        baseline_model,
        paths=fit_baseline_paths,
        steps=fit_steps,
        T=T,
        seed=fit_baseline_seed,
        cfg=experiment_cfg,
        batch_size=path_batch_size,
        payoff_names=payoff_names,
        S0=S0,
        strike=strike,
        r=r,
        asian_include_initial=asian_include_initial,
    )
    delta_base, _, _ = _stream_dataset(
        baseline_model,
        paths=delta_baseline_paths,
        steps=delta_steps,
        T=T,
        seed=delta_baseline_seed,
        cfg=experiment_cfg,
        batch_size=path_batch_size,
    )
    train_size = _safe_calibration_train_size(
        delta_baseline_paths, experiment_cfg.train_size
    )
    base_items: dict[int, tuple[Any, Any, Array]] = {}
    for level in experiment_cfg.signature_levels:
        level = int(level)
        raw = np.asarray(delta_base[level], dtype=float)
        scaler = fit_component_scaler(
            raw,
            train_size=train_size,
            method=experiment_cfg.scaling_method,
            floor=experiment_cfg.scaling_floor,
        )
        block_spec = _block_spec_for_level(delta_base, level, experiment_cfg)
        base_items[level] = (
            scaler,
            block_spec,
            scaler.transform(raw)[train_size:],
        )

    rows: list[dict[str, Any]] = []
    for stress_index, stress_model in enumerate(stress_models, start=1):
        stress_name = str(getattr(stress_model, "name", f"stress_{stress_index}"))
        delta_true, _, _ = _stream_dataset(
            stress_model,
            paths=delta_true_paths,
            steps=delta_steps,
            T=T,
            seed=delta_true_seed + 1000 * stress_index,
            cfg=experiment_cfg,
            batch_size=path_batch_size,
        )
        for payoff_name in payoff_names:
            for level_value in experiment_cfg.signature_levels:
                level = int(level_value)
                scaler, block_spec, baseline_ot = base_items[level]
                fit_scaled = scaler.transform(
                    np.asarray(fit_features[level], dtype=float)
                )
                targets = np.asarray(fit_payoffs[payoff_name], dtype=float)
                preliminary = _preliminary_score_joint_delta(
                    fit_features=fit_scaled,
                    targets=targets,
                    ot_baseline_features=baseline_ot,
                    ot_stress_features=scaler.transform(
                        np.asarray(delta_true[level], dtype=float)
                    ),
                    block_spec=block_spec,
                    cfg=experiment_cfg,
                    ot_seed_offset=1000 * stress_index + level,
                )
                model = _fit_surrogate_with_cfg(
                    fit_scaled,
                    targets,
                    experiment_cfg,
                    np.asarray(preliminary["regression_delta_vec"], dtype=float),
                )
                budget = _budget_payload(
                    model=model,
                    config=experiment_cfg,
                    block_spec=block_spec,
                    delta_override=np.asarray(preliminary["delta"], dtype=float),
                )
                baseline = _prediction_summary(
                    baseline_model,
                    fitted_model=model,
                    scaler=scaler,
                    level=level,
                    payoff_name=payoff_name,
                    paths=eval_baseline_paths,
                    steps=eval_steps,
                    T=T,
                    seed=eval_baseline_seed,
                    cfg=experiment_cfg,
                    batch_size=path_batch_size,
                    S0=S0,
                    strike=strike,
                    r=r,
                    asian_include_initial=asian_include_initial,
                    residual_buffers=True,
                )
                stress = _prediction_summary(
                    stress_model,
                    fitted_model=model,
                    scaler=scaler,
                    level=level,
                    payoff_name=payoff_name,
                    paths=eval_true_paths,
                    steps=eval_steps,
                    T=T,
                    seed=eval_true_seed + 1000 * stress_index,
                    cfg=experiment_cfg,
                    batch_size=path_batch_size,
                    S0=S0,
                    strike=strike,
                    r=r,
                    asian_include_initial=asian_include_initial,
                    residual_buffers=False,
                )
                base_price, stress_price = baseline["hard_mean"], stress["hard_mean"]
                kappa = float(budget["kappa"])
                rows.append({
                    "payoff": payoff_name,
                    "stress_model": stress_name,
                    "k": level,
                    "V_baseline_hard_eval": base_price,
                    "SE_baseline_hard_eval": baseline["hard_se"],
                    "V_stress_hard": stress_price,
                    "SE_stress_hard": stress["hard_se"],
                    "hard_true_gap": stress_price - base_price,
                    "E_baseline_surrogate": baseline["pred_mean"],
                    "E_stress_surrogate": stress["pred_mean"],
                    "surrogate_gap": stress["pred_mean"] - baseline["pred_mean"],
                    "kappa_eff": kappa,
                    "kappa_joint": kappa,
                    "surrogate_lower": baseline["pred_mean"] - kappa,
                    "surrogate_upper": baseline["pred_mean"] + kappa,
                    "price_anchored_lower": base_price - kappa,
                    "price_anchored_upper": base_price + kappa,
                    "eps_upper_eval": baseline["eps_upper"],
                    "eps_lower_eval": baseline["eps_lower"],
                    "tightness_abs_gap_over_kappa": (
                        abs(stress_price - base_price) / kappa if kappa > 0 else np.inf
                    ),
                    "rmse_train": float(model.rmse_train),
                    "rmse_test": float(model.rmse_test),
                    "r2_test": float(model.r2_test),
                    "n_active": int(model.n_active),
                    "theta_l1": float(getattr(model, "theta_l1", 0.0)),
                    "signature_backend": backend_used,
                    "ot_objective": float(
                        preliminary["ot_diag"].get("objective_value", np.nan)
                    ),
                })
    return pd.DataFrame(rows).sort_values(
        ["payoff", "stress_model", "k"]
    ).reset_index(drop=True)
