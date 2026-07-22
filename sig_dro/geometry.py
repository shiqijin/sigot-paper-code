"""Coordinatewise signature geometry used in Appendix E.1--E.2."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from .affine_payoff import LogPathModel, simulate_log_path_model
from .discrepancy import component_ot_vector
from .preprocess import fit_component_scaler
from .signatures import time_augmented_multichannel_signatures, time_augmented_signatures

Array = np.ndarray


@dataclass(frozen=True)
class FeatureBank:
    model_name: str
    raw_by_level: dict[int, Array]
    backend_used: str
    n_paths: int
    n_steps: int
    path_dim: int
    channel_names: tuple[str, ...]
    channel_spec: str


@dataclass(frozen=True)
class ParametricSweepSpec:
    sweep_name: str
    sweep_set: str
    model_family: str
    base_model: LogPathModel
    parameter_name: str
    parameter_values: tuple[float, ...]
    baseline_parameter_value: float | None = None
    baseline_relation: str = "structural_stress_family"
    interpretation: str = ""


def _levels(levels: Iterable[int]) -> tuple[int, ...]:
    result = tuple(sorted({int(level) for level in levels}))
    if not result or result[0] <= 0:
        raise ValueError("Signature levels must be positive and non-empty")
    return result


def _state_channels(paths: Array, channel_spec: str) -> tuple[Array | None, tuple[str, ...]]:
    spec = channel_spec.strip().lower().replace("-", "_")
    if spec == "time_price":
        return None, ("t", "X")
    if spec == "time_price_rv":
        increments = np.diff(paths, axis=1)
        rv = np.concatenate(
            [
                np.zeros((paths.shape[0], 1), dtype=float),
                np.cumsum(increments * increments, axis=1),
            ],
            axis=1,
        )
        return np.stack([paths, rv], axis=2), ("t", "X", "V")
    raise ValueError("channel_spec must be 'time_price' or 'time_price_rv'")


def compute_log_path_feature_bank(
    *,
    model: LogPathModel,
    paths: int,
    steps: int,
    T: float,
    seed: int,
    levels: Iterable[int],
    signature_backend: str = "iisignature",
    signature_type: str = "geometric",
    signature_representation: str = "signature",
    channel_spec: str = "time_price",
) -> FeatureBank:
    """Simulate one model and compute raw signatures at all requested levels."""
    level_tuple = _levels(levels)
    times, log_paths = simulate_log_path_model(
        model, paths=int(paths), steps=int(steps), T=float(T), seed=int(seed)
    )
    states, names = _state_channels(log_paths, channel_spec)
    if states is None:
        signatures, backend = time_augmented_signatures(
            times=times,
            paths=log_paths,
            max_level=max(level_tuple),
            backend=signature_backend,
            signature_type=signature_type,
            signature_representation=signature_representation,
            return_backend=True,
        )
    else:
        signatures, backend = time_augmented_multichannel_signatures(
            times=times,
            paths=states,
            max_level=max(level_tuple),
            backend=signature_backend,
            signature_type=signature_type,
            signature_representation=signature_representation,
            return_backend=True,
        )
    return FeatureBank(
        model_name=model.name,
        raw_by_level={
            level: np.asarray(signatures[level], dtype=float) for level in level_tuple
        },
        backend_used=str(backend),
        n_paths=int(paths),
        n_steps=int(steps),
        path_dim=len(names),
        channel_names=names,
        channel_spec=channel_spec,
    )


def pair_decoupled_budget(
    *,
    baseline_bank: FeatureBank,
    stress_bank: FeatureBank,
    level: int,
    scaling_method: str = "mad",
    scaling_floor: float = 1e-8,
    scaling_train_size: int | None = None,
    context: Mapping[str, Any] | None = None,
    scaling_reference_bank: FeatureBank | None = None,
) -> pd.DataFrame:
    """Return the coordinatewise empirical-W1 radius for one model pair."""
    level = int(level)
    raw0 = np.asarray(baseline_bank.raw_by_level[level], dtype=float)
    raw1 = np.asarray(stress_bank.raw_by_level[level], dtype=float)
    reference = baseline_bank if scaling_reference_bank is None else scaling_reference_bank
    raw_reference = np.asarray(reference.raw_by_level[level], dtype=float)
    if raw0.shape[1] != raw1.shape[1] or raw0.shape[1] != raw_reference.shape[1]:
        raise ValueError("Feature dimensions differ across geometry banks")
    train_size = raw_reference.shape[0] if scaling_train_size is None else min(
        int(scaling_train_size), raw_reference.shape[0]
    )
    scaler = fit_component_scaler(
        raw_reference,
        train_size=train_size,
        method=scaling_method,
        floor=scaling_floor,
    )
    delta = component_ot_vector(scaler.transform(raw0), scaler.transform(raw1))
    return pd.DataFrame([{
        **dict(context or {}),
        "baseline_model": baseline_bank.model_name,
        "stress_model": stress_bank.model_name,
        "signature_level": level,
        "feature_dim": int(delta.size),
        "n_baseline_paths": int(raw0.shape[0]),
        "n_stress_paths": int(raw1.shape[0]),
        "n_steps": baseline_bank.n_steps,
        "scaling_method": scaling_method,
        "scaling_train_size": train_size,
        "scaling_reference_model": reference.model_name,
        "channel_spec": baseline_bank.channel_spec,
        "channel_names": ",".join(baseline_bank.channel_names),
        "path_dim": baseline_bank.path_dim,
        "delta_l1": float(np.abs(delta).sum()),
        "delta_l2": float(np.linalg.norm(delta)),
        "delta_max": float(np.max(delta)),
    }])


def _sweep_model(
    base: LogPathModel, *, parameter_name: str, value: float, name: str
) -> LogPathModel:
    payload = asdict(base)
    if parameter_name not in payload:
        raise ValueError(f"Unknown LogPathModel field {parameter_name!r}")
    payload.update(name=name, **{parameter_name: float(value)})
    return LogPathModel(**payload)


def run_parametric_decoupled_geometry(
    *,
    baseline_model: LogPathModel,
    sweep_specs: Sequence[ParametricSweepSpec],
    paths: int,
    steps: int,
    T: float,
    baseline_seed: int,
    stress_seed: int,
    levels: Iterable[int],
    signature_backend: str = "iisignature",
    signature_type: str = "geometric",
    signature_representation: str = "signature",
    scaling_method: str = "mad",
    scaling_floor: float = 1e-8,
    scaling_train_size: int | None = None,
    replicate_id: int | None = None,
    sweep_seed_mode: str = "matched",
) -> pd.DataFrame:
    """Run all controlled parameter sweeps used by Appendix E.1."""
    level_tuple = _levels(levels)
    if sweep_seed_mode not in {"matched", "independent"}:
        raise ValueError("sweep_seed_mode must be 'matched' or 'independent'")
    baseline = compute_log_path_feature_bank(
        model=baseline_model,
        paths=paths,
        steps=steps,
        T=T,
        seed=baseline_seed,
        levels=level_tuple,
        signature_backend=signature_backend,
        signature_type=signature_type,
        signature_representation=signature_representation,
    )
    rows: list[pd.DataFrame] = []
    counter = 0
    for spec in sweep_specs:
        for value in map(float, spec.parameter_values):
            counter += 1
            token = f"{value:.6g}".replace("-", "m").replace(".", "p")
            stress_model = _sweep_model(
                spec.base_model,
                parameter_name=spec.parameter_name,
                value=value,
                name=f"{spec.sweep_name}_{spec.parameter_name}_{token}",
            )
            seed = baseline_seed if sweep_seed_mode == "matched" else stress_seed + 1000 * counter
            stress = compute_log_path_feature_bank(
                model=stress_model,
                paths=paths,
                steps=steps,
                T=T,
                seed=seed,
                levels=level_tuple,
                signature_backend=signature_backend,
                signature_type=signature_type,
                signature_representation=signature_representation,
            )
            context = {
                "section": "5.1.2",
                "sweep_name": spec.sweep_name,
                "sweep_set": spec.sweep_set,
                "model_family": spec.model_family,
                "parameter_name": spec.parameter_name,
                "parameter_value": value,
                "baseline_parameter_value": spec.baseline_parameter_value,
                "baseline_relation": spec.baseline_relation,
                "replicate_id": replicate_id,
                "delta_baseline_seed": baseline_seed,
                "delta_stress_seed": seed,
            }
            for level in level_tuple:
                rows.append(pair_decoupled_budget(
                    baseline_bank=baseline,
                    stress_bank=stress,
                    level=level,
                    scaling_method=scaling_method,
                    scaling_floor=scaling_floor,
                    scaling_train_size=scaling_train_size,
                    context=context,
                ))
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def summarize_geometry_replicates(
    frame: pd.DataFrame, group_cols: Sequence[str]
) -> pd.DataFrame:
    """Aggregate geometry replicates with matching mean/std column names."""
    if frame.empty:
        return frame.copy()
    rows: list[dict[str, Any]] = []
    seed_cols = {"replicate_id", "delta_baseline_seed", "delta_stress_seed"}
    for _, group in frame.groupby(list(group_cols), dropna=False, sort=True):
        row = {column: group.iloc[0][column] for column in group_cols}
        row["n_replicates"] = (
            int(group.replicate_id.nunique())
            if "replicate_id" in group
            else int(len(group))
        )
        for column in group.columns:
            if column in group_cols or column in seed_cols:
                continue
            series = group[column]
            if pd.api.types.is_numeric_dtype(series):
                values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
                row[column] = float(values.mean()) if len(values) else np.nan
                row[f"{column}_std"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                )
            else:
                values = series.dropna().astype(str).unique()
                row[column] = values[0] if len(values) == 1 else "mixed"
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)
