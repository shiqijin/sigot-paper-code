from __future__ import annotations

"""Aggregate per-replicate Section 5 outputs into publication tables."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


def read_csv(path: Path, required: Iterable[str]) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Missing experiment output: {path}")
    frame = pd.read_csv(path)
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    if frame.empty:
        raise ValueError(f"Experiment output is empty: {path}")
    return frame


def safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    den = pd.to_numeric(denominator, errors="coerce").replace(0.0, np.nan)
    return pd.to_numeric(numerator, errors="coerce") / den


def validate_replicate_grid(
    frame: pd.DataFrame,
    keys: list[str],
    expected_seeds: tuple[int, ...] | None,
) -> None:
    required = [*keys, "ot_seed"]
    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ValueError(f"Cannot validate replicate grid; missing columns: {missing}")
    work = frame.loc[:, required].copy()
    numeric = pd.to_numeric(work["ot_seed"], errors="raise")
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise ValueError("ot_seed values must be finite integers")
    work["ot_seed"] = numeric.astype(int)
    duplicate = work.duplicated([*keys, "ot_seed"], keep=False)
    if duplicate.any():
        sample = work.loc[duplicate, [*keys, "ot_seed"]].head(5).to_dict("records")
        raise ValueError(f"Duplicate replicate rows detected: {sample}")
    observed = work.groupby(keys, dropna=False, sort=True)["ot_seed"].agg(
        lambda values: tuple(sorted(int(value) for value in values))
    )
    if observed.empty:
        raise ValueError("No replicate cells were found")
    required_seeds = expected_seeds if expected_seeds is not None else observed.iloc[0]
    invalid = observed[observed != required_seeds]
    if not invalid.empty:
        sample = {str(index): list(value) for index, value in invalid.head(5).items()}
        raise ValueError(f"Replicate seed grid mismatch; expected {list(required_seeds)}, found {sample}")


def aggregate(frame: pd.DataFrame, keys: list[str], values: list[str]) -> pd.DataFrame:
    missing = [name for name in keys + values if name not in frame.columns]
    if missing:
        raise ValueError(f"Cannot aggregate missing columns: {missing}")
    work = frame.loc[:, keys + values].copy()
    for name in values:
        work[name] = pd.to_numeric(work[name], errors="raise")
    grouped = work.groupby(keys, dropna=False, sort=True)
    means = grouped[values].mean().rename(columns={name: f"{name}_mean" for name in values})
    stds = grouped[values].std(ddof=1).fillna(0.0).rename(columns={name: f"{name}_std" for name in values})
    counts = grouped.size().rename("n_replicates")
    return pd.concat([counts, means, stds], axis=1).reset_index()


def table_1(root: Path, expected_seeds: tuple[int, ...] | None) -> pd.DataFrame:
    path = root / "budget_calibration" / "diagnostics" / "section51_budget_calibration_results_replicates.csv"
    values = [
        "delta_decoupled_l1", "delta_joint_l1", "kappa_decoupled", "kappa_joint",
        "delta_joint_over_decoupled_l1", "joint_over_decoupled",
    ]
    raw = read_csv(path, ["stress_model", "ot_seed", *values])
    validate_replicate_grid(raw, ["stress_model"], expected_seeds)
    out = aggregate(raw, ["stress_model"], values)
    columns = ["stress_model", "n_replicates"]
    for name in values:
        columns.extend([f"{name}_mean", f"{name}_std"])
    return out.loc[:, columns]


def table_2(root: Path, expected_seeds: tuple[int, ...] | None) -> pd.DataFrame:
    path = root / "linear_expectation" / "diagnostics" / "linear_signature_expectation_results_replicates.csv"
    required = ["stress_model", "ot_seed", "baseline_mean", "stress_mean", "mean_shift", "kappa_joint", "upper_bound", "lower_bound"]
    raw = read_csv(path, required)
    validate_replicate_grid(raw, ["stress_model"], expected_seeds)
    raw = raw.copy()
    raw["shift_over_kappa"] = safe_ratio(raw["mean_shift"], raw["kappa_joint"])
    raw["robust_upper"] = raw["upper_bound"]
    raw["robust_lower"] = raw["lower_bound"]
    raw["upper_slack"] = raw["robust_upper"] - raw["stress_mean"]
    raw["abs_tightness_ratio"] = safe_ratio(raw["mean_shift"].abs(), raw["kappa_joint"])
    values = [
        "baseline_mean", "stress_mean", "mean_shift", "kappa_joint", "shift_over_kappa",
        "robust_upper", "robust_lower", "upper_slack", "abs_tightness_ratio",
    ]
    out = aggregate(raw, ["stress_model"], values)
    columns = [
        "stress_model", "n_replicates",
        "baseline_mean_mean", "baseline_mean_std", "stress_mean_mean", "stress_mean_std",
        "mean_shift_mean", "mean_shift_std", "kappa_joint_mean", "kappa_joint_std",
        "shift_over_kappa_mean", "shift_over_kappa_std", "robust_upper_mean", "robust_lower_mean",
        "upper_slack_mean", "abs_tightness_ratio_mean", "robust_upper_std", "robust_lower_std",
        "upper_slack_std", "abs_tightness_ratio_std",
    ]
    return out.loc[:, columns]


def table_3(root: Path, expected_seeds: tuple[int, ...] | None) -> pd.DataFrame:
    path = root / "linear_probability" / "diagnostics" / "linear_signature_probability_results_replicates.csv"
    required = [
        "stress_model", "threshold", "ot_seed", "baseline_prob", "stress_prob", "robust_prob_joint",
        "kappa_joint", "u_star_joint", "shifted_threshold_joint",
    ]
    raw = read_csv(path, required).copy()
    validate_replicate_grid(raw, ["stress_model", "threshold"], expected_seeds)
    raw["stress_over_base"] = safe_ratio(raw["stress_prob"], raw["baseline_prob"])
    raw["robust_over_stress"] = safe_ratio(raw["robust_prob_joint"], raw["stress_prob"])
    raw["robust_minus_stress"] = raw["robust_prob_joint"] - raw["stress_prob"]
    values = [
        "baseline_prob", "stress_prob", "robust_prob_joint", "stress_over_base",
        "robust_over_stress", "robust_minus_stress", "kappa_joint", "u_star_joint",
        "shifted_threshold_joint",
    ]
    out = aggregate(raw, ["stress_model", "threshold"], values)
    columns = [
        "stress_model", "threshold", "n_replicates", "baseline_prob_mean", "baseline_prob_std",
        "stress_prob_mean", "stress_prob_std", "robust_prob_joint_mean", "robust_prob_joint_std",
        "stress_over_base_mean", "stress_over_base_std", "robust_over_stress_mean", "robust_over_stress_std",
        "robust_minus_stress_mean", "kappa_joint_mean", "u_star_joint_mean", "shifted_threshold_joint_mean",
        "robust_minus_stress_std", "kappa_joint_std", "u_star_joint_std", "shifted_threshold_joint_std",
    ]
    return out.loc[:, columns]


def table_4(root: Path, expected_seeds: tuple[int, ...] | None) -> pd.DataFrame:
    path = root / "option_pricing" / "diagnostics" / "path_dependent_option_pricing_results_replicates.csv"
    required = [
        "payoff", "strike", "stress_sigma", "stress_sigma_shift", "ot_seed", "V_baseline_hard_eval",
        "V_stress_hard", "hard_true_gap", "kappa_eff", "tightness_abs_gap_over_kappa",
        "r2_test", "price_anchored_upper", "SE_stress_hard", "rmse_test",
    ]
    raw = read_csv(path, required).copy()
    validate_replicate_grid(
        raw, ["stress_sigma", "stress_sigma_shift", "strike", "payoff"], expected_seeds
    )
    raw["abs_shift_over_kappa"] = raw["tightness_abs_gap_over_kappa"]
    raw["upper_slack"] = raw["price_anchored_upper"] - raw["V_stress_hard"]
    values = [
        "V_baseline_hard_eval", "V_stress_hard", "hard_true_gap", "kappa_eff",
        "abs_shift_over_kappa", "r2_test", "price_anchored_upper", "upper_slack",
    ]
    grouped = aggregate(raw, ["stress_sigma", "stress_sigma_shift", "strike", "payoff"], values)

    key_cols = ["stress_sigma", "stress_sigma_shift", "strike"]
    pieces: list[pd.DataFrame] = []
    labels = {"asian_call": "asian", "lookback_call": "lookback"}
    for payoff, prefix in labels.items():
        part = grouped[grouped["payoff"] == payoff].drop(columns=["payoff"]).copy()
        if part.empty:
            raise ValueError(f"Option output is missing payoff {payoff!r}")
        rename = {"n_replicates": f"{prefix}_n_replicates"}
        for value in values:
            rename[f"{value}_mean"] = f"{prefix}_{value}_mean"
            rename[f"{value}_std"] = f"{prefix}_{value}_std"
        pieces.append(part.rename(columns=rename))
    out = pieces[0].merge(pieces[1], on=key_cols, how="outer", validate="one_to_one")
    columns = [
        *key_cols,
        "asian_n_replicates", "asian_V_baseline_hard_eval_mean", "asian_V_baseline_hard_eval_std",
        "asian_V_stress_hard_mean", "asian_V_stress_hard_std", "asian_hard_true_gap_mean", "asian_hard_true_gap_std",
        "asian_kappa_eff_mean", "asian_kappa_eff_std", "asian_abs_shift_over_kappa_mean", "asian_abs_shift_over_kappa_std",
        "asian_r2_test_mean", "asian_r2_test_std",
        "lookback_n_replicates", "lookback_V_baseline_hard_eval_mean", "lookback_V_baseline_hard_eval_std",
        "lookback_V_stress_hard_mean", "lookback_V_stress_hard_std", "lookback_hard_true_gap_mean", "lookback_hard_true_gap_std",
        "lookback_kappa_eff_mean", "lookback_kappa_eff_std", "lookback_abs_shift_over_kappa_mean", "lookback_abs_shift_over_kappa_std",
        "lookback_r2_test_mean", "lookback_r2_test_std",
        "asian_price_anchored_upper_mean", "asian_upper_slack_mean", "asian_price_anchored_upper_std", "asian_upper_slack_std",
        "lookback_price_anchored_upper_mean", "lookback_upper_slack_mean", "lookback_price_anchored_upper_std", "lookback_upper_slack_std",
    ]
    out = out.loc[:, columns].sort_values(key_cols).reset_index(drop=True)

    return out


def table_6(root: Path, expected_seeds: tuple[int, ...] | None) -> pd.DataFrame:
    paths = [
        root / "reserve_constant" / "diagnostics" / "reserve_constant_boundary_results_replicates.csv",
        root / "reserve_sine" / "diagnostics" / "reserve_sine_vulnerability_results_replicates.csv",
        root / "reserve_early" / "diagnostics" / "reserve_early_shortfall_results_replicates.csv",
        root / "reserve_drawdown" / "diagnostics" / "reserve_drawdown_results_replicates.csv",
    ]
    required = ["task", "u", "ot_seed", "psi_B", "psi_true", "psi_sig_rob", "kappa", "shift_star", "r2_test", "rmse_test", "n_active"]
    raw = pd.concat([read_csv(path, required) for path in paths], ignore_index=True)
    validate_replicate_grid(raw, ["task", "u"], expected_seeds)
    raw["target"] = raw["task"].astype(str)
    raw["brownian_over_cl"] = safe_ratio(raw["psi_B"], raw["psi_true"])
    raw["sigot_over_cl"] = safe_ratio(raw["psi_sig_rob"], raw["psi_true"])
    values = [
        "psi_B", "psi_true", "psi_sig_rob", "brownian_over_cl", "sigot_over_cl",
        "kappa", "shift_star", "r2_test", "rmse_test", "n_active",
    ]
    out = aggregate(raw, ["target", "u"], values)
    columns = [
        "target", "u", "n_replicates", "psi_B_mean", "psi_B_std", "psi_true_mean", "psi_true_std",
        "psi_sig_rob_mean", "psi_sig_rob_std", "brownian_over_cl_mean", "brownian_over_cl_std",
        "sigot_over_cl_mean", "sigot_over_cl_std", "kappa_mean", "kappa_std", "shift_star_mean", "shift_star_std",
        "r2_test_mean", "r2_test_std", "rmse_test_mean", "n_active_mean", "rmse_test_std", "n_active_std",
    ]
    order = {name: index for index, name in enumerate([
        "constant_boundary", "sine_vulnerability_am20", "early_shortfall_linear_rho1", "max_drawdown"
    ])}
    out["_order"] = out["target"].map(order).fillna(999)
    return out.sort_values(["_order", "u"]).drop(columns=["_order"]).loc[:, columns].reset_index(drop=True)


def table_7(root: Path, expected_seeds: tuple[int, ...] | None) -> pd.DataFrame:
    path = root / "bm19" / "diagnostics" / "bm19_ruin_benchmark_results_replicates.csv"
    required = [
        "representation_case", "u", "ot_seed", "psi_B_BM19", "psi_true", "psi_rob_j1", "psi_sig_rob",
        "kappa", "r2_test",
    ]
    raw = read_csv(path, required).copy()
    validate_replicate_grid(raw, ["representation_case", "u"], expected_seeds)
    raw["brownian_value"] = raw["psi_B_BM19"]
    raw["brownian_over_cl"] = safe_ratio(raw["brownian_value"], raw["psi_true"])
    raw["j1_over_cl"] = safe_ratio(raw["psi_rob_j1"], raw["psi_true"])
    raw["sigot_over_j1"] = safe_ratio(raw["psi_sig_rob"], raw["psi_rob_j1"])

    plain_common = raw[raw["representation_case"] == "plain"]
    extrema_common = raw[raw["representation_case"] == "extrema"]
    if plain_common.empty:
        raise ValueError("BM19 output is missing the plain representation")
    if extrema_common.empty:
        raise ValueError("BM19 output is missing the extrema representation")

    # Common BM19 quantities occur in both representation blocks. Pool both blocks
    # for their summary statistics and count the unique seed replicates separately.
    common = aggregate(
        raw,
        ["u"],
        ["brownian_value", "psi_true", "psi_rob_j1", "brownian_over_cl", "j1_over_cl"],
    )
    unique_counts = plain_common.groupby("u", dropna=False).size()
    extrema_counts = extrema_common.groupby("u", dropna=False).size()
    if not unique_counts.equals(extrema_counts):
        raise ValueError("BM19 plain/extrema replicate counts do not align")
    common["n_replicates"] = common["u"].map(unique_counts).astype(int)

    merged = common
    for case in ["plain", "extrema"]:
        part_raw = raw[raw["representation_case"] == case]
        if part_raw.empty:
            raise ValueError(f"BM19 output is missing the {case} representation")
        part = aggregate(part_raw, ["u"], ["psi_sig_rob", "sigot_over_j1", "kappa", "r2_test"])
        rename = {
            "n_replicates": f"{case}_n_replicates",
            "psi_sig_rob_mean": f"{case}_psi_sig_rob_mean",
            "psi_sig_rob_std": f"{case}_psi_sig_rob_std",
            "sigot_over_j1_mean": f"{case}_sigot_over_j1_mean",
            "sigot_over_j1_std": f"{case}_sigot_over_j1_std",
            "kappa_mean": f"{case}_kappa_mean", "kappa_std": f"{case}_kappa_std",
            "r2_test_mean": f"{case}_r2_test_mean", "r2_test_std": f"{case}_r2_test_std",
        }
        merged = merged.merge(part.rename(columns=rename), on="u", how="inner", validate="one_to_one")

    columns = [
        "u", "n_replicates", "brownian_value_mean", "brownian_value_std", "psi_true_mean", "psi_true_std",
        "psi_rob_j1_mean", "psi_rob_j1_std", "plain_psi_sig_rob_mean", "plain_psi_sig_rob_std",
        "extrema_psi_sig_rob_mean", "extrema_psi_sig_rob_std", "brownian_over_cl_mean", "brownian_over_cl_std",
        "j1_over_cl_mean", "j1_over_cl_std", "plain_sigot_over_j1_mean", "plain_sigot_over_j1_std",
        "extrema_sigot_over_j1_mean", "extrema_sigot_over_j1_std", "plain_n_replicates",
        "plain_kappa_mean", "plain_r2_test_mean", "plain_kappa_std", "plain_r2_test_std",
        "extrema_n_replicates", "extrema_kappa_mean", "extrema_r2_test_mean", "extrema_kappa_std", "extrema_r2_test_std",
    ]
    return merged.sort_values("u").loc[:, columns].reset_index(drop=True)
