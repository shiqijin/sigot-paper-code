"""Section 5 experiments, organized by manuscript table."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from sig_dro.affine_payoff import LogPathModel, default_stress_models, parse_theta_spec
from sig_dro.boundaries import make_boundary_spec
from sig_dro.configs import BrownianReserveConfig, CramerLundbergConfig, ExperimentConfig
from sig_dro.functionals import (
    PathFunctional,
    make_curved_boundary_ruin_depth_functional,
    make_drawdown_functional,
    make_early_shortfall_functional,
    ruin_depth,
)
from sig_dro.paper_outputs import prepare_output_tree, selected_frame, write_csv, write_selected_csv
from sig_dro.stability import (
    annotate_replicate,
    make_fixed_score_seed_replicates,
    make_seed_replicates,
    summarize_replicates,
)
from sig_dro.workflows.affine_scores import (
    calibrate_fixed_score,
    run_affine_signature_payoff,
    run_linear_halfspace_probability,
)
from sig_dro.workflows.options import run_path_option_expectation
from sig_dro.workflows.reserve import run_reserve_benchmark

from .settings import (
    bm19_thresholds,
    option_stresses,
    option_strikes,
    probability_eval_paths,
    reserve_eval_paths,
    reserve_thresholds,
    scale,
    seeds,
)


BASELINE = LogPathModel("baseline_gbm", "gbm", mu=.05, sigma=.20)
EXPECTATION_THETA = "X:-0.35,tX:0.15,Xt:-0.45,XX:0.20"
PROBABILITY_THETA = "Xt:-0.25,Xtt:-0.20,XXt:0.20,XXXt:-0.35"


def _dirs(root: Path, name: str) -> dict[str, Path]:
    return prepare_output_tree(root / name)


def _fixed_replicates(profile: str):
    return make_fixed_score_seed_replicates(
        delta_baseline_seeds=seeds(profile, 123),
        delta_stress_seeds=seeds(profile, 1123),
        eval_baseline_seeds=seeds(profile, 200123),
        eval_stress_seeds=seeds(profile, 201123),
        ot_seeds=seeds(profile, 0),
    )


def _replicates(profile: str):
    return make_seed_replicates(
        delta_baseline_seeds=seeds(profile, 123),
        delta_true_seeds=seeds(profile, 1123),
        fit_baseline_seeds=seeds(profile, 100000),
        eval_baseline_seeds=seeds(profile, 200000),
        eval_true_seeds=seeds(profile, 210000),
        ot_seeds=seeds(profile, 0),
    )


def _concat(parts: list[pd.DataFrame]) -> pd.DataFrame:
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def run_budget_calibration(profile: str, root: Path) -> pd.DataFrame:
    """Table 1: decoupled versus common-coupling budgets."""
    s, out = scale(profile), _dirs(root, "budget_calibration")
    parts: list[pd.DataFrame] = []
    for rep in _fixed_replicates(profile):
        rows = []
        for index, model in enumerate(default_stress_models(), start=1):
            result = calibrate_fixed_score(
                baseline_model=BASELINE,
                stress_model=model,
                theta_by_name=parse_theta_spec(EXPECTATION_THETA),
                delta_baseline_paths=32 if profile == "smoke" else s.delta_paths,
                delta_stress_paths=32 if profile == "smoke" else s.delta_paths,
                delta_steps=s.steps,
                delta_baseline_seed=rep.delta_baseline_seed,
                delta_stress_seed=rep.delta_stress_seed + 1000 * index,
                T=1.,
                path_batch_size=s.batch_size,
                signature_level=2,
                signature_backend="iisignature",
                signature_type="geometric",
                scaling_method="mad",
                scaling_floor=1e-8,
                ot_mode="subsampled",
                ot_batch_size=s.ot_batch_size,
                ot_batches=s.ot_batches,
                ot_epsilon=1e-2,
                ot_max_iter=s.ot_max_iter,
                ot_tol=s.ot_tol,
                ot_seed=rep.ot_seed + 2000 * index,
            )
            delta_joint = result["delta_joint"]
            delta_decoupled = result["delta_decoupled"]
            rows.append({
                "stress_model": model.name,
                "delta_decoupled_l1": float(delta_decoupled.sum()),
                "delta_joint_l1": float(delta_joint.sum()),
                "delta_joint_over_decoupled_l1": float(
                    delta_joint.sum() / delta_decoupled.sum()
                ),
                "kappa_decoupled": result["kappa_decoupled"],
                "kappa_joint": result["kappa_joint"],
                "joint_over_decoupled": result["joint_over_decoupled"],
                "ot_objective": result["ot_objective"],
            })
        parts.append(annotate_replicate(pd.DataFrame(rows), rep))
    raw = _concat(parts)
    write_csv(
        raw,
        out["diagnostics"] / "section51_budget_calibration_results_replicates.csv",
    )
    summary = summarize_replicates(raw, group_cols=["stress_model"])
    return write_selected_csv(
        summary,
        out["main"] / "section51_budget_calibration_summary.csv",
        [
            "stress_model",
            "delta_decoupled_l1",
            "delta_joint_l1",
            "kappa_decoupled",
            "kappa_joint",
        ],
    )


def _fixed_common(profile: str, rep) -> dict:
    s = scale(profile)
    return dict(
        baseline_model=BASELINE,
        stress_models=default_stress_models(),
        delta_baseline_paths=32 if profile == "smoke" else s.delta_paths,
        delta_stress_paths=32 if profile == "smoke" else s.delta_paths,
        delta_steps=s.steps,
        delta_baseline_seed=rep.delta_baseline_seed,
        delta_stress_seed=rep.delta_stress_seed,
        eval_baseline_paths=s.eval_paths,
        eval_stress_paths=s.eval_paths,
        eval_steps=s.steps,
        eval_baseline_seed=rep.eval_baseline_seed,
        eval_stress_seed=rep.eval_stress_seed,
        T=1.,
        path_batch_size=s.batch_size,
        signature_backend="iisignature",
        signature_type="geometric",
        scaling_method="mad",
        scaling_floor=1e-8,
        ot_mode="subsampled",
        ot_batch_size=512,
        ot_batches=16,
        ot_epsilon=1e-2,
        ot_max_iter=400,
        ot_tol=1e-9,
        ot_seed=rep.ot_seed,
    )


def run_fixed_scores(profile: str, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tables 2--3: fixed affine-signature expectation and tails."""
    expectation_parts, probability_parts = [], []
    exp_out = _dirs(root, "linear_expectation")
    prob_out = _dirs(root, "linear_probability")
    for rep in _fixed_replicates(profile):
        common = _fixed_common(profile, rep)
        expectation = run_affine_signature_payoff(
            theta_by_name=parse_theta_spec(EXPECTATION_THETA),
            signature_level=2,
            **common,
        )
        probability = run_linear_halfspace_probability(
            theta_by_name=parse_theta_spec(PROBABILITY_THETA),
            signature_level=4,
            thresholds=(20, 30, 40),
            **{
                **common,
                "eval_baseline_paths": probability_eval_paths(profile),
                "eval_stress_paths": probability_eval_paths(profile),
            },
        )
        expectation_parts.append(annotate_replicate(expectation, rep))
        probability_parts.append(annotate_replicate(probability, rep))
    exp_raw, prob_raw = _concat(expectation_parts), _concat(probability_parts)
    write_csv(
        exp_raw,
        exp_out["diagnostics"] / "linear_signature_expectation_results_replicates.csv",
    )
    write_csv(
        prob_raw,
        prob_out["diagnostics"] / "linear_signature_probability_results_replicates.csv",
    )
    exp = summarize_replicates(exp_raw, group_cols=["stress_model"])
    prob = summarize_replicates(prob_raw, group_cols=["stress_model", "threshold"])
    prob["stress_over_baseline"] = prob.stress_prob / prob.baseline_prob.replace(0., np.nan)
    prob["robust_over_stress_joint"] = prob.robust_prob_joint / prob.stress_prob.replace(0., np.nan)
    exp_main = write_selected_csv(
        exp,
        exp_out["main"] / "linear_signature_expectation_summary.csv",
        [
            "stress_model", "n_replicates", "baseline_mean", "baseline_mean_std",
            "stress_mean", "stress_mean_std", "mean_shift", "mean_shift_std",
            "kappa_joint", "kappa_joint_std", "upper_bound", "upper_bound_std",
            "lower_bound", "lower_bound_std", "covered_upper", "covered_lower",
            "covered_abs", "signed_tightness_ratio", "signed_tightness_ratio_std",
            "abs_tightness_ratio", "abs_tightness_ratio_std",
        ],
    )
    prob_main = write_selected_csv(
        prob,
        prob_out["main"] / "linear_signature_probability_summary.csv",
        [
            "stress_model", "threshold", "n_replicates", "baseline_prob",
            "baseline_prob_std", "stress_prob", "stress_prob_std",
            "baseline_tail_count", "baseline_tail_count_std", "stress_tail_count",
            "stress_tail_count_std", "kappa_joint", "kappa_joint_std", "u_star_joint",
            "u_star_joint_std", "shifted_threshold_joint", "shifted_threshold_joint_std",
            "robust_prob_joint", "robust_prob_joint_std", "robust_tail_count_joint",
            "robust_tail_count_joint_std", "stress_over_baseline",
            "robust_over_stress_joint", "covered_joint", "degenerate_joint",
        ],
    )
    return exp_main, prob_main


def _option_config(levels: Iterable[int], train_size: int, alpha: float, ot_seed: int) -> ExperimentConfig:
    levels = list(levels)
    return ExperimentConfig(
        thresholds=(), signature_levels=levels,
        train_size=train_size, regression_alpha=alpha, optimizer_max_iter=5000,
        surrogate_type="post_link", link_type="lip_spline", link_fit_mode="posthoc",
        link_lipschitz=1., link_n_knots=32, link_knot_quantile_low=.01,
        link_knot_quantile_high=.99, signature_backend="iisignature",
        signature_type="geometric", signature_representation="signature",
        scaling_method="mad", scaling_floor=1e-8, ot_mode="joint",
        ot_grouping="coordinate", ot_block_norm="l2",
        ot_lambda_mode="preliminary_score", ot_lambda_floor=1e-8,
        ot_lambda_normalize=True, ot_sinkhorn_epsilon=1e-2,
        ot_sinkhorn_max_iter=400, ot_sinkhorn_tol=1e-9,
        ot_batch_size=512, ot_batches=8, ot_seed=ot_seed,
    )


def run_options(
    profile: str,
    root: Path,
    *,
    payoffs: Iterable[str] = ("asian_call", "lookback_call"),
    strikes: Iterable[float] | None = None,
    stress_sigmas: Iterable[float] | None = None,
    levels: Iterable[int] = (4,),
    alpha: float = 1.,
    output_name: str = "option_pricing",
    write_outputs: bool = True,
) -> pd.DataFrame:
    """Table 4 and the Appendix E.3 option ablations."""
    s = scale(profile)
    out = _dirs(root, output_name) if write_outputs else None
    strikes = list(option_strikes(profile) if strikes is None else strikes)
    sigmas = list(option_stresses(profile) if stress_sigmas is None else stress_sigmas)
    baseline = LogPathModel("risk_neutral_gbm_baseline", "gbm", mu=.02, sigma=.20)
    models = [
        LogPathModel(f"gbm_vol_{sigma:.2f}".replace(".", "p"), "gbm", mu=.02, sigma=sigma)
        for sigma in sigmas
    ]
    parts: list[pd.DataFrame] = []
    for payoff in payoffs:
        for strike in strikes:
            for rep in _replicates(profile):
                frame = run_path_option_expectation(
                    baseline_model=baseline,
                    stress_models=models,
                    experiment_cfg=_option_config(levels, s.train_paths, alpha, rep.ot_seed),
                    payoff_names=[payoff],
                    S0=100., strike=strike, T=1., r=.02, q=0.,
                    asian_include_initial=False,
                    delta_baseline_paths=s.delta_paths,
                    delta_true_paths=s.delta_paths,
                    delta_steps=s.steps,
                    delta_baseline_seed=rep.delta_baseline_seed,
                    delta_true_seed=rep.delta_true_seed,
                    fit_baseline_paths=s.fit_paths,
                    fit_steps=s.steps,
                    fit_baseline_seed=rep.fit_baseline_seed,
                    eval_baseline_paths=s.eval_paths,
                    eval_true_paths=s.eval_paths,
                    eval_steps=s.steps,
                    eval_baseline_seed=rep.eval_baseline_seed,
                    eval_true_seed=rep.eval_true_seed,
                    path_batch_size=s.batch_size,
                )
                frame["strike"] = strike
                frame["stress_sigma"] = frame.stress_model.map({m.name: m.sigma for m in models})
                frame["stress_sigma_shift"] = frame.stress_sigma - .20
                parts.append(annotate_replicate(frame, rep))
    raw = _concat(parts)
    group = ["payoff", "strike", "stress_model", "stress_sigma", "stress_sigma_shift", "k"]
    summary = summarize_replicates(raw, group_cols=group)
    columns = [
        *group, "n_replicates", "V_baseline_hard_eval",
        "V_baseline_hard_eval_std", "V_stress_hard", "V_stress_hard_std",
        "hard_true_gap", "hard_true_gap_std", "E_baseline_surrogate",
        "E_baseline_surrogate_std", "kappa_eff", "kappa_eff_std",
        "tightness_abs_gap_over_kappa", "tightness_abs_gap_over_kappa_std",
        "surrogate_upper", "surrogate_upper_std", "rmse_test", "rmse_test_std",
        "r2_test", "r2_test_std",
    ]
    if out is None:
        return selected_frame(summary, columns)
    write_csv(raw, out["diagnostics"] / "path_dependent_option_pricing_results_replicates.csv")
    return write_selected_csv(
        summary, out["main"] / "path_dependent_option_pricing_summary.csv", columns
    )


def _reserve_config(
    thresholds: Iterable[float], train_size: int, ot_seed: int, extrema: bool
) -> ExperimentConfig:
    return ExperimentConfig(
        thresholds=list(thresholds), signature_levels=[4],
        train_size=train_size, regression_alpha=3.,
        optimizer_max_iter=5000, surrogate_type="post_link", link_type="lip_spline",
        link_fit_mode="posthoc", link_lipschitz=1., link_n_knots=32,
        link_knot_quantile_low=.01, link_knot_quantile_high=.99,
        signature_backend="iisignature", signature_type="geometric",
        signature_representation="signature", event_sense=">=", scaling_method="mad",
        scaling_floor=1e-8, add_running_max_channel=extrema,
        add_running_min_channel=extrema, ot_mode="joint", ot_grouping="coordinate",
        ot_block_norm="l2", ot_lambda_mode="preliminary_score", ot_lambda_floor=1e-8,
        ot_lambda_power=1., ot_lambda_normalize=True, ot_sinkhorn_epsilon=1e-2,
        ot_sinkhorn_max_iter=400, ot_sinkhorn_tol=1e-9, ot_batch_size=512,
        ot_batches=8, ot_seed=ot_seed,
    )


def _reserve_case(
    profile: str,
    root: Path,
    *,
    folder: str,
    stem: str,
    task: str,
    task_type: str,
    functional: PathFunctional,
    thresholds: Iterable[float],
    extrema: bool,
    closed_form: bool = False,
    bm19: bool = False,
    write_raw: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    s, out, n_eval = scale(profile), _dirs(root, folder), reserve_eval_paths(profile)
    base_delta = BrownianReserveConfig(steps=s.steps, paths=s.delta_paths, batch_size=s.batch_size)
    true_delta = CramerLundbergConfig(steps=s.steps, paths=s.delta_paths, batch_size=s.batch_size)
    base_fit = BrownianReserveConfig(steps=s.steps, paths=s.fit_paths, batch_size=s.batch_size)
    base_eval = BrownianReserveConfig(steps=s.steps, paths=n_eval, batch_size=s.batch_size)
    true_eval = CramerLundbergConfig(steps=s.steps, paths=n_eval, batch_size=s.batch_size)
    channels = "t,R,running_min_R,running_max_R" if extrema else "t,R"
    parts = []
    for rep in _replicates(profile):
        result = run_reserve_benchmark(
            functional=functional,
            delta_reserve_cfg=replace(base_delta, seed=rep.delta_baseline_seed),
            delta_true_cfg=replace(true_delta, seed=rep.delta_true_seed),
            fit_reserve_cfg=replace(base_fit, seed=rep.fit_baseline_seed),
            eval_reserve_cfg=replace(base_eval, seed=rep.eval_baseline_seed),
            eval_true_cfg=replace(true_eval, seed=rep.eval_true_seed),
            experiment_cfg=_reserve_config(thresholds, s.train_paths, rep.ot_seed, extrema),
            constant_boundary_baseline_closed_form=closed_form,
            attach_bm19_reference=bm19,
        )
        result["task"], result["task_type"] = task, task_type
        result["signature_input_channels"] = channels
        result["add_running_min_channel"] = extrema
        result["add_running_max_channel"] = extrema
        result["common_extrema_aware_lift"] = extrema
        parts.append(annotate_replicate(result, rep))
    raw = _concat(parts)
    if write_raw:
        write_csv(raw, out["diagnostics"] / f"{stem}_results_replicates.csv")
    summary = summarize_replicates(
        raw, group_cols=["task", "signature_input_channels", "k", "u"]
    )
    summary["baseline_over_true"] = summary.psi_B / summary.psi_true.replace(0., np.nan)
    summary["sigot_over_true"] = summary.psi_sig_rob / summary.psi_true.replace(0., np.nan)
    return raw, summary


RESERVE_COLUMNS = [
    "task", "task_type", "signature_input_channels", "u", "k", "n_replicates",
    "psi_B", "psi_B_std", "psi_true", "psi_true_std", "psi_sig_rob",
    "psi_sig_rob_std", "baseline_over_true",
    "sigot_over_true", "kappa", "kappa_std", "shift_star", "shift_star_std",
    "r2_test", "r2_test_std", "rmse_test", "rmse_test_std", "n_active", "n_active_std",
    "add_running_min_channel", "add_running_max_channel", "common_extrema_aware_lift",
]


def run_reserve_family(profile: str, root: Path) -> pd.DataFrame:
    """Table 6: four reserve-path functionals."""
    boundary = make_boundary_spec(boundary_type="sine_hump", amplitude=-20.)
    tasks = [
        ("reserve_constant", "reserve_constant_boundary", "constant_boundary",
         "constant_boundary_family", ruin_depth),
        ("reserve_sine", "reserve_sine_vulnerability", "sine_vulnerability_am20",
         "sinusoidal_vulnerability",
         make_curved_boundary_ruin_depth_functional(boundary, name="sine_vulnerability_am20")),
        ("reserve_early", "reserve_early_shortfall", "early_shortfall_linear_rho1",
         "early_shortfall", make_early_shortfall_functional(rho=1., weight_type="linear")),
        ("reserve_drawdown", "reserve_drawdown", "max_drawdown", "drawdown",
         make_drawdown_functional()),
    ]
    summaries = []
    for folder, stem, task, task_type, functional in tasks:
        _, summary = _reserve_case(
            profile, root, folder=folder, stem=stem, task=task, task_type=task_type,
            functional=functional, thresholds=reserve_thresholds(profile), extrema=True,
        )
        summaries.append(summary)
    combined = _concat(summaries).sort_values(["task", "u"])
    out = _dirs(root, "reserve_family")
    return write_selected_csv(
        combined, out["main"] / "reserve_path_risk_family_summary.csv", RESERVE_COLUMNS
    )


def _ratio(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(b) and b != 0 else np.nan


def _wide_bm19(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (k, u), group in summary.groupby(["k", "u"], sort=True):
        first = group.iloc[0]
        row = {name: first.get(name, np.nan) for name in
               ("n_replicates", "psi_B_BM19", "psi_B", "psi_true", "u_tilde_j1", "psi_rob_j1")}
        row.update({"u": u, "k": int(k)})
        for _, record in group.iterrows():
            case = record.representation_case
            for name in ("psi_sig_rob", "kappa", "shift_star", "r2_test", "n_active"):
                row[f"{name}_{case}"] = record.get(name, np.nan)
        row["baseline_over_true"] = _ratio(row["psi_B"], row["psi_true"])
        row["bm19_j1_over_true"] = _ratio(row["psi_rob_j1"], row["psi_true"])
        for case in ("plain", "extrema"):
            row[f"sigot_{case}_over_bm19_j1"] = _ratio(
                row[f"psi_sig_rob_{case}"], row["psi_rob_j1"]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["k", "u"])


def run_bm19(profile: str, root: Path) -> pd.DataFrame:
    """Table 7: BM19/J1 versus plain and extrema-aware Sig-OT."""
    raw_parts = []
    for case, extrema in (("plain", False), ("extrema", True)):
        raw, _ = _reserve_case(
            profile,
            root,
            folder="bm19",
            stem=f"bm19_{case}_temporary",
            task="bm19_ruin_benchmark",
            task_type="constant_boundary_bm19",
            functional=ruin_depth,
            thresholds=bm19_thresholds(profile),
            extrema=extrema,
            closed_form=True,
            bm19=True,
            write_raw=False,
        )
        raw["representation_case"] = case
        raw_parts.append(raw)
    raw = _concat(raw_parts)
    out = _dirs(root, "bm19")
    write_csv(raw, out["diagnostics"] / "bm19_ruin_benchmark_results_replicates.csv")
    summary = summarize_replicates(
        raw,
        group_cols=["representation_case", "signature_input_channels", "k", "u"],
    )
    wide = _wide_bm19(summary)
    return write_selected_csv(
        wide,
        out["main"] / "bm19_ruin_benchmark_summary.csv",
        [
            "u", "k", "n_replicates", "psi_B_BM19", "psi_B", "psi_true",
            "u_tilde_j1", "psi_rob_j1", "psi_sig_rob_plain", "psi_sig_rob_extrema",
            "baseline_over_true", "bm19_j1_over_true", "sigot_plain_over_bm19_j1",
            "sigot_extrema_over_bm19_j1", "kappa_plain", "kappa_extrema",
            "shift_star_plain", "shift_star_extrema", "r2_test_plain", "r2_test_extrema",
            "n_active_plain", "n_active_extrema",
        ],
    )
