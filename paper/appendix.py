"""Appendix E geometry experiments and design ablations."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")

from sig_dro.affine_payoff import LogPathModel
from sig_dro.geometry import (
    ParametricSweepSpec,
    compute_log_path_feature_bank,
    pair_decoupled_budget,
    run_parametric_decoupled_geometry,
    summarize_geometry_replicates,
)
from sig_dro.paper_outputs import prepare_output_tree, write_csv

from .section5 import run_options
from .settings import e1_grids, levels, option_strikes, option_stresses, scale, seeds


LABELS = {
    "baseline_gbm": "GBM", "gbm_vol_mismatch_022": "GBM-vol",
    "merton_moderate_jump": "Merton", "heston_moderate_leverage": "Heston",
    "cev_moderate_localvol": "CEV", "mild_regime_mixture": "Regime",
}
MODEL_ORDER = ["baseline_gbm", "gbm_vol_mismatch_022", "merton_moderate_jump",
               "cev_moderate_localvol", "heston_moderate_leverage", "mild_regime_mixture"]
SWEEP_LABELS = {
    "gbm_drift": ("GBM drift", r"$\mu$"),
    "gbm_volatility": ("GBM volatility", r"$\sigma$"),
    "merton_jump_intensity": ("Merton jump intensity", r"$\lambda$"),
    "heston_vol_of_vol": ("GBM--Heston vol-of-vol", r"$\xi_v$"),
    "heston_self_sigma_v": ("Heston self vol-of-vol", r"$\sigma_v$"),
    "cev_beta": ("CEV/local-vol elasticity", r"$\beta$"),
}


def _baseline(name: str = "baseline_gbm") -> LogPathModel:
    return LogPathModel(name, "gbm", mu=.05, sigma=.20)


def _heston(name: str) -> LogPathModel:
    return LogPathModel(name, "heston", mu=.05, v0=.04, theta_v=.04, kappa_v=3., xi_v=.35, rho=-.5)


def _e1_groups(profile: str):
    grid = e1_grids(profile)
    gbm = _baseline()
    heston = _heston("heston_base")
    gbm_specs = [
        ParametricSweepSpec("gbm_drift", "gbm_basic", "gbm", gbm, "mu", tuple(grid["gbm_drift"]), .05),
        ParametricSweepSpec("gbm_volatility", "gbm_basic", "gbm", gbm, "sigma", tuple(grid["gbm_volatility"]), .20),
        ParametricSweepSpec("merton_jump_intensity", "jump", "merton",
            LogPathModel("merton_jump_intensity_base", "merton", mu=.05, sigma=.20,
                         jump_intensity=0., jump_mean=-.12, jump_sd=.08),
            "jump_intensity", tuple(grid["merton_jump_intensity"]), 0.),
        ParametricSweepSpec("heston_vol_of_vol", "heston", "heston", heston,
                            "xi_v", tuple(grid["heston_vol_of_vol"]), 0.),
    ]
    self_heston = _heston("heston_self_baseline")
    cev = LogPathModel("cev_self_baseline", "cev", mu=.05, sigma=.20, beta=1.)
    return [
        (gbm, gbm_specs),
        (self_heston, [ParametricSweepSpec("heston_self_sigma_v", "heston_self", "heston",
                                           self_heston, "xi_v", tuple(grid["heston_self_sigma_v"]), .35)]),
        (cev, [ParametricSweepSpec("cev_beta", "localvol", "cev", cev, "beta",
                                   tuple(grid["cev_beta"]), 1.)]),
    ]


def _e1_table(summary: pd.DataFrame, normalized: bool = False) -> pd.DataFrame:
    out = summary.copy()
    out["sweep_order"] = out.sweep_name.map({name: i for i, name in enumerate(SWEEP_LABELS)})
    out["sweep"] = out.sweep_name.map(lambda x: SWEEP_LABELS[x][0])
    out["parameter_symbol"] = out.sweep_name.map(lambda x: SWEEP_LABELS[x][1])
    out["theta"], out["theta_0"] = out.parameter_value, out.baseline_parameter_value
    out["baseline_law"] = np.select(
        [out.baseline_model.eq("heston_self_baseline"), out.baseline_model.eq("cev_self_baseline")],
        ["Heston baseline", "CEV baseline"], default="GBM baseline")
    out["comparison_family"], out["k"] = out.model_family, out.signature_level
    out["Delta_k"], out["se_Delta_k"] = out.delta_l1, out.delta_l1_std
    cols = ["sweep_order", "sweep_name", "sweep", "parameter_symbol", "parameter_name", "theta", "theta_0",
            "baseline_law", "comparison_family", "k", "Delta_k", "se_Delta_k"]
    if normalized:
        out["Delta_k_sweep_max"] = out.groupby(["sweep_name", "signature_level"]).delta_l1.transform("max")
        out["widetilde_Delta_k"] = np.where(out.Delta_k_sweep_max > 0, out.Delta_k / out.Delta_k_sweep_max, 0.)
        cols += ["Delta_k_sweep_max", "widetilde_Delta_k"]
    return out[cols].sort_values(["sweep_order", "k", "theta"]).reset_index(drop=True)


def _plot_e1(table: pd.DataFrame, path: Path, value: str) -> None:
    import matplotlib.pyplot as plt
    normalized = value.startswith("widetilde")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2), sharey=normalized)
    handles = []
    for ax, (name, (title, symbol)) in zip(axes.ravel(), SWEEP_LABELS.items()):
        part = table[table.sweep_name == name]
        for k, group in part.groupby("k"):
            line, = ax.plot(group.theta, group[value], marker="o", ms=4, label=f"k={int(k)}")
            if len(handles) < part.k.nunique():
                handles.append(line)
        ax.set_title(title); ax.set_xlabel(symbol)
        ax.set_ylabel(r"$\widetilde{\Delta}_k(P_0,P_\theta)$" if normalized else r"$\Delta_k(P_0,P_\theta)$")
        ax.grid(alpha=.25)
    if len(handles) > 1:
        fig.legend(handles=handles, labels=[line.get_label() for line in handles],
                   loc="lower center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, .055 if len(handles) > 1 else 0, 1, 1)); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def run_e1(profile: str, root: Path) -> pd.DataFrame:
    """Reproduce all six parametric coverage sweeps and both E.1 figures."""
    s, ks, out = scale(profile), levels(profile, appendix=True), prepare_output_tree(root / "appendix_e1")
    n = 10_000 if profile == "paper" else 32
    parts = []
    for rep, (base_seed, stress_seed) in enumerate(zip(seeds(profile, 123), seeds(profile, 1123))):
        for baseline, specs in _e1_groups(profile):
            summary = run_parametric_decoupled_geometry(
                baseline_model=baseline, sweep_specs=specs, paths=n, steps=s.steps, T=1.,
                baseline_seed=base_seed, stress_seed=stress_seed, levels=ks,
                signature_backend="iisignature", signature_type="geometric",
                signature_representation="signature", scaling_method="mad", scaling_floor=1e-8,
                scaling_train_size=n, replicate_id=rep, sweep_seed_mode="matched",
            )
            summary["baseline_model"], summary["baseline_model_kind"] = baseline.name, baseline.kind
            parts.append(summary)
    raw = pd.concat(parts, ignore_index=True)
    group = ["section", "baseline_model", "baseline_model_kind", "sweep_set", "sweep_name", "model_family",
             "parameter_name", "parameter_value", "baseline_parameter_value", "baseline_relation", "signature_level"]
    summary = summarize_geometry_replicates(raw, group_cols=group)
    main, normalized = _e1_table(summary), _e1_table(summary, normalized=True)
    write_csv(raw, out["diagnostics"] / "geometry_parametric_sweep_summary_replicates.csv")
    write_csv(main, out["main"] / "appendix_e1_parametric_delta.csv")
    write_csv(main[main.k == 4], out["main"] / "appendix_e1_parametric_delta_k4.csv")
    write_csv(normalized, out["main"] / "appendix_e1_parametric_delta_normalized.csv")
    _plot_e1(main[main.k == 4], out["figures"] / "e1_parametric_raw_k4.png", "Delta_k")
    _plot_e1(normalized, out["figures"] / "e1_parametric_normalized_all_k.png", "widetilde_Delta_k")
    return main


def _model_library() -> list[LogPathModel]:
    return [
        _baseline(),
        LogPathModel("gbm_vol_mismatch_022", "gbm", mu=.05, sigma=.22),
        LogPathModel("merton_moderate_jump", "merton", mu=.05, sigma=.1865,
                     jump_intensity=.35, jump_mean=-.10, jump_sd=.07),
        _heston("heston_moderate_leverage"),
        LogPathModel("cev_moderate_localvol", "cev", mu=.05, sigma=.20, beta=.60),
        LogPathModel("mild_regime_mixture", "mixture", mixture_prob_crash=.10,
                     normal_mu=.05, normal_sigma=.20, crash_mu=-.05, crash_sigma=.30),
    ]


def _pair_matrix(summary: pd.DataFrame, channel: str, k: int, value: str) -> pd.DataFrame:
    mat = pd.DataFrame(np.zeros((6, 6)), index=MODEL_ORDER, columns=MODEL_ORDER)
    for _, row in summary[(summary.channel_spec == channel) & (summary.signature_level == k)].iterrows():
        mat.loc[row.pair_left, row.pair_right] = mat.loc[row.pair_right, row.pair_left] = row[value]
    mat = mat.rename(index=LABELS, columns=LABELS); mat.index.name = "model"
    return mat


def _plot_e2(matrix: pd.DataFrame, path: Path, title: str) -> None:
    import matplotlib.pyplot as plt
    values = matrix.to_numpy(float); mask = np.eye(len(matrix), dtype=bool)
    fig, ax = plt.subplots(figsize=(7.2, 6.0)); cmap = plt.get_cmap("viridis").copy(); cmap.set_bad("white")
    image = ax.imshow(np.ma.array(values, mask=mask), cmap=cmap)
    labels = list(matrix.columns)
    ax.set_xticks(range(6)); ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(range(6)); ax.set_yticklabels(labels)
    ax.set_title(title)
    threshold = np.nanmin(values[~mask]) + .55 * np.ptp(values[~mask])
    for i in range(6):
        for j in range(6):
            text = "—" if i == j else f"{values[i, j]:.1f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=8,
                    color="white" if i != j and values[i, j] >= threshold else "black")
    fig.colorbar(image, ax=ax, fraction=.046, pad=.04); fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=220, bbox_inches="tight"); plt.close(fig)


def run_e2(profile: str, root: Path) -> pd.DataFrame:
    """Reproduce the two model-library pairwise coverage matrices in E.2."""
    s, ks, out = scale(profile), ([2, 3, 4] if profile == "paper" else [2, 4]), prepare_output_tree(root / "appendix_e2")
    n, models = (10_000 if profile == "paper" else 32), _model_library()
    parts = []
    for channel in ("time_price", "time_price_rv"):
        for rep, (base_seed, stress_seed) in enumerate(zip(seeds(profile, 123), seeds(profile, 1123))):
            banks = {}
            for index, model in enumerate(models):
                seed = base_seed if index == 0 else stress_seed + 1000 * index
                banks[model.name] = compute_log_path_feature_bank(
                    model=model, paths=n, steps=s.steps, T=1., seed=seed, levels=ks,
                    signature_backend="iisignature", signature_type="geometric",
                    signature_representation="signature", channel_spec=channel)
            for i, left in enumerate(MODEL_ORDER):
                for right in MODEL_ORDER[i + 1:]:
                    for k in ks:
                        budget = pair_decoupled_budget(
                            baseline_bank=banks[left], stress_bank=banks[right], level=k,
                            scaling_method="mad", scaling_floor=1e-8, scaling_train_size=n,
                            scaling_reference_bank=banks["baseline_gbm"],
                            context={"section": "5.1.3", "diagnostic_kind": "pairwise_model_library",
                                     "channel_spec": channel, "pair_left": left, "pair_right": right,
                                     "replicate_id": rep})
                        parts.append(budget)
    raw = pd.concat(parts, ignore_index=True)
    group = ["section", "diagnostic_kind", "channel_spec", "pair_left", "pair_right", "signature_level"]
    summary = summarize_geometry_replicates(raw, group_cols=group)
    order = {LABELS[name]: index for index, name in enumerate(MODEL_ORDER)}
    rows = []
    for _, row in summary.iterrows():
        left, right = LABELS[row.pair_left], LABELS[row.pair_right]
        if order[right] < order[left]:
            left, right = right, left
        rows.append({"feature_path": r"$(t,X_t,\mathrm{RV}_t)$" if row.channel_spec == "time_price_rv" else r"$(t,X_t)$",
                     "k": row.signature_level, "P_i": left, "P_j": right,
                     "D_ij^(k)": row.delta_l1, "se_D_ij^(k)": row.delta_l1_std})
    long = pd.DataFrame(rows)
    long["_i"] = long.P_i.map(order); long["_j"] = long.P_j.map(order)
    long = long.sort_values(["feature_path", "k", "_i", "_j"]).drop(columns=["_i", "_j"])
    write_csv(raw, out["diagnostics"] / "pairwise_replicates.csv")
    write_csv(long, out["main"] / "appendix_e2_pairwise_radii_long.csv")
    for channel in ("time_price", "time_price_rv"):
        for k in ks:
            for value, suffix in (("delta_l1", ""), ("delta_l1_std", "_se")):
                matrix = _pair_matrix(summary, channel, k, value).round(1)
                write_csv(matrix.reset_index(), out["main"] / f"appendix_e2_pairwise_D_{channel}_k{k}{suffix}.csv")
        matrix = _pair_matrix(summary, channel, 4, "delta_l1")
        title = r"$(t,X_t,\mathrm{RV}_t)$, k=4" if channel.endswith("rv") else r"$(t,X_t)$, k=4"
        _plot_e2(matrix, out["figures"] / f"e2_pairwise_{channel}_k4.png", title)
    return long


def run_e3(profile: str, root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce the lookback penalty and signature-depth ablations in E.3."""
    strikes, stresses = option_strikes(profile), option_stresses(profile, ablation=True)
    alphas = [0., 1., 2., 3.] if profile == "paper" else [0., 1.]
    depths = [2, 3, 4, 5, 6] if profile == "paper" else [2, 4]
    tables = []
    for alpha in alphas:
        frame = run_options(profile, root, payoffs=["lookback_call"], strikes=strikes,
                            stress_sigmas=stresses, levels=[4], alpha=alpha,
                            output_name=f"appendix_e3/runs/alpha_{alpha:g}", write_outputs=False)
        frame["setting"], frame["ablation_group"] = f"alpha={alpha:g}", "alpha"
        tables.append(frame)
    for k in depths:
        frame = run_options(profile, root, payoffs=["lookback_call"], strikes=strikes,
                            stress_sigmas=stresses, levels=[k], alpha=1.,
                            output_name=f"appendix_e3/runs/k_{k}", write_outputs=False)
        frame["setting"], frame["ablation_group"] = f"k={k}", "depth"
        tables.append(frame)
    raw = pd.concat(tables, ignore_index=True)
    compact = pd.DataFrame({
        "setting": raw.setting, "stress_sigma": raw.stress_sigma,
        "stress_sigma_shift": raw.stress_sigma_shift, "K": raw.strike,
        "n_replicates": raw.n_replicates, "Base": raw.V_baseline_hard_eval,
        "Stress": raw.V_stress_hard, "Shift": raw.hard_true_gap,
        "kappa_hat": raw.kappa_eff, "abs_Shift_over_kappa": raw.tightness_abs_gap_over_kappa,
        "Fit_R2": raw.r2_test, "ablation_group": raw.ablation_group,
    })
    collected = root / "appendix_e3" / "main_results"; collected.mkdir(parents=True, exist_ok=True)
    alpha = compact[compact.ablation_group == "alpha"].drop(columns="ablation_group")
    depth = compact[compact.ablation_group == "depth"].drop(columns="ablation_group")
    write_csv(alpha, collected / "appendix_e3_alpha_ablation.csv")
    write_csv(depth, collected / "appendix_e3_signature_depth_ablation.csv")
    return alpha, depth
