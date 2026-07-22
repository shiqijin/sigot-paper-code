"""Execution profiles for the experiments in Sections 5 and Appendix E."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Scale:
    delta_paths: int
    fit_paths: int
    train_paths: int
    eval_paths: int
    steps: int
    batch_size: int
    replicates: int
    ot_batch_size: int
    ot_batches: int
    ot_max_iter: int
    ot_tol: float


PAPER = Scale(20_000, 20_000, 16_000, 500_000, 2_000, 256, 5, 512, 16, 400, 1e-9)
SMOKE = Scale(48, 64, 48, 128, 16, 16, 1, 16, 1, 40, 1e-7)


def scale(profile: str) -> Scale:
    try:
        return {"paper": PAPER, "smoke": SMOKE}[str(profile)]
    except KeyError as exc:
        raise ValueError("profile must be 'paper' or 'smoke'") from exc


def seeds(profile: str, start: int) -> list[int]:
    return [int(start) + i for i in range(scale(profile).replicates)]


def levels(profile: str, *, appendix: bool = False) -> list[int]:
    if appendix:
        return [2, 3, 4, 5, 6] if profile == "paper" else [2, 4]
    return [4]


def reserve_thresholds(profile: str) -> list[float]:
    return [100, 120, 140, 160, 180, 200] if profile == "paper" else [120]


def bm19_thresholds(profile: str) -> list[float]:
    return [50, 100, 150, 200, 250] if profile == "paper" else [100]


def option_strikes(profile: str) -> list[float]:
    return [95, 100, 105] if profile == "paper" else [100]


def option_stresses(profile: str, *, ablation: bool = False) -> list[float]:
    if ablation:
        return [0.15, 0.25]
    return [0.10, 0.15, 0.25, 0.30] if profile == "paper" else [0.15, 0.25]


def probability_eval_paths(profile: str) -> int:
    return 3_000_000 if profile == "paper" else scale(profile).eval_paths


def reserve_eval_paths(profile: str) -> int:
    return 5_000_000 if profile == "paper" else scale(profile).eval_paths


E1_GRIDS = {
    "paper": {
        "gbm_drift": [-.10, -.075, -.05, -.025, 0, .025, .05, .075, .10, .125, .15, .175, .20],
        "gbm_volatility": [.08, .10, .12, .14, .16, .18, .20, .22, .24, .26, .28, .30, .32],
        "merton_jump_intensity": [0, .10, .20, .35, .50, .75, 1, 1.25, 1.50, 1.75, 2],
        "heston_vol_of_vol": [0, .10, .20, .30, .40, .50, .65, .80, 1],
        "heston_self_sigma_v": [0, .10, .20, .35, .50, .65, .80, 1, 1.20],
        "cev_beta": [.40, .50, .60, .70, .80, .90, 1, 1.10, 1.20, 1.30, 1.50],
    },
    "smoke": {
        "gbm_drift": [-.10, .05, .20],
        "gbm_volatility": [.08, .20, .32],
        "merton_jump_intensity": [0, .75, 2],
        "heston_vol_of_vol": [0, .40, 1],
        "heston_self_sigma_v": [0, .50, 1.20],
        "cev_beta": [.40, 1, 1.50],
    },
}


def e1_grids(profile: str) -> dict[str, list[float]]:
    scale(profile)
    return E1_GRIDS[profile]
