from __future__ import annotations

"""Shared budget bookkeeping for the paper experiment pipelines."""

import numpy as np

from .blocks import build_block_spec
from .configs import ExperimentConfig
from .robust import effective_budget_blocks


Array = np.ndarray


def _block_spec_for_level(
    features_by_level: dict[int, Array],
    level: int,
    config: ExperimentConfig,
):
    feature_dim = int(features_by_level[level].shape[1])
    cumulative_dims = [
        int(features_by_level[j].shape[1])
        for j in range(1, int(level) + 1)
    ]
    return build_block_spec(
        feature_dim=feature_dim,
        grouping=config.ot_grouping,
        norm=config.ot_block_norm,
        cumulative_dims=cumulative_dims,
    )


def _budget_payload(
    model,
    config: ExperimentConfig,
    block_spec,
    delta_override: Array,
):
    if config.ot_mode != "joint":
        raise ValueError("The paper surrogate experiments use joint OT.")
    delta_blocks = np.asarray(delta_override, dtype=float).reshape(-1)
    if delta_blocks.shape[0] < block_spec.n_blocks:
        raise ValueError("delta_override shorter than number of blocks.")
    delta_blocks = delta_blocks[: block_spec.n_blocks]
    kappa, resolved_delta, alpha_vec, _ = effective_budget_blocks(
        model=model,
        block_spec=block_spec,
        delta_vec=delta_blocks,
    )
    return {
        "kappa": float(kappa),
        "resolved_delta": resolved_delta,
        "alpha": alpha_vec,
        "delta_scalar": None,
        "budget_kind": "block",
        "budget_dim": int(block_spec.n_blocks),
    }


def _empirical_event_probability(
    targets: Array,
    threshold: float,
    event_sense: str,
) -> float:
    values = np.asarray(targets, dtype=float)
    if event_sense == ">=":
        return float(np.mean(values >= threshold))
    if event_sense == "<=":
        return float(np.mean(values <= threshold))
    raise ValueError("event_sense must be '>=' or '<='.")
