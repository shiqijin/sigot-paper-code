"""Aligned random-seed replicates and mean/standard-deviation summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SeedReplicate:
    replicate_id: int
    delta_baseline_seed: int
    delta_true_seed: int
    fit_baseline_seed: int
    eval_baseline_seed: int
    eval_true_seed: int
    ot_seed: int

    def as_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class FixedScoreSeedReplicate:
    replicate_id: int
    delta_baseline_seed: int
    delta_stress_seed: int
    eval_baseline_seed: int
    eval_stress_seed: int
    ot_seed: int

    def as_dict(self) -> dict[str, int]:
        return {key: int(value) for key, value in asdict(self).items()}


def _broadcast(values: Sequence[int], count: int, name: str) -> list[int]:
    values = [int(value) for value in values]
    if len(values) == 1:
        return values * count
    if len(values) == count:
        return values
    raise ValueError(f"{name} must contain one or {count} aligned values")


def make_seed_replicates(
    *,
    delta_baseline_seeds: Sequence[int],
    delta_true_seeds: Sequence[int],
    fit_baseline_seeds: Sequence[int],
    eval_baseline_seeds: Sequence[int],
    eval_true_seeds: Sequence[int],
    ot_seeds: Sequence[int],
) -> list[SeedReplicate]:
    count = max(map(len, (
        delta_baseline_seeds, delta_true_seeds, fit_baseline_seeds,
        eval_baseline_seeds, eval_true_seeds, ot_seeds,
    )))
    columns = [
        _broadcast(delta_baseline_seeds, count, "delta baseline seeds"),
        _broadcast(delta_true_seeds, count, "delta true seeds"),
        _broadcast(fit_baseline_seeds, count, "fit baseline seeds"),
        _broadcast(eval_baseline_seeds, count, "evaluation baseline seeds"),
        _broadcast(eval_true_seeds, count, "evaluation true seeds"),
        _broadcast(ot_seeds, count, "OT seeds"),
    ]
    return [SeedReplicate(index, *(column[index] for column in columns))
            for index in range(count)]


def make_fixed_score_seed_replicates(
    *,
    delta_baseline_seeds: Sequence[int],
    delta_stress_seeds: Sequence[int],
    eval_baseline_seeds: Sequence[int],
    eval_stress_seeds: Sequence[int],
    ot_seeds: Sequence[int],
) -> list[FixedScoreSeedReplicate]:
    count = max(map(len, (
        delta_baseline_seeds, delta_stress_seeds, eval_baseline_seeds,
        eval_stress_seeds, ot_seeds,
    )))
    columns = [
        _broadcast(delta_baseline_seeds, count, "delta baseline seeds"),
        _broadcast(delta_stress_seeds, count, "delta stress seeds"),
        _broadcast(eval_baseline_seeds, count, "evaluation baseline seeds"),
        _broadcast(eval_stress_seeds, count, "evaluation stress seeds"),
        _broadcast(ot_seeds, count, "OT seeds"),
    ]
    return [FixedScoreSeedReplicate(index, *(column[index] for column in columns))
            for index in range(count)]


def annotate_replicate(frame: pd.DataFrame, replicate: Any) -> pd.DataFrame:
    result = frame.copy()
    for key, value in replicate.as_dict().items():
        result[key] = value
    return result


def summarize_replicates(
    frame: pd.DataFrame, *, group_cols: Sequence[str]
) -> pd.DataFrame:
    """Aggregate aligned replicates while retaining constant descriptive fields."""
    if frame.empty:
        return frame.copy()
    seed_cols = {
        "replicate_id", "delta_baseline_seed", "delta_true_seed",
        "delta_stress_seed", "fit_baseline_seed", "eval_baseline_seed",
        "eval_true_seed", "eval_stress_seed", "ot_seed",
    }
    rows: list[dict[str, Any]] = []
    for _, group in frame.groupby(list(group_cols), dropna=False, sort=True):
        row = {column: group.iloc[0][column] for column in group_cols}
        row["n_replicates"] = (
            int(group.replicate_id.nunique())
            if "replicate_id" in group
            else len(group)
        )
        for column in group.columns:
            if column in group_cols or column in seed_cols:
                continue
            series = group[column]
            if pd.api.types.is_bool_dtype(series):
                values = series.drop_duplicates()
                row[column] = bool(values.iloc[0]) if len(values) == 1 else "mixed"
            elif pd.api.types.is_numeric_dtype(series):
                values = pd.to_numeric(series, errors="coerce").dropna().astype(float)
                row[column] = float(values.mean()) if len(values) else np.nan
                row[f"{column}_std"] = (
                    float(values.std(ddof=1)) if len(values) > 1 else 0.0
                )
            else:
                values = series.dropna().astype(str).unique()
                row[column] = values[0] if len(values) == 1 else (
                    None if len(values) == 0 else "mixed"
                )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(list(group_cols)).reset_index(drop=True)
