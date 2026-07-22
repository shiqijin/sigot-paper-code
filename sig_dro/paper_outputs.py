from __future__ import annotations

"""Small CSV/output-directory helpers for the paper experiments."""

from pathlib import Path
from typing import Sequence

import pandas as pd


def prepare_output_tree(root: str | Path) -> dict[str, Path]:
    base = Path(root)
    paths = {
        "root": base,
        "main": base / "main_results",
        "diagnostics": base / "diagnostics",
        "figures": base / "figures",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def selected_frame(df: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise KeyError(f"Missing required output columns: {missing}")
    return df.loc[:, list(columns)].copy()


def write_selected_csv(df: pd.DataFrame, path: str | Path, columns: Sequence[str]) -> pd.DataFrame:
    out = selected_frame(df, columns)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    return out


def write_csv(df: pd.DataFrame, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
