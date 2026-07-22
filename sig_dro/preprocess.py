from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


@dataclass(frozen=True)
class ComponentScaler:
    scale: Array
    method: str
    floor: float

    def transform(self, features: Array) -> Array:
        features = np.asarray(features, dtype=float)
        if features.shape[-1] != self.scale.shape[0]:
            raise ValueError(
                f"Expected trailing dimension {self.scale.shape[0]}, got {features.shape[-1]}."
            )
        return features / self.scale


def _robust_scale_1d(x: Array, method: str, floor: float) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float(floor)
    if method != "mad":
        raise ValueError("The paper implementation uses MAD scaling.")
    med = float(np.median(x))
    val = 1.4826 * float(np.median(np.abs(x - med)))
    if not np.isfinite(val) or val < floor:
        val = float(max(floor, np.std(x, ddof=0)))
    return float(max(val, floor))


def fit_component_scaler(
    features: Array,
    train_size: int,
    method: str = "mad",
    floor: float = 1e-8,
) -> ComponentScaler:
    features = np.asarray(features, dtype=float)
    if features.ndim != 2:
        raise ValueError(f"features must be 2D, got {features.shape}.")
    if not (0 < train_size <= features.shape[0]):
        raise ValueError("train_size must lie between 1 and len(features).")
    train = features[:train_size]
    scales = np.array([_robust_scale_1d(train[:, j], method=method, floor=floor) for j in range(train.shape[1])], dtype=float)
    return ComponentScaler(scale=scales, method=method, floor=float(floor))
