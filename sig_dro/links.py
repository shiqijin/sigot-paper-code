from __future__ import annotations

"""Monotone one-Lipschitz spline used by the learned paper surrogates."""

from dataclasses import dataclass
from typing import Any

import numpy as np


Array = np.ndarray


@dataclass(frozen=True)
class MonotoneLink:
    link_type: str
    params: dict[str, Any]
    lip_const: float = 1.0
    is_monotone: bool = True

    def _spline_parameters(self) -> tuple[Array, Array, float]:
        if self.link_type != "lip_spline":
            raise ValueError("The paper implementation uses the lip_spline post-link.")
        return (
            np.asarray(self.params["knots"], dtype=float),
            np.asarray(self.params["slopes"], dtype=float),
            float(self.params.get("intercept", 0.0)),
        )

    def predict(self, scores: Array) -> Array:
        knots, slopes, intercept = self._spline_parameters()
        values = np.asarray(scores, dtype=float)
        return (intercept + _spline_basis(values, knots) @ slopes).reshape(values.shape)

    def derivative(self, scores: Array) -> Array:
        knots, slopes, _ = self._spline_parameters()
        return _spline_derivative(scores, knots, slopes)

    def inverse_threshold(self, threshold: float) -> float:
        """Return the left inverse inf{t:g(t)>=threshold}."""

        target = float(threshold)
        knots, _, _ = self._spline_parameters()
        center = float(np.median(knots))
        lo, hi = center - 1.0, center + 1.0
        for _ in range(80):
            if float(self.predict(np.array([lo]))[0]) <= target:
                break
            hi = lo
            lo = center - 2.0 * (center - lo + 1.0)
        else:
            return float("-inf")
        for _ in range(80):
            if float(self.predict(np.array([hi]))[0]) >= target:
                break
            lo = hi
            hi = center + 2.0 * (hi - center + 1.0)
        else:
            return float("inf")
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            if float(self.predict(np.array([mid]))[0]) >= target:
                hi = mid
            else:
                lo = mid
        return float(hi)


def _sigmoid(x: Array) -> Array:
    values = np.asarray(x, dtype=float)
    out = np.empty_like(values, dtype=float)
    positive = values >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    out[~positive] = exp_values / (1.0 + exp_values)
    return out


def _spline_basis(scores: Array, knots: Array) -> Array:
    """Integral basis for a monotone PWL spline with linear extrapolation."""

    values = np.asarray(scores, dtype=float).reshape(-1)
    knots = np.asarray(knots, dtype=float).reshape(-1)
    if knots.size < 2:
        raise ValueError("lip_spline requires at least two knots.")
    widths = np.diff(knots)
    if np.any(widths <= 0.0):
        raise ValueError("lip_spline knots must be strictly increasing.")
    basis = np.zeros((values.size, knots.size - 1), dtype=float)
    below = values < knots[0]
    basis[below, 0] = values[below] - knots[0]
    for index, width in enumerate(widths):
        basis[:, index] += np.minimum(np.maximum(values - knots[index], 0.0), width)
    above = values > knots[-1]
    basis[above, -1] += values[above] - knots[-1]
    return basis


def _spline_derivative(scores: Array, knots: Array, slopes: Array) -> Array:
    values = np.asarray(scores, dtype=float)
    flat = values.reshape(-1)
    knots = np.asarray(knots, dtype=float).reshape(-1)
    slopes = np.asarray(slopes, dtype=float).reshape(-1)
    indices = np.searchsorted(knots[1:], flat, side="right")
    indices = np.clip(indices, 0, slopes.size - 1)
    return slopes[indices].reshape(values.shape)
