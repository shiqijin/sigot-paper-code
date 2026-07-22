from __future__ import annotations

"""Sine boundary used by the reserve vulnerability experiment."""

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np


Array = np.ndarray


def _safe_float_token(value: float) -> str:
    raw = f"{float(value):g}"
    return raw.replace("-", "m").replace(".", "p").replace("+", "")


def _validate_times(times: Array) -> Array:
    grid = np.asarray(times, dtype=float).reshape(-1)
    if grid.size == 0:
        raise ValueError("Boundary evaluation requires a non-empty time grid.")
    if not np.all(np.isfinite(grid)):
        raise ValueError("Boundary evaluation time grid contains non-finite values.")
    if np.any(np.diff(grid) < 0):
        raise ValueError("Boundary evaluation time grid must be nondecreasing.")
    return grid


@dataclass(frozen=True)
class BoundarySpec:
    boundary_type: str
    name: str
    metadata: dict[str, Any]
    _evaluator: Callable[[Array], Array]

    def values(self, times: Array) -> Array:
        grid = _validate_times(times)
        values = np.asarray(self._evaluator(grid), dtype=float).reshape(-1)
        if values.shape != grid.shape:
            raise ValueError(
                f"Boundary '{self.name}' returned shape {values.shape}, expected {grid.shape}."
            )
        if not np.all(np.isfinite(values)):
            raise ValueError(f"Boundary '{self.name}' returned non-finite values.")
        return values


def make_boundary_spec(
    *,
    boundary_type: str = "sine_hump",
    amplitude: float = -20.0,
) -> BoundarySpec:
    """Construct the normalized-time sine boundary used in the paper."""

    if str(boundary_type).strip().lower() != "sine_hump":
        raise ValueError("The paper implementation uses boundary_type='sine_hump'.")
    amp = float(amplitude)
    if not np.isfinite(amp):
        raise ValueError("boundary amplitude must be finite.")

    def _sine_hump(times: Array) -> Array:
        horizon = float(times[-1] - times[0])
        z = np.zeros_like(times, dtype=float) if horizon <= 0.0 else (times - times[0]) / horizon
        return -amp * np.sin(np.pi * z) ** 2

    token = f"sinehump_a{_safe_float_token(amp)}"
    return BoundarySpec(
        boundary_type="sine_hump",
        name=f"curved_boundary_{token}",
        metadata={
            "boundary_type": "sine_hump",
            "boundary_amplitude": amp,
            "boundary_max_abs_slope_reference_normalized_time": abs(amp) * np.pi,
            "boundary_phi_formula": "-amplitude*sin(pi*(t-t0)/T)^2",
            "dynamic_threshold_formula": "u + amplitude*sin(pi*(t-t0)/T)^2",
        },
        _evaluator=_sine_hump,
    )
