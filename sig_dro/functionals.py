from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .boundaries import BoundarySpec

Array = np.ndarray


@dataclass(frozen=True)
class PathFunctional:
    """A batched path-functional interface used by the reserve benchmarks."""

    name: str
    evaluator: Callable[[Array, Array], Array]
    default_event_sense: str | None = None
    description: str = ""

    def __call__(self, times: Array, paths: Array) -> Array:
        values = np.asarray(self.evaluator(times, paths), dtype=float)
        if values.ndim != 1:
            raise ValueError(f"Functional '{self.name}' must return a 1D array, got shape {values.shape}.")
        if values.shape[0] != paths.shape[0]:
            raise ValueError(
                f"Functional '{self.name}' returned length {values.shape[0]}, expected {paths.shape[0]}."
            )
        return values


ruin_depth = PathFunctional(
    name="ruin_depth",
    evaluator=lambda times, paths: paths[:, 0] - np.min(paths, axis=1),
    default_event_sense=">=",
    description="R_0 - min_t R_t, so {ruin_depth >= u} is finite-horizon ruin at initial capital u.",
)


def make_curved_boundary_ruin_depth_functional(
    boundary: BoundarySpec,
    name: str | None = None,
) -> PathFunctional:
    """Build the time-dependent-boundary first-passage score.

    The functional is ``F_phi(R) = sup_t [R_0 - R_t + phi(t)]``.  For a
    threshold ``u``, the event ``F_phi(R) >= u`` is the corresponding dynamic
    boundary crossing event.  With ``phi = 0`` it reduces exactly to
    :data:`ruin_depth`.
    """

    if not isinstance(boundary, BoundarySpec):
        raise TypeError("boundary must be a BoundarySpec instance.")

    functional_name = name or boundary.name

    def _evaluator(times: Array, paths: Array) -> Array:
        paths_arr = np.asarray(paths, dtype=float)
        if paths_arr.ndim != 2:
            raise ValueError(f"Expected one-dimensional reserve paths with shape (batch, steps + 1), got {paths_arr.shape}.")
        phi = boundary.values(times)
        if phi.shape[0] != paths_arr.shape[1]:
            raise ValueError(
                f"Boundary grid length {phi.shape[0]} does not match path length {paths_arr.shape[1]}."
            )
        loss_pressure = paths_arr[:, [0]] - paths_arr
        return np.max(loss_pressure + phi[None, :], axis=1)

    desc = (
        "Time-dependent-boundary first-passage score sup_t[R_0 - R_t + phi(t)] "
        f"using boundary '{boundary.boundary_type}'."
    )
    return PathFunctional(
        name=functional_name,
        evaluator=_evaluator,
        default_event_sense=">=",
        description=desc,
    )

def make_drawdown_functional(name: str = "max_drawdown") -> PathFunctional:
    """Build the maximum-drawdown functional.

    The discrete implementation is

        F_DD(R) = max_t (max_{s<=t} R_s - R_t),

    evaluated on the observed simulation grid.  The runner may include a running
    maximum channel in the signature lift, but this functional itself is the
    target label and is not inserted as a channel.
    """

    def _evaluator(times: Array, paths: Array) -> Array:
        paths_arr = np.asarray(paths, dtype=float)
        if paths_arr.ndim != 2:
            raise ValueError(f"Expected reserve paths with shape (batch, steps + 1), got {paths_arr.shape}.")
        running_max = np.maximum.accumulate(paths_arr, axis=1)
        return np.max(running_max - paths_arr, axis=1)

    return PathFunctional(
        name=name,
        evaluator=_evaluator,
        default_event_sense=">=",
        description="Maximum drawdown max_t(max_{s<=t} R_s - R_t) on the simulation grid.",
    )


def make_early_shortfall_functional(
    *,
    rho: float = 1.0,
    weight_type: str = "linear",
    name: str | None = None,
) -> PathFunctional:
    """Build an early-shortfall supremum functional.

    The target is

        F_w(R) = sup_t w(t) (R_0 - R_t),

    The paper uses the linear weight ``w(t)=1+rho*(1-t/T)``, so early reserve
    shortfalls receive the largest weight.
    """

    rho_val = float(rho)
    kind = str(weight_type).strip().lower()
    if kind != "linear":
        raise ValueError("The paper implementation uses weight_type='linear'.")
    if not np.isfinite(rho_val):
        raise ValueError("rho must be finite.")

    def _weights(times: Array) -> Array:
        grid = np.asarray(times, dtype=float).reshape(-1)
        if grid.size == 0:
            raise ValueError("Early-shortfall functional requires a non-empty time grid.")
        horizon = float(grid[-1] - grid[0])
        if horizon <= 0.0:
            z = np.ones_like(grid, dtype=float)
        else:
            z = 1.0 - (grid - grid[0]) / horizon
        return 1.0 + rho_val * z

    def _evaluator(times: Array, paths: Array) -> Array:
        paths_arr = np.asarray(paths, dtype=float)
        if paths_arr.ndim != 2:
            raise ValueError(f"Expected reserve paths with shape (batch, steps + 1), got {paths_arr.shape}.")
        w = _weights(times)
        if w.shape[0] != paths_arr.shape[1]:
            raise ValueError(f"Weight grid length {w.shape[0]} does not match path length {paths_arr.shape[1]}.")
        loss_pressure = paths_arr[:, [0]] - paths_arr
        return np.max(loss_pressure * w[None, :], axis=1)

    functional_name = name or f"early_shortfall_linear_rho{str(rho_val).replace('-', 'm').replace('.', 'p')}"
    formula = "sup_t (1+rho*(1-t/T))(R_0-R_t)"
    return PathFunctional(
        name=functional_name,
        evaluator=_evaluator,
        default_event_sense=">=",
        description=f"Early-shortfall score {formula} with rho={rho_val:g}.",
    )
