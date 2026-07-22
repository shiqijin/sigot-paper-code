from __future__ import annotations

import numpy as np

Array = np.ndarray


def _validated_prices(log_paths: Array, S0: float) -> Array:
    x = np.asarray(log_paths, dtype=float)
    if x.ndim != 2:
        raise ValueError("log_paths must have shape (n_paths, n_steps + 1).")
    if x.shape[1] < 1:
        raise ValueError("log_paths must contain at least one monitoring time.")
    return float(S0) * np.exp(x)


def discounted_arithmetic_asian_call_payoff(
    log_paths: Array,
    *,
    S0: float,
    K: float,
    r: float = 0.0,
    T: float = 1.0,
    include_initial: bool = False,
) -> Array:
    """Discounted discretely monitored arithmetic Asian call.

    The default monitoring average excludes the deterministic initial price S_0,
    i.e. it uses the simulation grid points t_1, ..., t_m. Pass
    ``include_initial=True`` to include t_0 as well.
    """
    S = _validated_prices(log_paths, S0=S0)
    monitored = S if bool(include_initial) else S[:, 1:]
    if monitored.shape[1] == 0:
        monitored = S
    average_price = np.mean(monitored, axis=1)
    return np.exp(-float(r) * float(T)) * np.maximum(average_price - float(K), 0.0)


def discounted_fixed_strike_lookback_call_payoff(
    log_paths: Array,
    *,
    S0: float,
    K: float,
    r: float = 0.0,
    T: float = 1.0,
) -> Array:
    """Discounted fixed-strike lookback call monitored on the simulation grid.

    Payoff: exp(-rT) * (max_{grid} S_t - K)_+.
    """
    S = _validated_prices(log_paths, S0=S0)
    running_peak = np.max(S, axis=1)
    return np.exp(-float(r) * float(T)) * np.maximum(running_peak - float(K), 0.0)
