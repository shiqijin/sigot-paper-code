from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence


@dataclass(frozen=True)
class BrownianReserveConfig:
    """Configuration for the Brownian approximation of the Cramér–Lundberg reserve model."""

    T: float = 100.0
    steps: int = 2_000
    paths: int = 10_000
    nu: float = 1.0
    eta: float = 0.1
    alpha: float = 2.2
    seed: int = 0
    batch_size: int = 256
    normalization_eps: float = 1e-12


@dataclass(frozen=True)
class ExperimentConfig:
    """High-level experiment settings for signature-based robust probability experiments."""

    thresholds: Sequence[float] = (50.0, 100.0, 150.0, 200.0, 250.0)
    signature_levels: Sequence[int] = (2, 3, 4)
    train_size: int = 8_000
    regression_alpha: float = 1e-3
    optimizer_max_iter: int = 5000
    # Surrogate backend used by the paper computations.
    surrogate_type: Literal["post_link"] = "post_link"
    link_type: Literal["lip_spline"] = "lip_spline"
    link_fit_mode: Literal["posthoc"] = "posthoc"
    link_lipschitz: float = 1.0
    link_n_knots: int = 32
    link_knot_quantile_low: float = 0.01
    link_knot_quantile_high: float = 0.99
    event_sense: str = ">="
    signature_backend: Literal["iisignature"] = "iisignature"
    signature_type: Literal["geometric"] = "geometric"
    signature_representation: Literal["signature"] = "signature"
    scaling_method: Literal["mad"] = "mad"
    scaling_floor: float = 1e-8
    add_running_max_channel: bool = False
    add_running_min_channel: bool = False
    ot_mode: Literal["joint"] = "joint"
    ot_grouping: Literal["coordinate"] = "coordinate"
    ot_block_norm: Literal["l2"] = "l2"
    ot_lambda_mode: Literal["preliminary_score"] = "preliminary_score"
    ot_lambda_floor: float = 1e-8
    ot_lambda_power: float = 1.0
    ot_lambda_normalize: bool = True
    ot_sinkhorn_epsilon: float = 1e-2
    ot_sinkhorn_max_iter: int = 400
    ot_sinkhorn_tol: float = 1e-9
    ot_batch_size: int = 512
    ot_batches: int = 8
    ot_seed: int = 0


@dataclass(frozen=True)
class CramerLundbergConfig:
    """Configuration for the compound Poisson Cramér–Lundberg reserve model."""

    T: float = 100.0
    steps: int = 2_000
    paths: int = 10_000
    nu: float = 1.0
    eta: float = 0.1
    alpha: float = 2.2
    seed: int = 1
    batch_size: int = 256
    initial_capital: float | None = None
    normalization_eps: float = 1e-12
