from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .blocks import BlockSpec, block_dual_norm
from .regression import Surrogate

Array = np.ndarray
EventSense = Literal[">=", "<="]


@dataclass(frozen=True)
class RobustShiftResult:
    threshold: float
    shifted_threshold: float
    shift_star: float
    kappa: float
    theta_l1: float
    event_sense: str
    delta_scalar: float | None = None
    budget_dim: int = 0
    # For linked surrogates, thresholds below are in latent score space.
    score_threshold: float | None = None
    shifted_score_threshold: float | None = None
    threshold_space: str = "prediction"


def _is_linked_model(model) -> bool:
    return hasattr(model, "link") and getattr(model, "link_type", "identity") != "identity"


def truncated_moment_hat(t: float, margins: Array) -> float:
    margins = np.asarray(margins, dtype=float)
    return float(np.mean(margins * (margins <= t)))


def truncated_moment_inverse(kappa: float, margins: Array, tol: float = 1e-10, max_iter: int = 200) -> float:
    margins = np.asarray(margins, dtype=float)
    margins = margins[np.isfinite(margins)]
    margins = np.maximum(margins, 0.0)
    if margins.size == 0 or kappa <= 0.0:
        return 0.0

    t_lo = 0.0
    t_hi = float(np.max(margins))
    if kappa >= truncated_moment_hat(t_hi, margins):
        return t_hi

    for _ in range(max_iter):
        t_mid = 0.5 * (t_lo + t_hi)
        h_mid = truncated_moment_hat(t_mid, margins)
        if abs(h_mid - kappa) <= tol or (t_hi - t_lo) <= tol:
            return float(t_mid)
        if h_mid < kappa:
            t_lo = t_mid
        else:
            t_hi = t_mid
    return float(0.5 * (t_lo + t_hi))


def _resolve_delta_vec(
    feature_dim: int,
    delta: float | None = None,
    delta_vec: Array | None = None,
) -> tuple[Array, float | None]:
    if delta_vec is not None:
        arr = np.asarray(delta_vec, dtype=float).reshape(-1)
        if arr.size < feature_dim:
            raise ValueError(
                f"delta_vec has length {arr.size}, but the feature dimension is {feature_dim}."
            )
        return arr[:feature_dim], None
    if delta is None:
        raise ValueError("Provide either delta (scalar broadcast budget) or delta_vec.")
    return np.full(feature_dim, float(delta), dtype=float), float(delta)


def effective_budget_blocks(
    model: Surrogate,
    block_spec: BlockSpec,
    delta: float | None = None,
    delta_vec: Array | None = None,
) -> tuple[float, Array, Array, float | None]:
    resolved_delta_vec, delta_scalar = _resolve_delta_vec(feature_dim=block_spec.n_blocks, delta=delta, delta_vec=delta_vec)
    link_lip = float(getattr(model, "link_lip_const", 1.0))
    alpha_blocks = link_lip * np.array([
        block_dual_norm(model.theta[block.slc], block.norm)
        for block in block_spec.blocks
    ], dtype=float)
    alpha_blocks = alpha_blocks[:block_spec.n_blocks]
    kappa = float(np.dot(alpha_blocks, resolved_delta_vec))
    return kappa, resolved_delta_vec, alpha_blocks, delta_scalar


def score_threshold_for_model(model: Surrogate, threshold: float) -> tuple[float, str]:
    """Return the threshold in the model's event-score space.

    This avoids materialising a large feature matrix when only already-computed
    scores are available. Linked post-link surrogates use their latent score
    threshold; ordinary affine surrogates use the prediction threshold itself.
    """
    if _is_linked_model(model):
        if not hasattr(model, "inverse_threshold"):
            raise ValueError("Linked surrogate must expose inverse_threshold().")
        return float(model.inverse_threshold(float(threshold))), "score"
    return float(threshold), "prediction"


def pointwise_event_margin_from_scores(
    scores: Array,
    score_threshold: float,
    event_sense: EventSense,
) -> Array:
    scores = np.asarray(scores, dtype=float)
    if event_sense == ">=":
        raw = float(score_threshold) - scores
    elif event_sense == "<=":
        raw = scores - float(score_threshold)
    else:
        raise ValueError("event_sense must be '>=' or '<='.")
    return np.maximum(raw, 0.0)


def conservative_shift_from_scores(
    *,
    model: Surrogate,
    scores: Array,
    threshold: float,
    event_sense: EventSense,
    kappa: float,
    theta_l1: float | None = None,
    budget_dim: int = 0,
    tol: float = 1e-10,
) -> RobustShiftResult:
    score_threshold, threshold_space = score_threshold_for_model(model, threshold)
    margins = pointwise_event_margin_from_scores(scores, score_threshold, event_sense)
    shift_star = truncated_moment_inverse(kappa=float(kappa), margins=margins, tol=tol)
    shifted = float(score_threshold - shift_star) if event_sense == ">=" else float(score_threshold + shift_star)
    return RobustShiftResult(
        threshold=float(threshold),
        shifted_threshold=shifted,
        shift_star=float(shift_star),
        kappa=float(kappa),
        theta_l1=float(getattr(model, "theta_l1", 0.0) if theta_l1 is None else theta_l1),
        event_sense=event_sense,
        delta_scalar=None,
        budget_dim=int(budget_dim),
        score_threshold=float(score_threshold),
        shifted_score_threshold=float(shifted),
        threshold_space=threshold_space,
    )


def robust_probability_from_scores(
    scores: Array,
    shifted_score_threshold: float,
    event_sense: EventSense,
) -> float:
    scores = np.asarray(scores, dtype=float)
    if event_sense == ">=":
        return float(np.mean(scores >= float(shifted_score_threshold)))
    if event_sense == "<=":
        return float(np.mean(scores <= float(shifted_score_threshold)))
    raise ValueError("event_sense must be '>=' or '<='.")
