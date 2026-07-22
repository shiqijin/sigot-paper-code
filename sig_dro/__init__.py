"""Numerical core for the Sig-OT paper experiments."""

from .affine_payoff import LogPathModel
from .configs import BrownianReserveConfig, CramerLundbergConfig, ExperimentConfig
from .discrepancy import component_ot_vector, empirical_w1_1d, joint_ot_budget_vector
from .functionals import (
    make_curved_boundary_ruin_depth_functional,
    make_drawdown_functional,
    make_early_shortfall_functional,
    ruin_depth,
)
from .preprocess import ComponentScaler, fit_component_scaler
from .regression import fit_surrogate
from .robust import (
    conservative_shift_from_scores,
    effective_budget_blocks,
    robust_probability_from_scores,
    truncated_moment_hat,
    truncated_moment_inverse,
)
from .signatures import iisignature_available, time_augmented_signatures

__all__ = [
    "LogPathModel",
    "BrownianReserveConfig",
    "CramerLundbergConfig",
    "ExperimentConfig",
    "ComponentScaler",
    "fit_component_scaler",
    "component_ot_vector",
    "empirical_w1_1d",
    "joint_ot_budget_vector",
    "fit_surrogate",
    "time_augmented_signatures",
    "truncated_moment_hat",
    "truncated_moment_inverse",
    "conservative_shift_from_scores",
    "effective_budget_blocks",
    "robust_probability_from_scores",
    "iisignature_available",
    "ruin_depth",
    "make_curved_boundary_ruin_depth_functional",
    "make_early_shortfall_functional",
    "make_drawdown_functional",
]
