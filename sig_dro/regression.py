from __future__ import annotations

"""Weighted-lasso signature surrogates used in the paper."""

from dataclasses import dataclass
from typing import Union

import numpy as np
from scipy.optimize import minimize

from .links import MonotoneLink, _sigmoid, _spline_basis


Array = np.ndarray


@dataclass(frozen=True)
class AffineSurrogate:
    theta: Array
    beta: float
    rmse_train: float
    rmse_test: float
    r2_test: float
    n_active: int
    feature_mean: Array
    active_mask: Array
    regression_mode: str = "weighted_lasso"
    sign_template: Array | None = None
    objective_success: bool = True
    objective_message: str = ""

    @property
    def theta_norm(self) -> float:
        return float(np.linalg.norm(self.theta))

    @property
    def theta_l1(self) -> float:
        return float(np.sum(np.abs(self.theta)))

    @property
    def alpha_vec(self) -> Array:
        return np.abs(self.theta)

    def predict(self, features: Array) -> Array:
        return np.asarray(features, dtype=float) @ self.theta + self.beta

    def component_contributions(self, delta_vec: Array) -> Array:
        values = np.asarray(delta_vec, dtype=float)
        if values.shape[0] < self.theta.shape[0]:
            raise ValueError("delta_vec is shorter than the model dimension.")
        return np.abs(self.theta) * values[: self.theta.shape[0]]


@dataclass(frozen=True)
class LinkedSurrogate:
    theta: Array
    beta: float
    link: MonotoneLink
    rmse_train: float
    rmse_test: float
    r2_test: float
    n_active: int
    feature_mean: Array
    active_mask: Array
    regression_mode: str = "posthoc_lip_spline"
    sign_template: Array | None = None
    objective_success: bool = True
    objective_message: str = ""
    base_rmse_train: float | None = None
    base_rmse_test: float | None = None

    @property
    def link_type(self) -> str:
        return self.link.link_type

    @property
    def link_lip_const(self) -> float:
        return float(self.link.lip_const)

    @property
    def theta_norm(self) -> float:
        return float(np.linalg.norm(self.theta))

    @property
    def theta_l1(self) -> float:
        return float(np.sum(np.abs(self.theta)))

    @property
    def alpha_vec(self) -> Array:
        return np.abs(self.theta)

    def score(self, features: Array) -> Array:
        return np.asarray(features, dtype=float) @ self.theta + self.beta

    def predict(self, features: Array) -> Array:
        return self.link.predict(self.score(features))

    def inverse_threshold(self, threshold: float) -> float:
        return self.link.inverse_threshold(float(threshold))

    def component_contributions(self, delta_vec: Array) -> Array:
        values = np.asarray(delta_vec, dtype=float)
        if values.shape[0] < self.theta.shape[0]:
            raise ValueError("delta_vec is shorter than the model dimension.")
        return self.link_lip_const * np.abs(self.theta) * values[: self.theta.shape[0]]


def _rmse(y_true: Array, y_pred: Array) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def _r2(y_true: Array, y_pred: Array) -> float:
    denominator = float(np.sum((y_true - y_true.mean()) ** 2))
    if denominator <= 0.0:
        return float("nan")
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denominator)


def _sign_template_from_correlation(features: Array, targets: Array) -> Array:
    template = np.sign(features.T @ targets)
    template[template == 0.0] = 1.0
    return template.astype(float)


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return float(value - threshold)
    if value < -threshold:
        return float(value + threshold)
    return 0.0


def _fit_centered_weighted_lasso(
    features: Array,
    targets: Array,
    *,
    alpha: float,
    penalty_weights: Array,
    max_iter: int,
    tol: float = 1e-8,
) -> tuple[Array, bool, str, Array]:
    n_rows, n_columns = features.shape
    if n_columns == 0:
        return np.zeros(0, dtype=float), True, "no active columns", np.zeros(0, dtype=float)

    weights = np.asarray(penalty_weights, dtype=float).reshape(-1)
    if weights.shape[0] != n_columns:
        raise ValueError("penalty_weights has inconsistent dimension for weighted_lasso.")
    if np.any(weights < 0.0):
        raise ValueError("weighted_lasso requires non-negative penalty weights.")

    column_squares = np.sum(features * features, axis=0) / float(n_rows)
    theta = np.zeros(n_columns, dtype=float)
    residual = targets.copy()
    converged = False
    n_iter = 0
    for iteration in range(int(max_iter)):
        max_change = 0.0
        for column in range(n_columns):
            scale = float(column_squares[column])
            if scale <= 0.0:
                continue
            old = float(theta[column])
            values = features[:, column]
            residual = residual + values * old
            rho = float(np.dot(values, residual) / float(n_rows))
            weight = float(weights[column])
            new = _soft_threshold(rho, float(alpha) * weight) / scale if weight > 0.0 else rho / scale
            theta[column] = new
            residual = residual - values * new
            max_change = max(max_change, abs(new - old))
        n_iter = iteration + 1
        if max_change <= float(tol):
            converged = True
            break

    message = f"centered weighted lasso coordinate descent (n_iter={n_iter}, tol={tol:g})"
    return theta, converged, message, _sign_template_from_correlation(features, targets)


def fit_affine_surrogate(
    features: Array,
    targets: Array,
    train_size: int,
    alpha: float = 1e-3,
    variance_tol: float = 1e-12,
    delta_vec: Array | None = None,
    max_iter: int = 1000,
) -> AffineSurrogate:
    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    if features.ndim != 2 or targets.ndim != 1:
        raise ValueError("features must be 2D and targets must be 1D.")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features and targets have inconsistent first dimension.")
    if not (0 < int(train_size) < features.shape[0]):
        raise ValueError("train_size must be between 1 and len(features)-1.")

    train_features = features[:train_size]
    test_features = features[train_size:]
    train_targets = targets[:train_size]
    test_targets = targets[train_size:]
    feature_mean = train_features.mean(axis=0)
    centered_train = train_features - feature_mean
    active_mask = np.isfinite(np.var(centered_train, axis=0)) & (
        np.var(centered_train, axis=0) > float(variance_tol)
    )
    n_active = int(np.sum(active_mask))
    target_mean = float(train_targets.mean())
    centered_targets = train_targets - target_mean
    theta = np.zeros(train_features.shape[1], dtype=float)
    sign_template = np.zeros(train_features.shape[1], dtype=float)
    success, message = True, ""

    if n_active > 0:
        if delta_vec is None:
            weights = np.ones(train_features.shape[1], dtype=float)
        else:
            weights = np.asarray(delta_vec, dtype=float).reshape(-1)
            if weights.shape[0] < train_features.shape[1]:
                raise ValueError("delta_vec shorter than feature dimension in fit_affine_surrogate.")
            weights = weights[: train_features.shape[1]]
        theta_active, success, message, template_active = _fit_centered_weighted_lasso(
            centered_train[:, active_mask],
            centered_targets,
            alpha=float(alpha),
            penalty_weights=weights[active_mask],
            max_iter=int(max_iter),
        )
        theta[active_mask] = theta_active
        sign_template[active_mask] = template_active

    beta = float(target_mean - feature_mean @ theta)
    prediction_train = train_features @ theta + beta
    prediction_test = test_features @ theta + beta
    return AffineSurrogate(
        theta=theta,
        beta=beta,
        rmse_train=_rmse(train_targets, prediction_train),
        rmse_test=_rmse(test_targets, prediction_test),
        r2_test=_r2(test_targets, prediction_test),
        n_active=n_active,
        feature_mean=feature_mean,
        active_mask=active_mask,
        regression_mode="weighted_lasso",
        sign_template=sign_template,
        objective_success=success,
        objective_message=message,
    )


def _quantile_knots(scores: Array, n_knots: int, q_low: float, q_high: float) -> Array:
    finite = np.asarray(scores, dtype=float).reshape(-1)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        return np.linspace(-1.0, 1.0, max(2, int(n_knots)))
    quantiles = np.linspace(
        float(np.clip(q_low, 0.0, 0.49)),
        float(np.clip(q_high, 0.51, 1.0)),
        max(2, int(n_knots)),
    )
    knots = np.quantile(finite, quantiles)
    if np.min(np.diff(knots)) <= 1e-10:
        low, high = float(np.min(finite)), float(np.max(finite))
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = -1.0, 1.0
        padding = 1e-6 * max(1.0, abs(low), abs(high))
        knots = np.linspace(low - padding, high + padding, max(2, int(n_knots)))
    return np.asarray(knots, dtype=float)


def _fit_posthoc_lipschitz_spline(
    scores: Array,
    targets: Array,
    *,
    lipschitz: float,
    n_knots: int,
    q_low: float,
    q_high: float,
    max_iter: int,
) -> tuple[MonotoneLink, bool, str]:
    scores = np.asarray(scores, dtype=float).reshape(-1)
    targets = np.asarray(targets, dtype=float).reshape(-1)
    knots = _quantile_knots(scores, n_knots, q_low, q_high)
    lip = float(lipschitz)
    eps = 1e-4
    target_slope = min(lip * (1.0 - eps), 1.0 - eps)
    fraction = np.clip(target_slope / max(lip, 1e-12), eps, 1.0 - eps)
    raw_value = np.log(fraction / (1.0 - fraction))
    raw0 = np.full(knots.size - 1, raw_value, dtype=float)
    basis = _spline_basis(scores, knots)
    slopes0 = lip * _sigmoid(raw0)
    intercept0 = float(np.mean(scores - basis @ slopes0))
    initial = np.concatenate([[intercept0], raw0])

    def objective_gradient(values: Array) -> tuple[float, Array]:
        intercept = float(values[0])
        sigmoid = _sigmoid(np.asarray(values[1:], dtype=float))
        slopes = lip * sigmoid
        residual = intercept + basis @ slopes - targets
        loss = float(np.mean(residual**2))
        prediction_gradient = (2.0 / float(scores.size)) * residual
        gradient_intercept = float(np.sum(prediction_gradient))
        gradient_slopes = basis.T @ prediction_gradient
        gradient_raw = gradient_slopes * lip * sigmoid * (1.0 - sigmoid)
        return loss, np.concatenate([[gradient_intercept], gradient_raw])

    result = minimize(
        lambda values: objective_gradient(values)[0],
        initial,
        method="L-BFGS-B",
        jac=lambda values: objective_gradient(values)[1],
        options={"maxiter": int(max_iter), "ftol": 1e-10, "gtol": 1e-6, "maxls": 50},
    )
    values = np.asarray(result.x, dtype=float)
    slopes = lip * _sigmoid(values[1:])
    link = MonotoneLink(
        "lip_spline",
        {"knots": knots, "slopes": slopes, "intercept": float(values[0])},
        lip,
        True,
    )
    return link, bool(result.success), str(result.message)


def fit_linked_surrogate(
    features: Array,
    targets: Array,
    train_size: int,
    alpha: float = 1e-3,
    variance_tol: float = 1e-12,
    delta_vec: Array | None = None,
    max_iter: int = 1000,
    link_type: str = "lip_spline",
    link_fit_mode: str = "posthoc",
    link_lipschitz: float = 1.0,
    link_n_knots: int = 32,
    link_knot_quantile_low: float = 0.01,
    link_knot_quantile_high: float = 0.99,
) -> AffineSurrogate | LinkedSurrogate:
    affine = fit_affine_surrogate(
        features=features,
        targets=targets,
        train_size=train_size,
        alpha=alpha,
        variance_tol=variance_tol,
        delta_vec=delta_vec,
        max_iter=max_iter,
    )
    if str(link_type) == "identity":
        return affine
    if str(link_type) != "lip_spline" or str(link_fit_mode) != "posthoc":
        raise ValueError("The paper implementation uses a posthoc lip_spline link.")

    features = np.asarray(features, dtype=float)
    targets = np.asarray(targets, dtype=float)
    train_features, test_features = features[:train_size], features[train_size:]
    train_targets, test_targets = targets[:train_size], targets[train_size:]
    score_train = affine.predict(train_features)
    score_test = affine.predict(test_features)
    link, success, message = _fit_posthoc_lipschitz_spline(
        score_train,
        train_targets,
        lipschitz=float(link_lipschitz),
        n_knots=int(link_n_knots),
        q_low=float(link_knot_quantile_low),
        q_high=float(link_knot_quantile_high),
        max_iter=int(max_iter),
    )
    prediction_train = link.predict(score_train)
    prediction_test = link.predict(score_test)
    return LinkedSurrogate(
        theta=np.asarray(affine.theta, dtype=float),
        beta=float(affine.beta),
        link=link,
        rmse_train=_rmse(train_targets, prediction_train),
        rmse_test=_rmse(test_targets, prediction_test),
        r2_test=_r2(test_targets, prediction_test),
        n_active=int(affine.n_active),
        feature_mean=np.asarray(affine.feature_mean, dtype=float),
        active_mask=np.asarray(affine.active_mask, dtype=bool),
        regression_mode="posthoc_lip_spline",
        sign_template=affine.sign_template,
        objective_success=success,
        objective_message=message,
        base_rmse_train=affine.rmse_train,
        base_rmse_test=affine.rmse_test,
    )


def fit_surrogate(
    features: Array,
    targets: Array,
    train_size: int,
    alpha: float = 1e-3,
    variance_tol: float = 1e-12,
    delta_vec: Array | None = None,
    max_iter: int = 1000,
    surrogate_type: str = "post_link",
    link_type: str = "lip_spline",
    link_fit_mode: str = "posthoc",
    link_lipschitz: float = 1.0,
    link_n_knots: int = 32,
    link_knot_quantile_low: float = 0.01,
    link_knot_quantile_high: float = 0.99,
):
    if str(surrogate_type) not in {"post_link", "affine"}:
        raise ValueError("surrogate_type must be 'post_link' or 'affine'.")
    if str(surrogate_type) == "affine":
        link_type = "identity"
    return fit_linked_surrogate(
        features=features,
        targets=targets,
        train_size=train_size,
        alpha=alpha,
        variance_tol=variance_tol,
        delta_vec=delta_vec,
        max_iter=max_iter,
        link_type=link_type,
        link_fit_mode=link_fit_mode,
        link_lipschitz=link_lipschitz,
        link_n_knots=link_n_knots,
        link_knot_quantile_low=link_knot_quantile_low,
        link_knot_quantile_high=link_knot_quantile_high,
    )


Surrogate = Union[AffineSurrogate, LinkedSurrogate]
