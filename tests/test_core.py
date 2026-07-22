from __future__ import annotations

import unittest

import numpy as np

import sig_dro
from sig_dro.discrepancy import component_ot_vector, empirical_w1_1d, sinkhorn_uniform
from sig_dro.option_payoffs import (
    discounted_arithmetic_asian_call_payoff,
    discounted_fixed_strike_lookback_call_payoff,
)
from sig_dro.robust import truncated_moment_hat, truncated_moment_inverse


class CoreNumericsTests(unittest.TestCase):
    def test_package_imports_and_signature_backend(self) -> None:
        self.assertTrue(hasattr(sig_dro, "effective_budget_blocks"))
        self.assertTrue(sig_dro.iisignature_available())

    def test_empirical_wasserstein(self) -> None:
        x = np.array([0.0, 1.0, 2.0])
        y = np.array([1.0, 2.0, 3.0])
        self.assertAlmostEqual(empirical_w1_1d(x, y), 1.0)
        features = component_ot_vector(np.column_stack([x, 2 * x]), np.column_stack([y, 2 * y]))
        np.testing.assert_allclose(features, [1.0, 2.0])

    def test_sinkhorn_uniform_marginals(self) -> None:
        cost = np.array([[0.0, 1.0], [1.0, 0.0]])
        coupling = sinkhorn_uniform(cost, epsilon=0.1, max_iter=500, tol=1e-12)
        np.testing.assert_allclose(coupling.sum(axis=0), [0.5, 0.5], atol=1e-9)
        np.testing.assert_allclose(coupling.sum(axis=1), [0.5, 0.5], atol=1e-9)

    def test_truncated_moment_inverse(self) -> None:
        margins = np.array([0.0, 1.0, 2.0, 3.0])
        value = truncated_moment_inverse(0.25, margins)
        self.assertAlmostEqual(truncated_moment_hat(value, margins), 0.25, places=8)

    def test_path_option_payoffs(self) -> None:
        prices = np.array([[100.0, 110.0, 120.0], [100.0, 90.0, 80.0]])
        log_paths = np.log(prices / 100.0)
        asian = discounted_arithmetic_asian_call_payoff(log_paths, S0=100.0, K=100.0)
        lookback = discounted_fixed_strike_lookback_call_payoff(log_paths, S0=100.0, K=100.0)
        np.testing.assert_allclose(asian, [15.0, 0.0])
        np.testing.assert_allclose(lookback, [20.0, 0.0])


if __name__ == "__main__":
    unittest.main()
