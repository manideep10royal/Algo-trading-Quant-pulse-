"""risk_parity.py — Equal Risk Contribution allocation.

Solves for weights such that every asset contributes the same amount of
portfolio risk (vs. equal capital weights in 1/N).
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize


class RiskParity:
    def __init__(self, cov_matrix):
        # Accept pandas DataFrame or numpy array
        try:
            self.cov_matrix = cov_matrix.values
            self.tickers = list(cov_matrix.columns)
        except AttributeError:
            self.cov_matrix = np.asarray(cov_matrix)
            self.tickers = [f"asset_{i}" for i in range(self.cov_matrix.shape[0])]

    def _port_vol(self, w):
        return float(np.sqrt(w @ self.cov_matrix @ w))

    def _risk_contribution(self, w):
        sigma = self._port_vol(w)
        marginal = self.cov_matrix @ w
        return w * marginal / sigma if sigma > 0 else np.zeros_like(w)

    def _objective(self, w):
        rc = self._risk_contribution(w)
        # Penalise dispersion of risk contributions
        return float(np.sum((rc - rc.mean()) ** 2))

    def optimize(self) -> np.ndarray:
        n = self.cov_matrix.shape[0]
        x0 = np.array([1.0 / n] * n)
        bounds = tuple((1e-4, 1.0) for _ in range(n))
        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)
        res = minimize(
            self._objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-10},
        )
        return res.x if res.success else x0

    def risk_contributions(self, weights):
        """Helpful debug output — returns {ticker: % of portfolio risk}."""
        rc = self._risk_contribution(weights)
        total = rc.sum() or 1.0
        return {t: float(rc[i] / total) for i, t in enumerate(self.tickers)}
