"""portfolio_optimizer.py — Modern Portfolio Theory engine.

Two convex objectives:
- **Max Sharpe**  → maximise (Return − Rf) / Vol
- **Min Variance** → minimise portfolio variance

Both subject to:
    Σ w_i = 1
    0 ≤ w_i ≤ 1     (long-only, no leverage)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass
class OptimizationResult:
    method: str
    weights: np.ndarray
    tickers: List[str]
    expected_return: float
    volatility: float
    sharpe: float

    def as_dict(self) -> dict:
        return dict(zip(self.tickers, [round(float(w), 4) for w in self.weights]))


class PortfolioOptimizer:
    """Frequentist MPT optimizer driven by historical daily returns.

    Parameters
    ----------
    returns_df : DataFrame
        Columns are tickers, rows are daily simple returns.
    risk_free_rate : float
        Annualised risk-free rate (default 4%).
    """

    def __init__(self, returns_df: pd.DataFrame, risk_free_rate: float = 0.04):
        if returns_df.empty:
            raise ValueError("Empty returns frame.")
        self.returns_df = returns_df.dropna(how="all").fillna(0.0)
        self.tickers = list(self.returns_df.columns)
        self.risk_free_rate = float(risk_free_rate)
        self.mean_returns = self.returns_df.mean() * 252       # annualised
        self.cov_matrix = self.returns_df.cov() * 252

    # ------------------------------------------------------------------
    def portfolio_performance(self, weights: np.ndarray):
        ret = float(np.dot(weights, self.mean_returns))
        vol = float(np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights))))
        sharpe = (ret - self.risk_free_rate) / vol if vol > 0 else 0.0
        return ret, vol, sharpe

    def _negative_sharpe(self, w):
        return -self.portfolio_performance(w)[2]

    def _portfolio_variance(self, w):
        return float(np.dot(w.T, np.dot(self.cov_matrix, w)))

    # ------------------------------------------------------------------
    def _solve(self, objective, x0=None) -> np.ndarray:
        n = len(self.tickers)
        if x0 is None:
            x0 = np.array([1.0 / n] * n)
        bounds = tuple((0.0, 1.0) for _ in range(n))
        constraints = ({"type": "eq", "fun": lambda x: np.sum(x) - 1.0},)
        res = minimize(
            objective, x0, method="SLSQP",
            bounds=bounds, constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-9},
        )
        return res.x if res.success else x0

    def optimize_max_sharpe(self) -> OptimizationResult:
        w = self._solve(self._negative_sharpe)
        r, v, s = self.portfolio_performance(w)
        return OptimizationResult("max_sharpe", w, self.tickers, r, v, s)

    def optimize_min_variance(self) -> OptimizationResult:
        w = self._solve(self._portfolio_variance)
        r, v, s = self.portfolio_performance(w)
        return OptimizationResult("min_variance", w, self.tickers, r, v, s)

    def equal_weight(self) -> OptimizationResult:
        n = len(self.tickers)
        w = np.array([1.0 / n] * n)
        r, v, s = self.portfolio_performance(w)
        return OptimizationResult("equal_weight", w, self.tickers, r, v, s)
