"""monte_carlo.py — Forward portfolio simulation.

Generates `simulations` future equity paths given a fixed allocation,
sampling daily returns from a multivariate Normal calibrated on the
input returns DataFrame.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass
class MonteCarloResult:
    paths: np.ndarray          # shape (sims, days+1)
    final_values: np.ndarray   # shape (sims,)
    percentiles: dict          # {p05, p25, p50, p75, p95}
    var_95: float              # 1-day 95% VaR on final $
    cvar_95: float


class MonteCarloSimulator:
    def __init__(self, returns_df: pd.DataFrame):
        if returns_df.empty:
            raise ValueError("Empty returns frame.")
        self.returns_df = returns_df.dropna(how="all").fillna(0.0)
        self.tickers = list(self.returns_df.columns)
        self.mean_returns = self.returns_df.mean().values
        self.cov_matrix = self.returns_df.cov().values

    def simulate(
        self,
        weights: np.ndarray,
        initial_value: float = 100_000,
        simulations: int = 1000,
        days: int = 252,
        seed: int = 42,
    ) -> MonteCarloResult:
        rng = np.random.default_rng(seed)
        # Draw simulations × days × assets
        draws = rng.multivariate_normal(
            self.mean_returns, self.cov_matrix, size=(simulations, days)
        )
        # Portfolio daily returns
        port_daily = draws @ np.asarray(weights)
        growth = np.cumprod(1.0 + port_daily, axis=1)
        # Prepend initial value
        paths = np.concatenate(
            [np.full((simulations, 1), 1.0), growth], axis=1
        ) * initial_value
        final = paths[:, -1]

        pct = np.percentile(final, [5, 25, 50, 75, 95])
        percentiles = {
            "p05": float(pct[0]), "p25": float(pct[1]), "p50": float(pct[2]),
            "p75": float(pct[3]), "p95": float(pct[4]),
        }
        var_95 = float(initial_value - pct[0])
        # CVaR = expected loss given we're in the worst 5%
        worst = final[final <= pct[0]]
        cvar_95 = float(initial_value - worst.mean()) if len(worst) else 0.0

        return MonteCarloResult(paths, final, percentiles, var_95, cvar_95)
