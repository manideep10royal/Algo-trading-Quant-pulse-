"""efficient_frontier.py — Monte-Carlo random-portfolio frontier.

We sample N random weight vectors uniformly on the simplex (Dirichlet(1,…))
and evaluate Return / Vol / Sharpe for each. The convex hull of the upper
edge approximates the efficient frontier.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .portfolio_optimizer import PortfolioOptimizer


class EfficientFrontier:
    def __init__(self, optimizer: PortfolioOptimizer):
        self.opt = optimizer

    def generate_frontier(self, n_portfolios: int = 5000, seed: int = 42) -> pd.DataFrame:
        """Returns a DataFrame with columns: return, vol, sharpe + per-ticker weights."""
        rng = np.random.default_rng(seed)
        n_assets = len(self.opt.tickers)
        weights = rng.dirichlet(np.ones(n_assets), size=n_portfolios)

        rets = weights @ self.opt.mean_returns.values
        vols = np.sqrt(np.einsum("ij,jk,ik->i", weights, self.opt.cov_matrix.values, weights))
        sharpes = (rets - self.opt.risk_free_rate) / np.where(vols > 0, vols, np.nan)

        df = pd.DataFrame(weights, columns=self.opt.tickers)
        df.insert(0, "return", rets)
        df.insert(1, "vol", vols)
        df.insert(2, "sharpe", sharpes)
        return df

    @staticmethod
    def best_sharpe(frontier_df: pd.DataFrame) -> pd.Series:
        return frontier_df.iloc[frontier_df["sharpe"].idxmax()]

    @staticmethod
    def min_vol(frontier_df: pd.DataFrame) -> pd.Series:
        return frontier_df.iloc[frontier_df["vol"].idxmin()]
