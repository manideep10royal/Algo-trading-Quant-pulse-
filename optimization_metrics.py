"""optimization_metrics.py — Standard portfolio metrics.

All functions accept a `pandas.Series` of (daily) returns OR a numpy array
and operate without state.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class OptimizationMetrics:

    @staticmethod
    def sharpe_ratio(returns, risk_free_rate: float = 0.04) -> float:
        r = pd.Series(returns).dropna()
        if r.std() == 0 or r.empty:
            return 0.0
        excess = r.mean() * 252 - risk_free_rate
        return float(excess / (r.std() * np.sqrt(252)))

    @staticmethod
    def sortino_ratio(returns, risk_free_rate: float = 0.04) -> float:
        r = pd.Series(returns).dropna()
        downside = r[r < 0]
        if downside.std() == 0 or r.empty:
            return 0.0
        return float(((r.mean() * 252) - risk_free_rate) / (downside.std() * np.sqrt(252)))

    @staticmethod
    def max_drawdown(returns) -> float:
        r = pd.Series(returns).dropna()
        if r.empty:
            return 0.0
        cum = (1 + r).cumprod()
        peak = cum.cummax()
        dd = (cum - peak) / peak
        return float(dd.min())

    @staticmethod
    def cagr(returns, periods_per_year: int = 252) -> float:
        r = pd.Series(returns).dropna()
        if r.empty:
            return 0.0
        years = len(r) / periods_per_year
        final = float((1 + r).prod())
        if years <= 0 or final <= 0:
            return 0.0
        return float(final ** (1 / years) - 1)

    @staticmethod
    def diversification_ratio(weights, cov_matrix) -> float:
        """Weighted average of individual vols ÷ portfolio vol.
        Values > 1 indicate diversification benefit; 1 means none."""
        weights = np.asarray(weights)
        try:
            cov = cov_matrix.values
        except AttributeError:
            cov = np.asarray(cov_matrix)
        ind_vols = np.sqrt(np.diag(cov))
        weighted_vol = float(weights @ ind_vols)
        port_vol = float(np.sqrt(weights @ cov @ weights))
        return weighted_vol / port_vol if port_vol > 0 else 0.0

    @staticmethod
    def volatility(returns) -> float:
        r = pd.Series(returns).dropna()
        return float(r.std() * np.sqrt(252)) if not r.empty else 0.0
