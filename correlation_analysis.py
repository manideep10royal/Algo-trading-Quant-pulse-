"""correlation_analysis.py — Pairwise correlation utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd


class CorrelationAnalysis:
    @staticmethod
    def correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
        return returns_df.dropna(how="all").corr()

    @staticmethod
    def diversification_score(corr: pd.DataFrame) -> float:
        """0-1 score (higher = better diversified). Computed as
        1 − (mean of off-diagonal correlation values, clipped to [0, 1])."""
        if corr.empty or corr.shape[0] < 2:
            return 0.0
        vals = corr.values
        n = vals.shape[0]
        off_diag = vals[~np.eye(n, dtype=bool)]
        off_diag = off_diag[~np.isnan(off_diag)]
        if off_diag.size == 0:
            return 0.0
        return float(max(0.0, min(1.0, 1.0 - off_diag.mean())))

    @staticmethod
    def highest_correlated_pairs(corr: pd.DataFrame, top_n: int = 5):
        """Return the top_n most-correlated unique pairs."""
        if corr.empty:
            return []
        n = corr.shape[0]
        mask = np.triu(np.ones((n, n), dtype=bool), k=1)
        pairs = []
        for i in range(n):
            for j in range(n):
                if mask[i, j] and not np.isnan(corr.iat[i, j]):
                    pairs.append({
                        "a": corr.index[i], "b": corr.columns[j],
                        "corr": float(corr.iat[i, j]),
                    })
        pairs.sort(key=lambda d: d["corr"], reverse=True)
        return pairs[:top_n]
