"""rebalance_engine.py — Generate trade list from drift.

Inputs:
  current_weights : {ticker: weight} of CURRENT portfolio
  target_weights  : {ticker: weight} from the optimiser
  portfolio_value : $ value of the portfolio

Output:
  list of trades (ticker, side, $ amount, weight_change)
"""

from __future__ import annotations

from typing import Dict, List


class RebalanceEngine:
    @staticmethod
    def rebalance(
        current_weights: Dict[str, float],
        target_weights: Dict[str, float],
        portfolio_value: float,
        min_trade_pct: float = 0.005,
    ) -> List[Dict]:
        tickers = set(current_weights) | set(target_weights)
        trades = []
        for t in sorted(tickers):
            cur = current_weights.get(t, 0.0)
            tgt = target_weights.get(t, 0.0)
            delta_w = tgt - cur
            if abs(delta_w) < min_trade_pct:
                continue
            dollar = delta_w * portfolio_value
            trades.append({
                "ticker": t,
                "side": "BUY" if dollar > 0 else "SELL",
                "weight_delta": round(delta_w, 4),
                "dollar_amount": round(abs(dollar), 2),
                "current_weight": round(cur, 4),
                "target_weight": round(tgt, 4),
            })
        return trades
