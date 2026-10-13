"""allocation_engine.py — AI portfolio analyst (GPT-4o via Emergent key)."""

from __future__ import annotations

from typing import Dict

from ..alpha_generator import _llm_chat


class AIAllocationEngine:
    """Stateless wrapper that asks GPT-4o for a plain-English portfolio
    analysis. Never predicts prices; only interprets the allocation."""

    @staticmethod
    def analyze_portfolio(
        allocation: Dict[str, float],
        metrics: Dict[str, float],
        method: str = "max_sharpe",
    ) -> str:
        system = (
            "You are a senior portfolio strategist writing for a retail investor. "
            "Given an optimised allocation and its risk/return metrics, write a "
            "concise 5-7 sentence analysis covering: (1) diversification quality, "
            "(2) risk concentration, (3) any over- or under-weight positions, "
            "(4) one observation about the chosen optimisation method "
            "(max_sharpe / min_variance / risk_parity / equal_weight), and "
            "(5) one rebalancing tip. Do not predict prices."
        )
        user = (
            f"Optimisation method: {method}\n"
            f"Allocation: {allocation}\n"
            f"Metrics: {metrics}"
        )
        return _llm_chat(system, user, session_id=f"optimizer-{method}")

    @staticmethod
    def explain_monte_carlo(percentiles: Dict[str, float], var_95: float,
                            cvar_95: float, days: int) -> str:
        system = (
            "You are a quantitative risk analyst. Translate Monte Carlo "
            "simulation outputs (5/25/50/75/95 percentiles of terminal "
            "portfolio value, 95% VaR and CVaR over the simulation horizon) "
            "into a 4-5 sentence explanation suitable for a retail investor. "
            "Highlight downside risk and a realistic central scenario. Do "
            "not give financial advice."
        )
        user = (
            f"Horizon: {days} trading days\n"
            f"Percentiles: {percentiles}\n"
            f"VaR95 (terminal): {var_95:.2f}\n"
            f"CVaR95 (terminal): {cvar_95:.2f}"
        )
        return _llm_chat(system, user, session_id="mc-explain")
