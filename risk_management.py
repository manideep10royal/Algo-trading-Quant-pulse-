"""
risk_management.py — Portfolio risk-control engine.

Implements:
- Position sizing (fixed-fraction + volatility adjusted)
- Stop-loss / take-profit calculation
- Risk / Reward ratio
- Trade validation against exposure & drawdown rules
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .config import (
    DEFAULT_CAPITAL,
    MAX_RISK_PER_TRADE,
    MAX_PORTFOLIO_EXPOSURE,
    MAX_DAILY_DRAWDOWN,
    DEFAULT_STOP_LOSS_PCT,
    DEFAULT_TAKE_PROFIT_PCT,
)


@dataclass
class TradeValidation:
    is_valid: bool
    reason: str
    suggested_quantity: int
    stop_loss: float
    take_profit: float
    risk_amount: float
    risk_reward_ratio: float


def calculate_position_size(
    capital: float,
    entry_price: float,
    stop_loss_price: float,
    risk_per_trade: float = MAX_RISK_PER_TRADE,
) -> int:
    """How many shares can we buy without risking more than `risk_per_trade` of
    capital if the stop-loss gets hit?"""
    if entry_price <= 0 or stop_loss_price <= 0:
        return 0
    per_share_risk = abs(entry_price - stop_loss_price)
    if per_share_risk == 0:
        return 0
    max_risk_amount = capital * risk_per_trade
    qty = int(max_risk_amount // per_share_risk)
    # Never spend more than what we have (whole-share cap)
    qty = min(qty, int(capital // entry_price))
    return max(qty, 0)


def calculate_stop_loss(
    entry_price: float,
    side: str = "BUY",
    pct: float = DEFAULT_STOP_LOSS_PCT,
    volatility: Optional[float] = None,
) -> float:
    """Compute a stop-loss price.

    If a `volatility` (annualised std-dev of returns) is supplied, widen the
    stop proportionally so volatile names breathe."""
    if volatility is not None and volatility > 0:
        pct = max(pct, min(volatility * 0.25, 0.10))   # cap at 10%
    return entry_price * (1 - pct) if side == "BUY" else entry_price * (1 + pct)


def calculate_take_profit(
    entry_price: float,
    side: str = "BUY",
    pct: float = DEFAULT_TAKE_PROFIT_PCT,
) -> float:
    return entry_price * (1 + pct) if side == "BUY" else entry_price * (1 - pct)


def calculate_risk_reward(
    entry: float, stop: float, target: float
) -> float:
    risk = abs(entry - stop)
    reward = abs(target - entry)
    return reward / risk if risk else 0.0


def validate_trade(
    capital: float,
    entry_price: float,
    side: str = "BUY",
    current_exposure: float = 0.0,
    daily_pnl_pct: float = 0.0,
    volatility: Optional[float] = None,
) -> TradeValidation:
    """One-shot trade validator returning suggested size and reasoning."""
    if entry_price <= 0:
        return TradeValidation(False, "Invalid entry price.", 0, 0, 0, 0, 0)
    if daily_pnl_pct <= -MAX_DAILY_DRAWDOWN:
        return TradeValidation(
            False, "Daily drawdown limit hit — trading paused.", 0, 0, 0, 0, 0
        )
    if current_exposure >= MAX_PORTFOLIO_EXPOSURE:
        return TradeValidation(
            False, "Portfolio exposure limit reached.", 0, 0, 0, 0, 0
        )

    stop = calculate_stop_loss(entry_price, side, volatility=volatility)
    target = calculate_take_profit(entry_price, side)
    qty = calculate_position_size(capital, entry_price, stop)
    rr = calculate_risk_reward(entry_price, stop, target)
    risk_amount = qty * abs(entry_price - stop)

    if qty == 0:
        return TradeValidation(
            False, "Position size resolves to 0 (insufficient capital or wide stop).",
            0, stop, target, 0.0, rr
        )

    return TradeValidation(
        is_valid=True,
        reason="Trade meets all risk rules.",
        suggested_quantity=qty,
        stop_loss=round(stop, 2),
        take_profit=round(target, 2),
        risk_amount=round(risk_amount, 2),
        risk_reward_ratio=round(rr, 2),
    )


def default_capital() -> float:
    return DEFAULT_CAPITAL
