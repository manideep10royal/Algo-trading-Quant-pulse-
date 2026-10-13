"""
portfolio.py — Portfolio analytics & persistence.

Holdings + trades are persisted as simple CSV files inside `data/` so the MVP
needs no database. All money values are floats; for production you'd switch
to Decimal + a SQL store.

Public entry points
-------------------
- load_portfolio()          → dict with cash, positions, history
- save_portfolio(p)
- record_trade(p, trade)    → append a trade & adjust cash / positions
- portfolio_summary(p, prices)
- equity_curve(p)
- calculate_drawdown(series)
- calculate_sharpe_ratio(returns)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import DEFAULT_CAPITAL, DATA_DIR, PORTFOLIO_FILE, TRADES_FILE
from .utils import setup_logger

log = setup_logger("portfolio")

_STATE_FILE = DATA_DIR / "portfolio_state.json"


@dataclass
class Position:
    ticker: str
    quantity: int
    avg_price: float

    def market_value(self, price: float) -> float:
        return self.quantity * price

    def unrealized_pnl(self, price: float) -> float:
        return (price - self.avg_price) * self.quantity


@dataclass
class Trade:
    timestamp: str
    ticker: str
    side: str            # BUY / SELL
    quantity: int
    price: float
    total: float
    mode: str            # PAPER / LIVE
    note: str = ""


@dataclass
class Portfolio:
    cash: float = DEFAULT_CAPITAL
    initial_capital: float = DEFAULT_CAPITAL
    positions: Dict[str, Position] = field(default_factory=dict)
    history: List[Trade] = field(default_factory=list)
    equity_history: List[Dict] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def load_portfolio() -> Portfolio:
    if not _STATE_FILE.exists():
        p = Portfolio()
        save_portfolio(p)
        return p
    try:
        raw = json.loads(_STATE_FILE.read_text())
        positions = {
            k: Position(**v) for k, v in raw.get("positions", {}).items()
        }
        history = [Trade(**t) for t in raw.get("history", [])]
        return Portfolio(
            cash=raw.get("cash", DEFAULT_CAPITAL),
            initial_capital=raw.get("initial_capital", DEFAULT_CAPITAL),
            positions=positions,
            history=history,
            equity_history=raw.get("equity_history", []),
        )
    except Exception as exc:  # noqa: BLE001
        log.exception("Portfolio load failed; resetting. (%s)", exc)
        return Portfolio()


def save_portfolio(p: Portfolio) -> None:
    raw = {
        "cash": p.cash,
        "initial_capital": p.initial_capital,
        "positions": {k: asdict(v) for k, v in p.positions.items()},
        "history": [asdict(t) for t in p.history],
        "equity_history": p.equity_history,
    }
    _STATE_FILE.write_text(json.dumps(raw, indent=2, default=str))
    # Also dump human-readable CSVs
    if p.history:
        pd.DataFrame([asdict(t) for t in p.history]).to_csv(TRADES_FILE, index=False)
    if p.positions:
        pd.DataFrame([asdict(v) for v in p.positions.values()]).to_csv(
            PORTFOLIO_FILE, index=False
        )


def reset_portfolio(capital: Optional[float] = None) -> Portfolio:
    p = Portfolio(
        cash=capital or DEFAULT_CAPITAL,
        initial_capital=capital or DEFAULT_CAPITAL,
    )
    save_portfolio(p)
    return p


# ---------------------------------------------------------------------------
# Trade recording
# ---------------------------------------------------------------------------
def record_trade(
    p: Portfolio,
    ticker: str,
    side: str,
    quantity: int,
    price: float,
    mode: str = "PAPER",
    note: str = "",
) -> Trade:
    total = quantity * price
    if side == "BUY":
        if total > p.cash:
            raise ValueError("Insufficient cash for this BUY order.")
        p.cash -= total
        pos = p.positions.get(ticker)
        if pos:
            new_qty = pos.quantity + quantity
            pos.avg_price = (pos.avg_price * pos.quantity + price * quantity) / new_qty
            pos.quantity = new_qty
        else:
            p.positions[ticker] = Position(ticker, quantity, price)
    elif side == "SELL":
        pos = p.positions.get(ticker)
        if not pos or pos.quantity < quantity:
            raise ValueError("Not enough shares to sell.")
        pos.quantity -= quantity
        p.cash += total
        if pos.quantity == 0:
            del p.positions[ticker]
    else:
        raise ValueError(f"Unknown side {side!r}")

    trade = Trade(
        timestamp=datetime.now(timezone.utc).isoformat(),
        ticker=ticker, side=side, quantity=quantity, price=price,
        total=total, mode=mode, note=note,
    )
    p.history.append(trade)
    save_portfolio(p)
    return trade


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------
def calculate_portfolio_value(p: Portfolio, prices: Dict[str, float]) -> float:
    total = p.cash
    for tkr, pos in p.positions.items():
        total += pos.market_value(prices.get(tkr, pos.avg_price))
    return total


def calculate_daily_returns(equity: List[Dict]) -> pd.Series:
    if not equity:
        return pd.Series(dtype=float)
    df = pd.DataFrame(equity)
    if "value" not in df:
        return pd.Series(dtype=float)
    return df["value"].pct_change().dropna()


def calculate_drawdown(equity_values: pd.Series) -> pd.Series:
    if equity_values.empty:
        return equity_values
    peaks = equity_values.cummax()
    return (equity_values - peaks) / peaks


def calculate_sharpe_ratio(returns: pd.Series, risk_free: float = 0.04) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    excess = returns - (risk_free / 252)
    return float(np.sqrt(252) * excess.mean() / returns.std())


def calculate_cagr(initial: float, final: float, days: int) -> float:
    if initial <= 0 or days <= 0:
        return 0.0
    years = days / 365.25
    return (final / initial) ** (1 / years) - 1


def update_equity_history(p: Portfolio, prices: Dict[str, float]) -> Portfolio:
    value = calculate_portfolio_value(p, prices)
    p.equity_history.append(
        {"timestamp": datetime.now(timezone.utc).isoformat(), "value": round(value, 2)}
    )
    # keep at most last 500 snapshots
    p.equity_history = p.equity_history[-500:]
    save_portfolio(p)
    return p


def portfolio_summary(p: Portfolio, prices: Dict[str, float]) -> Dict:
    value = calculate_portfolio_value(p, prices)
    invested = sum(pos.avg_price * pos.quantity for pos in p.positions.values())
    market = sum(pos.market_value(prices.get(t, pos.avg_price)) for t, pos in p.positions.items())
    unrealized = market - invested
    total_return = (value - p.initial_capital) / p.initial_capital if p.initial_capital else 0.0
    exposure = (market / value) if value else 0.0

    equity_values = pd.Series([e["value"] for e in p.equity_history], dtype=float)
    dd = calculate_drawdown(equity_values).min() if not equity_values.empty else 0.0
    sharpe = calculate_sharpe_ratio(equity_values.pct_change().dropna())
    days = len(p.equity_history)
    cagr = calculate_cagr(p.initial_capital, value, days) if days > 1 else 0.0

    return {
        "value": round(value, 2),
        "cash": round(p.cash, 2),
        "invested": round(invested, 2),
        "market_value": round(market, 2),
        "unrealized_pnl": round(unrealized, 2),
        "total_return": round(total_return, 4),
        "exposure": round(exposure, 4),
        "max_drawdown": round(float(dd) if dd is not None else 0.0, 4),
        "sharpe": round(sharpe, 2),
        "cagr": round(cagr, 4),
        "n_positions": len(p.positions),
        "n_trades": len(p.history),
    }


def allocation_breakdown(p: Portfolio, prices: Dict[str, float]) -> pd.DataFrame:
    if not p.positions:
        return pd.DataFrame(columns=["Ticker", "Value", "Allocation"])
    rows = []
    total_value = calculate_portfolio_value(p, prices)
    rows.append({"Ticker": "CASH", "Value": p.cash,
                 "Allocation": p.cash / total_value if total_value else 0})
    for tkr, pos in p.positions.items():
        mv = pos.market_value(prices.get(tkr, pos.avg_price))
        rows.append({"Ticker": tkr, "Value": mv,
                     "Allocation": mv / total_value if total_value else 0})
    return pd.DataFrame(rows)
