"""
backtest.py — Historical strategy backtester.

A lightweight, dependency-free vectorised backtester. Given an OHLCV
DataFrame, it simulates day-by-day position changes against the signal
produced by one of the strategies and reports a standard set of performance
metrics.

Why not `backtesting.py`?
    For an MVP, an in-house engine is ~80 LOC, easy to read, easy to extend,
    and avoids another runtime dependency. Swap to `backtesting.py` later if
    you need limit orders / commissions modelling.

Supported strategies
--------------------
* sma     — SMA crossover (sma_strategy)
* rsi     — RSI mean reversion (rsi_strategy)
* macd    — MACD crossover (macd_strategy)
* combined— Majority vote of SMA + RSI + MACD (generate_signals)
* ml      — RandomForest / LogisticRegression alpha (alpha_generator)
* buyhold — Naive buy-and-hold baseline
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Dict, List

import numpy as np
import pandas as pd

from . import strategy as strat
from . import alpha_generator as alpha


STRATEGIES = ["sma", "rsi", "macd", "combined", "ml", "buyhold"]


@dataclass
class BacktestResult:
    strategy: str
    initial_capital: float
    final_equity: float
    total_return: float
    cagr: float
    sharpe: float
    max_drawdown: float
    win_rate: float
    n_trades: int
    equity_curve: pd.Series        # daily equity (indexed by date)


# ---------------------------------------------------------------------------
# Signal builders → produce a +1 / 0 / -1 position series indexed by date
# ---------------------------------------------------------------------------
def _signal_series(df: pd.DataFrame, name: str) -> pd.Series:
    if name == "sma":
        s = strat.sma_strategy(df)["Signal"]
    elif name == "rsi":
        s = strat.rsi_strategy(df)["Signal"]
    elif name == "macd":
        s = strat.macd_strategy(df)["Signal"]
    elif name == "combined":
        s = strat.generate_signals(df)["CombinedSignal"]
    elif name == "buyhold":
        s = pd.Series(1, index=df.index)
    elif name == "ml":
        s = _ml_signal_series(df)
    else:
        raise ValueError(f"Unknown strategy: {name}")
    # Shift by 1 so we trade on the NEXT bar — avoids look-ahead bias.
    return s.shift(1).fillna(0).astype(int).clip(-1, 1)


def _ml_signal_series(df: pd.DataFrame, train_frac: float = 0.6) -> pd.Series:
    """Train the ML model on the first `train_frac` of the data, then emit
    +1/-1/0 for the remaining bars based on predicted probability."""
    feats = alpha.create_features(df)
    if len(feats) < 80:
        return pd.Series(0, index=df.index)

    cutoff = int(len(feats) * train_frac)
    train_df = feats.iloc[:cutoff]
    test_df = feats.iloc[cutoff:]

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler().fit(train_df[alpha.FEATURE_COLS].values)
    model = RandomForestClassifier(
        n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
    ).fit(scaler.transform(train_df[alpha.FEATURE_COLS].values),
          train_df["Target"].values)

    probs = model.predict_proba(scaler.transform(test_df[alpha.FEATURE_COLS].values))
    buy_p = probs[:, 1]
    sig = np.where(buy_p >= 0.55, 1, np.where(buy_p <= 0.45, -1, 0))

    out = pd.Series(0, index=df.index)
    out.loc[test_df.index] = sig
    return out


# ---------------------------------------------------------------------------
# Core engine
# ---------------------------------------------------------------------------
def run_backtest(
    df: pd.DataFrame,
    strategy_name: str,
    initial_capital: float = 100_000.0,
    commission: float = 0.0005,        # 5 bps per trade
) -> BacktestResult:
    """Simulate a long/flat/short strategy and compute metrics."""
    if df is None or df.empty or "Close" not in df.columns:
        return _empty(strategy_name, initial_capital)

    df = df.copy()
    returns = df["Close"].pct_change().fillna(0)
    position = _signal_series(df, strategy_name)

    # Commission cost when position changes
    trade_costs = position.diff().abs().fillna(0) * commission
    strat_returns = position * returns - trade_costs

    equity_curve = (1 + strat_returns).cumprod() * initial_capital
    equity_curve.index = df.index
    equity_curve.name = strategy_name

    final = float(equity_curve.iloc[-1])
    total_return = final / initial_capital - 1
    days = max((df.index[-1] - df.index[0]).days, 1)
    years = days / 365.25
    cagr = (final / initial_capital) ** (1 / years) - 1 if years > 0 else 0.0
    sharpe = (
        np.sqrt(252) * strat_returns.mean() / strat_returns.std()
        if strat_returns.std() > 0 else 0.0
    )
    peaks = equity_curve.cummax()
    max_dd = float(((equity_curve - peaks) / peaks).min()) if not equity_curve.empty else 0.0

    # Win rate / trades — based on position-changes (entries)
    entries = position.diff().abs() > 0
    n_trades = int(entries.sum())
    bar_outcomes = strat_returns[position.shift(1).fillna(0).astype(bool)]
    wins = int((bar_outcomes > 0).sum())
    win_rate = wins / max(len(bar_outcomes), 1)

    return BacktestResult(
        strategy=strategy_name,
        initial_capital=initial_capital,
        final_equity=round(final, 2),
        total_return=round(float(total_return), 4),
        cagr=round(float(cagr), 4),
        sharpe=round(float(sharpe), 2),
        max_drawdown=round(max_dd, 4),
        win_rate=round(float(win_rate), 4),
        n_trades=n_trades,
        equity_curve=equity_curve,
    )


def compare_strategies(
    df: pd.DataFrame,
    strategies: List[str],
    initial_capital: float = 100_000.0,
) -> Dict[str, BacktestResult]:
    return {s: run_backtest(df, s, initial_capital) for s in strategies}


def results_to_frame(results: Dict[str, BacktestResult]) -> pd.DataFrame:
    rows = []
    for r in results.values():
        d = asdict(r)
        d.pop("equity_curve", None)
        rows.append(d)
    return pd.DataFrame(rows)


def _empty(name: str, capital: float) -> BacktestResult:
    return BacktestResult(
        strategy=name, initial_capital=capital, final_equity=capital,
        total_return=0.0, cagr=0.0, sharpe=0.0, max_drawdown=0.0,
        win_rate=0.0, n_trades=0, equity_curve=pd.Series(dtype=float),
    )
