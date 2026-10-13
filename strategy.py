"""
strategy.py — Classical technical-analysis trading strategies.

Each strategy function takes an OHLCV DataFrame (with `Close` column) and
returns the SAME DataFrame plus a `Signal` column containing one of:
    +1  → BUY
     0  → HOLD
    -1  → SELL
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import ta

SignalAction = Literal["BUY", "SELL", "HOLD"]


def _label(sig: int) -> SignalAction:
    return "BUY" if sig > 0 else ("SELL" if sig < 0 else "HOLD")


# ---------------------------------------------------------------------------
# Indicators / strategies
# ---------------------------------------------------------------------------
def sma_strategy(df: pd.DataFrame, short: int = 20, long: int = 50) -> pd.DataFrame:
    """Simple Moving Average crossover."""
    df = df.copy()
    df[f"SMA_{short}"] = df["Close"].rolling(short).mean()
    df[f"SMA_{long}"] = df["Close"].rolling(long).mean()
    df["Signal"] = 0
    df.loc[df[f"SMA_{short}"] > df[f"SMA_{long}"], "Signal"] = 1
    df.loc[df[f"SMA_{short}"] < df[f"SMA_{long}"], "Signal"] = -1
    return df


def rsi_strategy(
    df: pd.DataFrame,
    period: int = 14,
    overbought: int = 70,
    oversold: int = 30,
) -> pd.DataFrame:
    df = df.copy()
    df["RSI"] = ta.momentum.RSIIndicator(df["Close"], window=period).rsi()
    df["Signal"] = 0
    df.loc[df["RSI"] < oversold, "Signal"] = 1
    df.loc[df["RSI"] > overbought, "Signal"] = -1
    return df


def macd_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    macd = ta.trend.MACD(df["Close"])
    df["MACD"] = macd.macd()
    df["MACD_Signal"] = macd.macd_signal()
    df["MACD_Hist"] = macd.macd_diff()
    df["Signal"] = 0
    df.loc[df["MACD"] > df["MACD_Signal"], "Signal"] = 1
    df.loc[df["MACD"] < df["MACD_Signal"], "Signal"] = -1
    return df


def add_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add RSI, MACD, SMA20/50, Bollinger Bands & Volume MA in one go.
    Useful for the ML model feature set and the chart panels."""
    out = df.copy()
    out["RSI"] = ta.momentum.RSIIndicator(out["Close"], window=14).rsi()
    macd = ta.trend.MACD(out["Close"])
    out["MACD"] = macd.macd()
    out["MACD_Signal"] = macd.macd_signal()
    out["MACD_Hist"] = macd.macd_diff()
    out["SMA_20"] = out["Close"].rolling(20).mean()
    out["SMA_50"] = out["Close"].rolling(50).mean()
    bb = ta.volatility.BollingerBands(out["Close"], window=20, window_dev=2)
    out["BB_High"] = bb.bollinger_hband()
    out["BB_Low"] = bb.bollinger_lband()
    if "Volume" in out.columns:
        out["Volume_MA_20"] = out["Volume"].rolling(20).mean()
    return out


# ---------------------------------------------------------------------------
# Combined / ensemble signal
# ---------------------------------------------------------------------------
def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """Combine SMA + RSI + MACD signals into a single 'CombinedSignal'.
    Majority vote — adds transparency for the UI."""
    sma_df = sma_strategy(df)[["Signal"]].rename(columns={"Signal": "sma"})
    rsi_df = rsi_strategy(df)[["Signal"]].rename(columns={"Signal": "rsi"})
    macd_df = macd_strategy(df)[["Signal"]].rename(columns={"Signal": "macd"})
    out = add_all_indicators(df)
    out = out.join([sma_df, rsi_df, macd_df])
    out["Vote"] = out[["sma", "rsi", "macd"]].sum(axis=1)
    out["CombinedSignal"] = np.where(
        out["Vote"] >= 2, 1, np.where(out["Vote"] <= -2, -1, 0)
    )
    out["SignalLabel"] = out["CombinedSignal"].apply(_label)
    return out


def latest_signal(df: pd.DataFrame) -> dict:
    """Return a structured snapshot of the most recent technical signal."""
    sig_df = generate_signals(df)
    if sig_df.empty:
        return {"action": "HOLD", "rsi": None, "macd": None, "vote": 0}
    row = sig_df.iloc[-1]
    return {
        "action": _label(int(row["CombinedSignal"])),
        "rsi": float(row["RSI"]) if not pd.isna(row["RSI"]) else None,
        "macd": float(row["MACD"]) if not pd.isna(row["MACD"]) else None,
        "macd_signal": float(row["MACD_Signal"]) if not pd.isna(row["MACD_Signal"]) else None,
        "sma_20": float(row["SMA_20"]) if not pd.isna(row["SMA_20"]) else None,
        "sma_50": float(row["SMA_50"]) if not pd.isna(row["SMA_50"]) else None,
        "vote": int(row["Vote"]),
        "votes": {
            "sma": int(row["sma"]) if not pd.isna(row["sma"]) else 0,
            "rsi": int(row["rsi"]) if not pd.isna(row["rsi"]) else 0,
            "macd": int(row["macd"]) if not pd.isna(row["macd"]) else 0,
        },
    }
