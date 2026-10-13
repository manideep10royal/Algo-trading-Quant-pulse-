"""
market_data.py — Market data collection & preprocessing.

Primary source : yfinance
Optional source: Alpha Vantage (when ALPHAVANTAGE_API_KEY is configured)

All public functions return a clean pandas DataFrame containing standard OHLCV
columns (Open, High, Low, Close, Volume) indexed by Date.
"""

from __future__ import annotations

from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import requests

from .config import ALPHAVANTAGE_API_KEY, is_alphavantage_configured
from .utils import setup_logger

log = setup_logger("market_data")


# ---------------------------------------------------------------------------
# yfinance (default)
# ---------------------------------------------------------------------------
def fetch_stock_data(
    ticker: str,
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a single ticker via yfinance.

    Parameters
    ----------
    ticker   : e.g. "AAPL", "RELIANCE.NS"
    period   : 1d,5d,1mo,3mo,6mo,1y,2y,5y,10y,ytd,max
    interval : 1m,2m,5m,15m,30m,60m,90m,1h,1d,5d,1wk,1mo,3mo
    """
    try:
        df = yf.download(
            ticker, period=period, interval=interval,
            progress=False, auto_adjust=False, threads=False,
        )
        if df is None or df.empty:
            log.warning("No data returned for %s", ticker)
            return pd.DataFrame()

        # yfinance sometimes returns MultiIndex columns when a single ticker
        # is fetched together with auto_adjust — flatten it.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename_axis("Date").reset_index()
        df["Date"] = pd.to_datetime(df["Date"])
        df = df.set_index("Date")
        df = clean_data(df)
        df = calculate_returns(df)
        df = calculate_volatility(df)
        return df
    except Exception as exc:  # noqa: BLE001
        log.exception("yfinance failure for %s: %s", ticker, exc)
        return pd.DataFrame()


def fetch_multiple_stocks(
    tickers: List[str],
    period: str = "1y",
    interval: str = "1d",
) -> Dict[str, pd.DataFrame]:
    """Return a dict {ticker: DataFrame} for several tickers."""
    return {t: fetch_stock_data(t, period, interval) for t in tickers}


# ---------------------------------------------------------------------------
# Alpha Vantage (optional secondary)
# ---------------------------------------------------------------------------
def fetch_alphavantage_daily(symbol: str) -> pd.DataFrame:
    """Fetch daily OHLCV via Alpha Vantage. Requires ALPHAVANTAGE_API_KEY."""
    if not is_alphavantage_configured():
        log.info("Alpha Vantage not configured — skipping.")
        return pd.DataFrame()
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "TIME_SERIES_DAILY",
        "symbol": symbol,
        "outputsize": "compact",
        "apikey": ALPHAVANTAGE_API_KEY,
    }
    try:
        resp = requests.get(url, params=params, timeout=15)
        raw = resp.json().get("Time Series (Daily)", {})
        if not raw:
            return pd.DataFrame()
        df = pd.DataFrame(raw).T.astype(float)
        df.columns = ["Open", "High", "Low", "Close", "Volume"]
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        df = clean_data(df)
        df = calculate_returns(df)
        df = calculate_volatility(df)
        return df
    except Exception as exc:  # noqa: BLE001
        log.exception("Alpha Vantage failure for %s: %s", symbol, exc)
        return pd.DataFrame()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Drop fully empty rows and forward-fill small gaps."""
    if df.empty:
        return df
    df = df.copy()
    df = df.dropna(how="all")
    df = df.ffill().bfill()
    # Keep only standard columns when present
    keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
    return df[keep] if keep else df


def calculate_returns(df: pd.DataFrame) -> pd.DataFrame:
    if "Close" not in df.columns:
        return df
    df = df.copy()
    df["Returns"] = df["Close"].pct_change()
    df["LogReturns"] = np.log(df["Close"] / df["Close"].shift(1))
    return df


def calculate_volatility(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    if "Returns" not in df.columns:
        return df
    df = df.copy()
    df["Volatility"] = df["Returns"].rolling(window).std() * np.sqrt(252)
    return df


def latest_price(df: pd.DataFrame) -> Optional[float]:
    if df is None or df.empty or "Close" not in df.columns:
        return None
    return float(df["Close"].iloc[-1])
