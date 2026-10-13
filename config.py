"""
config.py — Central configuration management.

Loads all secrets and runtime settings from a .env file. Other modules import
from this file so we have one single place to manage configuration.

Why:
- Keep API keys out of source code.
- Easy to switch between paper/live, change capital, etc.
"""

from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from the module directory first, then fall back to /app/backend/.env
# so the Emergent universal LLM key is always available.
_MODULE_DIR = Path(__file__).resolve().parent
load_dotenv(_MODULE_DIR / ".env")
load_dotenv("/app/backend/.env", override=False)


def _get(key: str, default: str = "") -> str:
    value = os.environ.get(key, default)
    return value.strip() if isinstance(value, str) else value


# ---------------------------------------------------------------------------
# LLM / AI
# ---------------------------------------------------------------------------
EMERGENT_LLM_KEY: str = _get("EMERGENT_LLM_KEY")
OPENAI_MODEL: str = _get("OPENAI_MODEL", "gpt-4o")

# ---------------------------------------------------------------------------
# Broker — Zerodha Kite Connect
# ---------------------------------------------------------------------------
KITE_API_KEY: str = _get("KITE_API_KEY", "your_kite_api_key_here")
KITE_API_SECRET: str = _get("KITE_API_SECRET", "your_kite_api_secret_here")
KITE_ACCESS_TOKEN: str = _get("KITE_ACCESS_TOKEN", "")

# ---------------------------------------------------------------------------
# Market data
# ---------------------------------------------------------------------------
ALPHAVANTAGE_API_KEY: str = _get("ALPHAVANTAGE_API_KEY", "your_alphavantage_key_here")

# ---------------------------------------------------------------------------
# Trading parameters
# ---------------------------------------------------------------------------
DEFAULT_CAPITAL: float = float(_get("DEFAULT_CAPITAL", "100000"))
MAX_RISK_PER_TRADE: float = float(_get("MAX_RISK_PER_TRADE", "0.02"))      # 2%
MAX_PORTFOLIO_EXPOSURE: float = float(_get("MAX_PORTFOLIO_EXPOSURE", "0.80"))
MAX_DAILY_DRAWDOWN: float = float(_get("MAX_DAILY_DRAWDOWN", "0.05"))      # 5%
DEFAULT_STOP_LOSS_PCT: float = float(_get("DEFAULT_STOP_LOSS_PCT", "0.03"))
DEFAULT_TAKE_PROFIT_PCT: float = float(_get("DEFAULT_TAKE_PROFIT_PCT", "0.06"))

# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------
PAPER_TRADING: bool = _get("PAPER_TRADING", "true").lower() in {"1", "true", "yes"}

# ---------------------------------------------------------------------------
# Data paths
# ---------------------------------------------------------------------------
DATA_DIR: Path = _MODULE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
PORTFOLIO_FILE: Path = DATA_DIR / "portfolio.csv"
TRADES_FILE: Path = DATA_DIR / "trades.csv"


def is_kite_configured() -> bool:
    """Returns True only when real Kite credentials are present."""
    return (
        KITE_API_KEY
        and not KITE_API_KEY.startswith("your_")
        and KITE_API_SECRET
        and not KITE_API_SECRET.startswith("your_")
    )


def is_alphavantage_configured() -> bool:
    return ALPHAVANTAGE_API_KEY and not ALPHAVANTAGE_API_KEY.startswith("your_")


def is_llm_configured() -> bool:
    return bool(EMERGENT_LLM_KEY)
