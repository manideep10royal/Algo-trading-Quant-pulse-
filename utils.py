"""
utils.py — Reusable utility functions used across modules.

Includes a configured logger, currency / percentage formatters, and a few
helper math functions.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from typing import Optional


def setup_logger(name: str = "algo_trading", level: int = logging.INFO) -> logging.Logger:
    """Return a singleton logger with a sensible default format."""
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    )
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def format_currency(amount: float, symbol: str = "₹") -> str:
    """Pretty-print a number as currency (default ₹, Indian convention)."""
    if amount is None:
        return f"{symbol}0.00"
    try:
        return f"{symbol}{amount:,.2f}"
    except (TypeError, ValueError):
        return f"{symbol}0.00"


def format_percentage(value: float, decimals: int = 2) -> str:
    if value is None:
        return "0.00%"
    return f"{value * 100:.{decimals}f}%"


def calculate_percentage_change(old: float, new: float) -> float:
    if not old:
        return 0.0
    return (new - old) / old


def format_date(d: Optional[datetime]) -> str:
    if d is None:
        return "—"
    return d.strftime("%Y-%m-%d %H:%M")
