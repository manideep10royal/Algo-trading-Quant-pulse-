"""
execution.py — Trade execution layer.

Wraps Zerodha Kite Connect (real, fully wired) and Groww (stub — Groww's
official broker API is in private/approval-only beta). Provides a unified
broker interface so the dashboard doesn't need to know which is selected.

Always defaults to PAPER mode. Live execution only happens when:
    1. A broker session is active (e.g., Kite access_token valid), AND
    2. The caller explicitly passes `live=True`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict, List

from .config import (
    KITE_API_KEY,
    KITE_API_SECRET,
    KITE_ACCESS_TOKEN,
    PAPER_TRADING,
    is_kite_configured,
)
from .portfolio import Portfolio, record_trade
from .utils import setup_logger

log = setup_logger("execution")


@dataclass
class OrderResult:
    success: bool
    order_id: str
    mode: str            # PAPER / LIVE
    ticker: str
    side: str
    quantity: int
    price: float
    message: str


# ---------------------------------------------------------------------------
# Kite Connect (real) — runtime-credentials aware
# ---------------------------------------------------------------------------
def _kite_runtime(api_key: str = "", access_token: str = ""):
    """Build a KiteConnect client from runtime creds (overrides .env)."""
    api_key = api_key or KITE_API_KEY
    access_token = access_token or KITE_ACCESS_TOKEN
    if (not api_key) or api_key.startswith("your_") or not access_token:
        return None
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=api_key)
        kite.set_access_token(access_token)
        return kite
    except Exception as exc:  # noqa: BLE001
        log.exception("Kite init failed: %s", exc)
        return None


def _kite():
    """Kite client built from .env creds only (legacy path)."""
    return _kite_runtime()


def kite_login_url(api_key: str = "") -> Optional[str]:
    """Return the Kite Connect login URL for the given api_key (or .env)."""
    api_key = api_key or KITE_API_KEY
    if not api_key or api_key.startswith("your_"):
        return None
    try:
        from kiteconnect import KiteConnect
        return KiteConnect(api_key=api_key).login_url()
    except Exception as exc:  # noqa: BLE001
        log.exception("Kite login_url failed: %s", exc)
        return None


def kite_generate_session(
    request_token: str,
    api_key: str = "",
    api_secret: str = "",
) -> Dict:
    """Exchange request_token → access_token.

    Returns a dict with `access_token`, `user_id`, `user_name`, etc., or
    `{'error': ...}` on failure.
    """
    api_key = api_key or KITE_API_KEY
    api_secret = api_secret or KITE_API_SECRET
    if not api_key or api_key.startswith("your_"):
        return {"error": "Kite API key not configured."}
    if not api_secret or api_secret.startswith("your_"):
        return {"error": "Kite API secret not configured."}
    try:
        from kiteconnect import KiteConnect
        kite = KiteConnect(api_key=api_key)
        return kite.generate_session(request_token, api_secret=api_secret)
    except Exception as exc:  # noqa: BLE001
        log.exception("Kite session generation failed: %s", exc)
        return {"error": str(exc)}


def kite_profile(api_key: str, access_token: str) -> Dict:
    """Fetch the logged-in Kite user's profile (for the 'Connected as …' UI)."""
    k = _kite_runtime(api_key, access_token)
    if k is None:
        return {"error": "Kite client not available."}
    try:
        return k.profile()
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


def kite_holdings(api_key: str, access_token: str) -> List[Dict]:
    """Long-term holdings from the connected Kite account."""
    k = _kite_runtime(api_key, access_token)
    if k is None:
        return []
    try:
        return k.holdings()
    except Exception as exc:  # noqa: BLE001
        log.exception("Kite holdings fetch failed: %s", exc)
        return []


def kite_positions(api_key: str, access_token: str) -> Dict:
    """Net & day positions from the connected Kite account."""
    k = _kite_runtime(api_key, access_token)
    if k is None:
        return {"net": [], "day": []}
    try:
        return k.positions()
    except Exception as exc:  # noqa: BLE001
        log.exception("Kite positions fetch failed: %s", exc)
        return {"net": [], "day": []}


def kite_orders(api_key: str, access_token: str) -> List[Dict]:
    k = _kite_runtime(api_key, access_token)
    if k is None:
        return []
    try:
        return k.orders()
    except Exception as exc:  # noqa: BLE001
        log.exception("Kite orders fetch failed: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Groww (stub) — official API is in private beta / approval-only
# ---------------------------------------------------------------------------
def groww_status() -> Dict:
    """Return Groww integration status. Currently unavailable to the public."""
    return {
        "available": False,
        "message": (
            "Groww's official broker API is currently in private beta. "
            "Apply for access at api-support@groww.in and we'll wire your "
            "API key here once granted."
        ),
        "apply_email": "api-support@groww.in",
    }


def groww_place_order(*args, **kwargs) -> OrderResult:
    return OrderResult(
        success=False, order_id="", mode="PAPER", ticker=kwargs.get("ticker", ""),
        side=kwargs.get("side", ""), quantity=kwargs.get("quantity", 0),
        price=kwargs.get("price", 0.0),
        message="Groww API not yet available — falling back to paper.",
    )


# ---------------------------------------------------------------------------
# Public order API (unified)
# ---------------------------------------------------------------------------
_seen_orders: set = set()      # very simple in-memory de-dup


def _dedup_key(ticker: str, side: str, quantity: int, price: float) -> str:
    return f"{ticker}:{side}:{quantity}:{round(price, 2)}"


def place_buy_order(
    portfolio: Portfolio,
    ticker: str,
    quantity: int,
    price: float,
    live: bool = False,
    note: str = "",
    broker: str = "kite",
    kite_api_key: str = "",
    kite_access_token: str = "",
) -> OrderResult:
    return _place_order(portfolio, ticker, "BUY", quantity, price, live, note,
                        broker, kite_api_key, kite_access_token)


def place_sell_order(
    portfolio: Portfolio,
    ticker: str,
    quantity: int,
    price: float,
    live: bool = False,
    note: str = "",
    broker: str = "kite",
    kite_api_key: str = "",
    kite_access_token: str = "",
) -> OrderResult:
    return _place_order(portfolio, ticker, "SELL", quantity, price, live, note,
                        broker, kite_api_key, kite_access_token)


def _place_order(
    portfolio: Portfolio,
    ticker: str,
    side: str,
    quantity: int,
    price: float,
    live: bool,
    note: str,
    broker: str = "kite",
    kite_api_key: str = "",
    kite_access_token: str = "",
) -> OrderResult:
    if quantity <= 0 or price <= 0:
        return OrderResult(False, "", "PAPER", ticker, side, quantity, price,
                           "Invalid quantity or price.")

    key = _dedup_key(ticker, side, quantity, price)
    if key in _seen_orders:
        return OrderResult(False, "", "PAPER", ticker, side, quantity, price,
                           "Duplicate order blocked.")
    _seen_orders.add(key)

    # Decide live vs paper
    use_live = bool(live and broker == "kite")
    kite = _kite_runtime(kite_api_key, kite_access_token) if use_live else None
    if use_live and kite is None:
        return OrderResult(False, "", "PAPER", ticker, side, quantity, price,
                           "Live requested but broker session not active.")
    if live and broker == "groww":
        return groww_place_order(ticker=ticker, side=side, quantity=quantity, price=price)
    mode = "LIVE" if (use_live and kite is not None) else "PAPER"

    if mode == "LIVE":
        try:
            order_id = kite.place_order(
                variety=kite.VARIETY_REGULAR,
                exchange=kite.EXCHANGE_NSE,
                tradingsymbol=ticker.replace(".NS", ""),
                transaction_type=kite.TRANSACTION_TYPE_BUY if side == "BUY"
                                  else kite.TRANSACTION_TYPE_SELL,
                quantity=quantity,
                product=kite.PRODUCT_MIS,
                order_type=kite.ORDER_TYPE_MARKET,
            )
        except Exception as exc:  # noqa: BLE001
            log.exception("Live order failed: %s", exc)
            return OrderResult(False, "", "LIVE", ticker, side, quantity, price,
                               f"Live order failed: {exc}")
    else:
        order_id = f"PAPER-{len(portfolio.history)+1:06d}"

    # Always reflect in the local paper-portfolio book of record
    try:
        record_trade(portfolio, ticker, side, quantity, price, mode=mode, note=note)
    except ValueError as exc:
        return OrderResult(False, "", mode, ticker, side, quantity, price, str(exc))

    return OrderResult(True, str(order_id), mode, ticker, side, quantity, price,
                       f"{mode} {side} {quantity} {ticker} @ {price:.2f}")


def cancel_order(order_id: str, api_key: str = "", access_token: str = "") -> Dict:
    """Cancel a live Kite order. Paper orders are immediate so nothing to cancel."""
    kite = _kite_runtime(api_key, access_token)
    if kite is None:
        return {"success": False, "message": "Broker not connected — paper orders are instant."}
    try:
        kite.cancel_order(variety=kite.VARIETY_REGULAR, order_id=order_id)
        return {"success": True, "message": f"Order {order_id} cancelled."}
    except Exception as exc:  # noqa: BLE001
        return {"success": False, "message": str(exc)}


def fetch_positions(
    portfolio: Portfolio,
    api_key: str = "",
    access_token: str = "",
) -> List[Dict]:
    """Live positions if Kite is up, otherwise paper positions."""
    kite = _kite_runtime(api_key, access_token)
    if kite is not None:
        try:
            return kite.positions().get("net", [])
        except Exception as exc:  # noqa: BLE001
            log.exception("Kite positions fetch failed: %s", exc)
    return [
        {"tradingsymbol": p.ticker, "quantity": p.quantity, "average_price": p.avg_price}
        for p in portfolio.positions.values()
    ]
