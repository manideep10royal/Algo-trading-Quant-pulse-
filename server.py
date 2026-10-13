"""
QuantPulse — FastAPI backend.

Production-style modular monolith powering the React frontend.

Routes (all prefixed with /api):
- /auth/google/session, /auth/me, /auth/logout
- /broker/kite/login-url, /broker/kite/session, /broker/kite/status,
  /broker/kite/holdings, /broker/kite/positions, /broker/kite/disconnect
- /market/candles, /market/quote
- /strategies, /strategies/{id}/run
- /portfolio/paper, /portfolio/paper/orders, /portfolio/paper/reset
- /portfolio/live
- /risk/rules
- /ai/chat, /ai/insights
"""

from __future__ import annotations

import os
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional, List, Dict, Any

# Make /app importable so we can re-use the algo_trading_system package.
sys.path.insert(0, "/app")

import asyncio
import httpx
import pandas as pd
from fastapi import FastAPI, APIRouter, HTTPException, Request, Response, Cookie, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from motor.motor_asyncio import AsyncIOMotorClient
from pydantic import BaseModel, Field
from dotenv import load_dotenv

# Reuse all the existing quant logic.
from algo_trading_system import (
    market_data as md,
    strategy as strat,
    alpha_generator as alpha,
    risk_management as risk_mod,
    backtest as bt_mod,
    execution as ex_mod,
)
from algo_trading_system import optimizer as optz

load_dotenv("/app/backend/.env")
load_dotenv("/app/algo_trading_system/.env", override=False)

MONGO_URL = os.environ["MONGO_URL"]
DB_NAME = os.environ["DB_NAME"]

EMERGENT_AUTH_BASE = "https://demobackend.emergentagent.com/auth/v1/env"

mongo_client = AsyncIOMotorClient(MONGO_URL)
db = mongo_client[DB_NAME]

app = FastAPI(title="QuantPulse API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"https?://.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["set-cookie"],
)

router = APIRouter(prefix="/api")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class User(BaseModel):
    user_id: str
    email: str
    name: str
    picture: Optional[str] = None


async def _get_user_from_session(session_token: Optional[str]) -> Optional[Dict]:
    if not session_token:
        return None
    session = await db.user_sessions.find_one({"session_token": session_token}, {"_id": 0})
    if not session:
        return None
    expires_at = session.get("expires_at")
    if isinstance(expires_at, str):
        expires_at = datetime.fromisoformat(expires_at)
    if expires_at and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at and expires_at < datetime.now(timezone.utc):
        return None
    user = await db.users.find_one({"user_id": session["user_id"]}, {"_id": 0})
    return user


async def current_user(
    request: Request,
    session_token: Optional[str] = Cookie(default=None),
) -> User:
    """Resolve current user from cookie OR Authorization: Bearer header."""
    token = session_token
    if not token:
        auth = request.headers.get("authorization") or ""
        if auth.lower().startswith("bearer "):
            token = auth.split(" ", 1)[1].strip()
    user = await _get_user_from_session(token)
    if not user:
        raise HTTPException(status_code=401, detail="Not authenticated")
    return User(**user)


# ---------------------------------------------------------------------------
# AUTH
# ---------------------------------------------------------------------------
class SessionExchangeBody(BaseModel):
    session_id: str


@router.post("/auth/google/session")
async def auth_session(body: SessionExchangeBody, response: Response):
    """Exchange Emergent auth session_id for a persistent server cookie."""
    async with httpx.AsyncClient(timeout=15) as cli:
        r = await cli.get(
            f"{EMERGENT_AUTH_BASE}/oauth/session-data",
            headers={"X-Session-ID": body.session_id},
        )
        if r.status_code != 200:
            raise HTTPException(status_code=401, detail="Invalid session_id")
        data = r.json()

    email = data["email"]
    name = data.get("name", email)
    picture = data.get("picture")
    session_token = data["session_token"]

    existing = await db.users.find_one({"email": email}, {"_id": 0})
    if existing:
        user_id = existing["user_id"]
        await db.users.update_one(
            {"user_id": user_id},
            {"$set": {"name": name, "picture": picture,
                      "last_login": datetime.now(timezone.utc)}},
        )
    else:
        user_id = f"user_{uuid.uuid4().hex[:12]}"
        await db.users.insert_one({
            "user_id": user_id, "email": email, "name": name, "picture": picture,
            "created_at": datetime.now(timezone.utc),
        })

    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    await db.user_sessions.insert_one({
        "user_id": user_id, "session_token": session_token,
        "expires_at": expires_at, "created_at": datetime.now(timezone.utc),
    })

    response.set_cookie(
        key="session_token", value=session_token,
        max_age=7 * 24 * 60 * 60, path="/",
        secure=True, httponly=True, samesite="none",
    )
    return {"user_id": user_id, "email": email, "name": name, "picture": picture}


@router.get("/auth/me")
async def auth_me(user: User = Depends(current_user)):
    return user.model_dump()


@router.post("/auth/logout")
async def auth_logout(
    response: Response,
    session_token: Optional[str] = Cookie(default=None),
):
    if session_token:
        await db.user_sessions.delete_one({"session_token": session_token})
    response.delete_cookie("session_token", path="/")
    return {"ok": True}


# ---------------------------------------------------------------------------
# BROKER — Zerodha Kite
# ---------------------------------------------------------------------------
class KiteCredsBody(BaseModel):
    api_key: str
    api_secret: Optional[str] = ""
    request_token: Optional[str] = None


@router.post("/broker/kite/login-url")
async def kite_login_url(body: KiteCredsBody, user: User = Depends(current_user)):
    url = ex_mod.kite_login_url(body.api_key)
    if not url:
        raise HTTPException(status_code=400, detail="Invalid API key")
    # Persist creds (api_key only) on user record
    await db.users.update_one(
        {"user_id": user.user_id},
        {"$set": {"kite.api_key": body.api_key,
                  "kite.api_secret_set": bool(body.api_secret)}},
    )
    return {"login_url": url}


@router.post("/broker/kite/session")
async def kite_session(body: KiteCredsBody, user: User = Depends(current_user)):
    if not body.request_token:
        raise HTTPException(status_code=400, detail="request_token required")
    result = ex_mod.kite_generate_session(
        body.request_token, body.api_key, body.api_secret or "",
    )
    if result.get("error"):
        raise HTTPException(status_code=400, detail=result["error"])
    access_token = result.get("access_token", "")
    await db.users.update_one(
        {"user_id": user.user_id},
        {"$set": {
            "kite.api_key": body.api_key,
            "kite.access_token": access_token,
            "kite.user_name": result.get("user_name") or result.get("user_id"),
            "kite.connected_at": datetime.now(timezone.utc),
        }},
    )
    return {
        "connected": True,
        "user_name": result.get("user_name") or result.get("user_id"),
    }


async def _kite_creds(user_id: str):
    u = await db.users.find_one({"user_id": user_id}, {"_id": 0})
    kite = (u or {}).get("kite") or {}
    return kite.get("api_key", ""), kite.get("access_token", "")


@router.get("/broker/kite/status")
async def kite_status(user: User = Depends(current_user)):
    u = await db.users.find_one({"user_id": user.user_id}, {"_id": 0})
    kite = (u or {}).get("kite") or {}
    return {
        "connected": bool(kite.get("access_token")),
        "user_name": kite.get("user_name"),
        "connected_at": kite.get("connected_at"),
    }


@router.get("/broker/kite/holdings")
async def kite_holdings(user: User = Depends(current_user)):
    api_key, access_token = await _kite_creds(user.user_id)
    return {"holdings": ex_mod.kite_holdings(api_key, access_token)}


@router.get("/broker/kite/positions")
async def kite_positions(user: User = Depends(current_user)):
    api_key, access_token = await _kite_creds(user.user_id)
    return ex_mod.kite_positions(api_key, access_token)


@router.post("/broker/kite/disconnect")
async def kite_disconnect(user: User = Depends(current_user)):
    await db.users.update_one(
        {"user_id": user.user_id},
        {"$unset": {"kite": ""}},
    )
    return {"ok": True}


# ---------------------------------------------------------------------------
# MARKET DATA
# ---------------------------------------------------------------------------
@router.get("/market/candles")
async def market_candles(
    ticker: str, period: str = "1y", interval: str = "1d",
    user: User = Depends(current_user),
):
    df = md.fetch_stock_data(ticker, period=period, interval=interval)
    if df.empty:
        return {"ticker": ticker, "candles": []}
    out = []
    for ts, row in df.iterrows():
        out.append({
            "time": int(pd.Timestamp(ts).timestamp()),
            "open": float(row["Open"]), "high": float(row["High"]),
            "low": float(row["Low"]), "close": float(row["Close"]),
            "volume": int(row["Volume"]) if "Volume" in row and pd.notna(row["Volume"]) else 0,
        })
    return {"ticker": ticker, "candles": out}


@router.get("/market/quote")
async def market_quote(ticker: str, user: User = Depends(current_user)):
    df = md.fetch_stock_data(ticker, period="5d", interval="1d")
    if df.empty:
        raise HTTPException(404, "No data")
    last = df["Close"].iloc[-1]
    prev = df["Close"].iloc[-2] if len(df) > 1 else last
    return {
        "ticker": ticker, "price": float(last),
        "change": float(last - prev),
        "change_pct": float((last - prev) / prev) if prev else 0.0,
    }


# ---------------------------------------------------------------------------
# STRATEGIES — 10 prebuilt
# ---------------------------------------------------------------------------
PREBUILT_STRATEGIES = [
    {"id": "sma_cross", "name": "Moving Average Crossover", "description": "Long when SMA(20) crosses above SMA(50).", "risk_score": 3, "category": "Trend"},
    {"id": "rsi_reversal", "name": "RSI Reversal", "description": "Buy oversold (RSI<30), sell overbought (RSI>70).", "risk_score": 4, "category": "Mean Reversion"},
    {"id": "macd_momentum", "name": "MACD Momentum", "description": "Trade MACD line vs signal line crossovers.", "risk_score": 3, "category": "Momentum"},
    {"id": "bollinger_breakout", "name": "Bollinger Band Breakout", "description": "Long breakouts above upper band.", "risk_score": 4, "category": "Breakout"},
    {"id": "vwap_intraday", "name": "VWAP Intraday", "description": "Mean reversion around VWAP for intraday.", "risk_score": 4, "category": "Intraday"},
    {"id": "mean_reversion", "name": "Mean Reversion", "description": "Z-score reversion to 20-day mean.", "risk_score": 3, "category": "Mean Reversion"},
    {"id": "momentum", "name": "Momentum Strategy", "description": "12-1 month momentum factor.", "risk_score": 4, "category": "Momentum"},
    {"id": "breakout", "name": "Channel Breakout", "description": "Donchian channel breakout.", "risk_score": 4, "category": "Breakout"},
    {"id": "supertrend", "name": "Supertrend Strategy", "description": "Follows ATR-based Supertrend direction.", "risk_score": 3, "category": "Trend"},
    {"id": "ai_trend", "name": "AI Trend Prediction", "description": "ML (RandomForest) probability-based long/short.", "risk_score": 5, "category": "AI"},
]


@router.get("/strategies")
async def list_strategies(user: User = Depends(current_user)):
    return {"strategies": PREBUILT_STRATEGIES}


class StrategyRunBody(BaseModel):
    ticker: str
    period: str = "1y"
    interval: str = "1d"
    initial_capital: float = 100_000


def _strategy_to_backtest_name(sid: str) -> str:
    mapping = {
        "sma_cross": "sma", "rsi_reversal": "rsi", "macd_momentum": "macd",
        "bollinger_breakout": "combined", "vwap_intraday": "combined",
        "mean_reversion": "rsi", "momentum": "combined", "breakout": "combined",
        "supertrend": "macd", "ai_trend": "ml",
    }
    return mapping.get(sid, "combined")


@router.post("/strategies/{strategy_id}/run")
async def run_strategy(
    strategy_id: str, body: StrategyRunBody,
    user: User = Depends(current_user),
):
    df = md.fetch_stock_data(body.ticker, period=body.period, interval=body.interval)
    if df.empty:
        raise HTTPException(404, "No market data")

    # Live signal snapshot
    snapshot = strat.latest_signal(df)
    pred = alpha.predict_signal(df, model_type="random_forest")

    # Integrated backtest
    bt_name = _strategy_to_backtest_name(strategy_id)
    bt_result = bt_mod.run_backtest(df, bt_name, initial_capital=body.initial_capital)

    # Risk validation
    last_price = float(df["Close"].iloc[-1])
    val = risk_mod.validate_trade(body.initial_capital, last_price, "BUY")

    # Equity curve as list[{time, value}]
    eq = []
    if not bt_result.equity_curve.empty:
        for ts, v in bt_result.equity_curve.items():
            eq.append({"time": int(pd.Timestamp(ts).timestamp()), "value": float(v)})

    return {
        "strategy_id": strategy_id,
        "ticker": body.ticker,
        "snapshot": snapshot,
        "ml_prediction": {
            "action": pred.action, "buy_probability": pred.buy_probability,
            "sell_probability": pred.sell_probability, "confidence": pred.confidence,
            "accuracy": pred.model_accuracy,
        },
        "backtest": {
            "strategy": bt_result.strategy, "final_equity": bt_result.final_equity,
            "total_return": bt_result.total_return, "cagr": bt_result.cagr,
            "sharpe": bt_result.sharpe, "max_drawdown": bt_result.max_drawdown,
            "win_rate": bt_result.win_rate, "n_trades": bt_result.n_trades,
            "equity_curve": eq,
        },
        "risk_validation": {
            "is_valid": val.is_valid, "reason": val.reason,
            "suggested_quantity": val.suggested_quantity,
            "stop_loss": val.stop_loss, "take_profit": val.take_profit,
            "risk_amount": val.risk_amount, "risk_reward_ratio": val.risk_reward_ratio,
        },
    }


# ---------------------------------------------------------------------------
# PORTFOLIO — paper
# ---------------------------------------------------------------------------
DEFAULT_RISK_RULES = {
    "max_risk_per_trade": 0.02, "max_daily_drawdown": 0.05,
    "max_portfolio_exposure": 0.80, "stop_loss_pct": 0.03,
    "take_profit_pct": 0.06,
}


async def _get_paper_portfolio(user_id: str) -> Dict:
    doc = await db.paper_portfolios.find_one({"user_id": user_id}, {"_id": 0})
    if doc:
        return doc
    doc = {
        "user_id": user_id, "cash": 100_000.0, "initial_capital": 100_000.0,
        "positions": {}, "history": [],
        "created_at": datetime.now(timezone.utc),
    }
    await db.paper_portfolios.insert_one(doc)
    doc.pop("_id", None)
    return doc


@router.get("/portfolio/paper")
async def get_paper_portfolio(user: User = Depends(current_user)):
    p = await _get_paper_portfolio(user.user_id)
    # Compute summary
    prices = {}
    for tkr in p["positions"]:
        q = md.fetch_stock_data(tkr, period="5d", interval="1d")
        if not q.empty:
            prices[tkr] = float(q["Close"].iloc[-1])
    invested = sum(v["avg_price"] * v["quantity"] for v in p["positions"].values())
    market_value = sum(
        prices.get(t, v["avg_price"]) * v["quantity"]
        for t, v in p["positions"].items()
    )
    value = p["cash"] + market_value
    total_return = (value - p["initial_capital"]) / p["initial_capital"] if p["initial_capital"] else 0.0
    return {
        "cash": p["cash"], "initial_capital": p["initial_capital"],
        "positions": [
            {
                "ticker": t, "quantity": v["quantity"], "avg_price": v["avg_price"],
                "last_price": prices.get(t, v["avg_price"]),
                "market_value": prices.get(t, v["avg_price"]) * v["quantity"],
                "unrealized_pnl": (prices.get(t, v["avg_price"]) - v["avg_price"]) * v["quantity"],
            }
            for t, v in p["positions"].items()
        ],
        "history": p["history"][-50:][::-1],
        "summary": {
            "value": value, "invested": invested, "market_value": market_value,
            "unrealized_pnl": market_value - invested, "total_return": total_return,
            "n_positions": len(p["positions"]), "n_trades": len(p["history"]),
        },
    }


class OrderBody(BaseModel):
    ticker: str
    side: str  # BUY / SELL
    quantity: int
    price: Optional[float] = None
    note: str = ""


@router.post("/portfolio/paper/orders")
async def place_paper_order(body: OrderBody, user: User = Depends(current_user)):
    side = body.side.upper()
    if side not in {"BUY", "SELL"}:
        raise HTTPException(400, "Side must be BUY or SELL")
    price = body.price
    if price is None:
        q = md.fetch_stock_data(body.ticker, period="5d", interval="1d")
        if q.empty:
            raise HTTPException(404, "Could not fetch price")
        price = float(q["Close"].iloc[-1])
    total = price * body.quantity

    p = await _get_paper_portfolio(user.user_id)
    if side == "BUY":
        if total > p["cash"]:
            raise HTTPException(400, "Insufficient cash")
        p["cash"] -= total
        pos = p["positions"].get(body.ticker)
        if pos:
            new_q = pos["quantity"] + body.quantity
            pos["avg_price"] = (pos["avg_price"] * pos["quantity"] + price * body.quantity) / new_q
            pos["quantity"] = new_q
        else:
            p["positions"][body.ticker] = {"quantity": body.quantity, "avg_price": price}
    else:
        pos = p["positions"].get(body.ticker)
        if not pos or pos["quantity"] < body.quantity:
            raise HTTPException(400, "Not enough shares to sell")
        pos["quantity"] -= body.quantity
        p["cash"] += total
        if pos["quantity"] == 0:
            del p["positions"][body.ticker]

    trade = {
        "id": f"PAPER-{uuid.uuid4().hex[:8]}",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ticker": body.ticker, "side": side, "quantity": body.quantity,
        "price": price, "total": total, "mode": "PAPER", "note": body.note,
    }
    p["history"].append(trade)
    await db.paper_portfolios.update_one(
        {"user_id": user.user_id},
        {"$set": {"cash": p["cash"], "positions": p["positions"], "history": p["history"]}},
    )
    return {"ok": True, "trade": trade}


@router.post("/portfolio/paper/reset")
async def reset_paper(user: User = Depends(current_user)):
    await db.paper_portfolios.delete_one({"user_id": user.user_id})
    return await get_paper_portfolio(user)


# ---------------------------------------------------------------------------
# RISK
# ---------------------------------------------------------------------------
@router.get("/risk/rules")
async def get_risk_rules(user: User = Depends(current_user)):
    doc = await db.risk_rules.find_one({"user_id": user.user_id}, {"_id": 0})
    return doc["rules"] if doc else DEFAULT_RISK_RULES


@router.put("/risk/rules")
async def put_risk_rules(rules: Dict[str, float], user: User = Depends(current_user)):
    await db.risk_rules.update_one(
        {"user_id": user.user_id},
        {"$set": {"rules": rules, "updated_at": datetime.now(timezone.utc)}},
        upsert=True,
    )
    return rules


# ---------------------------------------------------------------------------
# AI INSIGHTS
# ---------------------------------------------------------------------------
class ChatBody(BaseModel):
    question: str
    context: Optional[str] = None


@router.post("/ai/chat")
async def ai_chat(body: ChatBody, user: User = Depends(current_user)):
    reply = await asyncio.to_thread(
        alpha.ai_assistant_response, body.question, body.context,
    )
    return {"reply": reply}


class InsightBody(BaseModel):
    ticker: str
    action: str
    technical: Dict[str, Any]
    ml: Optional[Dict[str, Any]] = None


@router.post("/ai/insights")
async def ai_insights(body: InsightBody, user: User = Depends(current_user)):
    pred = None
    if body.ml:
        from algo_trading_system.alpha_generator import AlphaPrediction
        pred = AlphaPrediction(
            action=body.ml.get("action", "HOLD"),
            buy_probability=body.ml.get("buy_probability", 0.5),
            sell_probability=body.ml.get("sell_probability", 0.5),
            confidence=body.ml.get("confidence", 0.5),
            model_accuracy=body.ml.get("accuracy", 0.5),
        )
    text = await asyncio.to_thread(
        alpha.generate_trade_explanation, body.ticker, body.action, body.technical, pred,
    )
    return {"text": text}


# ---------------------------------------------------------------------------
# HEALTH
# ---------------------------------------------------------------------------
@router.get("/")
async def root():
    return {"app": "QuantPulse", "version": "1.0.0", "status": "ok"}


app.include_router(router)
