"""
alpha_generator.py — AI-powered alpha generation.

Two layers:

1. **Machine Learning** (RandomForest + LogisticRegression)
   Predicts the probability that the *next bar's* return is positive.
   Features: RSI, MACD, MACD_Hist, SMA_20, SMA_50, Returns, Volatility, Volume.

2. **OpenAI GPT-4o (via emergentintegrations)**
   Used purely for explanation / market-intelligence layer — never to predict
   prices directly. Generates trade rationales, market summaries, and powers
   the AI assistant chat.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Optional, List, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import EMERGENT_LLM_KEY, OPENAI_MODEL, is_llm_configured
from .strategy import add_all_indicators
from .utils import setup_logger

log = setup_logger("alpha")

FEATURE_COLS = [
    "RSI", "MACD", "MACD_Hist", "SMA_20", "SMA_50",
    "Returns", "Volatility", "Volume",
]


# ---------------------------------------------------------------------------
# ML — features, train, predict
# ---------------------------------------------------------------------------
@dataclass
class AlphaPrediction:
    action: str            # BUY / SELL / HOLD
    buy_probability: float
    sell_probability: float
    confidence: float
    model_accuracy: float


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the feature matrix and binary target (next-bar return > 0)."""
    enriched = add_all_indicators(df)
    if "Returns" not in enriched.columns:
        enriched["Returns"] = enriched["Close"].pct_change()
    if "Volatility" not in enriched.columns:
        enriched["Volatility"] = enriched["Returns"].rolling(20).std() * np.sqrt(252)
    if "Volume" not in enriched.columns:
        enriched["Volume"] = 0.0
    enriched["Target"] = (enriched["Close"].shift(-1) > enriched["Close"]).astype(int)
    enriched = enriched.dropna(subset=FEATURE_COLS + ["Target"])
    return enriched


def train_model(df: pd.DataFrame, model_type: str = "random_forest"):
    """Train and return (model, scaler, accuracy)."""
    data = create_features(df)
    if len(data) < 60:
        log.warning("Not enough data to train (%d rows).", len(data))
        return None, None, 0.0

    X = data[FEATURE_COLS].values
    y = data["Target"].values

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, shuffle=False
    )
    if model_type == "logistic":
        model = LogisticRegression(max_iter=500)
    else:
        model = RandomForestClassifier(
            n_estimators=200, max_depth=6, random_state=42, n_jobs=-1
        )
    model.fit(X_train, y_train)
    acc = accuracy_score(y_test, model.predict(X_test)) if len(X_test) else 0.0
    return model, scaler, float(acc)


def predict_signal(df: pd.DataFrame, model_type: str = "random_forest") -> AlphaPrediction:
    """End-to-end: train on history and emit a prediction for the *latest* bar."""
    model, scaler, acc = train_model(df, model_type=model_type)
    if model is None:
        return AlphaPrediction("HOLD", 0.5, 0.5, 0.0, 0.0)

    feats = create_features(df)
    latest = feats[FEATURE_COLS].iloc[[-1]].values
    latest_scaled = scaler.transform(latest)
    probs = model.predict_proba(latest_scaled)[0]
    buy_p = float(probs[1])
    sell_p = float(probs[0])
    if buy_p >= 0.6:
        action = "BUY"
    elif sell_p >= 0.6:
        action = "SELL"
    else:
        action = "HOLD"
    confidence = float(max(buy_p, sell_p))
    return AlphaPrediction(action, buy_p, sell_p, confidence, acc)


def evaluate_model(df: pd.DataFrame, model_type: str = "random_forest") -> Dict[str, float]:
    _, _, acc = train_model(df, model_type)
    return {"accuracy": acc, "samples": len(create_features(df))}


# ---------------------------------------------------------------------------
# OpenAI GPT-4o — explanation / intelligence layer (via emergentintegrations)
# ---------------------------------------------------------------------------
def _llm_chat(system_message: str, user_message: str, session_id: str) -> str:
    """Synchronous wrapper around emergentintegrations LlmChat."""
    if not is_llm_configured():
        return ("AI explanations are disabled — EMERGENT_LLM_KEY not configured. "
                "Add the key to enable GPT-4o powered insights.")
    try:
        from emergentintegrations.llm.chat import LlmChat, UserMessage

        async def _run() -> str:
            chat = LlmChat(
                api_key=EMERGENT_LLM_KEY,
                session_id=session_id,
                system_message=system_message,
            ).with_model("openai", OPENAI_MODEL)
            return await chat.send_message(UserMessage(text=user_message))

        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                return asyncio.run_coroutine_threadsafe(_run(), loop).result(timeout=60)
        except RuntimeError:
            pass
        return asyncio.run(_run())
    except Exception as exc:  # noqa: BLE001
        log.exception("LLM call failed: %s", exc)
        return f"(AI explanation unavailable: {exc})"


def generate_trade_explanation(
    ticker: str,
    action: str,
    technical_snapshot: dict,
    ml_prediction: Optional[AlphaPrediction] = None,
) -> str:
    """Generate a natural-language rationale for a trade signal."""
    system = (
        "You are an experienced quantitative trading analyst. "
        "Given a stock ticker, a recommended action, technical indicators, "
        "and an ML prediction, write a concise, professional 3-4 sentence "
        "rationale. Do NOT predict prices. Focus on WHY the indicators "
        "support the action. Speak plainly; avoid jargon."
    )
    ml_part = (
        f"ML model: buy_prob={ml_prediction.buy_probability:.2f}, "
        f"sell_prob={ml_prediction.sell_probability:.2f}, "
        f"accuracy={ml_prediction.model_accuracy:.2f}."
    ) if ml_prediction else ""
    user = (
        f"Ticker: {ticker}\n"
        f"Action: {action}\n"
        f"Indicators: {technical_snapshot}\n"
        f"{ml_part}"
    )
    return _llm_chat(system, user, session_id=f"explain-{ticker}")


def generate_market_summary(tickers: List[str], snapshots: Dict[str, dict]) -> str:
    """Cross-ticker market summary written for a retail trader."""
    system = (
        "You are a senior market strategist writing a brief daily market summary "
        "for a retail trader. 4-6 sentences. Highlight overall sentiment, "
        "notable strengths/weaknesses across the listed tickers, and one "
        "actionable observation. Stay factual."
    )
    user = f"Tickers: {tickers}\nSnapshots: {snapshots}"
    return _llm_chat(system, user, session_id="market-summary")


def ai_assistant_response(question: str, context: Optional[str] = None) -> str:
    system = (
        "You are an AI trading assistant. Help the user understand trading "
        "concepts, technical indicators, risk management, and interpret the "
        "data they share. Never give financial advice or predict prices."
    )
    user = question if not context else f"Context:\n{context}\n\nQuestion: {question}"
    return _llm_chat(system, user, session_id="assistant")


def generate_backtest_commentary(ticker: str, comparison_table: str) -> str:
    """GPT-4o commentary on a backtest comparison table.
    Picks the most appropriate strategy for this ticker and explains why."""
    system = (
        "You are a senior quant analyst. Given a strategy comparison table from "
        "a single-asset backtest, write a concise 4-6 sentence commentary that: "
        "(1) identifies the best strategy by risk-adjusted return, (2) notes any "
        "strategy with poor drawdown or low win-rate, (3) gives ONE actionable "
        "observation. Do not predict future returns."
    )
    user = f"Ticker: {ticker}\n\nResults:\n{comparison_table}"
    return _llm_chat(system, user, session_id=f"backtest-{ticker}")
