"""
dashboard.py — Professional Streamlit dashboard.

Run from the project root:
    streamlit run algo_trading_system/dashboard.py
or from inside the folder:
    streamlit run dashboard.py
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path

# Make module imports work whether you run from /app or from /app/algo_trading_system
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from algo_trading_system import (
    market_data as md,
    strategy as strat,
    alpha_generator as alpha,
    risk_management as risk,
    portfolio as pf,
    execution as ex,
    backtest as bt,
    config as cfg,
)
from algo_trading_system.utils import format_currency, format_percentage


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="QuantPulse · AI Algo Trading",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS — dark fintech look
st.markdown(
    """
    <style>
    .main .block-container {padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1500px;}
    .kpi-card {
        background: linear-gradient(140deg,#111a33 0%,#0e1530 100%);
        border:1px solid #1f2a4d; border-radius:14px; padding:18px 20px;
        box-shadow: 0 6px 20px rgba(0,0,0,.25);
    }
    .kpi-label{color:#7f8db5;font-size:12px;letter-spacing:.08em;text-transform:uppercase;}
    .kpi-value{color:#e6edf7;font-size:26px;font-weight:600;margin-top:4px;}
    .kpi-delta-up{color:#22c55e;font-size:13px;}
    .kpi-delta-down{color:#ef4444;font-size:13px;}
    .badge {display:inline-block;padding:4px 10px;border-radius:999px;font-size:12px;font-weight:600;letter-spacing:.04em;}
    .badge-buy   {background:rgba(34,197,94,.12);color:#22c55e;border:1px solid rgba(34,197,94,.35);}
    .badge-sell  {background:rgba(239,68,68,.12);color:#ef4444;border:1px solid rgba(239,68,68,.35);}
    .badge-hold  {background:rgba(148,163,184,.12);color:#94a3b8;border:1px solid rgba(148,163,184,.35);}
    .panel-title{font-size:14px;color:#94a3b8;letter-spacing:.10em;text-transform:uppercase;margin:0 0 8px 0;}
    .stTabs [data-baseweb="tab-list"] {gap:6px;}
    .stTabs [data-baseweb="tab"] {
        background:#111a33;border:1px solid #1f2a4d;border-radius:10px;
        padding:8px 14px;color:#94a3b8;
    }
    .stTabs [aria-selected="true"]{background:#152150;color:#22d3ee;border-color:#22d3ee;}
    .small-muted{color:#7f8db5;font-size:12px;}
    hr{border:none;border-top:1px solid #1f2a4d;margin:14px 0;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "portfolio" not in st.session_state:
    st.session_state.portfolio = pf.load_portfolio()
if "ai_chat" not in st.session_state:
    st.session_state.ai_chat = []


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
    "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
]
TICKER_PLACEHOLDER = "— Select a ticker —"

with st.sidebar:
    st.markdown("### ⚡ QuantPulse")
    st.caption("AI-powered algo trading analytics")

    # ---------- Ticker picker ----------
    selected = st.selectbox(
        "Ticker", [TICKER_PLACEHOLDER] + DEFAULT_TICKERS,
        index=0, key="ticker_select",
    )
    custom = st.text_input("…or custom ticker", "", key="custom_ticker")
    # Quick-pick chips set session_state["quick_ticker"] (handled below in the landing screen)
    ticker = ""
    if custom.strip():
        ticker = custom.strip().upper()
    elif st.session_state.get("quick_ticker"):
        ticker = st.session_state["quick_ticker"]
    elif selected and selected != TICKER_PLACEHOLDER:
        ticker = selected
        st.session_state.pop("quick_ticker", None)

    period = st.select_slider(
        "History",
        options=["1mo", "3mo", "6mo", "1y", "2y", "5y"],
        value="1y", key="period_select",
    )
    interval = st.selectbox("Interval", ["1d", "1h", "1wk"], index=0, key="interval_select")

    st.markdown("---")
    st.markdown("#### ⚙️ Strategy")
    sma_short = st.slider("SMA short window", 5, 50, 20)
    sma_long = st.slider("SMA long window", 20, 200, 50)
    model_type = st.radio(
        "ML model", ["random_forest", "logistic"], horizontal=True, key="model_type"
    )

    # ---------- Broker connection ----------
    st.markdown("---")
    st.markdown("#### 🔌 Broker")
    broker = st.radio(
        "Broker", ["Zerodha Kite", "Groww"], horizontal=True, key="broker-radio"
    )

    if broker == "Zerodha Kite":
        # Pull saved creds from session, falling back to .env if present
        env_key = cfg.KITE_API_KEY if not cfg.KITE_API_KEY.startswith("your_") else ""
        env_secret = cfg.KITE_API_SECRET if not cfg.KITE_API_SECRET.startswith("your_") else ""
        kite_api_key = st.text_input(
            "API Key", value=st.session_state.get("kite_api_key", env_key),
            type="password", key="kite-api-key-input",
        )
        kite_api_secret = st.text_input(
            "API Secret", value=st.session_state.get("kite_api_secret", env_secret),
            type="password", key="kite-api-secret-input",
        )
        st.session_state["kite_api_key"] = kite_api_key
        st.session_state["kite_api_secret"] = kite_api_secret

        if kite_api_key:
            login_url = ex.kite_login_url(kite_api_key)
            if login_url:
                st.markdown(
                    f"<a href='{login_url}' target='_blank' "
                    "style='display:block;padding:8px 12px;background:#152150;"
                    "border-radius:8px;color:#22d3ee;text-decoration:none;"
                    "text-align:center;margin-bottom:8px;'>🔗 Open Kite Login</a>",
                    unsafe_allow_html=True,
                )
                st.caption("After login, copy `request_token=…` from the redirect URL.")

        req_token = st.text_input("Paste request_token", "", key="kite-req-token")
        if st.button("✅ Generate session", use_container_width=True, key="kite-gen-session"):
            res = ex.kite_generate_session(req_token, kite_api_key, kite_api_secret)
            if res.get("error"):
                st.error(res["error"])
            else:
                st.session_state["kite_access_token"] = res.get("access_token", "")
                st.session_state["kite_user_name"] = res.get("user_name") or res.get("user_id", "user")
                st.success(f"Connected as {st.session_state['kite_user_name']}")

        connected = bool(st.session_state.get("kite_access_token"))
        if connected:
            st.success(f"✅ Connected as {st.session_state.get('kite_user_name')}")
            if st.button("🔌 Disconnect", use_container_width=True, key="kite-disconnect"):
                for k in ["kite_access_token", "kite_user_name"]:
                    st.session_state.pop(k, None)
                st.rerun()
        else:
            st.caption("Not connected — paper trading only.")

        live_mode = st.toggle(
            "🔴 Enable LIVE trading",
            value=False, key="live-mode-toggle",
            disabled=not connected,
            help="Submits real orders to Kite. Requires an active session.",
        )
        if live_mode:
            st.warning("⚠️ LIVE mode — real orders will be placed.")
    else:
        # Groww — placeholder until API access is granted
        gw = ex.groww_status()
        st.info("🚧 " + gw["message"])
        st.text_input("Groww API Key (when available)", "", type="password",
                      key="groww-api-key", disabled=True)
        live_mode = False
        connected = False
        kite_api_key = ""
        kite_api_secret = ""

    st.markdown("---")
    st.markdown("#### 💼 Integrations")
    st.write("AI (GPT-4o): " + ("✅" if cfg.is_llm_configured() else "❌"))
    st.write("Kite session: " + ("✅ live" if st.session_state.get("kite_access_token") else "❌ paper"))
    st.write("Alpha Vantage: " + ("✅" if cfg.is_alphavantage_configured() else "❌ placeholder"))

    st.markdown("---")
    if st.button("🔄 Refresh data", use_container_width=True, key="refresh-btn"):
        st.cache_data.clear()
        st.rerun()
    if st.button("♻️ Reset paper portfolio", use_container_width=True, key="reset-btn"):
        st.session_state.portfolio = pf.reset_portfolio()
        st.success("Paper portfolio reset.")


# ---------------------------------------------------------------------------
# Data load (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False, ttl=300)
def load_data(t: str, p: str, i: str) -> pd.DataFrame:
    return md.fetch_stock_data(t, period=p, interval=i)


# ---------------------------------------------------------------------------
# Landing screen — no ticker selected yet
# ---------------------------------------------------------------------------
if not ticker:
    st.markdown(
        "<h2 style='margin-top:0'>👋 Welcome to QuantPulse</h2>"
        "<p class='small-muted'>AI-powered algo trading analytics. "
        "Pick a ticker below or use the sidebar to begin.</p>",
        unsafe_allow_html=True,
    )

    st.markdown("<div class='panel-title'>Quick-pick</div>",
                unsafe_allow_html=True)
    cols = st.columns(6)
    for i, t in enumerate(DEFAULT_TICKERS):
        if cols[i % 6].button(t, key=f"qp-{t}", use_container_width=True):
            st.session_state["quick_ticker"] = t
            st.rerun()

    st.markdown("<br/>", unsafe_allow_html=True)
    st.markdown("<div class='panel-title'>What you can do</div>",
                unsafe_allow_html=True)
    f1, f2, f3, f4 = st.columns(4)
    for col, title, body in [
        (f1, "📊 Charts", "Candlestick, SMA, Bollinger, RSI, MACD on one screen."),
        (f2, "🤖 AI Insights", "ML alpha (RF / LR) + GPT-4o rationale for every signal."),
        (f3, "📉 Backtest", "Compare 6 strategies on any ticker with one click."),
        (f4, "🔌 Broker", "Connect Zerodha Kite to place real orders."),
    ]:
        col.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>{title}</div>"
            f"<p style='color:#cbd5e1;font-size:13px;margin-top:6px'>{body}</p></div>",
            unsafe_allow_html=True,
        )
    st.stop()

# Note: quick_ticker is the source-of-truth between reruns once set; it's
# cleared automatically when the user explicitly picks from the dropdown.


with st.spinner(f"Loading {ticker} …"):
    df = load_data(ticker, period, interval)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown(
    f"## 📈 {ticker}  "
    f"<span class='small-muted'>· {period} · {interval}</span>",
    unsafe_allow_html=True,
)

if df.empty:
    st.error(f"No market data for {ticker}. Try a different symbol.")
    st.stop()

latest_close = float(df["Close"].iloc[-1])
prev_close = float(df["Close"].iloc[-2]) if len(df) > 1 else latest_close
day_change = (latest_close - prev_close) / prev_close if prev_close else 0.0
year_high = float(df["High"].max())
year_low = float(df["Low"].min())
avg_vol = float(df["Volume"].mean()) if "Volume" in df.columns else 0.0

# Signals & alpha
snapshot = strat.latest_signal(df)
ml_pred = alpha.predict_signal(df, model_type=model_type)


def signal_badge(action: str) -> str:
    cls = {"BUY": "badge-buy", "SELL": "badge-sell", "HOLD": "badge-hold"}[action]
    return f"<span class='badge {cls}'>{action}</span>"


# ---------------------------------------------------------------------------
# KPI cards
# ---------------------------------------------------------------------------
k1, k2, k3, k4, k5 = st.columns(5)
for col, label, value, sub in [
    (k1, "Last Price", format_currency(latest_close, "$"),
     f"<span class='kpi-delta-{'up' if day_change>=0 else 'down'}'>"
     f"{format_percentage(day_change)} today</span>"),
    (k2, "Period High", format_currency(year_high, "$"), ""),
    (k3, "Period Low", format_currency(year_low, "$"), ""),
    (k4, "Avg Volume", f"{avg_vol:,.0f}", ""),
    (k5, "Tech Signal", signal_badge(snapshot["action"]),
     f"<span class='small-muted'>Vote {snapshot['vote']}</span>"),
]:
    col.markdown(
        f"<div class='kpi-card'><div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>{sub}</div>",
        unsafe_allow_html=True,
    )

st.markdown("<hr/>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_chart, tab_signals, tab_ai, tab_risk, tab_paper, tab_portfolio, tab_backtest, tab_assistant = st.tabs(
    ["📊 Charts", "🎯 Signals", "🤖 AI Insights", "🛡️ Risk",
     "📝 Paper Trade", "💼 Portfolio", "📉 Backtest", "💬 AI Assistant"]
)

# =============================================================================
# CHARTS TAB
# =============================================================================
with tab_chart:
    enriched = strat.add_all_indicators(df).dropna()
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.55, 0.22, 0.23], vertical_spacing=0.04,
        subplot_titles=("Price / SMA / Bollinger", "RSI(14)", "MACD"),
    )
    fig.add_trace(
        go.Candlestick(
            x=enriched.index, open=enriched["Open"], high=enriched["High"],
            low=enriched["Low"], close=enriched["Close"], name="Price",
            increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
        ),
        row=1, col=1,
    )
    fig.add_trace(go.Scatter(x=enriched.index, y=enriched["SMA_20"],
                             name="SMA 20", line=dict(color="#22d3ee", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=enriched.index, y=enriched["SMA_50"],
                             name="SMA 50", line=dict(color="#a78bfa", width=1.5)), row=1, col=1)
    fig.add_trace(go.Scatter(x=enriched.index, y=enriched["BB_High"],
                             name="BB Upper", line=dict(color="#475569", width=1, dash="dot")),
                  row=1, col=1)
    fig.add_trace(go.Scatter(x=enriched.index, y=enriched["BB_Low"],
                             name="BB Lower", line=dict(color="#475569", width=1, dash="dot"),
                             fill="tonexty", fillcolor="rgba(71,85,105,0.08)"), row=1, col=1)

    fig.add_trace(go.Scatter(x=enriched.index, y=enriched["RSI"], name="RSI",
                             line=dict(color="#facc15", width=1.5)), row=2, col=1)
    fig.add_hline(y=70, line=dict(color="#ef4444", dash="dash"), row=2, col=1)
    fig.add_hline(y=30, line=dict(color="#22c55e", dash="dash"), row=2, col=1)

    fig.add_trace(go.Bar(x=enriched.index, y=enriched["MACD_Hist"],
                         name="MACD Hist", marker_color="#475569"), row=3, col=1)
    fig.add_trace(go.Scatter(x=enriched.index, y=enriched["MACD"],
                             name="MACD", line=dict(color="#22d3ee", width=1.5)), row=3, col=1)
    fig.add_trace(go.Scatter(x=enriched.index, y=enriched["MACD_Signal"],
                             name="MACD Signal", line=dict(color="#f472b6", width=1.5)),
                  row=3, col=1)

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0b1020", plot_bgcolor="#0b1020",
        height=720, margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", y=1.06),
        xaxis_rangeslider_visible=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# =============================================================================
# SIGNALS TAB
# =============================================================================
with tab_signals:
    cs1, cs2 = st.columns([1.2, 1])
    with cs1:
        st.markdown("<div class='panel-title'>Indicator votes</div>", unsafe_allow_html=True)
        votes = snapshot["votes"]
        st.markdown(
            f"""
            <div class='kpi-card'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span class='kpi-label'>SMA(20/50) Crossover</span>{signal_badge('BUY' if votes['sma']>0 else 'SELL' if votes['sma']<0 else 'HOLD')}
              </div><hr/>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span class='kpi-label'>RSI(14)</span>{signal_badge('BUY' if votes['rsi']>0 else 'SELL' if votes['rsi']<0 else 'HOLD')}
              </div><hr/>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span class='kpi-label'>MACD</span>{signal_badge('BUY' if votes['macd']>0 else 'SELL' if votes['macd']<0 else 'HOLD')}
              </div><hr/>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span class='kpi-label'>Final · Majority Vote</span>{signal_badge(snapshot['action'])}
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cs2:
        st.markdown("<div class='panel-title'>Latest indicators</div>", unsafe_allow_html=True)
        st.metric("RSI(14)", f"{snapshot['rsi']:.2f}" if snapshot["rsi"] is not None else "—")
        st.metric("MACD",
                  f"{snapshot['macd']:.4f}" if snapshot["macd"] is not None else "—",
                  f"{snapshot['macd'] - snapshot['macd_signal']:.4f}"
                  if snapshot['macd'] is not None and snapshot['macd_signal'] is not None else "")
        st.metric("SMA 20", f"{snapshot['sma_20']:.2f}" if snapshot["sma_20"] is not None else "—")
        st.metric("SMA 50", f"{snapshot['sma_50']:.2f}" if snapshot["sma_50"] is not None else "—")


# =============================================================================
# AI INSIGHTS TAB
# =============================================================================
with tab_ai:
    ca, cb = st.columns([1, 1])
    with ca:
        st.markdown("<div class='panel-title'>ML Alpha Prediction</div>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div class='kpi-card'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <span class='kpi-label'>Predicted Action</span>{signal_badge(ml_pred.action)}
              </div><hr/>
              <div style='display:flex;justify-content:space-between'>
                <span class='kpi-label'>Buy Probability</span>
                <span class='kpi-value' style='font-size:18px'>{ml_pred.buy_probability:.2%}</span>
              </div><hr/>
              <div style='display:flex;justify-content:space-between'>
                <span class='kpi-label'>Sell Probability</span>
                <span class='kpi-value' style='font-size:18px'>{ml_pred.sell_probability:.2%}</span>
              </div><hr/>
              <div style='display:flex;justify-content:space-between'>
                <span class='kpi-label'>Confidence</span>
                <span class='kpi-value' style='font-size:18px'>{ml_pred.confidence:.2%}</span>
              </div><hr/>
              <div style='display:flex;justify-content:space-between'>
                <span class='kpi-label'>Backtest Accuracy</span>
                <span class='kpi-value' style='font-size:18px'>{ml_pred.model_accuracy:.2%}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with cb:
        st.markdown("<div class='panel-title'>GPT-4o Trade Rationale</div>", unsafe_allow_html=True)
        if st.button("🪄 Generate explanation", key="gen-exp-btn", use_container_width=True):
            with st.spinner("GPT-4o thinking…"):
                txt = alpha.generate_trade_explanation(
                    ticker, snapshot["action"], snapshot, ml_pred,
                )
                st.session_state[f"exp_{ticker}"] = txt
        if exp := st.session_state.get(f"exp_{ticker}"):
            st.info(exp)
        else:
            st.caption("Click the button to generate a plain-English rationale.")

    st.markdown("<div class='panel-title' style='margin-top:24px'>Cross-ticker market summary</div>",
                unsafe_allow_html=True)
    summary_tickers = st.multiselect(
        "Pick tickers for the summary", DEFAULT_TICKERS,
        default=["AAPL", "MSFT", "NVDA"], key="summary-tickers",
    )
    if st.button("📰 Generate market summary", key="gen-sum-btn"):
        snaps = {}
        for t in summary_tickers:
            d = load_data(t, period, interval)
            if not d.empty:
                snaps[t] = strat.latest_signal(d)
        with st.spinner("GPT-4o composing market summary…"):
            st.session_state["market_summary"] = alpha.generate_market_summary(
                summary_tickers, snaps
            )
    if msum := st.session_state.get("market_summary"):
        st.success(msum)


# =============================================================================
# RISK TAB
# =============================================================================
with tab_risk:
    rk1, rk2 = st.columns([1, 1])
    with rk1:
        st.markdown("<div class='panel-title'>Trade sizer</div>", unsafe_allow_html=True)
        cap = st.number_input("Capital ($)", value=float(st.session_state.portfolio.cash),
                              min_value=0.0, step=1000.0, key="risk-cap")
        entry = st.number_input("Entry price", value=float(latest_close), key="risk-entry")
        side = st.selectbox("Side", ["BUY", "SELL"], key="risk-side")
        vol_value = float(df["Volatility"].iloc[-1]) if "Volatility" in df.columns and not pd.isna(df["Volatility"].iloc[-1]) else None
        validation = risk.validate_trade(cap, entry, side, volatility=vol_value)
    with rk2:
        st.markdown("<div class='panel-title'>Result</div>", unsafe_allow_html=True)
        badge = "✅" if validation.is_valid else "⛔"
        st.markdown(f"### {badge} {validation.reason}")
        st.metric("Suggested Qty", validation.suggested_quantity)
        st.metric("Stop-Loss", format_currency(validation.stop_loss, "$"))
        st.metric("Take-Profit", format_currency(validation.take_profit, "$"))
        st.metric("Risk Amount", format_currency(validation.risk_amount, "$"))
        st.metric("Risk / Reward", f"1 : {validation.risk_reward_ratio:.2f}")


# =============================================================================
# PAPER TRADE TAB
# =============================================================================
with tab_paper:
    p = st.session_state.portfolio
    is_live_active = bool(
        st.session_state.get("kite_access_token") and st.session_state.get("live-mode-toggle")
    )
    mode_badge = ("<span class='badge badge-sell'>🔴 LIVE</span>"
                  if is_live_active else "<span class='badge badge-hold'>📝 PAPER</span>")
    st.markdown(
        f"<div class='panel-title'>Order ticket · {mode_badge}</div>",
        unsafe_allow_html=True,
    )

    pt1, pt2 = st.columns([1, 1])
    with pt1:
        order_side = st.radio("Side", ["BUY", "SELL"], horizontal=True, key="order-side")
        suggested_qty = max(validation.suggested_quantity if validation.is_valid else 1, 1)
        order_qty = st.number_input("Quantity", min_value=1, value=int(suggested_qty), step=1,
                                    key="order-qty")
        order_price = st.number_input("Price", value=float(latest_close), step=0.5,
                                      key="order-price")
        note = st.text_input("Note (optional)", "", key="order-note")

        btn_label = ("🔴 Submit LIVE order" if is_live_active
                     else "📩 Submit paper order")
        if st.button(btn_label, type="primary", use_container_width=True,
                     key="submit-order-btn"):
            try:
                fn = ex.place_buy_order if order_side == "BUY" else ex.place_sell_order
                result = fn(
                    p, ticker, int(order_qty), float(order_price),
                    live=is_live_active, note=note,
                    broker="kite",
                    kite_api_key=st.session_state.get("kite_api_key", ""),
                    kite_access_token=st.session_state.get("kite_access_token", ""),
                )
                if result.success:
                    st.success(result.message)
                    st.session_state.portfolio = p
                else:
                    st.error(result.message)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))

    with pt2:
        st.markdown("<div class='panel-title'>Recent trades</div>", unsafe_allow_html=True)
        if p.history:
            hist_df = pd.DataFrame([t.__dict__ for t in p.history[-20:][::-1]])
            st.dataframe(hist_df, use_container_width=True, hide_index=True, height=380)
        else:
            st.caption("No trades yet — submit your first order on the left.")

        # Live broker positions/orders when connected
        if st.session_state.get("kite_access_token"):
            with st.expander("🔌 Broker positions & orders (live from Kite)", expanded=False):
                api_key = st.session_state.get("kite_api_key", "")
                tok = st.session_state.get("kite_access_token", "")
                positions = ex.kite_positions(api_key, tok).get("net", [])
                orders = ex.kite_orders(api_key, tok)
                holdings = ex.kite_holdings(api_key, tok)
                st.markdown("**Holdings**")
                st.dataframe(pd.DataFrame(holdings), use_container_width=True,
                             hide_index=True) if holdings else st.caption("No holdings.")
                st.markdown("**Net positions**")
                st.dataframe(pd.DataFrame(positions), use_container_width=True,
                             hide_index=True) if positions else st.caption("No positions.")
                st.markdown("**Today's orders**")
                st.dataframe(pd.DataFrame(orders), use_container_width=True,
                             hide_index=True) if orders else st.caption("No orders today.")


# =============================================================================
# PORTFOLIO TAB  (institutional-grade analytics suite — 6 sub-tabs)
# =============================================================================
with tab_portfolio:
    p = st.session_state.portfolio

    # Update equity history using current price for the loaded ticker
    prices = {ticker: latest_close}
    for tkr in p.positions:
        if tkr not in prices:
            d = load_data(tkr, "1mo", "1d")
            if not d.empty:
                prices[tkr] = float(d["Close"].iloc[-1])
    pf.update_equity_history(p, prices)
    summary = pf.portfolio_summary(p, prices)

    sub_holdings, sub_opt, sub_front, sub_mc, sub_corr, sub_reb = st.tabs([
        "📒 Holdings", "🎯 Optimizer", "📈 Efficient Frontier",
        "🎲 Monte Carlo", "🔗 Correlation", "♻️ Rebalance",
    ])

    # ---------------------------------------------------------------- HOLDINGS
    with sub_holdings:
        pc1, pc2, pc3, pc4 = st.columns(4)
        pc1.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>Portfolio Value</div>"
            f"<div class='kpi-value'>{format_currency(summary['value'],'$')}</div>"
            f"<span class='kpi-delta-{'up' if summary['total_return']>=0 else 'down'}'>"
            f"{format_percentage(summary['total_return'])} total</span></div>",
            unsafe_allow_html=True,
        )
        pc2.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>Cash</div>"
            f"<div class='kpi-value'>{format_currency(summary['cash'],'$')}</div>"
            f"<span class='small-muted'>{summary['n_positions']} positions</span></div>",
            unsafe_allow_html=True,
        )
        pc3.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>Unrealized P&L</div>"
            f"<div class='kpi-value'>{format_currency(summary['unrealized_pnl'],'$')}</div>"
            f"<span class='small-muted'>Exposure {format_percentage(summary['exposure'])}</span></div>",
            unsafe_allow_html=True,
        )
        pc4.markdown(
            f"<div class='kpi-card'><div class='kpi-label'>Sharpe / Max DD</div>"
            f"<div class='kpi-value'>{summary['sharpe']:.2f} · "
            f"{format_percentage(summary['max_drawdown'])}</div>"
            f"<span class='small-muted'>CAGR {format_percentage(summary['cagr'])}</span></div>",
            unsafe_allow_html=True,
        )
        st.markdown("<hr/>", unsafe_allow_html=True)
        eq_col, alloc_col = st.columns([1.5, 1])
        with eq_col:
            st.markdown("<div class='panel-title'>Equity curve</div>", unsafe_allow_html=True)
            if p.equity_history:
                eq_df = pd.DataFrame(p.equity_history)
                eq_df["timestamp"] = pd.to_datetime(eq_df["timestamp"])
                eqfig = go.Figure()
                eqfig.add_trace(go.Scatter(
                    x=eq_df["timestamp"], y=eq_df["value"], mode="lines",
                    line=dict(color="#22d3ee", width=2),
                    fill="tozeroy", fillcolor="rgba(34,211,238,0.08)", name="Equity"
                ))
                eqfig.update_layout(
                    template="plotly_dark", paper_bgcolor="#0b1020", plot_bgcolor="#0b1020",
                    height=320, margin=dict(l=10, r=10, t=10, b=10),
                    yaxis_title="$", showlegend=False,
                )
                st.plotly_chart(eqfig, use_container_width=True)
            else:
                st.caption("Equity curve will appear after your first portfolio snapshot.")
        with alloc_col:
            st.markdown("<div class='panel-title'>Allocation</div>", unsafe_allow_html=True)
            alloc = pf.allocation_breakdown(p, prices)
            if not alloc.empty:
                pie = go.Figure(go.Pie(
                    labels=alloc["Ticker"], values=alloc["Value"], hole=0.55,
                    marker=dict(colors=["#22d3ee", "#a78bfa", "#22c55e", "#facc15",
                                        "#f472b6", "#fb923c", "#38bdf8", "#94a3b8"]),
                ))
                pie.update_layout(
                    template="plotly_dark", paper_bgcolor="#0b1020",
                    plot_bgcolor="#0b1020", height=320, margin=dict(l=10, r=10, t=10, b=10),
                    showlegend=True, legend=dict(orientation="h", y=-0.05),
                )
                st.plotly_chart(pie, use_container_width=True)
            else:
                st.caption("Open a position to populate the allocation chart.")

        st.markdown("<div class='panel-title'>Holdings</div>", unsafe_allow_html=True)
        if p.positions:
            rows = []
            for tkr, pos in p.positions.items():
                price = prices.get(tkr, pos.avg_price)
                rows.append({
                    "Ticker": tkr, "Quantity": pos.quantity,
                    "Avg Price": pos.avg_price, "Last": price,
                    "Market Value": pos.market_value(price),
                    "Unrealized P&L": pos.unrealized_pnl(price),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.caption("No open positions.")

    # =============================================================
    # OPTIMIZER · FRONTIER · MC · CORR · REBALANCE — share inputs
    # =============================================================
    from algo_trading_system import optimizer as optz

    # Shared multi-asset selector
    OPT_UNIVERSE = [
        "AAPL", "MSFT", "NVDA", "GOOGL", "AMZN", "META", "TSLA",
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS",
        "BTC-USD", "ETH-USD",
    ]

    @st.cache_data(show_spinner=False, ttl=600)
    def _load_returns(tickers_tuple: tuple, lookback: str) -> pd.DataFrame:
        frames = {}
        for t in tickers_tuple:
            d = md.fetch_stock_data(t, period=lookback, interval="1d")
            if not d.empty and "Returns" in d.columns:
                frames[t] = d["Returns"]
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, axis=1).dropna(how="all").fillna(0.0)

    with sub_opt:
        st.markdown("<div class='panel-title'>Modern Portfolio Theory · "
                    "Max Sharpe / Min Variance / Risk Parity / Equal Weight</div>",
                    unsafe_allow_html=True)
        oc1, oc2 = st.columns([1, 1.5])
        with oc1:
            assets = st.multiselect("Asset universe", OPT_UNIVERSE,
                                    default=["AAPL", "MSFT", "NVDA", "GOOGL", "TSLA"],
                                    key="opt-assets")
            lookback = st.select_slider(
                "Lookback", options=["3mo", "6mo", "1y", "2y", "5y"],
                value="1y", key="opt-lookback",
            )
            method = st.radio(
                "Method", ["max_sharpe", "min_variance", "risk_parity", "equal_weight"],
                key="opt-method",
            )
            rf = st.slider("Risk-free rate (annual)", 0.0, 0.10, 0.04, 0.005,
                           key="opt-rf")
            run_opt = st.button("🚀 Optimize", type="primary",
                                use_container_width=True, key="opt-run")
        with oc2:
            if run_opt and len(assets) >= 2:
                with st.spinner("Crunching covariance & solving SLSQP …"):
                    rets = _load_returns(tuple(assets), lookback)
                if rets.empty or rets.shape[1] < 2:
                    st.error("Insufficient data for the chosen universe.")
                else:
                    op = optz.PortfolioOptimizer(rets, risk_free_rate=rf)
                    if method == "max_sharpe":
                        res = op.optimize_max_sharpe()
                    elif method == "min_variance":
                        res = op.optimize_min_variance()
                    elif method == "risk_parity":
                        rp = optz.RiskParity(op.cov_matrix)
                        w = rp.optimize()
                        r_, v_, s_ = op.portfolio_performance(w)
                        res = optz.OptimizationResult(
                            "risk_parity", w, op.tickers, r_, v_, s_)
                    else:
                        res = op.equal_weight()
                    st.session_state["opt_result"] = res
                    st.session_state["opt_returns"] = rets

            res = st.session_state.get("opt_result")
            if res:
                r1, r2, r3 = st.columns(3)
                r1.metric("Expected Return", format_percentage(res.expected_return))
                r2.metric("Volatility", format_percentage(res.volatility))
                r3.metric("Sharpe", f"{res.sharpe:.2f}")

                w_df = pd.DataFrame({
                    "Ticker": res.tickers,
                    "Weight": [round(float(w), 4) for w in res.weights],
                })
                pie = go.Figure(go.Pie(
                    labels=w_df["Ticker"], values=w_df["Weight"], hole=0.55,
                    marker=dict(colors=["#22d3ee", "#a78bfa", "#22c55e", "#facc15",
                                        "#f472b6", "#fb923c", "#38bdf8", "#94a3b8"]),
                ))
                pie.update_layout(
                    template="plotly_dark", paper_bgcolor="#0b1020",
                    plot_bgcolor="#0b1020", height=320,
                    margin=dict(l=10, r=10, t=10, b=10),
                    title=f"Optimised weights · {res.method}",
                    legend=dict(orientation="h", y=-0.05),
                )
                st.plotly_chart(pie, use_container_width=True)
                st.dataframe(w_df, use_container_width=True, hide_index=True)

                if st.button("🧠 AI Portfolio Analysis", key="opt-ai-btn"):
                    with st.spinner("GPT-4o analysing the allocation …"):
                        st.session_state["opt_ai"] = optz.AIAllocationEngine.analyze_portfolio(
                            res.as_dict(),
                            {"return": res.expected_return, "vol": res.volatility,
                             "sharpe": res.sharpe},
                            method=res.method,
                        )
                if txt := st.session_state.get("opt_ai"):
                    st.info(txt)
            else:
                st.caption("Pick ≥2 assets on the left and press Optimize.")

    # ------------------------------------------------------------ FRONTIER
    with sub_front:
        st.markdown("<div class='panel-title'>Efficient Frontier · "
                    "5000 random portfolios sampled on the simplex</div>",
                    unsafe_allow_html=True)
        rets = st.session_state.get("opt_returns")
        if rets is None or rets.empty:
            st.caption("Run an optimisation first (the Optimizer tab) — the same "
                       "asset universe is reused here.")
        else:
            op = optz.PortfolioOptimizer(rets,
                                         risk_free_rate=st.session_state.get("opt-rf", 0.04))
            ef = optz.EfficientFrontier(op)
            with st.spinner("Sampling 5000 portfolios …"):
                frontier = ef.generate_frontier(5000)
            best_s = ef.best_sharpe(frontier)
            min_v = ef.min_vol(frontier)

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=frontier["vol"], y=frontier["return"], mode="markers",
                marker=dict(color=frontier["sharpe"], colorscale="Viridis",
                            size=5, showscale=True,
                            colorbar=dict(title="Sharpe")),
                name="Random", text=None,
            ))
            fig.add_trace(go.Scatter(
                x=[best_s["vol"]], y=[best_s["return"]], mode="markers",
                marker=dict(color="#22d3ee", size=18, symbol="star",
                            line=dict(color="white", width=1)),
                name="Max Sharpe",
            ))
            fig.add_trace(go.Scatter(
                x=[min_v["vol"]], y=[min_v["return"]], mode="markers",
                marker=dict(color="#facc15", size=18, symbol="diamond",
                            line=dict(color="white", width=1)),
                name="Min Volatility",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0b1020",
                plot_bgcolor="#0b1020", height=460,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Volatility (annualised)",
                yaxis_title="Expected Return (annualised)",
                legend=dict(orientation="h", y=1.10),
            )
            st.plotly_chart(fig, use_container_width=True)

            c1, c2 = st.columns(2)
            c1.markdown("**⭐ Max Sharpe portfolio**")
            c1.dataframe(
                pd.DataFrame({"Ticker": op.tickers,
                              "Weight": [round(float(best_s[t]), 4) for t in op.tickers]}),
                use_container_width=True, hide_index=True,
            )
            c2.markdown("**🔶 Min Volatility portfolio**")
            c2.dataframe(
                pd.DataFrame({"Ticker": op.tickers,
                              "Weight": [round(float(min_v[t]), 4) for t in op.tickers]}),
                use_container_width=True, hide_index=True,
            )

    # ---------------------------------------------------------- MONTE CARLO
    with sub_mc:
        st.markdown("<div class='panel-title'>Monte Carlo simulation · "
                    "forward portfolio paths from current optimised weights</div>",
                    unsafe_allow_html=True)
        res = st.session_state.get("opt_result")
        rets = st.session_state.get("opt_returns")
        if not res or rets is None or rets.empty:
            st.caption("Run an optimisation first to populate the asset weights & returns.")
        else:
            mc1, mc2 = st.columns([1, 2])
            with mc1:
                mc_capital = st.number_input("Initial $", value=100_000.0,
                                             step=1000.0, key="mc-cap")
                mc_days = st.slider("Horizon (trading days)", 21, 504, 252,
                                    key="mc-days")
                mc_sims = st.slider("Simulations", 200, 5000, 1000, step=200,
                                    key="mc-sims")
                run_mc = st.button("🎲 Run simulation", type="primary",
                                   use_container_width=True, key="mc-run")
            with mc2:
                if run_mc:
                    with st.spinner("Simulating future paths …"):
                        sim = optz.MonteCarloSimulator(rets).simulate(
                            res.weights, initial_value=mc_capital,
                            simulations=mc_sims, days=mc_days,
                        )
                    st.session_state["mc_result"] = sim

                sim = st.session_state.get("mc_result")
                if sim is not None:
                    days_axis = np.arange(sim.paths.shape[1])
                    # Sample 200 paths for readability
                    idx = np.random.default_rng(42).choice(
                        sim.paths.shape[0], size=min(200, sim.paths.shape[0]),
                        replace=False,
                    )
                    fig = go.Figure()
                    for i in idx:
                        fig.add_trace(go.Scatter(
                            x=days_axis, y=sim.paths[i], mode="lines",
                            line=dict(color="rgba(34,211,238,0.07)", width=1),
                            showlegend=False, hoverinfo="skip",
                        ))
                    median = np.percentile(sim.paths, 50, axis=0)
                    fig.add_trace(go.Scatter(
                        x=days_axis, y=median, mode="lines",
                        line=dict(color="#22d3ee", width=3),
                        name="Median",
                    ))
                    fig.update_layout(
                        template="plotly_dark", paper_bgcolor="#0b1020",
                        plot_bgcolor="#0b1020", height=380,
                        margin=dict(l=10, r=10, t=10, b=10),
                        xaxis_title="Trading day",
                        yaxis_title="Portfolio value ($)",
                    )
                    st.plotly_chart(fig, use_container_width=True)

                    pct = sim.percentiles
                    g1, g2, g3, g4, g5 = st.columns(5)
                    g1.metric("p05", format_currency(pct["p05"], "$"))
                    g2.metric("p25", format_currency(pct["p25"], "$"))
                    g3.metric("Median", format_currency(pct["p50"], "$"))
                    g4.metric("p75", format_currency(pct["p75"], "$"))
                    g5.metric("p95", format_currency(pct["p95"], "$"))
                    r1, r2 = st.columns(2)
                    r1.metric("VaR 95% (terminal loss)",
                              format_currency(sim.var_95, "$"))
                    r2.metric("CVaR 95% (avg tail loss)",
                              format_currency(sim.cvar_95, "$"))

                    if st.button("🧠 AI risk commentary", key="mc-ai-btn"):
                        with st.spinner("GPT-4o explaining the tail …"):
                            st.session_state["mc_ai"] = optz.AIAllocationEngine.explain_monte_carlo(
                                pct, sim.var_95, sim.cvar_95, days=mc_days,
                            )
                    if txt := st.session_state.get("mc_ai"):
                        st.info(txt)

    # ---------------------------------------------------------- CORRELATION
    with sub_corr:
        st.markdown("<div class='panel-title'>Asset correlation heatmap & "
                    "diversification score</div>", unsafe_allow_html=True)
        rets = st.session_state.get("opt_returns")
        if rets is None or rets.empty:
            st.caption("Run an optimisation first — the same universe is reused here.")
        else:
            corr = optz.CorrelationAnalysis.correlation_matrix(rets)
            score = optz.CorrelationAnalysis.diversification_score(corr)
            top = optz.CorrelationAnalysis.highest_correlated_pairs(corr, 5)

            fig = go.Figure(data=go.Heatmap(
                z=corr.values, x=list(corr.columns), y=list(corr.index),
                colorscale="RdBu", zmin=-1, zmax=1, reversescale=True,
                text=corr.round(2).values, texttemplate="%{text}",
            ))
            fig.update_layout(
                template="plotly_dark", paper_bgcolor="#0b1020",
                plot_bgcolor="#0b1020", height=460,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

            sc1, sc2 = st.columns([1, 2])
            sc1.metric("Diversification score", f"{score:.2f}",
                       help="0 = perfectly correlated, 1 = perfectly diversified.")
            sc2.markdown("**Highest correlated pairs**")
            sc2.dataframe(pd.DataFrame(top), use_container_width=True, hide_index=True)

    # ------------------------------------------------------------- REBALANCE
    with sub_reb:
        st.markdown("<div class='panel-title'>Rebalance engine · target weights "
                    "from optimisation → trade list</div>", unsafe_allow_html=True)
        res = st.session_state.get("opt_result")
        if not res:
            st.caption("Optimise first — then we can compute the trades to "
                       "move your current allocation to the target.")
        else:
            target = res.as_dict()
            # Current weights from paper portfolio + the optimisation universe
            total_value = pf.calculate_portfolio_value(p, prices)
            current = {}
            for t in target.keys():
                pos = p.positions.get(t)
                if pos:
                    current[t] = pos.market_value(prices.get(t, pos.avg_price)) / total_value if total_value else 0.0
                else:
                    current[t] = 0.0
            rb1, rb2 = st.columns([1, 1])
            with rb1:
                pv = st.number_input("Portfolio value to rebalance ($)",
                                     value=float(total_value or 100_000.0),
                                     step=1000.0, key="reb-pv")
                min_pct = st.slider("Minimum trade size (%)", 0.0, 5.0, 0.5,
                                    0.1, key="reb-min") / 100
            trades = optz.RebalanceEngine.rebalance(current, target, pv, min_pct)
            with rb2:
                if not trades:
                    st.caption("No trades needed — portfolio already on target.")
                else:
                    df_trades = pd.DataFrame(trades)
                    df_trades["dollar_amount"] = df_trades["dollar_amount"].map(
                        lambda x: format_currency(x, "$"))
                    df_trades["current_weight"] = df_trades["current_weight"].map(format_percentage)
                    df_trades["target_weight"] = df_trades["target_weight"].map(format_percentage)
                    df_trades["weight_delta"] = df_trades["weight_delta"].map(format_percentage)
                    st.dataframe(df_trades, use_container_width=True, hide_index=True)


# =============================================================================
# BACKTEST TAB
# =============================================================================
with tab_backtest:
    st.markdown(
        "<div class='panel-title'>Strategy comparison · "
        f"{ticker} · {period} · {interval}</div>",
        unsafe_allow_html=True,
    )

    bc1, bc2 = st.columns([1, 1.5])
    with bc1:
        chosen = st.multiselect(
            "Strategies", bt.STRATEGIES,
            default=["sma", "rsi", "macd", "combined", "ml", "buyhold"],
            key="bt-strategies",
        )
        bt_capital = st.number_input(
            "Initial capital ($)", value=100_000.0, min_value=1_000.0,
            step=1_000.0, key="bt-capital",
        )
        commission_bps = st.slider(
            "Commission (bps per trade)", 0, 20, 5, key="bt-commission",
        )
        run_bt = st.button("▶️ Run backtest", type="primary",
                           use_container_width=True, key="run-backtest-btn")

    if run_bt and chosen:
        with st.spinner("Simulating strategies …"):
            results = {}
            for s in chosen:
                results[s] = bt.run_backtest(
                    df, s, initial_capital=float(bt_capital),
                    commission=commission_bps / 10_000,
                )
            st.session_state["bt_results"] = results

    results = st.session_state.get("bt_results")
    if results:
        # Equity-curve overlay
        with bc2:
            eqfig = go.Figure()
            palette = ["#22d3ee", "#a78bfa", "#22c55e", "#facc15",
                       "#f472b6", "#fb923c", "#38bdf8", "#94a3b8"]
            for i, r in enumerate(results.values()):
                if r.equity_curve.empty:
                    continue
                eqfig.add_trace(go.Scatter(
                    x=r.equity_curve.index, y=r.equity_curve.values,
                    mode="lines", name=r.strategy.upper(),
                    line=dict(color=palette[i % len(palette)], width=2),
                ))
            eqfig.update_layout(
                template="plotly_dark",
                paper_bgcolor="#0b1020", plot_bgcolor="#0b1020",
                height=380, margin=dict(l=10, r=10, t=30, b=10),
                title="Equity curves", yaxis_title="$",
                legend=dict(orientation="h", y=1.10),
            )
            st.plotly_chart(eqfig, use_container_width=True)

        # Metrics table
        st.markdown("<div class='panel-title'>Metrics</div>", unsafe_allow_html=True)
        rows = []
        for r in results.values():
            rows.append({
                "Strategy": r.strategy.upper(),
                "Final Equity": format_currency(r.final_equity, "$"),
                "Total Return": format_percentage(r.total_return),
                "CAGR": format_percentage(r.cagr),
                "Sharpe": f"{r.sharpe:.2f}",
                "Max DD": format_percentage(r.max_drawdown),
                "Win Rate": format_percentage(r.win_rate),
                "# Trades": r.n_trades,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # AI commentary
        st.markdown("<div class='panel-title'>GPT-4o commentary</div>",
                    unsafe_allow_html=True)
        if st.button("🧠 Generate AI commentary", key="bt-ai-btn"):
            with st.spinner("GPT-4o analysing …"):
                tbl_text = bt.results_to_frame(results).to_string(index=False)
                st.session_state["bt_commentary"] = alpha.generate_backtest_commentary(
                    ticker, tbl_text
                )
        if cmt := st.session_state.get("bt_commentary"):
            st.info(cmt)
    elif not chosen:
        st.caption("Pick at least one strategy and press Run.")
    else:
        st.caption("Configure parameters and press ▶️ Run backtest to compare strategies.")


# =============================================================================
# AI ASSISTANT TAB
# =============================================================================
with tab_assistant:
    st.markdown(
        "<div class='panel-title'>Ask GPT-4o anything about trading, indicators, "
        "risk, or your current data</div>", unsafe_allow_html=True,
    )
    for msg in st.session_state.ai_chat:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if user_q := st.chat_input("Ask the AI assistant…", key="ai-chat-input"):
        st.session_state.ai_chat.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)
        ctx = (f"Loaded ticker: {ticker}, last price {latest_close:.2f}. "
               f"Technical snapshot: {snapshot}. "
               f"ML prediction: action={ml_pred.action}, "
               f"buy_p={ml_pred.buy_probability:.2f}, "
               f"sell_p={ml_pred.sell_probability:.2f}.")
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                reply = alpha.ai_assistant_response(user_q, context=ctx)
            st.markdown(reply)
        st.session_state.ai_chat.append({"role": "assistant", "content": reply})


# Footer
st.markdown("<hr/>", unsafe_allow_html=True)
st.caption(
    f"QuantPulse · MVP build · {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')} · "
    "Educational analytics — not investment advice."
)
