# AI-Powered Algo Trading Platform (MVP)

A startup-style **AI Trading Analytics Platform** built with Python + Streamlit.
It combines classical technical-analysis strategies, a machine-learning alpha
layer, and an OpenAI GPT-4o intelligence layer for trade explanations and an
interactive market assistant.

> **Philosophy** — this is an **analytics & paper-trading** platform, not a
> high-frequency trading system. It is built to demonstrate AI insights,
> portfolio intelligence and risk management.

## Architecture

```
yfinance / Alpha Vantage  →  market_data.py
                                 │
                                 ▼
                          strategy.py  (SMA / RSI / MACD)
                                 │
                                 ▼
                       alpha_generator.py (ML + GPT-4o)
                                 │
                                 ▼
                       risk_management.py
                                 │
                                 ▼
                       execution.py  (Paper / Kite Connect)
                                 │
                                 ▼
                          portfolio.py
                                 │
                                 ▼
                          dashboard.py  (Streamlit UI)
```

## Features

* **Market Data** — yfinance (default) + optional Alpha Vantage
* **Technical Strategies** — SMA crossover, RSI, MACD with combined vote
* **ML Alpha Layer** — RandomForest / LogisticRegression with feature
  engineering (RSI, MACD, SMA, returns, volatility, volume)
* **OpenAI GPT-4o Intelligence** — trade explanations, market summaries,
  AI assistant chat (uses Emergent Universal LLM key — already wired)
* **Risk Engine** — position sizing, volatility-adjusted stop loss,
  risk/reward, exposure & drawdown limits
* **Paper Trading** — full order book with CSV persistence
* **Live Trading** — Zerodha Kite Connect (optional, behind a switch)
* **Portfolio Analytics** — equity curve, drawdown, Sharpe, CAGR, allocation
* **Dashboard** — dark fintech Streamlit UI with KPI cards, candlestick,
  RSI/MACD, equity curve, allocation pie, paper-trading panel and AI chat

## Installation

```bash
cd algo_trading_system
pip install -r requirements.txt
cp .env .env.local         # then edit .env.local with real keys
streamlit run dashboard.py
```

## Environment variables (`.env`)

| Variable                  | Purpose                                            |
|---------------------------|----------------------------------------------------|
| EMERGENT_LLM_KEY          | OpenAI GPT-4o via Emergent Universal LLM key       |
| OPENAI_MODEL              | Default: `gpt-4o`                                  |
| KITE_API_KEY/SECRET/TOKEN | Zerodha Kite Connect (optional)                    |
| ALPHAVANTAGE_API_KEY      | Alpha Vantage data source (optional)               |
| DEFAULT_CAPITAL           | Starting paper-trading capital                     |
| MAX_RISK_PER_TRADE        | Fraction of capital per trade (default 0.02)       |
| MAX_DAILY_DRAWDOWN        | Halt trading when daily drawdown exceeds this      |
| PAPER_TRADING             | `true` / `false` — defaults to paper               |

## How the AI layer works

* **ML model** — trains on every page load using the latest year of data,
  predicting whether the next bar's return will be positive. Outputs BUY/SELL
  probabilities and a confidence score.
* **GPT-4o** — never predicts prices. Used only to translate the technical
  + ML state into a plain-English rationale and to power the market summary
  and AI assistant chat.

## How risk management works

Every paper trade is routed through `validate_trade()` which:
1. Checks daily drawdown is within `MAX_DAILY_DRAWDOWN`.
2. Checks portfolio exposure is within `MAX_PORTFOLIO_EXPOSURE`.
3. Computes a volatility-adjusted stop-loss.
4. Sizes the position so the loss-on-stop ≤ `MAX_RISK_PER_TRADE × capital`.
5. Returns risk amount, stop, target, qty and the risk/reward ratio.

## Deployment

Streamlit Cloud, Railway or Render — point at `dashboard.py`.

## Roadmap

* WebSocket live ticks via Kite Connect ticker
* Telegram / email signal alerts
* Backtesting engine (`backtesting.py`)
* Multi-user accounts & SQL persistence
* Docker support
