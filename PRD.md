# PRD — QuantPulse · React + FastAPI Institutional Trading Platform

## Original problem statement
Build a PROFESSIONAL AI ALGO TRADING PLATFORM with an institutional-grade connected workflow: Google login → connect Zerodha Kite → portfolio auto-syncs → ticker+strategy selection → integrated signal generation with technical + alpha + risk + ML + AI reasoning + integrated backtest → paper OR live execution. Stack: React+TypeScript+Tailwind+ShadCN+Recharts/TradingView, FastAPI+Python, MongoDB, Redis (deferred), scikit-learn/XGBoost/OpenAI, Zerodha Kite, Docker.

## User choices (final)
- 1a. Replace Streamlit app entirely with React + FastAPI.
- 2a. MongoDB (Emergent-native). 2c. Skip Redis for v1.
- 3a. Emergent-managed Google Auth.
- 4a. TradingView Lightweight Charts (free).
- 5a. v1 must-haves only. Deferred: drag-and-drop strategy builder, voice commands, Telegram, multi-broker, LSTM/RL, mobile polish.

## What's been implemented (v1, 2026-05-14)
### Backend `/app/backend/server.py` (FastAPI + MongoDB)
- `/api/auth/google/session` — exchange Emergent session_id → server session cookie
- `/api/auth/me`, `/api/auth/logout`
- `/api/broker/kite/login-url` · `/session` · `/status` · `/holdings` · `/positions` · `/disconnect` (per-user Kite creds stored on user doc)
- `/api/market/candles` · `/api/market/quote` (yfinance)
- `/api/strategies` (10 prebuilt) · `/api/strategies/{id}/run` (signal + integrated backtest + ML alpha + risk validation)
- `/api/portfolio/paper` · `/api/portfolio/paper/orders` (BUY/SELL) · `/api/portfolio/paper/reset`
- `/api/risk/rules` (GET/PUT per-user)
- `/api/ai/chat` · `/api/ai/insights` — GPT-4o via emergentintegrations, offloaded via `asyncio.to_thread` (no event-loop deadlock)
- CORSMiddleware with allow_origin_regex + allow_credentials; same-origin cookie auth

### Frontend `/app/frontend/src/`
- `App.js` — React Router with `RouteSwitch` that captures `session_id=` in URL fragment
- Pages: `LoginPage`, `AuthCallback`, `DashboardPage`, `MarketPage`, `PortfolioPage`, `StrategyPage`, `SignalsPage`, `PaperTradingPage`, `RiskPage`, `AIInsightsPage`, `HistoryPage`, `SettingsPage`
- `components/AppShell.jsx` — sidebar with 10 nav items, user card, broker status, glass style
- `components/CandlestickChart.jsx` + `EquityChart.jsx` — TradingView Lightweight Charts v5 (`chart.addSeries(CandlestickSeries, ...)` API)
- `store/auth.js` — Zustand auth store (checkAuth, logout)
- `lib/api.js` — axios `baseURL: "/api"` + `withCredentials: true` (same-origin)
- Dark institutional theme via `App.css` + Tailwind; sonner toasts; lucide-react icons

### Reused quant modules `/app/algo_trading_system/`
`market_data`, `strategy`, `alpha_generator`, `risk_management`, `backtest`, `execution`, `optimizer/` — imported directly by FastAPI backend.

### Testing
- iteration_3: 14/16 backend, 14/15 frontend (blocked on AI async bug)
- iteration_4: **16/16 backend + all AI scenarios PASS, no regressions**

## Backlog
- Kite live-order placement wired into Paper Trading page (currently paper only)
- Custom strategy drag-and-drop builder
- WebSocket live ticks (Kite ticker)
- Advanced ML models (LSTM / RL)
- Telegram / email alerts
- Multi-broker (Groww stub already in codebase)
- Docker Compose + CI/CD
- Mobile responsive polish

## Next tasks
1. Redeploy from GitHub → Emergent production (fixes the currently-broken quant-analytics-5.emergent.host page)
2. User provides real Kite API keys → wire live order execution
3. Add trade export (CSV)
