# Test Credentials — QuantPulse (React + FastAPI rebuild)

## Emergent Google Auth (production flow)
- Any Google account works — Emergent Auth handles OAuth end-to-end.
- After login users land at `/dashboard`.
- No password to store; no allow-list.

## Seeded test session (for automation / testing agent)
Use these when the testing agent needs a pre-authenticated session **without going through Google OAuth**.

- `user_id`        : `test-user-quantpulse`
- `email`          : `test@quantpulse.dev`
- `name`           : `Test Trader`
- `session_token`  : `test_session_quantpulse_2026`
- Cookie to set    : `session_token=test_session_quantpulse_2026; Domain=<preview-host>; Path=/; Secure; HttpOnly; SameSite=None`
- Header alt       : `Authorization: Bearer test_session_quantpulse_2026`

Re-seed script:
```bash
mongosh mongodb://localhost:27017 --quiet --eval "
db = db.getSiblingDB('test_database');
db.users.deleteMany({user_id: 'test-user-quantpulse'});
db.user_sessions.deleteMany({session_token: 'test_session_quantpulse_2026'});
db.users.insertOne({user_id:'test-user-quantpulse', email:'test@quantpulse.dev', name:'Test Trader', picture:'https://api.dicebear.com/7.x/notionists/svg?seed=trader', created_at:new Date()});
db.user_sessions.insertOne({user_id:'test-user-quantpulse', session_token:'test_session_quantpulse_2026', expires_at:new Date(Date.now()+7*864e5), created_at:new Date()});
"
```

## Environment status
- EMERGENT_LLM_KEY: real (Universal Key) — GPT-4o enabled
- Kite Connect (per-user): user provides API Key + Secret in Settings page → OAuth flow via Kite login URL
- Alpha Vantage: not required (yfinance is default)
