# BTC Perp Trading Console

A local, single-operator console that trades **Kalshi BTCPERP perpetual futures
autonomously and live** using your production Kalshi margin account. Opus
(`claude-opus-4-8`) — or a deterministic quant strategy when no LLM key is set —
makes profit-focused decisions hour-by-hour and submits real orders at up to
**5.8x** leverage with no per-order confirmation once you arm live trading.

> ⚠️ **This places real orders with real money at leverage.** Read "Safety model"
> below. The author of this tool is not responsible for trading losses.

---

## Quick start

1. Put your Kalshi credentials in `trading_API_keys.txt` (already present):
   line 1 = API key id (UUID), then the RSA private key PEM block.
2. (Optional, for Opus decisions) set your Anthropic key:
   ```powershell
   $env:ANTHROPIC_API_KEY = "sk-ant-..."
   ```
3. Launch:
   ```powershell
   .\run.ps1
   ```
4. Open <http://127.0.0.1:8787>.

## Operating it (the flow you asked for)

1. Open the UI → it auto-connects to Kalshi **production**.
2. Product is **BTC Perp**; set **Timeframe = 15m**; leverage shows **5.8x**.
3. Tick **Use Actual Account Balance**, **Let Opus Decide**, **Auto-Submit Orders**.
4. Click **⚡ Enable Live Autonomous Trading** (you confirm this **once**).
5. Click **▶ Start Hourly Live Trading Loop** — the bot now trades by itself.
6. Stop anytime with **⏸ Stop Trading**, or **⛔ Kill Switch** (halts + disarms +
   cancels resting orders).

Every other button is live too: **Run One Live Decision Now**, **Backtest**,
**Tune Strategy** (auto-applies best params to the live loop), **Save Settings**.

## How decisions are made

Each cycle: fetch real balance/positions/orders → fetch BTCPERP orderbook +
candles → compute trend/momentum/mean-reversion/regime/ATR/RSI → deterministic
strategy forms a proposal → (optional) Opus reviews the full context and picks
the most profitable action → size from real equity × leverage → run risk checks →
submit/cancel/replace/reduce/close as needed.

**15m candles:** Kalshi's candlestick API only serves 1m/1h/1d, so the app pulls
1m candles and **aggregates them into 5m/15m locally**. The 15m timeframe is used
in live trading, backtest, and tuning alike.

## Safety model (read this)

You asked for "approve once, then trade." That is exactly what this does — there
is **no per-order confirmation, no tickets, no confirm phrases.** What remains:

- **One-time arm** (`Enable Live Autonomous Trading`) + **Kill Switch**.
- A few **automatic, non-prompting guardrails** that protect capital from a
  runaway bug (a bug at 5.8x can liquidate an account in minutes — that destroys
  profit, which is the opposite of your goal). All are in `backend/config.py`
  and adjustable, including **off**:
  - `max_leverage` (hard cap, 5.8x)
  - `max_position_notional_usd` (0 = equity × leverage)
  - `daily_loss_limit_*` circuit breaker (default 25% of day-start equity)
  - `min_liquidation_buffer_pct` / liquidation-risk block
  - `min_account_equity_usd`
- Everything else (spread, volatility, funding, …) is **warn-only** and never
  blocks a trade, per your preference for prioritizing profit.

## Strongly recommended first run

Set `environment` to `demo` in the Control Panel (or `state/settings.json`) and
do one **demo** session first to confirm signing, tickers, and order plumbing
against your account before going to production. The exact BTCPERP ticker and the
REST signing string are validated by the **Connect** button (it calls
`/margin/balance`). If Connect fails, see "Troubleshooting".

## Troubleshooting

- **Connect fails / 401:** the REST signing message is
  `timestamp_ms + METHOD + /trade-api/v2<path>`. If Kalshi rejects it, adjust
  `_sign`/`_request` in `backend/kalshi_client.py` (path prefix is the usual
  culprit).
- **Empty market data / wrong ticker:** confirm the live ticker via
  `GET /margin/markets`; set it in the Control Panel product field / settings.
- **Opus not used:** set `ANTHROPIC_API_KEY`; otherwise the deterministic
  strategy drives the loop (still fully autonomous).

## Files

```
backend/
  config.py          credentials + persisted settings + guardrail defaults
  kalshi_client.py   signed REST client for the margin/perps API
  market_data.py     candles (1m->5m/15m), indicators, orderbook metrics
  indicators.py      EMA / RSI / ROC / ATR / stdev
  account.py         real margin-account view
  strategy.py        deterministic quant proposal (tunable params)
  decision_engine.py Opus review + strategy fallback
  risk.py            sizing + checks (block vs warn)
  executor.py        submits/cancels real orders
  engine.py          the autonomous cycle + background loop
  tuning.py          param search -> auto-applies to live
  backtest.py        historical evaluation of the live strategy
  store.py           persisted state + logs
  app.py             FastAPI: serves UI + all endpoints
static/              the console UI (index.html, styles.css, app.js)
```
