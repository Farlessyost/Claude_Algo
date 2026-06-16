# BTC Perp Trading Console

A local operator console for researching and running autonomous Kalshi
BTCPERP perpetual-futures strategies. The app combines a FastAPI backend, a
browser-based trading console, live Kalshi margin-account telemetry,
deterministic strategy engines, ecological market-state diagnostics, tuning,
backtesting, and optional LLM-assisted review.

> Warning: this project can place real orders with real money at leverage when
> live trading is armed. It is experimental trading software, not financial
> advice. Use demo mode first, understand the code, and accept that leveraged
> trading can lose capital quickly.

## Running Agent

Screenshots below are from a running local console. Account balance and P&L
panels are intentionally kept out of frame.

![Control panel](docs/screenshots/control-panel.png)

![Agent telemetry](docs/screenshots/agent-telemetry.png)

## What It Does

- Connects to Kalshi BTCPERP margin/perps endpoints with signed REST requests.
- Maintains a live local dashboard for account state, market state, strategy
  decisions, risk checks, fills, logs, tuning, and LLM usage.
- Aggregates Kalshi 1m candles into local 2m, 3m, 5m, and 15m timeframes.
- Runs deterministic strategy variants including robust MPC, cost-aware MPC,
  mean reversion, trend, breakout, momentum, and blended alpha components.
- Includes ecology-inspired diagnostics such as the Trophic Information
  Forager, CSD governor, fast guard, forager harvest/cooldown, and autotomy
  loss-shedding layers.
- Supports backtesting, parameter sweeps, and live setting persistence through
  local JSON state.
- Can call an Anthropic model for optional review workflows when an API key is
  configured.

## Quick Start

1. Install Python 3.11+ on Windows.

2. Create a local Kalshi credential file named `trading_API_keys.txt` in the
   project root. The file is ignored by git.

   ```text
   <kalshi-api-key-id>
   -----BEGIN RSA PRIVATE KEY-----
   ...
   -----END RSA PRIVATE KEY-----
   ```

3. Optionally set an Anthropic key for LLM-assisted review.

   ```powershell
   $env:ANTHROPIC_API_KEY = "sk-ant-..."
   ```

4. Launch the console.

   ```powershell
   .\run.ps1
   ```

5. Open <http://127.0.0.1:8787>.

## Recommended First Run

Start in `demo` mode before using production credentials. Demo mode is the
right place to verify signing, endpoint compatibility, ticker configuration,
order plumbing, and UI behavior without risking production capital.

To switch environments, use the Control Panel in the UI or edit
`state/settings.json` after the first launch.

## Safety Model

Live trading requires a one-time arm action in the UI. After it is armed, the
loop can place orders without per-order confirmation until stopped or killed.

Guardrails live in `backend/config.py` and persisted settings. Important
controls include:

- `max_leverage`
- `max_position_notional_usd`
- `daily_loss_limit_*`
- `min_liquidation_buffer_pct`
- `min_account_equity_usd`
- kill switch and live-arm state

The application is intentionally local-first. Secrets, runtime state, logs,
keys, Python caches, virtual environments, and local agent config directories
are excluded by `.gitignore`.

## How Decisions Are Made

Each live cycle gathers the current account, positions, resting orders,
BTCPERP order book, candles, derived indicators, market regime, ecology state,
and risk context. Strategy modules produce a target action and sizing proposal.
Risk checks and execution rules then decide whether to hold, post, cross,
reduce, close, or stop.

The strategy path is deterministic by default. LLM review is optional and is
used only when an API key and relevant settings are enabled.

## Project Layout

```text
backend/
  app.py             FastAPI app, static UI serving, and API endpoints
  kalshi_client.py   Signed Kalshi REST client
  config.py          Settings, credential loading, and guardrail defaults
  store.py           Local state, logs, snapshots, and P&L history
  engine.py          Live trading cycle and background loop
  executor.py        Order submission, cancellation, and replacement
  risk.py            Position sizing and guardrail checks
  market_data.py     Candles, order book metrics, and aggregation
  indicators.py      EMA, RSI, ROC, ATR, volatility helpers
  signals.py         Strategy signal construction
  strategy.py        Deterministic strategy proposals
  tuning.py          Parameter search and auto-apply support
  backtest.py        Historical evaluation
  ecology.py         Trophic information network diagnostics
  csd.py             Critical-slowing-down risk governor
  forager.py         Profit harvest and refractory cooldown logic
  autotomy.py        Toxic-loss exit layer
  visual_review.py   Optional visual-review workflow

static/
  index.html         Main operator console
  app.js             UI state management and rendering
  styles.css         Dashboard styling
  watch.html         Watch view

docs/screenshots/   Running-console screenshots for documentation
```

## Troubleshooting

- `Connect` returns `401`: check the Kalshi signing string in
  `backend/kalshi_client.py`. The expected form is
  `timestamp_ms + METHOD + /trade-api/v2<path>`.
- Market data is empty: confirm the live ticker with Kalshi and update the
  product/ticker setting.
- LLM review does not run: set `ANTHROPIC_API_KEY` and enable the relevant UI
  settings.
- The UI shows stale state: stop the server, inspect local files in `state/`,
  and restart with `.\run.ps1`.

## Maintainer Notes

This repository is maintained as a public research and operator-console project
for autonomous market-agent experimentation. Contributions should preserve the
local-first secret model and treat live-trading behavior as high-risk code:
small changes in sizing, execution, or guardrails can have real financial
impact.
