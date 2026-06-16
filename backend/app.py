"""FastAPI backend for the BTC Perp Trading Console. Serves the UI and exposes
endpoints for every control. Run: uvicorn backend.app:app --port 8787"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import market_data
from .config import (ROOT, AGGRESSIVENESS_PRESETS, TRADE_SIZE_PRESETS,
                     anthropic_api_key)
from .engine import ENGINE
from .oblogger import OBLOGGER
from .store import STORE
from .tuning import TUNER

app = FastAPI(title="BTC Perp Trading Console")

STATIC_DIR = ROOT / "static"


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/watch")
def watch():
    return FileResponse(STATIC_DIR / "watch.html")


@app.get("/api/visual_review/latest.png")
def visual_review_image():
    """Serve the latest visual-review chart PNG. Returns 404 until the first
    review has been generated."""
    from .visual_review import VISUAL_REVIEW_IMAGE_PATH
    if not VISUAL_REVIEW_IMAGE_PATH.exists():
        return JSONResponse({"detail": "no visual review yet"}, status_code=404)
    return FileResponse(VISUAL_REVIEW_IMAGE_PATH, media_type="image/png")


# ------------------------------------------------------------------ state
@app.get("/api/state")
def get_state():
    snap = STORE.snapshot_for_ui()
    snap["opus_available"] = bool(anthropic_api_key())
    snap["ob_log"] = {"running": OBLOGGER.running(), "count": OBLOGGER.count,
                      "last_ts": OBLOGGER.last_ts, "error": OBLOGGER.last_error}
    return snap


@app.post("/api/connect")
def connect():
    return ENGINE.connect()


# ---------------------------------------------------------------- settings
class SettingsPatch(BaseModel):
    patch: dict


@app.post("/api/settings")
def update_settings(body: SettingsPatch):
    s = STORE.update_settings(body.patch)
    # reconnect if environment changed
    if "environment" in body.patch:
        ENGINE.connect()
    return s.to_dict()


@app.post("/api/save_settings")
def save_settings():
    STORE.save_settings()
    STORE.log("info", "settings", "Settings saved")
    return {"saved": True}


# -------------------------------------------------------- live arming / loop
@app.post("/api/enable_live")
def enable_live():
    """The single one-time arm for autonomous live trading."""
    STORE.update_settings({
        "live_autonomous_armed": True,
        "allow_live_orders": True,
        "auto_submit_orders": True,
        "kill_switch_engaged": False,
        "mode": "live_autonomous",
    })
    ENGINE.connect()
    STORE.log("warn", "live",
              "LIVE AUTONOMOUS TRADING ENABLED — bot may now place real orders")
    return {"armed": True, "connection": STORE.connection}


class StartLoopBody(BaseModel):
    duration_minutes: Optional[float] = None


@app.post("/api/start_loop")
def start_loop(body: Optional[StartLoopBody] = None):
    if not STORE.settings.live_autonomous_armed:
        return JSONResponse(
            {"error": "Enable Live Autonomous Trading first."}, status_code=400)
    mins = body.duration_minutes if body else None
    return ENGINE.start_loop(duration_minutes=mins)


@app.post("/api/stop")
def stop():
    return ENGINE.stop_loop()


@app.post("/api/kill")
def kill():
    return ENGINE.kill()


@app.post("/api/reset_kill")
def reset_kill():
    STORE.update_settings({"kill_switch_engaged": False})
    STORE.log("info", "kill", "Kill switch reset")
    return {"reset": True}


@app.post("/api/run_once")
def run_once():
    return ENGINE.run_cycle()


class AggressivenessBody(BaseModel):
    level: str


@app.post("/api/aggressiveness")
def set_aggressiveness(body: AggressivenessBody):
    level = (body.level or "").lower()
    preset = AGGRESSIVENESS_PRESETS.get(level)
    if preset is None:
        return JSONResponse(
            {"error": f"unknown level '{body.level}'. Use one of: "
                       + ", ".join(AGGRESSIVENESS_PRESETS)}, status_code=400)
    patch = {"aggressiveness": level, **preset}
    s = STORE.update_settings(patch)
    sp = s.strategy_params
    STORE.log("warn", "settings",
              f"Aggressiveness -> {level.upper()} "
              f"(band={sp.get('band')} deadband_bps={sp.get('deadband_bps')} "
              f"taker_k={s.taker_threshold_k} chase={s.chase_delay_bars} "
              f"model={s.model})")
    return {"aggressiveness": level, "applied": preset}


class TradeSizeBody(BaseModel):
    level: str


@app.post("/api/trade_size")
def set_trade_size(body: TradeSizeBody):
    level = (body.level or "").lower()
    preset = TRADE_SIZE_PRESETS.get(level)
    if preset is None:
        return JSONResponse(
            {"error": f"unknown level '{body.level}'. Use one of: "
                       + ", ".join(TRADE_SIZE_PRESETS)}, status_code=400)
    patch = {"trade_size": level, **preset}
    s = STORE.update_settings(patch)
    STORE.log("warn", "settings",
              f"Trade size -> {level.upper()} "
              f"(position_scale={s.position_scale}x; max_leverage cap "
              f"{s.max_leverage}x still applies)")
    return {"trade_size": level, "applied": preset}


# ------------------------------------------------------------- backtest
@app.post("/api/backtest")
def backtest_now():
    from . import backtest as bt
    client = ENGINE._ensure_client()
    if not client:
        return JSONResponse({"error": "not connected"}, status_code=400)
    s = STORE.settings
    candles = market_data.fetch_candles(client, s.ticker, s.timeframe, lookback_bars=600)
    fee = s.assumed_fee_bps / 1e4
    report = bt.run_backtest(candles, s.strategy_params, variant=s.strategy,
                             leverage=s.leverage_target, fee_pct=fee)
    STORE.log("info", "backtest",
              f"Backtest {s.strategy}@{s.timeframe} (fee {s.assumed_fee_bps}bps): "
              f"return {report.get('total_return_pct')}% "
              f"maxDD {report.get('max_drawdown_pct')}% trades {report.get('trades')}")
    return report


# ------------------------------------------------------------- tuning
class TuneBody(BaseModel):
    iterations: int = 60
    auto_apply: bool = True


@app.post("/api/tune")
def tune(body: TuneBody):
    client = ENGINE._ensure_client()
    if not client:
        return JSONResponse({"error": "not connected"}, status_code=400)
    s = STORE.settings
    candles = market_data.fetch_candles(client, s.ticker, s.timeframe, lookback_bars=800)
    return TUNER.start(candles, iterations=body.iterations,
                       auto_apply=body.auto_apply, leverage=s.leverage_target,
                       variant=s.strategy, fee_pct=s.assumed_fee_bps / 1e4)


@app.post("/api/tune/stop")
def tune_stop():
    TUNER.stop()
    return {"stopping": True}


# Mount static assets last so /api/* wins.
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.on_event("startup")
def _startup():
    STORE.log("info", "app", "BTC Perp Trading Console started")
    ENGINE.connect()
    OBLOGGER.start(lambda: ENGINE._ensure_client(), lambda: STORE.settings.ticker)
    STORE.log("info", "oblog", "Order-book logger started (15s interval)")
