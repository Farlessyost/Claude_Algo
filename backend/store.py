"""Lightweight persisted state: settings, rolling logs, P&L history, tuning
results and the latest snapshots shown in the UI. JSON-on-disk, thread-safe."""
from __future__ import annotations

import json
import threading
import time
from collections import deque
from datetime import datetime, timezone
from typing import Any, Deque, Optional

from .config import STATE_DIR, Settings

SETTINGS_FILE = STATE_DIR / "settings.json"
TUNING_FILE = STATE_DIR / "tuning.json"
LOG_FILE = STATE_DIR / "events.log"
LLM_USAGE_FILE = STATE_DIR / "llm_usage.jsonl"

_LOCK = threading.RLock()

MAX_EVENTS = 500

# USD per 1M tokens. Cache reads bill at ~0.1x input; writes ~1.25x. Kept inline
# so the running cost log doesn't depend on a network model-info call.
LLM_PRICING = {
    "claude-opus-4-8":   {"input": 5.0, "output": 25.0},
    "claude-opus-4-7":   {"input": 5.0, "output": 25.0},
    "claude-opus-4-6":   {"input": 5.0, "output": 25.0},
    "claude-sonnet-4-6": {"input": 3.0, "output": 15.0},
    "claude-haiku-4-5":  {"input": 1.0, "output": 5.0},
    "claude-haiku-4-5-20251001": {"input": 1.0, "output": 5.0},  # dated id the API echoes back
    "claude-fable-5":    {"input": 10.0, "output": 50.0},
}


def llm_call_cost_usd(model: str, input_tokens: int, output_tokens: int,
                       cache_read: int = 0, cache_write: int = 0) -> float:
    p = LLM_PRICING.get(model, {"input": 5.0, "output": 25.0})  # default to opus-tier
    return ((input_tokens * p["input"]
             + output_tokens * p["output"]
             + cache_read * p["input"] * 0.1
             + cache_write * p["input"] * 1.25) / 1_000_000.0)


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ecosystem_for_ui(eco: dict) -> dict:
    """Return an ecosystem snapshot with a usable multiscale block.

    Some downstream paths carry a trimmed ecosystem for decision records. The
    UI still needs the scale subnetworks, so reconstruct them from active
    nodes/edges when the full classifier output is missing that block.
    """
    out = dict(eco or {})
    if out.get("multiscale"):
        return out
    nodes = list(out.get("nodes") or [])
    edges_list = list(out.get("edges") or [])
    if not nodes:
        return out
    try:
        from . import ecology as _ecology
        series = {str(n): [0.0] for n in nodes}
        edges = {}
        for e in edges_list:
            a = e.get("from")
            b = e.get("to")
            if a is None or b is None:
                continue
            edges[(str(a), str(b))] = float(e.get("weight") or 0.0)
        out["multiscale"] = _ecology.multiscale_network(series, edges)
    except Exception:
        pass
    return out


class Store:
    def __init__(self) -> None:
        self.settings: Settings = self._load_settings()
        self.events: Deque[dict] = deque(maxlen=MAX_EVENTS)
        self.decisions: Deque[dict] = deque(maxlen=MAX_EVENTS)
        self.orders: Deque[dict] = deque(maxlen=MAX_EVENTS)
        self.fills: Deque[dict] = deque(maxlen=MAX_EVENTS)
        self.pnl_history: Deque[dict] = deque(maxlen=MAX_EVENTS)
        self.snapshots: Deque[dict] = deque(maxlen=MAX_EVENTS)

        # Latest live views for the UI panels
        self.account: dict = {}
        self.market: dict = {}
        # Multi-asset state + ecology snapshot (Trophic Information Forager).
        # Re-populated each trading-loop cycle; the UI reads these out of the
        # snapshot so the food-web/phase ring animates between bar closes.
        self.multiasset: dict = {}
        self.ecosystem: dict = {}
        # CSD risk governor: current risk + rolling history of abs_skew values
        # used for the live z-score. See backend/csd.py current_refined_risk.
        self.csd_state: dict = {"risk": 0.0, "skew_history": [], "skew_now": 0.0,
                                  "applied": False, "gated_at": None}
        # Latest visual chart review by Opus (see backend/visual_review.py).
        # Updated at engine startup and every visual_review_interval_seconds.
        self.visual_review: dict = {"ts": None, "last_ts": None,
                                      "trend": None, "concern": "OK",
                                      "note": "", "image_path": None,
                                      "model": None}
        # Latest blended-alpha decomposition for the alpha-decomposition diagram.
        self.blend_state: dict = {
            "enabled": False, "blended": 0.0,
            "weights": {}, "raw": {}, "parts": {},
        }
        # Fast-guard sub-cycle polling state (see backend/fast_guard.py).
        # Refreshed on every guard tick (~2-3s) so the UI knows whether
        # the guard is active, current intra-cycle drift, and emergency
        # trigger count.
        self.fast_guard_state: dict = {
            "enabled": False, "running": False,
            "last_poll_ts": None, "last_emergency_ts": None,
            "emergencies_triggered": 0,
            "current_intra_cycle_move_bps": 0.0,
            "last_cycle_price": None, "last_live_mid": None,
            "polls": 0,
        }
        # Foraging-cycle / profit-harvest layer (see backend/forager.py).
        # Tracks pnl_R, edge decay, harvest pressure, and the ecological
        # refractory (cooldown) state. Default-inert until forager_enabled.
        self.forager_state: dict = {
            "enabled": False, "in_cooldown": False, "cooldown_until": None,
            "cooldown_remaining_s": 0, "harvest_pressure": 0.0,
            "pnl_R": 0.0, "harvest_scale": 1.0, "reserve_prev": 0.0,
            "csd_prev": 0.0, "captured_cumulative": 0.0, "hunger": "ravenous",
        }
        # Forager harvest log + running captured-profit total, for the live
        # "captured profit over time" diagram (newest-first).
        self.forager_harvests: Deque[dict] = deque(maxlen=500)
        self.forager_captured_cumulative: float = 0.0
        # Autotomy Agent / ecological loss-shedding reflex (see
        # backend/autotomy.py). Always diagnostic; only mutates proposals when
        # settings.autotomy_enabled is true.
        self.autotomy_state: dict = {
            "enabled": False, "in_cooldown": False, "cooldown_until": None,
            "cooldown_remaining_s": 0, "autotomy_pressure": 0.0,
            "loss_R": 0.0, "confirmations": 0, "phase": "idle",
        }
        # Tracked entry price for the open position, used to fill in
        # unrealized_pnl when the /margin/positions endpoint reports 0 (lag).
        # {"entry": float, "sign": +1/-1} or None when flat.
        self.position_entry: Optional[dict] = None
        # Full candle history from the last cycle, so the fast-guard thread can
        # recompute forager features at the live mid between cycles.
        self.candles_full: list = []
        # Fast-tick orderbook from OBLOGGER (updates ~every 1-2s when ATR is
        # hot). Merged into the market view in snapshot_for_ui so the UI sees
        # fresh quotes between full trading-loop cycles.
        self.live_quote: dict = {}
        self.last_decision: dict = {}
        self.loop_status: dict = {
            "running": False,
            "mode": "idle",
            "cycles": 0,
            "last_cycle_ts": None,
            "next_cycle_ts": None,
            "last_error": None,
        }
        self.tuning: dict = self._load_tuning()
        self.connection: dict = {"connected": False, "detail": "not connected"}

        # Daily P&L baseline (for the circuit breaker)
        self.day_key: str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        self.day_start_equity: Optional[float] = None
        self.max_equity: Optional[float] = None  # for drawdown

        # LLM usage: in-memory ring of recent calls + aggregate totals replayed
        # from disk at startup. Persistence is a separate append-only JSONL so
        # the log survives restarts and can be analyzed offline.
        self.llm_recent: Deque[dict] = deque(maxlen=2000)
        self.llm_totals: dict = {
            "calls": 0, "tokens_input": 0, "tokens_output": 0,
            "cache_read": 0, "cache_write": 0, "cost_usd": 0.0,
            "by_model": {},
        }
        self._replay_llm_usage()

    # ---------------- settings ----------------
    def _load_settings(self) -> Settings:
        if SETTINGS_FILE.exists():
            try:
                return Settings.from_dict(json.loads(SETTINGS_FILE.read_text("utf-8")))
            except Exception:
                pass
        return Settings()

    def save_settings(self) -> None:
        with _LOCK:
            SETTINGS_FILE.write_text(
                json.dumps(self.settings.to_dict(), indent=2), encoding="utf-8"
            )

    def update_settings(self, patch: dict) -> Settings:
        with _LOCK:
            d = self.settings.to_dict()
            for k, v in (patch or {}).items():
                if k in d:
                    d[k] = v
            # nested strategy params merge
            if "strategy_params" in (patch or {}):
                sp = dict(self.settings.strategy_params)
                sp.update(patch["strategy_params"] or {})
                d["strategy_params"] = sp
            self.settings = Settings.from_dict(d)
            self.save_settings()
            return self.settings

    # ---------------- tuning ----------------
    def _load_tuning(self) -> dict:
        if TUNING_FILE.exists():
            try:
                return json.loads(TUNING_FILE.read_text("utf-8"))
            except Exception:
                pass
        return {
            "status": "idle",
            "best_score": None,
            "best_params": None,
            "tested": 0,
            "total": 0,
            "started_ts": None,
            "elapsed_s": 0,
            "eta_s": None,
            "current": None,
            "applied": False,
        }

    def save_tuning(self) -> None:
        with _LOCK:
            TUNING_FILE.write_text(json.dumps(self.tuning, indent=2), encoding="utf-8")

    # ---------------- events / logs ----------------
    def log(self, level: str, kind: str, message: str, **extra: Any) -> dict:
        ev = {
            "ts": now_iso(),
            "level": level,
            "kind": kind,
            "message": message,
            **extra,
        }
        with _LOCK:
            self.events.appendleft(ev)
        try:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(ev) + "\n")
        except Exception:
            pass
        return ev

    def add_decision(self, decision: dict) -> None:
        with _LOCK:
            self.last_decision = decision
            self.decisions.appendleft(decision)

    def add_order(self, order: dict) -> None:
        with _LOCK:
            self.orders.appendleft(order)

    def add_fill(self, fill: dict) -> None:
        with _LOCK:
            self.fills.appendleft(fill)

    def add_snapshot(self, snap: dict) -> None:
        with _LOCK:
            self.snapshots.appendleft(snap)

    def record_forager_harvest(self, captured_usd: float, reason=None, pnl_R=None) -> None:
        """Log a forager harvest and advance the running captured-profit total."""
        with _LOCK:
            self.forager_captured_cumulative += float(captured_usd or 0.0)
            self.forager_harvests.appendleft({
                "ts": now_iso(),
                "captured_usd": round(float(captured_usd or 0.0), 4),
                "cumulative_usd": round(self.forager_captured_cumulative, 4),
                "reason": reason, "pnl_R": pnl_R,
            })

    def record_pnl(self, equity: float, realized: float, unrealized: float) -> None:
        with _LOCK:
            self.pnl_history.appendleft(
                {"ts": now_iso(), "equity": equity,
                 "realized": realized, "unrealized": unrealized}
            )

    def roll_day_if_needed(self, equity: Optional[float]) -> None:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with _LOCK:
            if today != self.day_key:
                self.day_key = today
                self.day_start_equity = equity
                if bool(getattr(self.settings, "harvest_reserve_reset_daily", True)):
                    self.forager_captured_cumulative = 0.0
            if self.day_start_equity is None and equity is not None:
                self.day_start_equity = equity
            if equity is not None:
                self.max_equity = max(self.max_equity or equity, equity)

    # ---------------- LLM usage tracking ----------------
    def _replay_llm_usage(self) -> None:
        """Rebuild aggregate totals + the recent ring from the on-disk log."""
        if not LLM_USAGE_FILE.exists():
            return
        try:
            with LLM_USAGE_FILE.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    self._apply_llm_record(rec)
        except Exception:
            pass  # treat a corrupt log as empty rather than crash startup

    def _apply_llm_record(self, rec: dict) -> None:
        self.llm_recent.append(rec)
        t = self.llm_totals
        t["calls"] += 1
        t["tokens_input"] += int(rec.get("input_tokens") or 0)
        t["tokens_output"] += int(rec.get("output_tokens") or 0)
        t["cache_read"] += int(rec.get("cache_read") or 0)
        t["cache_write"] += int(rec.get("cache_write") or 0)
        t["cost_usd"] += float(rec.get("cost_usd") or 0.0)
        m = rec.get("model") or "unknown"
        by = t["by_model"].setdefault(
            m, {"calls": 0, "tokens_input": 0, "tokens_output": 0, "cost_usd": 0.0})
        by["calls"] += 1
        by["tokens_input"] += int(rec.get("input_tokens") or 0)
        by["tokens_output"] += int(rec.get("output_tokens") or 0)
        by["cost_usd"] += float(rec.get("cost_usd") or 0.0)

    def record_llm_call(self, model: str, input_tokens: int, output_tokens: int,
                         cache_read: int = 0, cache_write: int = 0) -> dict:
        cost = llm_call_cost_usd(model, input_tokens, output_tokens,
                                  cache_read, cache_write)
        rec = {
            "ts": now_iso(), "model": model,
            "input_tokens": int(input_tokens),
            "output_tokens": int(output_tokens),
            "cache_read": int(cache_read),
            "cache_write": int(cache_write),
            "cost_usd": round(cost, 6),
        }
        with _LOCK:
            self._apply_llm_record(rec)
            try:
                with LLM_USAGE_FILE.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
            except Exception:
                pass  # disk failure shouldn't break the trading loop
        return rec

    def llm_usage_summary(self) -> dict:
        """Aggregates for the UI panel — totals, today, last-hour rate."""
        now = datetime.now(timezone.utc)
        cutoff_24h = now.timestamp() - 86400
        cutoff_1h = now.timestamp() - 3600
        day_calls = day_cost = hour_calls = hour_cost = 0
        for r in self.llm_recent:
            try:
                ts = datetime.fromisoformat(r["ts"]).timestamp()
            except Exception:
                continue
            if ts >= cutoff_24h:
                day_calls += 1
                day_cost += float(r.get("cost_usd") or 0.0)
                if ts >= cutoff_1h:
                    hour_calls += 1
                    hour_cost += float(r.get("cost_usd") or 0.0)
        return {
            "totals": self.llm_totals,
            "last_24h": {"calls": day_calls, "cost_usd": round(day_cost, 4)},
            "last_1h":  {"calls": hour_calls, "cost_usd": round(hour_cost, 4)},
            "hourly_run_rate_usd": round(hour_cost, 4),  # alias for the UI
        }

    def _market_view_with_live_quote(self) -> dict:
        """Merge the fast-tick live quote into the market view: overrides
        orderbook best bid/ask + mid price, and mark-to-markets the most
        recent candle's close/high/low so the chart 'wiggles' between bar
        closes. If we have no live quote yet, returns market unchanged."""
        m = dict(self.market or {})
        lq = self.live_quote or {}
        if not lq:
            return m
        ob = dict(m.get("orderbook") or {})
        ob.update({
            "best_bid": lq.get("bid", ob.get("best_bid")),
            "best_ask": lq.get("ask", ob.get("best_ask")),
            "spread": (lq.get("ask") - lq.get("bid"))
                       if (lq.get("ask") is not None and lq.get("bid") is not None)
                       else ob.get("spread"),
            "spread_bps": lq.get("spread_bps", ob.get("spread_bps")),
        })
        m["orderbook"] = ob
        mid = lq.get("mid")
        if mid is not None:
            m["price"] = mid
            # Mark-to-market the in-progress (most recent) candle.
            candles = list(m.get("candles") or [])
            if candles:
                c = dict(candles[-1])
                c["close"] = mid
                if c.get("high") is None or mid > c["high"]:
                    c["high"] = mid
                if c.get("low") is None or mid < c["low"]:
                    c["low"] = mid
                candles[-1] = c
                m["candles"] = candles
        m["live_quote_ts"] = lq.get("ts")
        return m

    def snapshot_for_ui(self) -> dict:
        with _LOCK:
            return {
                "ts": now_iso(),
                "settings": self.settings.to_dict(),
                "connection": self.connection,
                "account": self.account,
                "market": self._market_view_with_live_quote(),
                "multiasset": self.multiasset,
                "ecosystem": _ecosystem_for_ui(self.ecosystem),
                "csd_state": dict(self.csd_state),
                "visual_review": dict(self.visual_review),
                "blend_state": dict(self.blend_state),
                "fast_guard_state": dict(self.fast_guard_state),
                "forager_state": dict(self.forager_state),
                "forager_harvests": list(self.forager_harvests)[:200],
                "autotomy_state": dict(self.autotomy_state),
                "last_decision": self.last_decision,
                "loop_status": self.loop_status,
                "tuning": self.tuning,
                "day_start_equity": self.day_start_equity,
                "max_equity": self.max_equity,
                "llm_usage": self.llm_usage_summary(),
                "logs": {
                    "events": list(self.events)[:120],
                    "decisions": list(self.decisions)[:60],
                    "orders": list(self.orders)[:60],
                    "fills": list(self.fills)[:60],
                    "pnl_history": list(self.pnl_history)[:120],
                    "snapshots": list(self.snapshots)[:30],
                    "llm_calls": list(self.llm_recent)[-60:],
                },
            }


STORE = Store()
