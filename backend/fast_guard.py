"""Sub-cycle polling guard.

The main trading loop sleeps for vol_aware_interval seconds between cycles
(5-30s on the conservative preset after 2026-06-15). Even with live-quote
MTM applied inside each cycle, anything that happens BETWEEN cycles is
invisible to the resilience layers until the next cycle fires.

This module runs a separate background thread that polls STORE.live_quote
on a fast cadence (default 3s) and triggers an immediate engine.run_cycle()
when the mid has moved more than fast_guard_emergency_move_bps since the
last cycle. The emergency cycle re-runs the full resilience stack (ATR,
CSD, ecology phase, blend) on the freshest live data and can adjust
position size accordingly.

Re-entrancy: a single threading.Lock keeps emergency runs from overlapping
with the main loop's scheduled cycle. If a cycle is already in progress
when the guard fires, the guard skips this tick and tries again next poll.

Public API:
  FAST_GUARD.start(engine, settings_provider, cycle_runner)
  FAST_GUARD.stop()
  FAST_GUARD.state -> dict (mirrored to STORE for the UI)
"""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Callable, Optional


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class FastGuard:
    def __init__(self) -> None:
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._cycle_lock = threading.Lock()
        self._cycle_runner: Optional[Callable[[], dict]] = None
        self._settings_provider: Optional[Callable[[], object]] = None
        self.state: dict = {
            "enabled": False,
            "last_poll_ts": None,
            "last_emergency_ts": None,
            "emergencies_triggered": 0,
            "current_intra_cycle_move_bps": 0.0,
            "last_cycle_price": None,
            "last_live_mid": None,
            "polls": 0,
            "running": False,
        }
        self._last_cycle_price: Optional[float] = None

    # ----------------------------------------------------- lifecycle
    def start(self, settings_provider: Callable[[], object],
              cycle_runner: Callable[[], dict]) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._settings_provider = settings_provider
        self._cycle_runner = cycle_runner
        self._stop.clear()
        self._last_cycle_price = None
        self.state["enabled"] = True
        self.state["running"] = True
        self._thread = threading.Thread(target=self._loop, daemon=True,
                                          name="fast-guard")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self.state["running"] = False

    # ----------------------------------------------------- hooks
    def cycle_completed(self, cycle_price: Optional[float]) -> None:
        """Called by the engine after each completed cycle so the guard
        knows the baseline price to measure intra-cycle drift against.

        Concurrency: stored as a single float assignment — atomic in CPython,
        no extra lock needed."""
        if cycle_price is not None and cycle_price > 0:
            self._last_cycle_price = float(cycle_price)
            self.state["last_cycle_price"] = float(cycle_price)
            self.state["current_intra_cycle_move_bps"] = 0.0

    # ----------------------------------------------------- loop
    def _loop(self) -> None:
        # Lazy import to avoid circular imports at module load
        from .store import STORE
        STORE.fast_guard_state = dict(self.state)
        while not self._stop.is_set():
            s = self._settings_provider() if self._settings_provider else None
            poll_seconds = float(getattr(s, "fast_guard_poll_seconds", 3) or 3)
            if not bool(getattr(s, "fast_guard_enabled", True)):
                # Disabled — sleep a couple polls and re-check the toggle
                self._wait(max(2.0, poll_seconds))
                continue

            self._tick(s)

            # Persist into STORE for the UI on every tick
            STORE.fast_guard_state = dict(self.state)
            self._wait(poll_seconds)

    def _wait(self, seconds: float) -> None:
        slept = 0.0
        step = 0.5
        while slept < seconds and not self._stop.is_set():
            d = min(step, seconds - slept)
            time.sleep(d)
            slept += d

    def _tick(self, settings) -> None:
        from .store import STORE
        self.state["polls"] = int(self.state.get("polls", 0)) + 1
        self.state["last_poll_ts"] = _now_iso()
        live_q = STORE.live_quote or {}
        mid = live_q.get("mid")
        if mid is None or mid <= 0:
            return
        self.state["last_live_mid"] = float(mid)
        base = self._last_cycle_price
        if base is None or base <= 0:
            return
        move_bps = (float(mid) - base) / base * 10_000.0
        self.state["current_intra_cycle_move_bps"] = round(move_bps, 2)
        emergency_bps = float(getattr(settings, "fast_guard_emergency_move_bps",
                                         10.0) or 10.0)
        move_trigger = abs(move_bps) >= emergency_bps

        # Forager sub-cycle harvest: every poll (~1s), check whether the forager
        # would harvest at the LIVE mid (profit appears and evaporates between
        # full cycles in churn). Read-only check; if true we fire a real cycle
        # which does the actual harvest + execution + cooldown.
        forager_trigger = False
        if bool(getattr(settings, "forager_fast_harvest", True)):
            try:
                from . import forager as _forager
                # Recompute ALL forager features fresh at the live mid (~1s) and
                # publish them so the panel + decision use 1s-fresh inputs, not
                # cycle-stale ones. live_signals merges over the cached state, so
                # cooldown/phase are preserved.
                sig = _forager.live_signals(settings, STORE, float(mid))
                STORE.forager_state = sig
                if not move_trigger:
                    forager_trigger = _forager.harvest_decision(settings, STORE, sig)
            except Exception:
                forager_trigger = False

        autotomy_trigger = False
        if bool(getattr(settings, "autotomy_fast_eject", True)):
            try:
                from . import autotomy as _autotomy
                sig = _autotomy.live_signals(settings, STORE, float(mid))
                STORE.autotomy_state = sig
                if not move_trigger and not forager_trigger:
                    autotomy_trigger = _autotomy.eject_decision(settings, STORE, sig)
            except Exception:
                autotomy_trigger = False

        if not (move_trigger or forager_trigger or autotomy_trigger):
            return

        # Threshold crossed: trigger an immediate cycle. Acquire the cycle
        # lock non-blocking so we never queue up behind an in-flight cycle.
        if not self._cycle_lock.acquire(blocking=False):
            return  # main loop / a previous emergency is already running
        try:
            if autotomy_trigger and not move_trigger and not forager_trigger:
                reason = "autotomy eject @ live mid"
            elif forager_trigger and not move_trigger:
                reason = "forager harvest @ live mid"
            else:
                reason = f"intra-cycle move {move_bps:+.2f}bps >= {emergency_bps:.1f}bps"
            self.state["last_emergency_ts"] = _now_iso()
            self.state["emergencies_triggered"] = int(
                self.state.get("emergencies_triggered", 0)) + 1
            STORE.log("warn", "fast_guard",
                       f"{reason} -> firing immediate cycle",
                       event="fast_guard_fire",
                       intra_cycle_move_bps=round(move_bps, 2),
                       threshold_bps=emergency_bps,
                       forager_harvest=bool(forager_trigger and not move_trigger),
                       autotomy_eject=bool(autotomy_trigger and not move_trigger and not forager_trigger))
            STORE.fast_guard_state = dict(self.state)
            if self._cycle_runner:
                try:
                    self._cycle_runner()
                except Exception as e:
                    STORE.log("warn", "fast_guard",
                               f"emergency cycle errored: {e}")
        finally:
            self._cycle_lock.release()

    # ----------------------------------------------------- cycle gate
    def acquire_cycle_lock(self) -> bool:
        """Used by the main loop to prevent the guard from firing during
        a scheduled cycle. Returns True if acquired; caller must release."""
        return self._cycle_lock.acquire(blocking=False)

    def release_cycle_lock(self) -> None:
        try:
            self._cycle_lock.release()
        except RuntimeError:
            pass   # lock wasn't held — defensive against double-release


FAST_GUARD = FastGuard()
