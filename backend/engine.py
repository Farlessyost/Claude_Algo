"""The autonomous trading engine: one cycle = fetch real state -> analyze ->
decide -> size -> risk-check -> execute -> log. A background thread runs the
loop hourly (or at the configured interval) until stopped or the kill switch is
engaged. The operator arms live trading ONCE; thereafter the loop trades on its
own with no per-order confirmation."""
from __future__ import annotations

import threading
import time
from datetime import datetime, timezone
from typing import Optional

from . import account as account_mod
from . import decision_engine, executor, market_data, risk, strategy
from .config import Credentials, Settings, compute_loop_interval, load_credentials
from .kalshi_client import KalshiClient
from .store import STORE


class Engine:
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._client: Optional[KalshiClient] = None
        self._creds: Optional[Credentials] = None
        self._lock = threading.RLock()
        self._deadline: Optional[float] = None
        self._peak_eq: Optional[float] = None
        self._cycles_since_retrain: int = 0

    # ----------------------------------------------------------- connection
    def connect(self) -> dict:
        creds = load_credentials()
        if not creds:
            STORE.connection = {"connected": False,
                                "detail": "key file missing/unparseable"}
            return STORE.connection
        self._creds = creds
        try:
            self._client = KalshiClient(creds, STORE.settings.environment)
            bal = self._client.get_balance()
            ok = "subaccount_balances" in bal
            fee_taker = None
            try:
                ft = self._client.get_fee_tiers()
                fee_taker = (ft.get("taker_fee_rates", {}) or {}).get(STORE.settings.ticker)
            except Exception:
                pass
            STORE.connection = {
                "connected": ok,
                "environment": STORE.settings.environment,
                "key_id_masked": creds.key_id[:8] + "…",
                "taker_fee_rate": fee_taker,
                "detail": "authenticated" if ok else "unexpected balance response",
            }
            STORE.log("info", "connection",
                      f"Connected to Kalshi {STORE.settings.environment}")
        except Exception as e:
            STORE.connection = {"connected": False,
                                "environment": STORE.settings.environment,
                                "detail": f"{e}"}
            STORE.log("error", "connection", f"Connection failed: {e}")
        return STORE.connection

    def _ensure_client(self) -> Optional[KalshiClient]:
        if self._client is None:
            self.connect()
        return self._client

    # --------------------------------------------------------------- a cycle
    def run_cycle(self) -> dict:
        s = STORE.settings
        if s.kill_switch_engaged:
            STORE.log("warn", "loop", "Kill switch engaged — cycle skipped")
            return {"skipped": "kill_switch"}

        client = self._ensure_client()
        if not client:
            STORE.log("error", "loop", "No client; cannot run cycle")
            return {"error": "no_client"}

        try:
            params = s.strategy_params
            market = market_data.build_market_view(client, s.ticker, s.timeframe, params)
            candles_full = market.pop("candles_full", market.get("candles", []))
            # Stash full candles so the fast-guard thread can recompute the
            # forager's dependent features (ecology / CSD / edge) at the live
            # mid every ~1s instead of using cycle-stale values.
            STORE.candles_full = candles_full

            # Mark-to-market the latest candle with the live quote BEFORE the
            # resilience / strategy layers consume it. The exchange candles
            # only close every 3m, but the orderbook logger pushes a fresh
            # mid every ~1-2s. Without this MTM, CSD risk / ATR / ecology
            # drivers see stale bar-close data — by the time the next bar
            # closes a fast move may already be over. With it, the in-flight
            # bar reflects current price, so all the resilience computations
            # are "as of now" within the cycle's tick.
            live_q = STORE.live_quote or {}
            live_mid = live_q.get("mid")
            if live_mid and candles_full:
                last = dict(candles_full[-1])
                last["close"] = float(live_mid)
                if last.get("high") is None or float(live_mid) > last["high"]:
                    last["high"] = float(live_mid)
                if last.get("low") is None or float(live_mid) < last["low"]:
                    last["low"] = float(live_mid)
                candles_full = candles_full[:-1] + [last]
                # Also refresh the market view's mirror so any downstream
                # consumer of market["candles"] sees the same MTM'd bar.
                if market.get("candles"):
                    mc = list(market["candles"])
                    mc[-1] = last
                    market["candles"] = mc
                market["price"] = float(live_mid)
                market["mtm_from_live_quote"] = True
            STORE.market = market

            account = account_mod.build_account_view(client, s.ticker, market.get("price"))
            _reconcile_position_entry(STORE, account, market.get("price"))
            STORE.account = account
            self._maybe_retrain(client, s, account)
            STORE.roll_day_if_needed(account.get("equity"))
            STORE.record_pnl(account.get("equity", 0.0),
                             0.0, account.get("unrealized_pnl", 0.0))
            STORE.add_snapshot({
                "ts": _now_iso(),
                "equity": account.get("equity"),
                "position": account.get("position_contracts"),
                "price": market.get("price"),
                "upnl": account.get("unrealized_pnl"),
            })

            # Ecology / Mycelial Alpha Network: always compute the
            # ecological phase (food-web nodes, keystone driver, organism
            # scores) so the UI can show it; only apply the phase size
            # multiplier to live sizing when settings.ecosystem_phase is on.
            from . import ecology as _ecology
            from . import multiasset as _multiasset
            try:
                multi = _multiasset.snapshot(s, client)
            except Exception as _e:
                STORE.log("warn", "ecology", f"multi-asset snapshot failed: {_e}")
                multi = {}
            STORE.multiasset = multi
            try:
                ecosystem = _ecology.classify(
                    candles_full, market, multi, params)
            except Exception as _e:
                STORE.log("warn", "ecology", f"ecology classify failed: {_e}")
                ecosystem = None
            STORE.ecosystem = ecosystem

            # CSD risk governor: compute current refined risk (abs_skew of
            # log-deviation, z-scored against rolling history). Validated
            # config in ablate_csd_refined.py: threshold 0.95 cuts max_dd
            # and lifts return on 3 of 4 walk-forward folds. Disabled by
            # default; enabled in aggressive/ultra presets.
            from . import csd as _csd
            try:
                closes_for_csd = [c["close"] for c in (candles_full or []) if c.get("close")]
                prev_csd_state = dict(STORE.csd_state or {})
                skew_hist = list(prev_csd_state.get("skew_history") or [])
                csd_risk_now, csd_skew_now = _csd.current_refined_risk(
                    closes_for_csd, skew_hist,
                    fv_period=int(getattr(s, "csd_governor_fv_period", 32) or 32),
                    window=int(getattr(s, "csd_governor_window", 96) or 96))
                skew_hist.append(csd_skew_now)
                if len(skew_hist) > 200:
                    skew_hist = skew_hist[-200:]
                # Adaptive threshold: drop from baseline to high_vol when ATR
                # crosses the breakpoint. This is what makes the governor
                # actually fire during fast regimes — the 0.95 baseline was
                # calibrated at ~0.08% ATR and is too lax at 0.2%+.
                atr_pct = float((market.get("features") or {}).get("atr_pct") or 0.0)
                base_thr = float(getattr(s, "csd_governor_threshold", 0.95) or 0.95)
                hi_thr = float(getattr(s, "csd_governor_threshold_high_vol", 0.80) or 0.80)
                atr_break = float(getattr(s, "csd_governor_atr_breakpoint", 0.20) or 0.20)
                eff_thr = hi_thr if atr_pct >= atr_break else base_thr
                from . import reflex as _reflex
                risk_dyn = _reflex.dynamics(prev_csd_state, "risk", csd_risk_now,
                                            threshold=eff_thr)
                skew_dyn = _reflex.dynamics(prev_csd_state, "skew", csd_skew_now)
                horizon = float(getattr(s, "csd_predictive_horizon_seconds", 12.0) or 12.0)
                impulse_gain = float(getattr(s, "csd_predictive_impulse_gain", 0.25) or 0.25)
                risk_velocity = max(0.0, float(risk_dyn.get("risk_velocity") or 0.0))
                risk_accel = max(0.0, float(risk_dyn.get("risk_acceleration") or 0.0))
                risk_impulse = max(0.0, float(risk_dyn.get("risk_impulse") or 0.0))
                risk_projected = max(
                    float(csd_risk_now),
                    float(csd_risk_now) + risk_velocity * horizon
                    + 0.5 * risk_accel * horizon * horizon
                    + impulse_gain * risk_impulse,
                )
                risk_projected = max(0.0, min(1.0, risk_projected))
                predictive_enabled = bool(getattr(s, "csd_predictive_enabled", True))
                risk_for_gate = risk_projected if predictive_enabled else float(csd_risk_now)
                ttt = risk_dyn.get("risk_time_to_threshold_s")
                STORE.csd_state = {
                    "risk": round(float(risk_for_gate), 4),
                    "risk_now": round(float(csd_risk_now), 4),
                    "risk_projected": round(float(risk_projected), 4),
                    "skew_now": round(float(csd_skew_now), 6),
                    "skew_history": skew_hist,
                    "threshold": round(eff_thr, 4),
                    "threshold_base": round(base_thr, 4),
                    "threshold_high_vol": round(hi_thr, 4),
                    "threshold_is_adaptive_high_vol": (atr_pct >= atr_break),
                    "predictive_enabled": predictive_enabled,
                    "predictive_horizon_s": round(horizon, 1),
                    "predictive_gate": bool(predictive_enabled and risk_for_gate > eff_thr),
                    "time_to_threshold_s": ttt,
                    **risk_dyn,
                    **skew_dyn,
                    "atr_pct": round(atr_pct, 4),
                    "enabled": bool(getattr(s, "csd_governor_enabled", False)),
                    "applied": False, "gated_at": None,
                }
            except Exception as _e:
                STORE.log("warn", "csd", f"csd risk compute failed: {_e}")

            # Visual review (chart -> Opus). Runs at most once per
            # settings.visual_review_interval_seconds, plus once on the very
            # first cycle after engine start (last_ts is None). Fires BEFORE
            # strategy.evaluate so the resulting STOP/CAUTION can gate the
            # current cycle's proposal, not just the next one's.
            from . import visual_review as _visual
            try:
                # Hand the visual layer a market dict that still includes the
                # full candle history (engine pops it earlier for STORE).
                mkt_for_review = dict(market)
                mkt_for_review["candles_full"] = candles_full
                _visual.maybe_run_review(
                    s, STORE, mkt_for_review, account,
                    list(STORE.decisions)[:20], client=client)
            except Exception as _e:
                STORE.log("warn", "visual", f"visual review failed: {_e}")

            # Build blend_context from rolling multiasset HISTORY. Each
            # component degrades gracefully when its required series is
            # missing — the historical Kalshi backtest validated the
            # spot-lead component; funding and OI are wired but default
            # weight=0 until they're separately backtested.
            blend_context = None
            if bool(getattr(s, "signal_blend_enabled", False)):
                # Inject blend weights / lookbacks into params so
                # signals.mpc_with_aux reads them. (Strategy_params is the
                # serialized dict that flows to signals.)
                params.setdefault("blend_enabled", True)
                for k in ("blend_w_mpc", "blend_w_spot_lead",
                          "blend_w_funding_fade", "blend_w_oi_pressure",
                          "blend_w_ecology_flow", "blend_w_visual",
                          "blend_spot_lookback", "blend_spot_history_for_std",
                          "blend_funding_history_for_std",
                          "blend_funding_persistence_threshold",
                          "blend_oi_lookback", "blend_oi_history_for_std",
                          "blend_ecology_lookback",
                          "blend_ecology_condition_spot_lead",
                          "blend_visual_conviction_ok",
                          "blend_visual_conviction_caution",
                          "blend_visual_conviction_stop"):
                    if hasattr(s, k):
                        params[k] = getattr(s, k)
                params["blend_enabled"] = True
                try:
                    from . import multiasset as _ma
                    spot_series = _ma.HISTORY.series("btc_spot", n=240)
                    spot_closes = [v for _, v in spot_series]
                    funding_series = _ma.HISTORY.series("binance_funding", n=240)
                    funding_history = [v for _, v in funding_series]
                    oi_series = _ma.HISTORY.series("hl_oi", n=240)
                    oi_history = [v for _, v in oi_series]
                    # Build a price series aligned to oi_history's cadence
                    # by pulling Kalshi closes at the same length. (Both
                    # series push once per cycle, so they share the
                    # cycle index.)
                    closes_for_oi = [c["close"] for c in
                                       (candles_full or [])][-len(oi_history):] \
                                       if oi_history else []
                    blend_context = {
                        "spot_closes": spot_closes,
                        "funding_history": funding_history,
                        "oi_history": oi_history,
                        "oi_price_history": closes_for_oi,
                        "ecosystem": ecosystem,
                    }
                    # Visual-review trend as a directional contributor. Only
                    # attach it when the directional use is enabled AND the
                    # latest review is fresh — the review refreshes every
                    # visual_review_interval_seconds but the loop runs more
                    # often, so a stale/failed review must not keep tilting.
                    if getattr(s, "visual_review_as_signal", False):
                        vr = STORE.visual_review or {}
                        last_ts = vr.get("last_ts")
                        max_age = float(getattr(
                            s, "visual_signal_max_age_seconds", 1800) or 1800)
                        if (vr.get("trend") and last_ts is not None
                                and (time.time() - float(last_ts)) <= max_age):
                            blend_context["visual_review"] = vr
                except Exception as _e:
                    STORE.log("warn", "blend",
                               f"blend_context build failed: {_e}")

            live_current_fraction = risk.current_mpc_fraction(s, account, market)
            proposal = strategy.evaluate(
                candles_full, params, account.get("position_contracts", 0.0),
                variant=s.strategy, market=market,
                ecosystem=ecosystem,
                ecosystem_apply=getattr(s, "ecosystem_phase", False),
                csd_state=STORE.csd_state,
                blend_context=blend_context,
                live_current_fraction=live_current_fraction)
            proposal["live_current_fraction"] = round(float(live_current_fraction), 6)

            # Surface the blended-alpha decomposition for the UI diagram. Same
            # math as signals.mpc_with_aux but exposed piece-by-piece. Failure
            # here is silent — the UI just won't update its alpha-decomp panel.
            try:
                from . import signals_blended as _sb
                STORE.blend_state = _sb.latest_alpha_decomposition(
                    candles_full, blend_context, params)
                if isinstance(STORE.blend_state, dict) and proposal.get("controller"):
                    controller = proposal.get("controller") or {}
                    STORE.blend_state["controller"] = controller
                    if controller.get("mpc_alpha") is not None:
                        raw = dict(STORE.blend_state.get("raw") or {})
                        parts = dict(STORE.blend_state.get("parts") or {})
                        weights = dict(STORE.blend_state.get("weights") or {})
                        raw["mpc"] = float(controller.get("mpc_alpha") or 0.0)
                        parts["mpc"] = round(
                            float(weights.get("mpc", 1.0) or 0.0) * raw["mpc"], 6)
                        STORE.blend_state["raw"] = raw
                        STORE.blend_state["parts"] = parts
                        STORE.blend_state["blended"] = round(
                            sum(float(v or 0.0) for v in parts.values()), 6)
                        diag = dict(STORE.blend_state.get("diagnostics") or {})
                        diag["controller_aligned"] = True
                        STORE.blend_state["diagnostics"] = diag
            except Exception as _e:
                STORE.log("warn", "blend",
                           f"blend decomposition failed: {_e}")
            # If the strategy gated the position, stamp the CSD state so the
            # UI / decision log shows the gate fired this cycle.
            if proposal.get("csd_gated"):
                STORE.csd_state["applied"] = True
                STORE.csd_state["gated_at"] = _now_iso()

            # Visual review STOP gate: if Opus said the bot is doing something
            # visibly stupid, block NEW entries (closes/reduces still allowed).
            if _visual.block_entry_if_stop(STORE, proposal, s):
                STORE.log("warn", "visual",
                           f"Entry blocked by visual review STOP: "
                           f"{(STORE.visual_review or {}).get('note', '')}",
                           event="visual_stop_block",
                           prior_action=proposal.get("visual_review_blocked_action"))

            # Tactical adverse-move reversal. MPC's target only updates on
            # bar close, so a sharp intra-bar jump against our position can
            # sit unanswered for up to a full bar. When we hold a position
            # and current price has moved against it by >= adverse_move_bps
            # (measured vs the prior bar's close), flip to a full-conviction
            # opposite-side position so we capitalize on the shift instead of
            # just dodging it. Sets force_taker on the proposal so it crosses
            # immediately regardless of preset.
            _apply_adverse_move_override(s, account, market, candles_full, proposal)

            # Autotomy Agent: ecological loss-shedding reflex. Symmetric to
            # the forager but for toxic losing positions: hard-exit only when
            # loss is paired with predator/cascade/CSD/reserve-collapse
            # confirmations. Diagnostics land on STORE.autotomy_state even
            # when disabled. See backend/autotomy.py.
            from . import autotomy as _autotomy
            try:
                astate = _autotomy.apply(s, STORE, account, market, proposal)
                if astate.get("ejected"):
                    STORE.log("warn", "autotomy",
                              f"Eject: {astate.get('eject_reason')} "
                              f"(cooldown {astate.get('cooldown_seconds')}s)",
                              event="autotomy_eject",
                              loss_R=astate.get("loss_R"),
                              pressure=astate.get("autotomy_pressure"))
                elif astate.get("blocked_entry"):
                    STORE.log("info", "autotomy",
                              f"Refractory blocked {astate.get('autotomy_blocked_action') or proposal.get('autotomy_blocked_action')} "
                              f"- loss cooldown ({astate.get('cooldown_remaining_s')}s left)",
                              event="autotomy_cooldown_block")
            except Exception as _e:
                STORE.log("warn", "autotomy", f"autotomy layer failed: {_e}")

            # Foraging cycle: ecological profit-harvest + refractory cooldown.
            # Runs as the FINAL exit/cooldown overlay before the decision
            # engine (which only reduces/blocks, never re-inflates). Trims/
            # closes when profit appears AND the edge is consumed; blocks NEW
            # entries while resting. Diagnostics land on STORE.forager_state
            # even when disabled. See backend/forager.py.
            from . import forager as _forager
            try:
                fstate = _forager.apply(s, STORE, account, market, proposal)
                if fstate.get("harvested"):
                    STORE.log("info", "forager",
                               f"Harvest: {fstate.get('harvest_reason')} "
                               f"(scale {fstate.get('harvest_scale')}, "
                               f"cooldown {fstate.get('cooldown_seconds')}s)",
                               event="forager_harvest",
                               pnl_R=fstate.get("pnl_R"))
                elif fstate.get("blocked_entry"):
                    STORE.log("info", "forager",
                               f"Refractory blocked {fstate.get('forager_blocked_action') or proposal.get('forager_blocked_action')} "
                               f"— resting ({fstate.get('cooldown_remaining_s')}s left)",
                               event="forager_cooldown_block")
            except Exception as _e:
                STORE.log("warn", "forager", f"forager layer failed: {_e}")

            decision = decision_engine.decide(
                s, account, market, proposal, log=STORE.log)

            sizing = risk.size_position(s, account, market, proposal)
            effective_action = risk.effective_action_from_sizing(account, sizing)
            sizing["effective_action"] = effective_action
            sizing["risk_increasing"] = risk.increases_risk(account, sizing)
            proposal["sized_action"] = effective_action
            if effective_action != decision.get("action"):
                original_action = decision.get("action", "HOLD")
                decision["action"] = effective_action
                decision["verb"] = decision_engine.ACTION_TO_VERB.get(
                    effective_action, effective_action)
                note = (f"Sized action corrected from {original_action} to "
                        f"{effective_action} after target/current comparison.")
                if effective_action in ("REDUCE", "CLOSE"):
                    reasons = list(decision.get("reasons_for") or [])
                    reasons.append(note)
                    decision["reasons_for"] = reasons
                else:
                    reasons = list(decision.get("reasons_against") or [])
                    reasons.append(note)
                    decision["reasons_against"] = reasons
            checks = risk.run_checks(
                s, account, market, sizing,
                STORE.day_start_equity, STORE.max_equity, proposal=proposal)

            decision_record = {
                "ts": _now_iso(),
                "verb": decision["verb"],
                "action": decision["action"],
                "confidence": decision["confidence"],
                "expected_edge_pct": decision["expected_edge_pct"],
                "urgency": decision.get("urgency", 0.0),
                "urgency_base": decision.get("urgency_base"),
                "taker_now": decision.get("taker_now", False),
                "force_taker": decision.get("force_taker", False),
                "urgency_override": decision.get("urgency_override"),
                "expected_profit_usd": decision.get("expected_profit_usd"),
                "expected_risk_usd": decision.get("expected_risk_usd"),
                "reasons_for": decision.get("reasons_for", []),
                "reasons_against": decision.get("reasons_against", []),
                "why_better_than_hold": decision.get("why_better_than_hold", ""),
                "source": decision.get("source"),
                "model_note": decision.get("model_note"),
                "proposal_action": proposal.get("action"),
                "proposal_target_fraction": proposal.get("target_fraction"),
                "proposal_target_fraction_base": proposal.get("target_fraction_base"),
                "proposal_rationale_for": proposal.get("rationale_for", []),
                "proposal_rationale_against": proposal.get("rationale_against", []),
                "ecosystem": decision.get("ecosystem"),
                "ecosystem_applied": decision.get("ecosystem_applied", False),
                "ecology_note": decision.get("ecology_note", ""),
                "sizing": sizing,
                "checks": checks,
                "proposal": proposal,
                "missed_opportunity": _missed_opportunity(proposal, checks, decision),
            }

            # Execute if allowed and not blocked.
            armed = s.live_autonomous_armed and s.allow_live_orders and s.auto_submit_orders
            if not armed:
                decision_record["execution"] = {"submitted": False,
                                                 "note": "not armed for live orders"}
            elif not checks["allow"]:
                decision_record["execution"] = {"submitted": False,
                                                 "note": "blocked by risk",
                                                 "blocks": checks["blocks"]}
                STORE.log("warn", "risk",
                          "Trade blocked: " + "; ".join(checks["blocks"]))
            else:
                ex = executor.execute(client, s, decision, sizing, market, account, STORE.log)
                decision_record["execution"] = ex
                for o in ex.get("orders", []):
                    STORE.add_order(o)

            STORE.add_decision(decision_record)
            STORE.loop_status["cycles"] += 1
            STORE.loop_status["last_cycle_ts"] = _now_iso()
            STORE.loop_status["last_error"] = None
            # Hand the fast guard the price we just acted on so it can
            # measure intra-cycle drift from that anchor.
            try:
                from .fast_guard import FAST_GUARD
                FAST_GUARD.cycle_completed(market.get("price"))
            except Exception:
                pass
            return decision_record
        except Exception as e:
            STORE.loop_status["last_error"] = str(e)
            STORE.log("error", "loop", f"Cycle error: {e}")
            return {"error": str(e)}

    def _maybe_retrain(self, client, s, account) -> None:
        """Auto-refresh strategy params when the live edge degrades (drawdown
        from peak) or on a schedule. The tuner's OOS gate prevents bad params
        from going live, so this can only help, never silently harm."""
        if not getattr(s, "auto_retrain", False):
            return
        from .tuning import TUNER
        eq = account.get("equity")
        if eq:
            self._peak_eq = max(self._peak_eq or eq, eq)
        self._cycles_since_retrain += 1
        dd = ((self._peak_eq - eq) / self._peak_eq * 100) if (self._peak_eq and eq) else 0.0
        trigger = None
        if dd >= s.retrain_drawdown_pct:
            trigger = f"live drawdown {dd:.1f}% >= {s.retrain_drawdown_pct:.0f}%"
        elif self._cycles_since_retrain >= s.retrain_every_cycles:
            trigger = "scheduled refresh"
        if trigger and not TUNER.is_running():
            try:
                candles = market_data.fetch_candles(
                    client, s.ticker, s.timeframe, lookback_bars=800)
                TUNER.start(candles, iterations=80, auto_apply=True,
                            leverage=s.leverage_target, variant=s.strategy,
                            fee_pct=s.assumed_fee_bps / 1e4)
                self._cycles_since_retrain = 0
                STORE.log("warn", "retrain",
                          f"Edge health: {trigger} -> auto-retrain started "
                          f"(applies only if out-of-sample positive)")
            except Exception as e:
                STORE.log("warn", "retrain", f"auto-retrain could not start: {e}")

    # ----------------------------------------------------------------- loop
    def start_loop(self, duration_minutes: Optional[float] = None) -> dict:
        with self._lock:
            if self._thread and self._thread.is_alive():
                return {"running": True, "note": "already running"}
            self._stop.clear()
            self._peak_eq = None
            self._cycles_since_retrain = 0
            self._deadline = (time.time() + duration_minutes * 60
                              if duration_minutes else None)
            self._thread = threading.Thread(target=self._loop, daemon=True)
            self._thread.start()
            # Spin up the fast guard alongside the main loop so it polls
            # live_quote between cycles and can trigger emergency runs.
            try:
                from .fast_guard import FAST_GUARD
                FAST_GUARD.start(
                    settings_provider=lambda: STORE.settings,
                    cycle_runner=self.run_cycle)
            except Exception as e:
                STORE.log("warn", "fast_guard", f"start failed: {e}")
            note = (f"auto-stops in {duration_minutes:.0f} min"
                    if duration_minutes else "runs until stopped")
            STORE.loop_status.update({"running": True, "mode": "autonomous-live",
                                      "deadline_ts": _iso_at(self._deadline)})
            STORE.log("info", "loop", f"Autonomous live trading loop STARTED ({note})")
            return {"running": True, "deadline": _iso_at(self._deadline)}

    def _loop(self):
        while not self._stop.is_set():
            if STORE.settings.kill_switch_engaged:
                STORE.log("warn", "loop", "Kill switch — loop pausing")
                break
            if self._deadline and time.time() >= self._deadline:
                STORE.log("info", "loop", "Loop deadline reached — auto-stopping")
                break
            self.run_cycle()
            # Vol-aware cadence: shorter intervals when ATR% is hot, longer
            # when quiet. Read the ATR% from the cycle's market view.
            atr_pct = ((STORE.market or {}).get("features", {}) or {}).get("atr_pct")
            interval = compute_loop_interval(STORE.settings, atr_pct)
            STORE.loop_status["interval_seconds"] = interval
            STORE.loop_status["atr_pct"] = atr_pct
            STORE.loop_status["next_cycle_ts"] = _iso_in(interval)
            # sleep in small slices (max 1s each) so Stop stays responsive.
            # For sub-second intervals (interval=1) we sleep exactly that.
            slept = 0.0
            slice_s = 1.0 if interval >= 1 else float(interval)
            while slept < interval:
                if self._stop.is_set() or STORE.settings.kill_switch_engaged:
                    break
                if self._deadline and time.time() >= self._deadline:
                    break
                step = min(slice_s, interval - slept)
                time.sleep(step)
                slept += step
        STORE.loop_status.update({"running": False, "mode": "idle",
                                  "next_cycle_ts": None})
        STORE.log("info", "loop", "Autonomous loop STOPPED")

    def stop_loop(self) -> dict:
        self._stop.set()
        # Tear down the fast guard too. It only makes sense while the main
        # loop is running — there's no scheduled cycle to short-circuit.
        try:
            from .fast_guard import FAST_GUARD
            FAST_GUARD.stop()
        except Exception:
            pass
        STORE.loop_status.update({"running": False, "mode": "idle"})
        STORE.log("info", "loop", "Stop requested")
        return {"running": False}

    def kill(self) -> dict:
        STORE.settings.kill_switch_engaged = True
        STORE.settings.allow_live_orders = False
        STORE.settings.live_autonomous_armed = False
        STORE.save_settings()
        self._stop.set()
        STORE.loop_status.update({"running": False, "mode": "killed"})
        # best-effort: cancel resting orders AND flatten any open position
        try:
            client = self._ensure_client()
            if client:
                executor.cancel_all_resting(client, STORE.settings.ticker, STORE.log)
                executor.flatten_position(client, STORE.settings.ticker, STORE.log)
        except Exception as e:
            STORE.log("warn", "kill", f"Could not fully flatten on kill: {e}")
        STORE.log("warn", "kill", "KILL SWITCH ENGAGED — trading halted")
        return {"killed": True}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _iso_in(seconds: int) -> str:
    from datetime import timedelta
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _iso_at(ts: Optional[float]) -> Optional[str]:
    if not ts:
        return None
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


_ADVERSE_FLIP_STATE = {"last_cycle": -10_000, "last_ts": 0.0}


def _reconcile_position_entry(store, account: dict, price) -> None:
    """Fill in unrealized_pnl when the /margin/positions endpoint reports 0 on a
    real position (it lags after fills / when the position is balance-inferred,
    which otherwise pegs the forager's pnl_R at 0 forever). Track an entry price
    across cycles: trust the broker's when nonzero, else snapshot the live price
    when a position first appears or flips side, then estimate
    unrealized_pnl = size * (price - entry)."""
    try:
        pos = float(account.get("position_contracts") or 0.0)
        px = float(price or 0.0)
        if abs(pos) < 1e-9:
            store.position_entry = None
            return
        sign = 1 if pos > 0 else -1
        broker_entry = float(account.get("entry_price") or 0.0)
        broker_upnl = float(account.get("unrealized_pnl") or 0.0)
        pe = store.position_entry
        if broker_entry > 0:
            store.position_entry = {"entry": broker_entry, "sign": sign}
        elif (not pe or pe.get("sign") != sign) and px > 0:
            # first observation of this position / a side flip — best estimate
            store.position_entry = {"entry": px, "sign": sign}
        entry = (store.position_entry or {}).get("entry")
        if abs(broker_upnl) < 1e-9 and entry and px > 0:
            account["unrealized_pnl"] = round(pos * (px - entry), 4)
            account["unrealized_pnl_estimated"] = True
            if broker_entry <= 0:
                account["entry_price"] = round(entry, 6)
    except Exception:
        pass


def _apply_adverse_move_override(settings, account: dict, market: dict,
                                 candles_full: list, proposal: dict) -> None:
    """Mutates `proposal` in place: when we hold a position and price has
    jumped against it past the configured bps threshold, flip the proposal
    to a full-conviction REVERSE on the opposite side and mark force_taker
    so the executor crosses immediately.

    Guards:
      - threshold > 0 (off by default)
      - we actually hold a position
      - proposal isn't already flipping/exiting
      - cooldown: didn't fire within the last N cycles (prevents back-to-
        back flips while a single move drifts back and forth across the
        threshold — the original implementation could fire every cycle)
    """
    thresh = float(getattr(settings, "adverse_move_bps", 0.0) or 0.0)
    if thresh <= 0.0:
        return
    pos = float(account.get("position_contracts") or 0.0)
    if abs(pos) < 1.0:
        return
    if proposal.get("action") in ("REVERSE_LONG", "REVERSE_SHORT", "CLOSE"):
        return   # already flipping/exiting; don't double-act

    cooldown = int(getattr(settings, "adverse_move_cooldown_cycles", 5) or 0)
    if cooldown > 0:
        cur_cycle = int(STORE.loop_status.get("cycles", 0) or 0)
        last_cycle = _ADVERSE_FLIP_STATE["last_cycle"]
        if cur_cycle - last_cycle < cooldown:
            return   # cooldown window — don't re-fire
    closes = [c.get("close") for c in (candles_full or [])
              if c.get("close")]
    if len(closes) < 2:
        return
    current = float(market.get("price") or closes[-1])
    ref = float(closes[-2])
    if current <= 0 or ref <= 0:
        return
    move_bps = (current - ref) / current * 10_000.0
    adverse_bps = -move_bps if pos > 0 else move_bps
    if adverse_bps < thresh:
        return
    direction = "LONG" if pos > 0 else "SHORT"
    new_action = "REVERSE_SHORT" if pos > 0 else "REVERSE_LONG"
    # MPC sizing reads target_fraction directly. Flip it to a full-conviction
    # opposite-side target so the controller's continuous sizer builds the
    # mirrored position (still clamped by max_leverage / max_notional).
    new_fraction = -1.0 if pos > 0 else 1.0
    STORE.log("warn", "decision",
              f"Adverse-move flip: {direction} hit by {adverse_bps:.1f}bps "
              f"(>= {thresh:.1f}bps) -> {new_action} (force taker)",
              event="adverse_reverse", direction=direction,
              adverse_bps=round(adverse_bps, 2), threshold_bps=thresh,
              prior_action=proposal.get("action"), new_action=new_action)
    proposal["action"] = new_action
    proposal["target_fraction"] = new_fraction
    proposal["desired_direction"] = "SHORT" if pos > 0 else "LONG"
    proposal["blended_score"] = new_fraction
    proposal["confidence"] = 1.0
    # Boost urgency above any plausible taker_threshold_k so the executor
    # crosses even when the verb (LONG/SHORT) doesn't itself bypass maker.
    proposal["urgency"] = max(float(proposal.get("urgency") or 0.0), 1.0)
    proposal["force_taker"] = True
    proposal["adverse_move_bps"] = round(adverse_bps, 2)
    # Stamp the cooldown so this can't re-fire next cycle while the move
    # drifts back across the threshold.
    _ADVERSE_FLIP_STATE["last_cycle"] = int(STORE.loop_status.get("cycles", 0) or 0)
    _ADVERSE_FLIP_STATE["last_ts"] = time.time()
    proposal.setdefault("rationale_for", []).insert(
        0, f"Tactical reversal: {adverse_bps:.1f}bps adverse move vs prev "
        f"bar close — flip to capitalize on the shift.")


def _missed_opportunity(proposal: dict, checks: dict, decision: dict) -> Optional[str]:
    if decision["verb"] == "HOLD" and proposal.get("confidence", 0) > 0.55:
        return (f"Strong signal (confidence {proposal['confidence']:.2f}) but action "
                f"resolved to HOLD — review thresholds.")
    if not checks["allow"] and proposal.get("confidence", 0) > 0.5:
        return ("Profitable signal blocked by a hard risk guardrail: "
                + "; ".join(checks["blocks"]))
    return None


ENGINE = Engine()
