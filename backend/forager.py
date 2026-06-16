"""Foraging-cycle layer: ecological profit-harvest + refractory (cooldown).

Models the bot as a forager that works an edge, HARVESTS the profit when the
ecological edge is consumed (not when a clock runs out), then RESTS until the
ecosystem resets or a new disturbance appears:

  1. Search        — (handled by the strategy/blend: finds the niche)
  2. Enter niche    — (strategy proposes a position)
  3. Harvest edge   — this layer: trim/close as profit appears AND edge decays
  4. Exit on decay  — full close when the edge is gone or CSD risk spikes
  5. Rest           — refractory cooldown: exits allowed, NEW entries blocked
  6. Re-enter       — cooldown ends on a new disturbance / MR-permission rebuild
                      / price back to the fair-value reset zone / time elapsed

Why: a common failure is "good trade -> profit -> bot keeps holding -> edge
decays -> churn reverses -> profit evaporates". Mean-reversion profit is often
temporary, so we harvest, flatten, and wait. The cooldown is ECOLOGICAL, not a
fixed timer — long when the edge has clearly been consumed (reserve recovered,
decomposer high, low disturbance), short/early-ended when a fresh disturbance
or scavenger (snap-back) signal returns.

Signal mapping (spec -> what this codebase actually exposes):
  pnl_R              -> unrealized_pnl / (position_notional * ATR%)  [profit in
                        per-bar-ATR-of-notional units; tunable thresholds]
  expected_edge      -> proposal["expected_edge_pct"]  (MPC: |target| * 0.6 —
                        decays to 0 as price reverts to fair value)
  edge_decay         -> 1 - edge/edge_ref, clamped [0,1]
  S_decomposer       -> ecosystem.organisms.scores.decomposer
  scavenger_score    -> ecosystem.organisms.scores.scavenger
  disturbance_score  -> ecosystem.drivers.disturbance  (||vol,spread,stretch,liq||/4)
  relative reserve   -> ecosystem.network_metrics.rel_reserve (Ulanowicz R/C)
  reserve_recovery   -> rel_reserve level; reserve_rising -> positive delta
  CSD risk (rising)  -> store.csd_state.risk vs the prior cycle

HarvestPressure (diagnostic scalar, exposed for the UI; the actual harvest
decision uses the explicit tiered rules below for clarity/safety):
  HP = realized_profit_score + decomposer + reserve_recovery + edge_decay
       - disturbance_score - scavenger_score

DEFAULT OFF (settings.forager_enabled). When off, apply() only computes
diagnostics into STORE.forager_state and never touches the proposal. This is
an UNBACKTESTED heuristic layer — enable it deliberately and watch it live.

Public API:
  apply(settings, store, account, market, proposal) -> dict
    Computes signals, manages the cooldown state machine, and (when enabled)
    mutates the proposal: trims/closes on harvest, blocks NEW entries during
    the refractory. Returns the diagnostic state (also stored on
    STORE.forager_state). Runs as the FINAL exit/cooldown overlay, just before
    the decision engine (which can only reduce/block, so it won't re-inflate a
    harvested position).
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

# New entries the refractory blocks. Reduces/closes are NEVER blocked — an
# exit must always be allowed (matches the visual-review gate's contract).
ENTRY_ACTIONS = {"ENTER_LONG", "ENTER_SHORT", "REVERSE_LONG", "REVERSE_SHORT", "ADD"}

# "How hungry" the forager is — one selectable knob controlling BOTH:
#   WHEN it eats — the profit/edge/churn thresholds (lower = harvests sooner),
#   HOW MUCH it eats — the per-tier bite sizes / "bounds for eating" below, and
#   HOW LONG it rests — the cooldown bounds.
# The *_scale fields are the fraction of the position KEPT after a harvest
# (so 0.0 = full close / devour, 0.85 = nibble 15% off / keep a runner). A
# lazy forager takes small bites and never fully closes on profit; a ravenous
# one flattens. Higher hunger = lower thresholds AND smaller scales AND shorter
# cooldown. Default RAVENOUS (eats fast and deep, ~5-10 min rest). Each preset
# fully defines behaviour; the raw forager_* settings are ignored.
FORAGER_HUNGER_PRESETS = {
    "lazy": {  # nibbles, always keeps a runner
        "profit_threshold_R": 0.6, "edge_decay_trim": 0.6, "tier2_R": 1.0,
        "tier3_R": 1.5, "decomposer": 0.6, "edge_gone": 0.05, "csd_high": 0.85,
        "churn_disturbance": 0.7, "churn_profit_R": 0.4,
        "tier1_scale": 0.85, "tier2_scale": 0.65, "tier3_scale": 0.40, "churn_scale": 0.50,
        "base_cooldown": 600, "cd_min": 240, "cd_max": 1800,
    },
    "steady": {
        "profit_threshold_R": 0.4, "edge_decay_trim": 0.5, "tier2_R": 0.7,
        "tier3_R": 1.0, "decomposer": 0.55, "edge_gone": 0.08, "csd_high": 0.8,
        "churn_disturbance": 0.6, "churn_profit_R": 0.3,
        "tier1_scale": 0.80, "tier2_scale": 0.55, "tier3_scale": 0.25, "churn_scale": 0.40,
        "base_cooldown": 420, "cd_min": 180, "cd_max": 1200,
    },
    "hungry": {
        "profit_threshold_R": 0.25, "edge_decay_trim": 0.4, "tier2_R": 0.5,
        "tier3_R": 0.7, "decomposer": 0.5, "edge_gone": 0.10, "csd_high": 0.7,
        "churn_disturbance": 0.5, "churn_profit_R": 0.2,
        "tier1_scale": 0.75, "tier2_scale": 0.40, "tier3_scale": 0.10, "churn_scale": 0.25,
        "base_cooldown": 300, "cd_min": 120, "cd_max": 720,
    },
    "ravenous": {  # devours — flattens on edge-gone and in churn
        "profit_threshold_R": 0.15, "edge_decay_trim": 0.3, "tier2_R": 0.3,
        "tier3_R": 0.5, "decomposer": 0.4, "edge_gone": 0.15, "csd_high": 0.6,
        "churn_disturbance": 0.4, "churn_profit_R": 0.1,
        "tier1_scale": 0.60, "tier2_scale": 0.25, "tier3_scale": 0.00, "churn_scale": 0.00,
        "base_cooldown": 240, "cd_min": 90, "cd_max": 600,
    },
}
DEFAULT_HUNGER = "ravenous"


def _resolve_params(settings) -> dict:
    """Resolve the active harvest/cooldown params from the hunger preset."""
    hunger = (getattr(settings, "forager_hunger", DEFAULT_HUNGER) or DEFAULT_HUNGER).lower()
    preset = FORAGER_HUNGER_PRESETS.get(hunger, FORAGER_HUNGER_PRESETS[DEFAULT_HUNGER])
    return dict(preset, hunger=hunger)


def _autoscale(P: dict, sig: dict, settings) -> dict:
    """Autoscale the hunger preset by the live regime / network-dynamic state.

    A brittle, disturbed, critically-slowing market — high disturbance, high CSD
    risk, organized/locked information graph (rel_ascendancy), and low adaptive
    reserve — makes the forager EAGER: it drops its profit bars, takes bigger
    bites, and rests for less time so it banks profit before a cascade unwinds
    it. A calm, diffuse, recovering market (low disturbance, high reserve) makes
    it PATIENT: higher bars, smaller bites, longer rest. Returns a copy of P
    with an added `eagerness` ∈ [0,1]. Toggle with forager_autoscale."""
    eager = 0.0
    if bool(getattr(settings, "forager_autoscale", True)):
        eager = _clip(
            0.45 * sig["disturbance_score"]
            + 0.30 * min(1.0, sig["csd_risk"])
            + 0.15 * sig.get("rel_ascendancy", 0.0)
            + 0.10 * (1.0 - sig["reserve_recovery"]),
            0.0, 1.0)
    Q = dict(P)
    Q["eagerness"] = round(eager, 3)
    if eager <= 0.0:
        return Q
    thr_mult = 1.0 - 0.5 * eager      # eat sooner (lower profit/edge bars)
    scale_mult = 1.0 - 0.5 * eager    # bigger bite (smaller "kept" scale)
    cd_mult = 1.0 - 0.6 * eager       # shorter rest
    for k in ("profit_threshold_R", "tier2_R", "tier3_R", "churn_profit_R",
              "edge_decay_trim", "churn_disturbance"):
        Q[k] = P[k] * thr_mult
    for k in ("tier1_scale", "tier2_scale", "tier3_scale", "churn_scale"):
        Q[k] = _clip(P[k] * scale_mult, 0.0, 1.0)
    for k in ("base_cooldown", "cd_min", "cd_max"):
        Q[k] = P[k] * cd_mult
    return Q


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _current_fraction(settings, account: dict, market: dict) -> float:
    """Signed current position as a fraction of max buying power (long>0,
    short<0) — same normalization the decision engine uses for trade sizing."""
    equity = float(account.get("equity") or 0.0)
    lev = float(getattr(settings, "leverage_target", 5.8) or 5.8)
    max_notional = float(getattr(settings, "max_position_notional_usd", 0.0) or 0.0)
    if max_notional <= 0:
        max_notional = equity * lev
    price = float((market or {}).get("price") or 0.0)
    pos = float(account.get("position_contracts") or 0.0)
    notional = abs(pos) * price
    frac = (notional / max_notional) if max_notional > 0 else 0.0
    return -frac if pos < 0 else frac


def _signals(settings, store, account: dict, market: dict, proposal: dict,
             prev: dict) -> dict:
    """Derive the ecological/profit scalars for this cycle."""
    pos = float(account.get("position_contracts") or 0.0)
    price = float((market or {}).get("price") or 0.0)
    upnl = float(account.get("unrealized_pnl") or 0.0)
    atr_pct = float(((market or {}).get("features") or {}).get("atr_pct") or 0.0)  # percent

    # pnl_R: profit normalized by one ATR of move on the current notional.
    notional = abs(pos) * price
    risk_usd = notional * (atr_pct / 100.0)
    pnl_R = (upnl / risk_usd) if risk_usd > 1e-9 else 0.0

    # Expected edge proxy: the strategy's own expected_edge_pct (decays toward 0
    # as price reverts to fair value and |target_fraction| shrinks).
    edge = float(proposal.get("expected_edge_pct") or 0.0)
    edge_ref = float(getattr(settings, "forager_edge_ref", 0.30) or 0.30)
    edge_decay = _clip(1.0 - (edge / edge_ref), 0.0, 1.0) if edge_ref > 0 else 0.0

    # Ecology: prefer the proposal's attached snapshot, else STORE's latest.
    eco = proposal.get("ecosystem") or (store.ecosystem or {})
    orgs = ((eco.get("organisms") or {}).get("scores") or {})
    decomposer = float(orgs.get("decomposer") or 0.0)
    scavenger = float(orgs.get("scavenger") or 0.0)
    drivers = eco.get("drivers") or {}
    disturbance = float(drivers.get("disturbance") or 0.0)
    disturbance_score = _clip(disturbance, 0.0, 1.0)   # already ~/4, clamp tail
    stretch_abs = abs(float(drivers.get("stretch_z") or 0.0))
    netm = eco.get("network_metrics") or {}
    reserve = float(netm.get("rel_reserve") or 0.0)
    rel_ascendancy = float(netm.get("rel_ascendancy") or 0.0)
    reserve_recovery = _clip(reserve, 0.0, 1.0)
    reserve_prev = float(prev.get("reserve_prev", reserve))
    reserve_rising = (reserve - reserve_prev) > 0.01

    csd = store.csd_state or {}
    csd_risk = float(csd.get("risk") or 0.0)
    csd_prev = float(prev.get("csd_prev", csd_risk))
    csd_rising = (csd_risk - csd_prev) > 0.01

    pnl_R_ref = float(getattr(settings, "forager_pnl_R_ref", 1.0) or 1.0)
    realized_profit_score = _clip(pnl_R / pnl_R_ref, 0.0, 1.0)

    harvest_pressure = round(
        realized_profit_score + decomposer + reserve_recovery + edge_decay
        - disturbance_score - scavenger, 3)

    from . import reflex as _reflex
    hard_pressure = float(getattr(settings, "forager_reflex_pressure_threshold", 1.25) or 1.25)
    dyn = {}
    dyn.update(_reflex.dynamics(prev, "harvest_pressure", harvest_pressure,
                                threshold=hard_pressure))
    dyn.update(_reflex.dynamics(prev, "edge_decay", edge_decay))
    dyn.update(_reflex.dynamics(prev, "decomposer", decomposer))
    dyn.update(_reflex.dynamics(prev, "scavenger", scavenger))

    horizon = float(getattr(settings, "forager_predictive_horizon_seconds", 12.0) or 12.0)
    impulse_gain = float(getattr(settings, "forager_predictive_impulse_gain", 0.35) or 0.35)
    hp_velocity = float(dyn.get("harvest_pressure_velocity") or 0.0)
    hp_accel = float(dyn.get("harvest_pressure_acceleration") or 0.0)
    hp_impulse = float(dyn.get("harvest_pressure_impulse") or 0.0)
    harvest_pressure_projected = harvest_pressure
    if bool(getattr(settings, "forager_predictive_enabled", True)):
        harvest_pressure_projected = (
            harvest_pressure
            + max(0.0, hp_velocity) * horizon
            + 0.5 * max(0.0, hp_accel) * horizon * horizon
            + impulse_gain * max(0.0, hp_impulse)
        )
    predictive_harvest = harvest_pressure_projected >= hard_pressure

    out = {
        "pnl_R": round(pnl_R, 3),
        "edge": round(edge, 4),
        "edge_decay": round(edge_decay, 3),
        "decomposer": round(decomposer, 3),
        "scavenger": round(scavenger, 3),
        "disturbance_score": round(disturbance_score, 3),
        "stretch_abs": round(stretch_abs, 3),
        "reserve": round(reserve, 4),
        "reserve_recovery": round(reserve_recovery, 3),
        "reserve_rising": reserve_rising,
        "rel_ascendancy": round(rel_ascendancy, 4),
        "csd_risk": round(csd_risk, 4),
        "csd_rising": csd_rising,
        "realized_profit_score": round(realized_profit_score, 3),
        "harvest_pressure": harvest_pressure,
        "harvest_pressure_projected": round(harvest_pressure_projected, 3),
        "predictive_harvest": bool(predictive_harvest),
    }
    out.update(dyn)
    return out


def _cooldown_seconds(P: dict, sig: dict) -> float:
    """Ecological (not clock) cooldown length:
        base * (1 + reserve_recovery) * (1 + decomposer) * (1 - disturbance)
    Long when the edge has clearly been consumed (reserve recovered, decomposer
    high, calm); short when there's still disturbance around. Clamped to the
    hunger preset's [cd_min, cd_max]."""
    base = float(P["base_cooldown"])
    cd = base * (1.0 + sig["reserve_recovery"]) * (1.0 + sig["decomposer"]) \
        * (1.0 - sig["disturbance_score"])
    return _clip(cd, float(P["cd_min"]), float(P["cd_max"]))


def live_signals(settings, store, live_mid) -> dict:
    """Recompute ALL of the forager's dependent features FRESH at the live mid,
    every fast-guard tick (~1s), instead of reusing cycle-stale values. Patches
    the last cached candle with the live mid (no API fetch) and re-runs
    ecology.classify + CSD + the MPC edge + pnl_R, so every input updates on the
    second. The multiasset network inputs stay cycle-cached (1s exchange polls
    would be abusive) — only the candle/price-derived features move at 1s, plus
    pnl_R. Returns the cached forager_state merged with the fresh signals; falls
    back to the cache on any failure."""
    cached = dict(store.forager_state or {})
    try:
        candles = list(getattr(store, "candles_full", None) or
                       (store.market or {}).get("candles") or [])
        px = float(live_mid or 0.0)
        if len(candles) < 8 or px <= 0:
            return cached
        last = dict(candles[-1])
        last["close"] = px
        last["high"] = max(float(last.get("high") or px), px)
        last["low"] = min(float(last.get("low") or px), px)
        candles = candles[:-1] + [last]
        params = dict(getattr(settings, "strategy_params", {}) or {})
        market = dict(store.market or {})
        market["price"] = px
        multi = store.multiasset or {}

        from . import ecology as _ecology
        eco = _ecology.classify(candles, market, multi, params)
        orgs = ((eco.get("organisms") or {}).get("scores") or {})
        drivers = eco.get("drivers") or {}
        netm = eco.get("network_metrics") or {}

        from . import csd as _csd
        closes = [c["close"] for c in candles if c.get("close")]
        skew_hist = list((store.csd_state or {}).get("skew_history") or [])
        try:
            csd_risk, _sk = _csd.current_refined_risk(
                closes, skew_hist,
                fv_period=int(getattr(settings, "csd_governor_fv_period", 32) or 32),
                window=int(getattr(settings, "csd_governor_window", 96) or 96))
        except Exception:
            csd_risk = float((store.csd_state or {}).get("risk") or 0.0)

        try:
            from . import signals as _signals
            pos_s, _u = _signals.mpc_with_aux(candles, params, blend_context=None)
            edge = abs(pos_s[-1]) * 0.6 if pos_s else 0.0
        except Exception:
            edge = float(cached.get("edge") or 0.0)

        acct = store.account or {}
        posq = float(acct.get("position_contracts") or 0.0)
        entry = (store.position_entry or {}).get("entry")
        atr_pct = float((market.get("features") or {}).get("atr_pct") or 0.0)
        risk = abs(posq) * px * (atr_pct / 100.0)
        upnl = posq * (px - entry) if entry else float(acct.get("unrealized_pnl") or 0.0)
        pnl_R = (upnl / risk) if risk > 1e-9 else 0.0

        disturbance = float(drivers.get("disturbance") or 0.0)
        reserve = float(netm.get("rel_reserve") or 0.0)
        edge_ref = float(getattr(settings, "forager_edge_ref", 0.30) or 0.30)
        edge_decay = _clip(1.0 - (edge / edge_ref), 0.0, 1.0) if edge_ref > 0 else 0.0
        decomposer = float(orgs.get("decomposer") or 0.0)
        scavenger = float(orgs.get("scavenger") or 0.0)
        disturbance_score = _clip(disturbance, 0.0, 1.0)
        reserve_recovery = _clip(reserve, 0.0, 1.0)
        pnl_R_ref = float(getattr(settings, "forager_pnl_R_ref", 1.0) or 1.0)
        realized_profit_score = _clip(pnl_R / pnl_R_ref, 0.0, 1.0)
        harvest_pressure = round(
            realized_profit_score + decomposer + reserve_recovery + edge_decay
            - disturbance_score - scavenger, 3)
        reserve_prev = float(cached.get("reserve_prev", reserve))
        csd_prev = float(cached.get("csd_prev", csd_risk))
        from . import reflex as _reflex
        hard_pressure = float(getattr(settings, "forager_reflex_pressure_threshold", 1.25) or 1.25)
        dyn = {}
        dyn.update(_reflex.dynamics(cached, "harvest_pressure", harvest_pressure,
                                    threshold=hard_pressure))
        dyn.update(_reflex.dynamics(cached, "edge_decay", edge_decay))
        dyn.update(_reflex.dynamics(cached, "decomposer", decomposer))
        dyn.update(_reflex.dynamics(cached, "scavenger", scavenger))
        out = {
            **cached,
            "pnl_R": round(pnl_R, 3),
            "edge": round(edge, 4),
            "edge_decay": round(edge_decay, 3),
            "decomposer": round(decomposer, 3),
            "scavenger": round(scavenger, 3),
            "disturbance_score": round(disturbance_score, 3),
            "stretch_abs": round(abs(float(drivers.get("stretch_z") or 0.0)), 3),
            "reserve": round(reserve, 4),
            "reserve_recovery": round(reserve_recovery, 3),
            "reserve_rising": (reserve - reserve_prev) > 0.01,
            "rel_ascendancy": round(float(netm.get("rel_ascendancy") or 0.0), 4),
            "csd_risk": round(float(csd_risk), 4),
            "csd_rising": (float(csd_risk) - csd_prev) > 0.01,
            "realized_profit_score": round(realized_profit_score, 3),
            "harvest_pressure": harvest_pressure,
            "live_recomputed": True,
            "ts": _now_iso(),
        }
        out.update(dyn)
        return out
    except Exception:
        return cached


def harvest_decision(settings, store, sig: dict) -> bool:
    """Given a (fresh) signal dict, would the forager harvest now? Read-only —
    no state mutation. Mirrors apply()'s harvest tiers."""
    if not bool(getattr(settings, "forager_enabled", False)):
        return False
    if sig.get("in_cooldown") or (store.forager_state or {}).get("in_cooldown"):
        return False
    pos = float((store.account or {}).get("position_contracts") or 0.0)
    if abs(pos) < 1e-9:
        return False
    pnl_R = float(sig.get("pnl_R") or 0.0)
    if pnl_R <= 0:
        return False
    P = _autoscale(_resolve_params(settings), sig, settings)
    edge_decay = float(sig.get("edge_decay") or 0.0)
    decomposer = float(sig.get("decomposer") or 0.0)
    edge = float(sig.get("edge") or 0.0)
    disturbance = float(sig.get("disturbance_score") or 0.0)
    csd_risk = float(sig.get("csd_risk") or 0.0)
    csd_rising = bool(sig.get("csd_rising"))
    if disturbance >= P["churn_disturbance"] and pnl_R >= P["churn_profit_R"]:
        return True
    if pnl_R >= P["profit_threshold_R"] and edge_decay >= P["edge_decay_trim"]:
        return True
    if pnl_R >= P["tier2_R"] and decomposer >= P["decomposer"]:
        return True
    if pnl_R >= P["tier3_R"] and edge <= P["edge_gone"]:
        return True
    if csd_rising and csd_risk >= P["csd_high"]:
        return True
    early, _reason = _early_harvest_reflex(settings, sig, P)
    if early:
        return True
    return False


def _early_harvest_reflex(settings, sig: dict, P: dict) -> tuple[bool, str]:
    """Second-order harvest: trim before hard tiers when pressure is turning."""
    if not bool(getattr(settings, "forager_reflex_enabled", True)):
        return False, ""
    pnl_R = float(sig.get("pnl_R") or 0.0)
    min_profit = max(float(P["profit_threshold_R"]) * 0.45,
                     float(getattr(settings, "forager_reflex_min_profit_R", 0.05) or 0.05))
    if pnl_R <= max(0.0, min_profit):
        return False, ""
    hp = float(sig.get("harvest_pressure") or 0.0)
    hp_projected = float(sig.get("harvest_pressure_projected", hp) or hp)
    soft = float(getattr(settings, "forager_reflex_soft_pressure", 0.45) or 0.45)
    if max(hp, hp_projected) < soft:
        return False, ""
    hp_delta = float(sig.get("harvest_pressure_delta") or 0.0)
    hp_accel = float(sig.get("harvest_pressure_acceleration") or 0.0)
    hp_impulse = float(sig.get("harvest_pressure_impulse") or 0.0)
    ttt = sig.get("harvest_pressure_time_to_threshold_s")
    max_ttt = float(getattr(settings, "forager_reflex_max_ttt_s", 20.0) or 20.0)
    min_delta = float(getattr(settings, "forager_reflex_min_pressure_delta", 0.10) or 0.10)
    pressure_turning = (
        hp_delta >= min_delta
        or hp_accel > 0.0
        or hp_impulse >= min_delta * 0.5
        or bool(sig.get("predictive_harvest"))
        or (ttt is not None and float(ttt) <= max_ttt)
    )
    if not pressure_turning:
        return False, ""

    edge_delta = float(sig.get("edge_decay_delta") or 0.0)
    dec_delta = float(sig.get("decomposer_delta") or 0.0)
    scav_delta = float(sig.get("scavenger_delta") or 0.0)
    edge_min = float(getattr(settings, "forager_reflex_min_edge_decay_delta", 0.04) or 0.04)
    eco_turning = (
        edge_delta >= edge_min
        or dec_delta >= 0.04
        or scav_delta <= -0.04
        or float(sig.get("edge_decay") or 0.0) >= float(P["edge_decay_trim"]) * 0.75
    )
    if not eco_turning:
        return False, ""
    reason = (f"predictive harvest reflex: HP {hp:+.2f}->{hp_projected:+.2f} "
              f"(d {hp_delta:+.2f}, impulse {hp_impulse:+.2f})")
    return True, reason


def would_harvest(settings, store, live_mid) -> bool:
    """Convenience: recompute fresh signals at the live mid and decide. The
    fast-guard tick uses live_signals + harvest_decision directly so it can also
    publish the fresh signals to the UI without recomputing twice."""
    return harvest_decision(settings, store, live_signals(settings, store, live_mid))


def apply(settings, store, account: dict, market: dict, proposal: dict) -> dict:
    """Foraging-cycle overlay. See module docstring. Mutates `proposal` only
    when forager_enabled; always writes diagnostics to STORE.forager_state."""
    prev = store.forager_state or {}
    now = time.time()
    enabled = bool(getattr(settings, "forager_enabled", False))
    sig = _signals(settings, store, account, market, proposal, prev)

    state = dict(sig)
    state.update({
        "ts": _now_iso(),
        "enabled": enabled,
        "reserve_prev": sig["reserve"],   # carry for next-cycle delta
        "csd_prev": sig["csd_risk"],
    })

    cooldown_until = prev.get("cooldown_until")
    cooldown_started_at = prev.get("cooldown_started_at")
    cooldown_seconds_cur = float(prev.get("cooldown_seconds") or 0.0)

    if not enabled:
        state.update({"in_cooldown": False, "cooldown_until": None,
                      "cooldown_remaining_s": 0, "harvest_scale": 1.0,
                      "harvested": False, "blocked_entry": False,
                      "hunger": _resolve_params(settings)["hunger"],
                      "eagerness": 0.0, "phase": "off",
                      "captured_cumulative": round(float(getattr(store, "forager_captured_cumulative", 0.0) or 0.0), 4),
                      "action": "disabled"})
        store.forager_state = state
        return state

    action = proposal.get("action")
    pos = float(account.get("position_contracts") or 0.0)
    have_position = abs(pos) > 1e-9

    # --- thresholds: hunger preset, then AUTOSCALED by the live regime/network
    # state (eagerness) so a brittle/disturbed market eats sooner & deeper. ---
    P = _autoscale(_resolve_params(settings), sig, settings)
    state["hunger"] = P["hunger"]
    state["eagerness"] = P["eagerness"]
    profit_threshold = float(P["profit_threshold_R"])
    tier2_R = float(P["tier2_R"])
    tier3_R = float(P["tier3_R"])
    edge_decay_trim = float(P["edge_decay_trim"])
    decomposer_thr = float(P["decomposer"])
    edge_gone = float(P["edge_gone"])
    csd_high = float(P["csd_high"])
    churn_dist = float(P["churn_disturbance"])
    churn_profit_R = float(P["churn_profit_R"])
    churn_scale = float(P["churn_scale"])
    tier1_scale = float(P["tier1_scale"])
    tier2_scale = float(P["tier2_scale"])
    tier3_scale = float(P["tier3_scale"])
    # reentry stays structural (not part of the hunger preset)
    reentry_dist = float(getattr(settings, "forager_reentry_disturbance", 0.7) or 0.7)
    reentry_scav = float(getattr(settings, "forager_reentry_scavenger", 0.6) or 0.6)
    reentry_edge = float(getattr(settings, "forager_reentry_edge", 0.20) or 0.20)
    reentry_fair = float(getattr(settings, "forager_reentry_fair_stretch", 0.5) or 0.5)
    # Cooldown duration override (UI knob): >0 = fixed N-second HARD timer that
    # blocks entries for the whole duration (no early exit). 0 = dynamic preset.
    cd_override = float(getattr(settings, "forager_cooldown_seconds", 0) or 0)
    # Minimum fraction of the cooldown that must elapse before any early re-exit
    # — without this floor the persistently-high scavenger/disturbance in a
    # churny regime clears the cooldown on the very next cycle and it never
    # actually blocks entries (the reported bug).
    min_rest_frac = float(getattr(settings, "forager_min_rest_frac", 0.6) or 0.6)

    # --- 1) Cooldown: active? end early on ecological reset / new disturbance,
    # but only AFTER the minimum rest has elapsed. ---
    in_cooldown = cooldown_until is not None and now < float(cooldown_until)
    reentry_reason = None
    if in_cooldown:
        rested = (now - float(cooldown_started_at)) if cooldown_started_at else 1e9
        min_rest = cooldown_seconds_cur if cd_override > 0 else (min_rest_frac * cooldown_seconds_cur)
        if rested >= min_rest:
            if sig["disturbance_score"] >= reentry_dist:
                reentry_reason = "new disturbance detected"
            elif sig["scavenger"] >= reentry_scav:
                reentry_reason = "scavenger / snap-back signal returned"
            elif sig["edge"] >= reentry_edge:
                reentry_reason = "MR permission rebuilt (edge recovered)"
            elif sig["stretch_abs"] <= reentry_fair:
                reentry_reason = "price back in fair-value reset zone"
        if reentry_reason:
            in_cooldown = False
            cooldown_until = None

    # --- 2) Harvest decision (only when holding a profitable position) ---
    # Collect every triggered tier, then take the MOST aggressive (smallest
    # scale) with its reason.
    harvest_scale = 1.0
    harvest_reason = None
    if have_position and sig["pnl_R"] > 0:
        candidates = []   # (scale, reason)
        # CHURN DEFENSE (references disturbance): in a disturbed/choppy regime
        # profit is temporary and high ATR suppresses pnl_R, so harvest on a LOW
        # profit bar the moment disturbance is elevated. This is the primary
        # churn protection the pnl_R-only tiers lacked.
        if sig["disturbance_score"] >= churn_dist and sig["pnl_R"] >= churn_profit_R:
            candidates.append((churn_scale,
                f"churn harvest: disturbance {sig['disturbance_score']:.2f} >= "
                f"{churn_dist:.2f} + profit -> lock in temporary profit"))
        # Tier 1: small profit + edge weakening -> light trim (bite size = hunger)
        if sig["pnl_R"] >= profit_threshold and sig["edge_decay"] >= edge_decay_trim:
            candidates.append((tier1_scale,
                f"profit + edge weakening -> trim to {tier1_scale:.0%}"))
        # Tier 2: profit + decomposer rising -> heavier trim
        if sig["pnl_R"] >= tier2_R and sig["decomposer"] >= decomposer_thr:
            candidates.append((tier2_scale,
                f"profit + decomposer rising -> trim to {tier2_scale:.0%}"))
        # Tier 3: profit + edge gone -> deep harvest (full close at high hunger)
        if sig["pnl_R"] >= tier3_R and sig["edge"] <= edge_gone:
            candidates.append((tier3_scale,
                f"profit + edge gone -> harvest to {tier3_scale:.0%}"))
        # CSD risk rising -> always a hard exit (risk event, hunger-independent)
        if sig["csd_rising"] and sig["csd_risk"] >= csd_high:
            candidates.append((0.0, "profit + CSD risk rising -> close aggressively"))
        early_reflex, early_reason = _early_harvest_reflex(settings, sig, P)
        if early_reflex:
            reflex_scale = _clip(float(getattr(settings, "forager_reflex_scale", 0.50) or 0.50),
                                 0.0, 1.0)
            candidates.append((min(tier1_scale, reflex_scale), early_reason))
        if candidates:
            harvest_scale, harvest_reason = min(candidates, key=lambda c: c[0])

    # --- 3) Apply harvest: reduce target toward flat; start the refractory ---
    harvested = harvest_scale < 1.0 and have_position
    if harvested:
        cur_frac = _current_fraction(settings, account, market)
        new_target = round(harvest_scale * cur_frac, 4)
        proposal["forager_harvested"] = True
        proposal["forager_prev_action"] = action
        proposal["forager_harvest_scale"] = harvest_scale
        proposal["target_fraction"] = new_target
        proposal["blended_score"] = new_target
        proposal["action"] = "CLOSE" if harvest_scale == 0.0 else "REDUCE"
        proposal.setdefault("rationale_for", []).insert(
            0, f"Forager harvest ({harvest_reason}); pnl_R {sig['pnl_R']:.2f}, "
               f"edge {sig['edge']:.3f}, decomposer {sig['decomposer']:.2f}")
        cd = cd_override if cd_override > 0 else _cooldown_seconds(P, sig)
        cooldown_until = now + cd
        cooldown_started_at = now
        cooldown_seconds_cur = cd
        in_cooldown = True
        state["cooldown_seconds"] = round(cd, 1)
        state["last_harvest_ts"] = _now_iso()
        state["last_harvest_scale"] = harvest_scale
        state["last_harvest_reason"] = harvest_reason
        # Record captured profit (locked in by the reduction) for the live
        # "captured profit over time" diagram. Captured ≈ open profit × the
        # fraction we just took off.
        captured = max(0.0, float(account.get("unrealized_pnl") or 0.0)) * (1.0 - harvest_scale)
        try:
            store.record_forager_harvest(captured, harvest_reason, sig.get("pnl_R"))
        except Exception:
            pass

    # --- 4) Refractory: block NEW entries (exits always allowed) ---
    blocked_entry = False
    if in_cooldown and not harvested and action in ENTRY_ACTIONS:
        remaining = int(float(cooldown_until) - now) if cooldown_until else 0
        proposal["forager_cooldown_blocked"] = True
        proposal["forager_blocked_action"] = action
        proposal["action"] = "HOLD"
        proposal["target_fraction"] = 0.0
        proposal["blended_score"] = 0.0
        proposal["confidence"] = 0.0
        proposal.setdefault("rationale_against", []).insert(
            0, f"Forager refractory: resting after harvest ({remaining}s left, "
               f"HP {sig['harvest_pressure']:+.2f})")
        blocked_entry = True

    # Foraging-cycle phase for the live diagram.
    if harvested:
        phase = "harvesting"
    elif in_cooldown:
        phase = "resting"
    elif reentry_reason:
        phase = "reentering"
    elif have_position:
        phase = "working"      # holding a position, watching the edge
    else:
        phase = "searching"    # flat, hunting for an entry

    state.update({
        "in_cooldown": in_cooldown,
        "cooldown_until": cooldown_until,
        "cooldown_started_at": cooldown_started_at if in_cooldown else None,
        "cooldown_seconds": round(cooldown_seconds_cur, 1) if in_cooldown else 0,
        "cooldown_remaining_s": (max(0, int(float(cooldown_until) - now))
                                  if cooldown_until else 0),
        "harvest_scale": harvest_scale,
        "harvest_reason": harvest_reason,
        "harvested": harvested,
        "blocked_entry": blocked_entry,
        "reentry_reason": reentry_reason,
        "phase": phase,
        "captured_cumulative": round(float(getattr(store, "forager_captured_cumulative", 0.0) or 0.0), 4),
    })
    store.forager_state = state
    return state
