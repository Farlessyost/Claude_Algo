"""Autotomy Agent: ecological loss-shedding reflex.

The forager exits because the profitable niche has been consumed.
The autotomy agent exits because the position itself has become dangerous:

    loss + toxic ecology + failed recovery -> hard exit + loss cooldown

It deliberately is NOT a simple stop loss. Small losses are normal in the
mean-reversion strategy; autotomy fires only when the loss occurs in the wrong
ecosystem: predator/cascade risk rising, CSD/skew stress, adaptive reserve
collapsing, the expected snap-back failing, and the original edge flipping.

DEFAULT OFF via settings.autotomy_enabled. Diagnostics are always computed into
STORE.autotomy_state so the UI can show what the reflex would do before it is
armed.
"""
from __future__ import annotations

import time
from datetime import datetime, timezone

ENTRY_ACTIONS = {"ENTER_LONG", "ENTER_SHORT", "REVERSE_LONG", "REVERSE_SHORT", "ADD"}

AUTOTOMY_AGGRESSION_PRESETS = {
    "lazy": {
        "pressure_threshold": 3.4, "loss_R": 0.45, "min_confirmations": 4,
        "cooldown_seconds": 120, "reflex_soft_pressure": 2.7,
        "reflex_loss_R": 0.25, "reflex_min_confirmations": 3,
        "reflex_min_pressure_delta": 0.18, "reflex_max_ttt_s": 14.0,
        "recovery_scavenger_block": 0.80,
    },
    "steady": {
        "pressure_threshold": 3.0, "loss_R": 0.35, "min_confirmations": 3,
        "cooldown_seconds": 180, "reflex_soft_pressure": 2.2,
        "reflex_loss_R": 0.18, "reflex_min_confirmations": 2,
        "reflex_min_pressure_delta": 0.15, "reflex_max_ttt_s": 18.0,
        "recovery_scavenger_block": 0.75,
    },
    "hungry": {
        "pressure_threshold": 2.55, "loss_R": 0.25, "min_confirmations": 2,
        "cooldown_seconds": 240, "reflex_soft_pressure": 1.8,
        "reflex_loss_R": 0.12, "reflex_min_confirmations": 2,
        "reflex_min_pressure_delta": 0.10, "reflex_max_ttt_s": 24.0,
        "recovery_scavenger_block": 0.85,
    },
    "ravenous": {
        "pressure_threshold": 2.15, "loss_R": 0.15, "min_confirmations": 2,
        "cooldown_seconds": 300, "reflex_soft_pressure": 1.35,
        "reflex_loss_R": 0.08, "reflex_min_confirmations": 1,
        "reflex_min_pressure_delta": 0.06, "reflex_max_ttt_s": 30.0,
        "recovery_scavenger_block": 0.92,
    },
}
DEFAULT_AGGRESSION = "ravenous"


def _resolve_params(settings) -> dict:
    level = (getattr(settings, "autotomy_aggression", DEFAULT_AGGRESSION)
             or DEFAULT_AGGRESSION).lower()
    preset = AUTOTOMY_AGGRESSION_PRESETS.get(
        level, AUTOTOMY_AGGRESSION_PRESETS[DEFAULT_AGGRESSION])
    P = dict(preset, aggression=level)
    if bool(getattr(settings, "autotomy_use_raw_thresholds", False)):
        for src, dst in (
            ("autotomy_pressure_threshold", "pressure_threshold"),
            ("autotomy_loss_R", "loss_R"),
            ("autotomy_min_confirmations", "min_confirmations"),
            ("autotomy_reflex_soft_pressure", "reflex_soft_pressure"),
            ("autotomy_reflex_loss_R", "reflex_loss_R"),
            ("autotomy_reflex_min_confirmations", "reflex_min_confirmations"),
            ("autotomy_reflex_min_pressure_delta", "reflex_min_pressure_delta"),
            ("autotomy_reflex_max_ttt_s", "reflex_max_ttt_s"),
        ):
            val = getattr(settings, src, None)
            if val is not None:
                P[dst] = val
    cd_override = float(getattr(settings, "autotomy_cooldown_seconds", 0) or 0)
    if cd_override > 0:
        P["cooldown_seconds"] = cd_override
    return P


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _current_fraction(settings, account: dict, market: dict) -> float:
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


def _side(x: float) -> int:
    return 1 if x > 1e-9 else -1 if x < -1e-9 else 0


def _proposal_target_side(proposal: dict) -> int:
    if proposal.get("target_fraction") is not None:
        return _side(float(proposal.get("target_fraction") or 0.0))
    action = proposal.get("action")
    if action in ("ENTER_LONG", "REVERSE_LONG"):
        return 1
    if action in ("ENTER_SHORT", "REVERSE_SHORT"):
        return -1
    if action == "CLOSE":
        return 0
    return 0


def _signals(settings, store, account: dict, market: dict, proposal: dict,
             prev: dict) -> dict:
    P = _resolve_params(settings)
    pos = float(account.get("position_contracts") or 0.0)
    price = float((market or {}).get("price") or 0.0)
    upnl = float(account.get("unrealized_pnl") or 0.0)
    atr_pct = float(((market or {}).get("features") or {}).get("atr_pct") or 0.0)
    notional = abs(pos) * price
    risk_usd = notional * (atr_pct / 100.0)
    pnl_R = (upnl / risk_usd) if risk_usd > 1e-9 else 0.0
    loss_R = max(0.0, -pnl_R)

    entry = (getattr(store, "position_entry", None) or {}).get("entry")
    if entry and price and pos:
        adverse_pct = ((entry - price) / entry * 100.0) if pos > 0 else ((price - entry) / entry * 100.0)
        adverse_excursion_R = max(0.0, adverse_pct / max(atr_pct, 1e-6))
    else:
        adverse_excursion_R = loss_R

    eco = proposal.get("ecosystem") or (store.ecosystem or {})
    orgs = ((eco.get("organisms") or {}).get("scores") or {})
    predator = float(orgs.get("predator") or 0.0)
    scavenger = float(orgs.get("scavenger") or 0.0)
    immune = float(orgs.get("immune") or 0.0)
    drivers = eco.get("drivers") or {}
    disturbance = _clip(float(drivers.get("disturbance") or 0.0), 0.0, 1.0)
    liq_proxy_z = max(0.0, float(drivers.get("liq_proxy_z") or 0.0))
    stretch_abs = abs(float(drivers.get("stretch_z") or 0.0))
    netm = eco.get("network_metrics") or {}
    reserve = _clip(float(netm.get("rel_reserve") or 0.0), 0.0, 1.0)
    rel_ascendancy = _clip(float(netm.get("rel_ascendancy") or 0.0), 0.0, 1.0)
    reserve_prev = float(prev.get("reserve_prev", reserve))
    reserve_delta = reserve - reserve_prev
    reserve_collapse = _clip(max(0.0, -reserve_delta) * 8.0 + max(0.0, 0.35 - reserve), 0.0, 1.0)

    csd = store.csd_state or {}
    csd_risk = float(csd.get("risk") or 0.0)
    csd_thr = float(csd.get("threshold") or getattr(settings, "csd_governor_threshold", 0.95) or 0.95)
    csd_prev = float(prev.get("csd_prev", csd_risk))
    csd_rising = (csd_risk - csd_prev) > 0.01
    skew_gate = 1.0 if csd_risk >= (0.8 * csd_thr) else 0.0

    cur_side = _side(pos)
    target_side = _proposal_target_side(proposal)
    edge = float(proposal.get("expected_edge_pct") or 0.0)
    edge_flip = 1.0 if (cur_side and target_side and target_side != cur_side) else 0.0
    edge_gone = 1.0 if edge <= float(getattr(settings, "forager_edge_gone", 0.10) or 0.10) else 0.0
    recovery_failed = 1.0 if (loss_R > 0.0 and stretch_abs > 0.8 and scavenger < 0.45) else 0.0
    cascade_risk = _clip(0.45 * disturbance + 0.25 * liq_proxy_z + 0.20 * predator
                         + 0.10 * rel_ascendancy, 0.0, 1.0)
    depth_recovery = _clip(float(drivers.get("depth_recover") or 1.0), 0.0, 1.5) / 1.5

    pressure = (
        min(2.0, loss_R)
        + min(1.5, adverse_excursion_R)
        + predator
        + cascade_risk
        + skew_gate
        + min(1.0, csd_risk)
        + reserve_collapse
        + edge_flip
        + recovery_failed
        - 0.50 * depth_recovery
        - 0.75 * scavenger
    )
    confirmations = sum([
        loss_R >= float(P["loss_R"]),
        predator >= 0.60 or immune >= 0.55,
        cascade_risk >= 0.65,
        skew_gate > 0.0 or (csd_rising and csd_risk >= 0.65),
        reserve_collapse >= 0.35,
        edge_flip > 0.0 or edge_gone > 0.0,
        recovery_failed > 0.0,
    ])

    from . import reflex as _reflex
    hard_pressure = float(P["pressure_threshold"])
    dyn = {}
    dyn.update(_reflex.dynamics(prev, "autotomy_pressure", pressure,
                                threshold=hard_pressure))
    dyn.update(_reflex.dynamics(prev, "predator_score", predator))
    dyn.update(_reflex.dynamics(prev, "cascade_risk", cascade_risk))
    dyn.update(_reflex.dynamics(prev, "reserve_collapse", reserve_collapse))
    dyn.update(_reflex.dynamics(prev, "scavenger_score", scavenger))

    horizon = float(getattr(settings, "autotomy_predictive_horizon_seconds", 12.0) or 12.0)
    impulse_gain = float(getattr(settings, "autotomy_predictive_impulse_gain", 0.35) or 0.35)
    p_velocity = float(dyn.get("autotomy_pressure_velocity") or 0.0)
    p_accel = float(dyn.get("autotomy_pressure_acceleration") or 0.0)
    p_impulse = float(dyn.get("autotomy_pressure_impulse") or 0.0)
    pressure_projected = pressure
    if bool(getattr(settings, "autotomy_predictive_enabled", True)):
        pressure_projected = (
            pressure
            + max(0.0, p_velocity) * horizon
            + 0.5 * max(0.0, p_accel) * horizon * horizon
            + impulse_gain * max(0.0, p_impulse)
        )

    out = {
        "pnl_R": round(pnl_R, 3),
        "loss_R": round(loss_R, 3),
        "adverse_excursion_R": round(adverse_excursion_R, 3),
        "predator_score": round(predator, 3),
        "scavenger_score": round(scavenger, 3),
        "immune_score": round(immune, 3),
        "cascade_risk": round(cascade_risk, 3),
        "skew_gate": bool(skew_gate),
        "csd_risk": round(csd_risk, 4),
        "csd_rising": csd_rising,
        "reserve": round(reserve, 4),
        "reserve_delta": round(reserve_delta, 4),
        "reserve_collapse": round(reserve_collapse, 3),
        "edge": round(edge, 4),
        "edge_flip": bool(edge_flip),
        "edge_gone": bool(edge_gone),
        "recovery_failed": bool(recovery_failed),
        "depth_recovery": round(depth_recovery, 3),
        "disturbance_score": round(disturbance, 3),
        "rel_ascendancy": round(rel_ascendancy, 4),
        "autotomy_pressure": round(pressure, 3),
        "autotomy_pressure_projected": round(pressure_projected, 3),
        "predictive_eject": bool(pressure_projected >= hard_pressure),
        "confirmations": int(confirmations),
        "aggression": P["aggression"],
    }
    out.update(dyn)
    return out


def _should_eject(settings, sig: dict, have_position: bool) -> bool:
    if not have_position:
        return False
    P = _resolve_params(settings)
    if float(sig.get("loss_R") or 0.0) < float(P["loss_R"]):
        return False
    if int(sig.get("confirmations") or 0) < int(P["min_confirmations"]):
        return False
    pressure = max(float(sig.get("autotomy_pressure") or 0.0),
                   float(sig.get("autotomy_pressure_projected") or 0.0))
    return pressure >= float(P["pressure_threshold"])


def _should_reflex_eject(settings, sig: dict, have_position: bool) -> bool:
    """Early autotomy: pressure is below hard threshold but rising fast."""
    if not bool(getattr(settings, "autotomy_reflex_enabled", True)):
        return False
    if not have_position:
        return False
    P = _resolve_params(settings)
    loss_R = float(sig.get("loss_R") or 0.0)
    if loss_R < float(P["reflex_loss_R"]):
        return False
    if int(sig.get("confirmations") or 0) < int(P["reflex_min_confirmations"]):
        return False
    pressure = float(sig.get("autotomy_pressure") or 0.0)
    pressure_projected = float(sig.get("autotomy_pressure_projected") or pressure)
    soft = float(P["reflex_soft_pressure"])
    if max(pressure, pressure_projected) < soft:
        return False

    p_delta = float(sig.get("autotomy_pressure_delta") or 0.0)
    p_accel = float(sig.get("autotomy_pressure_acceleration") or 0.0)
    p_impulse = float(sig.get("autotomy_pressure_impulse") or 0.0)
    ttt = sig.get("autotomy_pressure_time_to_threshold_s")
    max_ttt = float(P["reflex_max_ttt_s"])
    min_delta = float(P["reflex_min_pressure_delta"])
    pressure_turning = (
        p_delta >= min_delta
        or p_accel > 0.0
        or p_impulse >= min_delta * 0.5
        or bool(sig.get("predictive_eject"))
        or (ttt is not None and float(ttt) <= max_ttt)
    )
    if not pressure_turning:
        return False

    toxic_turning = (
        float(sig.get("predator_score_delta") or 0.0) > 0.03
        or float(sig.get("cascade_risk_delta") or 0.0) > 0.03
        or float(sig.get("reserve_collapse_delta") or 0.0) > 0.03
        or bool(sig.get("edge_flip"))
        or bool(sig.get("recovery_failed"))
        or bool(sig.get("skew_gate"))
    )
    recovery_helping = (
        float(sig.get("scavenger_score") or 0.0) >= float(P["recovery_scavenger_block"])
        and float(sig.get("scavenger_score_delta") or 0.0) > 0.02
    )
    return toxic_turning and not recovery_helping


def live_signals(settings, store, live_mid) -> dict:
    """Best-effort fresh autotomy diagnostics at the live mid for fast_guard."""
    cached = dict(getattr(store, "autotomy_state", {}) or {})
    try:
        from . import forager as _forager
        fs = _forager.live_signals(settings, store, live_mid)
        acct = dict(store.account or {})
        market = dict(store.market or {})
        market["price"] = float(live_mid or market.get("price") or 0.0)
        proposal = {
            "action": "HOLD",
            "target_fraction": 0.0,
            "expected_edge_pct": fs.get("edge", cached.get("edge", 0.0)),
            "ecosystem": store.ecosystem or {},
        }
        prev = getattr(store, "autotomy_state", {}) or {}
        sig = _signals(settings, store, acct, market, proposal, prev)
        return {**cached, **sig, "live_recomputed": True, "ts": _now_iso()}
    except Exception:
        return cached


def eject_decision(settings, store, sig: dict) -> bool:
    if not bool(getattr(settings, "autotomy_enabled", False)):
        return False
    if sig.get("in_cooldown") or (getattr(store, "autotomy_state", {}) or {}).get("in_cooldown"):
        return False
    pos = float((store.account or {}).get("position_contracts") or 0.0)
    have_position = abs(pos) > 1e-9
    return (_should_eject(settings, sig, have_position)
            or _should_reflex_eject(settings, sig, have_position))


def apply(settings, store, account: dict, market: dict, proposal: dict) -> dict:
    prev = getattr(store, "autotomy_state", {}) or {}
    now = time.time()
    P = _resolve_params(settings)
    enabled = bool(getattr(settings, "autotomy_enabled", False))
    sig = _signals(settings, store, account, market, proposal, prev)
    state = dict(sig)
    state.update({
        "ts": _now_iso(),
        "enabled": enabled,
        "reserve_prev": sig["reserve"],
        "csd_prev": sig["csd_risk"],
    })

    cooldown_until = prev.get("cooldown_until")
    in_cooldown = cooldown_until is not None and now < float(cooldown_until)
    state["in_cooldown"] = bool(in_cooldown)

    action = proposal.get("action")
    pos = float(account.get("position_contracts") or 0.0)
    have_position = abs(pos) > 1e-9

    ejected = False
    eject_reason = None
    hard_eject = _should_eject(settings, sig, have_position)
    reflex_eject = _should_reflex_eject(settings, sig, have_position)
    if enabled and not in_cooldown and (hard_eject or reflex_eject):
        cur_frac = _current_fraction(settings, account, market)
        proposal["autotomy_ejected"] = True
        proposal["autotomy_reflex_ejected"] = bool(reflex_eject and not hard_eject)
        proposal["autotomy_prev_action"] = action
        proposal["autotomy_pressure"] = sig["autotomy_pressure"]
        proposal["target_fraction"] = 0.0
        proposal["blended_score"] = 0.0
        proposal["action"] = "CLOSE"
        proposal["force_taker"] = True
        proposal["confidence"] = max(float(proposal.get("confidence") or 0.0), 0.95)
        mode = "reflex early eject" if reflex_eject and not hard_eject else "hard eject"
        eject_reason = (
            f"{mode}: loss_R {sig['loss_R']:.2f}, pressure {sig['autotomy_pressure']:.2f}, "
            f"confirmations {sig['confirmations']}: position became bait")
        proposal.setdefault("rationale_against", []).insert(0, f"Autotomy Agent: {eject_reason}")
        cd = float(P["cooldown_seconds"])
        cooldown_until = now + cd
        in_cooldown = True
        ejected = True
        state.update({
            "last_eject_ts": _now_iso(),
            "last_eject_reason": eject_reason,
            "ejected_fraction": round(cur_frac, 4),
            "cooldown_seconds": round(cd, 1),
        })

    blocked_entry = False
    if enabled and in_cooldown and not ejected and action in ENTRY_ACTIONS:
        remaining = int(float(cooldown_until) - now) if cooldown_until else 0
        proposal["autotomy_cooldown_blocked"] = True
        proposal["autotomy_blocked_action"] = action
        proposal["action"] = "HOLD"
        proposal["target_fraction"] = 0.0
        proposal["blended_score"] = 0.0
        proposal["confidence"] = 0.0
        proposal.setdefault("rationale_against", []).insert(
            0, f"Autotomy refractory: loss-shedding cooldown ({remaining}s left)")
        blocked_entry = True

    if ejected:
        phase = "ejecting"
    elif in_cooldown:
        phase = "cooldown"
    elif have_position and sig["loss_R"] > 0:
        phase = "watching_loss"
    elif have_position:
        phase = "attached"
    else:
        phase = "idle"

    state.update({
        "in_cooldown": bool(in_cooldown),
        "cooldown_until": cooldown_until if in_cooldown else None,
        "cooldown_remaining_s": (max(0, int(float(cooldown_until) - now))
                                  if cooldown_until else 0),
        "ejected": ejected,
        "eject_reason": eject_reason,
        "blocked_entry": blocked_entry,
        "phase": phase,
        "hard_eject_ready": bool(hard_eject),
        "reflex_eject_ready": bool(reflex_eject),
        "threshold": float(P["pressure_threshold"]),
        "loss_R_threshold": float(P["loss_R"]),
        "min_confirmations": int(P["min_confirmations"]),
        "reflex_loss_R_threshold": float(P["reflex_loss_R"]),
        "reflex_soft_pressure": float(P["reflex_soft_pressure"]),
        "cooldown_seconds_configured": float(P["cooldown_seconds"]),
    })
    store.autotomy_state = state
    return state
