"""Live strategy layer. Uses the validated signal variants in `signals.py`
(single source of truth shared with backtest + tuning). The desired direction is
the latest bar's target position from the selected variant; we then map desired
vs. current position to a concrete action.

`adaptive_blend` is kept as a legacy option but the validated default is
`regime` (trend / mean-reversion switch), which is the only family that held up
out-of-sample in walk-forward testing.
"""
from __future__ import annotations

from typing import List

from . import signals


def evaluate(candles: List[dict], params: dict, position_contracts: float,
             variant: str = "regime", market: dict = None,
             ecosystem: dict = None, ecosystem_apply: bool = False,
             csd_state: dict = None, blend_context: dict = None,
             live_current_fraction: float = None) -> dict:
    """Evaluate the live strategy for the latest bar. The MPC branch is the
    validated edge; other variants exist for comparison/tuning.

    `ecosystem` (optional): output of ecology.classify_phase(...) for the
    current bar. Always attached to the returned proposal (for the UI). The
    phase's `size_mult` is APPLIED to the MPC target_fraction only when
    `ecosystem_apply` is True (matches Settings.ecosystem_phase).

    `csd_state` (optional): {"risk", "threshold", "enabled", ...} from the
    engine. When enabled and risk > threshold, zeros the MPC fraction (gate
    mode). Always attached to the proposal for the UI panel.
    """
    if variant not in signals.VARIANTS:
        variant = "regime"

    # MPC variants get pos + urgency in one pass (urgency feeds the maker/taker
    # switch). robust_mpc keeps the same continuous target but conditions
    # gain/band/caps on ecology and uncertainty.
    if variant in ("mpc", "robust_mpc"):
        controller = {}
        if variant == "robust_mpc":
            pos_series, urgency_series, controller = signals.robust_mpc_with_diagnostics(
                candles, params, blend_context=blend_context, ecosystem=ecosystem,
                live_current_fraction=live_current_fraction)
        else:
            pos_series, urgency_series = signals.mpc_with_aux(
                candles, params, blend_context=blend_context,
                live_current_fraction=live_current_fraction)
        desired_sign = pos_series[-1] if pos_series else 0.0
        urgency = urgency_series[-1] if urgency_series else 0.0
        frac_base = desired_sign  # continuous target in [-1, 1] from MPC
        eco_mult = float((ecosystem or {}).get("size_mult", 1.0))
        if ecosystem_apply and variant == "mpc":
            # Multiply MAGNITUDE only; never flip sign. The MPC controller
            # picks side from regime-scaled mean-reversion alpha.
            frac = max(-1.0, min(1.0, frac_base * eco_mult))
        else:
            frac = frac_base
        # CSD risk governor: zero the fraction on cycles where the refined
        # skew-only CSD risk exceeds threshold. Acts AFTER ecology so the
        # ecology multiplier doesn't override a gated cycle. Records the gate
        # decision on the proposal for the UI / decision log.
        csd_gated = False
        csd_risk = float((csd_state or {}).get("risk") or 0.0)
        csd_threshold = float((csd_state or {}).get("threshold") or 0.95)
        csd_enabled = bool((csd_state or {}).get("enabled"))
        if csd_enabled and csd_risk > csd_threshold and abs(frac) > 0.001:
            frac = 0.0
            csd_gated = True
        conf = min(1.0, abs(frac))
        have_long = position_contracts > 0
        have_short = position_contracts < 0
        flat = position_contracts == 0
        if abs(frac) < 0.03:
            action = "CLOSE" if not flat else "HOLD"
        elif flat:
            action = "ENTER_LONG" if frac > 0 else "ENTER_SHORT"
        elif (frac > 0) == have_long:        # same direction -> track toward target
            action = "ADD"
        else:                                 # opposite sign -> flip
            action = "REVERSE_LONG" if frac > 0 else "REVERSE_SHORT"
        label = "Robust MPC" if variant == "robust_mpc" else "MPC"
        rationale = [f"{label} target {frac_base:+.2f} of max (band-controlled, low turnover)"]
        if variant == "robust_mpc" and ecosystem:
            rel_reserve = ((ecosystem.get("network_metrics") or {}).get("rel_reserve", 0))
            disturbance = ((ecosystem.get("drivers") or {}).get("disturbance", 0))
            rationale.insert(0, f"Robust ecology conditioning: phase "
                                f"{ecosystem.get('phase','?').upper()}, "
                                f"reserve {rel_reserve:.2f}, disturbance {disturbance:.2f}")
        elif ecosystem_apply and ecosystem and ecosystem.get("phase") not in (None, "churn"):
            rationale.insert(0, f"Ecology {ecosystem['phase'].upper()} "
                                f"(×{eco_mult:.2f}): {ecosystem.get('rationale','')}")
        if csd_gated:
            rationale.insert(0, f"CSD governor GATED: risk {csd_risk:.3f} > "
                                f"threshold {csd_threshold:.2f} — sitting out this cycle")
        if controller and controller.get("controller_move") == "hold_band":
            rationale.append(
                f"Controller held inside band: aim {controller.get('aim_capped', 0):+.3f}, "
                f"band {controller.get('band', 0):.3f}, target {controller.get('target_after', 0):+.3f}")
        return {
            "variant": variant,
            "desired_direction": "LONG" if frac > 0.03 else "SHORT" if frac < -0.03 else "FLAT",
            "blended_score": round(frac, 4),
            "target_fraction": round(frac, 4),
            "target_fraction_base": round(frac_base, 4),
            "confidence": round(conf, 4),
            "expected_edge_pct": round(conf * 0.6, 4),
            "urgency": round(urgency, 5),
            "action": action,
            "regime": _regime_label(candles, params),
            "rationale_for": rationale,
            "rationale_against": [],
            "components": {"target_fraction": round(frac, 4),
                           "urgency": round(urgency, 5),
                           "robust": variant == "robust_mpc",
                           "controller": controller},
            "controller": controller,
            "ecosystem": ecosystem,
            "ecosystem_applied": bool(ecosystem_apply or variant == "robust_mpc"),
            "csd_state": csd_state,
            "csd_gated": csd_gated,
        }

    fn = signals.VARIANTS[variant]
    pos_series = fn(candles, params)
    desired_sign = pos_series[-1] if pos_series else 0.0
    confidence = signals.confidence_for(variant, candles, params)
    # Expected edge proxy: confidence scaled by recent volatility.
    atr = signals.atr_series(candles, int(params.get("atr_period", 14)))
    last_close = candles[-1]["close"] if candles else 0.0
    atr_pct = (atr[-1] / last_close * 100) if (atr and atr[-1] and last_close) else 0.5
    expected_edge_pct = round(confidence * max(0.3, min(2.5, atr_pct)), 4)

    desired = "LONG" if desired_sign > 0 else "SHORT" if desired_sign < 0 else "FLAT"
    have_long = position_contracts > 0
    have_short = position_contracts < 0
    flat = position_contracts == 0

    action = "HOLD"
    rationale = []
    against = []

    if flat:
        if desired == "LONG":
            action = "ENTER_LONG"; rationale.append("Signal flipped long while flat.")
        elif desired == "SHORT":
            action = "ENTER_SHORT"; rationale.append("Signal flipped short while flat.")
        else:
            action = "HOLD"; against.append("Signal flat — no edge to act on.")
    elif have_long:
        if desired == "SHORT":
            action = "REVERSE_SHORT"; rationale.append("Long position; signal flipped short.")
        elif desired == "FLAT":
            action = "CLOSE"; rationale.append("Long position; signal went flat — exit.")
        else:
            action = "HOLD"; rationale.append("Long signal intact; holding.")
    elif have_short:
        if desired == "LONG":
            action = "REVERSE_LONG"; rationale.append("Short position; signal flipped long.")
        elif desired == "FLAT":
            action = "CLOSE"; rationale.append("Short position; signal went flat — exit.")
        else:
            action = "HOLD"; rationale.append("Short signal intact; holding.")

    # Urgency proxy for non-MPC variants: magnitude of position change requested.
    urgency_ns = signals.urgency_for(variant, candles, params)
    urgency = urgency_ns[-1] if urgency_ns else 0.0
    return {
        "variant": variant,
        "desired_direction": desired,
        "blended_score": desired_sign,
        "confidence": round(confidence, 4),
        "expected_edge_pct": expected_edge_pct,
        "urgency": round(urgency, 5),
        "action": action,
        "regime": _regime_label(candles, params),
        "rationale_for": rationale,
        "rationale_against": against,
        "components": {"target_position": desired_sign,
                       "atr_pct": round(atr_pct, 3),
                       "urgency": round(urgency, 5)},
        "ecosystem": ecosystem,
        "ecosystem_applied": False,   # only the MPC branch applies the multiplier
    }


def _regime_label(candles, params) -> str:
    closes = [c["close"] for c in candles]
    if len(closes) < int(params.get("ema_slow", 30)) + 2:
        return "insufficient-data"
    ef = signals.ema_series(closes, int(params["ema_fast"]))[-1]
    es = signals.ema_series(closes, int(params["ema_slow"]))[-1]
    sep = abs(ef - es) / closes[-1] * 100 if closes[-1] else 0
    return "trending" if sep > params.get("trend_gate", 0.1) else "ranging"
