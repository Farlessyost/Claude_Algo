"""Canonical signal library — the single source of truth used by live trading,
backtest, and tuning so they can never diverge.

Each variant maps a candle series + params to a target-position series in
{-1, 0, +1}. The live engine takes the LAST value as the desired direction;
backtest/tuning simulate the whole series. Validated by walk-forward analysis:
`regime` and `meanrev` on 5m showed positive out-of-sample returns at the
current (zero) fee level; trend/breakout/momentum did not survive costs.
"""
from __future__ import annotations

import math
from typing import Callable, List, Optional


# --------------------------------------------------------------- indicators
def ema_series(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2.0 / (period + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def rsi_series(closes: List[float], period: int = 14) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(closes)
    if len(closes) < period + 1:
        return out
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0); losses += max(-d, 0.0)
    ag = gains / period; al = losses / period
    out[period] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        ag = (ag * (period - 1) + max(d, 0.0)) / period
        al = (al * (period - 1) + max(-d, 0.0)) / period
        out[i] = 100.0 if al == 0 else 100 - 100 / (1 + ag / al)
    return out


def atr_series(candles: List[dict], period: int = 14) -> List[Optional[float]]:
    n = len(candles)
    out: List[Optional[float]] = [None] * n
    if n < period + 1:
        return out
    trs = [0.0]
    for i in range(1, n):
        h, l, pc = candles[i]["high"], candles[i]["low"], candles[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = sum(trs[1:period + 1]) / period
    out[period] = atr
    for i in range(period + 1, n):
        atr = (atr * (period - 1) + trs[i]) / period
        out[i] = atr
    return out


def roc_series(closes: List[float], period: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(closes)
    for i in range(period, len(closes)):
        p = closes[i - period]
        out[i] = (closes[i] - p) / p * 100.0 if p else None
    return out


def rolling_max(vals: List[float], n: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(vals)
    for i in range(n, len(vals)):
        out[i] = max(vals[i - n:i])
    return out


def rolling_min(vals: List[float], n: int) -> List[Optional[float]]:
    out: List[Optional[float]] = [None] * len(vals)
    for i in range(n, len(vals)):
        out[i] = min(vals[i - n:i])
    return out


# --------------------------------------------------------------- variants
def sig_trend(candles, p) -> List[float]:
    closes = [c["close"] for c in candles]
    ef = ema_series(closes, int(p["ema_fast"]))
    es = ema_series(closes, int(p["ema_slow"]))
    et = ema_series(closes, int(p["ema_trend"]))
    atr = atr_series(candles, int(p["atr_period"]))
    pos = [0.0] * len(closes)
    for i in range(len(closes)):
        a = atr[i]
        if a is None or closes[i] == 0:
            continue
        if a / closes[i] * 100 < p["min_atr_pct"]:
            continue
        if ef[i] > es[i] and closes[i] > et[i]:
            pos[i] = 1.0
        elif ef[i] < es[i] and closes[i] < et[i]:
            pos[i] = -1.0
    return pos


def sig_breakout(candles, p) -> List[float]:
    highs = [c["high"] for c in candles]; lows = [c["low"] for c in candles]
    closes = [c["close"] for c in candles]
    n = int(p["donchian"])
    hh = rolling_max(highs, n); ll = rolling_min(lows, n)
    pos = [0.0] * len(closes); cur = 0.0
    for i in range(len(closes)):
        if hh[i] is None:
            continue
        if closes[i] >= hh[i]:
            cur = 1.0
        elif closes[i] <= ll[i]:
            cur = -1.0
        pos[i] = cur
    return pos


def sig_meanrev(candles, p) -> List[float]:
    closes = [c["close"] for c in candles]
    rsi = rsi_series(closes, int(p["rsi_period"]))
    ef = ema_series(closes, int(p["ema_fast"]))
    es = ema_series(closes, int(p["ema_slow"]))
    pos = [0.0] * len(closes); cur = 0.0
    for i in range(len(closes)):
        r = rsi[i]
        if r is None:
            continue
        trending = (abs(ef[i] - es[i]) / closes[i] * 100 > p["trend_gate"]) if closes[i] else False
        if not trending:
            if r <= p["rsi_oversold"]:
                cur = 1.0
            elif r >= p["rsi_overbought"]:
                cur = -1.0
            elif 45 <= r <= 55:
                cur = 0.0
        pos[i] = cur
    return pos


def sig_momentum(candles, p) -> List[float]:
    closes = [c["close"] for c in candles]
    roc = roc_series(closes, int(p["roc_period"]))
    pos = [0.0] * len(closes)
    for i in range(len(closes)):
        v = roc[i]
        if v is None:
            continue
        if v > p["roc_threshold"]:
            pos[i] = 1.0
        elif v < -p["roc_threshold"]:
            pos[i] = -1.0
    return pos


def sig_regime(candles, p) -> List[float]:
    closes = [c["close"] for c in candles]
    ef = ema_series(closes, int(p["ema_fast"]))
    es = ema_series(closes, int(p["ema_slow"]))
    trend = sig_trend(candles, p); mr = sig_meanrev(candles, p)
    pos = [0.0] * len(closes)
    for i in range(len(closes)):
        if closes[i] == 0:
            continue
        sep = abs(ef[i] - es[i]) / closes[i] * 100
        pos[i] = trend[i] if sep > p["trend_gate"] else mr[i]
    return pos


def sig_mr_edge(candles, p) -> List[float]:
    """Validated 2m mean-reversion edge (see backend/edge.py + research)."""
    from . import edge as _edge
    closes = [c["close"] for c in candles]
    ep = {
        "vol_win": int(p.get("vol_win", 12)),
        "lookback": int(p.get("lookback", 2)),
        "beta": p.get("beta", 0.12),
        "k": p.get("k", 1.2),
        "z_cap": p.get("z_cap", 3.5),
        "deadband_bps": p.get("deadband_bps", 2.0),
    }
    return _edge.target_series(closes, ep)


def _std(xs) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def efficiency_ratio(closes: List[float], w: int) -> List[float]:
    """Kaufman efficiency ratio: |net move| / path length over w bars.
    ~1 = clean trend, ~0 = chop. Used as the MPC regime scaler."""
    out = [0.0] * len(closes)
    for i in range(w, len(closes)):
        net = abs(closes[i] - closes[i - w])
        path = sum(abs(closes[j] - closes[j - 1]) for j in range(i - w + 1, i + 1)) or 1e-9
        out[i] = net / path
    return out


def sig_mpc(candles, p) -> List[float]:
    """Cost-aware MPC (Garleanu-Pedersen style): alpha-proportional aim + a
    no-trade band + a regime scaler. Returns a CONTINUOUS target position in
    [-1, 1]; it only moves to the band edge around the aim, which keeps turnover
    low — the validated fix for the sub-spread cost problem."""
    return mpc_with_aux(candles, p)[0]


def _mpc_core(candles, p, blend_context: Optional[dict] = None,
              robust: bool = False, ecosystem: Optional[dict] = None,
              return_diagnostics: bool = False,
              live_current_fraction: Optional[float] = None):
    """Same as sig_mpc but also returns the per-bar urgency series |alpha|
    (the magnitude of the regime-scaled, gain-applied reversion aim). The live
    executor uses urgency to choose maker-vs-taker per cycle, and the hybrid
    backtester uses it for the same threshold.

    `blend_context` (optional): if provided AND p.get("blend_enabled") is True,
    the directional alpha components from signals_blended.py are added to
    MPC's reversion alpha with the configured weights. Schema:
        blend_context = {
            "spot_closes": List[float],     # rolling spot mid history
            "funding_history": List[float], # rolling funding rate history
            "oi_history": List[float],      # rolling OI history
            "oi_price_history": List[float], # price aligned to oi_history
        }
    Any missing key falls back to that component contributing 0.
    Validation: see backend/backtest_blended.py — spot-lead at w=0.30 lifted
    aggregate WF return +14.28pp and sharpe +0.85 over MPC-alone, OOS.
    """
    closes = [c["close"] for c in candles]
    n = len(closes)
    if n < 5:
        return [0.0] * n, [0.0] * n
    vw = int(p.get("vol_win", 12)); lb = int(p.get("lookback", 2))
    rw = int(p.get("regime_win", 8))
    beta = p.get("beta", 0.18); gain = p.get("gain", 1.0)
    band = p.get("band", 0.5); er_cap = p.get("er_cap", 1.0)
    robust_lambda = float(p.get("robust_lambda", 0.35))
    robust_disturbance_lambda = float(p.get("robust_disturbance_lambda", 0.45))
    phase_mod = {
        "producer":   {"gain": 0.90, "band": 0.85, "cap": 0.60},
        "predator":   {"gain": 0.35, "band": 1.60, "cap": 0.25},
        "exhaustion": {"gain": 0.80, "band": 0.95, "cap": 0.50},
        "scavenger":  {"gain": 1.35, "band": 0.25, "cap": 1.00},
        "decomposer": {"gain": 1.15, "band": 0.30, "cap": 0.70},
        "churn":      {"gain": 0.95, "band": 0.75, "cap": 0.70},
    }
    states = None
    if robust:
        try:
            from . import ecology as _ecology
            _, states = _ecology.phase_series_with_states(candles, p)
        except Exception:
            states = ["churn"] * n

    # Blend configuration (only the LATEST bar gets a blend tilt — the
    # historical bars use MPC-only since we don't have aligned spot/funding/OI
    # for the entire history in the live path. The backtest path uses a
    # separate dedicated walker in backend/backtest_blended.py.).
    blend_enabled = bool(p.get("blend_enabled", False))
    blend_params = None
    spot_tilt = funding_tilt = oi_tilt = ecology_tilt = visual_tilt = 0.0
    if blend_enabled and blend_context:
        try:
            from . import signals_blended as _sb
            blend_params = {
                "w_mpc": float(p.get("blend_w_mpc", 1.0)),
                "w_spot_lead": float(p.get("blend_w_spot_lead", 0.30)),
                "w_funding_fade": float(p.get("blend_w_funding_fade", 0.0)),
                "w_oi_pressure": float(p.get("blend_w_oi_pressure", 0.0)),
                "w_ecology_flow": float(p.get("blend_w_ecology_flow", 0.0)),
                "w_visual": float(p.get("blend_w_visual", 0.0)),
                "spot_lookback": int(p.get("blend_spot_lookback", 3)),
                "spot_history_for_std": int(p.get("blend_spot_history_for_std", 60)),
                "funding_history_for_std": int(p.get("blend_funding_history_for_std", 60)),
                "funding_persistence_threshold": float(
                    p.get("blend_funding_persistence_threshold", 0.5)),
                "oi_lookback": int(p.get("blend_oi_lookback", 3)),
                "oi_history_for_std": int(p.get("blend_oi_history_for_std", 60)),
                "ecology_lookback": int(p.get("blend_ecology_lookback", 3)),
                "ecology_condition_spot_lead": bool(
                    p.get("blend_ecology_condition_spot_lead", True)),
                "visual_conviction_ok": float(p.get("blend_visual_conviction_ok", 0.34)),
                "visual_conviction_caution": float(p.get("blend_visual_conviction_caution", 0.67)),
                "visual_conviction_stop": float(p.get("blend_visual_conviction_stop", 1.0)),
            }
            comps = _sb.compute_components(
                kalshi_closes=closes,
                spot_closes=blend_context.get("spot_closes"),
                funding_history=blend_context.get("funding_history"),
                oi_history=blend_context.get("oi_history"),
                oi_price_history=blend_context.get("oi_price_history"),
                ecosystem=blend_context.get("ecosystem"),
                visual_review=blend_context.get("visual_review"),
                params=blend_params)
            effective_weights = _sb.conditioned_component_weights(
                {
                    "spot_lead": blend_params["w_spot_lead"],
                    "funding_fade": blend_params["w_funding_fade"],
                    "oi_pressure": blend_params["w_oi_pressure"],
                    "ecology_flow": blend_params["w_ecology_flow"],
                    "visual_trend": blend_params["w_visual"],
                },
                blend_context.get("ecosystem"), blend_params)
            blend_params["w_spot_lead"] = float(
                effective_weights.get("spot_lead", blend_params["w_spot_lead"]))
            spot_tilt = float(comps.get("spot_lead", 0.0))
            funding_tilt = float(comps.get("funding_fade", 0.0))
            oi_tilt = float(comps.get("oi_pressure", 0.0))
            ecology_tilt = float(comps.get("ecology_flow", 0.0))
            visual_tilt = float(comps.get("visual_trend", 0.0))
        except Exception:
            # Never let blend math crash the controller — fall back to MPC-only.
            spot_tilt = funding_tilt = oi_tilt = ecology_tilt = visual_tilt = 0.0

    r = [0.0] + [(closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] else 0.0
                 for i in range(1, n)]
    er = efficiency_ratio(closes, rw)
    pos = [0.0] * n
    urgency = [0.0] * n
    cur = 0.0
    latest_diag = {}
    start = max(vw, lb, rw) + 1
    for i in range(start, n):
        move = sum(r[i - lb + 1:i + 1])
        vol = _std(r[i - vw:i]) or 1e-9
        alpha = -beta * (move / vol)                      # predicted reversion
        scale = max(0.0, 1.0 - er[i] / max(er_cap, 1e-6))  # fade less in trends
        phase = "churn"
        live_disturbance = 0.0
        live_reserve = 1.0
        live_csd_like = 0.0
        if robust:
            phase = states[i] if states and i < len(states) else "churn"
            if ecosystem and i == n - 1:
                phase = ecosystem.get("phase") or phase
                live_disturbance = float(((ecosystem.get("drivers") or {}).get("disturbance")) or 0.0)
                live_reserve = float(((ecosystem.get("network_metrics") or {}).get("rel_reserve")) or 1.0)
                live_csd_like = float(((ecosystem.get("organisms") or {}).get("scores") or {}).get("immune") or 0.0)
        mod = phase_mod.get(phase, phase_mod["churn"]) if robust else {"gain": 1.0, "band": 1.0, "cap": 1.0}
        recent_vols = []
        for j in range(vw * 2, i + 1):
            recent_vols.append(_std(r[j - vw:j]))
        vol_now = _std(r[i - vw:i]) or 0.0
        vol_base = (sum(recent_vols[-32:]) / len(recent_vols[-32:])) if recent_vols[-32:] else vol_now
        vol_stress = max(0.0, (vol_now / (vol_base or 1e-9)) - 1.0)
        robust_shrink = 1.0
        if robust:
            eco_stress = max(0.0, live_disturbance) + max(0.0, 1.0 - live_reserve) + live_csd_like
            robust_shrink = 1.0 / (1.0 + robust_lambda * vol_stress
                                   + robust_disturbance_lambda * eco_stress)
        mpc_alpha_base = gain * alpha * scale
        mpc_alpha = mod["gain"] * mpc_alpha_base * robust_shrink
        # Blend on the LATEST bar only — historical bars compute MPC-only because
        # we don't have aligned blend_context for the entire historical path
        # in live use. Walk-forward validation is done in backtest_blended.py.
        if blend_enabled and blend_params and i == n - 1:
            aim_raw = (blend_params["w_mpc"] * mpc_alpha
                        + blend_params["w_spot_lead"] * spot_tilt
                        + blend_params["w_funding_fade"] * funding_tilt
                        + blend_params["w_oi_pressure"] * oi_tilt
                        + blend_params["w_ecology_flow"] * ecology_tilt
                        + blend_params["w_visual"] * visual_tilt)
        else:
            aim_raw = mpc_alpha
        cap = float(mod["cap"])
        aim = max(-cap, min(cap, aim_raw))
        urgency[i] = abs(aim_raw)
        band_i = band * float(mod["band"])
        virtual_cur_before_anchor = cur
        live_anchor_applied = False
        if i == n - 1 and live_current_fraction is not None:
            try:
                cur = float(live_current_fraction)
            except (TypeError, ValueError):
                cur = 0.0
            cur = max(-cap, min(cap, cur))
            live_anchor_applied = True
        cur_before = cur
        controller_move = "hold_band"
        if cur < aim - band_i:
            cur = aim - band_i
            controller_move = "raise_to_band"
        elif cur > aim + band_i:
            cur = aim + band_i
            controller_move = "lower_to_band"
        cur = max(-cap, min(cap, cur))
        pos[i] = cur
        if return_diagnostics and i == n - 1:
            latest_diag = {
                "phase": phase,
                "alpha_base": round(float(mpc_alpha_base), 6),
                "mpc_alpha": round(float(mpc_alpha), 6),
                "aim_raw": round(float(aim_raw), 6),
                "aim_capped": round(float(aim), 6),
                "band": round(float(band_i), 6),
                "base_band": round(float(band), 6),
                "cap": round(float(cap), 6),
                "target_before": round(float(cur_before), 6),
                "target_after": round(float(cur), 6),
                "virtual_target_before_anchor": round(float(virtual_cur_before_anchor), 6),
                "live_current_fraction": (
                    round(float(live_current_fraction), 6)
                    if live_current_fraction is not None else None
                ),
                "live_anchor_applied": live_anchor_applied,
                "urgency": round(float(abs(aim_raw)), 6),
                "controller_move": controller_move,
                "robust_shrink": round(float(robust_shrink), 6),
                "vol_stress": round(float(vol_stress), 6),
                "efficiency_ratio": round(float(er[i]), 6),
                "scale": round(float(scale), 6),
                "phase_mod": dict(mod),
                "blend_enabled": bool(blend_enabled and blend_params),
                "spot_tilt": round(float(spot_tilt), 6),
                "funding_tilt": round(float(funding_tilt), 6),
                "oi_tilt": round(float(oi_tilt), 6),
                "ecology_tilt": round(float(ecology_tilt), 6),
                "visual_tilt": round(float(visual_tilt), 6),
            }
    if return_diagnostics:
        return pos, urgency, latest_diag
    return pos, urgency


def mpc_with_aux(candles, p, blend_context: Optional[dict] = None,
                 live_current_fraction: Optional[float] = None):
    """Same as sig_mpc but also returns the per-bar urgency series |alpha|
    (the magnitude of the regime-scaled, gain-applied reversion aim). The live
    executor uses urgency to choose maker-vs-taker per cycle, and the hybrid
    backtester uses it for the same threshold.

    `blend_context` (optional): if provided AND p.get("blend_enabled") is True,
    the directional alpha components from signals_blended.py are added to
    MPC's reversion alpha with the configured weights. Schema:
        blend_context = {
            "spot_closes": List[float],     # rolling spot mid history
            "funding_history": List[float], # rolling funding rate history
            "oi_history": List[float],      # rolling OI history
            "oi_price_history": List[float], # price aligned to oi_history
        }
    Any missing key falls back to that component contributing 0.
    Validation: see backend/backtest_blended.py â€” spot-lead at w=0.30 lifted
    aggregate WF return +14.28pp and sharpe +0.85 over MPC-alone, OOS.
    """
    return _mpc_core(candles, p, blend_context=blend_context, robust=False,
                     live_current_fraction=live_current_fraction)


def sig_robust_mpc(candles, p) -> List[float]:
    return robust_mpc_with_aux(candles, p)[0]


def robust_mpc_with_aux(candles, p, blend_context: Optional[dict] = None,
                        ecosystem: Optional[dict] = None,
                        live_current_fraction: Optional[float] = None):
    """Ecology-conditioned robust MPC.

    Keeps the validated MPC structure (alpha aim + no-trade band) but treats
    the ecology phase as uncertainty: predator/exhaustion/decomposer widen the
    no-trade band and cap position size, scavenger permits more mean-reversion
    size, and current live disturbance/reserve/immune scores shrink the latest
    aim. This gives nonlinear behavior without a fragile nonlinear optimizer.
    """
    return _mpc_core(candles, p, blend_context=blend_context,
                     robust=True, ecosystem=ecosystem,
                     live_current_fraction=live_current_fraction)


def robust_mpc_with_diagnostics(candles, p, blend_context: Optional[dict] = None,
                                ecosystem: Optional[dict] = None,
                                live_current_fraction: Optional[float] = None):
    """Robust MPC plus latest-bar controller diagnostics for live/UI wiring."""
    return _mpc_core(candles, p, blend_context=blend_context,
                     robust=True, ecosystem=ecosystem,
                     return_diagnostics=True,
                     live_current_fraction=live_current_fraction)


VARIANTS: dict[str, Callable] = {
    "mpc": sig_mpc,
    "robust_mpc": sig_robust_mpc,
    "mr_edge": sig_mr_edge,
    "regime": sig_regime, "meanrev": sig_meanrev, "trend": sig_trend,
    "breakout": sig_breakout, "momentum": sig_momentum,
}


# --------------------------------------------------------------- simulate
def simulate(candles, pos, leverage=5.8, fee_pct=0.0, funding_per_bar=0.0) -> dict:
    """Simulate a target-position series with 1-bar execution lag (no look-ahead)."""
    equity = 1.0; peak = 1.0; maxdd = 0.0
    prevpos = 0.0; trades = 0; marked = 0; wins = 0
    rets = []
    for i in range(len(candles) - 1):
        target = pos[i]
        if target != prevpos:
            equity -= equity * fee_pct * abs(target - prevpos) * leverage
            trades += 1
            prevpos = target
        px = candles[i]["close"]; nxt = candles[i + 1]["close"]
        if px > 0 and prevpos != 0:
            r = (nxt - px) / px * prevpos * leverage - funding_per_bar * abs(prevpos)
            before = equity
            equity *= (1 + r)
            rets.append(r); marked += 1
            if equity > before:
                wins += 1
        peak = max(peak, equity)
        maxdd = max(maxdd, (peak - equity) / peak if peak else 0.0)
    sharpe = 0.0
    if len(rets) > 2:
        m = sum(rets) / len(rets)
        sd = (sum((x - m) ** 2 for x in rets) / len(rets)) ** 0.5
        sharpe = (m / sd) * (len(rets) ** 0.5) if sd else 0.0
    return {
        "return_pct": round((equity - 1) * 100, 3),
        "total_return_pct": round((equity - 1) * 100, 3),
        "max_drawdown_pct": round(maxdd * 100, 3),
        "max_dd_pct": round(maxdd * 100, 3),
        "trades": trades, "bars": marked,
        "win_rate": round(wins / marked, 3) if marked else 0.0,
        "sharpe": round(sharpe, 3),
        "final_equity": round(equity, 4),
        "score": round((equity - 1) - maxdd * 1.5, 5),
    }


def urgency_for(variant: str, candles, p) -> List[float]:
    """Per-bar urgency used by the maker/taker switch. For MPC this is |alpha|
    (the regime-scaled reversion aim magnitude). For other variants we use the
    per-bar magnitude of the position change as a proxy."""
    if variant == "mpc":
        return mpc_with_aux(candles, p)[1]
    if variant == "robust_mpc":
        return robust_mpc_with_aux(candles, p)[1]
    fn = VARIANTS.get(variant)
    if not fn:
        return [0.0] * len(candles)
    pos = fn(candles, p)
    out = [0.0] * len(pos)
    prev = 0.0
    for i, t in enumerate(pos):
        out[i] = abs(t - prev)
        prev = t
    return out


def simulate_hybrid(candles, pos, urgency, *,
                    k: float = float("inf"),
                    chase_n: int = 0,
                    half_spread_bps: float = 7.0,
                    fee_bps: float = 0.0,
                    leverage: float = 5.8) -> dict:
    """Honest hybrid maker/taker fill simulator.

    For each desired position change at bar i with urgency u[i]:
      - if u[i] >= k -> TAKER: cross immediately at close, pay half-spread.
      - else -> MAKER: rest a limit at this bar's close. Fills on a later bar j
        only if that bar's range trades through the limit
        (low <= limit for a buy / high >= limit for a sell). If still unfilled
        after `chase_n` bars, we either chase (cross + pay half-spread) when
        chase_n > 0 finite, or count as MISSED and drop the order.

    Maker fills are NOT free — they're free of *spread* cost, but we still debit
    `fee_bps` per side so any non-zero maker fee can be modelled. Default fee=0
    matches Kalshi's current promo. The 1-bar execution lag and the requirement
    that the bar's range actually reach the limit prevents the "free fill"
    fiction that made naive maker backtests so optimistic.
    """
    n = len(candles)
    if n < 5:
        return {"error": "not enough candles"}
    hs = half_spread_bps / 1e4
    fee = fee_bps / 1e4
    equity = 1.0; peak = 1.0; maxdd = 0.0
    prevpos = 0.0
    pending = None   # {"target": float, "limit": float, "age": int, "submit_bar": int}
    trades_maker = 0; trades_taker = 0; trades_chase = 0; missed = 0
    rets = []
    wins = 0; marked = 0

    def _apply_cost(delta, cross: bool):
        nonlocal equity
        cost = (hs if cross else 0.0) + fee
        if cost and delta:
            equity -= equity * cost * abs(delta) * leverage

    for i in range(n - 1):
        bar = candles[i]
        nxt = candles[i + 1]
        target = pos[i]
        u = urgency[i] if i < len(urgency) else 0.0

        # 1) Try to fill any pending maker order on THIS bar (price must trade through limit)
        if pending is not None:
            delta_p = pending["target"] - prevpos
            limit = pending["limit"]
            filled = False
            if delta_p > 0 and bar["low"] <= limit:
                filled = True
            elif delta_p < 0 and bar["high"] >= limit:
                filled = True
            if filled:
                _apply_cost(delta_p, cross=False)
                prevpos = pending["target"]
                trades_maker += 1
                pending = None
            else:
                pending["age"] += 1
                # exceeded chase budget -> chase by crossing (taker)
                if pending["age"] > chase_n:
                    delta_c = pending["target"] - prevpos
                    _apply_cost(delta_c, cross=True)
                    prevpos = pending["target"]
                    trades_chase += 1
                    pending = None

        # 2) Generate a new order if signal differs from current position
        if pending is None and target != prevpos:
            delta = target - prevpos
            if u >= k:
                _apply_cost(delta, cross=True)
                prevpos = target
                trades_taker += 1
            else:
                pending = {"target": target, "limit": bar["close"],
                           "age": 0, "submit_bar": i}
        elif pending is not None and target != pending["target"]:
            # signal direction/size changed before fill: cancel old, decide afresh
            if target == prevpos:
                missed += 1
                pending = None
            else:
                delta = target - prevpos
                if u >= k:
                    _apply_cost(delta, cross=True)
                    prevpos = target
                    trades_taker += 1
                    pending = None
                else:
                    pending = {"target": target, "limit": bar["close"],
                               "age": 0, "submit_bar": i}

        # 3) Mark-to-market over the next bar at the position we actually hold
        px = bar["close"]; nx = nxt["close"]
        if px > 0 and prevpos != 0:
            r = (nx - px) / px * prevpos * leverage
            before = equity
            equity *= (1 + r)
            rets.append(r); marked += 1
            if equity > before:
                wins += 1
        peak = max(peak, equity)
        maxdd = max(maxdd, (peak - equity) / peak if peak else 0.0)

    sharpe = 0.0
    if len(rets) > 2:
        m = sum(rets) / len(rets)
        sd = (sum((x - m) ** 2 for x in rets) / len(rets)) ** 0.5
        sharpe = (m / sd) * (len(rets) ** 0.5) if sd else 0.0

    trades_total = trades_maker + trades_taker + trades_chase
    return {
        "return_pct": round((equity - 1) * 100, 3),
        "max_dd_pct": round(maxdd * 100, 3),
        "trades": trades_total,
        "trades_maker": trades_maker,
        "trades_taker": trades_taker,
        "trades_chase": trades_chase,
        "missed": missed,
        "fill_rate_maker": round(trades_maker / max(1, trades_maker + trades_chase + missed), 3),
        "bars": marked,
        "win_rate": round(wins / marked, 3) if marked else 0.0,
        "sharpe": round(sharpe, 3),
        "final_equity": round(equity, 4),
        "score": round((equity - 1) - maxdd * 1.5, 5),
    }


def confidence_for(variant: str, candles, p) -> float:
    """A bounded [0,1] confidence for the latest bar's signal strength."""
    closes = [c["close"] for c in candles]
    if len(closes) < 5:
        return 0.0
    if variant in ("mpc", "robust_mpc", "mr_edge"):
        s = VARIANTS[variant](candles, p)
        return min(1.0, abs(s[-1])) if s else 0.0
    if variant in ("meanrev", "regime"):
        r = rsi_series(closes, int(p["rsi_period"]))[-1]
        if r is None:
            return 0.3
        dist = max(p["rsi_oversold"] - r, r - p["rsi_overbought"], 0.0)
        return max(0.2, min(1.0, 0.4 + dist / 25.0))
    if variant == "trend":
        ef = ema_series(closes, int(p["ema_fast"]))[-1]
        es = ema_series(closes, int(p["ema_slow"]))[-1]
        sep = abs(ef - es) / closes[-1] * 100 if closes[-1] else 0
        return max(0.2, min(1.0, sep / max(p["trend_gate"], 1e-6) * 0.5))
    return 0.5
