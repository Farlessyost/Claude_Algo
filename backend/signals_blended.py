"""Directional alpha components that blend WITH the validated MPC mean-reversion
edge, not against it.

The current architecture treats MPC's target_fraction as the only directional
signal; everything else (ecology, CSD, visual review, LLM) can only REDUCE
or BLOCK that signal. That works in the validated regime (quiet, mean-reverting)
and bleeds in others (sustained trends). This module adds independent
directional contributors that can offset MPC when the market regime makes
pure reversion costly:

  spot_lead_tilt    — Coinbase BTC-USD leads Kalshi by ~3 min (validated in
                       the LSTM/CfC experiment, IC ~0.46-0.54). When spot has
                       moved but Kalshi hasn't, tilt toward the catch-up.
  funding_fade_tilt — Hyperliquid funding rate. Persistent + funding (longs
                       crowded) -> bias short. Classic perp-arb signal.
  oi_pressure_tilt  — Hyperliquid OI delta with sign preserved (the current
                       ecology only consumes |OI delta| as liq_proxy and
                       discards the direction). Combined with recent price
                       direction to map to a meaningful tilt.
  visual_trend_tilt — the periodic visual chart review (backend/visual_review.py).
                       An LLM looks at the recent candle chart and returns a
                       {trend, concern} read; we map trend->direction and
                       concern->conviction. A TREND contributor (with the move),
                       so like spot_lead it offsets MPC when reversion is costly.
                       Cannot be backtested via backtest_blended.py (no historical
                       LLM reviews exist) — weighted as the dominant directional
                       driver (w_visual default 2.0, 2x MPC); watch it live.

All three return a z-scored scalar that can be ADDED (weighted) to MPC's
alpha. Weights default to small values so MPC stays dominant; the backtest
in backend/backtest_blended.py validates whether non-zero weights actually
improve OOS performance.

NONE of these are wired into live trading by this module. signals.py and
strategy.py keep their existing behaviour. To enable, use the blended
signal helpers here from a future strategy variant after the backtest
passes.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple


# ----------------------------------------------------------- math helpers
def _std(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = sum(xs) / n
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def _log_return(later: float, earlier: float) -> float:
    if later <= 0 or earlier <= 0:
        return 0.0
    return math.log(later / earlier)


# ----------------------------------------------------------- components
def spot_lead_tilt(spot_closes: Sequence[float],
                    lookback: int = 3,
                    history_for_std: int = 60,
                    z_cap: float = 3.5) -> float:
    """Spot-leads-Kalshi tilt.

    Returns a z-scored log-return of spot over `lookback` bars, capped to
    [-z_cap, z_cap]. Positive = spot has moved up faster than its recent
    distribution -> Kalshi will follow -> bias long.

    Sign convention matches MPC's: positive = long bias, negative = short.
    But where MPC's alpha is REVERSION (against recent move), this is
    TREND (with recent move), so the two are independent contributors.
    """
    n = len(spot_closes)
    if n < lookback + 8:
        return 0.0
    ret_now = _log_return(spot_closes[-1], spot_closes[-1 - lookback])
    # Build the distribution of same-lookback returns over recent history.
    recent = spot_closes[-min(n, history_for_std + lookback):]
    if len(recent) < lookback + 8:
        return 0.0
    rets = []
    for i in range(lookback, len(recent)):
        rets.append(_log_return(recent[i], recent[i - lookback]))
    # Proper z-score: mean-subtract then divide by std. Without the mean term,
    # a market in steady drift fires a constant non-zero tilt every cycle.
    mu = _mean(rets)
    std = _std(rets)
    if std <= 0:
        return 0.0
    return _clip((ret_now - mu) / std, -z_cap, z_cap)


def funding_fade_tilt(funding_history: Sequence[float],
                       history_for_std: int = 60,
                       z_cap: float = 2.5,
                       persistence_threshold: float = 0.5) -> float:
    """Funding-rate fade tilt.

    When funding has been persistently positive (longs are paying shorts),
    the position is crowded long and tends to fade. Returns a NEGATIVE
    tilt in that case (bias short). Symmetric for negative funding.

    `persistence_threshold` is the minimum |z-score| needed before any tilt
    is emitted — below this, the funding signal is treated as noise.
    """
    n = len(funding_history)
    if n < 8:
        return 0.0
    recent = funding_history[-min(n, history_for_std):]
    if len(recent) < 8:
        return 0.0
    m = _mean(recent)
    s = _std(recent)
    if s <= 0:
        return 0.0
    cur = funding_history[-1]
    z = (cur - m) / s
    if abs(z) < persistence_threshold:
        return 0.0
    # Fade direction: positive funding (longs crowded) -> short bias = NEGATIVE
    return -_clip(z, -z_cap, z_cap)


def oi_pressure_tilt(price_history: Sequence[float],
                      oi_history: Sequence[float],
                      lookback: int = 3,
                      history_for_std: int = 60,
                      z_cap: float = 2.5) -> float:
    """OI-pressure tilt with sign preserved.

    Classifies the OI/price joint movement over `lookback` cycles:
       price up + OI up   : new longs entering   -> trend continues up
       price up + OI down : short covering       -> trend continues up
       price down + OI up : new shorts entering  -> trend continues down
       price down + OI down : longs unwinding    -> trend continues down

    All four cases yield a tilt in the direction of recent price.
    The MAGNITUDE comes from how unusual the |OI delta| is relative to its
    recent distribution (z-scored). Direction comes from the price sign.
    """
    n = min(len(price_history), len(oi_history))
    if n < lookback + 8:
        return 0.0
    p_now = price_history[-1]
    p_then = price_history[-1 - lookback]
    oi_now = oi_history[-1]
    oi_then = oi_history[-1 - lookback]
    if p_then <= 0 or oi_then <= 0:
        return 0.0
    price_ret = _log_return(p_now, p_then)
    oi_delta_pct = (oi_now - oi_then) / oi_then
    # z-score the OI delta against recent history
    recent_oi = oi_history[-min(n, history_for_std + lookback):]
    if len(recent_oi) < lookback + 8:
        return 0.0
    deltas = []
    for i in range(lookback, len(recent_oi)):
        prev = recent_oi[i - lookback]
        if prev > 0:
            deltas.append((recent_oi[i] - prev) / prev)
    if len(deltas) < 8:
        return 0.0
    mu = _mean(deltas)
    std = _std(deltas)
    if std <= 0:
        return 0.0
    # Proper z-score: how unusual is this delta vs the recent distribution,
    # not just "how big is the absolute value vs the recent volatility".
    oi_z = abs((oi_delta_pct - mu) / std)
    if oi_z < 0.5:
        return 0.0  # too quiet to act on
    # Sign of price_ret defines direction; magnitude scaled by oi_z
    direction = 1.0 if price_ret > 0 else -1.0 if price_ret < 0 else 0.0
    return _clip(direction * oi_z, -z_cap, z_cap)


def visual_trend_tilt(trend: Optional[str], concern: Optional[str],
                       conviction_ok: float = 0.34,
                       conviction_caution: float = 0.67,
                       conviction_stop: float = 1.0,
                       z_cap: float = 1.0) -> float:
    """Directional tilt from the periodic visual chart review.

    The visual review has an LLM read the recent candle chart and return
    {trend, concern, note}. `trend` is its read of the multi-bar direction the
    price action shows; `concern` is how visually obvious / alarming that move
    is for a reversion bot.

    Mapping:
        trend up        -> +  (bias long)
        trend down      -> -  (bias short)
        sideways/choppy -> 0
    Magnitude ("conviction") scales with `concern`: OK is a mild lean, CAUTION
    moderate, STOP the chart-reader shouting that an obvious sustained move is
    underway. STOP also fires the entry gate (block_entry_if_stop); the tilt
    and the gate are complementary — the gate is a hard block on NEW entries,
    the tilt is a soft push the rest of the time.

    Sign convention matches MPC's (positive = long bias). Returns 0.0 for an
    unknown/sideways trend or a missing review. This is a TREND contributor
    (with the move), independent of MPC's reversion alpha — exactly the kind
    of offset this module exists to provide.
    """
    t = (trend or "").strip().lower()
    direction = 1.0 if t == "up" else -1.0 if t == "down" else 0.0
    if direction == 0.0:
        return 0.0
    c = (concern or "OK").strip().upper()
    conviction = {
        "OK": conviction_ok,
        "CAUTION": conviction_caution,
        "STOP": conviction_stop,
    }.get(c, conviction_ok)
    return _clip(direction * conviction, -z_cap, z_cap)


def ecology_flow_tilt(kalshi_closes: Sequence[float],
                      ecosystem: Optional[dict],
                      lookback: int = 3,
                      z_cap: float = 2.0) -> float:
    """Directional tilt from the trophic network state.

    This is not another raw price signal. It converts the live ecological
    state into a small alpha term:
      - predator/exhaustion: continuation, follow recent pressure
      - scavenger/decomposer: reversion, fade current stretch
      - producer/churn: mostly neutral

    Positive = long bias, negative = short bias.
    """
    eco = ecosystem or {}
    phase = str(eco.get("phase") or "").lower()
    drivers = eco.get("drivers") or {}
    orgs = ((eco.get("organisms") or {}).get("scores") or {})
    if not kalshi_closes or len(kalshi_closes) < lookback + 8:
        return 0.0

    stretch_z = float(drivers.get("stretch_z") or 0.0)
    disturbance = float(drivers.get("disturbance_projected",
                         drivers.get("disturbance") or 0.0) or 0.0)
    reserve = float(((eco.get("network_metrics") or {}).get("rel_reserve")) or 1.0)
    predator = float(orgs.get("predator") or 0.0)
    scavenger = float(orgs.get("scavenger") or 0.0)
    decomposer = float(orgs.get("decomposer") or 0.0)

    p_now = float(kalshi_closes[-1])
    p_then = float(kalshi_closes[-1 - lookback])
    recent_dir = 1.0 if p_now > p_then else -1.0 if p_now < p_then else 0.0
    stretch_dir = -1.0 if stretch_z > 0 else 1.0 if stretch_z < 0 else 0.0

    if phase in ("predator", "exhaustion"):
        conviction = 0.35 + 0.35 * predator + 0.20 * min(1.0, disturbance) + 0.10 * max(0.0, 1.0 - reserve)
        return _clip(recent_dir * conviction, -z_cap, z_cap)
    if phase in ("scavenger", "decomposer"):
        snap = 0.40 * scavenger + 0.25 * decomposer + 0.25 * min(1.0, abs(stretch_z) / 3.0)
        # In disturbed snap-back zones, keep this useful but not dominant.
        conviction = snap * (1.0 - 0.35 * min(1.0, disturbance))
        return _clip(stretch_dir * conviction, -z_cap, z_cap)
    return 0.0


def conditioned_component_weights(weights: dict, ecosystem: Optional[dict],
                                  params: Optional[dict] = None) -> dict:
    """Scale raw component weights by ecology so alpha terms stop fighting.

    Spot lead is a useful lead/lag trend term, but the live loss mode has been
    spot lead overriding the reversion controller outside predator cascades.
    Treat it as a predator/exhaustion continuation tool, not as a universal
    alpha. Funding/OI are left at their configured weights.
    """
    p = params or {}
    out = dict(weights)
    if not bool(p.get("ecology_condition_spot_lead", True)):
        return out
    eco = ecosystem or {}
    phase = str(eco.get("phase") or "").lower()
    drivers = eco.get("drivers") or {}
    disturbance = float(drivers.get("disturbance_projected",
                         drivers.get("disturbance") or 0.0) or 0.0)
    orgs = ((eco.get("organisms") or {}).get("scores") or {})
    scavenger = float(orgs.get("scavenger") or 0.0)
    predator = float(orgs.get("predator") or 0.0)
    if phase == "predator":
        mult = 0.85 + 0.35 * min(1.0, predator)
    elif phase == "exhaustion":
        mult = 0.35 + 0.25 * min(1.0, predator)
    elif phase in ("scavenger", "decomposer"):
        # In snap-back/restoration, spot lead should be a context hint only.
        # Higher scavenger score means the trend-lead term is more likely to
        # fight the intended reversion.
        mult = 0.10 + 0.10 * min(1.0, disturbance) - 0.08 * min(1.0, scavenger)
    elif phase in ("producer", "churn"):
        mult = 0.10
    else:
        mult = 0.05
    out["spot_lead"] = float(out.get("spot_lead") or 0.0) * _clip(mult, 0.02, 1.20)
    return out


# ----------------------------------------------------------- blend
DEFAULT_BLEND = {
    "w_mpc": 1.0,            # weight on the validated reversion alpha
    "w_spot_lead": 0.30,     # weight on spot-lead trend tilt
    "w_funding_fade": 0.15,  # weight on funding-fade (small until validated)
    "w_oi_pressure": 0.15,   # weight on OI-pressure (small until validated)
    "w_visual": 0.0,         # visual-review trend REMOVED from the blend
    "w_ecology_flow": 0.0,    # live network-state alpha; default-zero until WF validation
    # Component parameters
    "spot_lookback": 3,
    "spot_history_for_std": 60,
    "funding_history_for_std": 60,
    "funding_persistence_threshold": 0.5,
    "oi_lookback": 3,
    "oi_history_for_std": 60,
    "ecology_lookback": 3,
    "ecology_condition_spot_lead": True,
    # Visual-trend conviction mapping (concern -> magnitude)
    "visual_conviction_ok": 0.34,
    "visual_conviction_caution": 0.67,
    "visual_conviction_stop": 1.0,
}


def compute_components(kalshi_closes: Sequence[float],
                        spot_closes: Optional[Sequence[float]] = None,
                        funding_history: Optional[Sequence[float]] = None,
                        oi_history: Optional[Sequence[float]] = None,
                        oi_price_history: Optional[Sequence[float]] = None,
                        ecosystem: Optional[dict] = None,
                        visual_review: Optional[dict] = None,
                        params: Optional[dict] = None) -> dict:
    """Compute every directional component on the latest available data.

    Each component returns 0.0 when its required input is missing. Returns a
    dict with: spot_lead, funding_fade, oi_pressure, ecology_flow,
    visual_trend. The caller blends these with MPC's alpha via weighted sum.

    `visual_review` is the latest {trend, concern, ...} dict from
    backend/visual_review.py (or None when unavailable / stale — the caller is
    responsible for the staleness check).

    `oi_price_history` is the price series ALIGNED to oi_history's cadence
    (HISTORY pushes from multiasset.snapshot). When OI samples once per
    cycle and price samples once per cycle, the two series share the same
    timeline so passing `kalshi_closes[-len(oi_history):]` works.
    """
    p = dict(DEFAULT_BLEND); p.update(params or {})
    out = {"spot_lead": 0.0, "funding_fade": 0.0,
           "oi_pressure": 0.0, "ecology_flow": 0.0, "visual_trend": 0.0}
    if spot_closes:
        out["spot_lead"] = spot_lead_tilt(
            spot_closes, lookback=int(p["spot_lookback"]),
            history_for_std=int(p["spot_history_for_std"]))
    if funding_history:
        out["funding_fade"] = funding_fade_tilt(
            funding_history,
            history_for_std=int(p["funding_history_for_std"]),
            persistence_threshold=float(p["funding_persistence_threshold"]))
    if oi_history and oi_price_history:
        out["oi_pressure"] = oi_pressure_tilt(
            oi_price_history, oi_history,
            lookback=int(p["oi_lookback"]),
            history_for_std=int(p["oi_history_for_std"]))
    if ecosystem:
        out["ecology_flow"] = ecology_flow_tilt(
            kalshi_closes, ecosystem,
            lookback=int(p.get("ecology_lookback", 3)))
    if visual_review:
        out["visual_trend"] = visual_trend_tilt(
            visual_review.get("trend"), visual_review.get("concern"),
            conviction_ok=float(p["visual_conviction_ok"]),
            conviction_caution=float(p["visual_conviction_caution"]),
            conviction_stop=float(p["visual_conviction_stop"]))
    return out


def latest_alpha_decomposition(candles: list, blend_context: Optional[dict],
                                 params: dict) -> dict:
    """Compute the latest bar's blended-alpha decomposition for display.

    Mirrors the math inside signals.mpc_with_aux but returns the per-piece
    contributions (raw + weighted + final blend) so the UI can render an
    alpha decomposition diagram. Returns zeros when the data is insufficient
    or blending is disabled in `params`.
    """
    out = {
        "weights": {"mpc": 0.0, "spot_lead": 0.0, "funding_fade": 0.0,
                     "oi_pressure": 0.0, "ecology_flow": 0.0,
                     "visual_trend": 0.0},
        "raw":     {"mpc": 0.0, "spot_lead": 0.0, "funding_fade": 0.0,
                     "oi_pressure": 0.0, "ecology_flow": 0.0,
                     "visual_trend": 0.0},
        "parts":   {"mpc": 0.0, "spot_lead": 0.0, "funding_fade": 0.0,
                     "oi_pressure": 0.0, "ecology_flow": 0.0,
                     "visual_trend": 0.0},
        "blended": 0.0,
        "enabled": False,
        "diagnostics": {},
    }
    if not candles or len(candles) < 5:
        return out
    if not bool(params.get("blend_enabled", False)):
        return out

    closes = [c["close"] for c in candles]
    n = len(closes)
    vw = int(params.get("vol_win", 12))
    lb = int(params.get("lookback", 2))
    rw = int(params.get("regime_win", 8))
    beta = float(params.get("beta", 0.18))
    gain = float(params.get("gain", 1.0))
    er_cap = float(params.get("er_cap", 1.0))

    from . import signals as _signals
    r = [0.0] + [(closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] else 0
                  for i in range(1, n)]
    er = _signals.efficiency_ratio(closes, rw)
    i = n - 1
    if i < max(vw, lb, rw):
        return out
    move = sum(r[i - lb + 1:i + 1])
    vol = _signals._std(r[i - vw:i]) or 1e-9
    scale = max(0.0, 1.0 - er[i] / max(er_cap, 1e-6))
    mpc_alpha = -beta * (move / vol) * scale * gain

    # Component params (mirror what mpc_with_aux uses)
    comp_params = {
        "spot_lookback": int(params.get("blend_spot_lookback", 3)),
        "spot_history_for_std": int(params.get("blend_spot_history_for_std", 60)),
        "funding_history_for_std": int(params.get("blend_funding_history_for_std", 60)),
        "funding_persistence_threshold": float(
            params.get("blend_funding_persistence_threshold", 0.5)),
        "oi_lookback": int(params.get("blend_oi_lookback", 3)),
        "oi_history_for_std": int(params.get("blend_oi_history_for_std", 60)),
        "ecology_lookback": int(params.get("blend_ecology_lookback", 3)),
        "ecology_condition_spot_lead": bool(params.get("blend_ecology_condition_spot_lead", True)),
        "visual_conviction_ok": float(params.get("blend_visual_conviction_ok", 0.34)),
        "visual_conviction_caution": float(params.get("blend_visual_conviction_caution", 0.67)),
        "visual_conviction_stop": float(params.get("blend_visual_conviction_stop", 1.0)),
    }
    comps = compute_components(
        kalshi_closes=closes,
        spot_closes=(blend_context or {}).get("spot_closes"),
        funding_history=(blend_context or {}).get("funding_history"),
        oi_history=(blend_context or {}).get("oi_history"),
        oi_price_history=(blend_context or {}).get("oi_price_history"),
        ecosystem=(blend_context or {}).get("ecosystem"),
        visual_review=(blend_context or {}).get("visual_review"),
        params=comp_params)

    weights = {
        "mpc":          float(params.get("blend_w_mpc", 1.0)),
        "spot_lead":    float(params.get("blend_w_spot_lead", 0.30)),
        "funding_fade": float(params.get("blend_w_funding_fade", 0.0)),
        "oi_pressure":  float(params.get("blend_w_oi_pressure", 0.0)),
        "ecology_flow": float(params.get("blend_w_ecology_flow", 0.0)),
        "visual_trend": float(params.get("blend_w_visual", 0.0)),
    }
    weights = conditioned_component_weights(
        weights, (blend_context or {}).get("ecosystem"), comp_params)
    raw = {
        "mpc": mpc_alpha,
        "spot_lead": float(comps.get("spot_lead", 0.0)),
        "funding_fade": float(comps.get("funding_fade", 0.0)),
        "oi_pressure": float(comps.get("oi_pressure", 0.0)),
        "ecology_flow": float(comps.get("ecology_flow", 0.0)),
        "visual_trend": float(comps.get("visual_trend", 0.0)),
    }
    parts = {k: round(weights[k] * raw[k], 6) for k in raw}
    blended = sum(parts.values())
    ctx = blend_context or {}
    diagnostics = {
        "spot_history_n": len(ctx.get("spot_closes") or []),
        "funding_history_n": len(ctx.get("funding_history") or []),
        "oi_history_n": len(ctx.get("oi_history") or []),
        "oi_price_history_n": len(ctx.get("oi_price_history") or []),
        "phase": ((ctx.get("ecosystem") or {}).get("phase")),
        "muted_components": [
            k for k, v in raw.items()
            if k != "mpc" and abs(float(v or 0.0)) >= 0.05
            and abs(float(weights.get(k) or 0.0)) < 1e-9
        ],
    }
    return {
        "weights": weights,
        "raw": {k: round(v, 6) for k, v in raw.items()},
        "parts": parts,
        "blended": round(blended, 6),
        "enabled": True,
        "diagnostics": diagnostics,
    }


def blend_with_mpc(mpc_alpha: float, components: dict,
                    params: Optional[dict] = None) -> Tuple[float, dict]:
    """Weighted sum of MPC's alpha + the directional components.

    Returns (blended_alpha, decomposition). Decomposition shows the
    contribution of each piece so the caller can log it for diagnostics.
    """
    p = dict(DEFAULT_BLEND); p.update(params or {})
    parts = {
        "mpc":          float(p["w_mpc"]) * float(mpc_alpha or 0.0),
        "spot_lead":    float(p["w_spot_lead"]) * float(components.get("spot_lead") or 0.0),
        "funding_fade": float(p["w_funding_fade"]) * float(components.get("funding_fade") or 0.0),
        "oi_pressure":  float(p["w_oi_pressure"]) * float(components.get("oi_pressure") or 0.0),
        "ecology_flow": float(p["w_ecology_flow"]) * float(components.get("ecology_flow") or 0.0),
        "visual_trend": float(p["w_visual"]) * float(components.get("visual_trend") or 0.0),
    }
    blended = sum(parts.values())
    return blended, parts
