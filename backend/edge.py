"""Mean-reversion edge for KXBTCPERP, built from the EDA findings:
  - short-horizon returns mean-revert (AR(1) coef strongly negative);
  - reversion is stronger in higher-vol regimes;
  - the most extreme moves do NOT revert (cap the fade).

Framing (MPC-lite): predict next-bar return as an AR model, take a position
proportional to predicted edge, but only when predicted edge > cost (deadband),
and cap when the move is so extreme it's likely a trend (continuation).
A fuzzy blend modulates aggressiveness by volatility regime.

This module is self-contained for the live engine to import.
"""
from __future__ import annotations

import math
from typing import List

PARAMS = {
    "tf": 3,            # base bars per decision (aggregation minutes)
    "lookback": 1,      # bars of return to fade
    "vol_win": 24,      # bars for vol estimate
    "beta": 0.18,       # AR mean-reversion strength (|autocorr| ~0.18)
    "k": 1.6,           # position aggressiveness on z-score
    "z_cap": 2.8,       # above this |z|, treat as trend -> cut fade
    "deadband_bps": 1.0,  # require predicted edge above this to act
    "vol_gate": 0.6,    # only scale up when vol >= this * median vol
}


def _ret(closes: List[float]) -> List[float]:
    return [0.0] + [(closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] else 0.0
                    for i in range(1, len(closes))]


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def target_series(closes: List[float], p: dict) -> List[float]:
    """Vectorized target position in [-1,1] for each bar (no look-ahead:
    pos[i] uses returns up to i, earns ret[i+1])."""
    r = _ret(closes)
    pos = [0.0] * len(closes)
    vw = int(p["vol_win"])
    lb = int(p["lookback"])
    for i in range(max(vw, lb) + 1, len(closes)):
        recent = r[i - lb + 1:i + 1]
        move = sum(recent)
        vol = _std(r[i - vw:i]) or 1e-9
        z = move / vol
        # AR prediction of next return (mean reversion): edge = -beta * move
        pred = -p["beta"] * move
        # extreme moves: likely continuation -> fade less (cap by z)
        damp = 1.0
        if abs(z) > p["z_cap"]:
            damp = max(0.0, 1.0 - (abs(z) - p["z_cap"]) / p["z_cap"])
        # fuzzy vol modulation: more aggressive when vol elevated
        volmod = min(1.5, max(0.4, vol / (_std(r[max(0, i - 240):i]) or vol)))
        raw = -math.tanh(p["k"] * z) * damp * volmod
        # deadband: act only if predicted edge beats cost
        if abs(pred) * 1e4 < p["deadband_bps"]:
            raw = 0.0
        pos[i] = max(-1.0, min(1.0, raw))
    return pos


def live_target(closes: List[float], p: dict) -> float:
    """Latest desired position in [-1,1] for the live engine."""
    s = target_series(closes, p)
    return s[-1] if s else 0.0
