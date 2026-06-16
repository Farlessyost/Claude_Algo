"""Backtest = run the selected signal variant over a candle series via the
shared simulator. Same code path the live strategy uses, so results transfer."""
from __future__ import annotations

from typing import List

from . import signals


def run_backtest(candles: List[dict], params: dict, variant: str = "regime",
                 leverage: float = 5.8, fee_pct: float = 0.0) -> dict:
    if len(candles) < 60:
        return {"error": "not enough candles", "bars": len(candles)}
    fn = signals.VARIANTS.get(variant, signals.VARIANTS["regime"])
    pos = fn(candles, params)
    rep = signals.simulate(candles, pos, leverage=leverage, fee_pct=fee_pct)
    rep["variant"] = variant
    rep["bars"] = len(candles)
    # equity curve for the UI
    eq = 1.0; curve = []; prev = 0.0
    for i in range(len(candles) - 1):
        t = pos[i]
        if t != prev:
            eq -= eq * fee_pct * abs(t - prev) * leverage; prev = t
        px = candles[i]["close"]; nxt = candles[i + 1]["close"]
        if px > 0 and prev != 0:
            eq *= (1 + (nxt - px) / px * prev * leverage)
        curve.append(round(eq, 6))
    rep["equity_curve"] = curve[-120:]
    return rep
