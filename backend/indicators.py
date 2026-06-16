"""Pure-python technical indicators. Inputs are lists of floats (close prices)
or OHLC dicts. No numpy dependency so the app stays trivial to install."""
from __future__ import annotations

from typing import List, Optional


def ema(values: List[float], period: int) -> List[float]:
    if not values:
        return []
    k = 2.0 / (period + 1.0)
    out = [values[0]]
    for v in values[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def rsi(closes: List[float], period: int = 14) -> Optional[float]:
    if len(closes) < period + 1:
        return None
    gains = 0.0
    losses = 0.0
    for i in range(1, period + 1):
        d = closes[i] - closes[i - 1]
        gains += max(d, 0.0)
        losses += max(-d, 0.0)
    avg_gain = gains / period
    avg_loss = losses / period
    for i in range(period + 1, len(closes)):
        d = closes[i] - closes[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(d, 0.0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0.0)) / period
    if avg_loss == 0:
        return 100.0
    rs = avg_gain / avg_loss
    return 100.0 - (100.0 / (1.0 + rs))


def roc(closes: List[float], period: int) -> Optional[float]:
    """Rate of change in percent over `period` bars."""
    if len(closes) < period + 1:
        return None
    past = closes[-period - 1]
    if past == 0:
        return None
    return (closes[-1] - past) / past * 100.0


def atr_pct(candles: List[dict], period: int = 14) -> Optional[float]:
    """Average true range as a percent of price."""
    if len(candles) < period + 1:
        return None
    trs = []
    for i in range(1, len(candles)):
        h = candles[i]["high"]
        l = candles[i]["low"]
        pc = candles[i - 1]["close"]
        tr = max(h - l, abs(h - pc), abs(l - pc))
        trs.append(tr)
    if len(trs) < period:
        return None
    atr = sum(trs[-period:]) / period
    last_close = candles[-1]["close"]
    if last_close == 0:
        return None
    return atr / last_close * 100.0


def stdev_pct(closes: List[float], period: int = 20) -> Optional[float]:
    if len(closes) < period:
        return None
    window = closes[-period:]
    mean = sum(window) / period
    var = sum((c - mean) ** 2 for c in window) / period
    if mean == 0:
        return None
    return (var ** 0.5) / mean * 100.0
