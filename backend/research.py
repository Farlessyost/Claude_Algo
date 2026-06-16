"""Strategy research harness for KXBTCPERP.

Goal: find a strategy/timeframe with a *real* out-of-sample edge after costs,
before any live trading. Key discipline:
  - Tune parameters on a TRAIN slice (older data), report on a TEST slice (newer).
  - Model taker fees on every position change + a funding proxy while in market.
  - Compare variants by TEST (out-of-sample) performance, not in-sample.

Run:  python -m backend.research [iterations]
"""
from __future__ import annotations

import random
import sys
from typing import Callable, List, Optional

from . import history
from .config import load_credentials
from .kalshi_client import KalshiClient
from .store import STORE

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
# Each variant builds a target-position series in [-1,1] from candles+params.
def sig_trend(candles, p):
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
            pos[i] = 0.0; continue
        if ef[i] > es[i] and closes[i] > et[i]:
            pos[i] = 1.0
        elif ef[i] < es[i] and closes[i] < et[i]:
            pos[i] = -1.0
    return pos


def sig_breakout(candles, p):
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


def sig_meanrev(candles, p):
    closes = [c["close"] for c in candles]
    rsi = rsi_series(closes, int(p["rsi_period"]))
    ef = ema_series(closes, int(p["ema_fast"]))
    es = ema_series(closes, int(p["ema_slow"]))
    pos = [0.0] * len(closes); cur = 0.0
    for i in range(len(closes)):
        r = rsi[i]
        if r is None:
            continue
        # only fade in a non-trending context
        trending = abs(ef[i] - es[i]) / closes[i] * 100 > p["trend_gate"] if closes[i] else False
        if not trending:
            if r <= p["rsi_oversold"]:
                cur = 1.0
            elif r >= p["rsi_overbought"]:
                cur = -1.0
            elif 45 <= r <= 55:
                cur = 0.0
        pos[i] = cur
    return pos


def sig_momentum(candles, p):
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


def sig_regime(candles, p):
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


VARIANTS: dict[str, Callable] = {
    "trend": sig_trend, "breakout": sig_breakout, "meanrev": sig_meanrev,
    "momentum": sig_momentum, "regime": sig_regime,
}

SPACE = {
    "ema_fast": [5, 8, 12, 20], "ema_slow": [20, 30, 50], "ema_trend": [50, 100, 150],
    "atr_period": [10, 14, 20], "min_atr_pct": [0.0, 0.02, 0.05],
    "rsi_period": [9, 14, 21], "rsi_oversold": [20.0, 25.0, 30.0],
    "rsi_overbought": [70.0, 75.0, 80.0], "roc_period": [3, 6, 10],
    "roc_threshold": [0.05, 0.1, 0.2], "donchian": [10, 20, 40],
    "trend_gate": [0.05, 0.1, 0.2],
}


# --------------------------------------------------------------- simulate
def simulate(candles, pos, leverage=5.8, fee_pct=0.0002, funding_per_bar=0.0):
    """Returns metrics for a target-position series. Fill at next bar open-ish
    (use close-to-close with 1-bar lag to avoid look-ahead)."""
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
    # crude annualization-free Sharpe proxy on per-bar returns
    sharpe = 0.0
    if len(rets) > 2:
        m = sum(rets) / len(rets)
        var = sum((x - m) ** 2 for x in rets) / len(rets)
        sd = var ** 0.5
        sharpe = (m / sd) * (len(rets) ** 0.5) if sd else 0.0
    return {
        "return_pct": round((equity - 1) * 100, 3),
        "max_dd_pct": round(maxdd * 100, 3),
        "trades": trades, "bars": marked,
        "win_rate": round(wins / marked, 3) if marked else 0.0,
        "sharpe": round(sharpe, 3),
        "score": round((equity - 1) - maxdd * 1.5, 5),
    }


def search(candles, variant_fn, iterations, leverage, fee_pct):
    best = None
    for _ in range(iterations):
        p = {k: random.choice(v) for k, v in SPACE.items()}
        pos = variant_fn(candles, p)
        m = simulate(candles, pos, leverage=leverage, fee_pct=fee_pct)
        if best is None or m["score"] > best[0]["score"]:
            best = (m, p)
    return best


def run(iterations=40, leverage=5.8, fee_pct=0.0002, max_days=14):
    client = KalshiClient(load_credentials(), STORE.settings.environment)
    one = history.fetch_full_1m(client, STORE.settings.ticker, max_days=max_days)
    print(f"Full 1m history: {len(one)} bars (~{len(one)/1440:.1f} days)\n")
    print(f"Costs: taker fee {fee_pct*1e4:.1f} bps/side, leverage {leverage}x")
    print(f"Validation: tune on first 70% (TRAIN), report last 30% (TEST/OOS)\n")

    rows = []
    for tf in ["1m", "5m", "15m"]:
        candles = history.to_timeframe(one, tf)
        if len(candles) < 200:
            print(f"[{tf}] only {len(candles)} bars — skipping"); continue
        split = int(len(candles) * 0.7)
        train, test = candles[:split], candles[split:]
        print(f"=== {tf}: {len(candles)} bars (train {len(train)} / test {len(test)}) ===")
        for name, fn in VARIANTS.items():
            best = search(train, fn, iterations, leverage, fee_pct)
            if not best:
                continue
            tr_m, p = best
            test_pos = fn(test, p)
            te = simulate(test, test_pos, leverage=leverage, fee_pct=fee_pct)
            rows.append((tf, name, tr_m, te, p))
            print(f"  {name:9s} TRAIN ret {tr_m['return_pct']:+7.2f}% "
                  f"| TEST ret {te['return_pct']:+7.2f}% dd {te['max_dd_pct']:5.2f}% "
                  f"trades {te['trades']:3d} win {te['win_rate']:.2f} sharpe {te['sharpe']:+.2f}")
        print()

    # Buy & hold benchmark on test of each tf for reference
    print("Benchmark (buy & hold, full series, no leverage):")
    for tf in ["1m", "5m", "15m"]:
        candles = history.to_timeframe(one, tf)
        if len(candles) < 2:
            continue
        bh = (candles[-1]["close"] - candles[0]["close"]) / candles[0]["close"] * 100
        print(f"  {tf}: {bh:+.2f}%")

    # Pick best by TEST score that is positive and reasonably traded
    viable = [r for r in rows if r[3]["return_pct"] > 0 and r[3]["trades"] >= 5]
    print("\n--- Out-of-sample winners (TEST return > 0) ---")
    if not viable:
        print("NONE. No variant/timeframe is profitable out-of-sample after costs.")
    else:
        viable.sort(key=lambda r: r[3]["score"], reverse=True)
        for tf, name, tr, te, p in viable[:5]:
            print(f"  {tf} {name}: TEST {te['return_pct']:+.2f}% dd {te['max_dd_pct']:.2f}% "
                  f"sharpe {te['sharpe']:+.2f} trades {te['trades']}")
    return rows, viable


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    run(iterations=iters)
