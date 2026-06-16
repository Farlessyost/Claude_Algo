"""Research lab for finding a real KXBTCPERP edge.

Discipline (to avoid self-deception on ~9 days of data):
  - Rich features incl. microstructure (spread, open interest), no look-ahead.
  - Signals at bar i use only info <= i; PnL realized on i -> i+1 move.
  - Edges judged out-of-sample via walk-forward, with per-trade t-stats and
    trade counts, not just total return.
  - History cached to disk so iteration is fast and we don't hammer the API.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import List, Optional

from .config import STATE_DIR, load_credentials
from .kalshi_client import KalshiClient, to_float

CACHE = STATE_DIR / "history_1m_rich.json"
WINDOW = 4990


def _rich_candle(c: dict) -> Optional[dict]:
    price = c.get("price") or {}
    bid = c.get("bid") or {}
    ask = c.get("ask") or {}
    close = to_float(price.get("close"))
    bclose = to_float(bid.get("close"))
    aclose = to_float(ask.get("close"))
    if close == 0:
        if bclose and aclose:
            close = (bclose + aclose) / 2
        else:
            return None
    o = to_float(price.get("open")) or close
    h = to_float(price.get("high")) or close
    l = to_float(price.get("low")) or close
    spread = (aclose - bclose) if (aclose and bclose) else 0.0
    return {
        "ts": int(c.get("end_period_ts", 0)),
        "open": o, "high": h, "low": l, "close": close,
        "bid": bclose or close, "ask": aclose or close, "spread": spread,
        "volume": to_float(c.get("volume")),
        "oi": to_float(c.get("open_interest")),
    }


def fetch_rich_1m(client: KalshiClient, ticker: str, max_days: int = 14) -> List[dict]:
    end_ts = int(time.time())
    floor_ts = end_ts - max_days * 24 * 3600
    rows: dict[int, dict] = {}
    cursor = end_ts
    for _ in range(max_days * 24 * 60 // WINDOW + 3):
        start = max(cursor - WINDOW * 60, floor_ts)
        try:
            raw = client.get_candlesticks(ticker, start, cursor, period_interval=1)
        except Exception:
            break
        got = [r for r in (_rich_candle(x) for x in raw.get("candlesticks", [])) if r]
        if not got:
            break
        for r in got:
            rows[r["ts"]] = r
        cursor = start - 60
        if cursor <= floor_ts:
            break
        time.sleep(0.15)
    return sorted(rows.values(), key=lambda r: r["ts"])


def clean(rows: List[dict]) -> List[dict]:
    """Drop corrupted candles (bad price/bid/ask) using a median price band."""
    cs = sorted(r["close"] for r in rows if 1.0 < r["close"] < 100.0)
    if not cs:
        return rows
    med = cs[len(cs) // 2]
    lo, hi = med * 0.5, med * 1.5
    out = []
    for r in rows:
        if not (lo < r["close"] < hi):
            continue
        bid = r["bid"] if lo < r["bid"] < hi else r["close"]
        ask = r["ask"] if lo < r["ask"] < hi else r["close"]
        spread = ask - bid if (ask >= bid and (ask - bid) < med * 0.05) else 0.0
        out.append({**r, "bid": bid, "ask": ask, "spread": spread,
                    "open": r["open"] if lo < r["open"] < hi else r["close"],
                    "high": r["high"] if lo < r["high"] < hi else r["close"],
                    "low": r["low"] if lo < r["low"] < hi else r["close"]})
    return out


def load(use_cache: bool = True, max_days: int = 14, do_clean: bool = True) -> List[dict]:
    if use_cache and CACHE.exists():
        rows = json.loads(CACHE.read_text("utf-8"))
    else:
        client = KalshiClient(load_credentials(), "production")
        rows = fetch_rich_1m(client, "KXBTCPERP", max_days=max_days)
        CACHE.write_text(json.dumps(rows), encoding="utf-8")
    return clean(rows) if do_clean else rows


def aggregate(rows: List[dict], minutes: int) -> List[dict]:
    if minutes <= 1:
        return rows
    out = []
    for i in range(0, len(rows), minutes):
        b = rows[i:i + minutes]
        if not b:
            continue
        out.append({
            "ts": b[-1]["ts"], "open": b[0]["open"],
            "high": max(x["high"] for x in b), "low": min(x["low"] for x in b),
            "close": b[-1]["close"], "bid": b[-1]["bid"], "ask": b[-1]["ask"],
            "spread": sum(x["spread"] for x in b) / len(b),
            "volume": sum(x["volume"] for x in b),
            "oi": b[-1]["oi"],
        })
    return out


# ------------------------------------------------------------------ stats
def mean(xs): return sum(xs) / len(xs) if xs else 0.0
def std(xs):
    if len(xs) < 2: return 0.0
    m = mean(xs); return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def autocorr(xs: List[float], lag: int) -> float:
    n = len(xs)
    if n <= lag + 2:
        return 0.0
    m = mean(xs)
    num = sum((xs[i] - m) * (xs[i - lag] - m) for i in range(lag, n))
    den = sum((x - m) ** 2 for x in xs)
    return num / den if den else 0.0


def tstat(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    s = std(xs)
    return (mean(xs) / s) * math.sqrt(len(xs)) if s else 0.0


def returns(closes: List[float]) -> List[float]:
    return [(closes[i] - closes[i - 1]) / closes[i - 1]
            for i in range(1, len(closes)) if closes[i - 1]]
