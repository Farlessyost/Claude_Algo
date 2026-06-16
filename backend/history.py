"""Paginated historical candle fetch. KXBTCPERP is young (launched 2026-06-03),
so we pull the full available 1m history by walking backwards in <=4990-candle
windows, then aggregate to any timeframe for research/backtesting."""
from __future__ import annotations

import time
from typing import List

from . import market_data
from .kalshi_client import KalshiClient

WINDOW = 4990  # candles per request (API cap is 5000)


def fetch_full_1m(client: KalshiClient, ticker: str, max_days: int = 14) -> List[dict]:
    end_ts = int(time.time())
    floor_ts = end_ts - max_days * 24 * 3600
    all_rows: dict[int, dict] = {}
    cursor = end_ts
    for _ in range(max_days * 24 * 60 // WINDOW + 3):
        start = max(cursor - WINDOW * 60, floor_ts)
        try:
            raw = client.get_candlesticks(ticker, start, cursor, period_interval=1)
        except Exception:
            break
        rows = [c for c in (market_data._candle_from_api(x)
                for x in raw.get("candlesticks", [])) if c]
        if not rows:
            break
        for r in rows:
            all_rows[r["ts"]] = r
        cursor = start - 60
        if cursor <= floor_ts:
            break
        time.sleep(0.15)  # be polite to the API
    out = sorted(all_rows.values(), key=lambda c: c["ts"])
    return out


def to_timeframe(one_min: List[dict], timeframe: str) -> List[dict]:
    minutes = market_data.TIMEFRAME_MINUTES.get(timeframe, 15)
    if minutes <= 1:
        return one_min
    return market_data.aggregate(one_min, minutes)
