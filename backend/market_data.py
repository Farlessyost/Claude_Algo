"""Fetches BTCPERP market data, aggregates 1m candles into the selected
timeframe (5m/15m have no native API support), and computes the feature set the
strategy and UI consume."""
from __future__ import annotations

import time
from typing import List, Optional

from . import indicators as ind
from .kalshi_client import KalshiClient, to_float

TIMEFRAME_MINUTES = {"1m": 1, "2m": 2, "3m": 3, "5m": 5, "15m": 15, "1h": 60}
# Periods the API serves natively (no local aggregation needed -> deeper history).
NATIVE_PERIOD = {"1m": 1, "1h": 60}
MAX_CANDLES = 4990  # API caps a single request at 5000


def _candle_from_api(c: dict) -> Optional[dict]:
    price = c.get("price") or {}
    o = to_float(price.get("open"))
    h = to_float(price.get("high"))
    l = to_float(price.get("low"))
    cl = to_float(price.get("close"))
    if cl == 0 and o == 0:
        return None
    return {
        "ts": int(c.get("end_period_ts", 0)),
        "open": o or cl, "high": h or cl, "low": l or cl, "close": cl,
        "volume": to_float(c.get("volume")),
    }


def aggregate(candles_1m: List[dict], minutes: int) -> List[dict]:
    """Aggregate 1-minute candles into `minutes`-sized buckets."""
    if minutes <= 1:
        return candles_1m
    out: List[dict] = []
    bucket: List[dict] = []
    bucket_size = minutes
    for c in candles_1m:
        bucket.append(c)
        if len(bucket) == bucket_size:
            out.append(_merge_bucket(bucket))
            bucket = []
    if bucket:
        out.append(_merge_bucket(bucket))
    return out


def _merge_bucket(bucket: List[dict]) -> dict:
    return {
        "ts": bucket[-1]["ts"],
        "open": bucket[0]["open"],
        "high": max(c["high"] for c in bucket),
        "low": min(c["low"] for c in bucket),
        "close": bucket[-1]["close"],
        "volume": sum(c["volume"] for c in bucket),
    }


def fetch_candles(client: KalshiClient, ticker: str, timeframe: str,
                  lookback_bars: int = 200) -> List[dict]:
    minutes = TIMEFRAME_MINUTES.get(timeframe, 15)
    end_ts = int(time.time())

    if timeframe in NATIVE_PERIOD:
        # Fetch the timeframe natively -> much deeper history available.
        period = NATIVE_PERIOD[timeframe]
        count = min(lookback_bars + 5, MAX_CANDLES)
        start_ts = end_ts - count * period * 60
        raw = client.get_candlesticks(ticker, start_ts, end_ts, period_interval=period)
        cs = [c for c in (_candle_from_api(x)
              for x in raw.get("candlesticks", [])) if c]
        cs.sort(key=lambda c: c["ts"])
        return cs[-lookback_bars:]

    # 5m / 15m have no native period: pull 1m candles and aggregate locally.
    minutes_needed = min(minutes * (lookback_bars + 5), MAX_CANDLES)
    start_ts = end_ts - minutes_needed * 60
    raw = client.get_candlesticks(ticker, start_ts, end_ts, period_interval=1)
    one_min = [c for c in (_candle_from_api(x)
               for x in raw.get("candlesticks", [])) if c]
    one_min.sort(key=lambda c: c["ts"])
    agg = aggregate(one_min, minutes)
    return agg[-lookback_bars:]


def orderbook_metrics(ob: dict) -> dict:
    book = ob.get("orderbook", {}) or {}
    bids = book.get("bids", []) or []
    asks = book.get("asks", []) or []

    def best(levels):
        if not levels:
            return None, None
        return to_float(levels[0][0]), to_float(levels[0][1])

    best_bid, bid_qty = best(bids)
    best_ask, ask_qty = best(asks)
    mid = None
    spread = None
    spread_bps = None
    if best_bid and best_ask:
        mid = (best_bid + best_ask) / 2.0
        spread = best_ask - best_bid
        if mid:
            spread_bps = spread / mid * 10_000.0
    bid_depth = sum(to_float(l[1]) for l in bids[:10])
    ask_depth = sum(to_float(l[1]) for l in asks[:10])
    return {
        "best_bid": best_bid, "best_ask": best_ask,
        "bid_qty": bid_qty, "ask_qty": ask_qty,
        "mid": mid, "spread": spread, "spread_bps": spread_bps,
        "bid_depth": bid_depth, "ask_depth": ask_depth,
        "depth_total": bid_depth + ask_depth,
    }


def compute_features(candles: List[dict], params: dict) -> dict:
    """Indicator panel + signals used by the strategy."""
    closes = [c["close"] for c in candles]
    out: dict = {
        "bars": len(candles),
        "last_close": closes[-1] if closes else None,
    }
    if len(closes) < 5:
        out.update({"trend": "n/a", "momentum": "n/a", "meanrev": "n/a",
                    "regime": "insufficient-data"})
        return out

    ema_fast = ind.ema(closes, int(params["ema_fast"]))
    ema_slow = ind.ema(closes, int(params["ema_slow"]))
    ema_trend = ind.ema(closes, int(params["ema_trend"]))
    rsi_v = ind.rsi(closes, int(params["rsi_period"]))
    roc_v = ind.roc(closes, int(params["roc_period"]))
    atr = ind.atr_pct(candles, int(params["atr_period"]))
    vol = ind.stdev_pct(closes, 20)

    # --- normalized component scores in [-1, 1] ---
    px = closes[-1]
    trend_score = 0.0
    if ema_fast and ema_slow:
        sep = (ema_fast[-1] - ema_slow[-1]) / px if px else 0.0
        trend_score = max(-1.0, min(1.0, sep * 200.0))
        if ema_trend:
            if px > ema_trend[-1]:
                trend_score = max(trend_score, 0.0) + 0.15
            else:
                trend_score = min(trend_score, 0.0) - 0.15
            trend_score = max(-1.0, min(1.0, trend_score))

    mom_score = 0.0
    if roc_v is not None:
        mom_score = max(-1.0, min(1.0, roc_v / max(params["roc_threshold"], 1e-6) / 5.0))

    mr_score = 0.0
    if rsi_v is not None:
        if rsi_v >= params["rsi_overbought"]:
            mr_score = -min(1.0, (rsi_v - params["rsi_overbought"]) / 20.0 + 0.3)
        elif rsi_v <= params["rsi_oversold"]:
            mr_score = min(1.0, (params["rsi_oversold"] - rsi_v) / 20.0 + 0.3)

    regime = "ranging"
    if atr is not None:
        if abs(trend_score) > 0.4 and atr > params["min_atr_pct"]:
            regime = "trending"
        elif atr is not None and atr > params["max_atr_pct"]:
            regime = "high-volatility"
        elif atr is not None and atr < params["min_atr_pct"]:
            regime = "quiet"

    # Realized move on the most recent bar. `recent_move_bps` is the unsigned
    # range/absolute pressure used for display/risk. `recent_move_signed_bps`
    # is directional close-vs-previous-close pressure used by execution. Do not
    # feed the unsigned value into a side-specific taker chase; that turns every
    # volatile bar into "cross the spread" regardless of order direction.
    recent_move_bps = 0.0
    recent_move_signed_bps = 0.0
    if candles and px:
        last = candles[-1]
        hi = float(last.get("high") or px)
        lo = float(last.get("low") or px)
        rng = max(0.0, hi - lo)
        ret_signed = (closes[-1] - closes[-2]) if len(closes) >= 2 else 0.0
        recent_move_signed_bps = (ret_signed / px * 10_000.0) if px else 0.0
        recent_move_bps = max(rng, abs(ret_signed)) / px * 10_000.0

    out.update({
        "ema_fast": ema_fast[-1] if ema_fast else None,
        "ema_slow": ema_slow[-1] if ema_slow else None,
        "ema_trend": ema_trend[-1] if ema_trend else None,
        "rsi": rsi_v,
        "roc": roc_v,
        "atr_pct": atr,
        "stdev_pct": vol,
        "recent_move_bps": round(recent_move_bps, 2),
        "recent_move_signed_bps": round(recent_move_signed_bps, 2),
        "trend_score": round(trend_score, 4),
        "momentum_score": round(mom_score, 4),
        "meanrev_score": round(mr_score, 4),
        "trend": _label(trend_score),
        "momentum": _label(mom_score),
        "meanrev": _label(mr_score),
        "regime": regime,
    })
    return out


def _label(score: float) -> str:
    if score > 0.15:
        return "bullish"
    if score < -0.15:
        return "bearish"
    return "neutral"


def build_market_view(client: KalshiClient, ticker: str, timeframe: str,
                      params: dict) -> dict:
    candles = fetch_candles(client, ticker, timeframe, lookback_bars=400)
    ob = client.get_orderbook(ticker, depth=10)
    obm = orderbook_metrics(ob)
    feats = compute_features(candles, params)
    funding = client.get_funding_rate_estimate(ticker)

    last = feats.get("last_close")
    mid = obm.get("mid") or last
    btc_price = (mid or last)
    return {
        "ticker": ticker,
        "timeframe": timeframe,
        "price": mid or last,
        "btc_price_per_contract": btc_price,
        "btc_spot_estimate": (btc_price * 10_000) if btc_price else None,
        "orderbook": obm,
        "features": feats,
        "funding": funding,
        "candles": candles[-60:],   # for the chart
        "candles_full": candles,    # for the strategy/indicators
        "refreshed_ts": _now_iso(),
    }


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
