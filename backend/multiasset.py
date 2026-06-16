"""Multi-asset / cross-venue data feeds for the Trophic Information Forager.

All sources are public (no API keys). Every fetcher returns None / {} on
failure so the ecology module degrades gracefully — a missing feed just
removes that node from the information network for the cycle.

Sources:
  - Coinbase Exchange public REST   : BTC-USD and ETH-USD spot mid + ticker
                                      plus top-book imbalance/spread
  - Kalshi (existing signed client) : KXETHPERP orderbook + candles
  - Hyperliquid public REST         : BTC perp open-interest, mark price,
                                       funding rate, premium vs oracle.
                                       Used as the liquidation-cascade
                                       proxy via |delta OI / prev OI|
                                       sampled at our cycle cadence.
                                      Expanded with ETH/SOL/HYPE breadth,
                                      total OI pressure, BTC book imbalance.
  - Deribit public REST             : BTC perp/term basis, OI, funding, IV skew
  - Kraken public REST              : venue-dispersion basis/spread/depth
  - Alternative.me public REST      : slow crypto-wide cap/dominance/breadth
  - mempool.space public REST       : BTC fee/mempool settlement pressure

Hyperliquid (replaced Binance 2026-06-15): Binance fapi.binance.com is
HTTP 451 geo-blocked from US networks. Hyperliquid is the largest BTC perp
DEX, has a clean public POST /info endpoint with no auth/no geo-block, and
returns markPx + funding + openInterest in a single call. We sample once
per trading cycle and push the OI series to HISTORY to derive the same
liq-proxy series the ecology consumes.

Caching is in-memory only; each call is single-shot so a stale/down feed
doesn't poison a later cycle. Timeouts are tight (4 s) so the trading loop
never blocks on a slow upstream.
"""
from __future__ import annotations

import threading
import time
from collections import deque
from typing import Any, Deque, Dict, Optional

import httpx


_TIMEOUT = 4.0
_USER_AGENT = "ClaudeAlgo/Ecology"
_CACHE: Dict[str, tuple] = {}


def _cached(key: str, ttl_s: float, fn):
    now = time.time()
    hit = _CACHE.get(key)
    if hit and (now - hit[0]) < ttl_s:
        return hit[1]
    val = fn()
    if val is not None:
        _CACHE[key] = (now, val)
    return val


# ---------------------------------------------------------------- HTTP wrap
def _get(url: str, params: Optional[dict] = None) -> Optional[Any]:
    try:
        with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}) as c:
            r = c.get(url, params=params)
            if r.status_code != 200:
                return None
            return r.json()
    except Exception:
        return None


def _post_json(url: str, body: dict) -> Optional[Any]:
    try:
        with httpx.Client(timeout=_TIMEOUT, headers={"User-Agent": _USER_AGENT}) as c:
            r = c.post(url, json=body)
            if r.status_code != 200:
                return None
            return r.json()
    except Exception:
        return None


# ---------------------------------------------------------- Coinbase (spot)
# Public endpoint, no auth: https://api.exchange.coinbase.com/products/{p}/ticker
def coinbase_ticker(product: str) -> Optional[dict]:
    j = _get(f"https://api.exchange.coinbase.com/products/{product}/ticker")
    if not j:
        return None
    try:
        bid = float(j.get("bid"))
        ask = float(j.get("ask"))
        price = float(j.get("price"))
        return {
            "product": product, "price": price,
            "bid": bid, "ask": ask,
            "mid": (bid + ask) / 2.0 if (bid and ask) else price,
            "spread_bps": ((ask - bid) / ((bid + ask) / 2.0) * 10_000.0)
                            if (bid and ask) else None,
            "volume_24h": float(j.get("volume") or 0.0),
            "ts": j.get("time"),
        }
    except (TypeError, ValueError):
        return None


def coinbase_book(product: str, depth: int = 10) -> Optional[dict]:
    """Top-book depth/imbalance from Coinbase level-2 public book."""
    j = _get(f"https://api.exchange.coinbase.com/products/{product}/book",
             params={"level": 2})
    if not j:
        return None
    try:
        bids = j.get("bids") or []
        asks = j.get("asks") or []
        if not bids or not asks:
            return None
        bid_px = float(bids[0][0]); ask_px = float(asks[0][0])
        bid_qty = sum(float(x[1]) for x in bids[:depth])
        ask_qty = sum(float(x[1]) for x in asks[:depth])
        mid = (bid_px + ask_px) / 2.0
        denom = bid_qty + ask_qty
        return {
            "product": product,
            "best_bid": bid_px,
            "best_ask": ask_px,
            "mid": mid,
            "bid_depth": bid_qty,
            "ask_depth": ask_qty,
            "depth_total": denom,
            "depth_imbalance": ((bid_qty - ask_qty) / denom) if denom else 0.0,
            "spread_bps": ((ask_px - bid_px) / mid * 10_000.0) if mid else None,
        }
    except (TypeError, ValueError, IndexError):
        return None


# ---------------------------------------------------------- Deribit BTC surface
_DERIBIT_FUT = "https://www.deribit.com/api/v2/public/get_book_summary_by_currency"


def deribit_btc_surface() -> Optional[dict]:
    """BTC derivatives surface: perp basis, term basis, OI, funding, IV skew.

    Option skew is best-effort and cached with the rest of the Deribit surface;
    it uses near-ATM put IV minus call IV for the nearest listed expiry.
    """
    fut = _get(_DERIBIT_FUT, params={"currency": "BTC", "kind": "future"})
    if not isinstance(fut, dict) or not isinstance(fut.get("result"), list):
        return None
    try:
        rows = fut["result"]
        perp = next((r for r in rows if r.get("instrument_name") == "BTC-PERPETUAL"), None)
        dated = [r for r in rows if r.get("instrument_name") != "BTC-PERPETUAL"]
        if not perp:
            return None
        delivery = float(perp.get("estimated_delivery_price") or 0.0)
        pmark = float(perp.get("mark_price") or perp.get("mid_price") or 0.0)
        total_oi = sum(float(r.get("open_interest") or 0.0) for r in rows)
        term = max(dated, key=lambda r: float(r.get("open_interest") or 0.0)) if dated else None
        tmark = float((term or {}).get("mark_price") or (term or {}).get("mid_price") or 0.0)
        perp_basis = ((pmark - delivery) / delivery * 10_000.0) if delivery else None
        term_basis = ((tmark - delivery) / delivery * 10_000.0) if (delivery and tmark) else None
        out = {
            "perp_mark": pmark,
            "estimated_delivery_price": delivery,
            "perp_basis_bps": perp_basis,
            "term_instrument": (term or {}).get("instrument_name"),
            "term_basis_bps": term_basis,
            "open_interest_total": total_oi,
            "perp_open_interest": float(perp.get("open_interest") or 0.0),
            "current_funding": float(perp.get("current_funding") or 0.0),
            "funding_8h": float(perp.get("funding_8h") or 0.0),
            "perp_spread_bps": (((float(perp.get("ask_price") or 0.0)
                                  - float(perp.get("bid_price") or 0.0))
                                 / pmark * 10_000.0) if pmark else None),
        }
        skew = deribit_btc_option_skew()
        if skew is not None:
            out["option_skew"] = skew.get("option_skew")
            out["option_skew_expiry"] = skew.get("expiry")
        return out
    except (TypeError, ValueError):
        return None


def deribit_btc_option_skew() -> Optional[dict]:
    j = _get(_DERIBIT_FUT, params={"currency": "BTC", "kind": "option"})
    if not isinstance(j, dict) or not isinstance(j.get("result"), list):
        return None
    try:
        rows = [r for r in j["result"] if r.get("mark_iv") is not None]
        if not rows:
            return None
        underlying = float(rows[0].get("underlying_price") or rows[0].get("estimated_delivery_price") or 0.0)
        if underlying <= 0:
            return None
        parsed = []
        for r in rows:
            name = r.get("instrument_name") or ""
            parts = name.split("-")
            if len(parts) < 4:
                continue
            expiry, strike_s, cp = parts[1], parts[2], parts[3]
            strike = float(strike_s)
            if abs(strike / underlying - 1.0) > 0.10:
                continue
            parsed.append((expiry, cp, float(r.get("mark_iv") or 0.0)))
        if not parsed:
            return None
        expiry = sorted({p[0] for p in parsed})[0]
        calls = [iv for e, cp, iv in parsed if e == expiry and cp == "C"]
        puts = [iv for e, cp, iv in parsed if e == expiry and cp == "P"]
        if not calls or not puts:
            return None
        return {"expiry": expiry, "option_skew": (sum(puts) / len(puts)) - (sum(calls) / len(calls))}
    except (TypeError, ValueError, IndexError):
        return None


# ---------------------------------------------------------- Kraken dispersion
def kraken_tickers() -> Optional[dict]:
    j = _get("https://api.kraken.com/0/public/Ticker", params={"pair": "XBTUSD,ETHUSD"})
    if not isinstance(j, dict) or j.get("error"):
        return None
    try:
        res = j.get("result") or {}
        btc = res.get("XXBTZUSD") or next(v for k, v in res.items() if "XBT" in k)
        eth = res.get("XETHZUSD") or next(v for k, v in res.items() if "ETH" in k)

        def one(x):
            ask = float(x["a"][0]); bid = float(x["b"][0])
            ask_qty = float(x["a"][1]); bid_qty = float(x["b"][1])
            mid = (ask + bid) / 2.0
            return {
                "bid": bid, "ask": ask, "mid": mid,
                "bid_qty": bid_qty, "ask_qty": ask_qty,
                "bbo_depth": bid_qty + ask_qty,
                "spread_bps": ((ask - bid) / mid * 10_000.0) if mid else None,
                "vwap_24h": float(x["p"][1]),
                "volume_24h": float(x["v"][1]),
            }
        return {"btc": one(btc), "eth": one(eth)}
    except (StopIteration, KeyError, TypeError, ValueError):
        return None


# ---------------------------------------------------------- Hyperliquid (perp DEX)
# Public POST /info endpoint, no auth, no geo-block. metaAndAssetCtxs returns
# the universe + per-asset context in a single call: funding, openInterest,
# markPx, midPx, premium, oraclePx, dayNtlVlm, dayBaseVlm, prevDayPx.
_HL_INFO = "https://api.hyperliquid.xyz/info"


def hyperliquid_btc_ctx() -> Optional[dict]:
    """Fetch current BTC perp context from Hyperliquid.

    Returns a flat dict ready for the ecology summary. Returns None on any
    network/parse failure so the snapshot loop drops the node gracefully.
    """
    j = _post_json(_HL_INFO, {"type": "metaAndAssetCtxs"})
    if not isinstance(j, list) or len(j) < 2:
        return None
    try:
        universe = j[0].get("universe") or []
        ctxs = j[1] or []
        btc_idx = -1
        for i, u in enumerate(universe):
            if u.get("name") == "BTC":
                btc_idx = i
                break
        if btc_idx < 0 or btc_idx >= len(ctxs):
            return None
        c = ctxs[btc_idx]
        mark = float(c.get("markPx") or 0.0)
        oracle = float(c.get("oraclePx") or 0.0)
        mid = float(c.get("midPx") or mark)
        prev = float(c.get("prevDayPx") or 0.0)
        day_ntl = float(c.get("dayNtlVlm") or 0.0)
        day_base = float(c.get("dayBaseVlm") or 0.0)
        oi_btc = float(c.get("openInterest") or 0.0)
        funding = float(c.get("funding") or 0.0)
        premium = float(c.get("premium") or 0.0)
        change_pct = ((mark - prev) / prev * 100.0) if prev else None
        return {
            "mark_price": mark,
            "mid_price": mid,
            "oracle_price": oracle,
            "prev_day_price": prev,
            "open_interest_btc": oi_btc,
            "open_interest_usd": oi_btc * mark if mark else 0.0,
            "funding_rate": funding,
            "premium": premium,
            "volume_24h_usd": day_ntl,
            "volume_24h_btc": day_base,
            "price_change_24h_pct": change_pct,
        }
    except (TypeError, ValueError, IndexError, AttributeError):
        return None


def hyperliquid_market_ctxs(coins=("BTC", "ETH", "SOL", "HYPE")) -> Optional[dict]:
    j = _post_json(_HL_INFO, {"type": "metaAndAssetCtxs"})
    if not isinstance(j, list) or len(j) < 2:
        return None
    try:
        universe = j[0].get("universe") or []
        ctxs = j[1] or []
        by = {}
        for i, u in enumerate(universe):
            name = u.get("name")
            if name in coins and i < len(ctxs):
                c = ctxs[i]
                mark = float(c.get("markPx") or 0.0)
                prev = float(c.get("prevDayPx") or 0.0)
                oi = float(c.get("openInterest") or 0.0)
                by[name] = {
                    "mark_price": mark,
                    "prev_day_price": prev,
                    "open_interest": oi,
                    "open_interest_usd": oi * mark if mark else 0.0,
                    "funding_rate": float(c.get("funding") or 0.0),
                    "premium": float(c.get("premium") or 0.0),
                    "change_pct": ((mark - prev) / prev * 100.0) if prev else None,
                }
        if not by:
            return None
        alt_changes = [v["change_pct"] for k, v in by.items()
                       if k != "BTC" and v.get("change_pct") is not None]
        return {
            "coins": by,
            "alt_breadth": (sum(1 for x in alt_changes if x > 0) / len(alt_changes))
                            if alt_changes else None,
            "alt_avg_change_pct": (sum(alt_changes) / len(alt_changes)) if alt_changes else None,
            "total_oi_usd": sum(v.get("open_interest_usd") or 0.0 for v in by.values()),
        }
    except (TypeError, ValueError, IndexError, AttributeError):
        return None


def hyperliquid_btc_book(depth: int = 10) -> Optional[dict]:
    j = _post_json(_HL_INFO, {"type": "l2Book", "coin": "BTC"})
    if not isinstance(j, dict):
        return None
    try:
        levels = j.get("levels") or []
        bids = levels[0] if len(levels) > 0 else []
        asks = levels[1] if len(levels) > 1 else []
        if not bids or not asks:
            return None
        bid_sz = sum(float(x.get("sz") or 0.0) for x in bids[:depth])
        ask_sz = sum(float(x.get("sz") or 0.0) for x in asks[:depth])
        bid = float(bids[0].get("px") or 0.0)
        ask = float(asks[0].get("px") or 0.0)
        mid = (bid + ask) / 2.0
        denom = bid_sz + ask_sz
        return {
            "bid_depth": bid_sz,
            "ask_depth": ask_sz,
            "depth_total": denom,
            "book_imbalance": ((bid_sz - ask_sz) / denom) if denom else 0.0,
            "spread_bps": ((ask - bid) / mid * 10_000.0) if mid else None,
        }
    except (TypeError, ValueError, IndexError, AttributeError):
        return None


# ---------------------------------------------------------- broad/settlement slow feeds
def crypto_breadth() -> Optional[dict]:
    glob = _get("https://api.alternative.me/v2/global/")
    tick = _get("https://api.alternative.me/v2/ticker/", params={"limit": 10, "convert": "USD"})
    if not isinstance(glob, dict):
        return None
    try:
        data = glob.get("data") or {}
        q = (data.get("quotes") or {}).get("USD") or {}
        out = {
            "total_market_cap_usd": float(q.get("total_market_cap") or 0.0),
            "total_volume_24h_usd": float(q.get("total_volume_24h") or 0.0),
            "btc_dominance": float(data.get("bitcoin_percentage_of_market_cap") or 0.0),
        }
        if isinstance(tick, dict) and isinstance(tick.get("data"), dict):
            changes = []
            alt_changes = []
            for _, coin in (tick.get("data") or {}).items():
                quote = ((coin.get("quotes") or {}).get("USD") or {})
                chg = quote.get("percent_change_1h")
                if chg is None:
                    continue
                chg = float(chg)
                changes.append(chg)
                sym = (coin.get("symbol") or "").upper()
                if sym not in ("BTC", "ETH"):
                    alt_changes.append(chg)
            if changes:
                out["top10_breadth"] = sum(1 for x in changes if x > 0) / len(changes)
            if alt_changes:
                out["risk_on_alt_breadth"] = sum(alt_changes) / len(alt_changes)
        return out
    except (TypeError, ValueError, AttributeError):
        return None


def mempool_pressure() -> Optional[dict]:
    fees = _get("https://mempool.space/api/v1/fees/recommended")
    mp = _get("https://mempool.space/api/mempool")
    if not isinstance(fees, dict):
        return None
    try:
        fastest = float(fees.get("fastestFee") or 0.0)
        half = float(fees.get("halfHourFee") or 0.0)
        hour = float(fees.get("hourFee") or 0.0)
        vsize = float((mp or {}).get("vsize") or 0.0) if isinstance(mp, dict) else 0.0
        count = float((mp or {}).get("count") or 0.0) if isinstance(mp, dict) else 0.0
        return {
            "fastest_fee": fastest,
            "half_hour_fee": half,
            "hour_fee": hour,
            "fee_pressure": (fastest + half + hour) / 3.0,
            "mempool_vsize": vsize,
            "mempool_count": count,
        }
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------- Kalshi ETH perp
_ETH_TICKER = "KXETHPERP"


def kalshi_eth_orderbook(client) -> Optional[dict]:
    if client is None:
        return None
    try:
        ob_raw = client.get_orderbook(_ETH_TICKER, depth=5)
        book = ob_raw.get("orderbook", {}) or {}
        bids = book.get("bids", []) or []
        asks = book.get("asks", []) or []
        if not bids or not asks:
            return None
        from .kalshi_client import to_float
        best_bid = to_float(bids[0][0])
        best_ask = to_float(asks[0][0])
        if not best_bid or not best_ask:
            return None
        mid = (best_bid + best_ask) / 2.0
        return {
            "ticker": _ETH_TICKER, "best_bid": best_bid, "best_ask": best_ask,
            "mid": mid, "spread": best_ask - best_bid,
            "spread_bps": ((best_ask - best_bid) / mid * 10_000.0) if mid else None,
            "depth_total": sum(to_float(l[1]) for l in (bids[:5] + asks[:5])),
        }
    except Exception:
        return None


# ----------------------------------------------- rolling per-node history
class MultiAssetHistory:
    """Thread-safe ring buffers per node. The ecology module pulls slices
    out of this to build the information network. Each cycle appends one
    snapshot; ~512 cycles of history is plenty for the lagged-correlation
    matrix used downstream."""

    MAX = 512

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._series: Dict[str, Deque[tuple]] = {}

    def push(self, node: str, ts: float, value: float) -> None:
        with self._lock:
            buf = self._series.setdefault(node, deque(maxlen=self.MAX))
            buf.append((ts, value))

    def series(self, node: str, n: int = 128) -> list:
        with self._lock:
            buf = self._series.get(node)
            if not buf:
                return []
            return list(buf)[-n:]

    def all_nodes(self) -> list:
        with self._lock:
            return list(self._series.keys())


HISTORY = MultiAssetHistory()


# ---------------------------------------------- one-shot snapshot per cycle
def snapshot(settings, kalshi_client) -> dict:
    """Build a multi-asset snapshot for one trading cycle. Each enabled feed
    runs once; failure is silent (None). The ecology module is the only
    consumer; it inspects what's present and computes the network over
    whatever nodes have data this cycle.

    The returned dict is also pushed into HISTORY so downstream nodes can
    look at the recent N samples for lagged-correlation estimation.
    """
    ts = time.time()
    out: dict = {"ts": ts, "sources": {}}

    if getattr(settings, "ecology_use_coinbase_spot", True):
        btc_spot = coinbase_ticker("BTC-USD")
        eth_spot = coinbase_ticker("ETH-USD")
        out["btc_spot"] = btc_spot
        out["eth_spot"] = eth_spot
        out["sources"]["coinbase"] = bool(btc_spot or eth_spot)
        if btc_spot and btc_spot.get("mid"):
            HISTORY.push("btc_spot", ts, btc_spot["mid"])
        if eth_spot and eth_spot.get("mid"):
            HISTORY.push("eth_spot", ts, eth_spot["mid"])

    if getattr(settings, "ecology_use_coinbase_book", True):
        cb_btc_book = coinbase_book("BTC-USD", depth=10)
        cb_eth_book = coinbase_book("ETH-USD", depth=10)
        out["coinbase_btc_book"] = cb_btc_book
        out["coinbase_eth_book"] = cb_eth_book
        out["sources"]["coinbase_book"] = bool(cb_btc_book or cb_eth_book)
        if cb_btc_book:
            if cb_btc_book.get("depth_imbalance") is not None:
                HISTORY.push("cb_btc_depth_imbalance", ts, cb_btc_book["depth_imbalance"])
            if cb_btc_book.get("spread_bps") is not None:
                HISTORY.push("cb_btc_spread_bps", ts, cb_btc_book["spread_bps"])
        if cb_eth_book and cb_eth_book.get("depth_imbalance") is not None:
            HISTORY.push("cb_eth_depth_imbalance", ts, cb_eth_book["depth_imbalance"])

    if getattr(settings, "ecology_use_kalshi_eth", True):
        eth_perp = kalshi_eth_orderbook(kalshi_client)
        out["eth_perp"] = eth_perp
        out["sources"]["kalshi_eth"] = bool(eth_perp)
        if eth_perp and eth_perp.get("mid"):
            HISTORY.push("eth_perp", ts, eth_perp["mid"])
        if eth_perp and eth_perp.get("spread_bps") is not None:
            HISTORY.push("eth_spread", ts, eth_perp["spread_bps"])

    # Hyperliquid replaces Binance (geo-blocked from US, HTTP 451). The
    # setting key kept its original name (ecology_use_binance_liq) for
    # backwards compatibility with saved settings.json files; the new alias
    # ecology_use_hyperliquid is also honored.
    use_hl = bool(getattr(settings, "ecology_use_hyperliquid",
                            getattr(settings, "ecology_use_binance_liq", True)))
    if use_hl:
        hl = hyperliquid_btc_ctx()
        out["hyperliquid_btc"] = hl
        out["sources"]["hyperliquid"] = bool(hl)
        if hl:
            # Push OI to history so we can derive cycle-over-cycle changes.
            # On the very first sample after startup we won't have a prev OI
            # yet — the next cycle gets the first delta. This is the same
            # cadence the Binance integration achieved (sampling every cycle)
            # but at our actual loop cadence rather than Binance's 5-minute
            # rollups.
            oi_now = hl.get("open_interest_btc") or 0.0
            HISTORY.push("hl_oi", ts, oi_now)
            recent_oi = HISTORY.series("hl_oi", n=2)
            if len(recent_oi) >= 2 and recent_oi[-2][1]:
                prev = recent_oi[-2][1]
                delta_pct = (oi_now - prev) / prev
                # Same node names the ecology graph consumes today, so the
                # downstream classifier doesn't need to change.
                HISTORY.push("liq_proxy", ts, abs(delta_pct))
                HISTORY.push("oi_change", ts, delta_pct)
            fr = hl.get("funding_rate")
            if fr is not None:
                HISTORY.push("binance_funding", ts, fr)

        if getattr(settings, "ecology_use_hyperliquid_breadth", True):
            hl_breadth = hyperliquid_market_ctxs()
            hl_book = hyperliquid_btc_book(depth=10)
            out["hyperliquid_breadth"] = hl_breadth
            out["hyperliquid_btc_book"] = hl_book
            out["sources"]["hyperliquid_breadth"] = bool(hl_breadth or hl_book)
            if hl_breadth:
                if hl_breadth.get("alt_breadth") is not None:
                    HISTORY.push("hl_alt_breadth", ts, hl_breadth["alt_breadth"])
                if hl_breadth.get("alt_avg_change_pct") is not None:
                    HISTORY.push("hl_alt_avg_change_pct", ts, hl_breadth["alt_avg_change_pct"])
                total_oi = hl_breadth.get("total_oi_usd") or 0.0
                HISTORY.push("hl_total_oi_usd", ts, total_oi)
                prev_oi = HISTORY.series("hl_total_oi_usd", n=2)
                if len(prev_oi) >= 2 and prev_oi[-2][1]:
                    HISTORY.push("hl_total_oi_change", ts,
                                 (total_oi - prev_oi[-2][1]) / prev_oi[-2][1])
                coins = hl_breadth.get("coins") or {}
                btc_prem = (coins.get("BTC") or {}).get("premium")
                if btc_prem is not None:
                    HISTORY.push("hl_premium", ts, btc_prem)
            if hl_book and hl_book.get("book_imbalance") is not None:
                HISTORY.push("hl_btc_book_imbalance", ts, hl_book["book_imbalance"])

    if getattr(settings, "ecology_use_deribit", True):
        db = _cached("deribit_btc_surface", 60.0, deribit_btc_surface)
        out["deribit_btc"] = db
        out["sources"]["deribit"] = bool(db)
        if db:
            for k, node in (
                ("perp_basis_bps", "deribit_perp_basis"),
                ("term_basis_bps", "deribit_term_basis"),
                ("funding_8h", "deribit_funding_8h"),
                ("option_skew", "deribit_option_skew"),
            ):
                if db.get(k) is not None:
                    HISTORY.push(node, ts, db[k])
            oi_total = db.get("open_interest_total") or 0.0
            HISTORY.push("deribit_oi_total", ts, oi_total)
            recent = HISTORY.series("deribit_oi_total", n=2)
            if len(recent) >= 2 and recent[-2][1]:
                HISTORY.push("deribit_oi_change", ts,
                             (oi_total - recent[-2][1]) / recent[-2][1])

    if getattr(settings, "ecology_use_kraken_dispersion", True):
        kr = kraken_tickers()
        out["kraken"] = kr
        out["sources"]["kraken"] = bool(kr)
        if kr:
            btc_mid = (kr.get("btc") or {}).get("mid")
            eth_mid = (kr.get("eth") or {}).get("mid")
            cb_btc_mid = ((out.get("btc_spot") or {}).get("mid"))
            if btc_mid and cb_btc_mid:
                HISTORY.push("kraken_coinbase_basis_bps",
                             ts, (btc_mid - cb_btc_mid) / cb_btc_mid * 10_000.0)
            if (kr.get("btc") or {}).get("spread_bps") is not None:
                HISTORY.push("kraken_spread_bps", ts, kr["btc"]["spread_bps"])
            if (kr.get("btc") or {}).get("bbo_depth") is not None:
                HISTORY.push("kraken_bbo_depth", ts, kr["btc"]["bbo_depth"])
            if btc_mid:
                HISTORY.push("kraken_btc_mid", ts, btc_mid)
            if eth_mid:
                HISTORY.push("kraken_eth_mid", ts, eth_mid)
            kb = HISTORY.series("kraken_btc_mid", n=2)
            ke = HISTORY.series("kraken_eth_mid", n=2)
            if len(kb) >= 2 and len(ke) >= 2 and kb[-2][1] and ke[-2][1]:
                btc_ret = (kb[-1][1] - kb[-2][1]) / kb[-2][1]
                eth_ret = (ke[-1][1] - ke[-2][1]) / ke[-2][1]
                HISTORY.push("kraken_eth_btc_lead", ts, eth_ret - btc_ret)

    if getattr(settings, "ecology_use_crypto_breadth", True):
        breadth = _cached("crypto_breadth", 300.0, crypto_breadth)
        out["crypto_breadth"] = breadth
        out["sources"]["crypto_breadth"] = bool(breadth)
        if breadth:
            for k, node in (
                ("total_market_cap_usd", "crypto_total_mcap"),
                ("btc_dominance", "btc_dominance"),
                ("top10_breadth", "top10_breadth"),
                ("risk_on_alt_breadth", "risk_on_alt_breadth"),
            ):
                if breadth.get(k) is not None:
                    HISTORY.push(node, ts, breadth[k])

    if getattr(settings, "ecology_use_mempool", True):
        memp = _cached("mempool_pressure", 60.0, mempool_pressure)
        out["mempool"] = memp
        out["sources"]["mempool"] = bool(memp)
        if memp:
            for k, node in (
                ("fastest_fee", "btc_fee_fastest"),
                ("half_hour_fee", "btc_fee_half_hour"),
                ("fee_pressure", "btc_fee_pressure"),
                ("mempool_vsize", "mempool_vsize"),
            ):
                if memp.get(k) is not None:
                    HISTORY.push(node, ts, memp[k])

    # ---- compact summary the UI can render directly ----
    out["summary"] = _summarize(out)
    return out


def _summarize(s: dict) -> dict:
    """Tiny, JSON-safe view for the UI panel."""
    btc = s.get("btc_spot") or {}
    eth = s.get("eth_spot") or {}
    eth_perp = s.get("eth_perp") or {}
    hl = s.get("hyperliquid_btc") or {}
    db = s.get("deribit_btc") or {}
    kr = s.get("kraken") or {}
    cb_btc_book = s.get("coinbase_btc_book") or {}
    hl_book = s.get("hyperliquid_btc_book") or {}
    hl_breadth = s.get("hyperliquid_breadth") or {}
    breadth = s.get("crypto_breadth") or {}
    memp = s.get("mempool") or {}
    # Pull recent OI delta directly from HISTORY so the UI always shows the
    # latest cycle-over-cycle change rather than the first-sample None.
    oi_series = HISTORY.series("hl_oi", n=2)
    oi_now = oi_series[-1][1] if oi_series else None
    oi_prev = oi_series[-2][1] if len(oi_series) >= 2 else None
    oi_delta_pct = ((oi_now - oi_prev) / oi_prev * 100.0) if (oi_now and oi_prev) else None
    return {
        "btc_spot_price": btc.get("mid"),
        "eth_spot_price": eth.get("mid"),
        "eth_perp_price": eth_perp.get("mid"),
        "eth_perp_spread_bps": eth_perp.get("spread_bps"),
        # Hyperliquid replaces the previous Binance fields. UI labels mirrored:
        # callers reading "binance_*" still get values from the new source so
        # the existing UI panels keep working with no JS changes.
        "binance_btc_last": hl.get("mark_price"),
        "binance_24h_change_pct": hl.get("price_change_24h_pct"),
        "binance_24h_range_pct": None,  # not in Hyperliquid metaAndAssetCtxs
        "binance_oi_now": hl.get("open_interest_btc"),
        "binance_oi_delta_pct_5m": oi_delta_pct,
        "binance_funding_rate": hl.get("funding_rate"),
        "hl_premium": hl.get("premium"),
        "hl_oracle_price": hl.get("oracle_price"),
        "hl_alt_breadth": hl_breadth.get("alt_breadth"),
        "hl_total_oi_usd": hl_breadth.get("total_oi_usd"),
        "hl_btc_book_imbalance": hl_book.get("book_imbalance"),
        "deribit_perp_basis_bps": db.get("perp_basis_bps"),
        "deribit_term_basis_bps": db.get("term_basis_bps"),
        "deribit_funding_8h": db.get("funding_8h"),
        "deribit_option_skew": db.get("option_skew"),
        "kraken_coinbase_basis_bps": (
            ((kr.get("btc") or {}).get("mid") - btc.get("mid")) / btc.get("mid") * 10_000.0
            if (kr.get("btc") or {}).get("mid") and btc.get("mid") else None),
        "kraken_spread_bps": (kr.get("btc") or {}).get("spread_bps"),
        "kraken_bbo_depth": (kr.get("btc") or {}).get("bbo_depth"),
        "cb_btc_depth_imbalance": cb_btc_book.get("depth_imbalance"),
        "cb_btc_spread_bps": cb_btc_book.get("spread_bps"),
        "crypto_total_mcap_usd": breadth.get("total_market_cap_usd"),
        "btc_dominance": breadth.get("btc_dominance"),
        "top10_breadth": breadth.get("top10_breadth"),
        "risk_on_alt_breadth": breadth.get("risk_on_alt_breadth"),
        "btc_fee_fastest": memp.get("fastest_fee"),
        "btc_fee_half_hour": memp.get("half_hour_fee"),
        "mempool_vsize": memp.get("mempool_vsize"),
        "feeds_ok": s.get("sources", {}),
    }
