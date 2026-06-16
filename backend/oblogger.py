"""Continuous order-book snapshot logger. Writes one JSONL line every `interval`
seconds to state/orderbook_log.jsonl so we can build a microstructure dataset
(order-book imbalance, depth, spread) and later backtest imbalance signals that
currently have NO history. Runs as a daemon thread alongside the server."""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone

from .config import STATE_DIR
from . import market_data

OB_LOG = STATE_DIR / "orderbook_log.jsonl"


class ObLogger:
    """Polls the live orderbook on a fast cadence and (a) writes a snapshot row
    to disk for the microstructure dataset, and (b) publishes a live quote to
    STORE so the UI can render bid/ask/mid and a mark-to-market chart between
    full trading-loop cycles. Cadence is vol-aware: fast when ATR% is hot, slow
    when quiet, with a separate file-write throttle so we don't flood disk
    with near-duplicate snapshots in dead markets."""

    def __init__(self, fast_interval: float = 1.0, slow_interval: float = 10.0,
                 disk_min_interval: float = 5.0):
        self._thread = None
        self._stop = threading.Event()
        self.fast_interval = fast_interval
        self.slow_interval = slow_interval
        self.disk_min_interval = disk_min_interval
        self.count = 0          # disk writes (microstructure rows)
        self.poll_count = 0     # in-memory poll count (UI ticks)
        self.last_ts = None
        self.last_error = None
        self._last_disk_ts = 0.0

    @property
    def interval(self):
        """Back-compat: the slowest cadence (used by the UI status pill)."""
        return self.slow_interval

    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, get_client, get_ticker):
        if self.running():
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, args=(get_client, get_ticker), daemon=True)
        self._thread.start()

    def _poll_interval(self) -> float:
        """Cadence of the live-mid feed. The forager's fast-harvest needs a ~1s
        mid, so poll 1s in any active market and back off only when the market
        is truly dead (so we still refresh the UI without spamming the API)."""
        # Lazy import to avoid an import cycle (store -> config -> ... -> here).
        from .store import STORE
        atr = ((STORE.market or {}).get("features", {}) or {}).get("atr_pct")
        if atr is None or atr >= 0.05:
            return self.fast_interval        # 1s whenever there's any movement
        return max(self.fast_interval, 4.0)  # 4s only in a dead-quiet market

    def _loop(self, get_client, get_ticker):
        from .store import STORE
        while not self._stop.is_set():
            poll_start = time.time()
            try:
                client = get_client()
                ticker = get_ticker()
                if client:
                    ob = client.get_orderbook(ticker, depth=10)
                    m = market_data.orderbook_metrics(ob)
                    bd = m.get("bid_depth") or 0.0
                    ad = m.get("ask_depth") or 0.0
                    bq = m.get("bid_qty") or 0.0
                    aq = m.get("ask_qty") or 0.0
                    ts_iso = datetime.now(timezone.utc).isoformat()
                    rec = {
                        "ts": ts_iso,
                        "bid": m.get("best_bid"), "ask": m.get("best_ask"),
                        "mid": m.get("mid"), "spread_bps": m.get("spread_bps"),
                        "bid_qty": bq, "ask_qty": aq,
                        "bid_depth": bd, "ask_depth": ad,
                        "imb_top": round((bq - aq) / (bq + aq), 4) if (bq + aq) else 0.0,
                        "imb_depth": round((bd - ad) / (bd + ad), 4) if (bd + ad) else 0.0,
                    }
                    # Always publish to STORE so the UI gets a fast tick.
                    STORE.live_quote = rec
                    try:
                        from . import multiasset as _multiasset
                        ts_num = poll_start
                        if rec.get("mid") is not None:
                            _multiasset.HISTORY.push("live_mid", ts_num, rec["mid"])
                        if rec.get("spread_bps") is not None:
                            _multiasset.HISTORY.push("live_spread_bps", ts_num, rec["spread_bps"])
                        _multiasset.HISTORY.push("live_imb_top", ts_num, rec.get("imb_top") or 0.0)
                        _multiasset.HISTORY.push("live_imb_depth", ts_num, rec.get("imb_depth") or 0.0)
                        _multiasset.HISTORY.push("live_depth_total", ts_num, bd + ad)
                    except Exception:
                        pass
                    # Throttle disk writes (microstructure dataset doesn't need
                    # sub-second granularity).
                    if poll_start - self._last_disk_ts >= self.disk_min_interval:
                        with open(OB_LOG, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec) + "\n")
                        self.count += 1
                        self._last_disk_ts = poll_start
                    self.poll_count += 1
                    self.last_ts = ts_iso
                    self.last_error = None
            except Exception as e:
                self.last_error = str(e)[:120]
            # Vol-aware sleep, broken into 0.5s slices so Stop is responsive.
            target = self._poll_interval()
            slept = 0.0
            while slept < target:
                if self._stop.is_set():
                    break
                step = min(0.5, target - slept)
                time.sleep(step)
                slept += step

    def stop(self):
        self._stop.set()


OBLOGGER = ObLogger(fast_interval=1.0, slow_interval=10.0, disk_min_interval=5.0)
