"""Fetch ~3 years of 1m BTC-USD candles from Coinbase Exchange (the public
unauthenticated endpoint). Binance's /api/v3/klines is blocked from US IPs
(HTTP 451), so Coinbase is the next-best public source with deep history.

Endpoint:    https://api.exchange.coinbase.com/products/BTC-USD/candles
Granularity: 60 seconds (1m).
Per call:    up to 300 candles. ~525k bars/year -> ~1750 calls/year.

We page BACKWARDS in 300-bar windows from the most recent unfilled timestamp
to the year-floor. The cache is resumable: each run reads existing rows,
fills any gap up to "now", and appends.

Run:  python -m backend.pretrain.fetch_spot [--years 3]
"""
from __future__ import annotations

import argparse
import gzip
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path
from typing import List

from ..config import STATE_DIR

OUT = STATE_DIR / "pretrain" / "btc_usd_1m.jsonl.gz"
OUT.parent.mkdir(parents=True, exist_ok=True)

BASE = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
GRAN = 60          # seconds
LIMIT = 300        # candles per request


def _request(start_iso: str, end_iso: str) -> List[list]:
    q = urllib.parse.urlencode({
        "start": start_iso, "end": end_iso, "granularity": GRAN,
    })
    url = f"{BASE}?{q}"
    req = urllib.request.Request(url, headers={
        "User-Agent": "claude-algo/1.0",
        "Accept": "application/json",
    })
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read().decode("utf-8"))


def _existing_times() -> set:
    if not OUT.exists():
        return set()
    seen = set()
    with gzip.open(OUT, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                seen.add(int(json.loads(line)["t"]))
            except Exception:
                continue
    return seen


def _iso(ts: int) -> str:
    """Coinbase wants RFC3339 timestamps in UTC."""
    from datetime import datetime, timezone
    return datetime.fromtimestamp(ts, timezone.utc).isoformat().replace("+00:00", "Z")


def fetch(years: float = 3.0) -> int:
    """Paginate BACKWARDS from now to (now - years). Each request returns up
    to 300 candles ordered newest-first; we de-duplicate against the cache."""
    seen = _existing_times()
    end_ts = int(time.time())
    floor_ts = end_ts - int(years * 365 * 24 * 3600)
    print(f"target window: {_iso(floor_ts)} -> {_iso(end_ts)}  "
          f"(cache has {len(seen):,} bars)")
    appended = 0
    cursor = end_ts
    backoff = 0.6
    f = gzip.open(OUT, "at", encoding="utf-8")
    try:
        while cursor > floor_ts:
            start = max(floor_ts, cursor - LIMIT * GRAN)
            tries = 0
            rows = None
            while True:
                try:
                    rows = _request(_iso(start), _iso(cursor))
                    break
                except urllib.error.HTTPError as e:
                    tries += 1
                    if e.code == 429 or 500 <= e.code < 600:
                        # rate limited or transient -> back off
                        sleep = min(30.0, backoff * (2 ** (tries - 1)))
                        print(f"  HTTP {e.code} at {start}; backoff {sleep:.1f}s")
                        time.sleep(sleep)
                        continue
                    print(f"  HTTP {e.code} at {start}; skipping window")
                    rows = []; break
                except Exception as e:
                    tries += 1
                    sleep = min(10.0, backoff * (2 ** (tries - 1)))
                    print(f"  network error at {start}: {e}; backoff {sleep:.1f}s")
                    time.sleep(sleep)
                    if tries > 6:
                        rows = []; break
            if not rows:
                cursor = start - GRAN
                continue
            # rows: [[time, low, high, open, close, volume], ...] newest first
            new_added = 0
            for k in rows:
                t = int(k[0])
                if t in seen:
                    continue
                seen.add(t)
                f.write(json.dumps({
                    "t": t * 1000,             # store as ms (matches Binance schema)
                    "o": float(k[3]), "h": float(k[2]),
                    "l": float(k[1]), "c": float(k[4]),
                    "v": float(k[5]),
                }) + "\n")
                appended += 1; new_added += 1
            oldest = min(int(k[0]) for k in rows)
            cursor = oldest - GRAN
            if appended and appended % 25_000 < new_added:
                f.flush()
                print(f"  ...{appended:,} bars added; back to {_iso(cursor)}")
            time.sleep(0.25)   # Coinbase public limit is ~10/s; stay well under
    finally:
        f.close()
    print(f"Appended {appended:,} bars to {OUT}")
    return appended


def load_all() -> list:
    rows = []
    if not OUT.exists():
        return rows
    with gzip.open(OUT, "rt", encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    rows.sort(key=lambda r: r["t"])
    out = []; last_t = -1
    for r in rows:
        if r["t"] == last_t:
            continue
        out.append(r); last_t = r["t"]
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=3.0)
    args = ap.parse_args()
    fetch(args.years)
    rows = load_all()
    if rows:
        print(f"Cache size: {len(rows):,} bars "
              f"({(rows[-1]['t']-rows[0]['t'])/(86400*1000):.1f} days)")
