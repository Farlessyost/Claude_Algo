"""Fetch BTC-USD 1-minute candles from Coinbase Exchange public API.

Resumable. Writes one .npz per 300-minute chunk to state/spot_1m/<start_unix>.npz.
Re-running skips chunks that already exist. Run direct:
    .venv\\Scripts\\python.exe -m backend.ml.fetch_spot [--years 3]
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "state" / "spot_1m"
OUT.mkdir(parents=True, exist_ok=True)

URL = "https://api.exchange.coinbase.com/products/BTC-USD/candles"
GRAN = 60
CHUNK_MIN = 300
CHUNK_SEC = CHUNK_MIN * GRAN
RATE_DELAY = 0.18  # ~5.5 req/s, well under the 10 req/s public cap


def chunk_path(start_unix: int) -> Path:
    return OUT / f"{start_unix}.npz"


def fetch_chunk(client: httpx.Client, start_unix: int) -> np.ndarray | None:
    start = datetime.fromtimestamp(start_unix, tz=timezone.utc).isoformat()
    end = datetime.fromtimestamp(start_unix + CHUNK_SEC - GRAN, tz=timezone.utc).isoformat()
    params = {"granularity": GRAN, "start": start, "end": end}
    for attempt in range(6):
        try:
            r = client.get(URL, params=params, timeout=20.0)
            if r.status_code == 429:
                time.sleep(2.0 + attempt)
                continue
            r.raise_for_status()
            data = r.json()
            if not isinstance(data, list):
                return None
            if not data:
                return np.zeros((0, 6), dtype=np.float64)
            arr = np.array(data, dtype=np.float64)  # [t, low, high, open, close, volume]
            arr = arr[np.argsort(arr[:, 0])]
            return arr
        except (httpx.HTTPError, httpx.TimeoutException):
            time.sleep(2.0 * (attempt + 1))
    return None


def main(years: float = 3.0) -> int:
    end_unix = int(time.time()) // GRAN * GRAN
    start_unix = end_unix - int(years * 365 * 24 * 3600)
    start_unix = (start_unix // CHUNK_SEC) * CHUNK_SEC
    chunks = list(range(start_unix, end_unix, CHUNK_SEC))
    todo = [c for c in chunks if not chunk_path(c).exists()]
    print(
        f"range {datetime.fromtimestamp(start_unix, tz=timezone.utc)} -> "
        f"{datetime.fromtimestamp(end_unix, tz=timezone.utc)} | "
        f"chunks total={len(chunks)} todo={len(todo)}",
        flush=True,
    )
    if not todo:
        return 0
    headers = {"User-Agent": "Claude_Algo/1.0 research"}
    failed = 0
    last_log = time.time()
    with httpx.Client(headers=headers) as client:
        for i, c in enumerate(todo):
            arr = fetch_chunk(client, c)
            if arr is None:
                failed += 1
                print(f"chunk {c} FAILED", file=sys.stderr, flush=True)
                time.sleep(1.0)
                continue
            np.savez_compressed(chunk_path(c), candles=arr)
            time.sleep(RATE_DELAY)
            if time.time() - last_log > 10.0:
                pct = 100.0 * (i + 1) / len(todo)
                ts = datetime.fromtimestamp(c, tz=timezone.utc)
                print(f"  {i+1}/{len(todo)} ({pct:.1f}%) at {ts}", flush=True)
                last_log = time.time()
    print(f"done. failed={failed}", flush=True)
    return failed


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--years", type=float, default=3.0)
    args = p.parse_args()
    sys.exit(0 if main(args.years) == 0 else 1)
