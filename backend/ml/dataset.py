"""Load and window the spot 1m candle archive.

`load_spot()` stitches state/spot_1m/*.npz into one chronological numpy array
[ts, low, high, open, close, volume]. Missing minutes (exchange downtime,
gaps in returned candles) are NOT forward-filled — we expose them via the
boolean continuity mask returned alongside the windowed tensors.

`build_windows()` returns input features X of shape (N, L, F) and targets
y of shape (N, H) where:
  X features = [log_return, log_volume_z, high_low_range_bps, close_vs_open_bps]
  y          = next H log returns (1m steps)
Windows that straddle a minute-gap (continuous slot missing) are dropped.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPOT_DIR = ROOT / "state" / "spot_1m"


def load_spot() -> np.ndarray:
    """Return chronological [N, 6] array: ts, low, high, open, close, volume."""
    files = sorted(SPOT_DIR.glob("*.npz"), key=lambda p: int(p.stem))
    if not files:
        raise FileNotFoundError(f"no candle chunks in {SPOT_DIR}")
    parts = []
    for f in files:
        with np.load(f) as z:
            parts.append(z["candles"])
    arr = np.concatenate(parts, axis=0) if parts else np.zeros((0, 6))
    # dedupe by timestamp, keep first occurrence (chunks can overlap at edges)
    arr = arr[np.argsort(arr[:, 0], kind="mergesort")]
    _, idx = np.unique(arr[:, 0], return_index=True)
    arr = arr[np.sort(idx)]
    return arr


def featurize(candles: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert raw OHLCV into per-bar feature vectors. Returns (features, valid_mask).

    Features per bar:
      0: log return = ln(close_t / close_{t-1})
      1: log volume normalized by rolling 60-bar median
      2: (high - low) / close in bps
      3: (close - open) / open in bps
    """
    n = candles.shape[0]
    ts = candles[:, 0].astype(np.int64)
    low = candles[:, 1]
    high = candles[:, 2]
    open_ = candles[:, 3]
    close = candles[:, 4]
    volume = candles[:, 5]

    # contiguity mask: bar i continuous if ts[i] - ts[i-1] == 60
    contig = np.zeros(n, dtype=bool)
    contig[1:] = (ts[1:] - ts[:-1]) == 60

    safe_open = np.where(open_ > 0, open_, 1.0)
    safe_close_prev = np.where(close > 0, close, 1.0)

    log_ret = np.zeros(n, dtype=np.float64)
    log_ret[1:] = np.log(np.maximum(close[1:], 1e-9) / np.maximum(close[:-1], 1e-9))
    log_ret[~contig] = 0.0  # treat gap-bars as no-return rather than huge jumps

    # rolling median volume in a 60-bar window for normalization
    log_vol = np.log1p(np.maximum(volume, 0.0))
    win = 60
    cs = np.cumsum(log_vol)
    roll_mean = np.zeros(n)
    roll_mean[win:] = (cs[win:] - cs[:-win]) / win
    roll_mean[:win] = roll_mean[win] if n > win else 0.0
    vol_z = log_vol - roll_mean

    range_bps = (high - low) / np.maximum(close, 1e-9) * 1e4
    body_bps = (close - safe_open) / safe_open * 1e4

    feats = np.stack([log_ret, vol_z, range_bps, body_bps], axis=1).astype(np.float32)
    valid = contig  # bar usable iff it has a previous continuous bar
    return feats, valid


def build_windows(
    feats: np.ndarray,
    valid: np.ndarray,
    L: int = 256,
    H: int = 16,
    stride: int = 8,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build (X, y, anchor_ts_idx). X: (N,L,F). y: (N,H) of future log returns.

    A window is kept only if all L+H bars in [i-L+1, i+H] are continuous,
    i.e. valid[i-L+2 ... i+H] all true (the L-th input bar at index i predicts
    y[0..H-1] = feats[i+1..i+H, 0]).
    """
    n, F = feats.shape
    X_list, y_list, idx_list = [], [], []
    for end in range(L - 1, n - H, stride):
        seg_valid = valid[end - L + 2 : end + H + 1]
        if seg_valid.size < L + H - 1 or not seg_valid.all():
            continue
        X_list.append(feats[end - L + 1 : end + 1])
        y_list.append(feats[end + 1 : end + 1 + H, 0])  # next H log returns
        idx_list.append(end)
    if not X_list:
        return (
            np.zeros((0, L, F), dtype=np.float32),
            np.zeros((0, H), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
        )
    return (
        np.stack(X_list).astype(np.float32),
        np.stack(y_list).astype(np.float32),
        np.array(idx_list, dtype=np.int64),
    )


def summary() -> None:
    """Print a quick summary of what's on disk."""
    arr = load_spot()
    n = arr.shape[0]
    if n == 0:
        print("no data")
        return
    from datetime import datetime, timezone

    t0 = datetime.fromtimestamp(int(arr[0, 0]), tz=timezone.utc)
    t1 = datetime.fromtimestamp(int(arr[-1, 0]), tz=timezone.utc)
    feats, valid = featurize(arr)
    span_min = (arr[-1, 0] - arr[0, 0]) / 60
    coverage = 100.0 * valid.sum() / max(span_min, 1)
    print(f"candles: {n:,} bars from {t0} to {t1}")
    print(f"span:    {span_min:,.0f} minutes; continuous bars: {valid.sum():,} ({coverage:.2f}%)")
    print(f"feature stats (log_ret, vol_z, range_bps, body_bps):")
    for i, name in enumerate(["log_ret", "vol_z", "range_bps", "body_bps"]):
        v = feats[valid, i]
        print(f"  {name:10s} mean={v.mean():+.4e} std={v.std():.4e} min={v.min():+.4e} max={v.max():+.4e}")


if __name__ == "__main__":
    summary()
