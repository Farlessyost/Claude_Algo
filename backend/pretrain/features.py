"""Feature engineering that's identical for Binance spot rows AND Kalshi
rows, so the encoder doesn't have to relearn anything venue-specific.

Per-bar inputs the encoder consumes:
  0. log return (close[t] / close[t-1])
  1. abs log return (proxy for instant vol)
  2. range_bps    = (high - low) / close * 1e4
  3. volume_z     = rolling z-score of log(volume + 1)

Outputs:
  X: shape (N, T, 4)  rolling windows of per-bar features
  y_ret: log return summed over the next K bars (the supervised target)
  y_vol: realized vol (std of returns) over the next K bars (log)
"""
from __future__ import annotations

import math
from typing import List, Tuple

import numpy as np


def per_bar_features(rows: List[dict]) -> np.ndarray:
    """rows must have keys: c (close), h (high), l (low), v (volume)."""
    n = len(rows)
    feat = np.zeros((n, 4), dtype=np.float32)
    closes = np.array([r["c"] for r in rows], dtype=np.float64)
    highs = np.array([r["h"] for r in rows], dtype=np.float64)
    lows = np.array([r["l"] for r in rows], dtype=np.float64)
    vols = np.array([r["v"] for r in rows], dtype=np.float64)
    logv = np.log(vols + 1.0)
    # rolling z-score of log volume
    win = 240   # 4h at 1m
    mean = np.zeros(n); std = np.ones(n)
    if n > win:
        s = np.cumsum(logv); s2 = np.cumsum(logv * logv)
        for i in range(win, n):
            a = s[i] - s[i - win]
            b = s2[i] - s2[i - win]
            m = a / win
            v = max(b / win - m * m, 1e-12)
            mean[i] = m; std[i] = math.sqrt(v)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.zeros(n)
        r[1:] = np.log(closes[1:] / np.maximum(closes[:-1], 1e-9))
    feat[:, 0] = r.astype(np.float32)
    feat[:, 1] = np.abs(r).astype(np.float32)
    rng = np.where(closes > 0, (highs - lows) / closes * 1e4, 0.0)
    feat[:, 2] = np.clip(rng, 0, 1000).astype(np.float32)
    vz = np.where(std > 0, (logv - mean) / std, 0.0)
    feat[:, 3] = np.clip(vz, -5, 5).astype(np.float32)
    return feat


def make_windows(features: np.ndarray, closes: np.ndarray,
                 seq_len: int = 60, horizon: int = 3
                 ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (X, y_ret, y_vol). Drops the last `horizon` rows so the labels
    are always realized."""
    n = features.shape[0]
    if n < seq_len + horizon + 2:
        return (np.zeros((0, seq_len, features.shape[1]), dtype=np.float32),
                np.zeros((0,), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))
    # log returns
    r = np.zeros(n)
    r[1:] = np.log(closes[1:] / np.maximum(closes[:-1], 1e-9))
    # Future K-bar sum return and realized vol
    fut_ret = np.zeros(n); fut_vol = np.zeros(n)
    csum = np.concatenate([[0.0], np.cumsum(r)])
    csum2 = np.concatenate([[0.0], np.cumsum(r * r)])
    for i in range(n - horizon):
        a = csum[i + horizon + 1] - csum[i + 1]            # sum r[i+1..i+horizon]
        b = csum2[i + horizon + 1] - csum2[i + 1]
        m = a / horizon
        v = max(b / horizon - m * m, 1e-12)
        fut_ret[i] = a
        fut_vol[i] = math.log(math.sqrt(v) + 1e-9)
    # Build windows ending at i (inclusive), label = fut_ret[i]
    starts = np.arange(seq_len - 1, n - horizon)
    X = np.stack([features[s - seq_len + 1: s + 1] for s in starts])
    y_ret = fut_ret[starts].astype(np.float32)
    y_vol = fut_vol[starts].astype(np.float32)
    return X.astype(np.float32), y_ret, y_vol
