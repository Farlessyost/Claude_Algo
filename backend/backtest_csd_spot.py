"""CSD backtest v2 on 3-year BTC spot 1m data, aggregated to 15m bars.

The v1 backtest on 9 days of Kalshi perp showed CSD risk has IC ~0.018 vs
forward vol (vs trailing vol's IC ~0.55) — likely because the dataset has
almost no regime shifts. This v2 fixes both problems:

  - Bigger dataset: 3y BTC spot 1m -> 15m bars (~105k samples, many regimes).
  - Better target: predict (forward_vol - trailing_vol) — the *change* in
    vol, not the level. Trailing vol predicts level trivially; the real
    CSD claim is that it leads the BREAK, not the persistence.
  - Drawdown prediction: a more directly useful target. AUC of CSD vs
    "did the next H bars contain a >X% drawdown".

Run:
    .\.venv\Scripts\python.exe -m backend.backtest_csd_spot
"""
from __future__ import annotations

import json
import math
import random
import statistics
import time as _time
from pathlib import Path
from typing import List, Tuple

import numpy as np

from . import csd
from .config import STATE_DIR
from .ml.dataset import load_spot


# ---------------------------------------------------------- aggregate to NM
def aggregate(candles: np.ndarray, minutes: int) -> List[dict]:
    """1m -> NM-minute bars. Drops bars that straddle a continuity gap."""
    if minutes <= 1:
        out = []
        for r in candles:
            out.append({"ts": int(r[0]), "low": r[1], "high": r[2],
                        "open": r[3], "close": r[4], "volume": r[5]})
        return out
    out = []
    block = minutes
    i = 0
    n = candles.shape[0]
    while i + block <= n:
        ts_first = int(candles[i, 0])
        ts_last = int(candles[i + block - 1, 0])
        if ts_last - ts_first != 60 * (block - 1):
            # gap in this window — skip a single 1m bar and retry
            i += 1
            continue
        block_arr = candles[i:i + block]
        out.append({
            "ts": ts_last,
            "open": float(block_arr[0, 3]),
            "high": float(block_arr[:, 2].max()),
            "low": float(block_arr[:, 1].min()),
            "close": float(block_arr[-1, 4]),
            "volume": float(block_arr[:, 5].sum()),
        })
        i += block
    return out


# ---------------------------------------------------------- helpers
def _returns(closes: List[float]) -> List[float]:
    out = [0.0]
    for i in range(1, len(closes)):
        p = closes[i - 1]
        out.append((closes[i] - p) / p if p else 0.0)
    return out


def _rolling_std(xs: List[float], w: int) -> List[float]:
    out = [0.0] * len(xs)
    for i in range(w, len(xs)):
        win = xs[i - w:i]
        m = sum(win) / w
        v = sum((x - m) ** 2 for x in win) / max(1, w - 1)
        out[i] = math.sqrt(v)
    return out


def _forward_vol(rets: List[float], i: int, H: int) -> float:
    end = min(len(rets), i + 1 + H)
    win = rets[i + 1:end]
    if len(win) < 2:
        return 0.0
    m = sum(win) / len(win)
    v = sum((x - m) ** 2 for x in win) / (len(win) - 1)
    return math.sqrt(v)


def _max_drawdown(closes: List[float], i: int, H: int) -> float:
    """Worst peak-to-trough drawdown across [i+1, i+H] (fractional)."""
    end = min(len(closes), i + 1 + H)
    if end <= i + 1:
        return 0.0
    peak = closes[i]
    mdd = 0.0
    for j in range(i + 1, end):
        peak = max(peak, closes[j])
        dd = (peak - closes[j]) / peak if peak else 0.0
        if dd > mdd:
            mdd = dd
    return mdd


def _spearman(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 4:
        return 0.0

    def ranks(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx = ranks(xs); ry = ranks(ys)
    n = len(xs)
    mx = sum(rx) / n; my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    den = dx * dy
    return num / den if den else 0.0


def _auc(scores: List[float], labels: List[int]) -> float:
    pos = [scores[i] for i in range(len(scores)) if labels[i] == 1]
    neg = [scores[i] for i in range(len(scores)) if labels[i] == 0]
    if not pos or not neg:
        return 0.5
    combined = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg],
                       key=lambda t: t[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if combined[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j + 1
    n_pos = len(pos); n_neg = len(neg)
    U = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return U / (n_pos * n_neg)


# ---------------------------------------------------------- rolling CSD
def build_csd_series(bars: List[dict],
                     fv_period: int = 32,
                     window: int = 96) -> List[float]:
    """Walk-forward CSD risk per bar. Uses log-volume diffs and range-bps as
    multivariate extras (these are stationary by construction)."""
    closes = [b["close"] for b in bars]
    log_vols = [math.log(max(b.get("volume", 1.0), 1.0)) for b in bars]
    d_logvol = [0.0] + [log_vols[i] - log_vols[i - 1] for i in range(1, len(log_vols))]
    range_bps = [((b["high"] - b["low"]) / b["close"]) if b["close"] > 0 else 0.0 for b in bars]
    extras = {"d_logvol": d_logvol, "range": range_bps}

    bundles = csd.rolling_metrics(closes, fv_period=fv_period,
                                    window=window, step=1, extras_series=extras)

    risk_hist_w = max(window, 200)
    risks: List[float] = [0.0] * len(bars)
    history: List[dict] = []
    for i, b in enumerate(bundles):
        if not b:
            continue
        ref = [h for h in history[-risk_hist_w:] if h]
        score = csd.csd_score(b, ref) if len(ref) >= 12 else 0.0
        risks[i] = csd.csd_risk(score)
        history.append(b)
    return risks


# ---------------------------------------------------------- tests
def test_predict_vol_level(bars, risks, H_set=(3, 6, 12, 24, 48)) -> dict:
    closes = [b["close"] for b in bars]
    rets = _returns(closes)
    trail = _rolling_std(rets, 24)
    out = {}
    for H in H_set:
        xs_r, xs_t, ys = [], [], []
        for i in range(len(bars) - H - 1):
            if risks[i] == 0.0 or trail[i] == 0.0:
                continue
            fv = _forward_vol(rets, i, H)
            if fv <= 0:
                continue
            xs_r.append(risks[i]); xs_t.append(trail[i]); ys.append(fv)
        out[H] = {
            "n": len(ys),
            "ic_csd": round(_spearman(xs_r, ys), 4),
            "ic_trail": round(_spearman(xs_t, ys), 4),
            "marginal": round(_spearman(xs_r, ys) - _spearman(xs_t, ys), 4),
        }
    return out


def test_predict_vol_change(bars, risks, H_set=(3, 6, 12, 24)) -> dict:
    """The harder/proper CSD test: predict (forward_vol - trailing_vol)."""
    closes = [b["close"] for b in bars]
    rets = _returns(closes)
    trail = _rolling_std(rets, 24)
    out = {}
    for H in H_set:
        xs_r, xs_t, ys = [], [], []
        for i in range(len(bars) - H - 1):
            if risks[i] == 0.0 or trail[i] == 0.0:
                continue
            fv = _forward_vol(rets, i, H)
            if fv <= 0:
                continue
            change = fv - trail[i]
            xs_r.append(risks[i]); xs_t.append(trail[i]); ys.append(change)
        out[H] = {
            "n": len(ys),
            "ic_csd_vs_change": round(_spearman(xs_r, ys), 4),
            "ic_trail_vs_change": round(_spearman(xs_t, ys), 4),
        }
    return out


def test_predict_drawdown(bars, risks, H: int = 12,
                           dd_threshold: float = 0.01) -> dict:
    """AUC of CSD risk predicting "does the next H bars contain a >X% peak-to-trough drawdown"."""
    closes = [b["close"] for b in bars]
    rets = _returns(closes)
    trail = _rolling_std(rets, 24)
    risks_arr, trail_arr, labels = [], [], []
    for i in range(len(bars) - H - 1):
        if risks[i] == 0.0 or trail[i] == 0.0:
            continue
        mdd = _max_drawdown(closes, i, H)
        risks_arr.append(risks[i])
        trail_arr.append(trail[i])
        labels.append(1 if mdd >= dd_threshold else 0)
    if not labels:
        return {"n": 0}
    return {
        "n": len(labels),
        "pos_count": sum(labels),
        "pos_rate": round(sum(labels) / len(labels), 4),
        "horizon_bars": H,
        "dd_threshold": dd_threshold,
        "auc_csd": round(_auc(risks_arr, labels), 4),
        "auc_trail": round(_auc(trail_arr, labels), 4),
    }


def test_quintile_dd(bars, risks, H: int = 12) -> dict:
    """Sort bars by CSD risk into quintiles, report forward MDD per quintile."""
    closes = [b["close"] for b in bars]
    pairs = []
    for i in range(len(bars) - H - 1):
        if risks[i] == 0.0:
            continue
        mdd = _max_drawdown(closes, i, H)
        fv = _forward_vol(_returns(closes), i, H) if False else None  # too slow per-bar
        pairs.append((risks[i], mdd))
    if len(pairs) < 25:
        return {"n": len(pairs)}
    pairs.sort(key=lambda p: p[0])
    Q = 5
    bs = len(pairs) // Q
    qs = []
    for q in range(Q):
        chunk = pairs[q * bs:(q + 1) * bs if q < Q - 1 else len(pairs)]
        qs.append({
            "q": q + 1, "n": len(chunk),
            "mean_risk": round(statistics.mean(p[0] for p in chunk), 4),
            "mean_fwd_mdd": round(statistics.mean(p[1] for p in chunk), 6),
            "p90_fwd_mdd": round(sorted(p[1] for p in chunk)[int(0.9 * len(chunk))], 6),
        })
    ratio = qs[-1]["mean_fwd_mdd"] / qs[0]["mean_fwd_mdd"] if qs[0]["mean_fwd_mdd"] else float("inf")
    return {"n": len(pairs), "horizon_bars": H,
             "quintiles": qs, "Q5_over_Q1_mdd": round(ratio, 3)}


def test_lead_lag(bars, risks, horizons=(-12, -6, -3, -1, 1, 3, 6, 12, 24)) -> dict:
    closes = [b["close"] for b in bars]
    rets = _returns(closes)
    out = {}
    for H in horizons:
        xs, ys = [], []
        for i in range(max(0, -H), len(bars) - max(0, H) - 1):
            if risks[i] == 0.0:
                continue
            if H > 0:
                fv = _forward_vol(rets, i, H)
            else:
                start = max(0, i + H); end = i + 1
                win = rets[start:end]
                if len(win) < 2:
                    continue
                m = sum(win) / len(win)
                v = sum((x - m) ** 2 for x in win) / (len(win) - 1)
                fv = math.sqrt(v)
            if fv <= 0:
                continue
            xs.append(risks[i]); ys.append(fv)
        out[H] = {"n": len(xs), "ic": round(_spearman(xs, ys), 4) if len(xs) >= 30 else None}
    return out


# ---------------------------------------------------------- main
def main(timeframe_min: int = 15, window: int = 96, fv_period: int = 32,
         max_bars: int = 0):
    t0 = _time.time()
    print("Loading 3y BTC spot 1m archive…")
    cs = load_spot()
    print(f"  raw 1m bars: {cs.shape[0]:,}")
    print(f"Aggregating to {timeframe_min}m bars (gap-aware)…")
    bars = aggregate(cs, timeframe_min)
    if max_bars and len(bars) > max_bars:
        bars = bars[-max_bars:]
        print(f"  truncated to last {max_bars:,} bars for speed")
    print(f"  {len(bars):,} {timeframe_min}m bars  ({(_time.time() - t0):.1f}s)\n")

    cache_path = STATE_DIR / f"csd_risks_cache_{timeframe_min}m_w{window}_fv{fv_period}.json"
    if cache_path.exists():
        print(f"Loading cached risks from {cache_path.name}…")
        risks = json.loads(cache_path.read_text("utf-8"))
        if len(risks) != len(bars):
            print(f"  cache mismatch ({len(risks)} vs {len(bars)}); recomputing")
            risks = None
        else:
            print(f"  loaded {len(risks):,} risks")
    else:
        risks = None
    if risks is None:
        print(f"Computing rolling CSD risk (window={window}, fv_period={fv_period})…")
        risks = build_csd_series(bars, fv_period=fv_period, window=window)
        cache_path.write_text(json.dumps(risks), encoding="utf-8")
        print(f"  cached to {cache_path.name}")
    nz = sum(1 for r in risks if r != 0.0)
    rmean = statistics.mean(r for r in risks if r != 0.0) if nz else 0
    rstd = statistics.stdev(r for r in risks if r != 0.0) if nz > 1 else 0
    print(f"  bars with risk defined : {nz:,}/{len(risks):,}")
    print(f"  risk mean ± std        : {rmean:.3f} ± {rstd:.3f}")
    print(f"  total elapsed          : {(_time.time() - t0):.1f}s\n")

    print("=" * 76)
    print("T1 — vs forward vol LEVEL (sanity: should be small if trail dominates)")
    print("=" * 76)
    t1 = test_predict_vol_level(bars, risks)
    print(f"  {'H':>5}{'n':>10}{'IC(CSD)':>12}{'IC(trail)':>12}{'marginal':>12}")
    for H, d in t1.items():
        print(f"  {H:>5}{d['n']:>10}{d['ic_csd']:>+12.4f}{d['ic_trail']:>+12.4f}{d['marginal']:>+12.4f}")

    print("\n" + "=" * 76)
    print("T2 — vs forward vol CHANGE (the proper CSD test)")
    print("=" * 76)
    t2 = test_predict_vol_change(bars, risks)
    print(f"  {'H':>5}{'n':>10}{'IC(CSD->dvol)':>16}{'IC(trail->dvol)':>18}")
    for H, d in t2.items():
        print(f"  {H:>5}{d['n']:>10}{d['ic_csd_vs_change']:>+16.4f}{d['ic_trail_vs_change']:>+18.4f}")

    print("\n" + "=" * 76)
    print("T3 — DRAWDOWN PREDICTION (H=12 bars, threshold 1%)")
    print("=" * 76)
    t3 = test_predict_drawdown(bars, risks, H=12, dd_threshold=0.01)
    print(f"  bars                : {t3.get('n', 0):,}")
    print(f"  positive event rate : {t3.get('pos_rate', 0):.3f}")
    print(f"  AUC (CSD)           : {t3.get('auc_csd', 0):.4f}")
    print(f"  AUC (trail vol)     : {t3.get('auc_trail', 0):.4f}")

    print("\n  Same test at threshold 2%…")
    t3b = test_predict_drawdown(bars, risks, H=12, dd_threshold=0.02)
    print(f"  positive event rate : {t3b.get('pos_rate', 0):.3f}")
    print(f"  AUC (CSD)           : {t3b.get('auc_csd', 0):.4f}")
    print(f"  AUC (trail vol)     : {t3b.get('auc_trail', 0):.4f}")

    print("\n" + "=" * 76)
    print("T4 — DRAWDOWN QUINTILES")
    print("=" * 76)
    t4 = test_quintile_dd(bars, risks, H=12)
    if "quintiles" in t4:
        print(f"  {'Q':>3}{'n':>8}{'mean_risk':>12}{'mean_fwd_mdd':>16}{'p90_fwd_mdd':>16}")
        for q in t4["quintiles"]:
            print(f"  {q['q']:>3}{q['n']:>8}{q['mean_risk']:>+12.4f}"
                  f"{q['mean_fwd_mdd']:>16.6f}{q['p90_fwd_mdd']:>16.6f}")
        print(f"  Q5/Q1 mean fwd MDD ratio: {t4['Q5_over_Q1_mdd']:.3f}")

    print("\n" + "=" * 76)
    print("T5 — LEAD-LAG PROFILE")
    print("=" * 76)
    t5 = test_lead_lag(bars, risks)
    print(f"  {'H':>5}{'n':>10}{'IC':>10}")
    for H, d in t5.items():
        ic = d['ic']
        print(f"  {H:>+5}{d['n']:>10}"
              f"{(ic if ic is not None else 0):>+10.4f}"
              f"{'  insuff' if ic is None else ''}")

    out = {"timeframe_min": timeframe_min, "window": window, "fv_period": fv_period,
            "n_bars": len(bars), "n_risk_defined": nz,
            "T1_vol_level": t1, "T2_vol_change": t2,
            "T3_drawdown_1pct": t3, "T3_drawdown_2pct": t3b,
            "T4_dd_quintiles": t4, "T5_lead_lag": t5}
    p = STATE_DIR / "csd_backtest_spot.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults written to {p}  (total {_time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
