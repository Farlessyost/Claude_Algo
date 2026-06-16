"""Validate the blended directional alpha against MPC-alone.

What we test:
  T1  Marginal IC of spot-lead tilt vs forward Kalshi return, controlling
       for MPC's own alpha. Real signal must beat zero AND add information
       beyond MPC.
  T2  Returns: simulate_hybrid run on Kalshi 3m bars using:
        - MPC alone (baseline, validated config)
        - MPC + spot_lead (different weights)
        - MPC + spot_lead + a synthetic funding tilt (just to confirm the
          plumbing — we don't have historical funding data so this is mocked)
  T3  Walk-forward 4 folds. Per-fold winner has to be consistent.

Only spot-lead is BACKTESTABLE here because we have 3-year BTC spot 1m
data. Funding rate and OI history would require pulling from another
source for historical data; until that's done, those components stay at
weight 0 and accumulate from live data only.

Run:
    .\.venv\Scripts\python.exe -m backend.backtest_blended
"""
from __future__ import annotations

import json
import math
import statistics
from typing import Dict, List, Optional, Tuple

from . import csd, lab, signals, signals_blended
from .config import DEFAULT_STRATEGY_PARAMS, STATE_DIR


# ---------------------------------------------------------- helpers
def _spearman(xs, ys):
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
    return num / (dx * dy) if (dx * dy) else 0.0


def _partial_spearman(x, y, z):
    if len(x) != len(y) or len(y) != len(z) or len(x) < 6:
        return 0.0
    rx = []  # ranks
    def ranks(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs); i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx = ranks(x); ry = ranks(y); rz = ranks(z)
    n = len(rx)
    mz = sum(rz) / n; my = sum(ry) / n
    den = sum((rz[i] - mz) ** 2 for i in range(n))
    if den <= 0:
        return 0.0
    b = sum((rz[i] - mz) * (ry[i] - my) for i in range(n)) / den
    a = my - b * mz
    resid = [ry[i] - (a + b * rz[i]) for i in range(n)]
    return _spearman(rx, resid)


# ---------------------------------------------------------- data alignment
def load_spot_aligned_to_kalshi(kalshi_bars: List[dict]) -> List[Optional[float]]:
    """For each Kalshi bar, return the spot 1m close at the closest spot ts
    <= Kalshi ts. Returns None for bars that fall outside the spot archive
    coverage."""
    from .ml.dataset import load_spot
    import numpy as np
    print("  loading spot 1m archive…", flush=True)
    arr = load_spot()
    spot_ts = arr[:, 0].astype(int)   # unix seconds
    spot_close = arr[:, 4]            # close column
    print(f"  spot bars: {len(spot_ts):,} (range {spot_ts[0]} .. {spot_ts[-1]})", flush=True)

    out: List[Optional[float]] = []
    for b in kalshi_bars:
        kts = int(b["ts"])
        if kts < spot_ts[0] or kts > spot_ts[-1] + 600:
            out.append(None)
            continue
        idx = int(np.searchsorted(spot_ts, kts, side="right") - 1)
        if idx < 0:
            out.append(None)
        else:
            out.append(float(spot_close[idx]))
    coverage = sum(1 for x in out if x is not None) / max(1, len(out))
    print(f"  alignment coverage: {coverage*100:.1f}%", flush=True)
    return out


# ---------------------------------------------------------- blended position series
def position_series_with_blend(candles: List[dict],
                                 spot_aligned: List[Optional[float]],
                                 params: dict,
                                 blend_params: Optional[dict] = None
                                 ) -> Tuple[List[float], List[float], List[float]]:
    """Walk a blended-signal series across `candles`. Returns (pos, urgency,
    alpha_blended). Uses MPC's controller logic so the band / regime scaler
    stay intact, but the alpha-aim is `mpc_alpha + w_spot * spot_lead`.

    spot_aligned[i] is the spot close at the same timestamp as candles[i].
    None entries fall through with spot_tilt=0 for that bar (degrade
    gracefully when out of coverage).
    """
    bp = dict(signals_blended.DEFAULT_BLEND)
    bp.update(blend_params or {})
    closes = [c["close"] for c in candles]
    n = len(closes)
    if n < 5:
        return [0.0] * n, [0.0] * n, [0.0] * n

    vw = int(params.get("vol_win", 12))
    lb = int(params.get("lookback", 2))
    rw = int(params.get("regime_win", 8))
    beta = float(params.get("beta", 0.18))
    gain = float(params.get("gain", 1.0))
    band = float(params.get("band", 0.5))
    er_cap = float(params.get("er_cap", 1.0))

    # 1-bar log returns of Kalshi (same as MPC's internal)
    r = [0.0] + [(closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] else 0.0
                 for i in range(1, n)]
    er = signals.efficiency_ratio(closes, rw)

    spot_lookback = int(bp["spot_lookback"])
    spot_hist_w = int(bp["spot_history_for_std"])
    w_spot = float(bp["w_spot_lead"])
    w_mpc = float(bp["w_mpc"])

    pos = [0.0] * n
    urgency = [0.0] * n
    blended = [0.0] * n
    cur = 0.0
    start = max(vw, lb, rw, spot_lookback + 8) + 1

    for i in range(start, n):
        # --- MPC alpha (same as signals.mpc_with_aux internals) ---
        move = sum(r[i - lb + 1:i + 1])
        vol = signals._std(r[i - vw:i]) or 1e-9
        mpc_alpha = -beta * (move / vol)
        scale = max(0.0, 1.0 - er[i] / max(er_cap, 1e-6))
        mpc_alpha *= scale * gain

        # --- spot-lead tilt ---
        spot_tilt = 0.0
        if w_spot > 0 and spot_aligned[i] is not None:
            # Build a rolling spot window aligned up to bar i, dropping None
            window = []
            for j in range(max(0, i - spot_hist_w * 2), i + 1):
                v = spot_aligned[j]
                if v is not None:
                    window.append(v)
            if len(window) >= spot_lookback + 8:
                spot_tilt = signals_blended.spot_lead_tilt(
                    window, lookback=spot_lookback, history_for_std=spot_hist_w)

        # --- blend ---
        # MPC is in z-score-ish units (vol-normalized reversion), spot is z-scored.
        # Both are comparable. Sum them with weights.
        aim_raw = w_mpc * mpc_alpha + w_spot * spot_tilt
        aim = max(-1.0, min(1.0, aim_raw))
        urgency[i] = abs(aim_raw)
        blended[i] = aim_raw
        if cur < aim - band:
            cur = aim - band
        elif cur > aim + band:
            cur = aim + band
        pos[i] = cur

    return pos, urgency, blended


# ---------------------------------------------------------- IC / return tests
def _returns(closes):
    out = [0.0]
    for i in range(1, len(closes)):
        p = closes[i - 1]
        out.append((closes[i] - p) / p if p else 0.0)
    return out


def test_marginal_ic(bars, spot_aligned, params,
                       H: int = 1, blend_params=None) -> dict:
    """IC of (MPC alpha alone) and (blended alpha) vs forward Kalshi return.
    Also: partial IC of spot_lead alone, controlling for MPC."""
    closes = [c["close"] for c in bars]
    rets = _returns(closes)

    # Build per-bar MPC alpha and spot tilt as separate columns
    bp = dict(signals_blended.DEFAULT_BLEND)
    bp.update(blend_params or {})
    spot_hist_w = int(bp["spot_history_for_std"])
    spot_lookback = int(bp["spot_lookback"])

    vw = int(params.get("vol_win", 12))
    lb = int(params.get("lookback", 2))
    rw = int(params.get("regime_win", 8))
    beta = float(params.get("beta", 0.18))
    gain = float(params.get("gain", 1.0))
    er_cap = float(params.get("er_cap", 1.0))

    r = [0.0] + [(closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] else 0.0
                 for i in range(1, len(closes))]
    er = signals.efficiency_ratio(closes, rw)

    n = len(bars)
    start = max(vw, lb, rw, spot_lookback + 8) + 1
    mpc_arr, spot_arr, blended_arr, y = [], [], [], []
    for i in range(start, n - H - 1):
        # MPC alpha
        move = sum(r[i - lb + 1:i + 1])
        vol = signals._std(r[i - vw:i]) or 1e-9
        mpc_a = -beta * (move / vol)
        scale = max(0.0, 1.0 - er[i] / max(er_cap, 1e-6))
        mpc_a *= scale * gain

        # Spot tilt
        s_a = 0.0
        if spot_aligned[i] is not None:
            window = [v for v in spot_aligned[max(0, i - spot_hist_w * 2):i + 1]
                       if v is not None]
            if len(window) >= spot_lookback + 8:
                s_a = signals_blended.spot_lead_tilt(
                    window, lookback=spot_lookback, history_for_std=spot_hist_w)

        # Forward H-bar return
        end = min(len(closes), i + 1 + H)
        if end <= i + 1:
            continue
        fwd_ret = (closes[end - 1] - closes[i]) / closes[i] if closes[i] else 0.0

        w_mpc = float(bp["w_mpc"]); w_spot = float(bp["w_spot_lead"])
        mpc_arr.append(mpc_a)
        spot_arr.append(s_a)
        blended_arr.append(w_mpc * mpc_a + w_spot * s_a)
        y.append(fwd_ret)

    ic_mpc = _spearman(mpc_arr, y)
    ic_spot = _spearman(spot_arr, y)
    ic_blended = _spearman(blended_arr, y)
    ic_spot_given_mpc = _partial_spearman(spot_arr, y, mpc_arr)
    return {
        "H": H, "n": len(y),
        "ic_mpc": round(ic_mpc, 4),
        "ic_spot_lead": round(ic_spot, 4),
        "ic_spot_lead_partial_on_mpc": round(ic_spot_given_mpc, 4),
        "ic_blended": round(ic_blended, 4),
        "marginal_blended_vs_mpc": round(ic_blended - ic_mpc, 4),
    }


def test_walkforward_returns(bars, spot_aligned, params, K: int = 4,
                              leverage: float = 5.8,
                              hs_bps: float = 4.0,
                              variants: Optional[dict] = None) -> dict:
    """Walk-forward simulate_hybrid returns. variants = {label: blend_params}.
    Always includes baseline (MPC-only). Returns per-fold + aggregate."""
    variants = variants or {}
    fs = len(bars) // K
    folds = []
    agg = {"baseline": {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0,
                          "trades": 0, "wins": 0}}
    for label in variants:
        agg[label] = {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0,
                       "trades": 0, "wins": 0}

    for fi in range(K):
        lo = fi * fs
        hi = len(bars) if fi == K - 1 else (fi + 1) * fs
        fold_bars = bars[lo:hi]
        fold_spot = spot_aligned[lo:hi]
        rec = {"fold": fi + 1, "bars": len(fold_bars)}

        # Baseline: MPC-only
        pos_b, urg_b = signals.mpc_with_aux(fold_bars, params)
        res_b = signals.simulate_hybrid(fold_bars, pos_b, urg_b,
                                          k=float("inf"), chase_n=3,
                                          half_spread_bps=hs_bps, fee_bps=0.0,
                                          leverage=leverage)
        rec["baseline"] = res_b
        for k in ("return_pct", "max_dd_pct", "sharpe", "trades"):
            v = res_b.get(k, 0.0)
            if k == "max_dd_pct":
                agg["baseline"][k] = max(agg["baseline"][k], v)
            else:
                agg["baseline"][k] += v

        for label, bp in variants.items():
            pos_v, urg_v, _ = position_series_with_blend(
                fold_bars, fold_spot, params, blend_params=bp)
            res_v = signals.simulate_hybrid(fold_bars, pos_v, urg_v,
                                              k=float("inf"), chase_n=3,
                                              half_spread_bps=hs_bps,
                                              fee_bps=0.0, leverage=leverage)
            rec[label] = res_v
            for k in ("return_pct", "max_dd_pct", "sharpe", "trades"):
                v = res_v.get(k, 0.0)
                if k == "max_dd_pct":
                    agg[label][k] = max(agg[label][k], v)
                else:
                    agg[label][k] += v
        folds.append(rec)

    # Sharpe is mean-of-fold averaged
    for label in agg:
        agg[label]["sharpe"] /= K
    return {"folds": folds, "agg": agg, "K": K, "hs_bps": hs_bps,
             "leverage": leverage}


# ---------------------------------------------------------- main
def make_params() -> dict:
    p = dict(DEFAULT_STRATEGY_PARAMS)
    p.update({"vol_win": 12, "lookback": 2, "beta": 0.25, "k": 1.2,
              "z_cap": 3.5, "deadband_bps": 1.0, "regime_win": 8,
              "er_cap": 1.0, "gain": 1.0, "band": 0.5})
    return p


def main():
    print("Loading Kalshi 3m bars…")
    rows = lab.load(use_cache=True)
    bars = lab.aggregate(rows, 3)
    print(f"  {len(bars):,} 3m bars\n")

    print("Aligning spot prices to Kalshi timestamps…")
    spot_aligned = load_spot_aligned_to_kalshi(bars)
    n_covered = sum(1 for v in spot_aligned if v is not None)
    print(f"  spot-aligned bars: {n_covered:,}/{len(bars):,}\n")

    params = make_params()

    print("=" * 78)
    print("T1 — MARGINAL IC vs forward 1-bar Kalshi return")
    print("=" * 78)
    bp = dict(signals_blended.DEFAULT_BLEND)
    t1 = test_marginal_ic(bars, spot_aligned, params, H=1, blend_params=bp)
    print(f"  H=1 bars, n={t1['n']:,}")
    print(f"  IC(MPC alone)            : {t1['ic_mpc']:+.4f}")
    print(f"  IC(spot_lead alone)      : {t1['ic_spot_lead']:+.4f}")
    print(f"  IC(spot_lead | MPC)      : {t1['ic_spot_lead_partial_on_mpc']:+.4f}  "
          f"({'positive' if t1['ic_spot_lead_partial_on_mpc'] > 0.01 else 'weak/noise'})")
    print(f"  IC(blended = MPC + spot) : {t1['ic_blended']:+.4f}")
    print(f"  marginal blended vs MPC  : {t1['marginal_blended_vs_mpc']:+.4f}")

    print("\n  At H=3 horizon (matches Kalshi 3m cadence):")
    t1b = test_marginal_ic(bars, spot_aligned, params, H=3, blend_params=bp)
    print(f"  H=3 bars, n={t1b['n']:,}")
    print(f"  IC(MPC alone)            : {t1b['ic_mpc']:+.4f}")
    print(f"  IC(spot_lead | MPC)      : {t1b['ic_spot_lead_partial_on_mpc']:+.4f}")
    print(f"  IC(blended = MPC + spot) : {t1b['ic_blended']:+.4f}")

    print("\n" + "=" * 78)
    print("T2/T3 — WALK-FORWARD HYBRID SIM (validated config: chase_n=3, hs=4bps)")
    print("=" * 78)
    variants = {
        "blend_w0.15": dict(signals_blended.DEFAULT_BLEND, w_spot_lead=0.15),
        "blend_w0.30": dict(signals_blended.DEFAULT_BLEND, w_spot_lead=0.30),
        "blend_w0.50": dict(signals_blended.DEFAULT_BLEND, w_spot_lead=0.50),
        "blend_w0.80": dict(signals_blended.DEFAULT_BLEND, w_spot_lead=0.80),
    }
    res = test_walkforward_returns(bars, spot_aligned, params, K=4,
                                     hs_bps=4.0, variants=variants)
    print(f"\n  {'fold':<6}{'baseline':>20}", end="")
    for label in variants:
        print(f"  {label:>16}", end="")
    print()
    for rec in res["folds"]:
        ret_b = rec["baseline"]["return_pct"]
        dd_b = rec["baseline"]["max_dd_pct"]
        print(f"  {rec['fold']:<6}{ret_b:>+10.2f}% dd{dd_b:>5.2f}%", end="")
        for label in variants:
            r = rec[label]
            print(f"  {r['return_pct']:>+8.2f}% dd{r['max_dd_pct']:>5.2f}%", end="")
        print()
    print(f"\n  aggregate:")
    base = res["agg"]["baseline"]
    print(f"    baseline     : ret {base['return_pct']:+.2f}%  "
          f"max_dd {base['max_dd_pct']:.2f}%  sharpe {base['sharpe']:+.2f}  "
          f"trades {base['trades']}")
    for label in variants:
        a = res["agg"][label]
        d_ret = a["return_pct"] - base["return_pct"]
        d_dd = a["max_dd_pct"] - base["max_dd_pct"]
        d_sh = a["sharpe"] - base["sharpe"]
        marker = "Y" if d_ret > 0 and d_sh > 0 else "?" if d_ret > -1 else "N"
        print(f"    {label:<13}: ret {a['return_pct']:+.2f}% (d {d_ret:+.2f}pp)  "
              f"max_dd {a['max_dd_pct']:.2f}% (d {d_dd:+.2f}pp)  "
              f"sharpe {a['sharpe']:+.2f} (d {d_sh:+.2f})  {marker}")

    # Save full results
    out = {
        "n_bars": len(bars), "n_spot_covered": n_covered,
        "T1_ic_h1": t1, "T1_ic_h3": t1b,
        "T23_walkforward": res,
    }
    p = STATE_DIR / "blended_backtest.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nResults written to {p}")


if __name__ == "__main__":
    main()
