"""Combined ablation: how much does each layer contribute when stacked?

Tests on Kalshi 3m bars (5,214) + aligned BTC spot 1m (3y archive),
4-fold walk-forward, validated maker config (chase_n=3, hs=4bps).

Ladder of variants:
  A. baseline           MPC-only, no governors, no blend
  B. + CSD skew gate    Refined skew-only CSD governor at gate T=0.95
  C. + adaptive thresh  CSD threshold drops to 0.80 when atr_pct >= 0.20
  D. + spot-lead blend  blend_w_spot_lead = 0.30 (the validated weight)
  E. ALL stacked        CSD + adaptive + blend at the same time

Reports per-fold and aggregate. Marginal contribution = (variant) - (prior).

Run:
    .\.venv\Scripts\python.exe -m backend.ablate_combined
"""
from __future__ import annotations

import json
import math
import statistics
from typing import List, Optional, Tuple

from . import csd, lab, signals, signals_blended
from .config import DEFAULT_STRATEGY_PARAMS, STATE_DIR
from .ablate_csd_refined import build_refined_risk, apply_gate
from .backtest_blended import (load_spot_aligned_to_kalshi,
                                  position_series_with_blend)


def make_params() -> dict:
    p = dict(DEFAULT_STRATEGY_PARAMS)
    p.update({"vol_win": 12, "lookback": 2, "beta": 0.25, "k": 1.2,
              "z_cap": 3.5, "deadband_bps": 1.0, "regime_win": 8,
              "er_cap": 1.0, "gain": 1.0, "band": 0.5})
    return p


def _rolling_std(xs, w):
    out = [0.0] * len(xs)
    for i in range(w, len(xs)):
        win = xs[i - w:i]
        m = sum(win) / w
        v = sum((x - m) ** 2 for x in win) / max(1, w - 1)
        out[i] = math.sqrt(v)
    return out


def adaptive_threshold_series(bars, risks, atr_breakpoint=0.20,
                                base_threshold=0.95,
                                hi_vol_threshold=0.80,
                                atr_window=14) -> List[float]:
    """Per-bar effective CSD threshold under the adaptive rule. ATR% is
    derived from the bar's high-low range so the test mirrors what the
    live engine sees as market.features.atr_pct."""
    n = len(bars)
    thrs = [base_threshold] * n
    if n < atr_window + 1:
        return thrs
    # Wilder-style ATR%
    trs = [0.0]
    for i in range(1, n):
        h = bars[i]["high"]; l = bars[i]["low"]; pc = bars[i - 1]["close"]
        trs.append(max(h - l, abs(h - pc), abs(l - pc)))
    atr = sum(trs[1:atr_window + 1]) / atr_window
    for i in range(atr_window + 1, n):
        atr = (atr * (atr_window - 1) + trs[i]) / atr_window
        px = bars[i]["close"]
        atr_pct = (atr / px * 100.0) if px else 0.0
        thrs[i] = hi_vol_threshold if atr_pct >= atr_breakpoint else base_threshold
    return thrs


def apply_adaptive_gate(pos, risks, thresholds):
    """Same as apply_gate but the threshold is per-bar."""
    return [p if risks[i] <= thresholds[i] else 0.0 for i, p in enumerate(pos)]


def run_hybrid(bars, pos, urgency, hs_bps=4.0, chase_n=3, leverage=5.8):
    return signals.simulate_hybrid(
        bars, pos, urgency,
        k=float("inf"), chase_n=chase_n,
        half_spread_bps=hs_bps, fee_bps=0.0, leverage=leverage)


def walk_forward(bars, spot_aligned, params, K=4, hs_bps=4.0, chase_n=3,
                  blend_weight=0.0,
                  csd_threshold_fixed=None,
                  csd_threshold_adaptive=False):
    """One pass per variant. Combines blend + csd gate (fixed or adaptive)."""
    fs = len(bars) // K
    folds = []
    agg = {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0, "trades": 0}
    for fi in range(K):
        lo = fi * fs
        hi = len(bars) if fi == K - 1 else (fi + 1) * fs
        fb = bars[lo:hi]
        fspot = spot_aligned[lo:hi]
        # Position series (with or without blend)
        if blend_weight > 0:
            bp = dict(signals_blended.DEFAULT_BLEND); bp["w_spot_lead"] = blend_weight
            pos, urg, _ = position_series_with_blend(fb, fspot, params, bp)
        else:
            pos, urg = signals.mpc_with_aux(fb, params)
        # CSD gate
        if csd_threshold_fixed is not None or csd_threshold_adaptive:
            risks = build_refined_risk(fb, mode="skew_only")
            if csd_threshold_adaptive:
                thrs = adaptive_threshold_series(fb, risks)
                pos = apply_adaptive_gate(pos, risks, thrs)
            else:
                pos = apply_gate(pos, risks, csd_threshold_fixed)
        res = run_hybrid(fb, pos, urg, hs_bps=hs_bps, chase_n=chase_n)
        folds.append({"fold": fi + 1, **{k: res.get(k) for k in
                       ("return_pct", "max_dd_pct", "sharpe", "trades")}})
        for k in ("return_pct", "trades"):
            agg[k] += res.get(k, 0)
        agg["max_dd_pct"] = max(agg["max_dd_pct"], res.get("max_dd_pct", 0))
        agg["sharpe"] += res.get("sharpe", 0.0)
    agg["sharpe"] /= K
    return {"folds": folds, "agg": agg}


def main():
    print("Loading Kalshi 3m bars + aligning spot 1m...")
    rows = lab.load(use_cache=True)
    bars = lab.aggregate(rows, 3)
    spot_aligned = load_spot_aligned_to_kalshi(bars)
    n_cov = sum(1 for v in spot_aligned if v is not None)
    print(f"  {len(bars):,} bars, {n_cov} spot-aligned\n")

    params = make_params()
    variants = [
        ("A baseline (MPC alone)",
         dict(blend_weight=0.0, csd_threshold_fixed=None,
               csd_threshold_adaptive=False)),
        ("B + CSD gate T=0.95",
         dict(blend_weight=0.0, csd_threshold_fixed=0.95,
               csd_threshold_adaptive=False)),
        ("C + CSD adaptive (0.80@hi-vol)",
         dict(blend_weight=0.0, csd_threshold_fixed=None,
               csd_threshold_adaptive=True)),
        ("D + spot-lead blend w=0.30",
         dict(blend_weight=0.30, csd_threshold_fixed=None,
               csd_threshold_adaptive=False)),
        ("E ALL stacked",
         dict(blend_weight=0.30, csd_threshold_fixed=None,
               csd_threshold_adaptive=True)),
    ]

    results = {}
    for label, kwargs in variants:
        print(f"Running {label} ...")
        results[label] = walk_forward(bars, spot_aligned, params, K=4, **kwargs)

    print("\n" + "=" * 92)
    print(" PER-FOLD RETURNS  (% / max_dd%)")
    print("=" * 92)
    print(f"  {'variant':<32}", end="")
    for fi in range(1, 5):
        print(f"  fold {fi}            ", end="")
    print()
    for label, _ in variants:
        a = results[label]
        print(f"  {label:<32}", end="")
        for f in a["folds"]:
            print(f"  {f['return_pct']:>+7.2f}% dd{f['max_dd_pct']:>4.1f}%", end="")
        print()

    print("\n" + "=" * 92)
    print(" AGGREGATE  (sum return / max DD / mean sharpe)")
    print("=" * 92)
    base_ret = results[variants[0][0]]["agg"]["return_pct"]
    base_dd = results[variants[0][0]]["agg"]["max_dd_pct"]
    base_sh = results[variants[0][0]]["agg"]["sharpe"]
    print(f"  {'variant':<32}{'return':>10}{'max_dd':>10}{'sharpe':>10}"
          f"{'d_ret':>10}{'d_dd':>10}{'d_sh':>10}")
    for label, _ in variants:
        a = results[label]["agg"]
        print(f"  {label:<32}{a['return_pct']:>+9.2f}%{a['max_dd_pct']:>+9.2f}%"
              f"{a['sharpe']:>+10.2f}{a['return_pct']-base_ret:>+9.2f}pp"
              f"{a['max_dd_pct']-base_dd:>+9.2f}pp"
              f"{a['sharpe']-base_sh:>+10.2f}")

    print("\n" + "=" * 92)
    print(" MARGINAL CONTRIBUTION (each layer vs the one above it)")
    print("=" * 92)
    prior_label = variants[0][0]
    for i in range(1, len(variants)):
        cur_label, _ = variants[i]
        prior = results[prior_label]["agg"]
        cur = results[cur_label]["agg"]
        d_ret = cur["return_pct"] - prior["return_pct"]
        d_dd = cur["max_dd_pct"] - prior["max_dd_pct"]
        d_sh = cur["sharpe"] - prior["sharpe"]
        print(f"  {cur_label:<32} d_ret {d_ret:+.2f}pp  "
              f"d_dd {d_dd:+.2f}pp  d_sh {d_sh:+.2f}")
        prior_label = cur_label

    out = {"variants": [{"label": l, "agg": results[l]["agg"],
                          "folds": results[l]["folds"]} for l, _ in variants]}
    p = STATE_DIR / "combined_ablation.json"
    p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nResults written to {p}")


if __name__ == "__main__":
    main()
