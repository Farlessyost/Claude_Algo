"""Sweep the maker/taker threshold k and chase-delay N over the cached 1m
history, using the HONEST hybrid fill simulator (`signals.simulate_hybrid`).

Honest sim recap:
  - Maker order rests at a limit at the bar's close. It only fills if a later
    bar's range trades through the limit (low<=limit for buys, high>=limit for
    sells). Otherwise it expires after chase_n bars and we either pay the
    half-spread to chase (taker) or stop trying (missed) depending on chase_n.
  - Taker orders cross immediately and pay the half-spread.
  - Fees are added on top of the spread cost.

Walk-forward: split the candle series into K sequential folds. For each fold
the in-sample window is everything before the test segment (anchored
expanding). Params (the validated default MPC params) are NOT re-tuned per
fold — we're sweeping (k, N, cost), not strategy params. Results show how the
costed PnL changes purely from the execution policy.

Run:  python -m backend.sweep_maker_taker
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import List

from . import lab, signals
from .config import DEFAULT_STRATEGY_PARAMS, STATE_DIR

# Knobs the live config supports. k = the urgency threshold above which we
# cross; chase_n = how many bars to leave a resting maker before chasing.
K_GRID = [math.inf, 3.0, 2.0, 1.5, 1.0, 0.5, 0.25, 0.0]
N_GRID = [0, 1, 2, 3, 6, math.inf]
HALF_SPREAD_BPS_GRID = [4.0, 7.0, 9.0]   # logged ~4-9 bps
FEE_BPS = 0.0                            # Kalshi promo today
LEVERAGE = 5.8
TIMEFRAME_MIN = 3                        # 3m bars (live default)
FOLDS = 4                                # walk-forward folds


def _fmt_k(k):
    return "inf" if math.isinf(k) else f"{k:g}"


def _fmt_n(n):
    return "inf" if math.isinf(n) else str(int(n))


def walk_forward(candles, pos, urgency, k, chase_n, hs_bps, fee_bps, folds):
    n = len(candles)
    if n < folds * 50:
        return None
    seg = n // folds
    rows = []
    for f in range(folds):
        a = f * seg
        b = (f + 1) * seg if f < folds - 1 else n
        cs = candles[a:b]
        ps = pos[a:b]
        us = urgency[a:b]
        r = signals.simulate_hybrid(cs, ps, us, k=k, chase_n=chase_n,
                                    half_spread_bps=hs_bps, fee_bps=fee_bps,
                                    leverage=LEVERAGE)
        rows.append(r)
    rets = [r["return_pct"] for r in rows]
    tot = 1.0
    for x in rets:
        tot *= (1 + x / 100.0)
    return {
        "per_fold": rets,
        "compounded_pct": (tot - 1) * 100,
        "pos_folds": sum(1 for x in rets if x > 0),
        "total_folds": len(rets),
        "trades_maker": sum(r["trades_maker"] for r in rows),
        "trades_taker": sum(r["trades_taker"] for r in rows),
        "trades_chase": sum(r["trades_chase"] for r in rows),
        "missed": sum(r["missed"] for r in rows),
        "fill_rate_maker": (sum(r["trades_maker"] for r in rows) /
                            max(1, sum(r["trades_maker"] + r["trades_chase"]
                                       + r["missed"] for r in rows))),
        "max_dd_pct": max(r["max_dd_pct"] for r in rows),
        "avg_sharpe": sum(r["sharpe"] for r in rows) / len(rows),
    }


def main():
    rows = lab.load(use_cache=True, do_clean=True)
    candles = lab.aggregate(rows, TIMEFRAME_MIN)
    print(f"Loaded {len(rows)} 1m rows -> {len(candles)} {TIMEFRAME_MIN}m bars")
    spread_bps = [r["spread"] / r["close"] * 1e4
                  for r in rows if r.get("spread") and r["close"]]
    if spread_bps:
        spread_bps.sort()
        med = spread_bps[len(spread_bps) // 2]
        avg = sum(spread_bps) / len(spread_bps)
        print(f"Empirical spread: median {med:.2f} bps, mean {avg:.2f} bps "
              f"(half-spread ~{med/2:.2f} bps)\n")

    params = dict(DEFAULT_STRATEGY_PARAMS)
    pos, urgency = signals.mpc_with_aux(candles, params)
    nz = [u for u in urgency if u > 0]
    if nz:
        nz.sort()
        print(f"Urgency distribution (|alpha|): "
              f"p10 {nz[len(nz)//10]:.3f} | p50 {nz[len(nz)//2]:.3f} | "
              f"p90 {nz[len(nz)*9//10]:.3f} | max {nz[-1]:.3f}\n")

    print(f"Walk-forward: {FOLDS} folds, {TIMEFRAME_MIN}m bars, lev {LEVERAGE}x, "
          f"fee {FEE_BPS} bps. (Maker fills only when bar range trades through limit.)\n")

    best = None
    all_rows = []
    for hs in HALF_SPREAD_BPS_GRID:
        print(f"===== half-spread = {hs:.1f} bps/side =====")
        print(f"  {'k':>5s} {'N':>3s}  {'comp%':>7s} {'pos':>4s} "
              f"{'mk':>4s} {'tk':>4s} {'ch':>4s} {'miss':>4s} {'fillR':>5s} "
              f"{'avgShp':>7s} {'maxDD%':>7s}  per-fold")
        for k in K_GRID:
            for nch in N_GRID:
                # All-taker (k=0) with no chase doesn't really need chase>0.
                # Pure-maker (k=inf) with chase=inf is today's pure-maker.
                rep = walk_forward(candles, pos, urgency,
                                   k=k, chase_n=nch,
                                   hs_bps=hs, fee_bps=FEE_BPS, folds=FOLDS)
                if not rep:
                    continue
                rec = {"hs_bps": hs, "k": ("inf" if math.isinf(k) else k),
                       "chase_n": ("inf" if math.isinf(nch) else int(nch)), **rep}
                all_rows.append(rec)
                print(f"  {_fmt_k(k):>5s} {_fmt_n(nch):>3s}  "
                      f"{rep['compounded_pct']:>+7.2f} "
                      f"{rep['pos_folds']:>1d}/{rep['total_folds']:<2d} "
                      f"{rep['trades_maker']:>4d} {rep['trades_taker']:>4d} "
                      f"{rep['trades_chase']:>4d} {rep['missed']:>4d} "
                      f"{rep['fill_rate_maker']:>5.2f} "
                      f"{rep['avg_sharpe']:>+7.2f} {rep['max_dd_pct']:>7.2f}  "
                      f"{['%+.1f' % x for x in rep['per_fold']]}")
                # Score: compounded return with a drawdown penalty.
                # Prefer combos that are also positive in MORE folds.
                score = (rep["compounded_pct"] - rep["max_dd_pct"] * 0.5
                         + rep["pos_folds"] * 1.0)
                if best is None or score > best["score"]:
                    best = {"score": score, **rec}
        print()

    print("===== BEST (compounded - 0.5*maxDD + pos_folds) =====")
    if best:
        print(f"  k={best['k']}  chase_n={best['chase_n']}  "
              f"half_spread={best['hs_bps']} bps  -> compounded "
              f"{best['compounded_pct']:+.2f}% across "
              f"{best['pos_folds']}/{best['total_folds']} positive folds, "
              f"maxDD {best['max_dd_pct']:.2f}%, avg sharpe "
              f"{best['avg_sharpe']:+.2f}")
        out = STATE_DIR / "sweep_maker_taker.json"
        out.write_text(json.dumps({"all": all_rows, "best": best}, indent=2),
                       encoding="utf-8")
        print(f"  (wrote full grid to {out})")
    else:
        print("  no usable results")
    return best


if __name__ == "__main__":
    main()
