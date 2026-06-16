"""Walk-forward validation for the promising variants. Anchored (expanding)
train window, sequential out-of-sample test folds, parameters re-tuned each fold
(mirrors the live auto-tune loop). Also reports fee sensitivity.

A variant earns trust only if it is positive out-of-sample across MOST folds and
both timeframes, and survives higher fees.

Run:  python -m backend.walkforward [iterations]
"""
from __future__ import annotations

import sys

from . import history, research
from .config import load_credentials
from .kalshi_client import KalshiClient
from .store import STORE


def walk(candles, variant_fn, folds, iterations, leverage, fee_pct):
    n = len(candles)
    start = int(n * 0.4)            # first 40% is the initial training base
    step = (n - start) // folds
    results = []
    for k in range(folds):
        tr_end = start + k * step
        te_end = start + (k + 1) * step if k < folds - 1 else n
        train = candles[:tr_end]
        test = candles[tr_end:te_end]
        if len(train) < 150 or len(test) < 30:
            continue
        best = research.search(train, variant_fn, iterations, leverage, fee_pct)
        if not best:
            continue
        _, p = best
        te = research.simulate(test, variant_fn(test, p),
                               leverage=leverage, fee_pct=fee_pct)
        results.append(te)
    return results


def run(iterations=40, leverage=5.8, folds=5):
    client = KalshiClient(load_credentials(), STORE.settings.environment)
    one = history.fetch_full_1m(client, STORE.settings.ticker, max_days=14)
    print(f"1m history: {len(one)} bars (~{len(one)/1440:.1f} days)")
    print(f"Walk-forward: {folds} expanding folds, re-tuned each fold "
          f"({iterations} candidates), leverage {leverage}x\n")

    candidates = ["regime", "meanrev"]
    for fee_bps in (2.0, 5.0, 10.0):
        fee = fee_bps / 1e4
        print(f"===== Taker fee {fee_bps:.0f} bps/side =====")
        for tf in ["5m", "15m"]:
            candles = history.to_timeframe(one, tf)
            for name in candidates:
                fn = research.VARIANTS[name]
                res = walk(candles, fn, folds, iterations, leverage, fee)
                if not res:
                    print(f"  {tf:>3} {name:8s}: (insufficient data)"); continue
                rets = [r["return_pct"] for r in res]
                pos = sum(1 for r in rets if r > 0)
                total = 1.0
                for r in rets:
                    total *= (1 + r / 100)
                compounded = (total - 1) * 100
                avg = sum(rets) / len(rets)
                trades = sum(r["trades"] for r in res)
                print(f"  {tf:>3} {name:8s}: folds+ {pos}/{len(rets)}  "
                      f"per-fold {['%+.1f' % x for x in rets]}  "
                      f"compounded {compounded:+.1f}%  avg {avg:+.2f}%  trades {trades}")
        print()


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    run(iterations=iters)
