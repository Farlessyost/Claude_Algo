"""Compare timeframes for KXBTCPERP: pull max history, tune each, and pick the
most profitable risk-adjusted timeframe. Favors a higher trade rate when the
risk-adjusted edge is comparable (per the operator's goal of aggressive growth
with controlled risk). Writes the winning timeframe + params into live settings.

Run:  python -m backend.optimize [iterations]
"""
from __future__ import annotations

import random
import sys

from . import backtest, market_data
from .config import DEFAULT_STRATEGY_PARAMS, load_credentials
from .kalshi_client import KalshiClient
from .store import STORE
from .tuning import SEARCH_SPACE

TIMEFRAMES = ["1m", "5m", "15m", "1h"]
MIN_BARS = 120  # need at least this many bars for a meaningful backtest


def tune_timeframe(candles, iterations, leverage):
    best = None
    candidates = [dict(DEFAULT_STRATEGY_PARAMS)]
    for _ in range(iterations - 1):
        cand = dict(DEFAULT_STRATEGY_PARAMS)
        cand.update({k: random.choice(v) for k, v in SEARCH_SPACE.items()})
        candidates.append(cand)
    for params in candidates:
        rep = backtest.run_backtest(candles, params, leverage=leverage)
        if "score" not in rep:
            continue
        if best is None or rep["score"] > best[0]["score"]:
            best = (rep, params)
    return best


def run(iterations=40, leverage=5.8):
    creds = load_credentials()
    client = KalshiClient(creds, STORE.settings.environment)
    ticker = STORE.settings.ticker
    print(f"Optimizing {ticker} ({STORE.settings.environment}), "
          f"{iterations} candidates/timeframe, leverage {leverage}x\n")

    results = {}
    for tf in TIMEFRAMES:
        candles = market_data.fetch_candles(client, ticker, tf, lookback_bars=600)
        n = len(candles)
        enough = n >= MIN_BARS
        status = "OK" if enough else f"LOW ({n}<{MIN_BARS})"
        print(f"[{tf:>3}] bars={n:<5} data={status}")
        if not enough:
            results[tf] = {"bars": n, "enough": False}
            continue
        rep, params = tune_timeframe(candles, iterations, leverage)
        results[tf] = {"bars": n, "enough": True, "report": rep, "params": params}
        print(f"       best score={rep['score']:+.4f} return={rep['total_return_pct']:+.2f}% "
              f"maxDD={rep['max_drawdown_pct']:.2f}% trades={rep['trades']} "
              f"win={rep['win_rate']:.2f}")

    # Choose: among timeframes with positive score, pick best score; if a faster
    # timeframe is within 15% of the top score but trades more, prefer it.
    viable = {tf: r for tf, r in results.items()
              if r.get("enough") and r.get("report")}
    if not viable:
        print("\nNo viable timeframe with enough data.")
        return results

    ranked = sorted(viable.items(), key=lambda kv: kv[1]["report"]["score"], reverse=True)
    top_tf, top = ranked[0]
    best_score = top["report"]["score"]
    chosen_tf, chosen = top_tf, top
    order = {"1m": 0, "5m": 1, "15m": 2, "1h": 3}
    for tf, r in viable.items():
        s = r["report"]["score"]
        faster = order[tf] < order[chosen_tf]
        more_trades = r["report"]["trades"] > chosen["report"]["trades"]
        close = s >= best_score - abs(best_score) * 0.15 - 0.05
        if faster and more_trades and close:
            chosen_tf, chosen = tf, r

    print(f"\nTop risk-adjusted: {top_tf} (score {best_score:+.4f})")
    print(f"Chosen for live:   {chosen_tf} "
          f"(score {chosen['report']['score']:+.4f}, "
          f"return {chosen['report']['total_return_pct']:+.2f}%, "
          f"trades {chosen['report']['trades']})")

    STORE.update_settings({"timeframe": chosen_tf,
                           "strategy_params": chosen["params"]})
    print(f"Applied timeframe={chosen_tf} + tuned params to live settings.")
    return results


if __name__ == "__main__":
    iters = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    run(iterations=iters)
