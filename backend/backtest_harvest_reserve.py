"""Backtest protected harvest reserve vs immediate compounding.

The live harvest reserve subtracts captured forager profit from deployable
equity, so the bot cannot immediately recycle harvested gains into the same
session's next/add size. Existing backtests compound all gains into the next
bar's notional. This script compares those capital policies on the same
validated robust-MPC target series.

Approximation discipline:
  - signal generation is unchanged;
  - target exposure = target_fraction * deployable_equity * leverage;
  - "captured" profit is estimated when exposure is reduced/closed/reversed
    while the current average-entry PnL is positive;
  - reserved profit remains in total equity but is excluded from deployable
    equity for future sizing.

Run:
    .\\.venv\\Scripts\\python.exe -m backend.backtest_harvest_reserve
"""
from __future__ import annotations

import json
import math
from typing import List

from . import lab, signals
from .config import DEFAULT_STRATEGY_PARAMS, STATE_DIR


def make_params() -> dict:
    p = dict(DEFAULT_STRATEGY_PARAMS)
    p.update({
        "vol_win": 12, "lookback": 2, "beta": 0.25, "k": 1.2,
        "z_cap": 3.5, "deadband_bps": 1.0, "regime_win": 8,
        "er_cap": 1.0, "gain": 1.0, "band": 0.5,
    })
    return p


def _ret_stats(equity_curve: List[float]) -> dict:
    if not equity_curve:
        return {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0}
    eq0 = equity_curve[0]
    ret = (equity_curve[-1] / eq0 - 1.0) * 100.0 if eq0 else 0.0
    peak = eq0
    dd = 0.0
    rets = []
    prev = eq0
    for x in equity_curve:
        peak = max(peak, x)
        dd = max(dd, (peak - x) / peak if peak else 0.0)
        if prev:
            rets.append((x - prev) / prev)
        prev = x
    if len(rets) > 2:
        m = sum(rets) / len(rets)
        s = math.sqrt(sum((r - m) ** 2 for r in rets) / (len(rets) - 1))
        sharpe = (m / s) * math.sqrt(365 * 24 * 20) if s else 0.0
    else:
        sharpe = 0.0
    return {
        "return_pct": round(ret, 3),
        "max_dd_pct": round(dd * 100.0, 3),
        "sharpe": round(sharpe, 3),
    }


def simulate_capital_policy(candles: List[dict], target_frac: List[float], *,
                            leverage: float = 5.8,
                            half_spread_bps: float = 4.0,
                            reserve_fraction: float = 0.0) -> dict:
    """Dollar-style simulator with optional protected harvest reserve."""
    n = min(len(candles), len(target_frac))
    if n < 3:
        return {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0}
    equity = 1.0
    reserve = 0.0
    exposure = 0.0              # signed dollars of risk exposure
    entry_price = None
    entry_sign = 0
    equity_curve = [equity]
    reserve_curve = [reserve]
    captured_total = 0.0
    harvest_events = 0
    cost = float(half_spread_bps) / 10_000.0

    for i in range(n - 1):
        price = float(candles[i]["close"])
        nxt = float(candles[i + 1]["close"])
        if price <= 0 or nxt <= 0:
            equity_curve.append(equity)
            reserve_curve.append(reserve)
            continue

        deployable = max(0.0, equity - reserve)
        desired = max(-1.0, min(1.0, float(target_frac[i] or 0.0)))
        desired_exposure = desired * deployable * leverage

        # Reserve captured profit on profitable reductions/closes/reversals.
        old_abs = abs(exposure)
        new_abs_same_side = (
            abs(desired_exposure)
            if exposure == 0.0 or desired_exposure == 0.0
            or (exposure > 0) == (desired_exposure > 0)
            else 0.0
        )
        reducing = old_abs > 1e-12 and new_abs_same_side < old_abs - 1e-12
        if reducing and entry_price:
            sign = 1.0 if exposure > 0 else -1.0
            open_pnl = old_abs * ((price - entry_price) / entry_price) * sign
            if open_pnl > 0:
                reduced_frac = (old_abs - new_abs_same_side) / old_abs
                captured = open_pnl * reduced_frac * max(0.0, min(1.0, reserve_fraction))
                reserve += max(0.0, min(captured, equity - reserve))
                captured_total += captured
                harvest_events += 1

        # Rebalance cost.
        delta = desired_exposure - exposure
        equity -= abs(delta) * cost
        equity = max(1e-9, equity)

        # Reset/roll average entry when direction changes or position grows.
        if abs(desired_exposure) < 1e-12:
            entry_price = None
            entry_sign = 0
        elif exposure == 0.0 or (exposure > 0) != (desired_exposure > 0):
            entry_price = price
            entry_sign = 1 if desired_exposure > 0 else -1
        elif abs(desired_exposure) > abs(exposure) and entry_price:
            # Weighted average entry on add.
            add_abs = abs(desired_exposure) - abs(exposure)
            entry_price = ((abs(exposure) * entry_price) + (add_abs * price)) / abs(desired_exposure)
            entry_sign = 1 if desired_exposure > 0 else -1
        exposure = desired_exposure

        # Mark to market over i -> i+1.
        equity += exposure * ((nxt - price) / price)
        reserve = min(max(0.0, reserve), max(0.0, equity))
        equity_curve.append(equity)
        reserve_curve.append(reserve)

    out = _ret_stats(equity_curve)
    out.update({
        "final_equity": round(equity, 6),
        "final_reserve": round(reserve, 6),
        "captured_total": round(captured_total, 6),
        "harvest_events": harvest_events,
        "avg_reserved_pct_equity": round(
            (sum((r / e) if e else 0.0 for r, e in zip(reserve_curve, equity_curve))
             / len(reserve_curve)) * 100.0, 3),
    })
    return out


def main():
    rows = lab.load(use_cache=True)
    bars = lab.aggregate(rows, 3)
    params = make_params()
    leverage = 5.8
    cost_grid = [0.0, 4.0, 7.0]
    K = 8
    n = len(bars)
    fold_size = n // K
    results = {"bars": n, "K": K, "costs": []}
    print(f"loaded {n:,} 3m bars")

    for hs in cost_grid:
        agg = {"reinvest": {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0},
               "reserve_all": {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0},
               "reserve_half": {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0}}
        folds = []
        print(f"\n=== half_spread={hs:.1f} bps ===")
        for k in range(K):
            lo = k * fold_size
            hi = n if k == K - 1 else (k + 1) * fold_size
            fold = bars[lo:hi]
            pos, _urg = signals.robust_mpc_with_aux(fold, params)
            reinvest = simulate_capital_policy(fold, pos, leverage=leverage,
                                                half_spread_bps=hs, reserve_fraction=0.0)
            reserve_all = simulate_capital_policy(fold, pos, leverage=leverage,
                                                  half_spread_bps=hs, reserve_fraction=1.0)
            reserve_half = simulate_capital_policy(fold, pos, leverage=leverage,
                                                   half_spread_bps=hs, reserve_fraction=0.5)
            row = {"fold": k + 1, "reinvest": reinvest,
                   "reserve_all": reserve_all, "reserve_half": reserve_half}
            folds.append(row)
            for label, stat in (("reinvest", reinvest), ("reserve_all", reserve_all),
                                ("reserve_half", reserve_half)):
                agg[label]["return_pct"] += stat["return_pct"]
                agg[label]["max_dd_pct"] = max(agg[label]["max_dd_pct"], stat["max_dd_pct"])
                agg[label]["sharpe"] += stat["sharpe"] / K
            print(f"  fold {k+1}: reinvest {reinvest['return_pct']:+6.2f}% dd {reinvest['max_dd_pct']:5.2f}% | "
                  f"reserve_all {reserve_all['return_pct']:+6.2f}% dd {reserve_all['max_dd_pct']:5.2f}% "
                  f"res {reserve_all['final_reserve']:.3f}")
        for label in agg:
            agg[label]["return_pct"] = round(agg[label]["return_pct"], 3)
            agg[label]["max_dd_pct"] = round(agg[label]["max_dd_pct"], 3)
            agg[label]["sharpe"] = round(agg[label]["sharpe"], 3)
        print("  AGG:")
        for label, stat in agg.items():
            print(f"    {label:<12} ret {stat['return_pct']:+7.2f}% "
                  f"dd {stat['max_dd_pct']:6.2f}% sh {stat['sharpe']:+.2f}")
        results["costs"].append({"half_spread_bps": hs, "folds": folds, "agg": agg})

    out = STATE_DIR / "harvest_reserve_backtest.json"
    out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nResults written to {out}")


if __name__ == "__main__":
    main()
