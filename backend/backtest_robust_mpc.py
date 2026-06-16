"""Backtest ecology-conditioned robust MPC against current MPC.

The robust controller keeps the MPC's low-turnover no-trade-band structure but
conditions gain, band width, and max size on the candle-only ecology phase.
Live mode additionally shrinks the latest aim using the full trophic network's
disturbance/reserve/immune scores.

Run:
    .\\.venv\\Scripts\\python.exe -m backend.backtest_robust_mpc
"""
from __future__ import annotations

import json
from typing import List

from . import ecology, lab, signals
from .config import DEFAULT_STRATEGY_PARAMS, STATE_DIR


def _apply_mult(pos: List[float], mult: List[float]) -> List[float]:
    out = []
    for i, p in enumerate(pos):
        m = mult[i] if i < len(mult) else 1.0
        out.append(max(-1.0, min(1.0, p * m)))
    return out


def _sim(candles, pos, urg, half_spread_bps: float, leverage: float) -> dict:
    return signals.simulate_hybrid(
        candles, pos, urg, k=float("inf"), chase_n=3,
        half_spread_bps=half_spread_bps, fee_bps=0.0, leverage=leverage)


def fold_stats(candles: List[dict], params: dict, half_spread_bps: float,
               leverage: float) -> dict:
    pos_mpc, urg_mpc = signals.mpc_with_aux(candles, params)
    mults, states = ecology.phase_series_with_states(candles, params)
    pos_eco = _apply_mult(pos_mpc, mults)
    pos_rob, urg_rob = signals.robust_mpc_with_aux(candles, params)

    phase_counts = {p: 0 for p in ecology.PHASES}
    for st in states:
        phase_counts[st] = phase_counts.get(st, 0) + 1

    return {
        "mpc": _sim(candles, pos_mpc, urg_mpc, half_spread_bps, leverage),
        "mpc_ecology_mult": _sim(candles, pos_eco, urg_mpc, half_spread_bps, leverage),
        "robust_mpc": _sim(candles, pos_rob, urg_rob, half_spread_bps, leverage),
        "phase_counts": phase_counts,
        "avg_abs_pos": {
            "mpc": round(sum(abs(x) for x in pos_mpc) / len(pos_mpc), 4),
            "mpc_ecology_mult": round(sum(abs(x) for x in pos_eco) / len(pos_eco), 4),
            "robust_mpc": round(sum(abs(x) for x in pos_rob) / len(pos_rob), 4),
        },
    }


def main() -> None:
    print("Loading cached 1m rich history from", lab.CACHE)
    rows = lab.load(use_cache=True)
    bars = lab.aggregate(rows, 3)
    print(f"  3m bars: {len(bars):,}")
    if len(bars) < 800:
        print("not enough bars")
        return

    params = dict(DEFAULT_STRATEGY_PARAMS)
    params.update({
        "vol_win": 24, "lookback": 1, "beta": 0.35, "k": 2.4,
        "z_cap": 2.0, "deadband_bps": 0.5, "regime_win": 20,
        "er_cap": 0.6, "gain": 1.0, "band": 0.3,
        "robust_lambda": 0.35, "robust_disturbance_lambda": 0.45,
    })
    K = 8
    leverage = 5.8
    costs = [0.0, 4.0, 7.0]
    n = len(bars)
    fold_size = n // K
    out = {"K": K, "bars": n, "params": params, "costs": []}

    for hs in costs:
        print(f"\n=== half_spread={hs:.1f} bps ===")
        block = {"half_spread_bps": hs, "folds": []}
        agg = {k: {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0, "wins": 0}
               for k in ("mpc", "mpc_ecology_mult", "robust_mpc")}
        for k in range(K):
            lo = k * fold_size
            hi = n if k == K - 1 else (k + 1) * fold_size
            fold = bars[lo:hi]
            r = fold_stats(fold, params, hs, leverage)
            block["folds"].append(r)
            base_ret = r["mpc"]["return_pct"]
            print(f"  fold {k+1}: "
                  f"mpc {r['mpc']['return_pct']:+6.2f}% dd {r['mpc']['max_dd_pct']:5.2f}% | "
                  f"eco {r['mpc_ecology_mult']['return_pct']:+6.2f}% dd {r['mpc_ecology_mult']['max_dd_pct']:5.2f}% | "
                  f"rob {r['robust_mpc']['return_pct']:+6.2f}% dd {r['robust_mpc']['max_dd_pct']:5.2f}%")
            for name in agg:
                x = r[name]
                agg[name]["return_pct"] += x["return_pct"]
                agg[name]["max_dd_pct"] = max(agg[name]["max_dd_pct"], x["max_dd_pct"])
                agg[name]["sharpe"] += x["sharpe"]
                if x["return_pct"] > base_ret:
                    agg[name]["wins"] += 1
        for name in agg:
            agg[name]["return_pct"] = round(agg[name]["return_pct"], 3)
            agg[name]["max_dd_pct"] = round(agg[name]["max_dd_pct"], 3)
            agg[name]["sharpe"] = round(agg[name]["sharpe"] / K, 3)
        block["agg"] = agg
        out["costs"].append(block)
        print("  AGG:")
        for name, x in agg.items():
            print(f"    {name:<16} ret {x['return_pct']:+7.2f}% "
                  f"maxDD {x['max_dd_pct']:5.2f}% sharpe {x['sharpe']:+.2f} "
                  f"wins_vs_mpc {x['wins']}/{K}")

    path = STATE_DIR / "robust_mpc_backtest.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults written to {path}")


if __name__ == "__main__":
    main()
