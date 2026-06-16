"""Walk-forward backtest of the Trophic Information Forager / Mycelial
Alpha Network on top of the validated MPC controller.

Same no-look-ahead discipline as the maker/taker sweep: for each bar i the
ecology classifier uses ONLY candle data up to bar i (multi-asset feeds
aren't backfilled — the on-platform-only ecology is what we honestly test
here; the live system gets the full multi-asset network).

For each cost level we report:
  - baseline MPC (pure-maker hybrid sim at half_spread bps)
  - ecology-modulated MPC (size_mult applied per bar from phase classifier)
  - state mix (how often each phase fires)

Run:
    .\\.venv\\Scripts\\python.exe -m backend.backtest_ecology
"""
from __future__ import annotations

import json
from typing import List

from . import lab, signals, ecology
from .config import DEFAULT_STRATEGY_PARAMS, STATE_DIR


def apply_mult(pos: List[float], mult: List[float]) -> List[float]:
    out = [0.0] * len(pos)
    for i in range(len(pos)):
        v = pos[i] * (mult[i] if i < len(mult) else 1.0)
        out[i] = max(-1.0, min(1.0, v))
    return out


def fmt_pct(x: float) -> str:
    return f"{x:+.2f}%"


def fold_stats(candles: List[dict], params: dict, leverage: float,
               half_spread_bps: float) -> dict:
    pos_base, urgency = signals.mpc_with_aux(candles, params)
    mults, states = ecology.phase_series_with_states(candles, params)
    pos_eco = apply_mult(pos_base, mults)

    base_fric = signals.simulate(candles, pos_base, leverage=leverage, fee_pct=0.0)
    eco_fric  = signals.simulate(candles, pos_eco,  leverage=leverage, fee_pct=0.0)
    base_hyb = signals.simulate_hybrid(candles, pos_base, urgency,
                                       k=float("inf"), chase_n=3,
                                       half_spread_bps=half_spread_bps,
                                       fee_bps=0.0, leverage=leverage)
    eco_hyb  = signals.simulate_hybrid(candles, pos_eco, urgency,
                                       k=float("inf"), chase_n=3,
                                       half_spread_bps=half_spread_bps,
                                       fee_bps=0.0, leverage=leverage)
    sc = {p: 0 for p in ecology.PHASES}
    for s in states:
        sc[s] = sc.get(s, 0) + 1
    active = sum(1 for m in mults if abs(m - 1.0) > 1e-6)
    avg_mult = sum(mults) / len(mults) if mults else 0.0
    return {
        "base_fric": base_fric, "eco_fric": eco_fric,
        "base_hyb": base_hyb,   "eco_hyb": eco_hyb,
        "state_counts": sc, "active_bars": active,
        "avg_mult": round(avg_mult, 3), "bars": len(candles),
    }


def main():
    print("Loading cached 1m rich history from", lab.CACHE)
    rows = lab.load(use_cache=True)
    print(f"  cleaned 1m bars: {len(rows):,}")
    bars = lab.aggregate(rows, 3)
    print(f"  aggregated to 3m bars: {len(bars):,}")
    if len(bars) < 600:
        print("  not enough bars to run walk-forward — abort.")
        return

    params = dict(DEFAULT_STRATEGY_PARAMS)
    params.update({
        "vol_win": 12, "lookback": 2, "beta": 0.25, "k": 1.2, "z_cap": 3.5,
        "deadband_bps": 1.0, "regime_win": 8, "er_cap": 1.0,
        "gain": 1.0, "band": 0.5,
    })
    leverage = 5.8
    cost_grid = [0.0, 4.0, 7.0, 9.0]
    K = 4
    n = len(bars)
    fold_size = n // K
    all_results = {"folds": [], "params": params, "leverage": leverage,
                   "bars_total": n, "K": K}

    for hs in cost_grid:
        print(f"\n=== cost level: half_spread={hs:.1f} bps ===")
        block = {"half_spread_bps": hs, "folds": []}
        a_bf = a_ef = a_bh = a_eh = 0.0
        w_fric = w_hyb = 0
        for k in range(K):
            lo = k * fold_size
            hi = n if k == K - 1 else (k + 1) * fold_size
            fold = bars[lo:hi]
            r = fold_stats(fold, params, leverage, hs)
            block["folds"].append(r)
            bf = r["base_fric"]["return_pct"]
            ef = r["eco_fric"]["return_pct"]
            bh = r["base_hyb"]["return_pct"]
            eh = r["eco_hyb"]["return_pct"]
            a_bf += bf; a_ef += ef; a_bh += bh; a_eh += eh
            if ef > bf: w_fric += 1
            if eh > bh: w_hyb += 1
            sc = r["state_counts"]
            print(f"  fold {k+1}/{K}  bars={r['bars']:>4}  avg_mult={r['avg_mult']:.2f}"
                  f"  pred={sc['predator']:>3}  exh={sc['exhaustion']:>3}"
                  f"  scav={sc['scavenger']:>3}  deco={sc['decomposer']:>3}"
                  f"  churn={sc['churn']:>3}  prod={sc['producer']:>4}")
            print(f"          fric  base {fmt_pct(bf)}  eco {fmt_pct(ef)}"
                  f"   |   hyb  base {fmt_pct(bh)}  eco {fmt_pct(eh)}")
            print(f"          trades_base hyb={r['base_hyb']['trades']:>3}"
                  f"  trades_eco hyb={r['eco_hyb']['trades']:>3}")
        block["agg"] = {
            "fric_base_total_pct": round(a_bf, 3),
            "fric_eco_total_pct":  round(a_ef, 3),
            "fric_delta_pct":      round(a_ef - a_bf, 3),
            "fric_eco_wins_folds": w_fric,
            "hyb_base_total_pct":  round(a_bh, 3),
            "hyb_eco_total_pct":   round(a_eh, 3),
            "hyb_delta_pct":       round(a_eh - a_bh, 3),
            "hyb_eco_wins_folds":  w_hyb,
        }
        print(f"  >>> AGG  fric  base {fmt_pct(a_bf)}  eco {fmt_pct(a_ef)}"
              f"   d {fmt_pct(a_ef - a_bf)}  wins {w_fric}/{K}")
        print(f"           hyb   base {fmt_pct(a_bh)}  eco {fmt_pct(a_eh)}"
              f"   d {fmt_pct(a_eh - a_bh)}  wins {w_hyb}/{K}")
        all_results["folds"].append(block)

    out_path = STATE_DIR / "ecology_backtest.json"
    out_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
