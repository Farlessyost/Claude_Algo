"""Direct ablation: does a CSD risk governor improve the validated maker/taker
strategy on Kalshi 3m bars? Or is it dead weight?

The signal is weak on its own (IC ~0.13 vs trail vol 0.55; AUC 0.55 vs 0.70),
but the drawdown-quintile test on spot data showed monotone Q5/Q1 = 1.27 —
the top CSD quintile has 27% larger forward max-drawdown than the bottom.
That's tail-event-shaped, not vol-shaped. So this script tests whether
*shutting off the strategy during top-quintile CSD risk* tightens the equity
curve enough to justify the wire-up.

Two governor variants compared against the unchanged baseline:
  - threshold_gate:  position = 0 whenever csd_risk > T
  - linear_scale:    position *= (1 - csd_risk * alpha) bounded to [0, 1]

Walk-forward over 4 folds, matching the validated maker/taker config sweep.

GO criterion: at least one governor variant
  - improves Sharpe by >= 5% AND does not reduce return_pct by more than 10%
  - OR reduces max_dd_pct by >= 20% with return_pct loss <= 15%

Otherwise the CSD signal is too weak to justify the wiring.

Run:
    .\.venv\Scripts\python.exe -m backend.ablate_csd_governor
"""
from __future__ import annotations

import json
import statistics
from typing import List, Tuple

from . import csd, lab, signals
from .config import DEFAULT_STRATEGY_PARAMS, STATE_DIR


def make_params() -> dict:
    p = dict(DEFAULT_STRATEGY_PARAMS)
    p.update({"vol_win": 12, "lookback": 2, "beta": 0.25, "k": 1.2,
              "z_cap": 3.5, "deadband_bps": 1.0, "regime_win": 8,
              "er_cap": 1.0, "gain": 1.0, "band": 0.5})
    return p


def compute_risks_on_bars(bars: List[dict],
                           fv_period: int = 32,
                           window: int = 96) -> List[float]:
    """Walk-forward CSD risk for these bars. Multivariate extras when
    available (spread / oi). Pure no-look-ahead."""
    closes = [b["close"] for b in bars]
    spreads = [float(b.get("spread", 0.0)) for b in bars]
    ois = [float(b.get("oi", 0.0)) for b in bars]
    extras = {}
    if any(spreads):
        d = [0.0] + [spreads[i] - spreads[i - 1] for i in range(1, len(spreads))]
        extras["spread"] = d
    if any(ois):
        d = [0.0] + [ois[i] - ois[i - 1] for i in range(1, len(ois))]
        extras["oi"] = d
    bundles = csd.rolling_metrics(closes, fv_period=fv_period,
                                    window=window, step=1,
                                    extras_series=extras or None)
    risk_hist_w = max(window, 200)
    risks = [0.0] * len(bars)
    history = []
    for i, b in enumerate(bundles):
        if not b:
            continue
        ref = [h for h in history[-risk_hist_w:] if h]
        score = csd.csd_score(b, ref) if len(ref) >= 12 else 0.0
        risks[i] = csd.csd_risk(score)
        history.append(b)
    return risks


def apply_threshold_gate(pos: List[float], risks: List[float],
                          threshold: float) -> List[float]:
    return [p if risks[i] <= threshold else 0.0 for i, p in enumerate(pos)]


def apply_linear_scale(pos: List[float], risks: List[float],
                        alpha: float) -> List[float]:
    out = []
    for i, p in enumerate(pos):
        scale = max(0.0, 1.0 - risks[i] * alpha)
        out.append(p * scale)
    return out


def run_hybrid(bars, pos, urgency, k, chase_n, hs_bps, fee_bps=0.0):
    res = signals.simulate_hybrid(bars, pos, urgency,
                                    k=(float("inf") if k is None else k),
                                    chase_n=chase_n,
                                    half_spread_bps=hs_bps,
                                    fee_bps=fee_bps)
    return res


def fold(bars, params, risks, k, chase_n, hs_bps):
    pos, urgency = signals.mpc_with_aux(bars, params)
    baseline = run_hybrid(bars, pos, urgency, k, chase_n, hs_bps)

    # threshold gates
    gate_results = {}
    for T in (0.80, 0.85, 0.90, 0.95):
        gpos = apply_threshold_gate(pos, risks, T)
        gate_results[T] = run_hybrid(bars, gpos, urgency, k, chase_n, hs_bps)

    # linear scales
    lin_results = {}
    for A in (0.25, 0.5, 0.75, 1.0):
        lpos = apply_linear_scale(pos, risks, A)
        lin_results[A] = run_hybrid(bars, lpos, urgency, k, chase_n, hs_bps)

    return baseline, gate_results, lin_results


def main():
    print("Loading Kalshi 3m bars…")
    rows = lab.load(use_cache=True)
    bars = lab.aggregate(rows, 3)
    print(f"  {len(bars):,} 3m bars\n")

    print("Computing rolling CSD risks (walk-forward)…")
    risks = compute_risks_on_bars(bars, fv_period=32, window=96)
    nz = sum(1 for r in risks if r > 0)
    print(f"  defined risks: {nz:,}/{len(risks):,}  "
          f"mean: {sum(r for r in risks if r > 0) / max(1, nz):.3f}\n")

    # validated config
    k = None; chase_n = 3; hs_bps = 4.0

    print(f"Validated maker/taker config: k={k}, chase_n={chase_n}, hs={hs_bps}bps\n")

    K = 4
    fs = len(bars) // K
    folds = []
    agg_base = {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0}
    agg_gate = {T: {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0}
                  for T in (0.80, 0.85, 0.90, 0.95)}
    agg_lin = {A: {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0}
                  for A in (0.25, 0.5, 0.75, 1.0)}

    for fi in range(K):
        lo = fi * fs
        hi = len(bars) if fi == K - 1 else (fi + 1) * fs
        fold_bars = bars[lo:hi]
        fold_risks = risks[lo:hi]
        b, gates, lins = fold(fold_bars, make_params(), fold_risks, k, chase_n, hs_bps)
        rec = {"fold": fi + 1, "bars": len(fold_bars),
                "base": {k_: b.get(k_) for k_ in ("return_pct", "max_dd_pct", "sharpe", "trades")},
                "gates": {T: {k_: gates[T].get(k_) for k_ in ("return_pct", "max_dd_pct", "sharpe", "trades")}
                            for T in gates},
                "lins": {A: {k_: lins[A].get(k_) for k_ in ("return_pct", "max_dd_pct", "sharpe", "trades")}
                            for A in lins}}
        folds.append(rec)
        agg_base["return_pct"] += b["return_pct"]
        agg_base["max_dd_pct"] = max(agg_base["max_dd_pct"], b["max_dd_pct"])
        agg_base["sharpe"] += b["sharpe"]
        for T, r in gates.items():
            agg_gate[T]["return_pct"] += r["return_pct"]
            agg_gate[T]["max_dd_pct"] = max(agg_gate[T]["max_dd_pct"], r["max_dd_pct"])
            agg_gate[T]["sharpe"] += r["sharpe"]
        for A, r in lins.items():
            agg_lin[A]["return_pct"] += r["return_pct"]
            agg_lin[A]["max_dd_pct"] = max(agg_lin[A]["max_dd_pct"], r["max_dd_pct"])
            agg_lin[A]["sharpe"] += r["sharpe"]

    # average sharpe across folds
    agg_base["sharpe"] /= K
    for T in agg_gate:
        agg_gate[T]["sharpe"] /= K
    for A in agg_lin:
        agg_lin[A]["sharpe"] /= K

    print("=" * 80)
    print("PER-FOLD RESULTS (return_pct / max_dd_pct / sharpe / trades)")
    print("=" * 80)
    for r in folds:
        print(f"\nFold {r['fold']} ({r['bars']} bars):")
        print(f"  baseline       : {r['base']['return_pct']:+.2f}%  dd {r['base']['max_dd_pct']:.2f}%  "
              f"sh {r['base']['sharpe']:+.2f}  trades {r['base']['trades']}")
        for T, x in r["gates"].items():
            print(f"  gate T={T:.2f}    : {x['return_pct']:+.2f}%  dd {x['max_dd_pct']:.2f}%  "
                  f"sh {x['sharpe']:+.2f}  trades {x['trades']}")
        for A, x in r["lins"].items():
            print(f"  scale alpha={A:.2f}: {x['return_pct']:+.2f}%  dd {x['max_dd_pct']:.2f}%  "
                  f"sh {x['sharpe']:+.2f}  trades {x['trades']}")

    print("\n" + "=" * 80)
    print("AGGREGATE (sum returns, max DD, mean sharpe across folds)")
    print("=" * 80)
    print(f"  {'variant':<22}{'ret%':>10}{'max_dd%':>12}{'sharpe':>10}{'d_ret':>10}{'d_dd':>10}{'d_sh':>10}")
    br, bd, bs = agg_base["return_pct"], agg_base["max_dd_pct"], agg_base["sharpe"]
    print(f"  {'baseline':<22}{br:>+10.2f}{bd:>+12.2f}{bs:>+10.2f}{0:>+10.2f}{0:>+10.2f}{0:>+10.2f}")
    for T, x in agg_gate.items():
        r_, d_, s_ = x["return_pct"], x["max_dd_pct"], x["sharpe"]
        print(f"  {f'gate T={T:.2f}':<22}{r_:>+10.2f}{d_:>+12.2f}{s_:>+10.2f}"
              f"{r_ - br:>+10.2f}{d_ - bd:>+10.2f}{s_ - bs:>+10.2f}")
    for A, x in agg_lin.items():
        r_, d_, s_ = x["return_pct"], x["max_dd_pct"], x["sharpe"]
        print(f"  {f'scale alpha={A:.2f}':<22}{r_:>+10.2f}{d_:>+12.2f}{s_:>+10.2f}"
              f"{r_ - br:>+10.2f}{d_ - bd:>+10.2f}{s_ - bs:>+10.2f}")

    # decision criterion
    print("\n" + "=" * 80)
    print("GO/NO-GO  (looking for any variant with delta_sharpe >= +0.05 vs baseline,")
    print("           OR delta_dd <= -0.20 * baseline_dd with ret loss <= 15%)")
    print("=" * 80)
    winners = []
    sharpe_thresh = 0.05
    dd_thresh = -0.20 * bd if bd > 0 else -1.0
    for label, x in [(f"gate T={T:.2f}", v) for T, v in agg_gate.items()] + \
                    [(f"scale alpha={A:.2f}", v) for A, v in agg_lin.items()]:
        d_sh = x["sharpe"] - bs
        d_dd = x["max_dd_pct"] - bd
        d_ret = x["return_pct"] - br
        ret_ok_for_dd = (d_ret >= -0.15 * abs(br)) if br else True
        ret_ok_for_sh = (d_ret >= -0.10 * abs(br)) if br else True
        if (d_sh >= sharpe_thresh and ret_ok_for_sh) or \
            (bd > 0 and d_dd <= dd_thresh and ret_ok_for_dd):
            winners.append({"variant": label, "d_sharpe": round(d_sh, 3),
                              "d_dd": round(d_dd, 3), "d_ret": round(d_ret, 3)})
    if winners:
        print("  GO: at least one variant cleared the bar:")
        for w in winners:
            print(f"    {w['variant']:<22}  d_sharpe {w['d_sharpe']:+.3f}  "
                  f"d_dd {w['d_dd']:+.3f}  d_ret {w['d_ret']:+.3f}")
    else:
        print("  NO-GO: no variant cleared either bar.")
        print("  Recommendation: do NOT wire CSD into the maker/taker switch.")

    out = {"k": k, "chase_n": chase_n, "hs_bps": hs_bps,
            "K_folds": K, "folds": folds,
            "agg_baseline": agg_base, "agg_gates": agg_gate, "agg_lin_scales": agg_lin,
            "winners": winners}
    p = STATE_DIR / "csd_ablation.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults written to {p}")


if __name__ == "__main__":
    main()
