"""Refined CSD governor ablation: use ONLY the components that carry signal
beyond trail vol.

The component diagnostic (csd_component_diag) found:
  - abs_skew has IC_dd|trail = +0.055, AUC=0.608, Q5/Q1=1.455 (genuinely
    orthogonal to trail vol, the real CSD signal we wanted).
  - variance of log-deviation has IC_dd|trail = +0.120 — strong, but partly
    overlaps with trail vol mechanically.
  - All other components (phi, recovery_rate, lf_power, max_eigenvalue,
    well_depth) had null or negative partial IC.

So the refined governor scores are:
  - refined_skew_only: sigmoid(z(abs_skew))
  - refined_combined:  sigmoid(0.5 * z(variance) + 1.0 * z(abs_skew))

Both compared against the validated maker/taker config baseline on Kalshi 3m
walk-forward.

Run:
    .\.venv\Scripts\python.exe -m backend.ablate_csd_refined
"""
from __future__ import annotations

import json
import math
import statistics
from typing import List

from . import csd, lab, signals
from .config import DEFAULT_STRATEGY_PARAMS, STATE_DIR


def make_params() -> dict:
    p = dict(DEFAULT_STRATEGY_PARAMS)
    p.update({"vol_win": 12, "lookback": 2, "beta": 0.25, "k": 1.2,
              "z_cap": 3.5, "deadband_bps": 1.0, "regime_win": 8,
              "er_cap": 1.0, "gain": 1.0, "band": 0.5})
    return p


def _zscore(x, hist):
    if len(hist) < 6:
        return 0.0
    m = sum(hist) / len(hist)
    var = sum((v - m) ** 2 for v in hist) / (len(hist) - 1) if len(hist) > 1 else 0
    s = math.sqrt(var)
    return (x - m) / s if s > 0 else 0.0


def _sigmoid(x):
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def build_refined_risk(bars, fv_period=32, window=96, hist_w=200,
                        mode="skew_only") -> List[float]:
    """Walk-forward refined CSD risk.

    mode = "skew_only" : sigmoid(z(abs_skew))
    mode = "combined"  : sigmoid(0.5*z(variance) + 1.0*z(abs_skew))
    """
    closes = [b["close"] for b in bars]
    bundles = csd.rolling_metrics(closes, fv_period=fv_period, window=window,
                                    step=1, extras_series=None)
    risks = [0.0] * len(bars)
    skew_hist = []
    var_hist = []
    for i, b in enumerate(bundles):
        if not b:
            continue
        sk = b.get("abs_skew", 0.0)
        var = b.get("variance", 0.0)
        if len(skew_hist) >= 12:
            z_sk = _zscore(sk, skew_hist[-hist_w:])
            z_var = _zscore(var, var_hist[-hist_w:])
            if mode == "skew_only":
                score = z_sk
            elif mode == "combined":
                score = 0.5 * z_var + 1.0 * z_sk
            else:
                raise ValueError(mode)
            risks[i] = _sigmoid(score)
        skew_hist.append(sk)
        var_hist.append(var)
    return risks


def apply_gate(pos, risks, T):
    return [p if risks[i] <= T else 0.0 for i, p in enumerate(pos)]


def apply_scale(pos, risks, alpha):
    return [p * max(0.0, 1.0 - risks[i] * alpha) for i, p in enumerate(pos)]


def run_hybrid(bars, pos, urgency, k, chase_n, hs_bps):
    return signals.simulate_hybrid(
        bars, pos, urgency,
        k=(float("inf") if k is None else k),
        chase_n=chase_n, half_spread_bps=hs_bps, fee_bps=0.0)


def walk_forward_one_signal(bars, risks, params, k, chase_n, hs_bps, K=4):
    fs = len(bars) // K
    agg_base = {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0}
    agg_gate = {T: {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0}
                  for T in (0.80, 0.85, 0.90, 0.95)}
    agg_scale = {A: {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0}
                  for A in (0.25, 0.5, 0.75, 1.0)}
    folds = []
    for fi in range(K):
        lo = fi * fs
        hi = len(bars) if fi == K - 1 else (fi + 1) * fs
        fb = bars[lo:hi]; fr = risks[lo:hi]
        pos, urg = signals.mpc_with_aux(fb, params)
        b = run_hybrid(fb, pos, urg, k, chase_n, hs_bps)
        gates = {T: run_hybrid(fb, apply_gate(pos, fr, T), urg, k, chase_n, hs_bps)
                  for T in (0.80, 0.85, 0.90, 0.95)}
        scales = {A: run_hybrid(fb, apply_scale(pos, fr, A), urg, k, chase_n, hs_bps)
                    for A in (0.25, 0.5, 0.75, 1.0)}
        folds.append({"fold": fi + 1, "bars": len(fb),
                       "base": b, "gates": gates, "scales": scales})
        agg_base["return_pct"] += b["return_pct"]
        agg_base["max_dd_pct"] = max(agg_base["max_dd_pct"], b["max_dd_pct"])
        agg_base["sharpe"] += b["sharpe"]
        for T in gates:
            x = gates[T]
            agg_gate[T]["return_pct"] += x["return_pct"]
            agg_gate[T]["max_dd_pct"] = max(agg_gate[T]["max_dd_pct"], x["max_dd_pct"])
            agg_gate[T]["sharpe"] += x["sharpe"]
        for A in scales:
            x = scales[A]
            agg_scale[A]["return_pct"] += x["return_pct"]
            agg_scale[A]["max_dd_pct"] = max(agg_scale[A]["max_dd_pct"], x["max_dd_pct"])
            agg_scale[A]["sharpe"] += x["sharpe"]
    agg_base["sharpe"] /= K
    for T in agg_gate:
        agg_gate[T]["sharpe"] /= K
    for A in agg_scale:
        agg_scale[A]["sharpe"] /= K
    return {"baseline": agg_base, "gates": agg_gate, "scales": agg_scale, "folds": folds}


def print_table(label, agg_base, variants):
    br, bd, bs = agg_base["return_pct"], agg_base["max_dd_pct"], agg_base["sharpe"]
    print(f"\n--- {label} ---")
    print(f"  {'variant':<22}{'ret%':>10}{'max_dd%':>12}{'sharpe':>10}"
          f"{'d_ret':>10}{'d_dd':>10}{'d_sh':>10}{'wins':>6}")
    print(f"  {'baseline':<22}{br:>+10.2f}{bd:>+12.2f}{bs:>+10.2f}"
          f"{0:>+10.2f}{0:>+10.2f}{0:>+10.2f}{'-':>6}")
    for name, x in variants:
        r_, d_, s_ = x["return_pct"], x["max_dd_pct"], x["sharpe"]
        # win = simultaneous d_ret >= -10% baseline AND d_dd < 0
        ret_ok = (r_ - br) >= -0.10 * abs(br) if br else True
        dd_ok = (d_ - bd) < 0
        wins = "Y" if (ret_ok and dd_ok) else " "
        print(f"  {name:<22}{r_:>+10.2f}{d_:>+12.2f}{s_:>+10.2f}"
              f"{r_ - br:>+10.2f}{d_ - bd:>+10.2f}{s_ - bs:>+10.2f}{wins:>6}")


def main():
    print("Loading Kalshi 3m bars…")
    rows = lab.load(use_cache=True)
    bars = lab.aggregate(rows, 3)
    print(f"  {len(bars):,} 3m bars\n")

    print("Computing refined CSD risks (skew_only and combined)…")
    risks_sk = build_refined_risk(bars, mode="skew_only")
    risks_cm = build_refined_risk(bars, mode="combined")

    nz_sk = sum(1 for r in risks_sk if r > 0)
    nz_cm = sum(1 for r in risks_cm if r > 0)
    print(f"  skew_only  : {nz_sk:,} defined, mean {sum(risks_sk)/max(1,nz_sk):.3f}")
    print(f"  combined   : {nz_cm:,} defined, mean {sum(risks_cm)/max(1,nz_cm):.3f}")

    k = None; chase_n = 3; hs_bps = 4.0
    print(f"\nValidated config: k={k}, chase_n={chase_n}, hs={hs_bps}bps\n")

    print("=" * 88)
    print("ABLATION (4-fold walk-forward)")
    print("=" * 88)
    res_sk = walk_forward_one_signal(bars, risks_sk, make_params(), k, chase_n, hs_bps)
    res_cm = walk_forward_one_signal(bars, risks_cm, make_params(), k, chase_n, hs_bps)

    base = res_sk["baseline"]

    variants_sk = [(f"gate T={T:.2f}", res_sk["gates"][T]) for T in res_sk["gates"]]
    variants_sk += [(f"scale a={A:.2f}", res_sk["scales"][A]) for A in res_sk["scales"]]
    print_table("REFINED-SKEW-ONLY", base, variants_sk)

    variants_cm = [(f"gate T={T:.2f}", res_cm["gates"][T]) for T in res_cm["gates"]]
    variants_cm += [(f"scale a={A:.2f}", res_cm["scales"][A]) for A in res_cm["scales"]]
    print_table("REFINED-COMBINED (skew + 0.5*var)", base, variants_cm)

    # Find the best single recommendation across both modes
    best_name = None; best_delta_dd = 0.0; best_x = None; best_mode = None
    for mode, variants in (("skew_only", variants_sk), ("combined", variants_cm)):
        for name, x in variants:
            ret_ok = (x["return_pct"] - base["return_pct"]) >= -0.10 * abs(base["return_pct"]) if base["return_pct"] else True
            if not ret_ok:
                continue
            d_dd = x["max_dd_pct"] - base["max_dd_pct"]
            if d_dd < best_delta_dd:
                best_delta_dd = d_dd
                best_name = name
                best_x = x
                best_mode = mode

    print("\n" + "=" * 88)
    print("RECOMMENDATION")
    print("=" * 88)
    if best_name:
        print(f"  Best variant: {best_mode}/{best_name}")
        print(f"    return_pct  {best_x['return_pct']:+.2f}  (baseline {base['return_pct']:+.2f}, delta {best_x['return_pct'] - base['return_pct']:+.2f})")
        print(f"    max_dd_pct  {best_x['max_dd_pct']:+.2f}  (baseline {base['max_dd_pct']:+.2f}, delta {best_x['max_dd_pct'] - base['max_dd_pct']:+.2f})")
        print(f"    sharpe      {best_x['sharpe']:+.2f}  (baseline {base['sharpe']:+.2f}, delta {best_x['sharpe'] - base['sharpe']:+.2f})")
    else:
        print("  No variant cleared the return-loss bar; refined signal does not help.")

    out = {"k": k, "chase_n": chase_n, "hs_bps": hs_bps,
            "skew_only": res_sk, "combined": res_cm,
            "recommended": {"mode": best_mode, "variant": best_name,
                              "stats": best_x} if best_name else None}
    p = STATE_DIR / "csd_refined_ablation.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults written to {p}")


if __name__ == "__main__":
    main()
