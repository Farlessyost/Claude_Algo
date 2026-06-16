"""Robustness tests for the Trophic Information Forager controller.

If the +2.3% improvement from `backtest_ecology.py` is real, it should survive:
  1. NULL-SHUFFLE: randomly permute the per-bar phase multiplier (same
     marginal distribution, no temporal alignment to the actual ecological
     state). If the alpha comes from temporal alignment, the real series
     should beat the shuffle reliably. If shuffles win too, the "alpha" was
     just spurious turnover.
  2. PER-PHASE ATTRIBUTION: 5 sub-tests, each isolating ONE phase's
     multiplier and setting the rest to 1.0. Tells us which phases
     actually generate the alpha and which are cargo-cult.
  3. THRESHOLD SENSITIVITY: perturb the phase multipliers by ±30%. If the
     result evaporates under small changes, it's overfit.
  4. 8-FOLD WALK-FORWARD: tighter aggregate over more folds.
  5. PER-BAR T-STAT: compute bar-by-bar PnL differences (eco - baseline)
     and report mean / (std / sqrt(N)) so we can see if the improvement is
     statistically significant or fits in noise.

Run:
    .\\.venv\\Scripts\\python.exe -m backend.backtest_ecology_robust
"""
from __future__ import annotations

import json
import random
import statistics
from typing import List, Tuple

from . import lab, signals, ecology
from .config import DEFAULT_STRATEGY_PARAMS, STATE_DIR


# ------------------------------------------------- shared helpers
def make_params() -> dict:
    p = dict(DEFAULT_STRATEGY_PARAMS)
    p.update({"vol_win": 12, "lookback": 2, "beta": 0.25, "k": 1.2,
              "z_cap": 3.5, "deadband_bps": 1.0, "regime_win": 8,
              "er_cap": 1.0, "gain": 1.0, "band": 0.5})
    return p


def apply_mult(pos: List[float], mult: List[float]) -> List[float]:
    out = [0.0] * len(pos)
    for i in range(len(pos)):
        v = pos[i] * (mult[i] if i < len(mult) else 1.0)
        out[i] = max(-1.0, min(1.0, v))
    return out


def per_bar_pnl(candles: List[dict], pos: List[float],
                leverage: float = 5.8) -> List[float]:
    """Per-bar P&L (held position over i -> i+1). Same convention as
    signals.simulate but returns the per-step return series so we can
    bootstrap / t-stat the difference between strategies."""
    out: List[float] = []
    for i in range(len(candles) - 1):
        px = candles[i]["close"]; nxt = candles[i + 1]["close"]
        target = pos[i]
        if px > 0 and target != 0:
            out.append((nxt - px) / px * target * leverage)
        else:
            out.append(0.0)
    return out


def t_stat(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = statistics.mean(xs)
    s = statistics.stdev(xs)
    return (m / s) * (len(xs) ** 0.5) if s else 0.0


def cumret(xs: List[float]) -> float:
    eq = 1.0
    for r in xs:
        eq *= (1.0 + r)
    return (eq - 1.0) * 100.0


# ------------------------------------------------- test 1: NULL shuffle
def test_null_shuffle(bars: List[dict], params: dict, n_shuffles: int = 50,
                      leverage: float = 5.8) -> dict:
    """Run K shuffles where the per-bar multiplier is randomly permuted.
    Compare the true vs shuffled return distribution.

    If the alpha is real, the true return should sit well into the right
    tail of the shuffle distribution (p < 0.05 = true beats >=95% of
    shuffles)."""
    pos_base, _ = signals.mpc_with_aux(bars, params)
    mults_true, _ = ecology.phase_series_with_states(bars, params)
    true_ret = cumret(per_bar_pnl(bars, apply_mult(pos_base, mults_true), leverage))
    base_ret = cumret(per_bar_pnl(bars, pos_base, leverage))

    shuffled_rets: List[float] = []
    mults_copy = list(mults_true)
    rng = random.Random(42)
    for _ in range(n_shuffles):
        rng.shuffle(mults_copy)
        shuffled_rets.append(cumret(per_bar_pnl(
            bars, apply_mult(pos_base, mults_copy), leverage)))
    shuffled_rets.sort()
    # how many shuffles did the true series beat?
    beats = sum(1 for s in shuffled_rets if true_ret > s)
    p_value = 1.0 - beats / max(1, n_shuffles)
    return {
        "base_pct": round(base_ret, 3),
        "true_pct": round(true_ret, 3),
        "shuffle_mean_pct": round(statistics.mean(shuffled_rets), 3),
        "shuffle_std_pct":  round(statistics.stdev(shuffled_rets) if len(shuffled_rets) > 1 else 0, 3),
        "shuffle_min_pct":  round(min(shuffled_rets), 3),
        "shuffle_max_pct":  round(max(shuffled_rets), 3),
        "shuffles": n_shuffles,
        "true_beats_n_shuffles": beats,
        "p_value": round(p_value, 4),
    }


# ------------------------------------------------- test 2: per-phase attribution
def test_per_phase(bars: List[dict], params: dict, leverage: float = 5.8) -> dict:
    """For each phase, run a configuration where ONLY that phase's mult is
    its real value and ALL other phases are set to 1.0. The per-phase
    contribution = (this-phase-only return) - (baseline return)."""
    pos_base, _ = signals.mpc_with_aux(bars, params)
    base_ret = cumret(per_bar_pnl(bars, pos_base, leverage))

    # Get the per-bar phase string array once
    _, states = ecology.phase_series_with_states(bars, params)
    full_mults = {
        "producer":   params.get("ecology_producer_mult", 1.0),
        "predator":   params.get("ecology_predator_mult", 0.2),
        "exhaustion": params.get("ecology_exhaustion_mult", 0.6),
        "scavenger":  params.get("ecology_scavenger_mult", 1.5),
        "decomposer": params.get("ecology_decomposer_mult", 0.5),
        "churn":      params.get("ecology_churn_mult", 1.0),
    }
    results = {}
    for phase, mult in full_mults.items():
        if abs(mult - 1.0) < 1e-9:
            results[phase] = {"contribution_pct": 0.0, "bars_active": 0,
                              "note": "mult == 1.0; no contribution"}
            continue
        per_bar = []
        active = 0
        for i, st in enumerate(states):
            if st == phase:
                per_bar.append(mult)
                active += 1
            else:
                per_bar.append(1.0)
        ret = cumret(per_bar_pnl(bars, apply_mult(pos_base, per_bar), leverage))
        results[phase] = {
            "mult": mult,
            "bars_active": active,
            "isolated_pct": round(ret, 3),
            "contribution_pct": round(ret - base_ret, 3),
        }
    return {"base_pct": round(base_ret, 3), "phases": results}


# ------------------------------- test 3: threshold sensitivity
def test_threshold_sensitivity(bars: List[dict], leverage: float = 5.8) -> dict:
    """Perturb each phase multiplier by ±30% and check the result holds."""
    base_params = make_params()
    pos_base, _ = signals.mpc_with_aux(bars, base_params)
    base_ret = cumret(per_bar_pnl(bars, pos_base, leverage))

    perturbations = [
        ("scavenger_low",   {"ecology_scavenger_mult": 1.0}),
        ("scavenger_high",  {"ecology_scavenger_mult": 2.0}),
        ("predator_low",    {"ecology_predator_mult": 0.1}),
        ("predator_high",   {"ecology_predator_mult": 0.4}),
        ("decomposer_low",  {"ecology_decomposer_mult": 0.3}),
        ("decomposer_high", {"ecology_decomposer_mult": 0.8}),
        ("exhaustion_low",  {"ecology_exhaustion_mult": 0.4}),
        ("exhaustion_high", {"ecology_exhaustion_mult": 0.9}),
        ("all_off",         {f"ecology_{p}_mult": 1.0 for p in ecology.PHASES}),
    ]
    out = {"baseline_pct": round(base_ret, 3), "scenarios": {}}
    for label, patch in perturbations:
        p = dict(base_params); p.update(patch)
        mults, _ = ecology.phase_series_with_states(bars, p)
        ret = cumret(per_bar_pnl(bars, apply_mult(pos_base, mults), leverage))
        out["scenarios"][label] = {
            "patch": patch,
            "eco_pct": round(ret, 3),
            "delta_pct": round(ret - base_ret, 3),
        }
    return out


# ------------------------------------------------- test 4: K-fold walk-forward
def k_fold(bars: List[dict], params: dict, K: int = 8,
            leverage: float = 5.8) -> dict:
    """Run a K-fold walk-forward, ecology vs baseline. Use frictionless +
    cost-aware hybrid (default 4 bps half-spread)."""
    n = len(bars)
    fs = n // K
    folds = []
    a_base = a_eco = 0.0
    wins = 0
    for k in range(K):
        lo = k * fs
        hi = n if k == K - 1 else (k + 1) * fs
        fold = bars[lo:hi]
        pos_base, _ = signals.mpc_with_aux(fold, params)
        mults, _ = ecology.phase_series_with_states(fold, params)
        pos_eco = apply_mult(pos_base, mults)
        b = cumret(per_bar_pnl(fold, pos_base, leverage))
        e = cumret(per_bar_pnl(fold, pos_eco, leverage))
        a_base += b; a_eco += e
        if e > b: wins += 1
        folds.append({"fold": k + 1, "bars": len(fold),
                      "base_pct": round(b, 3), "eco_pct": round(e, 3),
                      "delta_pct": round(e - b, 3)})
    return {"K": K, "agg_base_pct": round(a_base, 3),
            "agg_eco_pct":  round(a_eco, 3),
            "agg_delta_pct":round(a_eco - a_base, 3),
            "eco_wins": wins, "folds": folds}


# ------------------------------------------------- test 5: per-bar t-stat
def per_bar_significance(bars: List[dict], params: dict,
                          leverage: float = 5.8) -> dict:
    """Compute bar-by-bar PnL differences and report t-stat + bootstrap."""
    pos_base, _ = signals.mpc_with_aux(bars, params)
    mults, _ = ecology.phase_series_with_states(bars, params)
    pos_eco = apply_mult(pos_base, mults)
    pnl_base = per_bar_pnl(bars, pos_base, leverage)
    pnl_eco  = per_bar_pnl(bars, pos_eco,  leverage)
    n = min(len(pnl_base), len(pnl_eco))
    diff = [pnl_eco[i] - pnl_base[i] for i in range(n)]
    nz_diff = [d for d in diff if abs(d) > 1e-12]
    # block bootstrap (block=20 bars, 1000 resamples)
    rng = random.Random(1337)
    boot_means = []
    if len(diff) >= 60:
        block = 20
        blocks = [diff[i:i + block] for i in range(0, len(diff), block) if len(diff[i:i + block]) == block]
        for _ in range(1000):
            sample = []
            for _ in range(len(blocks)):
                sample.extend(blocks[rng.randrange(len(blocks))])
            boot_means.append(statistics.mean(sample) if sample else 0)
        boot_means.sort()
        lo95 = boot_means[int(0.025 * len(boot_means))]
        hi95 = boot_means[int(0.975 * len(boot_means))]
    else:
        lo95 = hi95 = 0.0
    return {
        "bars": n,
        "nonzero_diff_bars": len(nz_diff),
        "mean_diff_per_bar_bps": round(statistics.mean(diff) * 1e4, 4) if diff else 0,
        "std_diff_per_bar_bps":  round((statistics.stdev(diff) * 1e4) if len(diff) > 1 else 0, 4),
        "t_stat": round(t_stat(diff), 3),
        "boot_2.5%_pct":  round(lo95 * len(diff) * 100, 3),
        "boot_97.5%_pct": round(hi95 * len(diff) * 100, 3),
    }


# ------------------------------------------------- main
def main():
    print("Loading cached 1m rich history…")
    rows = lab.load(use_cache=True)
    bars = lab.aggregate(rows, 3)
    print(f"  {len(bars):,} 3m bars\n")
    params = make_params()

    print("=" * 72)
    print("TEST 1 — NULL SHUFFLE  (true vs random-permuted multipliers)")
    print("=" * 72)
    r1 = test_null_shuffle(bars, params, n_shuffles=80)
    print(f"  baseline MPC return    : {r1['base_pct']:+.2f}%")
    print(f"  true ecology return    : {r1['true_pct']:+.2f}%")
    print(f"  shuffle mean ± std     : {r1['shuffle_mean_pct']:+.2f}% ± {r1['shuffle_std_pct']:.2f}%")
    print(f"  shuffle range          : [{r1['shuffle_min_pct']:+.2f}%, {r1['shuffle_max_pct']:+.2f}%]")
    print(f"  true beats             : {r1['true_beats_n_shuffles']}/{r1['shuffles']} shuffles")
    print(f"  p-value (1-tail)       : {r1['p_value']:.4f}   "
          f"({'SIGNIFICANT' if r1['p_value'] < 0.05 else 'not significant'})")

    print("\n" + "=" * 72)
    print("TEST 2 — PER-PHASE ATTRIBUTION  (which phases generate the alpha?)")
    print("=" * 72)
    r2 = test_per_phase(bars, params)
    print(f"  baseline MPC: {r2['base_pct']:+.2f}%")
    print(f"  {'phase':<11}{'mult':>6}{'bars':>8}{'isolated_pct':>15}{'contribution':>15}")
    for ph in ("producer","predator","exhaustion","scavenger","decomposer","churn"):
        d = r2["phases"][ph]
        if "note" in d:
            print(f"  {ph:<11}{'':>6}{'':>8}{'':>15}{'(no mult)':>15}")
        else:
            print(f"  {ph:<11}{d['mult']:>6.2f}{d['bars_active']:>8}"
                  f"{d['isolated_pct']:>+14.2f}%{d['contribution_pct']:>+14.2f}%")

    print("\n" + "=" * 72)
    print("TEST 3 — THRESHOLD SENSITIVITY  (does it survive ±30% perturbations?)")
    print("=" * 72)
    r3 = test_threshold_sensitivity(bars)
    print(f"  baseline MPC: {r3['baseline_pct']:+.2f}%")
    print(f"  {'scenario':<22}{'eco_pct':>12}{'delta':>12}")
    for label, d in r3["scenarios"].items():
        print(f"  {label:<22}{d['eco_pct']:>+11.2f}%{d['delta_pct']:>+11.2f}%")

    print("\n" + "=" * 72)
    print("TEST 4 — 8-FOLD WALK-FORWARD")
    print("=" * 72)
    r4 = k_fold(bars, params, K=8)
    print(f"  {'fold':<6}{'bars':>6}{'base_pct':>12}{'eco_pct':>12}{'delta':>12}")
    for f in r4["folds"]:
        print(f"  {f['fold']:<6}{f['bars']:>6}{f['base_pct']:>+11.2f}%"
              f"{f['eco_pct']:>+11.2f}%{f['delta_pct']:>+11.2f}%")
    print(f"  AGG          {r4['agg_base_pct']:>+11.2f}%{r4['agg_eco_pct']:>+11.2f}%"
          f"{r4['agg_delta_pct']:>+11.2f}%  eco_wins {r4['eco_wins']}/{r4['K']}")

    print("\n" + "=" * 72)
    print("TEST 5 — PER-BAR T-STAT  (statistical significance of eco - base)")
    print("=" * 72)
    r5 = per_bar_significance(bars, params)
    print(f"  bars                          : {r5['bars']:,}")
    print(f"  bars where ecology disagreed  : {r5['nonzero_diff_bars']:,}")
    print(f"  mean per-bar diff             : {r5['mean_diff_per_bar_bps']:+.4f} bps")
    print(f"  std per-bar diff              : {r5['std_diff_per_bar_bps']:.4f} bps")
    print(f"  t-stat                        : {r5['t_stat']:+.3f}   "
          f"({'SIGNIFICANT' if abs(r5['t_stat']) > 2.0 else 'noisy'})")
    print(f"  block-bootstrap 95% CI on total return improvement:")
    print(f"     [{r5['boot_2.5%_pct']:+.2f}%, {r5['boot_97.5%_pct']:+.2f}%]")

    out = {"test1_null_shuffle": r1, "test2_per_phase": r2,
           "test3_thresholds": r3, "test4_8fold": r4,
           "test5_per_bar_significance": r5}
    out_path = STATE_DIR / "ecology_robust.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
