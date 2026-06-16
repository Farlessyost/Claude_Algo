"""Backtest predictive CSD gating against the current refined CSD governor.

This is intentionally offline-only: it reads cached Kalshi rich history,
aggregates to 3m candles, then compares:

    baseline                  no CSD gate
    raw_adaptive              current refined skew-only CSD risk
    predictive_*_adaptive     projected CSD risk using risk velocity,
                              acceleration, and fast/slow impulse

The live predictive horizon defaults to 12 seconds because the engine can now
update faster than candle close. Historical candles are 3 minutes, so this test
also includes 60s and 180s projections. Treat the 180s result as the fairer
bar-level proxy for "warn me before the next historical bar".

Run:
    .\\.venv\\Scripts\\python.exe -m backend.backtest_predictive_csd
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple

from . import csd, lab, reflex, signals
from .config import DEFAULT_STRATEGY_PARAMS, STATE_DIR


BAR_SECONDS = 180.0
HORIZONS_S = (12.0, 60.0, 180.0)


def make_params() -> dict:
    """Use the robust-MPC validation params as the comparison surface."""
    p = dict(DEFAULT_STRATEGY_PARAMS)
    p.update({
        "vol_win": 24,
        "lookback": 1,
        "beta": 0.35,
        "k": 2.4,
        "z_cap": 2.0,
        "deadband_bps": 0.5,
        "regime_win": 20,
        "er_cap": 0.6,
        "gain": 1.0,
        "band": 0.3,
        "robust_lambda": 0.35,
        "robust_disturbance_lambda": 0.45,
    })
    return p


def _zscore(x: float, hist: List[float]) -> float:
    if len(hist) < 6:
        return 0.0
    m = sum(hist) / len(hist)
    var = sum((v - m) ** 2 for v in hist) / (len(hist) - 1)
    s = math.sqrt(var)
    return (x - m) / s if s > 0 else 0.0


def _sigmoid(x: float) -> float:
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


def adaptive_thresholds(
    bars: List[dict],
    *,
    base_thr: float = 0.95,
    high_vol_thr: float = 0.80,
    atr_break_pct: float = 0.20,
    atr_period: int = 14,
) -> Tuple[List[float], List[float]]:
    atr = signals.atr_series(bars, atr_period)
    thresholds: List[float] = []
    atr_pct: List[float] = []
    for i, b in enumerate(bars):
        close = float(b.get("close") or 0.0)
        pct = (float(atr[i]) / close * 100.0) if atr[i] and close else 0.0
        atr_pct.append(pct)
        thresholds.append(high_vol_thr if pct >= atr_break_pct else base_thr)
    return thresholds, atr_pct


def build_predictive_risks(
    bars: List[dict],
    thresholds: List[float],
    *,
    fv_period: int = 32,
    window: int = 96,
    hist_w: int = 200,
    impulse_gain: float = 0.25,
) -> dict:
    """Walk-forward raw and projected refined CSD risk, no future bars used."""
    closes = [float(b["close"]) for b in bars]
    bundles = csd.rolling_metrics(closes, fv_period=fv_period, window=window)
    raw = [0.0] * len(bars)
    projected: Dict[str, List[float]] = {
        f"predictive_{int(h)}s_adaptive": [0.0] * len(bars)
        for h in HORIZONS_S
    }
    skew_hist: List[float] = []
    state: dict = {}
    dynamics: List[dict] = [{} for _ in bars]

    for i, bundle in enumerate(bundles):
        if not bundle:
            continue
        sk_now = float(bundle.get("abs_skew") or 0.0)
        risk_now = 0.0
        if len(skew_hist) >= 12:
            risk_now = _sigmoid(_zscore(sk_now, skew_hist[-hist_w:]))
        raw[i] = risk_now

        dyn = reflex.dynamics(
            state,
            "risk",
            risk_now,
            threshold=thresholds[i] if i < len(thresholds) else 0.95,
            now=i * BAR_SECONDS,
            fast_tau_s=6.0,
            slow_tau_s=30.0,
        )
        state.update(dyn)
        dynamics[i] = dict(dyn)

        velocity = max(0.0, float(dyn.get("risk_velocity") or 0.0))
        accel = max(0.0, float(dyn.get("risk_acceleration") or 0.0))
        impulse = max(0.0, float(dyn.get("risk_impulse") or 0.0))
        for h in HORIZONS_S:
            key = f"predictive_{int(h)}s_adaptive"
            projected[key][i] = _clamp01(
                max(
                    risk_now,
                    risk_now + velocity * h
                    + 0.5 * accel * h * h
                    + impulse_gain * impulse,
                )
            )

        skew_hist.append(sk_now)
        if len(skew_hist) > hist_w:
            skew_hist = skew_hist[-hist_w:]

    return {
        "raw_adaptive": raw,
        **projected,
        "dynamics": dynamics,
    }


def apply_gate(pos: List[float], risks: List[float],
               thresholds: List[float]) -> List[float]:
    return [
        p if i >= len(risks) or i >= len(thresholds) or risks[i] <= thresholds[i] else 0.0
        for i, p in enumerate(pos)
    ]


def run_signal(
    name: str,
    bars: List[dict],
    params: dict,
) -> Tuple[List[float], List[float]]:
    if name == "mpc":
        return signals.mpc_with_aux(bars, params)
    if name == "robust_mpc":
        return signals.robust_mpc_with_aux(bars, params)
    raise ValueError(name)


def sim(bars: List[dict], pos: List[float], urgency: List[float],
        half_spread_bps: float, leverage: float) -> dict:
    return signals.simulate_hybrid(
        bars,
        pos,
        urgency,
        k=float("inf"),
        chase_n=3,
        half_spread_bps=half_spread_bps,
        fee_bps=0.0,
        leverage=leverage,
    )


def empty_agg() -> dict:
    return {
        "return_pct": 0.0,
        "max_dd_pct": 0.0,
        "sharpe": 0.0,
        "trades": 0,
        "ret_wins": 0,
        "dd_wins": 0,
        "dual_wins": 0,
    }


def add_fold(agg: dict, x: dict, base: dict | None = None) -> None:
    agg["return_pct"] += float(x.get("return_pct") or 0.0)
    agg["max_dd_pct"] = max(agg["max_dd_pct"], float(x.get("max_dd_pct") or 0.0))
    agg["sharpe"] += float(x.get("sharpe") or 0.0)
    agg["trades"] += int(x.get("trades") or 0)
    if base:
        ret_win = float(x.get("return_pct") or 0.0) > float(base.get("return_pct") or 0.0)
        dd_win = float(x.get("max_dd_pct") or 0.0) < float(base.get("max_dd_pct") or 0.0)
        agg["ret_wins"] += int(ret_win)
        agg["dd_wins"] += int(dd_win)
        agg["dual_wins"] += int(ret_win and dd_win)


def finish_agg(agg: dict, k: int) -> dict:
    out = dict(agg)
    out["return_pct"] = round(out["return_pct"], 3)
    out["max_dd_pct"] = round(out["max_dd_pct"], 3)
    out["sharpe"] = round(out["sharpe"] / max(1, k), 3)
    return out


def gate_diagnostics(risk: List[float], raw: List[float],
                     thresholds: List[float], lookahead: int = 5) -> dict:
    gate_count = 0
    early_count = 0
    confirmed = 0
    leads: List[int] = []
    for i, r in enumerate(risk):
        thr = thresholds[i] if i < len(thresholds) else 0.95
        if r <= thr:
            continue
        gate_count += 1
        if i < len(raw) and raw[i] <= thr:
            early_count += 1
            for j in range(i + 1, min(len(raw), i + lookahead + 1)):
                thr_j = thresholds[j] if j < len(thresholds) else thr
                if raw[j] > thr_j:
                    confirmed += 1
                    leads.append(j - i)
                    break
    return {
        "gate_count": gate_count,
        "early_vs_raw_count": early_count,
        "early_confirmed_within_5_bars": confirmed,
        "avg_confirmed_lead_bars": round(sum(leads) / len(leads), 3) if leads else 0.0,
    }


def main() -> None:
    print("Loading cached rich history...")
    rows = lab.load(use_cache=True)
    bars = lab.aggregate(rows, 3)
    print(f"  3m bars: {len(bars):,}")
    if len(bars) < 800:
        print("not enough bars")
        return

    params = make_params()
    thresholds, atr_pct = adaptive_thresholds(
        bars,
        atr_period=int(params.get("atr_period", 14)),
    )
    risks = build_predictive_risks(bars, thresholds)
    risk_names = [
        "raw_adaptive",
        "predictive_12s_adaptive",
        "predictive_60s_adaptive",
        "predictive_180s_adaptive",
    ]
    threshold_sets = {
        "adaptive": thresholds,
        "fixed095": [0.95] * len(bars),
    }
    variant_specs = [
        ("raw_adaptive", "raw_adaptive", "adaptive"),
        ("predictive_12s_adaptive", "predictive_12s_adaptive", "adaptive"),
        ("predictive_60s_adaptive", "predictive_60s_adaptive", "adaptive"),
        ("predictive_180s_adaptive", "predictive_180s_adaptive", "adaptive"),
        ("raw_fixed095", "raw_adaptive", "fixed095"),
        ("predictive_180s_fixed095", "predictive_180s_adaptive", "fixed095"),
    ]
    diagnostics = {
        name: gate_diagnostics(risks[name], risks["raw_adaptive"], thresholds)
        for name in risk_names
    }

    print("Risk gate diagnostics:")
    for name in risk_names:
        d = diagnostics[name]
        print(
            f"  {name:<25} gates {d['gate_count']:>4} | "
            f"early {d['early_vs_raw_count']:>4} | "
            f"confirmed<=5 bars {d['early_confirmed_within_5_bars']:>4}"
        )

    k_folds = 8
    leverage = 5.8
    costs = [0.0, 4.0, 7.0]
    signals_to_test = ["mpc", "robust_mpc"]
    n = len(bars)
    fold_size = n // k_folds
    out = {
        "bars": n,
        "bar_seconds": BAR_SECONDS,
        "folds": k_folds,
        "params": params,
        "thresholds": {
            "base": 0.95,
            "high_vol": 0.80,
            "atr_break_pct": 0.20,
            "high_vol_bars": sum(1 for t in thresholds if t < 0.95),
            "avg_atr_pct": round(sum(atr_pct) / max(1, len(atr_pct)), 4),
        },
        "diagnostics": diagnostics,
        "costs": [],
    }

    for hs in costs:
        print(f"\n=== half_spread={hs:.1f} bps ===")
        cost_block = {"half_spread_bps": hs, "signals": {}}
        for sig_name in signals_to_test:
            aggs = {"baseline": empty_agg()}
            aggs.update({name: empty_agg() for name, _, _ in variant_specs})
            fold_rows = []
            print(f"  signal={sig_name}")
            for fold_idx in range(k_folds):
                lo = fold_idx * fold_size
                hi = n if fold_idx == k_folds - 1 else (fold_idx + 1) * fold_size
                fb = bars[lo:hi]
                pos, urg = run_signal(sig_name, fb, params)
                base = sim(fb, pos, urg, hs, leverage)
                add_fold(aggs["baseline"], base)

                fold_result = {
                    "fold": fold_idx + 1,
                    "bars": len(fb),
                    "baseline": base,
                    "variants": {},
                }
                line = (
                    f"    fold {fold_idx + 1}: "
                    f"base {base['return_pct']:+6.2f}% dd {base['max_dd_pct']:5.2f}%"
                )
                for name, risk_key, threshold_key in variant_specs:
                    fr = risks[risk_key][lo:hi]
                    ft = threshold_sets[threshold_key][lo:hi]
                    gated = apply_gate(pos, fr, ft)
                    x = sim(fb, gated, urg, hs, leverage)
                    add_fold(aggs[name], x, base)
                    fold_result["variants"][name] = x
                    if name in ("raw_adaptive", "predictive_180s_adaptive",
                                "raw_fixed095", "predictive_180s_fixed095"):
                        line += f" | {name.replace('_adaptive','')[:11]} {x['return_pct']:+6.2f}%"
                print(line)
                fold_rows.append(fold_result)

            final = {name: finish_agg(agg, k_folds) for name, agg in aggs.items()}
            cost_block["signals"][sig_name] = {
                "aggregate": final,
                "folds": fold_rows,
            }
            base = final["baseline"]
            print("    AGG:")
            for name, x in final.items():
                d_ret = x["return_pct"] - base["return_pct"]
                d_dd = x["max_dd_pct"] - base["max_dd_pct"]
                d_sh = x["sharpe"] - base["sharpe"]
                print(
                    f"      {name:<25} ret {x['return_pct']:+7.2f}% "
                    f"dd {x['max_dd_pct']:5.2f}% sh {x['sharpe']:+5.2f} "
                    f"dRet {d_ret:+7.2f} dDD {d_dd:+6.2f} dSh {d_sh:+5.2f} "
                    f"dual {x['dual_wins']}/{k_folds}"
                )
        out["costs"].append(cost_block)

    path = STATE_DIR / "predictive_csd_backtest.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults written to {path}")


if __name__ == "__main__":
    main()
