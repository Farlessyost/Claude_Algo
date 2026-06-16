"""Offline nonlinear-MPC research candidate.

This does NOT wire nonlinear MPC into live trading. It tests whether a simple
nonlinear controller deserves promotion against the current robust MPC:

    maximize over target q:
        alpha * q
      - risk_lambda * ecological_risk * q^2
      - turnover_lambda * |q - q_prev|^p
      - cap_barrier * (|q| / cap)^4

The optimizer is a deterministic grid search over q in [-cap, cap]. That keeps
the experiment dependency-free, repeatable, and safe to run in the background.

Run:
    .\\.venv\\Scripts\\python.exe -m backend.backtest_nonlinear_mpc
"""
from __future__ import annotations

import json
import math
from typing import Dict, List, Tuple

from . import ecology, lab, signals
from .config import DEFAULT_STRATEGY_PARAMS, STATE_DIR


def make_params() -> dict:
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


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = sum(xs) / len(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _grid(cap: float, n: int = 41) -> List[float]:
    if cap <= 0:
        return [0.0]
    if n < 3:
        n = 3
    return [-cap + 2.0 * cap * i / (n - 1) for i in range(n)]


PHASE_MOD = {
    "producer":   {"gain": 0.80, "cap": 0.60, "tox": 0.15},
    "predator":   {"gain": 0.35, "cap": 0.25, "tox": 1.30},
    "exhaustion": {"gain": 0.60, "cap": 0.45, "tox": 0.80},
    "scavenger":  {"gain": 1.20, "cap": 1.00, "tox": 0.25},
    "decomposer": {"gain": 0.70, "cap": 0.50, "tox": 0.55},
    "churn":      {"gain": 0.90, "cap": 0.70, "tox": 0.35},
}


def nonlinear_mpc_with_aux(candles: List[dict], p: dict,
                           cfg: dict) -> Tuple[List[float], List[float]]:
    closes = [float(c["close"]) for c in candles]
    n = len(closes)
    if n < 5:
        return [0.0] * n, [0.0] * n

    vw = int(p.get("vol_win", 12))
    lb = int(p.get("lookback", 2))
    rw = int(p.get("regime_win", 8))
    beta = float(p.get("beta", 0.18))
    gain = float(p.get("gain", 1.0))
    er_cap = float(p.get("er_cap", 1.0))
    risk_lambda = float(cfg.get("risk_lambda", 0.08))
    turnover_lambda = float(cfg.get("turnover_lambda", 0.08))
    turnover_power = float(cfg.get("turnover_power", 1.5))
    alpha_saturation = float(cfg.get("alpha_saturation", 1.0))
    cap_barrier = float(cfg.get("cap_barrier", 0.03))
    grid_n = int(cfg.get("grid_n", 41))

    try:
        _, states = ecology.phase_series_with_states(candles, p)
    except Exception:
        states = ["churn"] * n

    r = [0.0] + [
        (closes[i] - closes[i - 1]) / closes[i - 1] if closes[i - 1] else 0.0
        for i in range(1, n)
    ]
    er = signals.efficiency_ratio(closes, rw)
    pos = [0.0] * n
    urgency = [0.0] * n
    cur = 0.0
    recent_vols: List[float] = []
    start = max(vw, lb, rw) + 1

    for i in range(start, n):
        move = sum(r[i - lb + 1:i + 1])
        vol = _std(r[i - vw:i]) or 1e-9
        recent_vols.append(vol)
        vol_base = (sum(recent_vols[-32:]) / len(recent_vols[-32:])) if recent_vols[-32:] else vol
        vol_stress = max(0.0, (vol / (vol_base or 1e-9)) - 1.0)
        scale = max(0.0, 1.0 - er[i] / max(er_cap, 1e-6))
        phase = states[i] if i < len(states) else "churn"
        mod = PHASE_MOD.get(phase, PHASE_MOD["churn"])
        raw_alpha = -beta * (move / vol)
        alpha = math.tanh(alpha_saturation * gain * mod["gain"] * raw_alpha * scale)
        cap = float(mod["cap"])
        eco_risk = 1.0 + float(mod["tox"]) + float(p.get("robust_lambda", 0.35)) * vol_stress

        best_q = cur
        best_score = -1e99
        for q in _grid(cap, grid_n):
            abs_cap = abs(q) / max(cap, 1e-9)
            score = (
                alpha * q
                - risk_lambda * eco_risk * q * q
                - turnover_lambda * (abs(q - cur) ** turnover_power)
                - cap_barrier * (abs_cap ** 4)
            )
            if score > best_score:
                best_score = score
                best_q = q
        cur = max(-cap, min(cap, best_q))
        pos[i] = cur
        urgency[i] = abs(alpha) + 0.5 * abs(pos[i] - pos[i - 1])

    return pos, urgency


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
    return {"return_pct": 0.0, "max_dd_pct": 0.0, "sharpe": 0.0, "trades": 0, "wins": 0}


def add_fold(agg: dict, x: dict, base: dict | None = None) -> None:
    agg["return_pct"] += float(x.get("return_pct") or 0.0)
    agg["max_dd_pct"] = max(agg["max_dd_pct"], float(x.get("max_dd_pct") or 0.0))
    agg["sharpe"] += float(x.get("sharpe") or 0.0)
    agg["trades"] += int(x.get("trades") or 0)
    if base and float(x.get("return_pct") or 0.0) > float(base.get("return_pct") or 0.0):
        agg["wins"] += 1


def finish(agg: dict, k: int) -> dict:
    out = dict(agg)
    out["return_pct"] = round(out["return_pct"], 3)
    out["max_dd_pct"] = round(out["max_dd_pct"], 3)
    out["sharpe"] = round(out["sharpe"] / max(1, k), 3)
    return out


CONFIGS: List[dict] = [
    {"name": "nlm_balanced", "risk_lambda": 0.08, "turnover_lambda": 0.08,
     "turnover_power": 1.5, "alpha_saturation": 1.0, "cap_barrier": 0.03},
    {"name": "nlm_fast", "risk_lambda": 0.05, "turnover_lambda": 0.04,
     "turnover_power": 1.35, "alpha_saturation": 1.2, "cap_barrier": 0.02},
    {"name": "nlm_defensive", "risk_lambda": 0.14, "turnover_lambda": 0.10,
     "turnover_power": 1.6, "alpha_saturation": 0.9, "cap_barrier": 0.05},
    {"name": "nlm_sticky", "risk_lambda": 0.08, "turnover_lambda": 0.16,
     "turnover_power": 1.7, "alpha_saturation": 1.0, "cap_barrier": 0.03},
]


def main() -> None:
    print("Loading cached rich history...")
    rows = lab.load(use_cache=True)
    bars = lab.aggregate(rows, 3)
    print(f"  3m bars: {len(bars):,}")
    if len(bars) < 800:
        print("not enough bars")
        return

    params = make_params()
    k_folds = 8
    costs = [0.0, 4.0, 7.0]
    leverage = 5.8
    n = len(bars)
    fold_size = n // k_folds
    out = {"bars": n, "folds": k_folds, "params": params, "configs": CONFIGS, "costs": []}

    for hs in costs:
        print(f"\n=== half_spread={hs:.1f} bps ===")
        aggs: Dict[str, dict] = {"robust_mpc": empty_agg()}
        for cfg in CONFIGS:
            aggs[cfg["name"]] = empty_agg()
        folds = []
        for fi in range(k_folds):
            lo = fi * fold_size
            hi = n if fi == k_folds - 1 else (fi + 1) * fold_size
            fold = bars[lo:hi]
            pos_r, urg_r = signals.robust_mpc_with_aux(fold, params)
            base = sim(fold, pos_r, urg_r, hs, leverage)
            add_fold(aggs["robust_mpc"], base)
            row = {"fold": fi + 1, "robust_mpc": base, "nonlinear": {}}
            line = f"  fold {fi + 1}: robust {base['return_pct']:+6.2f}% dd {base['max_dd_pct']:5.2f}%"
            for cfg in CONFIGS:
                pos_n, urg_n = nonlinear_mpc_with_aux(fold, params, cfg)
                x = sim(fold, pos_n, urg_n, hs, leverage)
                add_fold(aggs[cfg["name"]], x, base)
                row["nonlinear"][cfg["name"]] = x
                line += f" | {cfg['name'].replace('nlm_', '')[:4]} {x['return_pct']:+6.2f}%"
            print(line)
            folds.append(row)
        agg = {name: finish(v, k_folds) for name, v in aggs.items()}
        base = agg["robust_mpc"]
        print("  AGG:")
        for name, x in agg.items():
            print(
                f"    {name:<15} ret {x['return_pct']:+7.2f}% "
                f"dd {x['max_dd_pct']:5.2f}% sh {x['sharpe']:+5.2f} "
                f"dRet {x['return_pct'] - base['return_pct']:+7.2f} "
                f"dDD {x['max_dd_pct'] - base['max_dd_pct']:+6.2f} "
                f"wins {x['wins']}/{k_folds}"
            )
        out["costs"].append({"half_spread_bps": hs, "aggregate": agg, "folds": folds})

    path = STATE_DIR / "nonlinear_mpc_backtest.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults written to {path}")


if __name__ == "__main__":
    main()
