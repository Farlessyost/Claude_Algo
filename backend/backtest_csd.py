"""Validate the Critical Slowing Down signal on historical KXBTCPERP bars.

The hypothesis: CSD risk @ t (computed only from history <= t) leads forward
realized volatility, large moves, and large drawdowns. If true, CSD is useful
as a *risk governor* — not as alpha, but as a gate that pulls position size
down before stress events and lets size go back to normal during stable churn.

The tests here decide go/no-go. If the signal is no better than a coin flip
against forward stress, we DON'T wire it into the live executor.

Tests run:
  T1  Predictive IC: rank correlation between CSD risk @ t and forward
       realized vol over [t+1, t+H]. Compare to the same IC for trailing vol
       (the obvious naive predictor) — CSD has to ADD predictive power.
  T2  AUC: how well does CSD risk @ t separate top-decile forward-vol
       outcomes from the rest?
  T3  Quintile attribution: forward stats by CSD-risk quintile (do top
       quintiles really have higher forward vol / drawdown?).
  T4  Null shuffle: shuffle the CSD risk series and recompute the IC. If
       shuffled ICs match the real one, the signal is noise.
  T5  Lead-lag profile: IC of CSD @ t vs forward vol at horizons H ∈
       {1, 3, 6, 12, 24} bars. A real leading indicator peaks at small
       positive H and decays.

Run:
    .\.venv\Scripts\python.exe -m backend.backtest_csd
"""
from __future__ import annotations

import json
import math
import random
import statistics
from typing import List, Tuple

from . import csd, lab
from .config import STATE_DIR


# ---------------------------------------------------------- helpers
def _returns(closes: List[float]) -> List[float]:
    out = [0.0]
    for i in range(1, len(closes)):
        p = closes[i - 1]
        out.append((closes[i] - p) / p if p else 0.0)
    return out


def _rolling_std(xs: List[float], w: int) -> List[float]:
    out = [0.0] * len(xs)
    for i in range(w, len(xs)):
        win = xs[i - w:i]
        m = sum(win) / w
        v = sum((x - m) ** 2 for x in win) / max(1, w - 1)
        out[i] = math.sqrt(v)
    return out


def _forward_vol(returns: List[float], i: int, H: int) -> float:
    """Realized vol over [i+1, i+H] (exclusive of current bar)."""
    end = min(len(returns), i + 1 + H)
    win = returns[i + 1:end]
    if len(win) < 2:
        return 0.0
    m = sum(win) / len(win)
    v = sum((x - m) ** 2 for x in win) / (len(win) - 1)
    return math.sqrt(v)


def _spearman(xs: List[float], ys: List[float]) -> float:
    """Spearman rank correlation. Ties broken by average rank."""
    if len(xs) != len(ys) or len(xs) < 4:
        return 0.0

    def ranks(vs):
        order = sorted(range(len(vs)), key=lambda i: vs[i])
        r = [0.0] * len(vs)
        i = 0
        while i < len(vs):
            j = i
            while j + 1 < len(vs) and vs[order[j + 1]] == vs[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r

    rx = ranks(xs); ry = ranks(ys)
    n = len(xs)
    mx = sum(rx) / n; my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    den = dx * dy
    return num / den if den else 0.0


def _auc(scores: List[float], labels: List[int]) -> float:
    """ROC AUC via Mann-Whitney U. labels are 0/1."""
    pos = [scores[i] for i in range(len(scores)) if labels[i] == 1]
    neg = [scores[i] for i in range(len(scores)) if labels[i] == 0]
    if not pos or not neg:
        return 0.5
    # rank-sum trick
    combined = sorted([(s, 1) for s in pos] + [(s, 0) for s in neg],
                       key=lambda t: t[0])
    rank_sum_pos = 0.0
    i = 0
    while i < len(combined):
        j = i
        while j + 1 < len(combined) and combined[j + 1][0] == combined[i][0]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            if combined[k][1] == 1:
                rank_sum_pos += avg_rank
        i = j + 1
    n_pos = len(pos); n_neg = len(neg)
    U = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    return U / (n_pos * n_neg)


# ---------------------------------------------------------- build CSD series
def build_csd_series(bars: List[dict],
                     fv_period: int = 32,
                     window: int = 96,
                     use_extras: bool = True) -> Tuple[List[dict], List[float]]:
    """Walk forward across the bar history, compute CSD bundle and risk at
    every bar (using only data <= i). Returns (per-bar bundles, per-bar risk).

    The risk is computed with z-scores against a TRAILING history of the same
    metrics (so it's a properly walk-forward, no-look-ahead estimate).
    """
    closes = [b["close"] for b in bars]
    extras_series = None
    if use_extras:
        # spread returns (centered) and log-volume changes
        spreads = [float(b.get("spread", 0.0)) for b in bars]
        vols = [float(b.get("volume", 0.0)) for b in bars]
        ois = [float(b.get("oi", 0.0)) for b in bars]
        # 1-bar diffs to make these stationary
        d_spread = [0.0] + [spreads[i] - spreads[i - 1] for i in range(1, len(spreads))]
        d_vol = [0.0] + [vols[i] - vols[i - 1] for i in range(1, len(vols))]
        d_oi = [0.0] + [ois[i] - ois[i - 1] for i in range(1, len(ois))]
        extras_series = {}
        if any(spreads):
            extras_series["spread"] = d_spread
        if any(vols):
            extras_series["volume"] = d_vol
        if any(ois):
            extras_series["oi"] = d_oi

    bundles = csd.rolling_metrics(closes, fv_period=fv_period,
                                    window=window, step=1,
                                    extras_series=extras_series)
    # Trailing window for z-score history (separate from the CSD window).
    risk_history_w = max(window, 200)
    risks: List[float] = [0.0] * len(bars)
    bundle_history: List[dict] = []
    for i, b in enumerate(bundles):
        if not b:
            continue
        # use only the last risk_history_w bundles as the reference history
        ref = [h for h in bundle_history[-risk_history_w:] if h]
        score = csd.csd_score(b, ref) if len(ref) >= 12 else 0.0
        risks[i] = csd.csd_risk(score)
        bundle_history.append(b)
    return bundles, risks


# ---------------------------------------------------------- tests
def test_predictive_ic(bars: List[dict], risks: List[float],
                        H_set=(1, 3, 6, 12, 24)) -> dict:
    """IC of CSD risk vs forward realized vol at multiple horizons.
    Also report IC of the trailing-vol baseline to see if CSD adds anything.
    """
    closes = [b["close"] for b in bars]
    rets = _returns(closes)
    trail_vol = _rolling_std(rets, 24)

    # Use only bars where both risk and trailing-vol are defined.
    out = {}
    for H in H_set:
        xs_risk, xs_trail, ys = [], [], []
        for i in range(len(bars) - H - 1):
            if risks[i] == 0.0 or trail_vol[i] == 0.0:
                continue
            fv = _forward_vol(rets, i, H)
            if fv <= 0:
                continue
            xs_risk.append(risks[i])
            xs_trail.append(trail_vol[i])
            ys.append(fv)
        ic_risk = _spearman(xs_risk, ys)
        ic_trail = _spearman(xs_trail, ys)
        out[H] = {
            "n": len(ys),
            "ic_csd_risk": round(ic_risk, 4),
            "ic_trailing_vol": round(ic_trail, 4),
            "marginal_ic": round(ic_risk - ic_trail, 4),
        }
    return out


def test_auc(bars: List[dict], risks: List[float], H: int = 6,
              top_q: float = 0.1) -> dict:
    """How well does CSD risk separate the top decile of forward-vol bars
    from the rest? Also compare against trailing-vol baseline.
    """
    closes = [b["close"] for b in bars]
    rets = _returns(closes)
    trail_vol = _rolling_std(rets, 24)

    pairs = []
    for i in range(len(bars) - H - 1):
        if risks[i] == 0.0 or trail_vol[i] == 0.0:
            continue
        fv = _forward_vol(rets, i, H)
        if fv <= 0:
            continue
        pairs.append((risks[i], trail_vol[i], fv))
    if not pairs:
        return {"n": 0}
    fvs = sorted(p[2] for p in pairs)
    cutoff = fvs[int((1 - top_q) * len(fvs))]
    risks_arr = [p[0] for p in pairs]
    trail_arr = [p[1] for p in pairs]
    labels = [1 if p[2] >= cutoff else 0 for p in pairs]
    return {
        "n": len(pairs),
        "pos_count": sum(labels),
        "auc_csd_risk": round(_auc(risks_arr, labels), 4),
        "auc_trailing_vol": round(_auc(trail_arr, labels), 4),
        "horizon_bars": H,
        "top_quantile": top_q,
    }


def test_quintile_attribution(bars: List[dict], risks: List[float],
                               H: int = 6) -> dict:
    """Bin (risk, forward-vol) pairs by risk quintile and report means.
    A real warning signal should show monotone forward-vol across quintiles.
    """
    closes = [b["close"] for b in bars]
    rets = _returns(closes)
    pairs = []
    for i in range(len(bars) - H - 1):
        if risks[i] == 0.0:
            continue
        fv = _forward_vol(rets, i, H)
        if fv <= 0:
            continue
        # also collect forward absolute return
        end = min(len(closes), i + 1 + H)
        if end <= i + 1:
            continue
        max_abs = max(abs((closes[j] - closes[i]) / closes[i]) for j in range(i + 1, end))
        pairs.append((risks[i], fv, max_abs))
    if len(pairs) < 25:
        return {"n": len(pairs)}
    pairs.sort(key=lambda p: p[0])
    Q = 5
    bs = len(pairs) // Q
    out = []
    for q in range(Q):
        chunk = pairs[q * bs:(q + 1) * bs if q < Q - 1 else len(pairs)]
        mean_risk = statistics.mean(p[0] for p in chunk)
        mean_fv = statistics.mean(p[1] for p in chunk)
        mean_maxabs = statistics.mean(p[2] for p in chunk)
        out.append({"q": q + 1, "n": len(chunk),
                    "mean_risk": round(mean_risk, 4),
                    "mean_fwd_vol": round(mean_fv, 6),
                    "mean_fwd_max_abs_ret": round(mean_maxabs, 6)})
    # ratio of Q5 to Q1 — quick read on monotonicity
    fv_ratio = out[-1]["mean_fwd_vol"] / out[0]["mean_fwd_vol"] if out[0]["mean_fwd_vol"] else float("inf")
    return {"n": len(pairs), "horizon_bars": H, "quintiles": out,
            "Q5_over_Q1_vol": round(fv_ratio, 3)}


def test_null_shuffle(bars: List[dict], risks: List[float],
                       n_shuffles: int = 200, H: int = 6) -> dict:
    """Compare real CSD IC against the distribution of ICs under random
    permutations of the risk series. p-value = fraction of shuffles whose IC
    exceeds the real one (one-tailed, looking for a positive predictive
    signal).
    """
    closes = [b["close"] for b in bars]
    rets = _returns(closes)
    xs, ys = [], []
    for i in range(len(bars) - H - 1):
        if risks[i] == 0.0:
            continue
        fv = _forward_vol(rets, i, H)
        if fv <= 0:
            continue
        xs.append(risks[i])
        ys.append(fv)
    if len(xs) < 30:
        return {"n": len(xs), "p_value": 1.0}
    real_ic = _spearman(xs, ys)
    rng = random.Random(1729)
    shuffles = []
    pool = list(xs)
    for _ in range(n_shuffles):
        rng.shuffle(pool)
        shuffles.append(_spearman(pool, ys))
    shuffles.sort()
    beats = sum(1 for s in shuffles if real_ic > s)
    p = 1.0 - beats / max(1, n_shuffles)
    return {
        "n": len(xs),
        "horizon_bars": H,
        "real_ic": round(real_ic, 4),
        "shuffle_mean_ic": round(statistics.mean(shuffles), 4),
        "shuffle_p95_ic": round(shuffles[int(0.95 * len(shuffles))], 4),
        "p_value": round(p, 4),
        "shuffles": n_shuffles,
    }


def test_lead_lag(bars: List[dict], risks: List[float]) -> dict:
    """IC vs forward vol at horizons -6..+24. A leading indicator should peak
    at small positive H and decay; a coincident/lagging signal won't."""
    closes = [b["close"] for b in bars]
    rets = _returns(closes)
    profile = {}
    horizons = [-6, -3, -1, 1, 3, 6, 12, 24]
    for H in horizons:
        xs, ys = [], []
        for i in range(max(0, -H), len(bars) - max(0, H) - 1):
            if risks[i] == 0.0:
                continue
            if H > 0:
                fv = _forward_vol(rets, i, H)
            else:
                # backward "forward vol": vol over [i+H, i]
                end = i + 1; start = max(0, i + H)
                win = rets[start:end]
                if len(win) < 2:
                    continue
                m = sum(win) / len(win)
                v = sum((x - m) ** 2 for x in win) / (len(win) - 1)
                fv = math.sqrt(v)
            if fv <= 0:
                continue
            xs.append(risks[i])
            ys.append(fv)
        if len(xs) >= 30:
            profile[H] = {"n": len(xs), "ic": round(_spearman(xs, ys), 4)}
        else:
            profile[H] = {"n": len(xs), "ic": None}
    return profile


# ---------------------------------------------------------- main
def main():
    print("Loading cached 1m rich history…")
    rows = lab.load(use_cache=True)
    bars = lab.aggregate(rows, 3)
    print(f"  {len(bars):,} 3m bars\n")

    print("Computing rolling CSD risk (window=96, fv_period=32, multivariate)…")
    bundles, risks = build_csd_series(bars, fv_period=32, window=96, use_extras=True)
    nonzero = sum(1 for r in risks if r != 0.0)
    rmean = statistics.mean(r for r in risks if r != 0.0) if nonzero else 0.0
    rstd = statistics.stdev(r for r in risks if r != 0.0) if nonzero > 1 else 0.0
    print(f"  bars with risk defined : {nonzero:,}/{len(risks):,}")
    print(f"  risk mean ± std        : {rmean:.3f} ± {rstd:.3f}\n")

    print("=" * 72)
    print("T1 — PREDICTIVE IC vs forward realized vol at multiple horizons")
    print("=" * 72)
    t1 = test_predictive_ic(bars, risks, H_set=(1, 3, 6, 12, 24))
    print(f"  {'H bars':>8}{'n':>8}{'IC(CSD risk)':>16}{'IC(trail vol)':>16}{'marginal':>12}")
    for H, d in t1.items():
        print(f"  {H:>8}{d['n']:>8}{d['ic_csd_risk']:>16.4f}{d['ic_trailing_vol']:>16.4f}{d['marginal_ic']:>+12.4f}")

    print("\n" + "=" * 72)
    print("T2 — AUC: separating the top-decile forward-vol bars (H=6)")
    print("=" * 72)
    t2 = test_auc(bars, risks, H=6, top_q=0.1)
    print(f"  bars              : {t2.get('n', 0):,}")
    print(f"  top-decile bars   : {t2.get('pos_count', 0):,}")
    print(f"  AUC (CSD risk)    : {t2.get('auc_csd_risk', 0):.4f}  "
          f"({'better than chance' if t2.get('auc_csd_risk', 0) > 0.55 else 'no separation'})")
    print(f"  AUC (trail vol)   : {t2.get('auc_trailing_vol', 0):.4f}")

    print("\n" + "=" * 72)
    print("T3 — QUINTILE ATTRIBUTION  (forward stats by risk quintile)")
    print("=" * 72)
    t3 = test_quintile_attribution(bars, risks, H=6)
    if "quintiles" in t3:
        print(f"  {'Q':>3}{'n':>6}{'mean_risk':>12}{'mean_fwd_vol':>16}{'mean_fwd_max_abs':>20}")
        for q in t3["quintiles"]:
            print(f"  {q['q']:>3}{q['n']:>6}{q['mean_risk']:>12.4f}"
                  f"{q['mean_fwd_vol']:>16.6f}{q['mean_fwd_max_abs_ret']:>20.6f}")
        print(f"  Q5/Q1 forward-vol ratio: {t3['Q5_over_Q1_vol']:.3f}  "
              f"({'monotone-ish (>=1.1)' if t3['Q5_over_Q1_vol'] >= 1.1 else 'weak/no separation'})")

    print("\n" + "=" * 72)
    print("T4 — NULL SHUFFLE  (does real CSD beat permuted risk labels?)")
    print("=" * 72)
    t4 = test_null_shuffle(bars, risks, n_shuffles=200, H=6)
    print(f"  bars              : {t4['n']:,}")
    print(f"  real IC           : {t4['real_ic']:+.4f}")
    print(f"  shuffle mean IC   : {t4['shuffle_mean_ic']:+.4f}")
    print(f"  shuffle 95% IC    : {t4['shuffle_p95_ic']:+.4f}")
    print(f"  p-value (1-tail)  : {t4['p_value']:.4f}  "
          f"({'SIGNIFICANT' if t4['p_value'] < 0.05 else 'not significant'})")

    print("\n" + "=" * 72)
    print("T5 — LEAD-LAG PROFILE  (IC at each horizon; leader peaks at small +H)")
    print("=" * 72)
    t5 = test_lead_lag(bars, risks)
    print(f"  {'H':>5}{'n':>8}{'IC':>10}")
    for H, d in t5.items():
        ic = d['ic']
        print(f"  {H:>+5}{d['n']:>8}{(ic if ic is not None else 0):>+10.4f}"
              f"{'  (insuff. data)' if ic is None else ''}")

    out = {"T1_ic_by_horizon": t1, "T2_auc": t2, "T3_quintiles": t3,
           "T4_null_shuffle": t4, "T5_lead_lag": t5,
           "n_bars": len(bars), "n_risk_defined": nonzero}
    out_path = STATE_DIR / "csd_backtest.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults written to {out_path}")


if __name__ == "__main__":
    main()
