"""Component-level diagnostic for the CSD signal.

The composite csd_risk overlaps heavily with trailing vol because phi(AR1)
and variance both rise mechanically when realized vol rises. This script
isolates each component — phi, recovery_rate, variance, low_freq_power,
abs_skew, max_eigenvalue, well_depth — and asks two questions per
component:

  1. Forward drawdown discrimination: AUC and Q5/Q1 ratio at the chosen
     drawdown threshold over the chosen horizon.
  2. PARTIAL IC vs forward vol, controlling for trail vol — i.e., does this
     component predict forward vol BEYOND what trail vol already explains?
     Done by residualizing forward vol on trail vol (rank-regression) and
     IC-ing the residual against the component.

Best components by these two checks become candidates for a refined
governor that doesn't double-count vol.

Run:
    .\.venv\Scripts\python.exe -m backend.csd_component_diag
"""
from __future__ import annotations

import json
import math
import statistics
from typing import List, Tuple

from . import csd, lab
from .config import STATE_DIR


# ---------------------------------------------------------- math helpers
def _returns(closes):
    out = [0.0]
    for i in range(1, len(closes)):
        p = closes[i - 1]
        out.append((closes[i] - p) / p if p else 0.0)
    return out


def _rolling_std(xs, w):
    out = [0.0] * len(xs)
    for i in range(w, len(xs)):
        win = xs[i - w:i]
        m = sum(win) / w
        v = sum((x - m) ** 2 for x in win) / max(1, w - 1)
        out[i] = math.sqrt(v)
    return out


def _forward_vol(rets, i, H):
    end = min(len(rets), i + 1 + H)
    win = rets[i + 1:end]
    if len(win) < 2:
        return 0.0
    m = sum(win) / len(win)
    v = sum((x - m) ** 2 for x in win) / (len(win) - 1)
    return math.sqrt(v)


def _max_drawdown(closes, i, H):
    end = min(len(closes), i + 1 + H)
    if end <= i + 1:
        return 0.0
    peak = closes[i]; mdd = 0.0
    for j in range(i + 1, end):
        peak = max(peak, closes[j])
        dd = (peak - closes[j]) / peak if peak else 0.0
        if dd > mdd:
            mdd = dd
    return mdd


def _ranks(vs):
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


def _spearman(xs, ys):
    if len(xs) != len(ys) or len(xs) < 4:
        return 0.0
    rx = _ranks(xs); ry = _ranks(ys)
    n = len(xs)
    mx = sum(rx) / n; my = sum(ry) / n
    num = sum((rx[i] - mx) * (ry[i] - my) for i in range(n))
    dx = math.sqrt(sum((rx[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ry[i] - my) ** 2 for i in range(n)))
    return num / (dx * dy) if (dx * dy) else 0.0


def _partial_spearman(x, y, z) -> float:
    """Spearman correlation of x and y, controlling for z. Computed via
    rank-residuals: residualize rank(y) on rank(z) by OLS, then spearman
    of x against the residual. Returns 0 on degenerate input."""
    if len(x) != len(y) or len(y) != len(z) or len(x) < 6:
        return 0.0
    rx = _ranks(x); ry = _ranks(y); rz = _ranks(z)
    n = len(rx)
    mz = sum(rz) / n
    my = sum(ry) / n
    den = sum((rz[i] - mz) ** 2 for i in range(n))
    if den <= 0:
        return 0.0
    b = sum((rz[i] - mz) * (ry[i] - my) for i in range(n)) / den
    a = my - b * mz
    resid = [ry[i] - (a + b * rz[i]) for i in range(n)]
    return _spearman(rx, resid)


def _auc(scores, labels):
    pos = [scores[i] for i in range(len(scores)) if labels[i] == 1]
    neg = [scores[i] for i in range(len(scores)) if labels[i] == 0]
    if not pos or not neg:
        return 0.5
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


# ---------------------------------------------------------- per-component pipeline
COMPONENTS = ["phi", "recovery_rate", "variance", "low_freq_power",
              "abs_skew", "max_eigenvalue", "well_depth"]


def build_bundles(bars, fv_period=32, window=96, multivariate=True):
    closes = [b["close"] for b in bars]
    extras = None
    if multivariate:
        spreads = [float(b.get("spread", 0.0)) for b in bars]
        ois = [float(b.get("oi", 0.0)) for b in bars]
        extras = {}
        if any(spreads):
            d = [0.0] + [spreads[i] - spreads[i - 1] for i in range(1, len(spreads))]
            extras["spread"] = d
        if any(ois):
            d = [0.0] + [ois[i] - ois[i - 1] for i in range(1, len(ois))]
            extras["oi"] = d
    return csd.rolling_metrics(closes, fv_period=fv_period, window=window,
                                  step=1, extras_series=extras or None)


def diagnose(bars, bundles, H=12, dd_threshold=0.005, trail_win=24,
              top_q=0.2) -> dict:
    """For every component, report:
      - n        (samples with all values defined)
      - ic       Spearman vs forward vol
      - ic_par   Spearman vs forward vol PARTIAL on trail vol
      - ic_dd    Spearman vs forward MDD
      - ic_dd_p  Spearman vs forward MDD PARTIAL on trail vol
      - auc_dd   AUC of THIS component vs binary "forward MDD >= threshold"
      - Q_ratio  Q5/Q1 forward MDD ratio when sorted by THIS component
      - mean_q1, mean_q5  the actual quintile means (sanity)
    """
    closes = [b["close"] for b in bars]
    rets = _returns(closes)
    trail = _rolling_std(rets, trail_win)
    n = len(bars)
    out = {}
    for comp in COMPONENTS:
        xs, ys_fv, ys_mdd, ys_tv = [], [], [], []
        for i in range(n - H - 1):
            bun = bundles[i] if i < len(bundles) else {}
            if not bun or comp not in bun or trail[i] == 0.0:
                continue
            v = bun[comp]
            fv = _forward_vol(rets, i, H)
            mdd = _max_drawdown(closes, i, H)
            if fv <= 0:
                continue
            xs.append(v); ys_fv.append(fv); ys_mdd.append(mdd); ys_tv.append(trail[i])
        if len(xs) < 50:
            out[comp] = {"n": len(xs)}
            continue

        ic_fv = _spearman(xs, ys_fv)
        ic_fv_p = _partial_spearman(xs, ys_fv, ys_tv)
        ic_dd = _spearman(xs, ys_mdd)
        ic_dd_p = _partial_spearman(xs, ys_mdd, ys_tv)

        # AUC for binary "forward MDD >= threshold"
        labels = [1 if m >= dd_threshold else 0 for m in ys_mdd]
        auc_dd = _auc(xs, labels)
        # Quintile binning of THIS component
        pairs = sorted(zip(xs, ys_mdd), key=lambda p: p[0])
        Q = 5
        bs = len(pairs) // Q
        q_means = []
        for q in range(Q):
            chunk = pairs[q * bs:(q + 1) * bs if q < Q - 1 else len(pairs)]
            q_means.append(statistics.mean(p[1] for p in chunk))
        q_ratio = q_means[-1] / q_means[0] if q_means[0] > 0 else float("inf")

        out[comp] = {
            "n": len(xs),
            "ic_fwd_vol": round(ic_fv, 4),
            "ic_fwd_vol_partial_trail": round(ic_fv_p, 4),
            "ic_fwd_mdd": round(ic_dd, 4),
            "ic_fwd_mdd_partial_trail": round(ic_dd_p, 4),
            "auc_fwd_mdd_pos": round(auc_dd, 4),
            "q1_mean_mdd": round(q_means[0], 6),
            "q5_mean_mdd": round(q_means[-1], 6),
            "q5_q1_ratio": round(q_ratio, 3),
            "pos_rate_dd_thresh": round(sum(labels) / len(labels), 4),
        }
    return out


def main():
    print("Loading Kalshi 3m bars…")
    rows = lab.load(use_cache=True)
    bars = lab.aggregate(rows, 3)
    print(f"  {len(bars):,} 3m bars\n")

    print("Computing bundles (multivariate)…")
    bundles_mv = build_bundles(bars, fv_period=32, window=96, multivariate=True)
    print("Computing bundles (univariate only)…")
    bundles_uv = build_bundles(bars, fv_period=32, window=96, multivariate=False)

    print("\nDiagnostic on Kalshi 3m  (H=12 bars / dd_threshold=0.5%)")
    print("=" * 95)
    diag_mv = diagnose(bars, bundles_mv, H=12, dd_threshold=0.005)
    print(f"{'component':<18}{'n':>7}{'IC_fv':>9}{'IC_fv|tv':>10}"
          f"{'IC_dd':>9}{'IC_dd|tv':>10}{'AUC_dd':>9}"
          f"{'Q5/Q1':>8}{'pos_rate':>10}")
    print("-" * 95)
    for comp in COMPONENTS:
        d = diag_mv.get(comp, {"n": 0})
        if "ic_fwd_vol" not in d:
            print(f"{comp:<18}{d.get('n', 0):>7}{'--':>9}{'--':>10}"
                  f"{'--':>9}{'--':>10}{'--':>9}{'--':>8}{'--':>10}")
            continue
        print(f"{comp:<18}{d['n']:>7}{d['ic_fwd_vol']:>+9.4f}"
              f"{d['ic_fwd_vol_partial_trail']:>+10.4f}"
              f"{d['ic_fwd_mdd']:>+9.4f}{d['ic_fwd_mdd_partial_trail']:>+10.4f}"
              f"{d['auc_fwd_mdd_pos']:>9.4f}{d['q5_q1_ratio']:>8.3f}"
              f"{d['pos_rate_dd_thresh']:>10.4f}")

    print("\nDiagnostic on spot 15m would be a stronger test — but the Kalshi")
    print("test is what actually matters for the live wiring. The 'partial'")
    print("columns show information BEYOND trail vol — a real CSD edge must")
    print("show positive ic_fv|tv or ic_dd|tv. If those are ~0, the component")
    print("is a slow function of trail vol.")

    out = {"H": 12, "dd_threshold": 0.005,
            "diag_kalshi_3m": diag_mv}
    p = STATE_DIR / "csd_component_diag.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nResults written to {p}")


if __name__ == "__main__":
    main()
