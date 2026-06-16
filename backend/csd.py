"""Critical Slowing Down (CSD) metrics on the price-vs-fair-value deviation series.

The market-as-ecosystem hypothesis: when the local equilibrium is resilient,
shocks decay quickly (low AR1, high recovery rate, small variance). Before a
regime transition, the recovery slows, autocorrelation rises, variance rises,
and low-frequency power rises. This module computes those early-warning
indicators on a rolling deviation series x_t = log(price) - log(EMA_fv).

Univariate (functions of a single deviation series):
    phi_ar1               OLS AR(1) coefficient. phi -> 1 means slow recovery.
    recovery_rate         -log(phi)/dt. Falls as phi approaches 1.
    variance              Sample variance of x_t.
    low_freq_power        Fraction of periodogram power in the lowest octave.
    skewness / kurtosis   Higher moments — abs(skew) often rises pre-transition.
    well_depth            -log P(x) range over the empirical density. Deep
                          well = strong mean-reversion attractor.

Multivariate (functions of a vector deviation Y_t):
    var1_a_matrix         OLS-fit VAR(1) coefficient matrix A.
    max_eigenvalue        Largest |eigenvalue| of A via power iteration.
                          >=1 means cross-coupled disturbances aren't damping.

Composite:
    csd_score             z(AR1) + z(var) + z(lf_power) + z(|skew|) - z(recovery)
                          on z-scores computed against THIS WINDOW's history.
    csd_risk              Sigmoid of the score.

All pure Python — no numpy / scipy — to stay consistent with the rest of the
backend. Windows of 64-256 bars run in microseconds; the O(N^2) DFT is fine.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple


# ----------------------------------------------------------- math helpers
def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _var(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    m = _mean(xs)
    return sum((x - m) ** 2 for x in xs) / (n - 1)


def _std(xs: Sequence[float]) -> float:
    return math.sqrt(_var(xs))


def _zscore(x: float, xs: Sequence[float]) -> float:
    s = _std(xs)
    return (x - _mean(xs)) / s if s else 0.0


def _sigmoid(x: float) -> float:
    if x > 50:
        return 1.0
    if x < -50:
        return 0.0
    return 1.0 / (1.0 + math.exp(-x))


def _ema(xs: Sequence[float], period: int) -> List[float]:
    """Simple in-place EMA. Mirrors signals.ema_series but kept local so csd
    has no dependency on signals."""
    if not xs:
        return []
    k = 2.0 / (period + 1.0)
    out = [xs[0]]
    for v in xs[1:]:
        out.append(v * k + out[-1] * (1 - k))
    return out


def deviation_series(closes: Sequence[float], fv_period: int) -> List[float]:
    """x_t = log(close_t) - log(ema_fv_t), in log-price units. The deviation is
    what the local linear system is supposed to be mean-reverting toward 0."""
    if not closes:
        return []
    fv = _ema(closes, fv_period)
    out: List[float] = []
    for i, c in enumerate(closes):
        f = fv[i]
        if c > 0 and f > 0:
            out.append(math.log(c) - math.log(f))
        else:
            out.append(0.0)
    return out


# ----------------------------------------------------------- univariate CSD
def phi_ar1(xs: Sequence[float]) -> float:
    """OLS slope of x_{t+1} on x_t (assumes mean-centered deviation series).

    Estimator: phi = sum(x_t * x_{t+1}) / sum(x_t^2). Numerically stable, no
    matrix solve, no centering required since x_t is already a deviation."""
    if len(xs) < 4:
        return 0.0
    num = den = 0.0
    for i in range(len(xs) - 1):
        num += xs[i] * xs[i + 1]
        den += xs[i] * xs[i]
    if den <= 0:
        return 0.0
    return num / den


def recovery_rate(phi: float, dt: float = 1.0) -> float:
    """-log(phi)/dt. Defined only for 0 < phi < 1 (mean-reverting). Returns 0
    for phi <= 0 (oscillation isn't slow recovery, it's overshoot) and a large
    capped value for phi very small (very fast). Returns 0 for phi >= 1 by
    convention — the system isn't recovering at all."""
    if phi <= 0.0 or phi >= 1.0 or dt <= 0:
        return 0.0
    return -math.log(phi) / dt


def variance(xs: Sequence[float]) -> float:
    return _var(xs)


def skewness(xs: Sequence[float]) -> float:
    n = len(xs)
    if n < 3:
        return 0.0
    m = _mean(xs)
    s = _std(xs)
    if s <= 0:
        return 0.0
    return sum(((x - m) / s) ** 3 for x in xs) / n


def kurtosis(xs: Sequence[float]) -> float:
    """Excess kurtosis (Gaussian -> 0)."""
    n = len(xs)
    if n < 4:
        return 0.0
    m = _mean(xs)
    s = _std(xs)
    if s <= 0:
        return 0.0
    return sum(((x - m) / s) ** 4 for x in xs) / n - 3.0


def low_freq_power(xs: Sequence[float], low_band_frac: float = 0.125) -> float:
    """Fraction of periodogram power that lives in the lowest band.

    Detrends by subtracting the mean. Computes a direct DFT (O(N^2)) — fine
    for the ~64-256 bar windows CSD uses. Returns power in frequencies
    [1, K] divided by total power [1, N/2], where K = max(1, floor(N * frac)).
    Low DC is excluded so a constant doesn't dominate.
    """
    n = len(xs)
    if n < 8:
        return 0.0
    m = _mean(xs)
    ys = [x - m for x in xs]
    half = n // 2
    cutoff = max(1, int(half * low_band_frac))
    total = 0.0
    low = 0.0
    for k in range(1, half + 1):
        # X_k = sum_t y_t * exp(-2 pi i k t / N)
        wk = 2.0 * math.pi * k / n
        re = im = 0.0
        for t, y in enumerate(ys):
            re += y * math.cos(wk * t)
            im -= y * math.sin(wk * t)
        p = re * re + im * im
        total += p
        if k <= cutoff:
            low += p
    if total <= 0:
        return 0.0
    return low / total


def well_depth(xs: Sequence[float], bins: int = 20) -> float:
    """Empirical potential U(x) = -log P(x) over a histogram, then the depth
    of the well = max U - min U over occupied bins. A deep well means the
    deviation series has a sharp mode (strong attractor). A shallow well
    means the system is wandering — pre-transition signature.

    Returns 0 when there isn't enough variance to form a histogram.
    """
    n = len(xs)
    if n < bins * 2:
        return 0.0
    lo, hi = min(xs), max(xs)
    if hi - lo <= 0:
        return 0.0
    counts = [0] * bins
    width = (hi - lo) / bins
    for x in xs:
        idx = int((x - lo) / width)
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    # smooth (add-one) and convert to -log p
    occupied = [c + 1 for c in counts]
    total = sum(occupied)
    probs = [c / total for c in occupied]
    us = [-math.log(p) for p in probs]
    return max(us) - min(us)


# ----------------------------------------------------------- multivariate VAR(1)
def _solve_normal(xtx: List[List[float]], xty: List[float]) -> List[float]:
    """Solve (X'X) b = X'y via Gauss-Jordan. Returns [0]*K on singular."""
    k = len(xty)
    a = [row[:] + [xty[i]] for i, row in enumerate(xtx)]
    for col in range(k):
        # partial pivot
        piv = col
        for r in range(col + 1, k):
            if abs(a[r][col]) > abs(a[piv][col]):
                piv = r
        if abs(a[piv][col]) < 1e-12:
            return [0.0] * k
        if piv != col:
            a[col], a[piv] = a[piv], a[col]
        # normalize
        d = a[col][col]
        for j in range(col, k + 1):
            a[col][j] /= d
        # eliminate
        for r in range(k):
            if r == col:
                continue
            f = a[r][col]
            if abs(f) < 1e-15:
                continue
            for j in range(col, k + 1):
                a[r][j] -= f * a[col][j]
    return [a[i][k] for i in range(k)]


def var1_a_matrix(Y: List[List[float]]) -> List[List[float]]:
    """Fit a VAR(1) Y_{t+1} = A Y_t + eps by row-wise OLS (no intercept;
    expects centered deviation series). Y is a list of K series, each length N.

    Returns the K x K matrix A. Returns the zero matrix if there aren't
    enough samples or the design is singular.
    """
    if not Y:
        return []
    k = len(Y)
    n = len(Y[0])
    if n < 8:
        return [[0.0] * k for _ in range(k)]
    # Build X = Y[:, :-1].T (rows = time t = 0..n-2, cols = k vars)
    rows = n - 1
    # X'X is k x k. X'Y_i is k. Compute X'X once.
    xtx = [[0.0] * k for _ in range(k)]
    for i in range(k):
        for j in range(i, k):
            s = 0.0
            for t in range(rows):
                s += Y[i][t] * Y[j][t]
            xtx[i][j] = s
            xtx[j][i] = s
    A = [[0.0] * k for _ in range(k)]
    for target in range(k):
        xty = [0.0] * k
        for i in range(k):
            s = 0.0
            yi = Y[i]; yt = Y[target]
            for t in range(rows):
                s += yi[t] * yt[t + 1]
            xty[i] = s
        A[target] = _solve_normal(xtx, xty)
    return A


def max_eigenvalue(A: List[List[float]], n_iter: int = 64) -> float:
    """Largest |eigenvalue| of square matrix A via power iteration. Uses a
    real, possibly-complex eigenvalue magnitude via Rayleigh quotient on A^2
    when needed. Returns 0 for an empty / 1x1 matrix with zero entry."""
    if not A or not A[0]:
        return 0.0
    k = len(A)
    if k == 1:
        return abs(A[0][0])
    # power iteration with deflation-free Rayleigh
    v = [1.0 / math.sqrt(k)] * k
    last = 0.0
    for _ in range(n_iter):
        # w = A v
        w = [sum(A[i][j] * v[j] for j in range(k)) for i in range(k)]
        nrm = math.sqrt(sum(x * x for x in w))
        if nrm < 1e-15:
            return 0.0
        v = [x / nrm for x in w]
        # Rayleigh quotient: lam = v' A v
        Av = [sum(A[i][j] * v[j] for j in range(k)) for i in range(k)]
        lam = sum(v[i] * Av[i] for i in range(k))
        if abs(lam - last) < 1e-9:
            break
        last = lam
    return abs(last)


# ----------------------------------------------------------- composite score
def csd_score(metrics_now: dict, metrics_history: List[dict]) -> float:
    """Combine the latest metrics into a single CSD score by z-scoring each
    component against the history of recent values. Sign convention:
       higher AR1, variance, low-freq power, |skew|  -> riskier  (+)
       higher recovery rate                          -> safer   (-)
       higher max_eigenvalue                         -> riskier (+)
       higher well_depth                             -> safer   (-)
    """
    if not metrics_history:
        return 0.0

    def col(key: str) -> List[float]:
        return [m.get(key, 0.0) for m in metrics_history if key in m]

    def z(key: str, sign: int) -> float:
        h = col(key)
        if len(h) < 6:
            return 0.0
        return sign * _zscore(metrics_now.get(key, 0.0), h)

    parts = [
        z("phi", +1),
        z("variance", +1),
        z("low_freq_power", +1),
        z("abs_skew", +1),
        z("max_eigenvalue", +1),
        z("recovery_rate", -1),
        z("well_depth", -1),
    ]
    return sum(parts)


def csd_risk(score: float, gain: float = 0.5) -> float:
    """Sigmoid squash of the raw score. gain controls how aggressively the
    sigmoid saturates — 0.5 is a moderate slope (a +2 score gives ~0.73)."""
    return _sigmoid(score * gain)


# ----------------------------------------------------------- top-level driver
def compute(closes: Sequence[float],
            fv_period: int = 32,
            window: int = 96,
            extras: Optional[dict] = None) -> dict:
    """Compute the full CSD metric bundle on a single window of closes.

    Returns a dict with: phi, recovery_rate, variance, low_freq_power, skew,
    abs_skew, kurt, well_depth, and (if `extras` includes other series of
    matching length) max_eigenvalue. The caller is responsible for keeping
    a rolling history of these bundles and computing the score / risk.

    `extras` is an optional dict of side series (e.g., {"spread": [...],
    "volume": [...]}) used to fit a multivariate VAR(1) alongside the
    deviation series. Each side series must already be detrended /
    centered around 0 (e.g., log differences or z-scores).
    """
    out = {
        "phi": 0.0, "recovery_rate": 0.0, "variance": 0.0,
        "low_freq_power": 0.0, "skew": 0.0, "abs_skew": 0.0,
        "kurt": 0.0, "well_depth": 0.0, "max_eigenvalue": 0.0,
    }
    if len(closes) < max(window, fv_period * 2):
        return out
    closes_w = list(closes[-window:])
    xs = deviation_series(closes_w, fv_period)
    # center for AR(1) (small bias correction)
    m = _mean(xs)
    xs_c = [x - m for x in xs]
    phi = phi_ar1(xs_c)
    out["phi"] = phi
    out["recovery_rate"] = recovery_rate(phi)
    out["variance"] = variance(xs_c)
    out["low_freq_power"] = low_freq_power(xs_c)
    sk = skewness(xs_c)
    out["skew"] = sk
    out["abs_skew"] = abs(sk)
    out["kurt"] = kurtosis(xs_c)
    out["well_depth"] = well_depth(xs_c)

    if extras:
        # Build matrix Y: rows = variables, cols = time. The price deviation
        # is variable 0; each extra is appended. Centered, equal length.
        rows: List[List[float]] = [xs_c]
        for _, series in extras.items():
            s = list(series[-window:])
            if len(s) < len(xs_c):
                continue
            s = s[-len(xs_c):]
            ms = _mean(s)
            rows.append([v - ms for v in s])
        if len(rows) >= 2:
            A = var1_a_matrix(rows)
            out["max_eigenvalue"] = max_eigenvalue(A)
    return out


def current_refined_risk(closes: Sequence[float],
                          skew_history: Sequence[float],
                          fv_period: int = 32,
                          window: int = 96) -> Tuple[float, float]:
    """Online refined CSD risk (skew_only mode) for live use.

    Returns (risk, skew_now). The caller appends skew_now to its persistent
    history and passes the trailing 200 values back next cycle.

    The refinement uses ONLY abs_skew of the price-deviation series — the
    component the diagnostic showed carries information beyond trail vol
    (partial IC 0.055, AUC 0.61, Q5/Q1 1.46). Other components (phi,
    recovery_rate, low_freq_power, max_eigenvalue, well_depth) had null
    partial IC and were dropped — they were averaging noise into the
    composite.

    Returns (0.0, 0.0) when there isn't enough data for a single window or
    enough history for the z-score.
    """
    if len(closes) < max(window, fv_period * 2):
        return 0.0, 0.0
    xs = deviation_series(list(closes[-window:]), fv_period)
    m = _mean(xs)
    xs_c = [x - m for x in xs]
    sk_now = abs(skewness(xs_c))
    if len(skew_history) < 12:
        return 0.0, sk_now
    z = _zscore(sk_now, list(skew_history))
    return _sigmoid(z), sk_now


def rolling_metrics(closes: Sequence[float],
                    fv_period: int = 32,
                    window: int = 96,
                    step: int = 1,
                    extras_series: Optional[dict] = None
                    ) -> List[dict]:
    """Walk a rolling window across `closes`, computing the CSD bundle at
    each step. Returns one dict per bar (the first `window` bars get an
    empty bundle).

    `extras_series`: dict of {name: list of same length as closes}. For each
    window we slice the corresponding piece. None means univariate-only.
    """
    n = len(closes)
    out: List[dict] = [dict() for _ in range(n)]
    if n < window:
        return out
    closes_list = list(closes)
    for i in range(window, n, step):
        w_closes = closes_list[i - window:i]
        extras_w = None
        if extras_series:
            extras_w = {k: v[i - window:i] for k, v in extras_series.items()}
        out[i] = compute(w_closes, fv_period=fv_period,
                          window=window, extras=extras_w)
    return out
