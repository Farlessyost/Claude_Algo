"""Trophic Information Forager / Mycelial Alpha Network.

Instead of asking "where will BTC go next?", this module asks:
  - which species (price, funding, liquidations, depth, ...) is currently
    driving the others?
  - which ecological phase are we in (producer / predator / exhaustion /
    scavenger / decomposer / churn)?
  - which organism (trading archetype) fits this phase best?
  - is the network in equilibrium, or has it just been disturbed?

The output is a phase string + a size multiplier (applied to the MPC
controller's |target_fraction| when settings.ecosystem_phase is on) plus
diagnostics the UI renders as a food-web + phase-ring animation.

The information network is built per-cycle from on-platform features
(price, vol, spread, depth, OI, funding) plus, when available, multi-asset
nodes from `multiasset.py` (BTC spot, ETH perp, Binance OI / liq proxy).
We approximate "information flow" with lagged Pearson correlation:
    flow_AB = corr(A[t-lag], B[t])
which is a directional, light-weight stand-in for transfer entropy that's
stable on ~hundreds of samples (proper TE needs binning + thousands).

Phases:
  PRODUCER     baseline market-making: low vol, tight spread, no flow
  PREDATOR     cascade in progress: vol_z>>0, spread_z>>0, OI dropping
  EXHAUSTION   predator slowing: vol still high but decelerating, spread peaked
  SCAVENGER    snap-back zone: stretched price, vol decelerating, spread normalizing
  DECOMPOSER   liquidity restoration: vol back near baseline, spread compressing
  CHURN        normal range, default state

Organisms (scored every cycle; the live engine still trades one position,
the mycelial allocator is informational for now):
  predator     momentum follower (would profit from cascade continuation)
  scavenger    mean-reversion bot (would profit from snap-back)
  decomposer   spread-collecting market-maker
  mycelium     meta-router (allocates to the others)
  immune       risk shutdown (suggests stand-aside under disturbance)
  producer     passive liquidity in quiet markets
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from . import signals
from . import multiasset

PHASES = ("producer", "predator", "exhaustion", "scavenger", "decomposer", "churn")
ORGANISMS = ("predator", "scavenger", "decomposer", "mycelium", "immune", "producer")

# Phase succession order (clockwise on the UI ring). The ring is a 6-cycle
# with churn at center: producer -> predator -> exhaustion -> scavenger
# -> decomposer -> producer (or back to churn).
SUCCESSION = ("producer", "predator", "exhaustion", "scavenger",
              "decomposer", "churn")

TIME_SCALES = ("seconds", "minutes", "hours", "days")

NODE_TIME_SCALE = {
    # 1s microstructure / fast guard layer
    "live_mid": "seconds",
    "live_spread_bps": "seconds",
    "live_imb_top": "seconds",
    "live_imb_depth": "seconds",
    "live_depth_total": "seconds",
    "cb_btc_depth_imbalance": "seconds",
    "cb_btc_spread_bps": "seconds",
    "cb_eth_depth_imbalance": "seconds",
    "hl_btc_book_imbalance": "seconds",
    # cycle/minute trading layer
    "price": "minutes",
    "vol": "minutes",
    "range": "minutes",
    "volume": "minutes",
    "spread": "minutes",
    "oi_change": "minutes",
    "eth_perp": "minutes",
    "btc_spot": "minutes",
    "eth_spot": "minutes",
    "liq_proxy": "minutes",
    "funding_bn": "minutes",
    "deribit_perp_basis": "minutes",
    "deribit_term_basis": "minutes",
    "deribit_oi_change": "minutes",
    "kraken_coinbase_basis_bps": "minutes",
    "kraken_spread_bps": "minutes",
    "kraken_bbo_depth": "minutes",
    "kraken_eth_btc_lead": "minutes",
    "hl_alt_breadth": "minutes",
    "hl_alt_avg_change_pct": "minutes",
    "hl_total_oi_change": "minutes",
    "hl_premium_z": "minutes",
    # slower structure
    "deribit_funding_8h": "hours",
    "deribit_option_skew": "hours",
    "btc_fee_fastest": "hours",
    "btc_fee_half_hour": "hours",
    "btc_fee_pressure": "hours",
    "mempool_congestion_z": "hours",
    "top10_breadth": "hours",
    "risk_on_alt_breadth": "hours",
    "crypto_total_mcap_change": "days",
    "btc_dominance_change": "days",
}


# ----------------------------------------------------------- math helpers
def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _zscore(x: float, xs: List[float]) -> float:
    s = _std(xs)
    return (x - _mean(xs)) / s if s else 0.0


def _pearson(xs: List[float], ys: List[float]) -> float:
    n = min(len(xs), len(ys))
    if n < 4:
        return 0.0
    xs = xs[-n:]; ys = ys[-n:]
    mx = _mean(xs); my = _mean(ys)
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = math.sqrt(sum((xs[i] - mx) ** 2 for i in range(n)))
    dy = math.sqrt(sum((ys[i] - my) ** 2 for i in range(n)))
    den = dx * dy
    return num / den if den else 0.0


def _diffs(xs: List[float]) -> List[float]:
    return [xs[i] - xs[i - 1] for i in range(1, len(xs))]


def _returns(xs: List[float]) -> List[float]:
    out = []
    for i in range(1, len(xs)):
        p = xs[i - 1]
        out.append((xs[i] - p) / p if p else 0.0)
    return out


def _rolling_z_series(xs: List[float], min_hist: int = 6) -> List[float]:
    """Per-sample z-score against prior history. Flat until enough history."""
    out: List[float] = []
    for i, x in enumerate(xs):
        hist = xs[max(0, i - 32):i]
        out.append(_zscore(x, hist) if len(hist) >= min_hist else 0.0)
    return out


# ------------------------------------------- node-series extraction
def _extract_node_series(candles: List[dict], market: dict, multi: dict,
                        win: int) -> Dict[str, List[float]]:
    """Return a dict of node_name -> recent timeseries (length up to `win`).
    Each series is a per-bar SCALAR derived from on-platform candles, the
    live market view, and the multi-asset history. Series are returned as
    "rates" / "changes" where appropriate so the lagged-correlation matrix
    captures directional flow rather than level coupling."""
    out: Dict[str, List[float]] = {}
    closes = [c["close"] for c in (candles or [])][-win:]
    if len(closes) < 8:
        return out
    out["price"] = _returns(closes)                                 # price returns
    # realized vol per-bar: |return|
    out["vol"] = [abs(r) for r in out["price"]]
    # range vol per-bar: high-low / close, in bps
    rng_vol = []
    for c in (candles or [])[-(len(out["price"]) + 1):]:
        px = c.get("close") or 0.0
        if px:
            rng_vol.append((float(c.get("high") or px) - float(c.get("low") or px))
                            / px * 10_000.0)
        else:
            rng_vol.append(0.0)
    out["range"] = rng_vol[-len(out["price"]):]
    # volume per bar
    vols = [float(c.get("volume") or 0.0) for c in (candles or [])[-win:]]
    out["volume"] = _diffs(vols)
    # OI per bar (if available on rich candles)
    ois = [float(c.get("oi") or 0.0) for c in (candles or [])[-win:] if c.get("oi") is not None]
    if len(ois) >= 8:
        out["oi_change"] = _diffs(ois)
    # spread per bar (rich candles carry it)
    spreads = [float(c.get("spread") or 0.0) for c in (candles or [])[-win:]]
    if any(spreads):
        out["spread"] = spreads[-len(out["price"]):]

    # Live orderbook spread + depth — only a single sample, but multi-asset
    # history (multiasset.HISTORY) carries the rolling series.
    live_mid = multiasset.HISTORY.series("live_mid", n=max(win, 180))
    if len(live_mid) >= 8:
        out["live_mid"] = _returns([v for _, v in live_mid])
    for node in ("live_spread_bps", "live_imb_top", "live_imb_depth", "live_depth_total"):
        hist = multiasset.HISTORY.series(node, n=max(win, 180))
        if len(hist) >= 6:
            out[node] = [v for _, v in hist]

    eth_perp = multiasset.HISTORY.series("eth_perp", n=win)
    if len(eth_perp) >= 8:
        out["eth_perp"] = _returns([v for _, v in eth_perp])
    btc_spot = multiasset.HISTORY.series("btc_spot", n=win)
    if len(btc_spot) >= 8:
        out["btc_spot"] = _returns([v for _, v in btc_spot])
    eth_spot = multiasset.HISTORY.series("eth_spot", n=win)
    if len(eth_spot) >= 8:
        out["eth_spot"] = _returns([v for _, v in eth_spot])
    liq_proxy = multiasset.HISTORY.series("liq_proxy", n=win)
    if len(liq_proxy) >= 6:
        out["liq_proxy"] = [v for _, v in liq_proxy]
    funding_bn = multiasset.HISTORY.series("binance_funding", n=win)
    if len(funding_bn) >= 6:
        out["funding_bn"] = [v for _, v in funding_bn]

    # Expanded public-stream nodes. These are intentionally transformed
    # signals (basis, spreads, depth imbalance, OI deltas, breadth, z-scores),
    # not raw duplicate prices.
    direct_nodes = [
        "cb_btc_depth_imbalance", "cb_btc_spread_bps", "cb_eth_depth_imbalance",
        "deribit_perp_basis", "deribit_term_basis", "deribit_oi_change",
        "deribit_funding_8h", "deribit_option_skew",
        "kraken_coinbase_basis_bps", "kraken_spread_bps", "kraken_bbo_depth",
        "kraken_eth_btc_lead",
        "hl_alt_breadth", "hl_alt_avg_change_pct", "hl_total_oi_change",
        "hl_btc_book_imbalance",
        "top10_breadth", "risk_on_alt_breadth",
        "btc_fee_fastest", "btc_fee_half_hour", "btc_fee_pressure",
    ]
    for node in direct_nodes:
        hist = multiasset.HISTORY.series(node, n=win)
        if len(hist) >= 6:
            out[node] = [v for _, v in hist]

    for raw_node, out_node in (
        ("crypto_total_mcap", "crypto_total_mcap_change"),
        ("btc_dominance", "btc_dominance_change"),
    ):
        hist = multiasset.HISTORY.series(raw_node, n=win)
        if len(hist) >= 8:
            out[out_node] = _returns([v for _, v in hist])

    for raw_node, out_node in (
        ("hl_premium", "hl_premium_z"),
        ("mempool_vsize", "mempool_congestion_z"),
    ):
        hist = multiasset.HISTORY.series(raw_node, n=win)
        if len(hist) >= 8:
            out[out_node] = _rolling_z_series([v for _, v in hist])
    return out


# ---------------------------- information-network (lagged correlation)
def information_network(series: Dict[str, List[float]], lag: int = 1
                        ) -> Dict[Tuple[str, str], float]:
    """Estimate directed information flow A -> B as corr(A[t-lag], B[t]).
    Symmetric correlation goes into both (A,B) and (B,A); the directional
    edge weight is the ABSOLUTE value of the lagged correlation.

    A node's out-strength is the sum of |edge| over its out-neighbours and
    is reported as the centrality / "keystone" score. Self-edges are skipped.
    """
    edges: Dict[Tuple[str, str], float] = {}
    nodes = list(series.keys())
    for a in nodes:
        sa = series[a]
        if len(sa) < lag + 8:
            continue
        for b in nodes:
            if a == b:
                continue
            sb = series[b]
            if len(sb) < lag + 8:
                continue
            n = min(len(sa) - lag, len(sb))
            r = _pearson(sa[-(n + lag):-lag] if lag else sa[-n:],
                          sb[-n:])
            edges[(a, b)] = round(r, 4)
    return edges


def network_centrality(edges: Dict[Tuple[str, str], float],
                        nodes: List[str]) -> Dict[str, float]:
    out = {n: 0.0 for n in nodes}
    for (a, b), w in edges.items():
        out[a] = out.get(a, 0.0) + abs(w)
    # normalize so the most central node = 1.0 (UI uses this for glow intensity)
    mx = max(out.values()) if out else 0.0
    if mx > 0:
        for k in out:
            out[k] = round(out[k] / mx, 4)
    return out


def network_entropy(edges: Dict[Tuple[str, str], float]) -> float:
    """Shannon-style entropy over the absolute edge weights (normalized).
    Low entropy = a few edges dominate (concentrated info flow, dangerous).
    High entropy = flow is diffuse (healthy ecosystem)."""
    ws = [abs(v) for v in edges.values() if abs(v) > 1e-6]
    if not ws:
        return 0.0
    s = sum(ws)
    ps = [w / s for w in ws]
    return -sum(p * math.log(p) for p in ps if p > 0)


def ascendancy_metrics(edges: Dict[Tuple[str, str], float]) -> dict:
    """Ulanowicz ascendancy / development capacity / reserve on the directed
    information graph. Edges carry signed lagged-correlation weights;
    throughput uses |W_ij| since flows are non-negative quantities.

    Definitions:
      TST = Σ_ij |W_ij|                 (total system throughput)
      p_ij = |W_ij| / TST              (normalized flow)
      p_i_out = Σ_j p_ij               (out-marginal)
      p_j_in  = Σ_i p_ij               (in-marginal)
      AMI = Σ_ij p_ij log(p_ij / (p_i_out * p_j_in))   (organized flow info)
      A = TST * AMI                    (ascendancy: organized capacity)
      H = -Σ_ij p_ij log(p_ij)         (network Shannon entropy)
      C = TST * H                      (development capacity)
      R = C - A                        (reserve / adaptive overhead)

    Returned `rel_ascendancy` = A/C and `rel_reserve` = R/C lie in [0, 1]
    and sum to ~1. They're the regime-shape diagnostics: rel_A high =
    organized & brittle; rel_R high = diffuse & adaptive.

    Logs and ratios use a small eps to stay numerically safe on empty /
    degenerate graphs. Self-edges are not present in the input by
    construction (information_network skips a==b).
    """
    EPS = 1e-12
    ws = {k: abs(v) for k, v in edges.items() if abs(v) > 0}
    TST = sum(ws.values())
    if TST <= 0:
        return {"TST": 0.0, "AMI": 0.0, "A": 0.0, "H": 0.0, "C": 0.0, "R": 0.0,
                "rel_ascendancy": 0.0, "rel_reserve": 0.0, "n_active_edges": 0}
    # Normalized flows
    p = {k: w / TST for k, w in ws.items()}
    # Marginals over distinct node names
    nodes = set()
    for (a, b) in p.keys():
        nodes.add(a); nodes.add(b)
    p_out = {n: 0.0 for n in nodes}
    p_in = {n: 0.0 for n in nodes}
    for (a, b), v in p.items():
        p_out[a] += v
        p_in[b] += v
    # AMI = Σ p_ij log( p_ij / (p_i_out * p_j_in) )
    AMI = 0.0
    H = 0.0
    for (a, b), v in p.items():
        denom = (p_out[a] * p_in[b]) + EPS
        # Mutual-info term. v=0 already filtered above; log is safe.
        AMI += v * math.log(v / denom)
        H -= v * math.log(v + EPS)
    A = TST * AMI
    C = TST * H
    R = C - A
    rel_A = (A / C) if C > 0 else 0.0
    rel_R = (R / C) if C > 0 else 0.0
    # Clamp negative AMI rounding artifacts (can happen on small numerics)
    if AMI < 0 and AMI > -1e-9:
        AMI = 0.0; A = 0.0; rel_A = 0.0; rel_R = 1.0 if C > 0 else 0.0
    return {
        "TST": round(TST, 4),
        "AMI": round(AMI, 6),
        "A": round(A, 4),
        "H": round(H, 6),
        "C": round(C, 4),
        "R": round(R, 4),
        "rel_ascendancy": round(rel_A, 4),
        "rel_reserve": round(rel_R, 4),
        "n_active_edges": len(p),
    }


def _node_scale(node: str) -> str:
    return NODE_TIME_SCALE.get(node, "minutes")


def multiscale_network(series: Dict[str, List[float]],
                       edges: Dict[Tuple[str, str], float]) -> dict:
    """Break the trophic graph into seconds/minutes/hours/days subnetworks.

    The full graph remains the source of truth. Subnetworks summarize flows
    within each scale; cross-scale edges summarize information sharing between
    adjacent and non-adjacent ecological layers.
    """
    subnets = {}
    for scale in TIME_SCALES:
        nodes = [n for n in series.keys() if _node_scale(n) == scale]
        intra = {(a, b): w for (a, b), w in edges.items()
                 if a in nodes and b in nodes}
        cent = network_centrality(intra, nodes) if nodes else {}
        keystone = max(cent.items(), key=lambda kv: kv[1])[0] if cent else None
        subnets[scale] = {
            "nodes": nodes,
            "edges": [{"from": a, "to": b, "weight": w}
                      for (a, b), w in intra.items() if abs(w) >= 0.05][:120],
            "centrality": cent,
            "keystone": keystone,
            "net_entropy": round(network_entropy(intra), 3),
            "network_metrics": ascendancy_metrics(intra),
        }

    grouped: Dict[Tuple[str, str], List[float]] = {}
    detailed = []
    for (a, b), w in edges.items():
        sa = _node_scale(a)
        sb = _node_scale(b)
        if sa == sb:
            continue
        grouped.setdefault((sa, sb), []).append(abs(w))
        if abs(w) >= 0.08:
            detailed.append({
                "from": a, "to": b, "from_scale": sa, "to_scale": sb,
                "weight": w,
            })
    cross = []
    for (sa, sb), ws in grouped.items():
        cross.append({
            "from_scale": sa,
            "to_scale": sb,
            "weight": round(sum(ws) / len(ws), 4) if ws else 0.0,
            "n_edges": len(ws),
        })
    cross.sort(key=lambda x: abs(x["weight"]), reverse=True)
    detailed.sort(key=lambda x: abs(x["weight"]), reverse=True)
    return {
        "scale_order": list(TIME_SCALES),
        "subnetworks": subnets,
        "cross_scale_edges": cross,
        "cross_scale_edge_detail": detailed[:160],
    }


def _observed_nodes_from_multi(multi: dict) -> List[str]:
    """Nodes whose feeds are present, even if their history is still warming."""
    summary = (multi or {}).get("summary") or {}
    mapping = {
        "btc_spot_price": "btc_spot",
        "eth_spot_price": "eth_spot",
        "eth_perp_price": "eth_perp",
        "binance_funding_rate": "funding_bn",
        "binance_oi_delta_pct_5m": "oi_change",
        "hl_premium": "hl_premium_z",
        "hl_alt_breadth": "hl_alt_breadth",
        "hl_total_oi_usd": "hl_total_oi_change",
        "hl_btc_book_imbalance": "hl_btc_book_imbalance",
        "deribit_perp_basis_bps": "deribit_perp_basis",
        "deribit_term_basis_bps": "deribit_term_basis",
        "deribit_funding_8h": "deribit_funding_8h",
        "deribit_option_skew": "deribit_option_skew",
        "kraken_coinbase_basis_bps": "kraken_coinbase_basis_bps",
        "kraken_spread_bps": "kraken_spread_bps",
        "kraken_bbo_depth": "kraken_bbo_depth",
        "cb_btc_depth_imbalance": "cb_btc_depth_imbalance",
        "cb_btc_spread_bps": "cb_btc_spread_bps",
        "crypto_total_mcap_usd": "crypto_total_mcap_change",
        "btc_dominance": "btc_dominance_change",
        "top10_breadth": "top10_breadth",
        "risk_on_alt_breadth": "risk_on_alt_breadth",
        "btc_fee_fastest": "btc_fee_fastest",
        "btc_fee_half_hour": "btc_fee_half_hour",
        "mempool_vsize": "mempool_congestion_z",
    }
    out = []
    for key, node in mapping.items():
        if summary.get(key) is not None:
            out.append(node)
    return out


def _augment_multiscale(scale_net: dict, observed_nodes: List[str]) -> dict:
    """Add observed-but-not-yet-correlatable nodes to their scale buckets."""
    if not observed_nodes:
        return scale_net
    sub = scale_net.setdefault("subnetworks", {})
    for scale in TIME_SCALES:
        sn = sub.setdefault(scale, {
            "nodes": [], "edges": [], "centrality": {}, "keystone": None,
            "net_entropy": 0.0, "network_metrics": ascendancy_metrics({}),
        })
        nodes = list(sn.get("nodes") or [])
        for node in observed_nodes:
            if _node_scale(node) == scale and node not in nodes:
                nodes.append(node)
                (sn.setdefault("centrality", {}))[node] = 0.0
        sn["nodes"] = nodes
        if sn.get("keystone") is None and nodes:
            sn["keystone"] = nodes[0]
    scale_net["scale_order"] = list(TIME_SCALES)
    return scale_net


# ----------------------------------------- phase classification
def _classify_phase(scalars: Dict[str, float]) -> Tuple[str, str]:
    """Return (phase, rationale) given a dict of scalar drivers:
      vol_z         realized-vol z vs baseline
      vol_decel     vol[t] - vol[t-k], <0 means decelerating
      spread_z      live spread z vs baseline
      stretch_z     price stretch from EMA fair value, z-scored
      depth_recover live depth ratio vs rolling median (>1 = recovered)
      oi_change_z   z-score of OI rate-of-change (negative = unwinding)
      liq_proxy_z   z-score of |OI rate-of-change| (high = forced unwinding)
      disturbance   ||(vol_z, spread_z, stretch_z, liq_proxy_z)||_1 / 4

    Calibration (2026-06-15): the original thresholds were too strict — a
    market with vol_z=1.69, stretch_z=-1.40, and vol_decel=0.00011 (i.e.
    essentially flat) fell through every branch and landed on PRODUCER,
    which is wrong. Adjustments:
      - vol_decel uses |vd| < eps to mean "flat" (counts toward either side)
      - SCAVENGER stretch threshold lowered 1.5 -> 1.2
      - EXHAUSTION accepts flat vol_decel
      - DISTURBED fallback before PRODUCER catches the "elevated but doesn't
        fit a named pattern" case so the size_mult drops below 1.0
    """
    vz = scalars.get("vol_z", 0.0)
    vd = scalars.get("vol_decel", 0.0)
    sz = scalars.get("spread_z", 0.0)
    stz = abs(scalars.get("stretch_z", 0.0))
    dr = scalars.get("depth_recover", 1.0)
    liqz = scalars.get("liq_proxy_z", 0.0)
    dist = scalars.get("disturbance", 0.0)
    EPS = 0.0005  # vol_decel below this magnitude counts as "flat"
    rel_A = scalars.get("rel_ascendancy", 0.0)
    rel_R = scalars.get("rel_reserve", 0.0)

    # PREDATOR: cascade in progress -> high vol, spread widening, OI unwinding,
    # vol NOT decelerating
    if vz > 1.5 and (sz > 1.0 or liqz > 1.5) and vd >= -EPS:
        return ("predator",
                f"cascade — vol_z {vz:+.2f}, spread_z {sz:+.2f}, "
                f"liq_z {liqz:+.2f}, vol still accelerating")
    # Ascendancy override before EXHAUSTION: a graph that has locked into one
    # dominant pathway under stress is BRITTLE — closer to predator than to
    # a graceful exhaustion phase. Catches the case where the level signals
    # (vol_z, stretch) look like exhaustion but the network structure says
    # the system is still organized around one driver.
    if rel_A > 0.55 and dist > 0.7:
        return ("predator",
                f"brittle (graph locked): rel_ascendancy {rel_A:.2f} dominant + "
                f"disturbance {dist:.2f} — one pathway driving the system")
    # EXHAUSTION: vol elevated AND decelerating (or flat) AND stretch present.
    # The "or flat" is critical — vol_decel ~ 0 on small samples is the norm
    # for the cycle-to-cycle vol change; the old strict < 0 missed real
    # exhaustion regimes.
    if vz > 1.0 and vd <= EPS and stz > 0.8:
        return ("exhaustion",
                f"predator slowing — vol_z {vz:+.2f} (decel {vd:+.2f}), "
                f"stretch_z {stz:+.2f}")
    # SCAVENGER: stretched, vol decelerating-or-flat, spread compressing
    if stz > 1.2 and vd <= EPS and sz < 0.8:
        return ("scavenger",
                f"snap-back zone — stretch_z {stz:+.2f}, vol decel {vd:+.2f}, "
                f"spread normalizing")
    # DECOMPOSER: vol normalized but stretch lingering; depth recovering
    if vz < 0.6 and sz < 0.6 and dr > 0.8 and stz > 0.4:
        return ("decomposer",
                f"liquidity restoring — vol_z {vz:+.2f}, spread_z {sz:+.2f}, "
                f"depth_recover {dr:.2f}")
    # CHURN: low everything, near fair value
    if vz < 0.3 and sz < 0.3 and stz < 0.6 and dist < 0.7:
        return ("churn",
                f"normal range — vol_z {vz:+.2f}, spread_z {sz:+.2f}, "
                f"stretch_z {stz:+.2f}")
    # Ascendancy / reserve regime checks (low-rel_A side). Diffuse stress
    # with no clean driver -> exhaustion (size_mult 0.6) rather than a
    # quiet PRODUCER label that doesn't reduce exposure.
    if rel_A < 0.20 and dist > 0.5:
        return ("exhaustion",
                f"diffuse stress: rel_ascendancy {rel_A:.2f} low + "
                f"disturbance {dist:.2f} — no clean edge")
    # NEW: DISTURBED fallback before PRODUCER. Catches the "elevated but
    # doesn't fit a named pattern" case so the engine knows something is
    # off rather than reporting calm-baseline PRODUCER. Mapped to exhaustion
    # phase (which has size_mult 0.6) so the controller automatically
    # reduces exposure during these uncertain regimes.
    if vz > 0.8 or stz > 1.0 or dist > 0.7:
        return ("exhaustion",
                f"disturbed (no clean pattern) — vol_z {vz:+.2f}, "
                f"stretch_z {stz:+.2f}, disturbance {dist:.2f}")
    # PRODUCER (default fallback): quiet liquidity provision
    return ("producer",
            f"baseline — vol_z {vz:+.2f}, spread_z {sz:+.2f}, "
            f"stretch_z {stz:+.2f}")


# Risk ordering for projected phases. Used only to allow fast seconds-layer
# stress to pull the phase forward into a more defensive state.
def _phase_risk_rank(phase: str) -> int:
    return {
        "producer": 0,
        "churn": 1,
        "decomposer": 2,
        "scavenger": 2,
        "exhaustion": 3,
        "predator": 4,
    }.get(phase, 1)


# ----------------------------------------- organism scoring + mycelium
def _score_organisms(phase: str, scalars: Dict[str, float],
                      centrality: Dict[str, float]) -> Dict[str, dict]:
    """Score each organism's fit to the current ecological conditions.
    Each score is a float in [0, 1]; the mycelial allocator softmaxes
    these into a suggested capital fraction.

    The intuition:
      - In PREDATOR phase, the predator organism is in its element
        (a cascade) — but our validated edge is mean-reversion, so we
        WOULDN'T allocate capital to it; instead the IMMUNE organism
        suggests sitting out.
      - In SCAVENGER phase, the scavenger (= MPC reversion bot) is in
        its sweet spot — high allocation.
      - In CHURN, the decomposer market-maker collects the spread.
    """
    vz = scalars.get("vol_z", 0.0)
    sz = scalars.get("spread_z", 0.0)
    stz = abs(scalars.get("stretch_z", 0.0))
    vd = scalars.get("vol_decel", 0.0)
    dist = scalars.get("disturbance", 0.0)

    pred = max(0.0, min(1.0, 0.2 + 0.35 * max(0.0, vz) + 0.25 * (1 if vd >= 0 else 0)))
    scav = max(0.0, min(1.0, 0.2 + 0.4 * stz + 0.25 * max(0.0, -vd) + 0.15 * (1 if sz < 0 else 0)))
    deco = max(0.0, min(1.0, 0.4 + 0.4 * max(0.0, -sz) + 0.2 * max(0.0, -vz)))
    prod = max(0.0, min(1.0, 0.5 + 0.4 * max(0.0, -vz) + 0.1 * max(0.0, -sz)))
    immu = max(0.0, min(1.0, 0.1 + 0.6 * max(0.0, dist - 1.0) + 0.3 * max(0.0, vz - 1.5)))
    myce = max(0.0, min(1.0, 0.3 + 0.5 * (1.0 - dist / 4.0)))

    # phase prior: bump the organism whose niche matches the phase
    phase_bump = {
        "predator":   {"predator": 0.3, "immune": 0.35},
        "exhaustion": {"immune": 0.25, "mycelium": 0.15},
        "scavenger":  {"scavenger": 0.4, "mycelium": 0.15},
        "decomposer": {"decomposer": 0.3, "producer": 0.15},
        "producer":   {"producer": 0.25, "decomposer": 0.2},
        "churn":      {"decomposer": 0.2, "producer": 0.15},
    }.get(phase, {})
    scores = {
        "predator":   round(min(1.0, pred + phase_bump.get("predator", 0)), 3),
        "scavenger":  round(min(1.0, scav + phase_bump.get("scavenger", 0)), 3),
        "decomposer": round(min(1.0, deco + phase_bump.get("decomposer", 0)), 3),
        "mycelium":   round(min(1.0, myce + phase_bump.get("mycelium", 0)), 3),
        "immune":     round(min(1.0, immu + phase_bump.get("immune", 0)), 3),
        "producer":   round(min(1.0, prod + phase_bump.get("producer", 0)), 3),
    }
    # softmax-like allocation suggestion
    exps = {k: math.exp(v * 3) for k, v in scores.items()}
    total = sum(exps.values()) or 1.0
    alloc = {k: round(v / total, 3) for k, v in exps.items()}
    return {"scores": scores, "allocation": alloc}


# ----------------------------------------- master classifier
def classify(candles: List[dict], market: dict, multi: dict,
             params: dict) -> dict:
    """Master ecology classification for a single cycle. Returns a dict the
    engine attaches to the proposal (always) and the UI renders.

    Keys returned:
      phase, rationale, size_mult
      nodes:        list of active node names
      edges:        list of {from, to, weight} (directed, lagged-corr)
      centrality:   {node: 0..1}
      keystone:     name of the most-central node
      net_entropy:  Shannon entropy of normalized edge weights
      drivers:      {vol_z, vol_decel, spread_z, stretch_z, depth_recover,
                      oi_change_z, liq_proxy_z, disturbance}
      organisms:    {scores: {...}, allocation: {...}}
      multi_summary: pass-through of multiasset.summary
    """
    win = max(20, int(params.get("ecology_win", 64)))
    lag = max(1, int(params.get("ecology_te_lag", 1)))
    vol_win = max(4, int(params.get("ecology_vol_win", 10)))
    disturb_z = float(params.get("ecology_disturbance_z", 1.5))

    series = _extract_node_series(candles, market, multi, win)
    observed_nodes = _observed_nodes_from_multi(multi)
    if len(series) < 2 or "price" not in series or len(series["price"]) < 16:
        scale_net = _augment_multiscale(multiscale_network({}, {}), observed_nodes)
        return {
            "phase": "producer", "rationale": "insufficient-data",
            "size_mult": 1.0, "nodes": sorted(set(list(series.keys()) + observed_nodes)),
            "edges": [], "centrality": {}, "keystone": None,
            "net_entropy": 0.0, "drivers": {}, "organisms": _score_organisms(
                "producer", {"vol_z": 0, "vol_decel": 0, "spread_z": 0,
                             "stretch_z": 0, "disturbance": 0}, {}),
            "multiscale": scale_net,
            "multi_summary": (multi or {}).get("summary"),
        }

    edges = information_network(series, lag=lag)
    centrality = network_centrality(edges, list(series.keys()))
    keystone = max(centrality.items(), key=lambda kv: kv[1])[0] if centrality else None
    netH = network_entropy(edges)
    netmetrics = ascendancy_metrics(edges)
    scale_net = _augment_multiscale(multiscale_network(series, edges), observed_nodes)

    # --- scalar drivers used by the phase classifier ---
    closes = [c["close"] for c in candles]
    rets = series["price"]
    vol_now = _std(rets[-vol_win:]) if len(rets) >= vol_win else _std(rets)
    vol_history = []
    for i in range(vol_win, len(rets)):
        vol_history.append(_std(rets[i - vol_win:i]))
    vol_z = _zscore(vol_now, vol_history) if len(vol_history) >= 6 else 0.0
    # acceleration of vol: recent vs slightly older
    if len(vol_history) >= 4:
        vol_decel = vol_now - _mean(vol_history[-4:-1])
    else:
        vol_decel = 0.0

    spread_z = 0.0
    if "spread" in series and len(series["spread"]) >= 8:
        sp = series["spread"]
        spread_z = _zscore(sp[-1], sp[:-1])

    # depth recovery: live depth_total vs recent average (best-effort)
    depth_recover = 1.0
    live_depth = (market or {}).get("orderbook", {}).get("depth_total") or 0
    eth_depth_series = multiasset.HISTORY.series("eth_perp", n=win)  # placeholder
    if live_depth and eth_depth_series:
        avg_depth = _mean([v for _, v in eth_depth_series[-20:]])
        depth_recover = (live_depth / avg_depth) if avg_depth else 1.0

    # stretch z: price vs EMA fair-value, scaled by recent vol
    fv_win = max(8, win // 3)
    fv_series = signals.ema_series(closes, fv_win)
    fv = fv_series[-1] if fv_series else closes[-1]
    stretch_raw = (closes[-1] - fv) / fv if fv else 0.0
    # normalize by sqrt(N)*std(rets)
    stretch_norm = stretch_raw / (vol_now or 1e-6)
    stretch_z = max(-5.0, min(5.0, stretch_norm))

    oi_change_z = 0.0
    if "oi_change" in series and len(series["oi_change"]) >= 8:
        oc = series["oi_change"]
        oi_change_z = _zscore(oc[-1], oc[:-1])

    liq_proxy_z = 0.0
    if "liq_proxy" in series and len(series["liq_proxy"]) >= 6:
        lq = series["liq_proxy"]
        liq_proxy_z = _zscore(lq[-1], lq[:-1])

    disturbance = (abs(vol_z) + abs(spread_z) + abs(stretch_z) + abs(liq_proxy_z)) / 4.0

    drivers = {
        "vol_z": round(vol_z, 3),
        "vol_decel": round(vol_decel, 5),
        "spread_z": round(spread_z, 3),
        "stretch_z": round(stretch_z, 3),
        "depth_recover": round(depth_recover, 3),
        "oi_change_z": round(oi_change_z, 3),
        "liq_proxy_z": round(liq_proxy_z, 3),
        "disturbance": round(disturbance, 3),
    }
    seconds_metrics = (((scale_net.get("subnetworks") or {}).get("seconds") or {})
                       .get("network_metrics") or {})
    fast_vol_z = 0.0
    if "live_mid" in series and len(series["live_mid"]) >= 8:
        fm = [abs(x) for x in series["live_mid"]]
        fast_vol_z = _zscore(fm[-1], fm[:-1])
    fast_spread_z = 0.0
    if "live_spread_bps" in series and len(series["live_spread_bps"]) >= 8:
        sp = series["live_spread_bps"]
        fast_spread_z = _zscore(sp[-1], sp[:-1])
    fast_imbalance_z = 0.0
    if "live_imb_depth" in series and len(series["live_imb_depth"]) >= 8:
        im = series["live_imb_depth"]
        fast_imbalance_z = _zscore(im[-1], im[:-1])
    fast_depth_drop_z = 0.0
    if "live_depth_total" in series and len(series["live_depth_total"]) >= 8:
        dp = series["live_depth_total"]
        fast_depth_drop_z = max(0.0, -_zscore(dp[-1], dp[:-1]))
    seconds_disturbance = (
        abs(fast_vol_z) + max(0.0, fast_spread_z)
        + abs(fast_imbalance_z) + fast_depth_drop_z
    ) / 4.0
    seconds_rel_A = float(seconds_metrics.get("rel_ascendancy") or 0.0)
    disturbance_projected = max(
        disturbance,
        0.65 * disturbance + 0.55 * seconds_disturbance
        + max(0.0, seconds_rel_A - 0.55),
    )
    drivers["seconds_rel_ascendancy"] = seconds_metrics.get("rel_ascendancy", 0.0)
    drivers["seconds_rel_reserve"] = seconds_metrics.get("rel_reserve", 0.0)
    drivers["seconds_edges"] = seconds_metrics.get("n_active_edges", 0)
    drivers["seconds_vol_z"] = round(fast_vol_z, 3)
    drivers["seconds_spread_z"] = round(fast_spread_z, 3)
    drivers["seconds_imbalance_z"] = round(fast_imbalance_z, 3)
    drivers["seconds_depth_drop_z"] = round(fast_depth_drop_z, 3)
    drivers["seconds_disturbance"] = round(seconds_disturbance, 3)
    drivers["disturbance_projected"] = round(disturbance_projected, 3)
    scalars = dict(drivers)

    # Pass ascendancy/reserve into the classifier so it can flag brittle
    # one-pathway regimes and diffuse no-edge regimes alongside the vol/
    # stretch/spread thresholds.
    scalars["rel_ascendancy"] = netmetrics.get("rel_ascendancy", 0.0)
    scalars["rel_reserve"] = netmetrics.get("rel_reserve", 0.0)
    phase_now, rationale_now = _classify_phase(scalars)
    projected_scalars = dict(scalars)
    projected_scalars["disturbance"] = disturbance_projected
    phase_projected, rationale_projected = _classify_phase(projected_scalars)
    if (seconds_disturbance >= 1.2 and disturbance_projected >= 0.9
            and phase_projected in ("producer", "churn", "scavenger", "decomposer")):
        phase_projected = (
            "predator"
            if seconds_rel_A > 0.55 or fast_spread_z > 1.5 or fast_depth_drop_z > 1.5
            else "exhaustion"
        )
        rationale_projected = (
            f"seconds-layer shock: sec_dist {seconds_disturbance:.2f}, "
            f"spread_z {fast_spread_z:+.2f}, depth_drop_z {fast_depth_drop_z:+.2f}, "
            f"sec_A {seconds_rel_A:.2f}"
        )
    use_projected = (
        _phase_risk_rank(phase_projected) > _phase_risk_rank(phase_now)
        and (seconds_disturbance >= 0.8 or disturbance_projected >= disturbance + 0.25)
    )
    phase = phase_projected if use_projected else phase_now
    rationale = (f"projected: {rationale_projected} | now: {rationale_now}"
                 if use_projected else rationale_now)
    scalars["phase_now"] = phase_now
    scalars["phase_projected"] = phase_projected
    scalars["projected_phase_applied"] = use_projected
    drivers["phase_now"] = phase_now
    drivers["phase_projected"] = phase_projected
    drivers["projected_phase_applied"] = use_projected
    mult_map = {
        "producer":   float(params.get("ecology_producer_mult", 1.0)),
        "predator":   float(params.get("ecology_predator_mult", 0.2)),
        "exhaustion": float(params.get("ecology_exhaustion_mult", 0.6)),
        "scavenger":  float(params.get("ecology_scavenger_mult", 1.5)),
        "decomposer": float(params.get("ecology_decomposer_mult", 0.5)),
        "churn":      float(params.get("ecology_churn_mult", 1.0)),
    }
    size_mult = max(0.0, min(2.5, mult_map.get(phase, 1.0)))

    # Serialize edges as a flat list for the UI (filter out weak ones)
    edge_list = []
    for (a, b), w in edges.items():
        if abs(w) >= 0.05:
            edge_list.append({"from": a, "to": b, "weight": w})

    organisms = _score_organisms(phase, scalars, centrality)

    return {
        "phase": phase,
        "rationale": rationale,
        "size_mult": round(size_mult, 3),
        "nodes": sorted(set(list(series.keys()) + observed_nodes)),
        "edges": edge_list,
        "centrality": centrality,
        "keystone": keystone,
        "net_entropy": round(netH, 3),
        "network_metrics": netmetrics,
        "drivers": drivers,
        "organisms": organisms,
        "multiscale": scale_net,
        "multi_summary": (multi or {}).get("summary"),
    }


# ----------------------------------- vectorized phase series for backtest
def phase_size_mult_series(candles: List[dict], params: dict) -> List[float]:
    """Per-bar phase-multiplier series for the backtester. Uses ONLY
    on-platform data (rich candles' price/spread/volume/oi); the multi-asset
    feeds aren't available historically without a long pull. Each bar i is
    classified from candles[:i+1] only (no look-ahead) and the corresponding
    size_mult is returned.

    Returned list is the same length as `candles`; bars before the warmup
    window have multiplier 1.0 (passes through MPC unchanged).
    """
    n = len(candles)
    out = [1.0] * n
    if n < 40:
        return out

    win = max(20, int(params.get("ecology_win", 64)))
    vol_win = max(4, int(params.get("ecology_vol_win", 10)))
    mult_map = {
        "producer":   float(params.get("ecology_producer_mult", 1.0)),
        "predator":   float(params.get("ecology_predator_mult", 0.2)),
        "exhaustion": float(params.get("ecology_exhaustion_mult", 0.6)),
        "scavenger":  float(params.get("ecology_scavenger_mult", 1.5)),
        "decomposer": float(params.get("ecology_decomposer_mult", 0.5)),
        "churn":      float(params.get("ecology_churn_mult", 1.0)),
    }
    closes = [c["close"] for c in candles]
    rets_all = _returns(closes)
    spreads_all = [float(c.get("spread") or 0.0) for c in candles]
    has_spread = any(spreads_all)
    has_oi = any(c.get("oi") for c in candles)
    ois_all = [float(c.get("oi") or 0.0) for c in candles]
    fv_win = max(8, win // 3)
    fv_series = signals.ema_series(closes, fv_win)

    start = max(win, vol_win * 2 + 1, fv_win + 1)
    for i in range(start, n):
        rets = rets_all[max(0, i - win):i]
        if len(rets) < vol_win * 2:
            continue
        vol_now = _std(rets[-vol_win:])
        vol_history = [_std(rets[j - vol_win:j])
                       for j in range(vol_win, len(rets))]
        vol_z = _zscore(vol_now, vol_history) if len(vol_history) >= 6 else 0.0
        vol_decel = vol_now - _mean(vol_history[-4:-1]) if len(vol_history) >= 4 else 0.0

        spread_z = 0.0
        if has_spread:
            sp = spreads_all[max(0, i - win):i + 1]
            if len(sp) >= 8 and any(sp[:-1]):
                spread_z = _zscore(sp[-1], sp[:-1])

        fv = fv_series[i] if fv_series[i] else closes[i]
        stretch_raw = (closes[i] - fv) / fv if fv else 0.0
        stretch_z = max(-5.0, min(5.0, stretch_raw / (vol_now or 1e-6)))

        oi_change_z = 0.0
        liq_proxy_z = 0.0
        if has_oi:
            oc = _diffs(ois_all[max(0, i - win):i + 1])
            if len(oc) >= 8:
                oi_change_z = _zscore(oc[-1], oc[:-1])
                liq_proxy_z = _zscore(abs(oc[-1]), [abs(v) for v in oc[:-1]])

        disturbance = (abs(vol_z) + abs(spread_z) + abs(stretch_z) + abs(liq_proxy_z)) / 4.0
        scalars = {
            "vol_z": vol_z, "vol_decel": vol_decel,
            "spread_z": spread_z, "stretch_z": stretch_z,
            "depth_recover": 1.0,
            "oi_change_z": oi_change_z, "liq_proxy_z": liq_proxy_z,
            "disturbance": disturbance,
        }
        phase, _ = _classify_phase(scalars)
        out[i] = max(0.0, min(2.5, mult_map.get(phase, 1.0)))
    return out


def phase_series_with_states(candles: List[dict], params: dict
                              ) -> Tuple[List[float], List[str]]:
    """Same as phase_size_mult_series but also returns the per-bar phase
    string array (so the backtester can report state counts)."""
    n = len(candles)
    mults = [1.0] * n
    states = ["producer"] * n
    if n < 40:
        return mults, states
    # The full function above is duplicated only to expose states; keep
    # logic in sync if you tweak the classifier thresholds.
    win = max(20, int(params.get("ecology_win", 64)))
    vol_win = max(4, int(params.get("ecology_vol_win", 10)))
    mult_map = {
        "producer":   float(params.get("ecology_producer_mult", 1.0)),
        "predator":   float(params.get("ecology_predator_mult", 0.2)),
        "exhaustion": float(params.get("ecology_exhaustion_mult", 0.6)),
        "scavenger":  float(params.get("ecology_scavenger_mult", 1.5)),
        "decomposer": float(params.get("ecology_decomposer_mult", 0.5)),
        "churn":      float(params.get("ecology_churn_mult", 1.0)),
    }
    closes = [c["close"] for c in candles]
    rets_all = _returns(closes)
    spreads_all = [float(c.get("spread") or 0.0) for c in candles]
    has_spread = any(spreads_all)
    has_oi = any(c.get("oi") for c in candles)
    ois_all = [float(c.get("oi") or 0.0) for c in candles]
    fv_win = max(8, win // 3)
    fv_series = signals.ema_series(closes, fv_win)
    start = max(win, vol_win * 2 + 1, fv_win + 1)
    for i in range(start, n):
        rets = rets_all[max(0, i - win):i]
        if len(rets) < vol_win * 2:
            continue
        vol_now = _std(rets[-vol_win:])
        vol_history = [_std(rets[j - vol_win:j])
                       for j in range(vol_win, len(rets))]
        vol_z = _zscore(vol_now, vol_history) if len(vol_history) >= 6 else 0.0
        vol_decel = vol_now - _mean(vol_history[-4:-1]) if len(vol_history) >= 4 else 0.0
        spread_z = 0.0
        if has_spread:
            sp = spreads_all[max(0, i - win):i + 1]
            if len(sp) >= 8 and any(sp[:-1]):
                spread_z = _zscore(sp[-1], sp[:-1])
        fv = fv_series[i] if fv_series[i] else closes[i]
        stretch_raw = (closes[i] - fv) / fv if fv else 0.0
        stretch_z = max(-5.0, min(5.0, stretch_raw / (vol_now or 1e-6)))
        oi_change_z = 0.0; liq_proxy_z = 0.0
        if has_oi:
            oc = _diffs(ois_all[max(0, i - win):i + 1])
            if len(oc) >= 8:
                oi_change_z = _zscore(oc[-1], oc[:-1])
                liq_proxy_z = _zscore(abs(oc[-1]), [abs(v) for v in oc[:-1]])
        disturbance = (abs(vol_z) + abs(spread_z) + abs(stretch_z) + abs(liq_proxy_z)) / 4.0
        phase, _ = _classify_phase({
            "vol_z": vol_z, "vol_decel": vol_decel,
            "spread_z": spread_z, "stretch_z": stretch_z,
            "depth_recover": 1.0,
            "oi_change_z": oi_change_z, "liq_proxy_z": liq_proxy_z,
            "disturbance": disturbance,
        })
        states[i] = phase
        mults[i] = max(0.0, min(2.5, mult_map.get(phase, 1.0)))
    return mults, states
