"""Summarize live event/orderbook logs for execution and churn postmortems.

Run from the repo root:
    python -m backend.analyze_live_events --since 2026-06-16T04:00:00Z
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median

from .config import STATE_DIR


def _parse_ts(raw: str | None) -> datetime | None:
    if not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None


def _load_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line in path.read_text("utf-8", errors="ignore").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def _pct(xs: list[float], q: float) -> float:
    if not xs:
        return 0.0
    ys = sorted(xs)
    idx = min(len(ys) - 1, max(0, int(round((len(ys) - 1) * q))))
    return ys[idx]


def _event_time(e: dict) -> datetime | None:
    return _parse_ts(e.get("ts") or e.get("time"))


def _event_name(e: dict) -> str:
    return str(e.get("event") or e.get("type") or e.get("message") or "")


def _f(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def summarize_events(events: list[dict]) -> dict:
    kinds = Counter(str(e.get("kind") or e.get("channel") or "unknown") for e in events)
    names = Counter(_event_name(e) for e in events if _event_name(e))
    orders = [e for e in events if (e.get("kind") == "order" or "fill_count" in e)]
    submitted = [e for e in orders if _event_name(e) == "submitted"]
    filled = [e for e in submitted if _f(e.get("fill_count")) > 0.0]
    zero = [e for e in submitted if _f(e.get("fill_count")) <= 0.0]
    maker = [e for e in filled if bool(e.get("maker"))]
    taker = [e for e in filled if not bool(e.get("maker"))]

    by_verb: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0, 0.0, 0.0, 0.0])
    # orders, contracts, notional, taker orders, taker contracts
    for e in filled:
        verb = str(e.get("verb") or "unknown")
        count = _f(e.get("fill_count") or e.get("count"))
        price = _f(e.get("avg_fill_price") or e.get("price"))
        rec = by_verb[verb]
        rec[0] += 1
        rec[1] += count
        rec[2] += count * price
        if not bool(e.get("maker")):
            rec[3] += 1
            rec[4] += count

    realized = _reconstruct_realized(filled)
    return {
        "events": len(events),
        "kinds": kinds.most_common(12),
        "event_names": names.most_common(20),
        "orders_submitted": len(submitted),
        "orders_filled": len(filled),
        "orders_zero_fill": len(zero),
        "maker_fills": len(maker),
        "taker_fills": len(taker),
        "filled_contracts": round(sum(_f(e.get("fill_count") or e.get("count")) for e in filled), 2),
        "maker_contracts": round(sum(_f(e.get("fill_count") or e.get("count")) for e in maker), 2),
        "taker_contracts": round(sum(_f(e.get("fill_count") or e.get("count")) for e in taker), 2),
        "filled_notional": round(sum(_f(e.get("fill_count") or e.get("count")) * _f(e.get("avg_fill_price") or e.get("price")) for e in filled), 2),
        "by_verb": {k: [round(x, 3) for x in v] for k, v in sorted(by_verb.items())},
        **realized,
    }


def _reconstruct_realized(filled: list[dict]) -> dict:
    pos = 0.0
    avg = 0.0
    realized = 0.0
    closes: list[float] = []
    flips = 0
    max_abs_pos = 0.0
    for e in sorted(filled, key=lambda x: _event_time(x) or datetime.min.replace(tzinfo=timezone.utc)):
        qty = _f(e.get("fill_count") or e.get("count"))
        px = _f(e.get("avg_fill_price") or e.get("price"))
        if qty <= 0 or px <= 0:
            continue
        signed = qty if e.get("side") == "bid" else -qty
        old_pos = pos
        if pos == 0 or (pos > 0 and signed > 0) or (pos < 0 and signed < 0):
            new_abs = abs(pos) + abs(signed)
            avg = ((avg * abs(pos)) + (px * abs(signed))) / new_abs if new_abs else 0.0
            pos += signed
        else:
            close_qty = min(abs(pos), abs(signed))
            pnl = (px - avg) * close_qty if pos > 0 else (avg - px) * close_qty
            realized += pnl
            closes.append(pnl)
            residual = abs(signed) - close_qty
            pos += signed
            if residual > 1e-9 and old_pos * pos < 0:
                flips += 1
                avg = px
            elif abs(pos) < 1e-9:
                pos = 0.0
                avg = 0.0
        max_abs_pos = max(max_abs_pos, abs(pos))
    wins = [x for x in closes if x > 0]
    losses = [x for x in closes if x < 0]
    return {
        "approx_realized_pnl": round(realized, 4),
        "end_position": round(pos, 3),
        "end_avg_price": round(avg, 4),
        "max_abs_position": round(max_abs_pos, 3),
        "close_events": len(closes),
        "close_wins": len(wins),
        "close_losses": len(losses),
        "gross_win": round(sum(wins), 4),
        "gross_loss": round(sum(losses), 4),
        "avg_close_pnl": round(mean(closes), 5) if closes else 0.0,
        "flips": flips,
    }


def summarize_orderbook(rows: list[dict]) -> dict:
    spreads = [_f(r.get("spread_bps")) for r in rows if r.get("spread_bps") is not None]
    moves = [_f(r.get("recent_move_bps")) for r in rows if r.get("recent_move_bps") is not None]
    return {
        "orderbook_rows": len(rows),
        "spread_bps_mean": round(mean(spreads), 3) if spreads else 0.0,
        "spread_bps_p50": round(median(spreads), 3) if spreads else 0.0,
        "spread_bps_p95": round(_pct(spreads, 0.95), 3),
        "spread_bps_max": round(max(spreads), 3) if spreads else 0.0,
        "recent_move_bps_p95": round(_pct(moves, 0.95), 3),
        "recent_move_bps_p99": round(_pct(moves, 0.99), 3),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default=None, help="UTC ISO timestamp, e.g. 2026-06-16T04:00:00Z")
    args = ap.parse_args()
    since = _parse_ts(args.since) if args.since else None

    events = _load_jsonl(STATE_DIR / "events.log")
    orderbook = _load_jsonl(STATE_DIR / "orderbook_log.jsonl")
    if since:
        events = [e for e in events if (_event_time(e) and _event_time(e) >= since)]
        orderbook = [r for r in orderbook if (_event_time(r) and _event_time(r) >= since)]

    summary = {
        "since": since.isoformat() if since else None,
        "events": summarize_events(events),
        "orderbook": summarize_orderbook(orderbook),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
