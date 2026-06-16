"""Periodic visual sanity check: render the recent candle chart and ask Opus
"are we doing something stupid?".

The validated mean-reversion strategy is essentially blind to trend context
beyond the few features in its alpha. Once every `visual_review_interval_seconds`
(plus once at engine startup) we generate a PNG of the last ~100 candles, mark
the current position direction and recent trades, attach the latest ecology
phase + CSD risk, and send the whole thing to Opus with vision.

Opus replies with structured JSON: {trend, concern, note}. If concern == 'STOP'
and the next cycle's proposal is a NEW ENTRY (ENTER_LONG/ENTER_SHORT or REVERSE_*),
the engine blocks it and the bot sits out until the next visual review clears.
Existing positions are NEVER force-flattened by this layer — it's an entry
gate, not a kill switch.

Cost: ~1500-3500 input tokens (image + context) + ~200 output tokens per call.
At a 10-min cadence that's ~144 calls/day. At Opus 4.8 pricing the daily cost
is on the order of a few dollars; this layer is opt-in via
settings.visual_review_enabled.

Public API:
  maybe_run_review(settings, store, market, account, last_decisions) -> dict|None
    Checks the timing, generates+sends if due, mutates STORE.visual_review.
    Returns the review dict on a fresh call, None when skipped or disabled.

  block_entry_if_stop(store, proposal) -> bool
    Decision-time check. If the latest review's concern is STOP and the
    proposal is a new entry, mutates the proposal to HOLD and returns True.
"""
from __future__ import annotations

import base64
import io
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")  # headless; never touch a window backend in the loop
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.dates as mdates  # noqa: E402

from .config import STATE_DIR, anthropic_api_key  # noqa: E402


VISUAL_REVIEW_IMAGE_PATH = STATE_DIR / "visual_review_latest.png"


SYSTEM_PROMPT = """You are the visual brain for an autonomous BTC perpetual \
trading bot. The bot runs a statistically validated mean-reversion edge on \
Kalshi's KXBTCPERP contract, but that edge is blind to multi-bar trend \
context — you provide that context. Your read does TWO jobs: (1) your `trend` \
call is fed directly into the bot's DIRECTIONAL signal (it tilts the target \
position with the trend), and (2) your `concern` level flags how strongly the \
trend is in force AND whether the bot is fighting an obvious move.

Judge the genuine multi-bar direction — not every wiggle. A small countertrend \
position is normal and expected for a reversion bot, so do NOT call a trend on \
noise. Only report up/down when the price action shows a real, sustained \
directional move; otherwise sideways or choppy.

You receive a chart PNG with TWO stacked panels and a small JSON context \
(current position, recent decisions, ecology phase, CSD risk):
  - TOP panel = 3-minute candles (~5 hours). This is the TREND ANCHOR. Base \
    your `trend` read on THIS panel — it is the multi-bar regime the edge is \
    blind to. Current position and recent trades are marked here.
  - 1-minute candles panel = NEAR-TERM CONFIRMATION only. Use it to judge \
    whether the 3m trend is actually in force right now, NOT to pick a different \
    direction. If the 1m contradicts the 3m (e.g. 3m sloping up but 1m has \
    rolled over), that means LOWER conviction — lean toward sideways/choppy — \
    it does NOT mean flip the call.
  - P/L panel = account EQUITY over time (overall trajectory of the account). \
    Falling equity while you keep signalling the same direction is a warning.
  - Open-trade UNREALIZED P/L panel = the current position's profit over time, \
    with a zero line. READ ITS SLOPE: if it rose and is now rolling over / \
    fading while price stalls, the edge is being consumed — lower conviction, \
    lean toward sideways/choppy (CAUTION) so the bot harvests rather than holds. \
    A steadily climbing unrealized line in the trend's direction supports higher \
    conviction.
These P/L panels are decision inputs, not just price — weigh the profit \
trajectory alongside the candles.

Respond with ONLY a JSON object, no prose, no markdown fences:

{
  "trend": "up" | "down" | "sideways" | "choppy",
  "concern": "OK" | "CAUTION" | "STOP",
  "note": "short human-readable observation (under 200 chars)"
}

Concern levels — this ALSO sets how hard `trend` feeds the directional signal \
(higher concern = stronger conviction in the trend):
  OK      = no strong/obvious trend; the reversion edge is fine. Low \
            directional conviction.
  CAUTION = a directional trend is present/forming but the 1m is mixed or the \
            move isn't yet decisive. Moderate conviction.
  STOP    = an obvious, sustained one-way move confirmed on BOTH panels — the \
            bot should be trading WITH it, not fading it. High conviction.

Be honest. If there is no real trend, say sideways/OK — don't invent one. \
Don't call a strong trend on a single-panel move the other panel contradicts."""


# ----------------------------------------------------------- chart rendering
def _draw_candles(ax, candles: List[dict], panel_label: Optional[str] = None):
    """Draw OHLC candles onto `ax` (dark theme), with the standard grid/axis
    styling. Returns (opens, highs, lows, closes, ts) so the caller can add
    annotations. Pure drawing — no figure creation or I/O."""
    cs = candles[-100:]
    ts = [datetime.fromtimestamp(c.get("ts", 0), tz=timezone.utc) for c in cs]
    opens = [c["open"] for c in cs]
    highs = [c["high"] for c in cs]
    lows = [c["low"] for c in cs]
    closes = [c["close"] for c in cs]

    ax.set_facecolor("#0c0f17")
    if len(ts) >= 2:
        bar_seconds = max(30.0, (ts[-1] - ts[0]).total_seconds() / max(1, len(ts) - 1))
    else:
        bar_seconds = 60.0
    width_days = (bar_seconds * 0.7) / 86400.0
    for o, h, l, c, t in zip(opens, highs, lows, closes, ts):
        color = "#22c55e" if c >= o else "#ef4444"
        ax.vlines(t, l, h, color=color, linewidth=0.8, zorder=1)
        ax.add_patch(plt.Rectangle(
            (mdates.date2num(t) - width_days / 2, min(o, c)),
            width_days, max(abs(c - o), 1e-9),
            facecolor=color, edgecolor=color, linewidth=0.5, zorder=2))

    ax.grid(True, color="#1f2937", linestyle="-", linewidth=0.5, alpha=0.6)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#1f2937")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%m-%d"))
    if panel_label:
        ax.text(0.008, 0.96, panel_label, transform=ax.transAxes,
                ha="left", va="top", color="#9ca3af", fontsize=9,
                fontweight="bold", zorder=6)
    return opens, highs, lows, closes, ts


def _draw_line(ax, series, panel_label=None, zero_line=False, fill=True):
    """Dark-theme line panel. `series` = list of (label, color, [(datetime,
    value), ...]). With fill=True the area is shaded to zero (good for a signed
    series around 0); with fill=False the axis autoscales to the data range
    (good for a large-magnitude series like equity)."""
    ax.set_facecolor("#0c0f17")
    any_pts = False
    n_series = 0
    for label, color, pts in series:
        if not pts:
            continue
        any_pts = True
        n_series += 1
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, color=color, linewidth=1.4, zorder=3, label=label)
        if fill:
            ax.fill_between(xs, ys, 0, color=color, alpha=0.07, zorder=1)
    if zero_line:
        ax.axhline(0, color="#475569", linewidth=0.7, linestyle="--", zorder=2)
    ax.grid(True, color="#1f2937", linestyle="-", linewidth=0.5, alpha=0.6)
    ax.tick_params(colors="#9ca3af", labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#1f2937")
    ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=6))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M\n%m-%d"))
    if panel_label:
        ax.text(0.008, 0.96, panel_label, transform=ax.transAxes, ha="left",
                va="top", color="#9ca3af", fontsize=9, fontweight="bold", zorder=6)
    if n_series > 1:
        ax.legend(loc="upper left", fontsize=7, framealpha=0.0,
                  labelcolor="#cbd5e1", bbox_to_anchor=(0.12, 1.0))
    if not any_pts:
        ax.text(0.5, 0.5, "no P/L history yet", ha="center", va="center",
                transform=ax.transAxes, color="#64748b", fontsize=9)


def _parse_pnl(pnl_history):
    """pnl_history is newest-first {ts, equity, realized, unrealized}. Returns
    chronological (equity_pts, unrealized_pts, realized_pts)."""
    eq, un, re_ = [], [], []
    for row in reversed(pnl_history or []):
        try:
            t = datetime.fromisoformat(str(row.get("ts")).replace("Z", "+00:00"))
        except Exception:
            continue
        if row.get("equity") is not None:
            eq.append((t, float(row["equity"])))
        if row.get("unrealized") is not None:
            un.append((t, float(row["unrealized"])))
        if row.get("realized") is not None:
            re_.append((t, float(row["realized"])))
    return eq, un, re_


def render_chart(candles: List[dict], position_contracts: float,
                  recent_decisions: List[dict],
                  ticker: str = "KXBTCPERP",
                  csd_risk: Optional[float] = None,
                  ecology_phase: Optional[str] = None,
                  interval: str = "3m",
                  candles_1m: Optional[List[dict]] = None,
                  pnl_history: Optional[List[dict]] = None) -> bytes:
    """Render the review chart as a multi-panel PNG. Pure function — no I/O.

    Panels, top to bottom (each included when its data is present):
      1. `interval` candles — TREND ANCHOR (position + recent decisions marked)
      2. 1-minute candles — NEAR-TERM CONFIRMATION (when candles_1m given)
      3. P/L — account equity + open-trade unrealized over time (when pnl_history given)
      4. Captured (realized) P/L over time (when pnl_history given)
    The P/L panels give the model the profit trajectory to base harvest/decay
    judgements on, not just price."""
    if not candles:
        # Render a tiny stub so the loop still produces an image; the model is
        # told via context that data is unavailable.
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "no candles", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()

    has_1m = bool(candles_1m)
    eq_pts, un_pts, re_pts = _parse_pnl(pnl_history)
    has_pnl = bool(eq_pts or un_pts or re_pts)

    # Assemble the panel stack: anchor (always) + near-term + P/L + captured.
    rows, ratios = ["anchor"], [3]
    if has_1m:
        rows.append("near"); ratios.append(2)
    if has_pnl:
        rows += ["pnl", "captured"]; ratios += [2, 2]
    n = len(rows)
    fig, axes = plt.subplots(
        n, 1, figsize=(10, 2.0 + 1.7 * n), facecolor="#0c0f17",
        gridspec_kw={"height_ratios": ratios})
    if n == 1:
        axes = [axes]
    axmap = dict(zip(rows, axes))
    ax = axmap["anchor"]
    ax1 = axmap.get("near")
    multi = n > 1

    # --- Primary (trend-anchor) panel: interval candles + annotations ---
    opens, highs, lows, closes, ts = _draw_candles(
        ax, candles,
        panel_label=(f"{interval} · trend anchor" if multi else None))

    # Recent decisions: arrows near the corresponding ts.
    for d in (recent_decisions or [])[-20:]:
        try:
            d_ts = d.get("ts")
            if not d_ts:
                continue
            t = datetime.fromisoformat(str(d_ts).replace("Z", "+00:00"))
            verb = d.get("verb") or (d.get("decision") or {}).get("verb")
            if verb in ("LONG", "ADD") and "submitted" in str(d):
                ax.scatter(t, lows[-1] * 0.999 if lows else 0,
                           marker="^", color="#7dd3fc", s=44, zorder=5,
                           edgecolors="black", linewidths=0.5)
            elif verb in ("SHORT", "REDUCE", "CLOSE"):
                ax.scatter(t, highs[-1] * 1.001 if highs else 0,
                           marker="v", color="#facc15", s=44, zorder=5,
                           edgecolors="black", linewidths=0.5)
        except (ValueError, TypeError, KeyError):
            continue

    # Header banner
    pos_label = ("LONG" if position_contracts > 0
                  else "SHORT" if position_contracts < 0 else "FLAT")
    pos_color = ("#86efac" if position_contracts > 0
                  else "#ff7a7a" if position_contracts < 0 else "#94a3b8")
    last_px = closes[-1] if closes else 0
    header = (f"{ticker} · {interval} · last {last_px:.4f} · "
              f"position: {pos_label} {abs(position_contracts):.0f}")
    if csd_risk is not None:
        header += f"  ·  CSD risk {csd_risk:.2f}"
    if ecology_phase:
        header += f"  ·  ecology: {ecology_phase}"
    ax.set_title(header, color="#e5e7eb", fontsize=12, loc="left", pad=10)

    # Position bias band
    if abs(position_contracts) > 0.01 and ts:
        ax.axhspan(min(lows), max(highs), facecolor=pos_color, alpha=0.04)

    # --- Near-term confirmation panel: 1m candles only ---
    if ax1 is not None:
        _draw_candles(ax1, candles_1m,
                       panel_label="1m · near-term confirmation")

    # --- P/L panel: account equity over time (its own scale) ---
    if "pnl" in axmap:
        _draw_line(axmap["pnl"], [("equity", "#38bdf8", eq_pts)],
                    panel_label="P/L · account equity over time",
                    zero_line=False, fill=False)

    # --- Decision panel: open-trade UNREALIZED P/L trajectory (own scale,
    # zero line) — its slope is the harvest/decay signal. ---
    if "captured" in axmap:
        _draw_line(axmap["captured"], [("open-trade unrealized P/L", "#f59e0b", un_pts)],
                    panel_label="open-trade unrealized P/L (read the slope)", zero_line=True)

    fig.autofmt_xdate()
    plt.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


# ----------------------------------------------------------- Opus call
def ask_opus_for_chart_review(model: str, png_bytes: bytes,
                                context: dict) -> Optional[dict]:
    """Send chart + context to Opus and parse the JSON response.
    Returns None on any failure (network, parse, missing API key)."""
    api_key = anthropic_api_key()
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        b64 = base64.b64encode(png_bytes).decode("ascii")
        msg = client.messages.create(
            model=model,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": b64}},
                    {"type": "text",
                      "text": "Context:\n" + json.dumps(context, indent=2)
                              + "\n\nRespond with the JSON object only."},
                ],
            }],
        )
        # Record usage to the cost telemetry (so the UI's $/h panel includes
        # the visual-review spend).
        try:
            from .store import STORE
            u = getattr(msg, "usage", None)
            if u is not None:
                STORE.record_llm_call(
                    model=getattr(msg, "model", None) or model,
                    input_tokens=int(getattr(u, "input_tokens", 0) or 0),
                    output_tokens=int(getattr(u, "output_tokens", 0) or 0),
                    cache_read=int(getattr(u, "cache_read_input_tokens", 0) or 0),
                    cache_write=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
                )
        except Exception:
            pass
        text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
        # Use the same lenient extractor as the decision-engine path. Models
        # — Haiku especially — often emit `{json}\n\nthen prose`, which a
        # strict json.loads() rejects with "Extra data".
        from .decision_engine import _extract_json_object
        data = _extract_json_object(text)
        # Normalize the three required fields. Unknown values fall back to
        # safe defaults (concern -> OK) so a malformed response can't gate
        # trading unintentionally.
        return {
            "trend": str(data.get("trend") or "sideways").lower(),
            "concern": (str(data.get("concern") or "OK").upper()
                          if str(data.get("concern", "")).upper() in ("OK", "CAUTION", "STOP")
                          else "OK"),
            "note": str(data.get("note") or "")[:240],
        }
    except Exception:
        return None


# ----------------------------------------------------------- orchestration
def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def maybe_run_review(settings, store, market: dict, account: dict,
                      last_decisions: List[dict], client=None) -> Optional[dict]:
    """Engine hook. Checks the configured cadence; if due, renders the chart,
    sends it to the model, updates STORE.visual_review, and writes the PNG to
    STATE_DIR for the UI.

    `client` (optional KalshiClient): when provided, a 1-minute candle series is
    fetched for the near-term confirmation panel. Fetched only when a review
    actually fires (every visual_review_interval_seconds), so it adds at most
    one 1m request per review, not per cycle. Without it, the chart falls back
    to a single primary-timeframe panel.

    Fires on the periodic cadence OR when the CSD / ecological-resilience
    governor is firing (subject to visual_review_event_min_interval_seconds).

    Returns the new review dict on a fresh call (with a "trigger" field of
    "periodic" or "csd_fire"). Returns None when:
      - feature is disabled
      - neither the periodic timer nor a governor-fire event is due
      - no candles available yet
      - the model call failed (in which case STORE.visual_review is left intact)
    """
    if not getattr(settings, "visual_review_enabled", False):
        return None

    interval = int(getattr(settings, "visual_review_interval_seconds", 600) or 600)
    state = store.visual_review or {}
    last_ts = state.get("last_ts")
    now = time.time()
    elapsed = None if last_ts is None else (now - float(last_ts))

    # Two ways a review can fire:
    #   (1) periodic — every `interval` seconds (baseline cadence).
    #   (2) event — the CSD / ecological-resilience governor is firing right now
    #       (enabled AND risk >= effective threshold). This catches fast regime
    #       breaks between scheduled reviews. A SEPARATE, shorter cooldown
    #       (event_min) bounds the cost so a sustained CSD storm can't fire a
    #       review every cycle — worst case is one review per event_min seconds.
    due_periodic = (elapsed is None) or (elapsed >= interval)
    csd = store.csd_state or {}
    csd_firing = (
        bool(getattr(settings, "visual_review_on_csd_fire", True))
        and bool(csd.get("enabled"))
        and csd.get("risk") is not None
        and csd.get("threshold") is not None
        and float(csd["risk"]) >= float(csd["threshold"]))
    event_min = int(getattr(settings, "visual_review_event_min_interval_seconds", 60) or 60)
    due_event = csd_firing and ((elapsed is None) or (elapsed >= event_min))

    if not (due_periodic or due_event):
        return None
    trigger = "periodic" if due_periodic else "csd_fire"

    candles = (market or {}).get("candles_full") or (market or {}).get("candles") or []
    if not candles:
        # Nothing to render yet; pretend the timer didn't fire so we retry next cycle.
        return None

    position = float((account or {}).get("position_contracts") or 0.0)
    csd_risk = float((store.csd_state or {}).get("risk") or 0.0)
    eco_phase = ((store.ecosystem or {}).get("phase") if store.ecosystem else None)
    ticker = getattr(settings, "ticker", "KXBTCPERP")
    timeframe = getattr(settings, "timeframe", "3m")

    # Near-term 1m candles for the confirmation panel. Prefer a series the
    # caller already attached to `market`; otherwise fetch fresh via the client.
    # This only runs when a review actually fires (the timer check above already
    # returned for non-due cycles), so it's at most one extra 1m request per
    # review. Best-effort — on failure we fall back to the single-panel chart.
    candles_1m = (market or {}).get("candles_1m") or []
    if not candles_1m and timeframe != "1m" and client is not None:
        try:
            from . import market_data as _md
            candles_1m = _md.fetch_candles(client, ticker, "1m", lookback_bars=120)
        except Exception:
            candles_1m = []

    # P/L history (equity / unrealized / realized over time) for the two extra
    # panels the model bases harvest/decay judgements on.
    pnl_history = list(getattr(store, "pnl_history", []) or [])[:240]

    png = render_chart(
        candles=candles, position_contracts=position,
        recent_decisions=last_decisions or [],
        ticker=ticker, csd_risk=csd_risk,
        ecology_phase=eco_phase, interval=timeframe,
        candles_1m=candles_1m or None,
        pnl_history=pnl_history or None)
    try:
        VISUAL_REVIEW_IMAGE_PATH.write_bytes(png)
    except Exception:
        pass  # disk write failure shouldn't fail the cycle

    # Compact context for the model
    context = {
        "ticker": ticker, "timeframe": timeframe,
        "chart_panels": (["3m trend anchor", "1m near-term confirmation"]
                          if candles_1m else [f"{timeframe} only"]),
        "position_contracts": position,
        "position_direction": ("LONG" if position > 0 else "SHORT" if position < 0 else "FLAT"),
        "entry_price": (account or {}).get("entry_price"),
        "unrealized_pnl": (account or {}).get("unrealized_pnl"),
        "equity": (account or {}).get("equity"),
        "csd_risk": csd_risk,
        "csd_gate_threshold": (store.csd_state or {}).get("threshold"),
        "ecology_phase": eco_phase,
        "recent_decisions_summary": [
            {"ts": d.get("ts"), "verb": d.get("verb"),
              "action": d.get("action"), "submitted": bool(
                  (d.get("execution") or {}).get("submitted"))}
            for d in (last_decisions or [])[-8:]
        ],
    }

    model = getattr(settings, "visual_review_model",
                       getattr(settings, "model", "claude-haiku-4-5-20251001"))
    review = ask_opus_for_chart_review(model, png, context)
    if review is None:
        # Don't update last_ts on failure — we want to retry next cycle.
        return None

    review_record = {
        "ts": _now_iso(),
        "last_ts": now,
        "trend": review["trend"],
        "concern": review["concern"],
        "note": review["note"],
        "model": model,
        "trigger": trigger,
        "image_path": str(VISUAL_REVIEW_IMAGE_PATH),
    }
    store.visual_review = review_record
    return review_record


# ----------------------------------------------------------- decision-time gate
_ENTRY_ACTIONS = {"ENTER_LONG", "ENTER_SHORT", "REVERSE_LONG", "REVERSE_SHORT", "ADD"}


def block_entry_if_stop(store, proposal: dict, settings) -> bool:
    """If the latest visual review's concern justifies it AND the proposal
    is a new entry/add/reverse, mutate the proposal to HOLD and return True.

    Closes and reduces are NEVER blocked — exiting an existing position must
    remain possible even when the visual layer is screaming.

    Blocks fire on STOP when visual_review_block_entries_on_stop is on, and
    on CAUTION when visual_review_block_entries_on_caution is on. STOP can be
    disabled independently if you want CAUTION-only gating.
    """
    review = store.visual_review or {}
    concern = (review.get("concern") or "OK").upper()
    block_stop = bool(getattr(settings, "visual_review_block_entries_on_stop", True))
    block_caution = bool(getattr(settings, "visual_review_block_entries_on_caution", False))
    should_block = ((concern == "STOP" and block_stop)
                     or (concern == "CAUTION" and block_caution))
    if not should_block:
        return False
    action = proposal.get("action")
    if action not in _ENTRY_ACTIONS:
        return False
    # Block the entry. Preserve the original action for logging.
    proposal["visual_review_blocked"] = True
    proposal["visual_review_blocked_action"] = action
    proposal["visual_review_blocked_concern"] = concern
    proposal["action"] = "HOLD"
    proposal["target_fraction"] = 0.0
    proposal["blended_score"] = 0.0
    proposal["confidence"] = 0.0
    proposal.setdefault("rationale_against", []).insert(
        0, f"Visual review {concern}: {review.get('note', '(no note)')}")
    return True
