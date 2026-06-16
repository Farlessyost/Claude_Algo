"""Combines the deterministic strategy proposal with an optional Opus review to
produce the final Decision the executor acts on.

If `let_opus_decide` is on and ANTHROPIC_API_KEY is set, Opus (claude-opus-4-8)
receives the full account + market + strategy context and returns a structured
JSON action. If Opus is unavailable or errors, we fall back to the strategy
proposal so the autonomous loop never stalls."""
from __future__ import annotations

import json
import re
from typing import Optional


def _extract_json_object(text: str) -> dict:
    """Parse the first JSON object from a model response, tolerating trailing
    prose. Haiku in particular tends to emit `{json}\n\nThis is because...`
    which makes a strict json.loads() fail with "Extra data". raw_decode
    parses from the first { and ignores anything after the matching close.

    Order of attempts:
      1. Strip markdown code fences if present.
      2. raw_decode from the first '{'.
      3. Strip a trailing comment-style block (handles "{...}\n# note").
      4. Last-ditch regex grab between the first '{' and last '}'.

    Raises json.JSONDecodeError if nothing parses.
    """
    s = (text or "").strip()
    if s.startswith("```"):
        s = s.strip("`").strip()
        if s.startswith("json"):
            s = s[4:].lstrip()
    start = s.find("{")
    if start < 0:
        raise json.JSONDecodeError("no '{' found", s, 0)
    # Primary: raw_decode handles trailing prose cleanly
    try:
        obj, _ = json.JSONDecoder().raw_decode(s[start:])
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Fallback: greedy match from first '{' to last '}'
    end = s.rfind("}")
    if end > start:
        candidate = s[start:end + 1]
        # Strip line-end JS-style comments which some models emit
        candidate = re.sub(r"//[^\n]*", "", candidate)
        return json.loads(candidate)
    raise json.JSONDecodeError("no parseable JSON object", s, start)

from .config import Settings, anthropic_api_key

# Map proposal actions -> final decision verbs shown in the UI.
ACTION_TO_VERB = {
    "ENTER_LONG": "LONG", "REVERSE_LONG": "LONG", "ADD": "ADD",
    "ENTER_SHORT": "SHORT", "REVERSE_SHORT": "SHORT",
    "REDUCE": "REDUCE", "CLOSE": "CLOSE", "HOLD": "HOLD",
}

SYSTEM_PROMPT = """You are the risk-overlay brain of an autonomous BTC perpetual \
futures bot on Kalshi (ticker KXBTCPERP, 1 contract = 1/10,000 BTC). You trade \
ONLY this product, up to the configured leverage cap.

IMPORTANT: the `strategy_proposal` you receive comes from a STATISTICALLY \
VALIDATED mean-reversion edge (out-of-sample t-stat > 3, positive across every \
walk-forward fold). Your job is NOT to second-guess it with discretionary \
trend/momentum opinions — that discretion is unvalidated and tends to destroy \
the edge. Default to FOLLOWING the proposal's direction/action. Only deviate \
(reduce or hold) when there is a concrete RISK reason: e.g. liquidation danger, \
abnormal spread/illiquidity, stale or inconsistent data, or margin exhaustion. \
If the proposal says act and nothing is wrong, act with it.

ECOLOGY LAYER (Trophic Information Forager): the proposal may include an \
`ecosystem` block with a current ecological phase classification of the \
market. The phases mean:
  producer    baseline market-making; quiet liquidity provision
  predator    cascade in progress (high vol_z, spread widening, OI unwinding) \
— mean-reversion is DANGEROUS here; lean toward REDUCE/HOLD even if the \
strategy proposal says act
  exhaustion  predator slowing (vol still high but decelerating, stretch_z \
elevated) — small probe is reasonable
  scavenger   snap-back zone (stretched price, vol decelerating, spread \
normalizing) — this is the IDEAL fill window for the mean-reversion edge; \
follow the proposal aggressively
  decomposer  liquidity restoration (vol normalized, spread compressing) — \
typical reversion-to-fair conditions
  churn       normal range; baseline behavior
The proposal also carries `organisms.allocation` — a softmax-style fitness \
score across {predator, scavenger, decomposer, mycelium, immune, producer}. \
When `immune` has the highest score, the ecosystem itself is signaling \
"infection / toxic flow" and you should be conservative.

USE THE ECOLOGY AS RISK CONTEXT, not as a second signal: if phase=predator AND \
liq_proxy_z is high AND the strategy says enter — the mean-reversion edge is \
likely standing in front of a real cascade. Prefer REDUCE/HOLD and explain. \
If phase=scavenger or decomposer and the proposal says act — that's exactly \
when the validated edge fires; act with it.

EXECUTION POLICY (maker/taker): the validated baseline is MAKER. The system \
rests post-only limits and captures the spread. Going taker (crossing the \
spread) costs the half-spread and is usually a net loser at this edge size. \
The default for `taker_now` is FALSE. Only set `taker_now: true` when ALL \
THREE are true:
  (a) a clear reversion pattern is forming (high |urgency|/|alpha| in the \
      proposal AND the proposal is non-HOLD on a confirmed signal),
  (b) the live spread is tight: market.orderbook.spread_bps <= 10,
  (c) you would otherwise miss the trade (the move looks imminent and a \
      passive fill is unlikely in time).
You CANNOT downgrade an aggressive cycle to maker — that's not your call. \
This field is a one-way promote only.

You receive: the real margin account state, live market features, the orderbook, \
funding, the validated strategy proposal, and the live ecology. Confirm or \
risk-adjust the proposal.

Respond with ONLY a JSON object, no prose:
{
  "action": "ENTER_LONG|ENTER_SHORT|ADD|REDUCE|CLOSE|REVERSE_LONG|REVERSE_SHORT|HOLD",
  "confidence": 0.0-1.0,
  "expected_edge_pct": number,
  "expected_profit_usd": number,
  "expected_risk_usd": number,
  "reasons_for": ["..."],
  "reasons_against": ["..."],
  "why_better_than_hold": "...",
  "ecology_note": "one-sentence rationale citing phase/keystone/organism if relevant",
  "taker_now": true|false,
  "urgency_override": number   // optional multiplier on the proposal's urgency \
(>=1 boosts; values <1 are ignored — Opus cannot soften aggression)
}"""


def decide(settings: Settings, account: dict, market: dict, proposal: dict,
           log=lambda *a, **k: None) -> dict:
    base = _from_proposal(proposal)
    action = proposal.get("action", "HOLD")

    # Tactical strategy-layer reflex bypass: when the engine's adverse-move
    # override has flipped the proposal to a full-conviction REVERSE, that is
    # a reactive trade that should NOT wait for or be gated by the LLM. The
    # whole point of the reflex is to react before a slower discretionary
    # check kicks in. Sonnet would otherwise add latency and could even
    # override the action to HOLD, defeating the override entirely.
    if proposal.get("adverse_move_bps") is not None:
        base["source"] = "strategy"
        base["model_note"] = (f"adverse-move REVERSE "
                              f"({proposal.get('adverse_move_bps'):.1f}bps) — "
                              f"LLM gate bypassed (tactical reflex)")
        log("warn", "decision", base["model_note"])
        return base

    if not (settings.let_opus_decide and anthropic_api_key()):
        base["source"] = "strategy"
        base["model_note"] = ("Opus disabled" if not settings.let_opus_decide
                              else "ANTHROPIC_API_KEY not set — using strategy")
        return base

    # Cost gate: only spend an Opus call when the validated strategy actually
    # proposes a trade. On HOLD cycles there is nothing for the risk-overlay to
    # confirm, so skip the API call entirely (no tokens billed). Hard risk
    # checks still run every cycle regardless, independent of Opus.
    if settings.opus_only_on_signal and action == "HOLD":
        base["source"] = "strategy"
        base["model_note"] = "HOLD — Opus call skipped (cost gate)"
        return base

    try:
        model_used, trade_fraction = _pick_decision_model(settings, account, market, proposal)
        opus = _ask_opus(settings, account, market, proposal, model_used)
        if opus:
            opus["source"] = "opus"
            opus["model_note"] = (f"Decided by {model_used} "
                                    f"(trade size {trade_fraction:.2f} of max)")
            opus["model_used"] = model_used
            return opus
    except Exception as e:  # never let the loop die on an LLM hiccup
        log("warn", "model", f"LLM call failed, using strategy fallback: {e}")

    base["source"] = "strategy"
    base["model_note"] = "Opus errored — strategy fallback"
    return base


def _from_proposal(proposal: dict) -> dict:
    action = proposal.get("action", "HOLD")
    # Strategy-layer force_taker (set by the adverse-move override) propagates
    # straight through. Both the executor and the Opus path read this; Opus
    # is allowed to FURTHER promote but never demote.
    force_taker = bool(proposal.get("force_taker", False))
    return {
        "action": action,
        "verb": ACTION_TO_VERB.get(action, "HOLD"),
        "confidence": proposal.get("confidence", 0.0),
        "expected_edge_pct": proposal.get("expected_edge_pct", 0.0),
        "urgency": proposal.get("urgency", 0.0),
        "taker_now": force_taker,
        "force_taker": force_taker,
        "expected_profit_usd": None,
        "expected_risk_usd": None,
        "reasons_for": proposal.get("rationale_for", []),
        "reasons_against": proposal.get("rationale_against", []),
        "why_better_than_hold": (
            "Blended signal exceeds the configured entry/exit thresholds."
            if action != "HOLD" else "No action has positive expected edge now."),
        "ecology_note": "",   # only set when Opus/Sonnet path is taken
        "ecosystem": proposal.get("ecosystem"),
        "ecosystem_applied": bool(proposal.get("ecosystem_applied", False)),
    }


def _pick_decision_model(settings: Settings, account: dict, market: dict,
                          proposal: dict) -> tuple[str, float]:
    """Pick which LLM should approve this trade.

    Tier rule: trades larger than `large_trade_fraction_threshold` of max
    buying power go to `model_large_trade` (Sonnet by default); smaller
    trades go to `model_small_trade` (Haiku by default). Size is measured
    as |target_fraction - current_fraction| where fractions are normalized
    to max buying power. Returns (model_id, trade_fraction).

    Falls back to `settings.model` if either tier model is blank.
    """
    small = (getattr(settings, "model_small_trade", "") or "").strip()
    large = (getattr(settings, "model_large_trade", "") or "").strip()
    threshold = float(getattr(settings, "large_trade_fraction_threshold", 0.4) or 0.4)

    # Current position as a fraction of max buying power
    equity = float(account.get("equity") or 0.0)
    leverage = float(getattr(settings, "leverage_target", 5.8) or 5.8)
    max_notional = float(getattr(settings, "max_position_notional_usd", 0.0) or 0.0)
    if max_notional <= 0:
        max_notional = equity * leverage
    price = float(market.get("price") or 0.0)
    current_contracts = float(account.get("position_contracts") or 0.0)
    current_notional = abs(current_contracts) * price
    current_fraction = (current_notional / max_notional) if max_notional > 0 else 0.0
    # signed current fraction (long > 0, short < 0)
    if current_contracts < 0:
        current_fraction = -current_fraction

    target_fraction = float(proposal.get("target_fraction") or 0.0)
    delta_fraction = abs(target_fraction - current_fraction)

    if delta_fraction >= threshold:
        chosen = large or settings.model
    else:
        chosen = small or settings.model
    return chosen, delta_fraction


def _ask_opus(settings: Settings, account: dict, market: dict,
              proposal: dict, model_override: Optional[str] = None) -> Optional[dict]:
    import anthropic  # imported lazily

    client = anthropic.Anthropic(api_key=anthropic_api_key())
    # Trim the ecology block before sending — drop weak edges and only keep
    # the keystone-relevant centrality. Keeps the prompt under ~300 tokens
    # while preserving the risk-overlay-relevant signal.
    eco_full = proposal.get("ecosystem") or {}
    eco_trim = None
    if eco_full:
        eco_trim = {
            "phase": eco_full.get("phase"),
            "phase_rationale": eco_full.get("rationale"),
            "size_mult": eco_full.get("size_mult"),
            "applied_to_live": bool(proposal.get("ecosystem_applied")),
            "keystone": eco_full.get("keystone"),
            "net_entropy": eco_full.get("net_entropy"),
            "drivers": eco_full.get("drivers"),
            "organisms": (eco_full.get("organisms") or {}).get("allocation"),
            "multi_summary": eco_full.get("multi_summary"),
        }
    proposal_trim = {k: v for k, v in proposal.items() if k != "ecosystem"}
    context = {
        "leverage_cap": settings.max_leverage,
        "leverage_target": settings.leverage_target,
        "timeframe": settings.timeframe,
        "account": {k: account.get(k) for k in (
            "equity", "available_balance", "position_contracts",
            "position_direction", "entry_price", "unrealized_pnl",
            "effective_leverage", "notional_exposure", "margin_safety",
            "liquidation_risk", "open_orders_count")},
        "market": {
            "price": market.get("price"),
            "features": market.get("features"),
            "orderbook": {k: market.get("orderbook", {}).get(k) for k in (
                "best_bid", "best_ask", "spread_bps", "depth_total")},
            "funding": market.get("funding"),
        },
        "strategy_proposal": proposal_trim,
        "ecosystem": eco_trim,
    }
    msg = client.messages.create(
        model=(model_override or settings.model),
        max_tokens=1024,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": json.dumps(context)}],
    )
    # Cost telemetry — record every call's usage to the persistent log so the UI
    # panel can show $/hour and per-model breakdowns. Wrapped to never break the
    # loop on a usage-field shape change or import-order issue.
    try:
        from .store import STORE
        u = getattr(msg, "usage", None)
        if u is not None:
            STORE.record_llm_call(
                model=getattr(msg, "model", None) or settings.model,
                input_tokens=int(getattr(u, "input_tokens", 0) or 0),
                output_tokens=int(getattr(u, "output_tokens", 0) or 0),
                cache_read=int(getattr(u, "cache_read_input_tokens", 0) or 0),
                cache_write=int(getattr(u, "cache_creation_input_tokens", 0) or 0),
            )
    except Exception:
        pass
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    data = _extract_json_object(text)
    action = data.get("action", "HOLD")
    base_urgency = float(proposal.get("urgency", 0.0))
    # ONE-WAY PROMOTE. Opus can ONLY make a maker cycle into a taker — never
    # the other way around. `urgency_override` is treated as a multiplier on
    # base urgency: values >=1 amplify (helps clear the k gate); values <1
    # are clamped to 1.0 (no softening). `taker_now: true` is a hard
    # promotion — it sets force_taker so the executor crosses even when
    # taker_threshold_k is None (pure-maker mode).
    mult_raw = data.get("urgency_override")
    try:
        mult = float(mult_raw) if mult_raw is not None else None
    except (TypeError, ValueError):
        mult = None
    if mult is not None and mult < 1.0:
        mult = 1.0   # demotion attempt -> ignored
    effective_urgency = base_urgency * mult if mult is not None else base_urgency
    # OR Opus's taker_now with the proposal's force_taker so a strategy-layer
    # adverse-move flip (one-way promote) can't be silently demoted by Opus.
    force_taker = bool(data.get("taker_now", False)) or bool(
        proposal.get("force_taker", False))
    return {
        "action": action,
        "verb": ACTION_TO_VERB.get(action, "HOLD"),
        "confidence": float(data.get("confidence", proposal.get("confidence", 0.0))),
        "expected_edge_pct": float(data.get("expected_edge_pct",
                                           proposal.get("expected_edge_pct", 0.0))),
        "urgency": effective_urgency,
        "urgency_base": base_urgency,
        "taker_now": force_taker,
        "force_taker": force_taker,
        "urgency_override": mult,
        "expected_profit_usd": data.get("expected_profit_usd"),
        "expected_risk_usd": data.get("expected_risk_usd"),
        "reasons_for": data.get("reasons_for", []),
        "reasons_against": data.get("reasons_against", []),
        "why_better_than_hold": data.get("why_better_than_hold", ""),
        "ecology_note": data.get("ecology_note", ""),
        "ecosystem": proposal.get("ecosystem"),
        "ecosystem_applied": bool(proposal.get("ecosystem_applied", False)),
    }
