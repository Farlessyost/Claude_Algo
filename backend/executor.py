"""Translates a sized Decision into real Kalshi margin orders and manages
resting orders (cancel/replace). On Kalshi perps: side 'bid' = buy (increase
long / cover short), side 'ask' = sell (increase short / sell long)."""
from __future__ import annotations

import time
import uuid
from typing import Optional

from .config import Settings, bar_seconds
from .kalshi_client import KalshiClient, KalshiError, to_float


def _client_order_id() -> str:
    return f"btcperp-{int(time.time()*1000)}-{uuid.uuid4().hex[:8]}"


# Tracks the most recent resting maker order so we can chase it to a taker
# once enough wall-clock time has passed. chase_delay_bars * bar_seconds is
# the threshold; tracking time (not cycles) keeps the semantic stable when
# the vol-aware loop drops cycle interval to seconds.
_RESTING = {"order_id": None, "side": None, "cycles": 0, "placed_at": 0.0,
            "drifts": 0}

# Consecutive zero-fill submits on the same side. When the book runs one
# way, a fresh maker each cycle stays at the wrong price and never fills.
# After consecutive_zerofill_taker_at submits on the same side without
# anything filling, escalate this submit to taker. Cleared on any fill.
_ZEROFILL = {"side": None, "count": 0}


def _reset_resting():
    _RESTING.update({"order_id": None, "side": None, "cycles": 0,
                     "placed_at": 0.0, "drifts": 0})


def _on_fill(_side=None):
    _ZEROFILL.update({"side": None, "count": 0})


def cancel_all_resting(client: KalshiClient, ticker: str, log) -> list:
    cancelled = []
    try:
        resp = client.get_orders(ticker=ticker, status="resting")
        for o in resp.get("orders", []) or []:
            oid = o.get("order_id") or o.get("id")
            if not oid:
                continue
            try:
                client.cancel_order(oid)
                cancelled.append(oid)
                log("info", "order", f"Cancelled resting order {oid}",
                    order_id=oid, event="cancel")
            except KalshiError as e:
                log("warn", "order", f"Cancel failed for {oid}: {e}")
    except Exception as e:
        log("warn", "order", f"Could not list resting orders: {e}")
    return cancelled


def flatten_position(client: KalshiClient, ticker: str, log, max_tries: int = 6) -> dict:
    """Close any open position using balance.position_value as the source of
    truth (the positions endpoint can lag). reduce_only so it can only flatten."""
    closed = {"closed": False, "tries": 0}
    for i in range(max_tries):
        try:
            bal = client.get_balance()
            sub = next((s for s in bal.get("subaccount_balances", [])
                        if s.get("subaccount") == 0), {})
            pv = to_float(sub.get("position_value"))
        except Exception as e:
            log("warn", "kill", f"flatten: balance read failed: {e}"); break
        if pv < 0.5:
            closed["closed"] = True
            break
        # determine direction + size
        try:
            pr = client.get_positions(ticker=ticker)
            p = next((x for x in pr.get("positions", [])
                      if x.get("market_ticker") == ticker), {})
            size = to_float(p.get("position"))
            ob = client.get_orderbook(ticker, depth=5).get("orderbook", {})
            bids = ob.get("bids", []); asks = ob.get("asks", [])
            if size >= 0:  # long (or unknown -> assume long) sell at bid
                price = round(to_float(bids[0][0]), 4) if bids else None
                side = "ask"
            else:
                price = round(to_float(asks[0][0]), 4) if asks else None
                side = "bid"
            count = abs(size) if abs(size) > 0 else round(pv / (price or 1), 2)
            if not price or count <= 0:
                log("warn", "kill", "flatten: no price/size; retrying"); time.sleep(1.0); continue
            r = client.create_order(ticker=ticker, side=side, count=count, price=price,
                                    client_order_id=_client_order_id(),
                                    time_in_force="immediate_or_cancel", reduce_only=True)
            log("warn", "kill", f"flatten: {side} {count} @ {price} "
                f"(filled {to_float(r.get('fill_count'))})", event="flatten")
            closed["tries"] = i + 1
        except KalshiError as e:
            log("warn", "kill", f"flatten order failed: {e}")
        time.sleep(1.2)
    return closed


def execute(client: KalshiClient, settings: Settings, decision: dict,
            sizing: dict, market: dict, account: dict, log) -> dict:
    """Submit the order(s) required to move toward the target position."""
    ticker = settings.ticker
    delta = sizing.get("delta_contracts") or 0.0
    verb = decision.get("verb", "HOLD")
    result = {"submitted": False, "verb": verb, "delta": delta, "orders": []}

    side = "bid" if delta > 0 else "ask"
    count = abs(delta)
    ob = market.get("orderbook", {}) or {}
    # Use the original action, not just the display verb. ACTION_TO_VERB maps
    # reversals to LONG/SHORT, but execution/maker semantics need the raw action.
    action = decision.get("action") or verb

    # --- Maker/taker switch ---
    # urgency comes from the strategy proposal (|alpha| for MPC). When urgency
    # is over the threshold k we cross the spread; otherwise we rest a maker.
    # k=None collapses to today's pure-maker behavior. Opus can promote a
    # maker cycle to taker via decision["force_taker"]=True, BUT only when the
    # live spread is tight (the (b) gate from the prompt). Opus can never
    # demote a taker cycle to maker — once urgency or force_taker says cross,
    # we cross.
    urgency = float(decision.get("urgency") or 0.0)
    k = settings.taker_threshold_k
    spread_bps = float((ob.get("spread_bps") or 0.0))
    force_taker = bool(decision.get("force_taker", False))
    # Aggressive override: if the most recent bar's realized move exceeds the
    # configured threshold, treat this cycle as a taker. This catches the
    # "large movement" case visible in the logs where price walks 30-50bps in
    # 5-10 minutes while a resting maker gets repriced and never fills.
    feats = (market.get("features") or {}) if isinstance(market, dict) else {}
    recent_move_bps = float(feats.get("recent_move_bps") or 0.0)
    recent_move_signed_bps = float(feats.get("recent_move_signed_bps") or 0.0)
    move_thresh = float(getattr(settings, "move_force_taker_bps", 0.0) or 0.0)
    move_supports_side = (
        (side == "bid" and recent_move_signed_bps >= move_thresh)
        or (side == "ask" and recent_move_signed_bps <= -move_thresh)
    )
    if (abs(delta) >= 1.0 and verb != "HOLD" and move_thresh > 0.0
            and move_supports_side and not force_taker and spread_bps <= 20.0):
        log("warn", "order",
            f"Signed recent-move {recent_move_signed_bps:+.1f}bps supports {side} "
            f"(abs pressure {recent_move_bps:.1f}bps >= {move_thresh:.1f}bps) "
            f"(spread {spread_bps:.1f}bps) -> forcing taker this cycle",
            event="move_taker", recent_move_bps=recent_move_bps,
            recent_move_signed_bps=recent_move_signed_bps,
            move_threshold_bps=move_thresh)
        force_taker = True
    # Safety gate on Opus promotion: don't cross a wide spread even if Opus
    # said go. This is the (b) condition from the system prompt enforced in
    # code as a belt-and-suspenders check.
    MAX_TAKER_SPREAD_BPS = 10.0
    if force_taker and spread_bps > MAX_TAKER_SPREAD_BPS:
        log("warn", "order",
            f"Opus taker_now=true but spread {spread_bps:.1f}bps > "
            f"{MAX_TAKER_SPREAD_BPS:.0f}bps -> ignoring promotion (stay maker)",
            event="taker_blocked")
        force_taker = False

    maker_allowed = (getattr(settings, "maker_mode", False)
                     and verb not in ("CLOSE", "REVERSE_LONG", "REVERSE_SHORT")
                     and action not in ("CLOSE", "REVERSE_LONG", "REVERSE_SHORT"))
    if k is None:
        urgency_taker = False           # pure maker baseline
    else:
        urgency_taker = urgency >= float(k)
    # Consecutive zero-fill escalation: when N maker submits on the same side
    # have all failed to get any fill, the book is running one way and the
    # passive limit is permanently stale. Cross THIS submit. Counter is reset
    # whenever a fill happens (see _on_fill).
    zerofill_limit = int(getattr(settings, "consecutive_zerofill_taker_at", 0) or 0)
    zerofill_taker = False
    if (zerofill_limit > 0 and _ZEROFILL["side"] == side
            and _ZEROFILL["count"] >= zerofill_limit and spread_bps <= 20.0):
        log("warn", "order",
            f"Consecutive-zerofill chase: {_ZEROFILL['count']} prior maker "
            f"submits on {side} side returned no fill -> crossing now",
            event="zerofill_chase", consecutive_zerofills=_ZEROFILL["count"],
            zerofill_limit=zerofill_limit)
        zerofill_taker = True
    # ONE-WAY PROMOTE: maker_allowed && not(urgency_taker || force_taker || zerofill_taker).
    maker = (maker_allowed
              and not (urgency_taker or force_taker or zerofill_taker))
    TICK = 0.0001
    KEEP_TOL = 3 * TICK   # keep a resting maker order if within this of the inside

    # What's currently resting (so a maker order can KEEP its queue position)?
    try:
        resting = (client.get_orders(ticker=ticker, status="resting").get("orders", [])
                   if settings.allow_live_orders else [])
    except Exception:
        resting = []

    if verb == "HOLD" or abs(delta) < 1.0:
        if resting:
            cancel_all_resting(client, ticker, log)   # signal flat -> pull orders
        _reset_resting()
        log("info", "decision", f"HOLD — no order (delta {delta:.2f})")
        return result

    # CHASE: if our tracked resting maker on the desired side has been alive for
    # >= chase_delay_bars cycles, escalate this cycle to a taker. (Pulling the
    # order forces a fresh IOC at the inside.)
    chase_n = int(getattr(settings, "chase_delay_bars", 999) or 999)
    chase_seconds = chase_n * bar_seconds(getattr(settings, "timeframe", "5m"))
    elapsed = (time.time() - _RESTING["placed_at"]) if _RESTING["placed_at"] else 0.0
    if (maker and settings.allow_live_orders
            and _RESTING["order_id"] and _RESTING["side"] == side
            and elapsed >= chase_seconds
            and any((o.get("order_id") or o.get("id")) == _RESTING["order_id"]
                    for o in resting)):
        log("warn", "order",
            f"Maker chase: resting {elapsed:.0f}s unfilled "
            f"(>= {chase_seconds:.0f}s = {chase_n} bars) -> crossing now",
            event="chase")
        cancel_all_resting(client, ticker, log)
        _reset_resting()
        maker = False

    if maker:
        price = ob.get("best_bid") if side == "bid" else ob.get("best_ask")
        tif = "good_till_canceled"
        post_only = True
    else:
        price = sizing.get("limit_price") or market.get("price")
        tif = "immediate_or_cancel"
        post_only = False
    if not price or price <= 0:
        log("error", "order", "No valid limit price; skipping submit")
        return result

    # MAKER queue management: if we already have a resting order on the desired
    # side that's still near the inside, LEAVE IT so it keeps its place in the
    # queue and has time to fill. Only cancel/replace if it's the wrong side or
    # has drifted away from the book. (This is what makes passive fills possible;
    # cancelling every cycle was sabotaging fills.)
    if maker and settings.allow_live_orders:
        # If the order we were tracking isn't in the book anymore (filled, or
        # cancelled out-of-band), our cycle/drift counters are stale — reset
        # so a fresh maker order starts from a clean drift count.
        if _RESTING["order_id"] and not any(
                (o.get("order_id") or o.get("id")) == _RESTING["order_id"]
                for o in resting):
            _reset_resting()
        wrong = [o for o in resting if o.get("side") != side]
        same = [o for o in resting if o.get("side") == side]
        if wrong:
            cancel_all_resting(client, ticker, log); same = []; _reset_resting()
        inside = price
        if same and inside is not None:
            rp = to_float(same[0].get("price"))
            if abs(rp - inside) <= KEEP_TOL:
                oid = same[0].get("order_id") or same[0].get("id")
                if _RESTING["order_id"] == oid and _RESTING["side"] == side:
                    _RESTING["cycles"] += 1
                else:
                    # Adopting an existing order (probably from a restart);
                    # start the clock now — best we can do without an
                    # exchange-side placement timestamp.
                    _RESTING.update({"order_id": oid, "side": side,
                                      "cycles": 1, "placed_at": time.time(),
                                      "drifts": 0})
                log("info", "order",
                    f"Maintaining resting maker {side} {same[0].get('remaining_count')} "
                    f"@ {rp:.4f} (inside {inside:.4f}) — kept "
                    f"{_RESTING['cycles']}/{chase_n} cycles before chase",
                    event="kept", side=side, price=rp,
                    age_cycles=_RESTING["cycles"])
                result.update({"kept": True, "side": side, "price": rp, "maker": True,
                               "age_cycles": _RESTING["cycles"]})
                return result
            # Drifted away from the inside. The plain reprice path resets the
            # chase clock, so during a sustained move the chase never fires.
            # Carry the drift counter across the cancel, and if we've drifted
            # the configured limit, escalate to taker THIS cycle instead of
            # reposting another doomed maker.
            prior_drifts = (_RESTING["drifts"]
                            if _RESTING["side"] == side else 0)
            new_drifts = prior_drifts + 1
            cancel_all_resting(client, ticker, log)
            drift_limit = int(getattr(settings, "chase_after_drifts", 999) or 999)
            if new_drifts >= drift_limit:
                log("warn", "order",
                    f"Maker drift chase: drifted {new_drifts}x "
                    f"(>= {drift_limit}) -> crossing now instead of reposting",
                    event="drift_chase", drifts=new_drifts,
                    drift_limit=drift_limit)
                _reset_resting()
                maker = False
            else:
                # Keep the counter alive across the reprice. The new order's
                # id will be stamped in when it's submitted below.
                _RESTING.update({"order_id": None, "side": side, "cycles": 0,
                                 "placed_at": 0.0, "drifts": new_drifts})
    elif settings.allow_live_orders and resting:
        cancel_all_resting(client, ticker, log)       # taker path: clear then IOC
        _reset_resting()

    # If drift_chase above flipped maker -> False mid-flight, price/tif/post_only
    # were assigned with maker values; swap them out for taker before submit.
    if not maker and post_only:
        price = sizing.get("limit_price") or market.get("price")
        tif = "immediate_or_cancel"
        post_only = False
        if not price or price <= 0:
            log("error", "order", "No valid limit price after drift_chase; skipping submit")
            return result

    current = account.get("position_contracts") or 0.0
    target = sizing.get("target_contracts") or 0.0
    # Kalshi forbids reduce_only on GTC/post-only orders — it only works with
    # IOC/FOK. So reduce_only is set ONLY for taker exits; maker reduces rely on
    # correct delta sizing (and the per-cycle cancel/replace) to avoid crossing 0.
    reduce_only = (settings.reduce_only_on_exit and verb in ("REDUCE", "CLOSE")
                   and not maker)
    # Reversal can't be reduce_only (it flips through zero) — send full delta.

    if not settings.allow_live_orders:
        log("warn", "order",
            f"Live orders not enabled — simulated only ({'maker' if maker else 'taker'} "
            f"delta {delta:+.2f} {side} @ {price:.4f})", event="simulated")
        result.update({"simulated": True, "side": side, "count": count,
                       "price": price, "maker": maker})
        return result

    coid = _client_order_id()
    try:
        resp = client.create_order(
            ticker=ticker, side=side, count=count, price=round(price, 4),
            client_order_id=coid,
            time_in_force=tif,
            reduce_only=reduce_only,
            post_only=post_only,
        )
        order_rec = {
            "ts": _now_iso(), "verb": verb, "side": side, "count": count,
            "price": round(price, 4), "reduce_only": reduce_only, "maker": maker,
            "client_order_id": coid,
            "order_id": resp.get("order_id"),
            "fill_count": to_float(resp.get("fill_count")),
            "remaining_count": to_float(resp.get("remaining_count")),
            "avg_fill_price": to_float(resp.get("average_fill_price")),
            "event": "submitted",
        }
        result["orders"].append(order_rec)
        result["submitted"] = True
        # If this was a fresh maker, start the chase timer at age=1 for the
        # next cycle's accounting; if it was a taker, clear any tracking.
        fill_now = float(order_rec.get("fill_count") or 0.0)
        if maker and order_rec.get("remaining_count", 0) > 0:
            _RESTING.update({"order_id": order_rec.get("order_id"),
                             "side": side, "cycles": 1,
                             "placed_at": time.time()})
        else:
            _reset_resting()
        # Track consecutive zero-fill maker submits on the same side so the
        # next cycle's executor can decide whether to escalate to a taker.
        if maker and fill_now <= 0.0:
            if _ZEROFILL["side"] == side:
                _ZEROFILL["count"] += 1
            else:
                _ZEROFILL.update({"side": side, "count": 1})
        else:
            _on_fill()
        log("info", "order",
            f"LIVE {verb}: {side} {count:.2f} @ {price:.4f} "
            f"(filled {order_rec['fill_count']:.2f}, "
            f"{'maker' if maker else 'taker'})", **order_rec)
    except KalshiError as e:
        msg = str(e)
        # "post only cross" is the post-only safety doing its job: by the time
        # the order reaches Kalshi the book moved a tick. The executor will
        # repost on the next cycle. Demote to warn so it doesn't drown the log.
        if "post only cross" in msg.lower():
            log("warn", "order",
                f"Post-only rejected (book moved); will repost next cycle: {msg}")
        else:
            log("error", "order", f"Order rejected: {msg}")
        result["error"] = msg
    return result


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
