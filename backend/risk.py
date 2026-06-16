"""Position sizing from the real account + leverage target, plus the risk
checks. Most checks are WARN (non-blocking) per the operator's preference for
prioritizing profit. A small set are hard BLOCKS that protect capital from a
runaway bug at leverage; these never prompt the user, they just clamp/abort."""
from __future__ import annotations

from typing import Optional

from .config import Settings


def harvest_reserved_usd(settings: Settings) -> float:
    """Captured forager profit protected from same-session redeploy."""
    if not bool(getattr(settings, "harvest_reserve_enabled", True)):
        return 0.0
    frac = max(0.0, min(1.0, float(getattr(settings, "harvest_reserve_fraction", 1.0) or 0.0)))
    if frac <= 0.0:
        return 0.0
    try:
        from .store import STORE
        captured = float(getattr(STORE, "forager_captured_cumulative", 0.0) or 0.0)
    except Exception:
        captured = 0.0
    return max(0.0, captured * frac)


def deployable_equity(settings: Settings, equity: float) -> tuple[float, float]:
    """Equity available for new risk after protecting harvested profit."""
    equity = float(equity or 0.0)
    reserve = min(equity, harvest_reserved_usd(settings))
    return max(0.0, equity - reserve), reserve


def max_deployable_notional(settings: Settings, deployable: float) -> float:
    derived = float(deployable or 0.0) * float(settings.max_leverage)
    explicit = float(getattr(settings, "max_position_notional_usd", 0.0) or 0.0)
    return min(explicit, derived) if explicit > 0 else derived


def current_mpc_fraction(settings: Settings, account: dict, market: dict) -> float:
    """Normalize the real live position into MPC target-fraction units."""
    equity = float(account.get("equity") or 0.0)
    deployable, _reserved = deployable_equity(settings, equity)
    price = float(market.get("price") or 0.0)
    contracts = float(account.get("position_contracts") or 0.0)
    lev = min(float(settings.leverage_target), float(settings.max_leverage))
    scale = float(getattr(settings, "position_scale", 1.0) or 1.0)
    denom = deployable * lev * scale
    if price <= 0.0 or denom <= 0.0:
        return 0.0
    return contracts * price / denom


def effective_action_from_sizing(account: dict, sizing: dict) -> str:
    """Classify the actual order implied by target/current contracts.

    Strategy labels are directional intentions. After sizing, the real delta
    may be a reduction even if the strategy label said ADD. Executor side is
    delta-based, so this keeps decision/risk/maker-taker semantics honest.
    """
    current = float(account.get("position_contracts") or 0.0)
    target = float(sizing.get("target_contracts") or 0.0)
    delta = float(sizing.get("delta_contracts") or 0.0)
    if abs(delta) < 1.0:
        return "HOLD"
    if abs(target) < 1e-9:
        return "CLOSE"
    if abs(current) < 1e-9:
        return "ENTER_LONG" if target > 0 else "ENTER_SHORT"
    if (current > 0) != (target > 0):
        return "REVERSE_LONG" if target > 0 else "REVERSE_SHORT"
    if abs(target) > abs(current):
        return "ADD"
    return "REDUCE"


def increases_risk(account: dict, sizing: dict) -> bool:
    current = float(account.get("position_contracts") or 0.0)
    target = float(sizing.get("target_contracts") or 0.0)
    delta = float(sizing.get("delta_contracts") or 0.0)
    if abs(delta) < 1.0:
        return False
    if abs(current) < 1e-9:
        return abs(target) > 0
    if abs(target) < 1e-9:
        return False
    if (current > 0) != (target > 0):
        return True
    return abs(target) > abs(current) + 1e-9


def mpc_confidence_scale(settings: Settings, confidence: float) -> float:
    """Nonlinear exposure scale for risk-increasing MPC movement.

    Confidence is an empirical probability/quality proxy, not a direction.
    A 7-8% confidence target should be a tiny probe, not a nearly 1x position.
    """
    if not bool(getattr(settings, "mpc_confidence_sizing_enabled", True)):
        return 1.0
    conf = max(0.0, min(1.0, float(confidence or 0.0)))
    full_at = max(1e-6, float(getattr(settings, "mpc_confidence_full_at", 0.55) or 0.55))
    power = max(0.25, float(getattr(settings, "mpc_confidence_power", 1.5) or 1.5))
    floor = max(0.0, min(1.0, float(getattr(settings, "mpc_min_confidence_scale", 0.0) or 0.0)))
    scaled = min(1.0, (conf / full_at) ** power)
    return max(floor, scaled)


def size_position(settings: Settings, account: dict, market: dict,
                  proposal: dict) -> dict:
    """Compute target contracts/notional for the proposed action."""
    equity = account.get("equity") or 0.0
    deployable, reserved = deployable_equity(settings, equity)
    price = market.get("price") or 0.0
    current = account.get("position_contracts") or 0.0
    lev = min(settings.leverage_target, settings.max_leverage)

    # Confidence-scaled target leverage: lean in when edge is strong.
    conf = proposal.get("confidence") or 0.0
    target_lev = lev * max(0.35, min(1.0, 0.5 + conf * 0.6))
    # Trade-size dial: scale the targeted notional. Final position is still
    # clamped by max_notional below, so Heavy can't exceed the leverage cap.
    scale_nonmpc = float(getattr(settings, "position_scale", 1.0) or 1.0)
    target_notional = deployable * target_lev * scale_nonmpc

    if price <= 0 or equity <= 0:
        return {"target_contracts": 0.0, "delta_contracts": 0.0,
                "target_notional": 0.0, "target_leverage": 0.0,
                "deployable_equity": round(deployable, 2),
                "harvest_reserved_usd": round(reserved, 2),
                "note": "no price/equity"}

    # MPC path: size DIRECTLY to the controller's continuous target fraction
    # (preserves partial positions / low turnover). Bypasses the discrete
    # action-based sizing used by the other variants.
    target_fraction = proposal.get("target_fraction")
    if target_fraction is not None:
        import math
        # Trade-size dial: scale the controller's target_fraction. The
        # max-notional clamp below still hard-bounds the result, so Heavy
        # (2.0) can never push past the platform leverage cap.
        scale = float(getattr(settings, "position_scale", 1.0) or 1.0)
        cap_notional = max_deployable_notional(settings, deployable)
        raw = scale * target_fraction * deployable * lev / price
        max_contracts = cap_notional / price
        raw = max(-max_contracts, min(max_contracts, raw))
        current_int = float(round(current))
        conf_scale = mpc_confidence_scale(settings, conf)
        raw_before_confidence = raw
        if _target_increases_risk(current_int, raw):
            raw = current_int + (raw - current_int) * conf_scale
        target = math.copysign(math.floor(abs(raw)), raw) if raw else 0.0
        delta = float(round(target - current_int))
        return {
            "target_contracts": float(target),
            "delta_contracts": delta,
            "target_notional": round(abs(target) * price, 2),
            "target_leverage": round((abs(target) * price / equity) if equity else 0.0, 2),
            "deployable_equity": round(deployable, 2),
            "harvest_reserved_usd": round(reserved, 2),
            "deployable_notional_cap": round(cap_notional, 2),
            "mpc_confidence_scale": round(conf_scale, 4),
            "target_contracts_before_confidence": round(raw_before_confidence, 4),
            "limit_price": _limit_price(settings, market, delta),
        }

    full_target_contracts = target_notional / price
    action = proposal.get("action", "HOLD")
    desired = proposal.get("desired_direction", "FLAT")

    target = current
    if action in ("ENTER_LONG", "REVERSE_LONG"):
        target = full_target_contracts
    elif action in ("ENTER_SHORT", "REVERSE_SHORT"):
        target = -full_target_contracts
    elif action == "ADD":
        # add a third of a full clip in the current direction
        step = full_target_contracts * 0.34
        target = current + (step if current > 0 else -step)
    elif action == "REDUCE":
        target = current * 0.5
    elif action == "CLOSE":
        target = 0.0
    elif action == "HOLD":
        target = current

    # clamp to max notional
    max_notional = max_deployable_notional(settings, deployable)
    max_contracts = max_notional / price
    if abs(target) > max_contracts:
        target = max_contracts if target > 0 else -max_contracts

    # Kalshi requires WHOLE-INTEGER contract counts -> floor magnitude toward 0
    # (never round up past the leverage/notional cap). Current position is held
    # as an integer on the exchange, so round it too before differencing.
    import math
    target = math.copysign(math.floor(abs(target)), target) if target else 0.0
    current_int = float(round(current))
    delta = float(round(target - current_int))
    target = float(target)
    return {
        "target_contracts": target,
        "delta_contracts": delta,
        "target_notional": round(abs(target) * price, 2),
        "target_leverage": round((abs(target) * price / equity) if equity else 0.0, 2),
        "deployable_equity": round(deployable, 2),
        "harvest_reserved_usd": round(reserved, 2),
        "deployable_notional_cap": round(max_notional, 2),
        "limit_price": _limit_price(settings, market, delta),
    }


def _target_increases_risk(current: float, target: float) -> bool:
    if abs(target - current) < 1.0:
        return False
    if abs(current) < 1e-9:
        return abs(target) > 0.0
    if abs(target) < 1e-9:
        return False
    if (current > 0) != (target > 0):
        return True
    return abs(target) > abs(current) + 1e-9


def adaptive_signal_floors(settings: Settings, market: dict, sizing: dict,
                           proposal: Optional[dict] = None) -> tuple[float, float, dict]:
    """Return adaptive edge/urgency floors for risk-increasing trades.

    The baseline floors are intentionally low because confidence-aware sizing
    already shrinks weak entries. This multiplier raises the bar in costly or
    fragile environments and lowers it in cleaner snap-back conditions.
    """
    base_edge = float(getattr(settings, "min_risk_increase_edge_pct", 0.0) or 0.0)
    base_urg = float(getattr(settings, "min_risk_increase_urgency", 0.0) or 0.0)
    if not bool(getattr(settings, "adaptive_signal_floor_enabled", True)):
        return base_edge, base_urg, {"mult": 1.0, "mode": "fixed"}

    proposal = proposal or {}
    ecosystem = proposal.get("ecosystem") or {}
    phase = str(ecosystem.get("phase") or "").lower()
    phase_mult = {
        "scavenger": 0.72,
        "decomposer": 0.82,
        "producer": 0.90,
        "churn": 1.00,
        "exhaustion": 1.05,
        "predator": 1.45,
    }.get(phase, 1.00)

    ob = market.get("orderbook", {}) or {}
    spread_bps = ob.get("spread_bps")
    try:
        spread_bps = float(spread_bps)
    except (TypeError, ValueError):
        spread_bps = None
    # Cheap book: relax a bit. Wide book: require more signal to overcome
    # crossing/missed-fill risk. Capped to avoid turning this into a kill switch.
    if spread_bps is None:
        spread_mult = 1.00
    elif spread_bps <= 10:
        spread_mult = 0.88
    else:
        spread_mult = min(1.35, 0.88 + spread_bps / 140.0)

    atr_pct = (market.get("features", {}) or {}).get("atr_pct")
    try:
        atr_pct = float(atr_pct)
    except (TypeError, ValueError):
        atr_pct = None
    # More volatility can improve opportunity, but it also means a weak signal
    # is easier to overtrade. Keep this as a mild adjustment.
    if atr_pct is None:
        vol_mult = 1.00
    elif atr_pct < 0.06:
        vol_mult = 0.90
    elif atr_pct > 0.35:
        vol_mult = 1.18
    else:
        vol_mult = 1.00

    lev = float(sizing.get("target_leverage") or 0.0)
    lev_mult = 0.90 + min(0.35, max(0.0, lev) * 0.18)

    conf_scale = sizing.get("mpc_confidence_scale")
    try:
        conf_scale = float(conf_scale)
    except (TypeError, ValueError):
        conf_scale = None
    # If confidence sizing has already crushed the target, the hard floor can
    # relax; the position is already a probe. Full-size targets keep full bar.
    if conf_scale is None:
        conf_mult = 1.00
    else:
        conf_mult = 0.68 + 0.32 * min(1.0, max(0.0, conf_scale)) ** 0.5

    mult = phase_mult * spread_mult * vol_mult * lev_mult * conf_mult
    lo = float(getattr(settings, "adaptive_signal_floor_min_mult", 0.55) or 0.55)
    hi = float(getattr(settings, "adaptive_signal_floor_max_mult", 2.20) or 2.20)
    mult = max(lo, min(hi, mult))
    detail = {
        "mode": "adaptive",
        "mult": round(mult, 3),
        "phase": phase or "unknown",
        "phase_mult": round(phase_mult, 3),
        "spread_bps": spread_bps,
        "spread_mult": round(spread_mult, 3),
        "atr_pct": atr_pct,
        "vol_mult": round(vol_mult, 3),
        "lev_mult": round(lev_mult, 3),
        "confidence_mult": round(conf_mult, 3),
    }
    return base_edge * mult, base_urg * mult, detail


def _limit_price(settings: Settings, market: dict, delta: float) -> Optional[float]:
    ob = market.get("orderbook", {}) or {}
    best_bid = ob.get("best_bid")
    best_ask = ob.get("best_ask")
    mid = ob.get("mid") or market.get("price")
    if mid is None:
        return None
    off = settings.limit_offset_bps / 10_000.0
    if delta > 0:  # buying -> pay up toward/above ask for a fast fill
        ref = best_ask or mid
        return round(ref * (1 + off), 6)
    elif delta < 0:  # selling -> hit toward/below bid
        ref = best_bid or mid
        return round(ref * (1 - off), 6)
    return round(mid, 6)


def run_checks(settings: Settings, account: dict, market: dict, sizing: dict,
               day_start_equity: Optional[float], max_equity: Optional[float],
               proposal: Optional[dict] = None) -> dict:
    """Return {checks:[...], blocks:[...], warnings:[...], allow:bool}."""
    checks = []
    blocks = []
    warnings = []

    def add(name, status, detail):
        checks.append({"name": name, "status": status, "detail": detail})
        if status == "block":
            blocks.append(f"{name}: {detail}")
        elif status == "warn":
            warnings.append(f"{name}: {detail}")

    equity = account.get("equity") or 0.0
    price = market.get("price") or 0.0
    spread_bps = (market.get("orderbook", {}) or {}).get("spread_bps")
    atr = (market.get("features", {}) or {}).get("atr_pct")
    target_lev = sizing.get("target_leverage") or 0.0

    # ---- HARD BLOCKS (capital protection; never prompt) ----
    if equity < settings.min_account_equity_usd:
        add("account_balance", "block",
            f"Equity ${equity:,.2f} below floor ${settings.min_account_equity_usd:,.2f}")
    else:
        add("account_balance", "ok", f"Equity ${equity:,.2f}")

    if target_lev > settings.max_leverage + 1e-6:
        add("leverage", "block",
            f"Target {target_lev:.2f}x exceeds cap {settings.max_leverage:.2f}x")
    else:
        add("leverage", "ok", f"Target {target_lev:.2f}x <= cap {settings.max_leverage:.2f}x")

    deployable, reserved = deployable_equity(settings, equity)
    max_notional = max_deployable_notional(settings, deployable)
    if (sizing.get("target_notional") or 0.0) > max_notional + 1.0:
        add("notional", "block",
            f"Notional ${sizing.get('target_notional'):,.0f} > cap ${max_notional:,.0f}")
    else:
        add("notional", "ok", f"Notional ${sizing.get('target_notional', 0):,.0f}")
    if reserved > 0:
        add("harvest_reserve", "ok",
            f"Protected ${reserved:,.2f}; deployable equity ${deployable:,.2f}")

    if increases_risk(account, sizing):
        edge = float((proposal or {}).get("expected_edge_pct") or 0.0)
        urgency = float((proposal or {}).get("urgency") or 0.0)
        min_edge, min_urg, floor_detail = adaptive_signal_floors(
            settings, market, sizing, proposal)
        floor_note = (
            f"adaptive x{floor_detail.get('mult', 1.0):.2f}, "
            f"phase {floor_detail.get('phase', 'unknown')}"
            if floor_detail.get("mode") == "adaptive" else "fixed"
        )
        edge_weak = min_edge > 0 and edge < min_edge
        urgency_weak = min_urg > 0 and urgency < min_urg
        if edge_weak and urgency_weak:
            add("signal_floor", "block",
                f"Risk increase signal weak: edge {edge:.2f}% < {min_edge:.2f}% "
                f"and urgency {urgency:.3f} < {min_urg:.3f} ({floor_note})")
        elif edge_weak:
            add("edge_floor", "warn",
                f"Risk increase edge {edge:.2f}% < floor {min_edge:.2f}% ({floor_note})")
        elif min_edge > 0:
            add("edge_floor", "ok",
                f"Risk increase edge {edge:.2f}% >= floor {min_edge:.2f}% ({floor_note})")
        if urgency_weak and not edge_weak:
            add("urgency_floor", "warn",
                f"Risk increase urgency {urgency:.3f} < floor {min_urg:.3f} ({floor_note})")
        elif min_urg > 0 and not (edge_weak and urgency_weak):
            add("urgency_floor", "ok",
                f"Risk increase urgency {urgency:.3f} >= floor {min_urg:.3f} ({floor_note})")
    else:
        add("signal_floor", "ok", "not increasing exposure")

    # daily loss circuit breaker
    if settings.daily_loss_limit_enabled and day_start_equity:
        limit = settings.daily_loss_limit_usd or (day_start_equity * 0.25)
        day_pnl = equity - day_start_equity
        if day_pnl <= -abs(limit):
            add("daily_loss", "block",
                f"Daily P&L ${day_pnl:,.2f} hit limit -${abs(limit):,.0f}")
        else:
            add("daily_loss", "ok", f"Daily P&L ${day_pnl:,.2f}")
    else:
        add("daily_loss", "ok", "disabled")

    # liquidation buffer (only blocks opening/adding, not reducing/closing)
    liq = account.get("liquidation_risk", "low")
    if liq == "critical":
        add("liquidation_risk", "block", "Maintenance margin critically high")
    else:
        add("liquidation_risk", "ok" if liq == "low" else "warn",
            f"Liquidation risk: {liq}")

    # market data sanity
    if price <= 0:
        add("market_data", "block", "No valid price")
    else:
        add("market_data", "ok", f"Price ${price:,.4f}")

    ob = market.get("orderbook", {}) or {}
    if not ob.get("best_bid") or not ob.get("best_ask"):
        add("orderbook", "block", "Empty orderbook")
    else:
        add("orderbook", "ok",
            f"Depth {ob.get('depth_total', 0):,.0f} contracts")

    # ---- WARN ONLY (do not block trading) ----
    if spread_bps is not None and spread_bps > settings.max_spread_bps:
        add("spread", "warn", f"Spread {spread_bps:.0f}bps > {settings.max_spread_bps:.0f}bps")
    else:
        add("spread", "ok", f"Spread {spread_bps:.0f}bps" if spread_bps else "n/a")

    if atr is not None and atr > settings.strategy_params.get("max_atr_pct", 5.0):
        add("volatility", "warn", f"ATR {atr:.2f}% elevated")
    else:
        add("volatility", "ok", f"ATR {atr:.2f}%" if atr is not None else "n/a")

    funding = market.get("funding") or {}
    add("funding", "ok",
        f"{funding.get('funding_rate', funding.get('rate', 'n/a'))}")

    add("margin", "ok" if account.get("margin_safety") != "danger" else "warn",
        f"Margin safety: {account.get('margin_safety', 'n/a')}")
    add("open_orders", "ok", f"{account.get('open_orders_count', 0)} resting")
    add("position_exposure", "ok",
        f"{account.get('position_contracts', 0)} contracts, "
        f"{account.get('effective_leverage', 0):.2f}x")

    allow = len(blocks) == 0
    return {"checks": checks, "blocks": blocks, "warnings": warnings, "allow": allow}
