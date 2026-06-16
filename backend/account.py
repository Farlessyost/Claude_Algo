"""Assembles the real margin-account view from balance + positions + orders."""
from __future__ import annotations

from typing import Optional

from .kalshi_client import KalshiClient, to_float


def build_account_view(client: KalshiClient, ticker: str,
                       price: Optional[float]) -> dict:
    bal = client.get_balance()
    subs = bal.get("subaccount_balances", []) or []
    primary = next((s for s in subs if s.get("subaccount") == 0),
                   subs[0] if subs else {})

    equity = to_float(primary.get("account_equity"))
    available = to_float(primary.get("available_balance"))
    initial_margin = to_float(primary.get("initial_margin"))
    maintenance_margin = to_float(primary.get("maintenance_margin"))
    resting_margin = to_float(primary.get("resting_orders_margin"))
    position_value = to_float(primary.get("position_value"))

    pos_resp = client.get_positions(ticker=ticker)
    positions = pos_resp.get("positions", []) or []
    pos = next((p for p in positions if p.get("market_ticker") == ticker),
               positions[0] if positions else {})

    size = to_float(pos.get("position"))          # signed contracts (can lag!)
    entry = to_float(pos.get("entry_price"))
    upnl = to_float(pos.get("unrealized_pnl"))
    margin_used = to_float(pos.get("margin_used"))
    roe = pos.get("roe")

    # The positions endpoint can lag right after a fill while balance updates
    # immediately. balance.position_value is SIGNED (negative = short). Reconcile
    # so leverage is never under-reported and short exposure isn't missed
    # (prevents piling on into a hidden position in either direction).
    pos_notional = abs(size) * (price or entry or 0.0)
    bal_notional = abs(position_value)             # magnitude of exposure
    bal_sign = -1.0 if position_value < 0 else 1.0  # direction from balance
    inconsistent = abs(pos_notional - bal_notional) > max(0.5, 0.1 * bal_notional)
    if abs(size) < 1e-6 and bal_notional > 0.5 and price:
        # positions endpoint lagging — infer size + direction from balance
        size = bal_sign * (bal_notional / price)
        pos_notional = bal_notional
    notional = max(pos_notional, bal_notional)
    eff_leverage = (notional / equity) if equity else 0.0

    orders_resp = client.get_orders(ticker=ticker, status="resting")
    open_orders = orders_resp.get("orders", []) or []

    risk = client.get_risk()

    direction = "FLAT"
    if size > 0:
        direction = "LONG"
    elif size < 0:
        direction = "SHORT"

    liq_risk = "low"
    margin_safety = "healthy"
    if equity:
        mm_ratio = maintenance_margin / equity if equity else 0.0
        if mm_ratio > 0.8:
            liq_risk, margin_safety = "critical", "danger"
        elif mm_ratio > 0.55:
            liq_risk, margin_safety = "elevated", "caution"
        elif mm_ratio > 0.3:
            liq_risk, margin_safety = "moderate", "ok"

    return {
        "equity": equity,
        "available_balance": available,
        "initial_margin": initial_margin,
        "maintenance_margin": maintenance_margin,
        "resting_orders_margin": resting_margin,
        "position_value": position_value,
        "position_contracts": size,
        "position_direction": direction,
        "entry_price": entry,
        "unrealized_pnl": upnl,
        "margin_used": margin_used,
        "roe": roe,
        "notional_exposure": notional,
        "effective_leverage": eff_leverage,
        "open_orders": open_orders,
        "open_orders_count": len(open_orders),
        "margin_safety": margin_safety,
        "liquidation_risk": liq_risk,
        "balance_position_value": bal_notional,
        "position_endpoint_consistent": not inconsistent,
        "risk_raw": risk,
        "refreshed_ts": _now_iso(),
    }


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
