"""Smoke-test risk sizing guardrails."""
from backend import risk
from backend.config import Settings
from backend.store import STORE


def main():
    old_capture = STORE.forager_captured_cumulative
    try:
        settings = Settings(
            harvest_reserve_enabled=True,
            harvest_reserve_fraction=1.0,
            max_leverage=5.0,
            leverage_target=5.0,
            position_scale=1.0,
            max_position_notional_usd=0.0,
            mpc_confidence_sizing_enabled=False,
        )
        account = {"equity": 100.0, "position_contracts": 0.0}
        market = {"price": 10.0, "orderbook": {"mid": 10.0}}
        proposal = {"target_fraction": 0.5, "confidence": 0.5}

        STORE.forager_captured_cumulative = 0.0
        base = risk.size_position(settings, account, market, proposal)

        STORE.forager_captured_cumulative = 20.0
        reserved = risk.size_position(settings, account, market, proposal)
        checks = risk.run_checks(settings, account, market, reserved, 100.0, 100.0)

        print("harvest reserve sizing:")
        print(f"  base target_notional    : {base['target_notional']}")
        print(f"  reserved target_notional: {reserved['target_notional']}")
        print(f"  deployable equity       : {reserved['deployable_equity']}")
        print(f"  harvest reserve         : {reserved['harvest_reserved_usd']}")

        assert base["target_notional"] == 250.0, base
        assert reserved["target_notional"] == 200.0, reserved
        assert reserved["deployable_equity"] == 80.0, reserved
        assert reserved["harvest_reserved_usd"] == 20.0, reserved
        assert any(c["name"] == "harvest_reserve" for c in checks["checks"]), checks

        # A lower same-side MPC target is a REDUCE, not ADD. This mirrors the
        # UI failure mode where robust MPC was aiming at +114 contracts while
        # the live account was already larger on the same side.
        floor_settings = Settings(
            harvest_reserve_enabled=False,
            max_leverage=5.0,
            leverage_target=5.0,
            position_scale=1.0,
            min_risk_increase_edge_pct=0.06,
            min_risk_increase_urgency=0.03,
        )
        reduce_account = {"equity": 1000.0, "position_contracts": 205.0}
        reduce_market = {
            "price": 6.5783,
            "orderbook": {
                "mid": 6.5783,
                "best_bid": 6.57,
                "best_ask": 6.59,
                "spread_bps": 30.4,
            },
        }
        reduce_sizing = {"target_contracts": 114.0, "delta_contracts": -91.0,
                         "target_notional": 750.22, "target_leverage": 0.75}
        reduce_proposal = {"expected_edge_pct": 0.15, "urgency": 0.154}
        reduce_checks = risk.run_checks(
            floor_settings, reduce_account, reduce_market, reduce_sizing,
            1000.0, 1000.0, proposal=reduce_proposal)

        assert risk.effective_action_from_sizing(reduce_account, reduce_sizing) == "REDUCE"
        assert risk.increases_risk(reduce_account, reduce_sizing) is False
        assert reduce_checks["allow"], reduce_checks
        assert any(c["name"] == "signal_floor" and c["status"] == "ok"
                   for c in reduce_checks["checks"]), reduce_checks

        # Marginal live-scale signals should be allowed; confidence sizing is
        # already shrinking the target, so the floor only blocks dead signals.
        weak_account = {"equity": 1000.0, "position_contracts": 0.0}
        weak_sizing = {"target_contracts": 114.0, "delta_contracts": 114.0,
                       "target_notional": 750.22, "target_leverage": 0.75}
        marginal_checks = risk.run_checks(
            floor_settings, weak_account, reduce_market, weak_sizing,
            1000.0, 1000.0, proposal={"expected_edge_pct": 0.08, "urgency": 0.041})

        assert risk.effective_action_from_sizing(weak_account, weak_sizing) == "ENTER_LONG"
        assert risk.increases_risk(weak_account, weak_sizing) is True
        assert marginal_checks["allow"], marginal_checks

        weak_checks = risk.run_checks(
            floor_settings, weak_account, reduce_market, weak_sizing,
            1000.0, 1000.0, proposal={"expected_edge_pct": 0.05, "urgency": 0.006})

        assert not weak_checks["allow"], weak_checks
        assert any("signal_floor" in b for b in weak_checks["blocks"]), weak_checks

        adaptive_market = {
            "price": 6.5783,
            "orderbook": {"mid": 6.5783, "best_bid": 6.57, "best_ask": 6.58,
                          "spread_bps": 8.0},
            "features": {"atr_pct": 0.08},
        }
        adaptive_sizing = dict(weak_sizing, target_leverage=0.15,
                               mpc_confidence_scale=0.08)
        scavenger_checks = risk.run_checks(
            floor_settings, weak_account, adaptive_market, adaptive_sizing,
            1000.0, 1000.0,
            proposal={"expected_edge_pct": 0.045, "urgency": 0.021,
                      "ecosystem": {"phase": "scavenger"}})
        assert scavenger_checks["allow"], scavenger_checks

        predator_market = {
            "price": 6.5783,
            "orderbook": {"mid": 6.5783, "best_bid": 6.50, "best_ask": 6.65,
                          "spread_bps": 220.0},
            "features": {"atr_pct": 0.42},
        }
        predator_checks = risk.run_checks(
            floor_settings, weak_account, predator_market,
            dict(weak_sizing, target_leverage=1.2, mpc_confidence_scale=1.0),
            1000.0, 1000.0,
            proposal={"expected_edge_pct": 0.07, "urgency": 0.035,
                      "ecosystem": {"phase": "predator"}})
        assert not predator_checks["allow"], predator_checks
        assert any("signal_floor" in b for b in predator_checks["blocks"]), predator_checks

        # Low confidence should shrink the displayed MPC target itself, not
        # merely rely on a later risk block. The screenshot had ~7.8%
        # confidence and a raw +0.078 target_fraction, which should be a
        # dust-size probe at most.
        low_conf_settings = Settings(
            harvest_reserve_enabled=False,
            max_leverage=5.8,
            leverage_target=5.8,
            position_scale=1.0,
            mpc_confidence_sizing_enabled=True,
            mpc_confidence_full_at=0.55,
            mpc_confidence_power=1.5,
            mpc_min_confidence_scale=0.0,
        )
        low_conf_account = {"equity": 510.0, "position_contracts": 0.0}
        low_conf_market = {
            "price": 6.5866,
            "orderbook": {"mid": 6.5866, "best_bid": 6.58, "best_ask": 6.59},
        }
        low_conf_proposal = {
            "target_fraction": 0.078,
            "confidence": 0.0779,
        }
        low_conf_sizing = risk.size_position(
            low_conf_settings, low_conf_account, low_conf_market, low_conf_proposal)

        assert low_conf_sizing["target_contracts_before_confidence"] > 30.0, low_conf_sizing
        assert low_conf_sizing["mpc_confidence_scale"] < 0.06, low_conf_sizing
        assert low_conf_sizing["target_contracts"] <= 2.0, low_conf_sizing

        # Reductions remain available; confidence damping should not trap risk.
        reduce_low_conf_account = {"equity": 1000.0, "position_contracts": 205.0}
        reduce_low_conf_proposal = {"target_fraction": 0.15, "confidence": 0.05}
        reduce_low_conf_sizing = risk.size_position(
            low_conf_settings, reduce_low_conf_account, reduce_market, reduce_low_conf_proposal)
        assert reduce_low_conf_sizing["target_contracts"] > 100.0, reduce_low_conf_sizing
        assert reduce_low_conf_sizing["delta_contracts"] < 0.0, reduce_low_conf_sizing
        print("\nOK")
    finally:
        STORE.forager_captured_cumulative = old_capture


if __name__ == "__main__":
    main()
