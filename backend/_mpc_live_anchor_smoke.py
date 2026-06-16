"""Smoke-test that live MPC targets are anchored to the real account position."""
from backend import signals


def _synthetic_candles():
    out = []
    px = 100.0
    for i in range(80):
        if i >= 78:
            px *= 1.003
        elif i % 2 == 0:
            px *= 1.00005
        else:
            px *= 0.99995
        out.append({"open": px, "high": px * 1.0005,
                    "low": px * 0.9995, "close": px})
    return out


def main():
    params = {
        "vol_win": 12,
        "lookback": 2,
        "regime_win": 8,
        "beta": 0.18,
        "gain": 1.0,
        "band": 0.5,
        "er_cap": 1.0,
        "blend_enabled": False,
    }
    ecosystem = {
        "phase": "decomposer",
        "drivers": {"disturbance": 0.32},
        "network_metrics": {"rel_reserve": 0.91},
        "organisms": {"scores": {"immune": 0.0}},
    }
    candles = _synthetic_candles()

    pos_replay, _urg_replay, replay = signals.robust_mpc_with_diagnostics(
        candles, params, ecosystem=ecosystem)
    pos_live, _urg_live, live = signals.robust_mpc_with_diagnostics(
        candles, params, ecosystem=ecosystem, live_current_fraction=0.0)

    print("replay target:", pos_replay[-1], replay)
    print("live target  :", pos_live[-1], live)

    assert replay["live_anchor_applied"] is False, replay
    assert live["live_anchor_applied"] is True, live
    assert abs(live["target_before"]) < 1e-9, live
    if abs(live["aim_capped"]) <= live["band"]:
        assert abs(pos_live[-1]) < 1e-9, live
    print("\nOK")


if __name__ == "__main__":
    main()
