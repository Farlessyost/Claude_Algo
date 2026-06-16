"""Fast integration smoke: CSD governor zeros a nonzero MPC target."""
from __future__ import annotations

from backend import lab, strategy
from backend.config import DEFAULT_STRATEGY_PARAMS


def _params() -> dict:
    p = dict(DEFAULT_STRATEGY_PARAMS)
    p.update({
        "vol_win": 12,
        "lookback": 2,
        "beta": 0.25,
        "k": 1.2,
        "z_cap": 3.5,
        "deadband_bps": 1.0,
        "regime_win": 8,
        "er_cap": 1.0,
        "gain": 1.0,
        "band": 0.5,
    })
    return p


def _find_nonzero_slice(bars, params):
    for end in range(min(len(bars), 260), len(bars) + 1, 25):
        sample = bars[max(0, end - 240):end]
        prop = strategy.evaluate(
            sample,
            params,
            position_contracts=0.0,
            variant="mpc",
            csd_state={"risk": 0.0, "threshold": 0.95, "enabled": False},
        )
        if abs(float(prop.get("target_fraction_base") or 0.0)) > 0.001:
            return sample, prop
    raise AssertionError("could not find a nonzero MPC target slice")


def main() -> None:
    rows = lab.load(use_cache=True)
    bars = lab.aggregate(rows, 3)
    print(f"loaded {len(bars):,} 3m bars")
    params = _params()
    sample, base = _find_nonzero_slice(bars, params)
    print(f"base target: {base['target_fraction_base']:+.4f}")

    forced = strategy.evaluate(
        sample,
        params,
        position_contracts=0.0,
        variant="mpc",
        csd_state={"risk": 1.0, "threshold": 0.95, "enabled": True},
    )
    assert forced["csd_gated"] is True, forced
    assert abs(float(forced["target_fraction"])) < 1e-9, forced
    print("forced high-risk CSD gate: OK")

    disabled = strategy.evaluate(
        sample,
        params,
        position_contracts=0.0,
        variant="mpc",
        csd_state={"risk": 1.0, "threshold": 0.95, "enabled": False},
    )
    assert disabled["csd_gated"] is False, disabled
    assert abs(float(disabled["target_fraction_base"]) - float(base["target_fraction_base"])) < 1e-9
    print("disabled governor leaves target intact: OK")
    print("\nOK")


if __name__ == "__main__":
    main()
