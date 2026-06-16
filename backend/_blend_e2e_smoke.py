"""Smoke-test the end-to-end blend wiring through signals -> strategy."""
from backend import lab, signals, strategy
from backend.config import DEFAULT_STRATEGY_PARAMS

rows = lab.load(use_cache=True)
bars = lab.aggregate(rows, 3)
print(f"loaded {len(bars):,} 3m bars")

# Build params with blend enabled
params = dict(DEFAULT_STRATEGY_PARAMS)
params.update({"vol_win": 12, "lookback": 2, "beta": 0.25, "k": 1.2,
                "z_cap": 3.5, "deadband_bps": 1.0, "regime_win": 8,
                "er_cap": 1.0, "gain": 1.0, "band": 0.5})
params["blend_enabled"] = True
params["blend_w_mpc"] = 1.0
params["blend_w_spot_lead"] = 0.30

# Simulate spot history from the actual aligned data (use ML dataset)
from backend.backtest_blended import load_spot_aligned_to_kalshi
spot_aligned = load_spot_aligned_to_kalshi(bars)
spot_closes_full = [v for v in spot_aligned if v is not None]
print(f"spot history length: {len(spot_closes_full):,}")

blend_context = {
    "spot_closes": spot_closes_full[-240:],   # last 240 bars
    "funding_history": [],
    "oi_history": [],
    "oi_price_history": [],
}

# Test 1: MPC alone (no blend)
params_noblend = dict(params); params_noblend["blend_enabled"] = False
pos_a, urg_a = signals.mpc_with_aux(bars[-200:], params_noblend, blend_context=None)
print(f"\nMPC alone (last bar):  pos={pos_a[-1]:+.4f}  urgency={urg_a[-1]:.4f}")

# Test 2: Blended (just last bar should be blended)
pos_b, urg_b = signals.mpc_with_aux(bars[-200:], params, blend_context=blend_context)
print(f"MPC + blend (last bar): pos={pos_b[-1]:+.4f}  urgency={urg_b[-1]:.4f}")
print(f"Historical pos same (should be):  {pos_a[-2]:+.4f}  vs  {pos_b[-2]:+.4f}")

# Test 3: strategy.evaluate with blend_context
prop = strategy.evaluate(
    bars[-200:], params, position_contracts=0.0,
    variant="mpc", market=None,
    blend_context=blend_context)
print(f"\nstrategy.evaluate MPC + blend:")
print(f"  action: {prop['action']}")
print(f"  target_fraction: {prop['target_fraction']}")
print(f"  target_fraction_base: {prop['target_fraction_base']}")
print(f"  urgency: {prop['urgency']}")

# Test 4: same call with blend_context=None should fall back
prop2 = strategy.evaluate(
    bars[-200:], params_noblend, position_contracts=0.0,
    variant="mpc", market=None,
    blend_context=None)
print(f"\nstrategy.evaluate MPC alone:")
print(f"  action: {prop2['action']}")
print(f"  target_fraction: {prop2['target_fraction']}")

# Test 5: visual-review trend as a directional contributor. A strong DOWN/STOP
# read with a heavy weight should push the last-bar aim more negative than the
# same blend without it.
params_vis = dict(params)
params_vis["blend_w_visual"] = 1.0   # exaggerated so the effect is unmistakable
ctx_no_vis = dict(blend_context)
ctx_vis = dict(blend_context)
ctx_vis["visual_review"] = {"trend": "down", "concern": "STOP", "last_ts": 1.0}
pos_c, _ = signals.mpc_with_aux(bars[-200:], params_vis, blend_context=ctx_no_vis)
pos_d, _ = signals.mpc_with_aux(bars[-200:], params_vis, blend_context=ctx_vis)
print(f"\nvisual trend (down/STOP, w=1.0):")
print(f"  without visual (last bar): pos={pos_c[-1]:+.4f}")
print(f"  with visual    (last bar): pos={pos_d[-1]:+.4f}  (expect <= without)")
assert pos_d[-1] <= pos_c[-1] + 1e-9, (pos_c[-1], pos_d[-1])
print("\nOK")
