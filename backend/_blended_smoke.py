"""Smoke-test the directional alpha components."""
from backend.signals_blended import (
    spot_lead_tilt, funding_fade_tilt, oi_pressure_tilt, visual_trend_tilt,
    compute_components, blend_with_mpc, conditioned_component_weights, DEFAULT_BLEND,
)

# 1) Spot lead — synthetic price walks
import math, random
random.seed(1)
spot_flat = [100.0 + random.gauss(0, 0.05) for _ in range(120)]
spot_uptrend = [100.0 * (1 + i * 0.001) + random.gauss(0, 0.05) for i in range(120)]
spot_downtrend = [100.0 * (1 - i * 0.001) + random.gauss(0, 0.05) for i in range(120)]
print("spot_lead_tilt:")
print(f"  flat   : {spot_lead_tilt(spot_flat):.4f}  (expect ~0)")
print(f"  uptrend: {spot_lead_tilt(spot_uptrend):+.4f}  (expect positive)")
print(f"  downtr.: {spot_lead_tilt(spot_downtrend):+.4f}  (expect negative)")

# 2) Funding fade
funding_neutral = [random.gauss(1e-5, 1e-5) for _ in range(80)]
funding_long_crowded = funding_neutral[:-1] + [5e-5]   # spike positive
funding_short_crowded = funding_neutral[:-1] + [-5e-5]
print("\nfunding_fade_tilt:")
print(f"  neutral         : {funding_fade_tilt(funding_neutral):+.4f}")
print(f"  long crowded    : {funding_fade_tilt(funding_long_crowded):+.4f}  (expect NEGATIVE — fade)")
print(f"  short crowded   : {funding_fade_tilt(funding_short_crowded):+.4f}  (expect POSITIVE — fade)")

# 3) OI pressure
prices = [100.0 + i * 0.1 for i in range(120)]   # mild uptrend
oi_rising = [1000.0 + i * 2 for i in range(120)]   # OI rising
oi_falling = [1200.0 - i * 2 for i in range(120)]  # OI falling
print("\noi_pressure_tilt (price uptrend):")
print(f"  oi rising  : {oi_pressure_tilt(prices, oi_rising):+.4f}  (price+ oi+ -> bullish)")
print(f"  oi falling : {oi_pressure_tilt(prices, oi_falling):+.4f}  (price+ oi-  -> still bullish [short covering])")
prices_down = [100.0 - i * 0.1 for i in range(120)]
print("\noi_pressure_tilt (price downtrend):")
print(f"  oi rising  : {oi_pressure_tilt(prices_down, oi_rising):+.4f}  (price- oi+ -> bearish)")
print(f"  oi falling : {oi_pressure_tilt(prices_down, oi_falling):+.4f}  (price- oi- -> bearish)")

# 4) Visual-review trend tilt
print("\nvisual_trend_tilt (direction x concern-conviction):")
print(f"  up/OK       : {visual_trend_tilt('up', 'OK'):+.4f}  (expect +0.34)")
print(f"  up/CAUTION  : {visual_trend_tilt('up', 'CAUTION'):+.4f}  (expect +0.67)")
print(f"  down/STOP   : {visual_trend_tilt('down', 'STOP'):+.4f}  (expect -1.00)")
print(f"  sideways/OK : {visual_trend_tilt('sideways', 'OK'):+.4f}  (expect 0)")
print(f"  choppy/STOP : {visual_trend_tilt('choppy', 'STOP'):+.4f}  (expect 0 — no direction)")
print(f"  missing     : {visual_trend_tilt(None, None):+.4f}  (expect 0)")

# 5) compute_components includes visual_trend when a review is passed
comps_vr = compute_components(
    kalshi_closes=[100.0] * 80,
    visual_review={"trend": "down", "concern": "STOP"})
print(f"\ncompute_components visual_trend (down/STOP): "
      f"{comps_vr['visual_trend']:+.4f}  (expect -1.00)")
assert abs(comps_vr["visual_trend"] + 1.0) < 1e-9, comps_vr

# 6) Blend including visual
components = {"spot_lead": 1.5, "funding_fade": -0.5,
             "oi_pressure": 0.8, "visual_trend": -1.0}
mpc = -0.4   # MPC says fade (mean revert), modest signal
blended, parts = blend_with_mpc(mpc, components)
print(f"\nblend: mpc=-0.4 + spot=+1.5 + funding=-0.5 + oi=+0.8 + visual=-1.0")
print(f"  blended={blended:.4f}")
print(f"  parts={parts}")
assert "visual_trend" in parts, parts

# 7) Ecology-conditioned spot lead: keep it mostly predator-only.
base_w = {"spot_lead": 0.30, "funding_fade": 0.0, "oi_pressure": 0.0,
          "ecology_flow": 0.0, "visual_trend": 0.0}
pred_w = conditioned_component_weights(
    base_w,
    {"phase": "predator", "drivers": {"disturbance": 0.8},
     "organisms": {"scores": {"predator": 0.8}}},
)["spot_lead"]
scav_w = conditioned_component_weights(
    base_w,
    {"phase": "scavenger", "drivers": {"disturbance": 0.5},
     "organisms": {"scores": {"scavenger": 1.0}}},
)["spot_lead"]
churn_w = conditioned_component_weights(base_w, {"phase": "churn"})["spot_lead"]
print("\necology-conditioned spot lead weights:")
print(f"  predator : {pred_w:.4f}")
print(f"  scavenger: {scav_w:.4f}")
print(f"  churn    : {churn_w:.4f}")
assert pred_w > 0.25, pred_w
assert scav_w <= 0.03, scav_w
assert churn_w <= 0.03, churn_w
print("\nOK")
