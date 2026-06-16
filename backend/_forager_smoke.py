"""Smoke-test the foraging-cycle layer with hunger presets + churn defense.

Default hunger is RAVENOUS: profit_threshold_R 0.15, edge_decay_trim 0.3,
tier2 0.3, tier3 0.5, decomposer 0.4, edge_gone 0.15, csd_high 0.6,
churn_disturbance 0.4, churn_profit_R 0.1, churn_scale 0.0 (full close),
cooldown base 240 / min 90 / max 600.
"""
import time
from backend import forager
from backend.config import Settings

# Hunger preset resolves + default is ravenous.
assert forager._resolve_params(Settings())["hunger"] == "ravenous"
assert forager._resolve_params(Settings(forager_hunger="lazy"))["base_cooldown"] == 600
# Bite sizes ("bounds for eating") scale with hunger: lazy keeps a runner on
# every tier; ravenous devours (tier3/churn -> full close).
lazy = forager._resolve_params(Settings(forager_hunger="lazy"))
rav = forager._resolve_params(Settings(forager_hunger="ravenous"))
assert lazy["tier3_scale"] == 0.40 and lazy["churn_scale"] == 0.50, lazy
assert rav["tier3_scale"] == 0.0 and rav["churn_scale"] == 0.0, rav
assert rav["tier1_scale"] < lazy["tier1_scale"], "ravenous bites deeper on tier1"
print("preset resolve OK (default ravenous; bite-sizes scale with hunger)")


class FakeStore:
    forager_captured_cumulative = 0.0
    def __init__(self, eco, csd, forager_state=None):
        self.ecosystem = eco
        self.csd_state = csd
        self.forager_state = forager_state or {}
        self.harvests = []
    def record_forager_harvest(self, captured, reason=None, pnl_R=None):
        self.forager_captured_cumulative += captured
        self.harvests.append({"captured": captured, "reason": reason})
    def log(self, *a, **k): pass


def eco(decomposer=0.2, scavenger=0.2, disturbance=0.2, stretch_z=2.0, rel_reserve=0.5):
    return {"organisms": {"scores": {"decomposer": decomposer, "scavenger": scavenger}},
            "drivers": {"disturbance": disturbance, "stretch_z": stretch_z},
            "network_metrics": {"rel_reserve": rel_reserve}}


MKT = {"price": 6.5, "features": {"atr_pct": 0.2}}
def acct(upnl, pos=10.0):
    return {"position_contracts": pos, "price": 6.5, "equity": 100.0, "unrealized_pnl": upnl}
def upnl_for_R(R):  # pnl_R = upnl / (10*6.5*0.2/100) = upnl/0.13
    return R * 0.13

# Base preset behaviour tested with autoscale OFF (isolates the preset);
# autoscale is checked separately at the end.
S = Settings(forager_enabled=True, forager_hunger="ravenous", forager_autoscale=False)

# A) disabled -> no mutation
prop = {"action": "ENTER_LONG", "target_fraction": 0.5, "expected_edge_pct": 0.3}
st = forager.apply(Settings(forager_enabled=False), FakeStore(eco(), {"risk": 0.1}),
                   acct(upnl_for_R(2.0)), MKT, prop)
assert prop["action"] == "ENTER_LONG" and not st["enabled"]
print("A disabled            -> no mutation OK")

# B) tier1 only -> ravenous tier1_scale 0.60 (pnl_R 0.2, edge 0.18 -> edge_decay 0.40, calm)
prop = {"action": "ADD", "target_fraction": 0.11, "expected_edge_pct": 0.18}
st = forager.apply(S, FakeStore(eco(disturbance=0.2), {"risk": 0.1}), acct(upnl_for_R(0.2)), MKT, prop)
print(f"B tier1 (R0.2)        -> {prop['action']} scale={st['harvest_scale']} ({st['harvest_reason']})")
assert prop["action"] == "REDUCE" and st["harvest_scale"] == 0.60

# C) tier3 full close (pnl_R 0.6, edge gone, calm)
prop = {"action": "ADD", "target_fraction": 0.11, "expected_edge_pct": 0.05}
st = forager.apply(S, FakeStore(eco(disturbance=0.2), {"risk": 0.1}), acct(upnl_for_R(0.6)), MKT, prop)
print(f"C tier3 (R0.6)        -> {prop['action']} scale={st['harvest_scale']}")
assert prop["action"] == "CLOSE" and st["harvest_scale"] == 0.0

# D) CHURN full close (small profit + disturbance, no other tier fires)
prop = {"action": "ADD", "target_fraction": 0.11, "expected_edge_pct": 0.25}
storeD = FakeStore(eco(disturbance=0.6, decomposer=0.2), {"risk": 0.1})
st = forager.apply(S, storeD, acct(upnl_for_R(0.2)), MKT, prop)
print(f"D CHURN (R0.2,dist.6) -> {prop['action']} scale={st['harvest_scale']} ({st['harvest_reason']})")
assert prop["action"] == "CLOSE" and "churn" in (st["harvest_reason"] or "")
assert storeD.forager_captured_cumulative > 0, "captured profit should be recorded"
print(f"   captured recorded: ${storeD.forager_captured_cumulative:.4f}")

# E) CSD rising -> close
prop = {"action": "ADD", "target_fraction": 0.11, "expected_edge_pct": 0.25}
storeE = FakeStore(eco(disturbance=0.2), {"risk": 0.9}, {"csd_prev": 0.5, "reserve_prev": 0.5})
st = forager.apply(S, storeE, acct(upnl_for_R(0.2)), MKT, prop)
print(f"E csd-rising          -> {prop['action']} ({st['harvest_reason']})")
assert prop["action"] == "CLOSE" and st["csd_rising"]

# F) below all bars -> untouched (pnl_R 0.05)
prop = {"action": "ADD", "target_fraction": 0.11, "expected_edge_pct": 0.25}
st = forager.apply(S, FakeStore(eco(disturbance=0.2), {"risk": 0.1}), acct(upnl_for_R(0.05)), MKT, prop)
print(f"F below-bars          -> {prop['action']} scale={st['harvest_scale']}")
assert prop["action"] == "ADD" and st["harvest_scale"] == 1.0

# G) refractory blocks a new entry
future = time.time() + 600
prop = {"action": "ENTER_LONG", "target_fraction": 0.5, "expected_edge_pct": 0.05}
storeG = FakeStore(eco(disturbance=0.2, scavenger=0.2, stretch_z=2.0), {"risk": 0.1},
                   {"cooldown_until": future, "csd_prev": 0.1, "reserve_prev": 0.5})
st = forager.apply(S, storeG, acct(0.0, pos=0.0), MKT, prop)
print(f"G refractory block    -> {prop['action']} blocked={st['blocked_entry']}")
assert prop["action"] == "HOLD" and st["blocked_entry"]

# H) close allowed in cooldown
prop = {"action": "CLOSE", "target_fraction": 0.0, "expected_edge_pct": 0.05}
storeH = FakeStore(eco(disturbance=0.2, stretch_z=2.0), {"risk": 0.1},
                   {"cooldown_until": future, "csd_prev": 0.1, "reserve_prev": 0.5})
st = forager.apply(S, storeH, acct(0.0, pos=5.0), MKT, prop)
print(f"H close in cooldown   -> {prop['action']} blocked={st['blocked_entry']}")
assert prop["action"] == "CLOSE" and not st["blocked_entry"]

# I) cooldown ends early on disturbance
prop = {"action": "ENTER_LONG", "target_fraction": 0.5, "expected_edge_pct": 0.05}
storeI = FakeStore(eco(disturbance=0.9, stretch_z=2.0), {"risk": 0.1},
                   {"cooldown_until": future, "csd_prev": 0.1, "reserve_prev": 0.5})
st = forager.apply(S, storeI, acct(0.0, pos=0.0), MKT, prop)
print(f"I disturbance reset    -> {prop['action']} reentry={st['reentry_reason']}")
assert prop["action"] == "ENTER_LONG" and not st["in_cooldown"]

# K) MIN-REST FLOOR (the reported bug): within the rest period, persistently
#    high scavenger does NOT clear the cooldown -> entry STILL blocked.
nowt = time.time()
prop = {"action": "ENTER_LONG", "target_fraction": 0.5, "expected_edge_pct": 0.05}
storeK = FakeStore(eco(scavenger=1.0, disturbance=0.3, stretch_z=2.0), {"risk": 0.1},
                   {"cooldown_until": nowt + 600, "cooldown_started_at": nowt - 10,
                    "cooldown_seconds": 600, "csd_prev": 0.1, "reserve_prev": 0.5})
st = forager.apply(S, storeK, acct(0.0, pos=0.0), MKT, prop)
print(f"K min-rest holds       -> {prop['action']} blocked={st['blocked_entry']} (scavenger 1.0, 10s into 600s rest)")
assert prop["action"] == "HOLD" and st["blocked_entry"]

# L) past the min-rest, a strong signal DOES clear it -> entry allowed
prop = {"action": "ENTER_LONG", "target_fraction": 0.5, "expected_edge_pct": 0.05}
storeL = FakeStore(eco(scavenger=1.0, disturbance=0.3, stretch_z=2.0), {"risk": 0.1},
                   {"cooldown_until": nowt + 600, "cooldown_started_at": nowt - 400,
                    "cooldown_seconds": 600, "csd_prev": 0.1, "reserve_prev": 0.5})
st = forager.apply(S, storeL, acct(0.0, pos=0.0), MKT, prop)
print(f"L past min-rest        -> {prop['action']} reentry={st['reentry_reason']}")
assert prop["action"] == "ENTER_LONG" and not st["in_cooldown"]

# M) fixed-override cooldown is a HARD timer: blocks even past 60% with a strong
#    signal (no early exit at all).
prop = {"action": "ENTER_LONG", "target_fraction": 0.5, "expected_edge_pct": 0.9}
Sov = Settings(forager_enabled=True, forager_hunger="ravenous",
               forager_autoscale=False, forager_cooldown_seconds=120)
storeM = FakeStore(eco(scavenger=1.0, disturbance=0.9, stretch_z=0.1), {"risk": 0.1},
                   {"cooldown_until": nowt + 120, "cooldown_started_at": nowt - 100,
                    "cooldown_seconds": 120, "csd_prev": 0.1, "reserve_prev": 0.5})
st = forager.apply(Sov, storeM, acct(0.0, pos=0.0), MKT, prop)
print(f"M fixed-override hard  -> {prop['action']} blocked={st['blocked_entry']} (100s into 120s, no early exit)")
assert prop["action"] == "HOLD" and st["blocked_entry"]

# J) AUTOSCALE: a brittle/disturbed/high-CSD regime raises eagerness and lowers
#    the effective profit bar vs a calm one.
calm = forager._autoscale(forager._resolve_params(Settings(forager_hunger="hungry")),
                          {"disturbance_score": 0.1, "csd_risk": 0.1, "rel_ascendancy": 0.1,
                           "reserve_recovery": 0.9}, Settings())
brittle = forager._autoscale(forager._resolve_params(Settings(forager_hunger="hungry")),
                            {"disturbance_score": 0.9, "csd_risk": 0.9, "rel_ascendancy": 0.7,
                             "reserve_recovery": 0.1}, Settings())
print(f"J autoscale eagerness  -> calm={calm['eagerness']} brittle={brittle['eagerness']}")
print(f"   profit bar          -> calm={calm['profit_threshold_R']:.3f} brittle={brittle['profit_threshold_R']:.3f}")
assert brittle["eagerness"] > calm["eagerness"]
assert brittle["profit_threshold_R"] < calm["profit_threshold_R"]
assert brittle["base_cooldown"] < calm["base_cooldown"]

print("\nOK")
