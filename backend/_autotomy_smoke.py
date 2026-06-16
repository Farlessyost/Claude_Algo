"""Smoke-test the Autotomy Agent.

Autotomy is the forager's loss-side sibling:
  - small loss alone -> no mutation
  - loss + toxic ecology -> CLOSE + cooldown
  - cooldown blocks new entries but never blocks exits
"""
import time

from backend import autotomy
from backend.config import Settings


class FakeStore:
    def __init__(self, eco, csd, autotomy_state=None, entry=6.5):
        self.ecosystem = eco
        self.csd_state = csd
        self.autotomy_state = autotomy_state or {}
        self.position_entry = {"entry": entry, "sign": 1}
        self.account = {}
        self.market = {}


def eco(predator=0.2, scavenger=0.7, immune=0.1, disturbance=0.2,
        liq_z=0.0, stretch_z=0.5, reserve=0.8, ascend=0.1, depth=1.2):
    return {
        "organisms": {"scores": {
            "predator": predator, "scavenger": scavenger, "immune": immune,
        }},
        "drivers": {
            "disturbance": disturbance, "liq_proxy_z": liq_z,
            "stretch_z": stretch_z, "depth_recover": depth,
        },
        "network_metrics": {"rel_reserve": reserve, "rel_ascendancy": ascend},
    }


MKT = {"price": 6.5, "features": {"atr_pct": 0.2}}


def acct(upnl, pos=10.0):
    return {"position_contracts": pos, "price": 6.5, "equity": 100.0,
            "unrealized_pnl": upnl}


def upnl_for_R(R):
    return R * (10.0 * 6.5 * 0.2 / 100.0)


S = Settings(autotomy_enabled=True, autotomy_aggression="steady",
             autotomy_cooldown_seconds=120)

assert autotomy._resolve_params(Settings(autotomy_aggression="ravenous"))["loss_R"] < \
       autotomy._resolve_params(Settings(autotomy_aggression="steady"))["loss_R"]
assert autotomy._resolve_params(Settings(autotomy_aggression="ravenous"))["min_confirmations"] <= 2
print("preset resolve OK (ravenous is faster than steady)")

# A) disabled -> diagnostic only, no mutation
prop = {"action": "ADD", "target_fraction": 0.2, "expected_edge_pct": 0.3}
st = autotomy.apply(Settings(autotomy_enabled=False), FakeStore(eco(), {"risk": 0.1}),
                    acct(-upnl_for_R(2.0)), MKT, prop)
assert prop["action"] == "ADD" and not st["enabled"]
print("A disabled        -> no mutation OK")

# B) small loss alone -> no eject
prop = {"action": "ADD", "target_fraction": 0.2, "expected_edge_pct": 0.3}
st = autotomy.apply(S, FakeStore(eco(), {"risk": 0.1}), acct(-upnl_for_R(0.2)), MKT, prop)
print(f"B small loss      -> {prop['action']} pressure={st['autotomy_pressure']}")
assert prop["action"] == "ADD" and not st["ejected"]

# B2) same moderate toxic loss: steady waits, ravenous ejects.
moderate_toxic = eco(predator=0.7, scavenger=0.2, immune=0.5, disturbance=0.8,
                     liq_z=1.0, stretch_z=1.2, reserve=0.3, ascend=0.6, depth=0.4)
prev_mod = {"reserve_prev": 0.6, "csd_prev": 0.3}
prop = {"action": "ADD", "target_fraction": 0.2, "expected_edge_pct": 0.05}
st = autotomy.apply(S, FakeStore(moderate_toxic, {"risk": 0.65, "threshold": 0.8},
                                 prev_mod, entry=6.58),
                    acct(-upnl_for_R(0.14)), MKT, prop)
print(f"B2 steady toxic   -> {prop['action']} pressure={st['autotomy_pressure']} confirms={st['confirmations']}")
assert prop["action"] == "ADD" and not st["ejected"]
prop = {"action": "ADD", "target_fraction": 0.2, "expected_edge_pct": 0.05}
st = autotomy.apply(Settings(autotomy_enabled=True, autotomy_aggression="ravenous"),
                    FakeStore(moderate_toxic, {"risk": 0.65, "threshold": 0.8},
                              prev_mod, entry=6.58),
                    acct(-upnl_for_R(0.14)), MKT, prop)
print(f"B3 ravenous toxic -> {prop['action']} pressure={st['autotomy_pressure']} confirms={st['confirmations']}")
assert prop["action"] == "CLOSE" and st["ejected"]

# C) toxic losing ecosystem -> hard close
toxic = eco(predator=0.9, scavenger=0.1, immune=0.8, disturbance=0.95,
            liq_z=2.0, stretch_z=1.5, reserve=0.15, ascend=0.75, depth=0.2)
prev = {"reserve_prev": 0.7, "csd_prev": 0.4}
prop = {"action": "ADD", "target_fraction": -0.4, "expected_edge_pct": 0.02}
store = FakeStore(toxic, {"risk": 0.9, "threshold": 0.8}, prev, entry=6.62)
st = autotomy.apply(S, store, acct(-upnl_for_R(0.7)), MKT, prop)
print(f"C toxic loss      -> {prop['action']} pressure={st['autotomy_pressure']} confirms={st['confirmations']}")
assert prop["action"] == "CLOSE" and st["ejected"] and prop["force_taker"]
assert st["in_cooldown"] and st["cooldown_remaining_s"] > 0

# D) cooldown blocks a new entry
future = time.time() + 90
prop = {"action": "ENTER_LONG", "target_fraction": 0.5, "expected_edge_pct": 0.5}
st = autotomy.apply(S, FakeStore(eco(), {"risk": 0.1}, {"cooldown_until": future}),
                    acct(0.0, pos=0.0), MKT, prop)
print(f"D cooldown entry  -> {prop['action']} blocked={st['blocked_entry']}")
assert prop["action"] == "HOLD" and st["blocked_entry"]

# E) cooldown never blocks an exit
prop = {"action": "CLOSE", "target_fraction": 0.0, "expected_edge_pct": 0.1}
st = autotomy.apply(S, FakeStore(eco(), {"risk": 0.1}, {"cooldown_until": future}),
                    acct(-upnl_for_R(0.4), pos=5.0), MKT, prop)
print(f"E cooldown close  -> {prop['action']} blocked={st['blocked_entry']}")
assert prop["action"] == "CLOSE" and not st["blocked_entry"]

print("\nOK")
