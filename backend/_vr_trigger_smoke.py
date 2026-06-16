"""Smoke-test the periodic vs CSD-fire trigger logic in maybe_run_review."""
import time
from backend import visual_review as vr
from backend.config import Settings

# Stub the model call + chart render so the test is free and fast.
vr.ask_opus_for_chart_review = lambda model, png, ctx: {
    "trend": "down", "concern": "STOP", "note": "x"}
vr.render_chart = lambda **k: b"PNG"


class FakeStore:
    def __init__(self, age_seconds, firing):
        self.visual_review = {"last_ts": time.time() - age_seconds}
        self.csd_state = {"enabled": True, "threshold": 0.8,
                          "risk": (0.9 if firing else 0.5)}
        self.ecosystem = {}

    def log(self, *a, **k):
        pass


s = Settings(visual_review_enabled=True, visual_review_interval_seconds=150,
             visual_review_event_min_interval_seconds=60,
             visual_review_on_csd_fire=True)
mkt = {"candles_full": [{"ts": 1, "open": 1, "high": 1, "low": 1, "close": 1}] * 10,
       "candles": []}
acct = {"position_contracts": 0.0}


def trig(age, firing):
    r = vr.maybe_run_review(s, FakeStore(age, firing), mkt, acct, [])
    return r["trigger"] if r else None


a = trig(10, True)
b = trig(70, True)
c = trig(70, False)
d = trig(200, False)
print("A recent(10s)+firing   ->", a, "(expect None: event cooldown)")
print("B 70s+firing           ->", b, "(expect csd_fire)")
print("C 70s+not-firing       ->", c, "(expect None)")
print("D 200s (past periodic) ->", d, "(expect periodic)")
assert a is None and b == "csd_fire" and c is None and d == "periodic", (a, b, c, d)
print("OK")
