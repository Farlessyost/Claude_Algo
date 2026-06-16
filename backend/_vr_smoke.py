"""Smoke-test the visual_review module without hitting Opus."""
from __future__ import annotations

import time
from backend import visual_review, lab
from backend.config import Settings
from backend.store import Store


# 1) Render a chart from real Kalshi candles
rows = lab.load(use_cache=True)
bars = lab.aggregate(rows, 3)
print(f"loaded {len(bars):,} 3m bars")

# Fake recent decisions
fake_decisions = [
    {"ts": "2026-06-15T08:00:00+00:00", "verb": "LONG", "action": "ENTER_LONG",
      "execution": {"submitted": True}},
    {"ts": "2026-06-15T08:15:00+00:00", "verb": "SHORT", "action": "REVERSE_SHORT",
      "execution": {"submitted": True}},
]

png = visual_review.render_chart(
    candles=bars[-100:], position_contracts=39.0,
    recent_decisions=fake_decisions,
    ticker="KXBTCPERP", csd_risk=0.42,
    ecology_phase="scavenger", interval="3m")
print(f"chart PNG (single panel): {len(png):,} bytes (first 8: {png[:8]!r})")

# 1b) Dual-panel render: 3m anchor + 1m near-term confirmation.
png2 = visual_review.render_chart(
    candles=bars[-100:], position_contracts=39.0,
    recent_decisions=fake_decisions,
    ticker="KXBTCPERP", csd_risk=0.42,
    ecology_phase="scavenger", interval="3m",
    candles_1m=rows[-120:])
assert png2[:8] == b"\x89PNG\r\n\x1a\n", png2[:8]
print(f"chart PNG (dual panel) : {len(png2):,} bytes")

# 2) Test block_entry_if_stop
class FakeStore:
    visual_review = {"concern": "STOP", "note": "test stop scenario"}

fake_settings = Settings(visual_review_block_entries_on_stop=True)
proposal = {"action": "ENTER_LONG", "target_fraction": 0.4, "blended_score": 0.5, "confidence": 0.6}
print(f"\nbefore block_entry_if_stop: action={proposal['action']}, frac={proposal['target_fraction']}")
blocked = visual_review.block_entry_if_stop(FakeStore(), proposal, fake_settings)
print(f"blocked={blocked} -> action={proposal['action']}, frac={proposal['target_fraction']}, "
       f"visual_review_blocked={proposal.get('visual_review_blocked')}")

# 3) Same but with concern=OK should NOT block
class FakeStoreOK:
    visual_review = {"concern": "OK", "note": "all good"}

proposal2 = {"action": "ENTER_LONG", "target_fraction": 0.4}
blocked2 = visual_review.block_entry_if_stop(FakeStoreOK(), proposal2, fake_settings)
print(f"\nwith OK: blocked={blocked2} -> action={proposal2['action']} (should stay ENTER_LONG)")

# 4) Test that CLOSE is NEVER blocked even when STOP
proposal3 = {"action": "CLOSE", "target_fraction": 0.0}
blocked3 = visual_review.block_entry_if_stop(FakeStore(), proposal3, fake_settings)
print(f"\nwith STOP + CLOSE: blocked={blocked3} (should be False — exits must always go through)")

# 5) Save the chart so we can eyeball it
visual_review.VISUAL_REVIEW_IMAGE_PATH.write_bytes(png)
print(f"\nWrote sample chart to {visual_review.VISUAL_REVIEW_IMAGE_PATH}")

# 6) Test maybe_run_review timing (won't actually call Opus because no API key
# is loaded into env unless the file is present)
import time as _t
store = Store()
mkt = {"candles_full": bars, "candles": bars[-30:], "price": bars[-1]["close"]}
acct = {"position_contracts": 0.0, "equity": 137.8, "entry_price": 0,
         "unrealized_pnl": 0}
s = Settings(visual_review_enabled=False)  # disabled -> should return None
r = visual_review.maybe_run_review(s, store, mkt, acct, [])
print(f"\nwith enabled=False: maybe_run_review returned {r}")

s2 = Settings(visual_review_enabled=True, visual_review_interval_seconds=600)
# This will TRY to call Opus. If no API key, returns None.
print("calling maybe_run_review with enabled=True (may or may not call Opus depending on key)...")
r2 = visual_review.maybe_run_review(s2, store, mkt, acct, [])
print(f"  result: {r2}")
