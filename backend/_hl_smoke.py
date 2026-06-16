"""Smoke-test the Hyperliquid replacement."""
from backend import multiasset
from backend.config import Settings

ctx = multiasset.hyperliquid_btc_ctx()
print("hyperliquid_btc_ctx() ->")
if ctx is None:
    print("  FAILED (None)")
else:
    for k, v in ctx.items():
        print(f"  {k:<24}: {v}")

# Run snapshot with a Settings-like object — provide kalshi_client=None so we
# don't try to authenticate. The Hyperliquid block should still populate.
s = Settings(ecology_use_hyperliquid=True, ecology_use_coinbase_spot=True,
              ecology_use_kalshi_eth=False)
snap = multiasset.snapshot(s, kalshi_client=None)
print("\nsnapshot() summary ->")
for k, v in snap.get("summary", {}).items():
    print(f"  {k:<26}: {v}")
print("\nsources:", snap.get("sources"))

# Second call to verify the OI delta computation kicks in
import time
time.sleep(1.5)
snap2 = multiasset.snapshot(s, kalshi_client=None)
print("\nsecond snapshot summary (OI delta should now be present) ->")
for k, v in snap2.get("summary", {}).items():
    print(f"  {k:<26}: {v}")
