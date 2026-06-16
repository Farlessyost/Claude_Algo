import json
from pathlib import Path
d = json.load(open(Path("state/csd_refined_ablation.json"), "r"))
print("Per-fold detail for skew_only:")
print(f"{'fold':<5}{'baseline ret':>16}{'baseline dd':>14}{'gate0.95 ret':>16}{'gate0.95 dd':>14}{'gate0.95 sh':>14}")
for f in d["skew_only"]["folds"]:
    base = f["base"]; g = f["gates"]["0.95"]
    print(f"{f['fold']:<5}{base['return_pct']:>+15.2f}%{base['max_dd_pct']:>+13.2f}%"
          f"{g['return_pct']:>+15.2f}%{g['max_dd_pct']:>+13.2f}%{g['sharpe']:>+14.2f}")
print()
print("Per-fold detail for combined gate0.90:")
print(f"{'fold':<5}{'baseline ret':>16}{'baseline dd':>14}{'gate0.90 ret':>16}{'gate0.90 dd':>14}{'gate0.90 sh':>14}")
for f in d["combined"]["folds"]:
    base = f["base"]; g = f["gates"]["0.9"]
    print(f"{f['fold']:<5}{base['return_pct']:>+15.2f}%{base['max_dd_pct']:>+13.2f}%"
          f"{g['return_pct']:>+15.2f}%{g['max_dd_pct']:>+13.2f}%{g['sharpe']:>+14.2f}")
