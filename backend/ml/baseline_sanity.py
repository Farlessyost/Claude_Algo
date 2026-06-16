"""Sanity check: how much of the Kalshi forward-return predictability comes
from trivial cross-venue lag vs the LSTM's learned features?

Reports IC of (a) last 1-min spot log return, (b) last 3-min spot log return
against the same Kalshi 3-min forward log return target used in head_train.
If a simple baseline already explains most of the IC, the LSTM is mostly
recovering market-microstructure lag (Kalshi quote drift behind spot), not
adding tradable alpha. Writes state/ml/baseline_sanity.json.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from backend.ml.dataset import featurize, load_spot
from backend.ml.head_train import load_kalshi_bars

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "state" / "ml" / "baseline_sanity.json"


def main(target_min: int = 3) -> dict:
    spot = load_spot()
    feats, valid = featurize(spot)
    ts_spot = spot[:, 0].astype(np.int64)
    spot_idx = {int(t): i for i, t in enumerate(ts_spot)}

    k = load_kalshi_bars()
    ts_k = k[:, 0].astype(np.int64)
    close_k = k[:, 4]

    xs_3m, xs_1m, ys = [], [], []
    for i in range(target_min, len(ts_k) - target_min):
        t = int(ts_k[i])
        si = spot_idx.get(t)
        if si is None:
            continue
        if int(ts_k[i + target_min]) != t + target_min * 60:
            continue
        if int(ts_k[i - target_min]) != t - target_min * 60:
            continue
        if not (valid[si] and valid[si - 1] and valid[si - 2]):
            continue
        sc_now = spot[si, 4]
        sc_3m_ago = spot[si - target_min, 4]
        if sc_now <= 0 or sc_3m_ago <= 0:
            continue
        xs_3m.append(math.log(sc_now / sc_3m_ago))
        xs_1m.append(float(feats[si, 0]))
        ys.append(math.log(close_k[i + target_min] / close_k[i]))

    xa = np.asarray(xs_3m)
    xb = np.asarray(xs_1m)
    yy = np.asarray(ys)

    def ic(a, b):
        if a.std() == 0 or b.std() == 0:
            return 0.0
        return float(np.corrcoef(a, b)[0, 1])

    res = {
        "n": int(len(yy)),
        "ic_prior_3min_spot_vs_kalshi_fwd": ic(xa, yy),
        "ic_prior_1min_spot_vs_kalshi_fwd": ic(xb, yy),
        "note": (
            "If the 1-min IC is already near the head IC, the LSTM is mostly "
            "recovering cross-venue lag, not novel structure."
        ),
    }
    OUT.write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return res


if __name__ == "__main__":
    main()
