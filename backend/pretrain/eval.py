"""Walk-forward evaluation of the hybrid pipeline (frozen encoder + Kalshi
head) against the validated MPC baseline. Reads only — never writes to the
live settings, never touches the engine.

Run:  python -m backend.pretrain.eval
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .. import lab, signals
from ..config import DEFAULT_STRATEGY_PARAMS, STATE_DIR
from . import features as F
from .encoder import LSTMEncoder
from .head import KalshiHead, _kalshi_to_spot_rows, _mr_edge_alpha

ARTIFACTS = STATE_DIR / "pretrain"
LEVERAGE = 5.8
TIMEFRAME_MIN = 3
FOLDS = 4
HALF_SPREAD_BPS = 4.0


def _load_model():
    meta = np.load(ARTIFACTS / "norm.npz")
    mu, sd = meta["mu"], meta["sd"]
    seq_len = int(meta["seq_len"]); horizon = int(meta["horizon"])
    encoder = LSTMEncoder(n_features=int(mu.shape[0]))
    encoder.load_state_dict(torch.load(ARTIFACTS / "encoder.pt",
                                       map_location="cpu", weights_only=True))
    encoder.eval()
    head_meta = json.loads((ARTIFACTS / "head_meta.json").read_text())
    head = KalshiHead(feat_dim=int(head_meta["head_input_dim"]))
    head.load_state_dict(torch.load(ARTIFACTS / "head.pt",
                                    map_location="cpu", weights_only=True))
    head.eval()
    return encoder, head, mu, sd, seq_len, horizon


def _predict_series(candles_1m):
    encoder, head, mu, sd, seq_len, horizon = _load_model()
    spot_like = _kalshi_to_spot_rows(candles_1m)
    feats = F.per_bar_features(spot_like)
    feats = (feats - mu) / sd
    closes = np.array([r["c"] for r in spot_like], dtype=np.float64)
    n = feats.shape[0]
    preds = np.zeros(n, dtype=np.float32)
    alpha = _mr_edge_alpha(closes)
    # batch the encoder forward over rolling windows
    starts = np.arange(seq_len - 1, n)
    with torch.no_grad():
        for i in range(0, len(starts), 1024):
            idx = starts[i:i + 1024]
            X = np.stack([feats[s - seq_len + 1: s + 1] for s in idx])
            xb = torch.from_numpy(X.astype(np.float32))
            z = encoder(xb)
            a = torch.from_numpy(alpha[idx].astype(np.float32).reshape(-1, 1))
            pr = head(torch.cat([z, a], dim=1)).numpy()
            preds[idx] = pr
    return preds


def _pos_from_pred(pred: np.ndarray, deadband: float = 0.0005) -> np.ndarray:
    """Convert per-bar predicted return into a target position in [-1, 1].
    Sign(pred) outside the deadband; clip magnitude to 1."""
    out = np.zeros_like(pred)
    out[pred > deadband] = 1.0
    out[pred < -deadband] = -1.0
    return out


def main():
    if not (ARTIFACTS / "head.pt").exists():
        raise SystemExit("Need to train the head first: "
                         "python -m backend.pretrain.head")
    rows = lab.load(use_cache=True, do_clean=True)
    candles_1m = rows
    print(f"Kalshi 1m bars: {len(candles_1m):,}")

    print("predicting on full 1m series...")
    pred_1m = _predict_series(candles_1m)

    # Aggregate to TIMEFRAME_MIN bars (mean predicted return in each bucket;
    # take the last value of the bucket).
    n = len(candles_1m)
    bars = lab.aggregate(candles_1m, TIMEFRAME_MIN)
    print(f"-> {len(bars)} {TIMEFRAME_MIN}m bars")
    pred_tf = []
    for i in range(0, n, TIMEFRAME_MIN):
        chunk = pred_1m[i:i + TIMEFRAME_MIN]
        pred_tf.append(float(chunk[-1]) if len(chunk) else 0.0)
    pred_tf = np.array(pred_tf[:len(bars)])

    # Hybrid model positions + urgency proxy = |pred|
    pos_hybrid = _pos_from_pred(pred_tf)
    urg_hybrid = np.abs(pred_tf)

    # MPC baseline
    params = dict(DEFAULT_STRATEGY_PARAMS)
    pos_mpc, urg_mpc = signals.mpc_with_aux(bars, params)
    pos_mpc = np.array(pos_mpc); urg_mpc = np.array(urg_mpc)

    print("\nWalk-forward comparison (pure-maker, chase_n=3, hs=4bps):\n")
    seg = len(bars) // FOLDS
    print(f"  fold       hybrid_ret%   mpc_ret%   hybrid_dd%   mpc_dd%   "
          f"hyb_mk hyb_ch  mpc_mk mpc_ch")
    rets_h = []; rets_m = []
    for f in range(FOLDS):
        a = f * seg
        b = (f + 1) * seg if f < FOLDS - 1 else len(bars)
        cs = bars[a:b]
        ph = pos_hybrid[a:b].tolist(); uh = urg_hybrid[a:b].tolist()
        pm = pos_mpc[a:b].tolist(); um = urg_mpc[a:b].tolist()
        rh = signals.simulate_hybrid(cs, ph, uh, k=float("inf"), chase_n=3,
                                     half_spread_bps=HALF_SPREAD_BPS,
                                     leverage=LEVERAGE)
        rm = signals.simulate_hybrid(cs, pm, um, k=float("inf"), chase_n=3,
                                     half_spread_bps=HALF_SPREAD_BPS,
                                     leverage=LEVERAGE)
        rets_h.append(rh["return_pct"]); rets_m.append(rm["return_pct"])
        print(f"  {f+1:>4d}     {rh['return_pct']:>+10.2f}   "
              f"{rm['return_pct']:>+8.2f}   {rh['max_dd_pct']:>9.2f}   "
              f"{rm['max_dd_pct']:>6.2f}   {rh['trades_maker']:>4d} "
              f"{rh['trades_chase']:>4d}   "
              f"{rm['trades_maker']:>4d} {rm['trades_chase']:>4d}")
    def _comp(xs):
        t = 1.0
        for x in xs: t *= (1 + x / 100)
        return (t - 1) * 100
    print(f"\n  compounded:  hybrid {_comp(rets_h):+.2f}%   "
          f"mpc {_comp(rets_m):+.2f}%")
    print(f"  pos folds:   hybrid {sum(1 for x in rets_h if x>0)}/{FOLDS}   "
          f"mpc {sum(1 for x in rets_m if x>0)}/{FOLDS}")
    print("\n(Hybrid model is RESEARCH ONLY — not wired into the live engine.)")


if __name__ == "__main__":
    main()
