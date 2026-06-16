"""Train the small Kalshi head on the frozen pretrained encoder.

Pipeline:
  Kalshi 1m candles --(features)--> encoder (FROZEN) --(64-d z)-->
        [z || mr_edge_alpha] --MLP--> next-K-bar return prediction.

The mr_edge alpha (the validated linear reversion signal) is appended to z so
the head only has to learn the *residual* on top of what we already know.

Run:  python -m backend.pretrain.head [--epochs 30] [--batch 256]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..config import STATE_DIR
from .. import lab
from . import features as F
from .encoder import LSTMEncoder

ARTIFACTS = STATE_DIR / "pretrain"


def _kalshi_to_spot_rows(candles):
    """Map Kalshi candles into the {t,o,h,l,c,v} schema features.per_bar_features
    expects (matches the Binance fetcher's output)."""
    return [{"t": c["ts"], "o": c["open"], "h": c["high"],
             "l": c["low"], "c": c["close"], "v": c.get("volume", 0.0)}
            for c in candles]


def _mr_edge_alpha(closes: np.ndarray, vol_win: int = 12, lookback: int = 2,
                   beta: float = 0.18) -> np.ndarray:
    """Compute the validated mr_edge alpha as a per-bar number, identical
    shape to closes. Mirrors backend.edge / backend.signals.mpc_with_aux."""
    n = len(closes)
    r = np.zeros(n)
    r[1:] = (closes[1:] - closes[:-1]) / np.maximum(closes[:-1], 1e-9)
    out = np.zeros(n, dtype=np.float32)
    for i in range(max(vol_win, lookback) + 1, n):
        move = r[i - lookback + 1: i + 1].sum()
        vol = r[i - vol_win:i].std() or 1e-9
        out[i] = -beta * (move / vol)
    return out


class KalshiHead(nn.Module):
    def __init__(self, feat_dim: int, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seq_len", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.3)
    args = ap.parse_args()

    if not (ARTIFACTS / "encoder.pt").exists():
        raise SystemExit("Need to pretrain first: python -m backend.pretrain.train")

    meta = np.load(ARTIFACTS / "norm.npz")
    mu, sd = meta["mu"], meta["sd"]
    encoder = LSTMEncoder(n_features=int(mu.shape[0]))
    encoder.load_state_dict(torch.load(ARTIFACTS / "encoder.pt",
                                       map_location="cpu", weights_only=True))
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad_(False)

    rows = lab.load(use_cache=True, do_clean=True)
    candles_1m = rows
    print(f"Kalshi 1m bars: {len(candles_1m):,}")
    spot_like = _kalshi_to_spot_rows(candles_1m)
    feats = F.per_bar_features(spot_like)
    feats = (feats - mu) / sd
    closes = np.array([r["c"] for r in spot_like], dtype=np.float64)
    X_seq, y_ret, _ = F.make_windows(feats, closes, args.seq_len, args.horizon)
    print(f"  windows: {X_seq.shape}")

    if X_seq.shape[0] < 100:
        raise SystemExit("Not enough Kalshi data — fetch more first.")

    # Encode windows with the frozen encoder (do it in chunks to keep RAM small).
    zs = []
    encoder.eval()
    with torch.no_grad():
        for i in range(0, X_seq.shape[0], 1024):
            xb = torch.from_numpy(X_seq[i:i + 1024])
            zs.append(encoder(xb).numpy())
    Z = np.concatenate(zs, axis=0)   # (N, 64)

    # mr_edge alpha aligned to each window (use end-of-window index)
    alpha = _mr_edge_alpha(closes)
    starts = np.arange(args.seq_len - 1, len(closes) - args.horizon)
    a = alpha[starts].astype(np.float32).reshape(-1, 1)
    Xh = np.concatenate([Z, a], axis=1).astype(np.float32)
    print(f"  head input: {Xh.shape}, target: {y_ret.shape}")

    # chronological val split — no leakage from future into past
    n = Xh.shape[0]
    cut = int(n * (1 - args.val_frac))
    Xt = torch.from_numpy(Xh[:cut]); yt = torch.from_numpy(y_ret[:cut])
    Xv = torch.from_numpy(Xh[cut:]); yv = torch.from_numpy(y_ret[cut:])

    head = KalshiHead(feat_dim=Xh.shape[1])
    opt = torch.optim.Adam(head.parameters(), lr=args.lr)
    bs = args.batch
    n_train = Xt.shape[0]
    best = float("inf")
    for ep in range(1, args.epochs + 1):
        head.train()
        idx = torch.randperm(n_train)
        tl = 0.0; ntb = 0
        for i in range(0, n_train, bs):
            j = idx[i:i + bs]
            pr = head(Xt[j])
            loss = nn.functional.mse_loss(pr, yt[j])
            opt.zero_grad(); loss.backward(); opt.step()
            tl += loss.item(); ntb += 1
        head.eval()
        with torch.no_grad():
            pv = head(Xv)
            v_mse = nn.functional.mse_loss(pv, yv).item()
            # IC vs label
            p_c = pv - pv.mean(); y_c = yv - yv.mean()
            ic = (p_c * y_c).sum().item() / max(1e-12,
                  (p_c.pow(2).sum().sqrt() * y_c.pow(2).sum().sqrt()).item())
            # IC of the linear baseline (mr_edge alpha column alone)
            a_v = Xv[:, -1]
            a_c = a_v - a_v.mean()
            ic_base = (a_c * y_c).sum().item() / max(1e-12,
                  (a_c.pow(2).sum().sqrt() * y_c.pow(2).sum().sqrt()).item())
        print(f"epoch {ep:>3d}  train_mse {tl/max(1,ntb):.6g}  "
              f"val_mse {v_mse:.6g}  val_IC {ic:+.4f}  "
              f"baseline_IC {ic_base:+.4f}")
        if v_mse < best:
            best = v_mse
            torch.save(head.state_dict(), ARTIFACTS / "head.pt")
    with open(ARTIFACTS / "head_meta.json", "w") as f:
        json.dump({"seq_len": args.seq_len, "horizon": args.horizon,
                   "best_val_mse": best, "head_input_dim": Xh.shape[1]}, f)
    print(f"saved head to {ARTIFACTS/'head.pt'} (best val mse {best:.6g})")


if __name__ == "__main__":
    main()
