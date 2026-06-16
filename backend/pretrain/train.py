"""Pretrain the LSTM encoder on Binance spot 1m data.

Multi-task supervised: predict next-K-bar log return + realized vol from a
60-bar window of features. The encoder weights are saved separately; heads
are discarded.

Run:  python -m backend.pretrain.train [--epochs 5] [--batch 4096]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from ..config import STATE_DIR
from . import fetch_spot
from .encoder import MultiTaskModel
from .features import make_windows, per_bar_features

ARTIFACTS = STATE_DIR / "pretrain"
ARTIFACTS.mkdir(parents=True, exist_ok=True)

SEQ_LEN = 60
HORIZON = 3


def _build_dataset():
    rows = fetch_spot.load_all()
    if len(rows) < SEQ_LEN + HORIZON + 100:
        raise SystemExit("Need to fetch spot history first: "
                         "python -m backend.pretrain.fetch_spot")
    closes = np.array([r["c"] for r in rows], dtype=np.float64)
    feats = per_bar_features(rows)
    X, y_ret, y_vol = make_windows(feats, closes, SEQ_LEN, HORIZON)
    return X, y_ret, y_vol


def _norm_stats(X: np.ndarray):
    """Compute per-feature mean/std across train set so the model trains on
    standardized inputs. Saved so the live encoder loader can reuse them."""
    flat = X.reshape(-1, X.shape[-1])
    mu = flat.mean(axis=0); sd = flat.std(axis=0)
    sd = np.where(sd > 1e-6, sd, 1.0)
    return mu.astype(np.float32), sd.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val_frac", type=float, default=0.15)
    args = ap.parse_args()

    print("loading dataset...")
    t0 = time.time()
    X, y_ret, y_vol = _build_dataset()
    print(f"  {X.shape[0]:,} windows of shape {X.shape[1:]} in "
          f"{time.time()-t0:.1f}s")

    mu, sd = _norm_stats(X)
    X = (X - mu) / sd
    np.savez(ARTIFACTS / "norm.npz", mu=mu, sd=sd,
             seq_len=SEQ_LEN, horizon=HORIZON)

    # chronological train/val split (no shuffling across the boundary!)
    n = X.shape[0]
    cut = int(n * (1 - args.val_frac))
    Xt = torch.from_numpy(X[:cut]); yrt = torch.from_numpy(y_ret[:cut])
    yvt = torch.from_numpy(y_vol[:cut])
    Xv = torch.from_numpy(X[cut:]); yrv = torch.from_numpy(y_ret[cut:])
    yvv = torch.from_numpy(y_vol[cut:])
    train_loader = DataLoader(TensorDataset(Xt, yrt, yvt),
                              batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(TensorDataset(Xv, yrv, yvv),
                            batch_size=args.batch, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiTaskModel(n_features=X.shape[-1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    print(f"training on {device}, {sum(p.numel() for p in model.parameters()):,} params")

    best_val = float("inf")
    for ep in range(1, args.epochs + 1):
        model.train()
        tl_ret = tl_vol = 0.0; ntb = 0
        for xb, yrb, yvb in train_loader:
            xb = xb.to(device); yrb = yrb.to(device); yvb = yvb.to(device)
            pr, pv, _ = model(xb)
            loss_r = torch.nn.functional.mse_loss(pr, yrb)
            loss_v = torch.nn.functional.mse_loss(pv, yvb)
            loss = loss_r + 0.3 * loss_v   # vol task is auxiliary
            opt.zero_grad(); loss.backward(); opt.step()
            tl_ret += loss_r.item(); tl_vol += loss_v.item(); ntb += 1
        model.eval()
        vl_ret = vl_vol = 0.0; nvb = 0
        ic_num = ic_den_p = ic_den_y = 0.0
        with torch.no_grad():
            for xb, yrb, yvb in val_loader:
                xb = xb.to(device); yrb = yrb.to(device); yvb = yvb.to(device)
                pr, pv, _ = model(xb)
                vl_ret += torch.nn.functional.mse_loss(pr, yrb).item()
                vl_vol += torch.nn.functional.mse_loss(pv, yvb).item()
                nvb += 1
                # rank-free Pearson IC accumulator
                pr_c = pr - pr.mean(); yb_c = yrb - yrb.mean()
                ic_num += (pr_c * yb_c).sum().item()
                ic_den_p += (pr_c * pr_c).sum().item()
                ic_den_y += (yb_c * yb_c).sum().item()
        ic = ic_num / max(1e-12, (ic_den_p ** 0.5) * (ic_den_y ** 0.5))
        val_loss = (vl_ret / max(1, nvb)) + 0.3 * (vl_vol / max(1, nvb))
        print(f"epoch {ep}/{args.epochs}  "
              f"train_ret_mse {tl_ret/max(1,ntb):.6g}  "
              f"val_ret_mse {vl_ret/max(1,nvb):.6g}  "
              f"val_vol_mse {vl_vol/max(1,nvb):.4g}  "
              f"val_IC {ic:+.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.encoder.state_dict(), ARTIFACTS / "encoder.pt")
            torch.save(model.state_dict(), ARTIFACTS / "full_model.pt")
            with open(ARTIFACTS / "meta.txt", "w") as f:
                f.write(f"seq_len={SEQ_LEN}\nhorizon={HORIZON}\n"
                        f"val_ret_mse={vl_ret/max(1,nvb):.6g}\nval_IC={ic:+.4f}\n")
    print(f"saved encoder to {ARTIFACTS/'encoder.pt'} "
          f"(best val loss {best_val:.6g})")


if __name__ == "__main__":
    main()
