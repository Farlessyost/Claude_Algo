"""Self-supervised pretraining of the LSTM encoder on BTC-USD 1m spot data.

Objective: from an L=256-bar window of features, predict the next H=16 log
returns. The encoder learns a temporal embedding that's useful for any
downstream forecasting task on this market.

Time split: windows whose end-time is before PRETRAIN_END_UTC go to the
pretraining set; the most recent slice is held back for the Kalshi-head
work in head_train.py. Within the pretraining set, the final 10% by time
is used as validation.

Usage:
    .venv\\Scripts\\python.exe -m backend.ml.pretrain
Outputs:
    state/ml/encoder.pt           — encoder weights
    state/ml/pretrain_log.json    — losses per epoch + config
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backend.ml.dataset import build_windows, featurize, load_spot
from backend.ml.encoder import EncoderConfig, LSTMEncoder, PretrainHead

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "state" / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Anything at or after this UTC instant is reserved for Kalshi alignment +
# walk-forward eval. Kalshi history begins 2026-06-03, so this gives a clean
# 1083-day pretraining horizon with no overlap.
PRETRAIN_END_UTC = datetime(2026, 6, 3, tzinfo=timezone.utc).timestamp()


def main(
    L: int = 256,
    H: int = 16,
    stride: int = 16,
    hidden: int = 96,
    layers: int = 2,
    batch_size: int = 256,
    epochs: int = 6,
    lr: float = 1e-3,
    seed: int = 42,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    print(f"loading spot data...", flush=True)
    candles = load_spot()
    print(f"  candles: {candles.shape[0]:,} bars", flush=True)
    feats, valid = featurize(candles)

    # Restrict to pretraining era
    ts = candles[:, 0].astype(np.int64)
    pre_mask = ts < int(PRETRAIN_END_UTC)
    feats_pre = feats[pre_mask]
    valid_pre = valid[pre_mask]
    print(
        f"  pretrain era: {pre_mask.sum():,} bars (last <= "
        f"{datetime.fromtimestamp(int(ts[pre_mask][-1]), tz=timezone.utc) if pre_mask.any() else 'n/a'})",
        flush=True,
    )

    print(f"building windows L={L} H={H} stride={stride}...", flush=True)
    X, y, anchors = build_windows(feats_pre, valid_pre, L=L, H=H, stride=stride)
    print(f"  windows: {X.shape[0]:,}", flush=True)
    if X.shape[0] < 1000:
        raise RuntimeError("not enough windows to train; let the fetch finish first")

    # Time-ordered 90/10 split
    n = X.shape[0]
    cut = int(n * 0.9)
    X_tr, X_va = X[:cut], X[cut:]
    y_tr, y_va = y[:cut], y[cut:]
    print(f"  train={X_tr.shape[0]:,} val={X_va.shape[0]:,}", flush=True)

    # Per-feature standardisation, fit on train only
    mu = X_tr.reshape(-1, X_tr.shape[-1]).mean(axis=0)
    sd = X_tr.reshape(-1, X_tr.shape[-1]).std(axis=0) + 1e-6
    X_tr = (X_tr - mu) / sd
    X_va = (X_va - mu) / sd
    # Targets: log returns are already centered around 0; scale by train std
    ty_sd = y_tr.std() + 1e-9
    y_tr_n = y_tr / ty_sd
    y_va_n = y_va / ty_sd

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr_n)),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    va_loader = DataLoader(
        TensorDataset(torch.from_numpy(X_va), torch.from_numpy(y_va_n)),
        batch_size=batch_size,
        shuffle=False,
    )

    cfg = EncoderConfig(
        n_features=X.shape[-1], hidden=hidden, num_layers=layers, horizon=H
    )
    enc = LSTMEncoder(cfg)
    head = PretrainHead(cfg.hidden, H)
    params = list(enc.parameters()) + list(head.parameters())
    n_params = sum(p.numel() for p in params)
    print(f"  encoder+head params: {n_params:,}", flush=True)

    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    history: list[dict] = []
    best_va = float("inf")
    t0 = time.time()
    for ep in range(1, epochs + 1):
        enc.train(); head.train()
        tr_loss = 0.0; n_tr = 0
        for xb, yb in tr_loader:
            opt.zero_grad()
            z = enc(xb)
            yhat = head(z)
            loss = loss_fn(yhat, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            tr_loss += loss.item() * xb.size(0); n_tr += xb.size(0)
        tr_loss /= max(n_tr, 1)

        enc.eval(); head.eval()
        va_loss = 0.0; n_va = 0
        ic_sum = 0.0; ic_cnt = 0
        with torch.no_grad():
            for xb, yb in va_loader:
                z = enc(xb)
                yhat = head(z)
                va_loss += loss_fn(yhat, yb).item() * xb.size(0); n_va += xb.size(0)
                # 1-step-ahead Pearson IC
                a = yhat[:, 0].numpy()
                b = yb[:, 0].numpy()
                if a.std() > 0 and b.std() > 0:
                    ic_sum += np.corrcoef(a, b)[0, 1]; ic_cnt += 1
        va_loss /= max(n_va, 1)
        ic1 = ic_sum / max(ic_cnt, 1)
        sched.step()

        elapsed = time.time() - t0
        print(
            f"  ep {ep}/{epochs}  tr={tr_loss:.5f}  va={va_loss:.5f}  ic1={ic1:+.4f}  "
            f"lr={opt.param_groups[0]['lr']:.2e}  t={elapsed:.0f}s",
            flush=True,
        )
        history.append({"epoch": ep, "tr": tr_loss, "va": va_loss, "ic1": ic1})

        if va_loss < best_va:
            best_va = va_loss
            torch.save(
                {
                    "encoder_state": enc.state_dict(),
                    "encoder_cfg": cfg.__dict__,
                    "feature_mu": mu.tolist(),
                    "feature_sd": sd.tolist(),
                    "target_sd": float(ty_sd),
                    "window_L": L,
                    "horizon_H": H,
                },
                OUT_DIR / "encoder.pt",
            )

    log = {
        "config": {
            "L": L, "H": H, "stride": stride, "hidden": hidden, "layers": layers,
            "batch_size": batch_size, "epochs": epochs, "lr": lr, "seed": seed,
            "pretrain_end_utc": PRETRAIN_END_UTC,
            "n_train_windows": int(X_tr.shape[0]),
            "n_val_windows": int(X_va.shape[0]),
        },
        "history": history,
        "best_va": best_va,
    }
    with open(OUT_DIR / "pretrain_log.json", "w") as f:
        json.dump(log, f, indent=2)
    print(f"saved encoder + log. best va={best_va:.5f}", flush=True)
    return log


if __name__ == "__main__":
    main()
