"""Train a small MLP head on frozen LSTM encoder features for Kalshi forward
returns. Walk-forward eval reported.

Pipeline:
  1) Load Kalshi 1m rich history (state/history_1m_rich.json)
  2) For each Kalshi bar t with enough preceding spot context, build a
     standardised L-bar spot feature window ending at t.
  3) Push through frozen encoder -> hidden vector z (B, H_dim).
  4) Train a small MLP head to predict the next-K-minute Kalshi log return.
  5) Walk-forward eval: 4 folds, train on past, eval on next slice.

Outputs:
  state/ml/head.pt
  state/ml/walkforward_ml.json
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from backend.ml.dataset import featurize, load_spot
from backend.ml.encoder import EncoderConfig, KalshiHead, LSTMEncoder

ROOT = Path(__file__).resolve().parents[2]
KALSHI_HIST = ROOT / "state" / "history_1m_rich.json"
ENCODER_PATH = ROOT / "state" / "ml" / "encoder.pt"
OUT_DIR = ROOT / "state" / "ml"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_kalshi_bars() -> np.ndarray:
    """Return [N, 5] = [ts, open, high, low, close] sorted by ts, cleaned."""
    raw = json.loads(KALSHI_HIST.read_text())
    rows = []
    for r in raw:
        c = r.get("close")
        if c is None or not isinstance(c, (int, float)) or c <= 0:
            continue
        rows.append([r["ts"], r["open"], r["high"], r["low"], c])
    arr = np.array(rows, dtype=np.float64)
    arr = arr[np.argsort(arr[:, 0])]
    # Drop duplicates and obvious corruption (median-band filter, mirroring lab.clean)
    med = np.median(arr[:, 4])
    band = (arr[:, 4] > med * 0.1) & (arr[:, 4] < med * 10.0)
    arr = arr[band]
    return arr


def _build_encoder_from_ckpt(ckpt: dict):
    """Construct the right encoder class from a saved checkpoint."""
    kind = ckpt.get("kind", "lstm")
    if kind == "lstm":
        cfg = EncoderConfig(**ckpt["encoder_cfg"])
        enc = LSTMEncoder(cfg)
        hidden = cfg.hidden
    elif kind in ("cfc", "ltc"):
        from backend.ml.encoder_liquid import LiquidConfig, LiquidEncoder
        cfg = LiquidConfig(**ckpt["encoder_cfg"])
        enc = LiquidEncoder(cfg)
        hidden = cfg.hidden
    else:
        raise ValueError(f"unknown encoder kind in checkpoint: {kind}")
    enc.load_state_dict(ckpt["encoder_state"])
    enc.eval()
    for p in enc.parameters():
        p.requires_grad_(False)
    return enc, cfg, hidden


def build_kalshi_examples(
    encoder_path: Path,
    target_minutes: int = 3,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, object, dict]:
    """Materialise (Z, y, ts) where Z is encoder output per Kalshi bar."""
    ckpt = torch.load(encoder_path, map_location="cpu", weights_only=False)
    L = ckpt["window_L"]
    mu = np.asarray(ckpt["feature_mu"], dtype=np.float32)
    sd = np.asarray(ckpt["feature_sd"], dtype=np.float32)
    enc, cfg, _hidden = _build_encoder_from_ckpt(ckpt)

    spot = load_spot()
    feats, valid = featurize(spot)
    ts_spot = spot[:, 0].astype(np.int64)
    spot_idx = {int(t): i for i, t in enumerate(ts_spot)}

    kalshi = load_kalshi_bars()
    ts_k = kalshi[:, 0].astype(np.int64)
    close_k = kalshi[:, 4]

    X_chunks = []
    y_list = []
    ts_list = []
    skipped_no_spot = 0
    skipped_gap = 0
    skipped_target = 0

    for i in range(len(ts_k) - target_minutes):
        t = int(ts_k[i])
        si = spot_idx.get(t)
        if si is None or si < L - 1:
            skipped_no_spot += 1
            continue
        window_valid = valid[si - L + 2 : si + 1]
        if not window_valid.all():
            skipped_gap += 1
            continue
        # Forward Kalshi log return over target_minutes
        ts_tgt = t + target_minutes * 60
        # Find nearest Kalshi bar at ts_tgt (must be contiguous)
        if i + target_minutes >= len(ts_k) or int(ts_k[i + target_minutes]) != ts_tgt:
            skipped_target += 1
            continue
        c0 = close_k[i]
        c1 = close_k[i + target_minutes]
        if c0 <= 0 or c1 <= 0:
            skipped_target += 1
            continue
        win = feats[si - L + 1 : si + 1]
        win = (win - mu) / sd
        X_chunks.append(win.astype(np.float32))
        y_list.append(math.log(c1 / c0))
        ts_list.append(t)

    print(
        f"kalshi examples: kept={len(X_chunks)} skipped(no_spot={skipped_no_spot}, "
        f"gap={skipped_gap}, target={skipped_target})",
        flush=True,
    )
    if not X_chunks:
        raise RuntimeError("no aligned Kalshi/spot examples")

    X = np.stack(X_chunks)  # (N, L, F)
    y = np.asarray(y_list, dtype=np.float32)
    ts_arr = np.asarray(ts_list, dtype=np.int64)

    # Run encoder in batches
    Z = []
    with torch.no_grad():
        for i in range(0, X.shape[0], 512):
            xb = torch.from_numpy(X[i : i + 512])
            Z.append(enc(xb).numpy())
    Z = np.concatenate(Z, axis=0).astype(np.float32)
    return Z, y, ts_arr, cfg, ckpt


def train_head_one_fold(
    Z_tr: np.ndarray,
    y_tr: np.ndarray,
    Z_va: np.ndarray,
    y_va: np.ndarray,
    hidden: int,
    epochs: int = 40,
    lr: float = 5e-4,
    batch_size: int = 256,
    seed: int = 0,
) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Standardise y on train
    yt_mu = float(y_tr.mean())
    yt_sd = float(y_tr.std() + 1e-9)
    y_tr_n = (y_tr - yt_mu) / yt_sd

    tr_loader = DataLoader(
        TensorDataset(torch.from_numpy(Z_tr), torch.from_numpy(y_tr_n.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )
    head = KalshiHead(hidden)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.MSELoss()

    best_state = None
    best_va_mse = float("inf")
    history = []
    for ep in range(epochs):
        head.train()
        for zb, yb in tr_loader:
            opt.zero_grad()
            pred = head(zb)
            loss = loss_fn(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
        head.eval()
        with torch.no_grad():
            pred_va = head(torch.from_numpy(Z_va)).numpy() * yt_sd + yt_mu
        mse = float(np.mean((pred_va - y_va) ** 2))
        # Information coefficient (Pearson) and rank IC
        if pred_va.std() > 0 and y_va.std() > 0:
            ic = float(np.corrcoef(pred_va, y_va)[0, 1])
        else:
            ic = 0.0
        history.append({"ep": ep, "va_mse": mse, "ic": ic})
        if mse < best_va_mse:
            best_va_mse = mse
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        pred_va = head(torch.from_numpy(Z_va)).numpy() * yt_sd + yt_mu
    ic = float(np.corrcoef(pred_va, y_va)[0, 1]) if pred_va.std() > 0 and y_va.std() > 0 else 0.0

    def _ranks(a):
        order = a.argsort().argsort().astype(np.float64)
        return (order - order.mean()) / (order.std() + 1e-9)
    rank_ic = float(np.mean(_ranks(pred_va) * _ranks(y_va)))

    return {
        "ic": ic,
        "rank_ic": rank_ic,
        "va_mse": best_va_mse,
        "n_val": int(Z_va.shape[0]),
        "pred_va": pred_va,
        "y_va": y_va,
        "history": history,
        "state": best_state,
        "yt_mu": yt_mu, "yt_sd": yt_sd,
    }


def walkforward(
    Z: np.ndarray, y: np.ndarray, ts: np.ndarray, hidden: int, n_folds: int = 4
) -> dict:
    n = Z.shape[0]
    fold_size = n // (n_folds + 1)
    folds = []
    for k in range(1, n_folds + 1):
        tr_end = k * fold_size
        va_end = min(n, (k + 1) * fold_size)
        Z_tr, y_tr = Z[:tr_end], y[:tr_end]
        Z_va, y_va = Z[tr_end:va_end], y[tr_end:va_end]
        print(f"fold {k}: train [0:{tr_end}] val [{tr_end}:{va_end}]", flush=True)
        out = train_head_one_fold(Z_tr, y_tr, Z_va, y_va, hidden=hidden, seed=k)
        sig = np.sign(out["pred_va"])
        pnl = sig * out["y_va"]
        pnl_total = float(pnl.sum())
        pnl_mean = float(pnl.mean())
        pnl_sd = float(pnl.std() + 1e-12)
        t_stat = pnl_mean / pnl_sd * math.sqrt(pnl.size)
        hit = float((sig * out["y_va"] > 0).mean())
        folds.append(
            {
                "fold": k,
                "ic": out["ic"],
                "rank_ic": out["rank_ic"],
                "va_mse": out["va_mse"],
                "n_val": out["n_val"],
                "pnl_total_logret": pnl_total,
                "pnl_mean_logret": pnl_mean,
                "t_stat": t_stat,
                "hit_rate": hit,
                "ts_start": int(ts[tr_end]) if tr_end < n else None,
                "ts_end": int(ts[va_end - 1]) if va_end <= n else None,
            }
        )

    tr_end = n_folds * fold_size
    final = train_head_one_fold(Z[:tr_end], y[:tr_end], Z[tr_end:], y[tr_end:], hidden=hidden, seed=99)
    torch.save(
        {
            "head_state": final["state"],
            "yt_mu": final["yt_mu"],
            "yt_sd": final["yt_sd"],
            "hidden": hidden,
        },
        OUT_DIR / "head.pt",
    )
    return {
        "folds": folds,
        "mean_ic": float(np.mean([f["ic"] for f in folds])),
        "mean_rank_ic": float(np.mean([f["rank_ic"] for f in folds])),
        "mean_pnl_total": float(np.mean([f["pnl_total_logret"] for f in folds])),
        "mean_t_stat": float(np.mean([f["t_stat"] for f in folds])),
        "mean_hit_rate": float(np.mean([f["hit_rate"] for f in folds])),
        "final_holdout_ic": final["ic"],
        "final_holdout_rank_ic": final["rank_ic"],
        "n_examples": n,
    }


def main(target_minutes: int = 3, encoder_path: Path | None = None, tag: str = "") -> dict:
    t0 = time.time()
    ep = encoder_path or ENCODER_PATH
    print(f"building examples from {ep.name}...", flush=True)
    Z, y, ts, cfg, ckpt = build_kalshi_examples(ep, target_minutes=target_minutes)
    print(f"Z={Z.shape} y={y.shape} ts span {int(ts[0])}..{int(ts[-1])}", flush=True)
    print(f"y stats: mean={y.mean():+.5e} std={y.std():.5e}", flush=True)

    res = walkforward(Z, y, ts, hidden=cfg.hidden, n_folds=4)
    res["target_minutes"] = target_minutes
    res["encoder"] = ep.name
    res["elapsed_s"] = round(time.time() - t0, 1)
    out_name = f"walkforward_ml{('_' + tag) if tag else ''}.json"
    with open(OUT_DIR / out_name, "w") as f:
        json.dump(res, f, indent=2)
    print("done.", flush=True)
    print(json.dumps({k: v for k, v in res.items() if k != "folds"}, indent=2))
    return res


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--encoder", default=str(ENCODER_PATH))
    p.add_argument("--tag", default="")
    p.add_argument("--target-minutes", type=int, default=3)
    a = p.parse_args()
    main(target_minutes=a.target_minutes, encoder_path=Path(a.encoder), tag=a.tag)
