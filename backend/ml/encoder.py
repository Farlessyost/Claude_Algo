"""LSTM encoder + pretraining head.

Encoder takes a (B, L, F) window of features and returns:
  - hidden: (B, H_dim)  — final-layer hidden state, used as the feature vector
The pretraining head is a small MLP that maps the encoder hidden to H future
log returns. After pretraining, the encoder is reused (weights frozen) by the
Kalshi head trained in head_train.py.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class EncoderConfig:
    n_features: int = 4
    hidden: int = 96
    num_layers: int = 2
    dropout: float = 0.1
    horizon: int = 16


class LSTMEncoder(nn.Module):
    def __init__(self, cfg: EncoderConfig):
        super().__init__()
        self.cfg = cfg
        self.lstm = nn.LSTM(
            input_size=cfg.n_features,
            hidden_size=cfg.hidden,
            num_layers=cfg.num_layers,
            batch_first=True,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
        )
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(cfg.hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, F)
        out, (h, _c) = self.lstm(x)
        last = out[:, -1, :]  # (B, hidden)
        z = self.ln(self.act(self.proj(last)))
        return z


class PretrainHead(nn.Module):
    """Map encoder hidden vector to H future log returns."""

    def __init__(self, hidden: int, horizon: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, horizon),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class KalshiHead(nn.Module):
    """Small MLP from frozen encoder features to a scalar (forward Kalshi return)."""

    def __init__(self, hidden: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)
