"""Small LSTM encoder + multi-task head used for self-supervised pretraining.

Architecture (intentionally small — we have a lot of data per parameter):
  input: per-bar features (log-return, abs-return, range-bps, volume-z)
  -> Linear(F -> 32)
  -> LSTM(32 -> 64, 2 layers, dropout 0.1)
  -> the encoder representation is the last-step hidden state (64-d)
  -> two task heads:
        ret_head:  Linear(64 -> 1)  predicts next-K-bar log return
        vol_head:  Linear(64 -> 1)  predicts next-K-bar realized vol (log)

Encoder weights are saved separately so the downstream Kalshi head can load
them in eval mode and use the 64-d representation as features.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class LSTMEncoder(nn.Module):
    def __init__(self, n_features: int = 4, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Linear(n_features, 32)
        self.lstm = nn.LSTM(input_size=32, hidden_size=hidden,
                            num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.hidden = hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, F) -> z: (B, hidden)
        x = torch.tanh(self.proj(x))
        out, _ = self.lstm(x)
        return out[:, -1, :]   # last-step hidden state


class MultiTaskModel(nn.Module):
    """Encoder + two task heads. Training drops the heads; only encoder.pt is
    kept for downstream use."""
    def __init__(self, n_features: int = 4, hidden: int = 64,
                 layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.encoder = LSTMEncoder(n_features, hidden, layers, dropout)
        self.ret_head = nn.Linear(hidden, 1)
        self.vol_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        return self.ret_head(z).squeeze(-1), self.vol_head(z).squeeze(-1), z
