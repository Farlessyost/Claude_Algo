"""Liquid encoder using ncps (Hasani et al.) — wraps LTC or CfC as a drop-in
for LSTMEncoder. CfC is the closed-form continuous-time variant of LTC and
is the practical choice on CPU; LTC ODE integration at L=256, hidden=96 is
~10s/batch and infeasible.

Same interface as encoder.LSTMEncoder: forward(x) -> (B, hidden) final hidden.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn as nn
from ncps.torch import CfC, LTC
from ncps.wirings import AutoNCP


@dataclass
class LiquidConfig:
    n_features: int = 4
    hidden: int = 96
    horizon: int = 16
    kind: Literal["cfc", "ltc"] = "cfc"
    wiring: Literal["fc", "ncp"] = "fc"
    ncp_output_size: int = 24
    ncp_sparsity: float = 0.5
    ncp_seed: int = 42


class LiquidEncoder(nn.Module):
    def __init__(self, cfg: LiquidConfig):
        super().__init__()
        self.cfg = cfg
        if cfg.wiring == "ncp":
            wiring = AutoNCP(
                units=cfg.hidden,
                output_size=cfg.ncp_output_size,
                sparsity_level=cfg.ncp_sparsity,
                seed=cfg.ncp_seed,
            )
            rnn_units = wiring
        elif cfg.wiring == "fc":
            rnn_units = cfg.hidden
        else:
            raise ValueError(f"unknown wiring: {cfg.wiring}")

        if cfg.kind == "cfc":
            self.rnn = CfC(cfg.n_features, rnn_units)
        elif cfg.kind == "ltc":
            self.rnn = LTC(cfg.n_features, rnn_units)
        else:
            raise ValueError(f"unknown kind: {cfg.kind}")

        # ncps returns final hidden of size = total units, regardless of wiring,
        # so downstream projection size is cfg.hidden either way.
        self.proj = nn.Linear(cfg.hidden, cfg.hidden)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(cfg.hidden)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.rnn(x)
        z = self.ln(self.act(self.proj(h)))
        return z
