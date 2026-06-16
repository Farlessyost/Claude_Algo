"""Hybrid LSTM research pipeline. Pretrain an encoder on years of BTC spot,
freeze it, train a small head on the much smaller Kalshi KXBTCPERP history.

NOT wired into the live engine. Run from the command line:
  python -m backend.pretrain.fetch_spot
  python -m backend.pretrain.train
  python -m backend.pretrain.head
  python -m backend.pretrain.eval
"""
