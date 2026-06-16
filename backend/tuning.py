"""Tuning pipeline. Randomized search over the selected variant's parameters,
scored by the shared backtest. Runs in a background thread, reports progress,
and on completion writes the best params straight into live settings so the
autonomous loop uses them with no manual copy step.

To avoid overfitting to today's promotional zero fees, tuning scores trades with
a small assumed fee (settings.assumed_fee_bps) by default."""
from __future__ import annotations

import random
import threading
import time
from typing import List, Optional

from . import backtest
from .store import STORE

# Search ranges aligned to signals.py parameters.
SEARCH_SPACE = {
    # mpc controller params
    "gain": [0.5, 1.0, 2.0, 3.0],
    "band": [0.1, 0.2, 0.35, 0.5],
    "regime_win": [8, 12, 20],
    "er_cap": [0.4, 0.6, 1.0],
    "robust_lambda": [0.15, 0.35, 0.6, 0.9],
    "robust_disturbance_lambda": [0.2, 0.45, 0.8, 1.2],
    # mr_edge params
    "vol_win": [12, 24, 48],
    "lookback": [1, 2, 3],
    "beta": [0.12, 0.18, 0.25, 0.35],
    "k": [0.8, 1.2, 1.8, 2.4],
    "z_cap": [2.0, 2.8, 3.5],
    "deadband_bps": [0.0, 1.0, 2.0, 4.0],
    # other variants
    "ema_fast": [5, 8, 12, 20],
    "ema_slow": [20, 30, 50],
    "ema_trend": [50, 100, 150],
    "atr_period": [10, 14, 20],
    "min_atr_pct": [0.0, 0.02, 0.05],
    "rsi_period": [9, 14, 21],
    "rsi_oversold": [20.0, 25.0, 30.0],
    "rsi_overbought": [70.0, 75.0, 80.0],
    "roc_period": [3, 6, 10],
    "roc_threshold": [0.05, 0.1, 0.2],
    "donchian": [10, 20, 40],
    "trend_gate": [0.05, 0.1, 0.2],
}


class Tuner:
    def __init__(self):
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()

    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self, candles: List[dict], iterations: int = 60,
              auto_apply: bool = True, leverage: float = 5.8,
              variant: str = "regime", fee_pct: float = 0.0003) -> dict:
        if self.is_running():
            return {"status": "already_running"}
        if len(candles) < 120:
            return {"status": "error", "detail": "not enough history to tune"}
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._run,
            args=(candles, iterations, auto_apply, leverage, variant, fee_pct),
            daemon=True)
        self._thread.start()
        return {"status": "started", "iterations": iterations, "variant": variant}

    def stop(self):
        self._stop.set()

    def _run(self, candles, iterations, auto_apply, leverage, variant, fee_pct):
        start = time.time()
        # Validate out-of-sample: tune on first 70%, report the held-out 30%.
        split = int(len(candles) * 0.7)
        train, test = candles[:split], candles[split:]
        STORE.tuning.update({
            "status": "running", "tested": 0, "total": iterations,
            "started_ts": time.time(), "best_score": None, "best_params": None,
            "applied": False, "current": None, "eta_s": None,
            "what": f"Search {len(SEARCH_SPACE)} params for '{variant}' "
                    f"(train {len(train)}/test {len(test)} bars, fee {fee_pct*1e4:.1f}bps)",
        })
        STORE.save_tuning()
        STORE.log("info", "tuning", f"Tuning '{variant}': {iterations} candidates")

        best_score = None; best_params = None; best_train = None
        cur = dict(STORE.settings.strategy_params)
        candidates = [cur]
        for _ in range(iterations - 1):
            cand = dict(cur)
            cand.update({k: random.choice(v) for k, v in SEARCH_SPACE.items()})
            candidates.append(cand)

        for i, params in enumerate(candidates):
            if self._stop.is_set():
                STORE.tuning["status"] = "stopped"; STORE.save_tuning()
                STORE.log("warn", "tuning", "Tuning stopped"); return
            tr = backtest.run_backtest(train, params, variant, leverage, fee_pct)
            score = tr.get("score")
            if score is not None and (best_score is None or score > best_score):
                best_score, best_params, best_train = score, params, tr

            elapsed = time.time() - start; done = i + 1
            eta = (elapsed / done) * (iterations - done) if done else None
            STORE.tuning.update({
                "tested": done, "total": iterations,
                "elapsed_s": round(elapsed, 1), "eta_s": round(eta, 1) if eta else None,
                "best_score": best_score, "best_params": best_params,
                "current": {"score": score, "return": tr.get("total_return_pct")},
            })
            if done % 5 == 0 or done == iterations:
                STORE.save_tuning()

        # Out-of-sample check of the winner
        oos = backtest.run_backtest(test, best_params, variant, leverage, fee_pct) if best_params else {}
        STORE.tuning["status"] = "complete"
        STORE.tuning["elapsed_s"] = round(time.time() - start, 1)
        STORE.tuning["eta_s"] = 0
        STORE.tuning["oos"] = {"return_pct": oos.get("total_return_pct"),
                               "max_dd_pct": oos.get("max_drawdown_pct"),
                               "trades": oos.get("trades"), "sharpe": oos.get("sharpe")}
        STORE.tuning["best_report"] = best_train

        # Only auto-apply if the winner is also positive OUT-OF-SAMPLE.
        if auto_apply and best_params and (oos.get("total_return_pct") or -1) > 0:
            STORE.update_settings({"strategy_params": best_params})
            STORE.tuning["applied"] = True
            STORE.log("info", "tuning",
                      f"Applied params: train {best_train.get('total_return_pct')}% / "
                      f"OOS {oos.get('total_return_pct')}% ({variant})")
        else:
            STORE.tuning["applied"] = False
            STORE.log("warn", "tuning",
                      f"NOT applied — OOS return {oos.get('total_return_pct')}% not positive")
        STORE.save_tuning()
        STORE.log("info", "tuning", "Tuning complete")


TUNER = Tuner()
