"""Second-order reflex dynamics for ecological overlays.

Forager and autotomy both act on pressures that can change faster than the
main strategy loop. Level thresholds catch obvious cases; these helpers add
the reflex layer: is pressure rising, accelerating, and likely to cross soon?
"""
from __future__ import annotations

import math
import time
from typing import Optional


def _clip(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def dynamics(prev: dict, name: str, level: float, *,
             threshold: Optional[float] = None,
             now: Optional[float] = None,
             fast_tau_s: float = 6.0,
             slow_tau_s: float = 30.0) -> dict:
    """Return first/second-order state for one scalar pressure.

    Keys are prefixed with `name`, for example `autotomy_pressure_velocity`.
    Velocity/acceleration are per second; delta is the raw change since the
    last sample, which is easier to reason about in cycle diagnostics.
    """
    now = float(now if now is not None else time.time())
    level = float(level or 0.0)
    prev_level = float(prev.get(f"{name}_prev", level) or 0.0)
    prev_velocity = float(prev.get(f"{name}_velocity", 0.0) or 0.0)
    prev_ts = float(prev.get(f"{name}_ts_epoch", now) or now)
    dt = _clip(now - prev_ts, 1.0, 300.0)

    delta = level - prev_level
    velocity = delta / dt
    acceleration = (velocity - prev_velocity) / dt

    fast_prev = float(prev.get(f"{name}_ema_fast", prev_level) or 0.0)
    slow_prev = float(prev.get(f"{name}_ema_slow", prev_level) or 0.0)
    af = 1.0 - math.exp(-dt / max(1e-6, fast_tau_s))
    aslow = 1.0 - math.exp(-dt / max(1e-6, slow_tau_s))
    ema_fast = fast_prev + af * (level - fast_prev)
    ema_slow = slow_prev + aslow * (level - slow_prev)
    impulse = ema_fast - ema_slow

    ttt = None
    if threshold is not None and velocity > 1e-9 and level < threshold:
        ttt = max(0.0, (float(threshold) - level) / velocity)

    return {
        f"{name}_prev": round(level, 6),
        f"{name}_delta": round(delta, 6),
        f"{name}_velocity": round(velocity, 6),
        f"{name}_acceleration": round(acceleration, 6),
        f"{name}_ema_fast": round(ema_fast, 6),
        f"{name}_ema_slow": round(ema_slow, 6),
        f"{name}_impulse": round(impulse, 6),
        f"{name}_time_to_threshold_s": round(ttt, 1) if ttt is not None else None,
        f"{name}_ts_epoch": now,
    }


def rising_fast(dyn: dict, name: str, *,
                min_delta: float = 0.0,
                min_velocity: float = 0.0,
                min_acceleration: Optional[float] = None,
                min_impulse: Optional[float] = None,
                max_time_to_threshold_s: Optional[float] = None) -> bool:
    """Whether a pressure has enough positive motion to act before threshold."""
    delta = float(dyn.get(f"{name}_delta") or 0.0)
    velocity = float(dyn.get(f"{name}_velocity") or 0.0)
    acceleration = float(dyn.get(f"{name}_acceleration") or 0.0)
    impulse = float(dyn.get(f"{name}_impulse") or 0.0)
    ttt = dyn.get(f"{name}_time_to_threshold_s")

    if delta < min_delta and velocity < min_velocity:
        return False
    if min_acceleration is not None and acceleration < min_acceleration:
        return False
    if min_impulse is not None and impulse < min_impulse:
        return False
    if max_time_to_threshold_s is not None and ttt is not None:
        return float(ttt) <= max_time_to_threshold_s
    return True
