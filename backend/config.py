"""Configuration, credential loading and persisted settings for the
BTC Perp Trading Console.

Nothing in here ever prints secret material. The RSA private key is loaded
into memory only and never written to the state store or logs.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
STATE_DIR = ROOT / "state"
STATE_DIR.mkdir(exist_ok=True)

# Where the Kalshi API key id + RSA private key live. Format expected:
#   line 1: API key id (UUID)
#   following lines: -----BEGIN RSA PRIVATE KEY----- ... -----END ... -----
DEFAULT_KEY_FILE = ROOT / "trading_API_keys.txt"

# Kalshi REST base URLs (margin / perps share the trade-api/v2 prefix).
PROD_BASE = "https://external-api.kalshi.com/trade-api/v2"
DEMO_BASE = "https://external-api.demo.kalshi.co/trade-api/v2"

# BTC perp contract (ticker KXBTCPERP): 1 contract = 0.0001 BTC = 1/10,000 BTC.
# Price field is per-contract USD.
DEFAULT_TICKER = "KXBTCPERP"
CONTRACT_BTC_FRACTION = 1.0 / 10_000.0


@dataclass
class Credentials:
    key_id: str
    private_key_pem: str  # held in memory only

    @property
    def loaded(self) -> bool:
        return bool(self.key_id and self.private_key_pem)


def load_credentials(path: Path | str = DEFAULT_KEY_FILE) -> Optional[Credentials]:
    """Parse the key file. Returns None if the file is missing/unparseable.

    The first non-empty, non-PEM line is treated as the key id; the PEM block
    is captured verbatim.
    """
    p = Path(path)
    if not p.exists():
        return None
    text = p.read_text(encoding="utf-8")
    lines = text.splitlines()
    key_id = ""
    for ln in lines:
        s = ln.strip()
        if not s:
            continue
        if s.startswith("-----BEGIN"):
            break
        key_id = s
        break
    begin = text.find("-----BEGIN")
    end_marker = "-----END"
    end = text.find(end_marker)
    pem = ""
    if begin != -1 and end != -1:
        end_line_end = text.find("\n", end)
        if end_line_end == -1:
            end_line_end = len(text)
        pem = text[begin:end_line_end].strip() + "\n"
    if not key_id or not pem:
        return None
    return Credentials(key_id=key_id, private_key_pem=pem)


# ---------------------------------------------------------------------------
# Persisted, user-editable settings. These are safe to serialize (no secrets).
# ---------------------------------------------------------------------------

# Parameters consumed by signals.py (single source of truth for live/backtest/tune).
DEFAULT_STRATEGY_PARAMS = {
    # --- mpc: cost-aware MPC controller (primary). Continuous target position
    # with a no-trade band -> low turnover; survives realistic costs at ~breakeven
    # (validated by walk-forward). vol_win/lookback/beta shared with mr_edge. ---
    "gain": 1.0,        # aim gain on the z-scored reversion alpha
    "band": 0.5,        # no-trade band half-width (cost-sizing)
    "regime_win": 8,    # bars for the efficiency-ratio regime scaler
    "er_cap": 1.0,      # efficiency above which reversion aim is fully scaled out
    # --- mr_edge mean-reversion params (also feed the MPC alpha) ---
    "vol_win": 12,
    "lookback": 2,
    "beta": 0.18,
    "k": 1.2,
    "z_cap": 3.5,
    "deadband_bps": 1.0,
    # Trend / regime (other variants)
    "ema_fast": 8,
    "ema_slow": 30,
    "ema_trend": 100,
    "trend_gate": 0.1,
    # Mean reversion (classic variant)
    "rsi_period": 14,
    "rsi_overbought": 75.0,
    "rsi_oversold": 25.0,
    # Momentum / breakout
    "roc_period": 6,
    "roc_threshold": 0.1,
    "donchian": 20,
    # Volatility (ATR%) gating
    "atr_period": 14,
    "min_atr_pct": 0.02,
    "max_atr_pct": 5.0,
    # --- Ecology / mycelial controller (see backend/ecology.py). The phase
    # classifier and organism scorer use these knobs; the multiplier table
    # maps each phase to a size multiplier applied to MPC |target_fraction|
    # when settings.ecosystem_phase is on. ---
    "ecology_win": 64,               # rolling bars used to compute network baselines
    "ecology_vol_win": 10,           # window for realized vol
    "ecology_te_lag": 1,             # lag (bars) for lagged-correlation TE proxy
    "ecology_disturbance_z": 1.5,    # |sum-of-z-shifts| above this = "disturbance"
    "ecology_predator_mult": 0.2,    # cascade in progress -> shrink
    "ecology_exhaustion_mult": 0.6,  # predator slowing -> probe
    "ecology_scavenger_mult": 1.5,   # snap-back zone -> accelerate
    "ecology_decomposer_mult": 0.5,  # liquidity recovering -> unwind
    "ecology_churn_mult": 1.0,       # baseline (normal regime)
    "ecology_producer_mult": 1.0,    # quiet liquidity provision phase
}


@dataclass
class Settings:
    environment: str = "production"      # production | demo
    ticker: str = DEFAULT_TICKER
    timeframe: str = "5m"                # MPC validated best on 5m
    leverage_target: float = 5.8
    strategy: str = "robust_mpc"         # robust_mpc | mpc | mr_edge | regime | meanrev | trend | breakout | momentum
    assumed_fee_bps: float = 0.4         # maker-ish cost used for tuning/backtest realism
    model: str = "claude-sonnet-4-6"
    mode: str = "live_autonomous"        # live_autonomous | dry_run

    # Behavioural switches (mirror the UI checkboxes)
    use_actual_balance: bool = True
    allow_live_orders: bool = False      # becomes True only after explicit arm
    let_opus_decide: bool = False   # LLM trade verifier REMOVED (degraded perf) — deterministic only
    opus_only_on_signal: bool = True     # only spend an Opus call on candidate trades (non-HOLD)
    auto_submit_orders: bool = True
    do_not_ask_again: bool = False
    trust_strategy: bool = False

    # The single one-time arm flag for autonomous live trading.
    live_autonomous_armed: bool = False
    kill_switch_engaged: bool = False

    loop_interval_seconds: int = 300     # legacy; used as max when vol_aware_interval is on
    maker_mode: bool = True              # post-only limit orders (capture, don't pay, spread)

    # --- Vol-aware loop interval. When on, the loop sleeps a shorter interval
    # while ATR% is hot and a longer one while quiet, so the bot reacts fast
    # during real moves without burning API calls on dead markets.
    # interpolated linearly between (atr_pct_for_max -> interval_seconds_max)
    # and (atr_pct_for_min -> interval_seconds_min). Below max-threshold: max
    # interval. Above min-threshold: min interval. Each aggressiveness preset
    # writes its own bounds.
    vol_aware_interval: bool = True
    interval_seconds_min: int = 60       # floor — used when ATR% is hot
    interval_seconds_max: int = 300      # ceiling — used when ATR% is quiet
    atr_pct_for_min: float = 0.5         # at/above this ATR%, hit min interval
    atr_pct_for_max: float = 0.05        # at/below this, sit at max interval

    # --- Maker/taker switch (validated by hybrid backtest, signals.simulate_hybrid)
    # urgency = |alpha| from the MPC controller; if urgency >= taker_threshold_k
    # the executor crosses (IOC taker), else it posts (maker). chase_delay_bars
    # is the cycles a maker can rest unfilled before being escalated to a taker
    # chase. taker_threshold_k=None collapses to pure-maker (today's behavior);
    # k=0 is all-taker (validated NEGATIVE — don't do it). Defaults are
    # placeholders; the sweep script writes the winning combo here. ---
    taker_threshold_k: Optional[float] = None
    chase_delay_bars: int = 999
    half_spread_bps: float = 7.0         # used for sim + live cost realism
    # --- Aggressive override #1: drift-aware chase. The normal chase clock is
    # wall-clock time since the maker was placed; but when price walks away from
    # us we cancel-and-reprice every cycle, which resets that clock and the
    # chase never fires (the bot just chases the book passively forever, losing
    # the move). chase_after_drifts counts reprices on the same desired side and
    # crosses once we've drifted this many times. 999=off (no drift-based
    # promotion). 1 = cross on the first reprice. ---
    chase_after_drifts: int = 999
    # --- Aggressive override #2: realized-move detector. When the most recent
    # bar's high-low range (in bps of mid) >= this, force taker for the cycle.
    # 0.0=off. Set ~8 in "ultra" so we cross during fast moves rather than
    # resting and getting left behind. ---
    move_force_taker_bps: float = 0.0
    # --- Maker-fill escalation: after this many consecutive zero-fill maker
    # submits on the same side, escalate the NEXT submit to taker. This is
    # the cleanest defense against the trending-market failure mode where a
    # fresh maker every cycle keeps repricing to the wrong side of a one-way
    # book and never gets a fill. 0=off. 3-5 is reasonable on conservative.
    consecutive_zerofill_taker_at: int = 0
    # --- Aggressive override #3: tactical adverse-move reversal. When we
    # hold a position and price has jumped against it by >= this many bps
    # (current mid vs close of previous bar), the engine flips the proposal
    # to a full-conviction REVERSE_* on the opposite side and marks
    # force_taker so the executor crosses immediately. Capitalizes on the
    # shift rather than just dodging it. 0.0=off.
    #
    # CALIBRATION NOTE: on KXBTCPERP at ~$6.59/contract, one tick = 1.5 bps,
    # and normal 3m bar moves are 2-5 bps. Anything <15 bps fires on noise.
    # Use 20-30 bps for "real adverse move", combined with the cooldown
    # below so a single move doesn't trigger a back-to-back flip storm.
    adverse_move_bps: float = 0.0
    # Cycles to wait after firing before adverse-move can fire again. Stops
    # the override from re-flipping every cycle when a slow walk drifts back
    # and forth across the threshold. 0 = no cooldown. ---
    adverse_move_cooldown_cycles: int = 5

    # --- Trophic Information Forager / Mycelial Alpha Network. When on, the
    # ecology module builds a live information network over on-platform nodes
    # (BTC perp price, volatility, spread, depth, OI, volume, funding, plus
    # multi-asset nodes when available — BTC spot via Coinbase, ETH perp via
    # Kalshi, BTC liquidations via Binance), classifies the current ecological
    # phase (producer/predator/exhaustion/scavenger/decomposer/churn), scores
    # six organisms (predator, scavenger, decomposer, mycelium router,
    # immune, producer/market-maker), and modulates the MPC's target_fraction
    # by the phase multiplier. Always computed for the UI; only APPLIED to
    # live sizing when this flag is on. The "wildest" version: a self-
    # observing information ecosystem instead of a price predictor.
    ecosystem_phase: bool = False

    # --- CSD risk governor (validated by ablate_csd_refined.py on Kalshi 3m
    # 4-fold walk-forward). Uses the skew-only refined signal — abs(skewness)
    # of the price-vs-EMA-fair-value log-deviation series, z-scored against
    # trailing history and sigmoid-squashed. When enabled, zeros the MPC
    # target_fraction on cycles where risk > threshold. Validated config:
    # threshold=0.95 gave aggregate +7.02% vs +4.63% baseline with max_dd
    # 4.89% vs 5.48% on the Kalshi 3m 4-fold WF (3 of 4 folds improved).
    # Off by default; enabled in aggressive/ultra presets.
    csd_governor_enabled: bool = False
    csd_governor_threshold: float = 0.95
    csd_governor_window: int = 96
    csd_governor_fv_period: int = 32
    # Adaptive CSD threshold: when realized vol (ATR%) is high, the skew
    # distribution shifts and the 0.95 baseline (calibrated at ~0.08% ATR)
    # never triggers. When atr_pct >= csd_governor_atr_breakpoint, the
    # effective threshold drops to csd_governor_threshold_high_vol so the
    # gate actually fires during fast regimes. Set high_vol == threshold
    # to disable the adaptive behaviour.
    csd_governor_atr_breakpoint: float = 0.20
    csd_governor_threshold_high_vol: float = 0.80
    # Predictive CSD: gate on short-horizon projected risk, not only the
    # current skew level. Uses first/second-order risk dynamics plus EMA impulse
    # so the governor can fire while the transition is forming.
    csd_predictive_enabled: bool = True
    csd_predictive_horizon_seconds: float = 12.0
    csd_predictive_impulse_gain: float = 0.25

    # --- Blended directional alpha (see backend/signals_blended.py +
    # backend/backtest_blended.py). When enabled, the LATEST cycle's
    # MPC alpha is blended with the spot-lead / funding-fade / OI-pressure
    # components. Validated on 4-fold WF Kalshi 3m: w_spot_lead=0.30
    # lifted aggregate return +14.3pp and sharpe +0.85 over MPC-alone,
    # with lower max-dd. Funding and OI components are wired but
    # default-zero pending historical-data backtests.
    signal_blend_enabled: bool = True
    blend_w_mpc: float = 1.0
    blend_w_spot_lead: float = 0.30
    blend_w_funding_fade: float = 0.0   # unvalidated; awaiting historical data
    blend_w_oi_pressure: float = 0.0    # unvalidated; awaiting historical data
    blend_w_ecology_flow: float = 0.0    # network-state alpha; validated before activation
    blend_w_visual: float = 0.0         # visual-review trend REMOVED from the blend (degraded perf)
    blend_spot_lookback: int = 3
    blend_spot_history_for_std: int = 60
    blend_funding_history_for_std: int = 60
    blend_funding_persistence_threshold: float = 0.5
    blend_oi_lookback: int = 3
    blend_oi_history_for_std: int = 60
    blend_ecology_lookback: int = 3
    blend_ecology_condition_spot_lead: bool = True
    # Visual-trend conviction mapping: concern level -> tilt magnitude. The
    # review's `trend` sets direction (up=+, down=-, sideways/choppy=0); the
    # `concern` level scales how hard it tilts — OK is a mild lean, STOP is the
    # chart-reader shouting an obvious sustained move (and also fires the entry
    # gate). Tilt = direction * conviction, fed into the blend at blend_w_visual.
    blend_visual_conviction_ok: float = 0.34
    blend_visual_conviction_caution: float = 0.67
    blend_visual_conviction_stop: float = 1.0

    # --- Foraging cycle: ecological profit-harvest + refractory (forager.py).
    # The bot works an edge, HARVESTS profit when the ecological edge is
    # consumed (not on a timer), then RESTS (blocks new entries, allows exits)
    # until the ecosystem resets or a new disturbance appears. DEFAULT OFF —
    # unbacktested heuristic; enable deliberately and watch live. Diagnostics
    # (pnl_R, harvest_pressure, cooldown) populate STORE.forager_state even
    # when off, so you can preview behaviour before arming it.
    forager_enabled: bool = False
    # "How hungry" the forager is — a single selectable knob instead of the raw
    # parameters. Resolves to a bundle of harvest thresholds + cooldown lengths
    # in forager.FORAGER_HUNGER_PRESETS. Levels: lazy / steady / hungry /
    # ravenous. Default RAVENOUS: eats profit fast and rests only ~5-10 min.
    forager_hunger: str = "ravenous"
    # Autoscale the hunger preset by the live regime / network-dynamic state:
    # a brittle, disturbed, critically-slowing market (high disturbance / CSD /
    # ascendancy, low reserve) makes the forager eat sooner, deeper, and rest
    # less; a calm/recovering one makes it patient. See forager._autoscale.
    forager_autoscale: bool = True
    # Cooldown duration override (UI-adjustable). 0 = dynamic (hunger preset +
    # autoscale). >0 = fixed N-second HARD refractory that blocks new entries
    # for the whole duration (no early re-exit). forager_min_rest_frac is the
    # fraction of a DYNAMIC cooldown that must elapse before any early re-exit
    # is allowed — without it the persistently-high scavenger/disturbance in
    # churn clears the cooldown instantly and it never blocks entries.
    forager_cooldown_seconds: int = 0
    forager_min_rest_frac: float = 0.6
    # Sub-cycle harvest: let the fast-guard thread (~1s polls) run a read-only
    # forager check at the live mid and fire an immediate harvest cycle the
    # instant profit crosses a tier — so the forager acts on the second instead
    # of waiting for the next full 3-20s trade cycle. Requires fast_guard_enabled.
    forager_fast_harvest: bool = True
    forager_predictive_enabled: bool = True
    forager_predictive_horizon_seconds: float = 12.0
    forager_predictive_impulse_gain: float = 0.35
    forager_reflex_enabled: bool = True
    forager_reflex_pressure_threshold: float = 1.25
    forager_reflex_soft_pressure: float = 0.45
    forager_reflex_min_profit_R: float = 0.05
    forager_reflex_min_pressure_delta: float = 0.10
    forager_reflex_min_edge_decay_delta: float = 0.04
    forager_reflex_max_ttt_s: float = 20.0
    forager_reflex_scale: float = 0.50
    # The fields below are the legacy raw knobs. They are NO LONGER read by the
    # forager (the hunger preset drives everything) — kept only so old
    # settings.json files still load. Tune via forager_hunger instead.
    # Harvest tiers (pnl_R = profit in per-bar-ATR-of-notional units). Most
    # aggressive (smallest scale) wins. Thresholds lowered 2026-06-15 so it
    # harvests SOONER — churn profit evaporates before pnl_R can build, and
    # high ATR in churn suppresses pnl_R, so the bars must be low.
    #   disturbance >= churn_disturbance AND pnl_R >= churn_profit_R -> churn_scale (CHURN DEFENSE)
    #   pnl_R >= profit_threshold_R       AND edge_decay >= edge_decay_trim -> trim to 75%
    #   pnl_R >= tier2_profit_R           AND decomposer >= thr       -> trim to 40%
    #   pnl_R >= tier3_profit_R           AND edge <= edge_gone        -> full close
    #   csd risk rising AND >= csd_high                              -> full close
    forager_profit_threshold_R: float = 0.25   # tier-1 min profit (was 0.5)
    forager_tier2_profit_R: float = 0.5        # tier-2 min profit (was 1.0)
    forager_tier3_profit_R: float = 0.75       # tier-3 min profit (was 1.5)
    forager_edge_decay_trim: float = 0.4       # tier-1 edge_decay floor (was 0.5)
    forager_decomposer_threshold: float = 0.5  # (was 0.6)
    forager_edge_ref: float = 0.30        # expected_edge_pct that = "full edge" (for edge_decay)
    forager_edge_gone: float = 0.10       # edge at/below this = consumed (was 0.05)
    forager_csd_high: float = 0.7         # CSD risk level that triggers aggressive close (was 0.8)
    forager_pnl_R_ref: float = 1.0        # normalizer for realized_profit_score in HarvestPressure
    # CHURN DEFENSE (the primary fix): in a disturbed/choppy regime profit is
    # temporary, so flatten on a LOW profit bar the moment disturbance is up.
    # churn_scale 0.0 = full close; raise toward 0.4 to keep a runner. This is
    # eager by design — disturbance >= 0.5 is common, so any profitable position
    # in a disturbed market gets harvested. Soften via these three knobs.
    forager_churn_disturbance: float = 0.5     # disturbance_score that means "churn"
    forager_churn_profit_R: float = 0.2        # min profit to harvest in churn (low — ATR suppresses pnl_R here)
    forager_churn_scale: float = 0.0           # target after churn harvest (0.0 = full close)
    # Ecological cooldown: base * (1+reserve_recovery) * (1+decomposer) * (1-disturbance)
    forager_base_cooldown_seconds: int = 300
    forager_cooldown_min_seconds: int = 60
    forager_cooldown_max_seconds: int = 1800
    # Early cooldown exit (re-enable entries) when any of these fire:
    forager_reentry_disturbance: float = 0.7   # new disturbance
    forager_reentry_scavenger: float = 0.6     # snap-back signal returns
    forager_reentry_edge: float = 0.20         # MR permission rebuilt
    forager_reentry_fair_stretch: float = 0.5  # price back in fair-value reset zone (|stretch_z|)

    # --- Autotomy Agent: ecological loss-shedding reflex (backend/autotomy.py).
    # Symmetric to the forager, but for bad positions instead of consumed
    # winners. The forager says "this niche has been harvested"; autotomy says
    # "this position has become bait." When enabled, a losing position is hard
    # exited only when loss is paired with toxic ecology: predator/cascade risk,
    # CSD/skew stress, reserve collapse, failed recovery, and/or edge flip.
    # Diagnostics populate STORE.autotomy_state even when disabled.
    autotomy_enabled: bool = False
    autotomy_aggression: str = "ravenous"  # lazy / steady / hungry / ravenous
    autotomy_use_raw_thresholds: bool = False
    autotomy_pressure_threshold: float = 3.0
    autotomy_loss_R: float = 0.35
    autotomy_min_confirmations: int = 3
    autotomy_cooldown_seconds: int = 0    # 0 = preset cooldown; >0 fixed seconds
    autotomy_fast_eject: bool = True
    autotomy_predictive_enabled: bool = True
    autotomy_predictive_horizon_seconds: float = 12.0
    autotomy_predictive_impulse_gain: float = 0.35
    autotomy_reflex_enabled: bool = True
    autotomy_reflex_soft_pressure: float = 2.2
    autotomy_reflex_loss_R: float = 0.18
    autotomy_reflex_min_confirmations: int = 2
    autotomy_reflex_min_pressure_delta: float = 0.15
    autotomy_reflex_max_ttt_s: float = 18.0

    # --- Fast guard: sub-cycle polling thread (backend/fast_guard.py). The
    # main loop sleeps for vol_aware_interval seconds between cycles; this
    # thread polls STORE.live_quote every fast_guard_poll_seconds and
    # triggers an immediate cycle when intra-cycle mid moves more than
    # fast_guard_emergency_move_bps. Catches the "fast move happens between
    # cycles" failure mode where the resilience layers don't see it until
    # the next scheduled cycle fires.
    fast_guard_enabled: bool = True
    fast_guard_poll_seconds: int = 1
    fast_guard_emergency_move_bps: float = 10.0

    # --- Visual chart review (see backend/visual_review.py). At engine
    # startup and every N seconds, render the recent candle chart with
    # current position + recent decisions annotated, send it to Opus with
    # vision, and get back {trend, concern, note}. When concern == STOP
    # and the next proposal is a new entry, the engine blocks it (existing
    # positions are never force-flattened by this layer).
    # Default ON; requires ANTHROPIC_API_KEY. Cost is roughly a few dollars
    # per day at a 10-min cadence on opus-4-8 — disable for low-cost runs.
    visual_review_enabled: bool = False   # visual chart tracker REMOVED (degraded perf)
    visual_review_interval_seconds: int = 600        # 10 minutes
    visual_review_block_entries_on_stop: bool = True
    # Also gate on CAUTION, not just STOP. Useful in volatile regimes where
    # the bot's mean-reversion edge is most likely to fade against a sustained
    # one-way move — Opus typically calls those CAUTION not STOP because they
    # aren't catastrophic, but they're exactly when we want to slow entries.
    visual_review_block_entries_on_caution: bool = True
    # Haiku 4.5 for the image review: it has vision, it's the cheapest tier,
    # and the chart sanity-check ("is the bot fighting an obvious trend?") is a
    # coarse visual judgement, not a fine-grained quantitative one. This is the
    # ONE call kept off Sonnet — it runs most often (periodic + CSD-fire), so
    # the cheap tier matters most here. Everything else (trade decisions) is
    # Sonnet; nothing uses Opus.
    visual_review_model: str = "claude-haiku-4-5-20251001"
    # Use the review's trend as a directional blend contributor (in addition
    # to the STOP/CAUTION entry gate above). When on, the latest {trend,
    # concern} is mapped to a tilt (see blend_visual_conviction_*) and added
    # to the blended alpha at weight blend_w_visual. Gated on signal_blend_enabled.
    visual_review_as_signal: bool = False   # visual not fed to the blend (tracker removed)
    # Drop the visual tilt if the latest review is older than this. The review
    # refreshes every visual_review_interval_seconds but the loop runs more
    # often; this stops a stale/failed review from tilting indefinitely.
    visual_signal_max_age_seconds: int = 1800
    # Event trigger: also fire a review the moment the CSD / ecological-
    # resilience governor is firing (risk >= effective threshold), not just on
    # the periodic timer — catches fast regime breaks between scheduled reviews.
    # `event_min_interval` is a SEPARATE, shorter cooldown that bounds cost: a
    # sustained CSD storm fires at most one review per this many seconds (not
    # per cycle). Set it below visual_review_interval_seconds for any effect.
    visual_review_on_csd_fire: bool = True
    visual_review_event_min_interval_seconds: int = 60

    # --- Tiered decision model. The deterministic strategy proposal goes
    # through an LLM risk overlay before execution. Haiku 4.5 is fast and
    # cheap; it's plenty for confirming small, validated-edge trades. Sonnet
    # 4.6 is reserved for larger trades where the LLM's judgement might
    # actually matter. "Large" = |target_fraction - current_fraction| >=
    # large_trade_fraction_threshold, where the fractions are normalized
    # to max buying power. 0.4 means a >= 40%-of-max-buying-power move.
    # `model` (the legacy single-model setting above) is retained as the
    # fallback when these are blank. Set either to "" to fall back.
    model_small_trade: str = "claude-sonnet-4-6"   # all LLM calls on Sonnet (never Opus/Haiku)
    model_large_trade: str = "claude-sonnet-4-6"
    large_trade_fraction_threshold: float = 0.4

    # Pull external data sources for the ecology network. All are public
    # endpoints (no keys). If any feed is unreachable, the ecology degrades
    # gracefully (that node drops out of the network for that cycle).
    ecology_use_coinbase_spot: bool = True   # BTC-USD + ETH-USD spot
    ecology_use_coinbase_book: bool = True   # Coinbase top-book imbalance/spread
    ecology_use_kalshi_eth: bool = True      # KXETHPERP from existing Kalshi auth
    # Hyperliquid (BTC perp DEX) for OI / funding / mark price. Replaced the
    # old Binance integration on 2026-06-15 — Binance fapi.binance.com returns
    # HTTP 451 (geo-block) from US networks. Hyperliquid has a public
    # POST /info endpoint with no auth and no geo-block, and returns all the
    # fields the ecology consumed from Binance (openInterest, funding,
    # markPx, midPx, dayNtlVlm) in a single call.
    #
    # Setting kept as both names: ecology_use_hyperliquid is the new canonical
    # name. ecology_use_binance_liq is preserved so old saved settings.json
    # files continue to parse and toggle the same source. Default ON because
    # the data source actually works now.
    ecology_use_hyperliquid: bool = True
    ecology_use_binance_liq: bool = True
    ecology_use_hyperliquid_breadth: bool = True
    ecology_use_deribit: bool = True
    ecology_use_kraken_dispersion: bool = True
    ecology_use_crypto_breadth: bool = True
    ecology_use_mempool: bool = True

    # Auto-retraining: refresh params when the edge degrades or on a schedule.
    # The tuner only APPLIES params that pass its out-of-sample gate, so a bad
    # regime can't push bad params live.
    auto_retrain: bool = True
    retrain_drawdown_pct: float = 8.0    # live drawdown from peak -> retrain
    retrain_every_cycles: int = 480      # also retrain on a schedule (~1 day at 3m)

    # ---- Automatic, non-prompting guardrails (adjustable; some can be off) ----
    max_leverage: float = 5.8            # hard cap on effective leverage
    max_position_notional_usd: float = 0.0   # 0 => derived from equity*leverage
    daily_loss_limit_usd: float = 0.0        # 0 => derived as 25% of equity
    daily_loss_limit_enabled: bool = True
    min_liquidation_buffer_pct: float = 8.0  # block opening if closer than this
    min_account_equity_usd: float = 25.0     # refuse to trade below this
    min_risk_increase_edge_pct: float = 0.06 # block risk-adds only if edge and urgency are both weak
    min_risk_increase_urgency: float = 0.03  # weak single side is a warning, not a hard stop
    adaptive_signal_floor_enabled: bool = True
    adaptive_signal_floor_min_mult: float = 0.55
    adaptive_signal_floor_max_mult: float = 2.20
    mpc_confidence_sizing_enabled: bool = True
    mpc_confidence_full_at: float = 0.55      # confidence where MPC target is allowed at full size
    mpc_confidence_power: float = 1.5         # >1 sharply shrinks weak-confidence risk adds
    mpc_min_confidence_scale: float = 0.0     # 0 => no minimum clip for weak entries
    harvest_reserve_enabled: bool = False    # risk-control toggle; backtest did not prove profit alpha
    harvest_reserve_fraction: float = 1.0    # 1.0 = reserve all captured forager profit
    harvest_reserve_reset_daily: bool = True # clear the reserve when the UTC trading day rolls

    # Order behaviour
    limit_offset_bps: float = 2.0        # cross the spread by this for fast fills
    max_spread_bps: float = 60.0         # warn threshold (not a hard block)
    reduce_only_on_exit: bool = True

    # Trade aggressiveness preset selector. UI flips this; the matching entry
    # in AGGRESSIVENESS_PRESETS is applied via /api/aggressiveness.
    aggressiveness: str = "conservative"

    # Trade-size preset selector. Independent of aggressiveness — controls how
    # much of available buying power each trade uses. UI flips this; the
    # TRADE_SIZE_PRESETS entry is applied via /api/trade_size.
    trade_size: str = "standard"

    # Multiplier on the MPC controller's target_fraction. Applied in
    # risk.size_position before the max_leverage hard clamp, so Heavy=2.0
    # pushes the bot toward fuller utilization without ever exceeding the cap.
    position_scale: float = 1.0

    strategy_params: dict = field(default_factory=lambda: dict(DEFAULT_STRATEGY_PARAMS))

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Settings":
        base = cls()
        for k, v in (d or {}).items():
            if hasattr(base, k):
                setattr(base, k, v)
        # Ensure strategy params always have all keys
        merged = dict(DEFAULT_STRATEGY_PARAMS)
        merged.update(base.strategy_params or {})
        base.strategy_params = merged
        return base


# Trade-aggressiveness presets. Each level patches a small set of live params.
# Conservative == the validated edge (mostly maker, wide deadband). Moderate
# tightens MPC band + deadband but remains maker-first. Aggressive lowers band
# further AND lets the executor cross the spread (finite taker_threshold_k)
# on high-urgency cycles. Each preset deliberately writes the whole execution
# personality; otherwise stale ultra fields in settings.json can survive a UI
# switch back to moderate and silently turn a maker strategy into a taker one.
AGGRESSIVENESS_PRESETS = {
    "conservative": {
        "strategy": "robust_mpc",
        "taker_threshold_k": None,
        "chase_delay_bars": 4,
        "chase_after_drifts": 999,
        "consecutive_zerofill_taker_at": 4,
        "move_force_taker_bps": 0.0,
        "adverse_move_bps": 0.0,
        # CSD governor off for the validated maker baseline — the strategy
        # already trades infrequently and rarely takes on tail risk. The
        # governor's edge is on aggressive/ultra cycles.
        "csd_governor_enabled": False,
        # FAST-POLL bounds (lowered 2026-06-15). Old bounds 60-300s let
        # 3m-bar regime shifts go undetected because ATR / CSD / ecology
        # only updated once per cycle. With Haiku at the small-trade tier
        # (~$0.003/cycle) and live-quote MTM applied inside the cycle, we
        # can now poll the full resilience stack on a ~5-30s cadence and
        # actually catch a fast move before it eats equity.
        "interval_seconds_min": 5,
        "interval_seconds_max": 30,
        # Low call volume — Opus 4.8 for best risk-overlay nuance.
        "model": "claude-sonnet-4-6",
        "strategy_params": {"gain": 1.0, "band": 0.5, "deadband_bps": 1.0},
    },
    "moderate": {
        "strategy": "robust_mpc",
        "taker_threshold_k": None,
        "chase_delay_bars": 3,
        "chase_after_drifts": 999,
        "consecutive_zerofill_taker_at": 3,
        # Directional recent-move taker override. This now uses signed
        # close-to-close pressure in executor.py, so it only crosses when the
        # move agrees with the intended order side. 18 bps is above the
        # overnight p99-ish one-minute drift, so normal chop stays maker.
        "move_force_taker_bps": 18.0,
        "adverse_move_bps": 0.0,
        "csd_governor_enabled": False,
        "interval_seconds_min": 3,
        "interval_seconds_max": 20,
        "model": "claude-sonnet-4-6",
        "strategy_params": {"gain": 1.0, "band": 0.3, "deadband_bps": 0.5},
    },
    "aggressive": {
        "strategy": "robust_mpc",
        "taker_threshold_k": 0.5,
        "chase_delay_bars": 2,
        "chase_after_drifts": 3,
        "consecutive_zerofill_taker_at": 2,
        "move_force_taker_bps": 14.0,
        "adverse_move_bps": 25.0,
        # CSD governor on at the validated threshold — aggressive presets size
        # up and cross spreads more often, so the drawdown-shaping benefit is
        # most valuable here. Conservative/moderate stay OFF.
        "csd_governor_enabled": True,
        "csd_governor_threshold": 0.95,
        "interval_seconds_min": 1,
        "interval_seconds_max": 60,
        # High call volume — Sonnet 4.6 (~5x cheaper) is plenty for the risk-overlay
        # JSON contract, and the validated edge runs without LLM input anyway.
        "model": "claude-sonnet-4-6",
        "strategy_params": {"gain": 1.2, "band": 0.15, "deadband_bps": 0.0},
    },
    # Ultra: takes the maker baseline off entirely on conviction signals.
    # k=0.05 means nearly any non-trivial urgency crosses; chase=0 means a
    # resting maker is escalated to taker on the very next cycle; the two
    # drift/move overrides catch sustained moves that would otherwise leave us
    # repricing forever (which is what the live logs were doing — see
    # 00:42-00:55 sustained walk with zero fills). Expect higher fill rate,
    # higher cost; only worth it when you'd rather pay the spread than miss
    # the trade.
    "ultra": {
        "strategy": "robust_mpc",
        "taker_threshold_k": 0.05,
        "chase_delay_bars": 0,
        "chase_after_drifts": 1,
        "consecutive_zerofill_taker_at": 1,
        "move_force_taker_bps": 8.0,
        "csd_governor_enabled": True,
        "csd_governor_threshold": 0.95,
        # Adverse-move reversal — 25 bps is ~17 ticks on KXBTCPERP, well
        # above the 2-5 bps normal-bar noise. Combined with the cooldown,
        # this fires on real moves and stays quiet on chop.
        "adverse_move_bps": 25.0,
        "adverse_move_cooldown_cycles": 5,
        "interval_seconds_min": 1,
        "interval_seconds_max": 30,
        "model": "claude-sonnet-4-6",
        "strategy_params": {"gain": 1.4, "band": 0.10, "deadband_bps": 0.0},
    },
}


# Trade-size presets. Multiplies the MPC controller's target_fraction (the
# fraction of max buying power it asks for) — Light sizes down, Heavy sizes up.
# Leverage cap (settings.max_leverage = 5.8, Kalshi's platform ceiling) still
# hard-clamps the resulting position, so Heavy cannot push past it; what changes
# is how often the bot actually approaches the cap vs leaving capital idle.
TRADE_SIZE_PRESETS = {
    "light":    {"position_scale": 0.5},
    "standard": {"position_scale": 1.0},
    "heavy":    {"position_scale": 2.0},
}


def bar_seconds(timeframe: str) -> int:
    """Convert a timeframe like '5m'/'1h' to seconds. Falls back to 300s."""
    tf = (timeframe or "").lower().strip()
    try:
        if tf.endswith("m"):
            return max(1, int(tf[:-1])) * 60
        if tf.endswith("h"):
            return max(1, int(tf[:-1])) * 3600
    except ValueError:
        pass
    return 300


def compute_loop_interval(settings: "Settings", atr_pct: Optional[float]) -> int:
    """Linearly interpolate the loop interval between max (quiet) and min
    (hot) based on ATR%. Returns seconds, clamped to [min, max]."""
    smin = max(1, int(settings.interval_seconds_min))
    smax = max(smin, int(settings.interval_seconds_max))
    if not settings.vol_aware_interval or atr_pct is None:
        return smax
    hi = float(settings.atr_pct_for_min)   # ATR% at which we hit min interval
    lo = float(settings.atr_pct_for_max)   # ATR% at/below which we use max
    if hi <= lo:
        return smax
    a = max(lo, min(hi, float(atr_pct)))
    # frac=0 at lo (quiet -> max), frac=1 at hi (hot -> min)
    frac = (a - lo) / (hi - lo)
    return int(round(smax + (smin - smax) * frac))


ANTHROPIC_KEY_FILE = STATE_DIR / "anthropic_key.txt"


def anthropic_api_key() -> Optional[str]:
    """Env var takes precedence; otherwise a gitignored local secret file."""
    env = os.environ.get("ANTHROPIC_API_KEY")
    if env:
        return env
    if ANTHROPIC_KEY_FILE.exists():
        v = ANTHROPIC_KEY_FILE.read_text(encoding="utf-8").strip()
        return v or None
    return None
