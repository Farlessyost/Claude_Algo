"use strict";
const $ = (id) => document.getElementById(id);
const fmt = (n, d = 2) => (n === null || n === undefined || isNaN(n))
  ? "—" : Number(n).toLocaleString(undefined, {minimumFractionDigits: d, maximumFractionDigits: d});
const usd = (n, d = 2) => (n === null || n === undefined || isNaN(n)) ? "—" : "$" + fmt(n, d);
const pct = (n, d = 2) => (n === null || n === undefined || isNaN(n)) ? "—" : fmt(n, d) + "%";

let CURRENT_TAB = "events";
let SETTINGS = {};

async function api(path, body) {
  const opt = {method: body === undefined ? "GET" : "POST",
    headers: {"Content-Type": "application/json"}};
  if (body !== undefined) opt.body = JSON.stringify(body);
  const r = await fetch(path, opt);
  return r.json();
}

function msg(text, kind) {
  const el = $("control-msg");
  el.textContent = text;
  el.className = "msg " + (kind || "");
}

async function patchSettings(patch) {
  SETTINGS = await api("/api/settings", {patch});
}

// ---------------- wiring ----------------
function wire() {
  $("sel-timeframe").onchange = (e) => patchSettings({timeframe: e.target.value});
  $("sel-strategy").onchange = (e) => patchSettings({strategy: e.target.value});
  $("sel-mode").onchange = (e) => patchSettings({mode: e.target.value});
  $("inp-leverage").onchange = (e) => patchSettings({leverage_target: parseFloat(e.target.value)});
  $("sel-model").onchange = (e) => patchSettings({model: e.target.value});

  const cmap = {"chk-balance":"use_actual_balance","chk-live":"allow_live_orders",
    "chk-opus":"let_opus_decide","chk-auto":"auto_submit_orders",
    "chk-noask":"do_not_ask_again","chk-trust":"trust_strategy",
    "chk-ecosystem":"ecosystem_phase",
    "chk-csd":"csd_governor_enabled",
    "chk-forager-enabled":"forager_enabled",
    "chk-autotomy-enabled":"autotomy_enabled"};
  for (const [id, key] of Object.entries(cmap)) {
    if ($(id)) $(id).onchange = (e) => patchSettings({[key]: e.target.checked});
  }
  wireForagerHunger();

  $("btn-connect").onclick = async () => {
    msg("Connecting…");
    const r = await api("/api/connect", {});
    msg(r.connected ? "Connected ✓ " + (r.detail||"") : "Connection failed: " + (r.detail||""),
        r.connected ? "ok" : "err");
    refresh();
  };

  $("btn-enable").onclick = async () => {
    if (!confirm("Enable LIVE autonomous trading?\n\nThe bot will place REAL orders on your "
      + "production Kalshi margin account at up to 5.8x leverage WITHOUT asking again. "
      + "You can stop it anytime with Stop or Kill Switch.\n\nArm live trading now?")) return;
    const r = await api("/api/enable_live", {});
    msg(r.armed ? "LIVE AUTONOMOUS TRADING ARMED ⚡ — bot can now trade real orders." : "Failed",
        r.armed ? "ok" : "err");
    refresh();
  };
  $("btn-startloop").onclick = async () => {
    const sel = $("sel-loop-duration");
    const durMin = sel && sel.value ? parseFloat(sel.value) : null;
    const body = durMin ? {duration_minutes: durMin} : {};
    const isContinuous = !durMin;
    if (isContinuous && !confirm(
        "CONTINUOUS TRADING: the loop will run until you press Stop or Kill. "
        + "There is NO time limit.\n\nThe bot will keep placing real orders on your "
        + "Kalshi margin account at the configured aggressiveness/leverage. Make sure "
        + "Live Autonomous Trading is armed and the kill switch is clear.\n\nStart now?"))
      return;
    const r = await api("/api/start_loop", body);
    const label = isContinuous ? "Continuous trading started — runs until stopped ▶"
      : `Bounded run started — auto-stops in ${durMin} min ▶`;
    msg(r.error ? r.error : label, r.error ? "err" : "ok");
    refresh();
  };
  $("btn-stop").onclick = async () => { await api("/api/stop", {}); msg("Trading stopped.","ok"); refresh(); };
  $("btn-kill").onclick = async () => {
    if (!confirm("KILL SWITCH: halt the loop, disarm live trading, and cancel resting orders. Continue?")) return;
    await api("/api/kill", {}); msg("KILL SWITCH ENGAGED ⛔","err"); refresh();
  };
  $("btn-runonce").onclick = async () => {
    msg("Running one live decision cycle…");
    const r = await api("/api/run_once", {});
    msg("Cycle complete: " + (r.verb || r.error || r.skipped || "done"), r.error ? "err":"ok");
    refresh();
  };
  $("btn-backtest").onclick = async () => {
    msg("Backtesting on " + (SETTINGS.timeframe||"") + " candles…");
    const r = await api("/api/backtest", {});
    if (r.error) { msg("Backtest: " + r.error, "err"); return; }
    msg(`Backtest: return ${pct(r.total_return_pct)} · maxDD ${pct(r.max_drawdown_pct)} · `
      + `${r.trades} trades · win ${pct((r.win_rate||0)*100)}`, "ok");
  };
  const tune = async () => {
    const r = await api("/api/tune", {iterations: 60, auto_apply: true});
    msg(r.status === "started" ? "Tuning started — best params auto-apply to live."
      : "Tuning: " + (r.detail || r.status), r.status==="started"?"ok":"err");
  };
  $("btn-tune").onclick = tune;
  $("btn-tune2").onclick = tune;
  $("btn-tune-stop").onclick = async () => { await api("/api/tune/stop", {}); };
  $("btn-save").onclick = async () => { await api("/api/save_settings", {}); msg("Settings saved ✓","ok"); };

  document.querySelectorAll(".tab").forEach(t => t.onclick = () => {
    document.querySelectorAll(".tab").forEach(x => x.classList.remove("active"));
    t.classList.add("active"); CURRENT_TAB = t.dataset.tab; renderLogs(LAST_LOGS);
  });

  document.querySelectorAll("button.agg").forEach(b => b.onclick = async () => {
    const level = b.dataset.level;
    if (level === "aggressive" && !confirm(
        "AGGRESSIVE: the bot will cross the spread on conviction signals and "
        + "trade on much smaller signals. The validated edge is sub-spread, so "
        + "this can turn the strategy NEGATIVE. Risk-overlay model also "
        + "switches to Sonnet 4.6 (cheaper, ~equivalent for this task).\n\n"
        + "Continue?")) return;
    if (level === "ultra" && !confirm(
        "ULTRA: takes the maker baseline off entirely. Crosses on nearly any "
        + "non-trivial signal (k=0.05), crosses on the first reprice instead "
        + "of resting (chase_after_drifts=1), and force-crosses when the most "
        + "recent bar's move exceeds 8bps (move_force_taker_bps=8). Expect "
        + "many more fills but you'll pay the half-spread on most of them, "
        + "which can turn the strategy NEGATIVE versus Aggressive. Use only "
        + "when missing trades on big moves is worse than paying the spread.\n\n"
        + "Continue?")) return;
    document.querySelectorAll("button.agg").forEach(x => {
      x.classList.toggle("active", x === b);
      x.setAttribute("aria-pressed", x === b ? "true" : "false");
    });
    b.classList.add("saving");
    msg(`Switching to ${level}…`);
    try {
      const r = await api("/api/aggressiveness", {level});
      msg(r.error ? r.error : `Aggressiveness -> ${level.toUpperCase()} ✓`,
          r.error ? "err" : "ok");
    } catch (e) {
      msg("Aggressiveness switch failed: " + e, "err");
    } finally {
      b.classList.remove("saving");
    }
    refresh();
  });

  document.querySelectorAll("button.size").forEach(b => b.onclick = async () => {
    const level = b.dataset.level;
    if (level === "heavy" && !confirm(
        "HEAVY: doubles the MPC controller's sizing — positions approach the "
        + "5.8x leverage cap far more often, so most of your equity is at work "
        + "on any open trade. Cap still hard-clamps (can't exceed 5.8x), and "
        + "liquidation buffer / daily-loss limit / min equity still apply, but "
        + "per-trade equity at risk is materially larger. Combined with "
        + "AGGRESSIVE this is the bot's highest-utilization mode.\n\nContinue?"))
      return;
    document.querySelectorAll("button.size").forEach(x => {
      x.classList.toggle("active", x === b);
      x.setAttribute("aria-pressed", x === b ? "true" : "false");
    });
    b.classList.add("saving");
    msg(`Switching trade size to ${level}…`);
    try {
      const r = await api("/api/trade_size", {level});
      msg(r.error ? r.error : `Trade size -> ${level.toUpperCase()} ✓`,
          r.error ? "err" : "ok");
    } catch (e) {
      msg("Trade size switch failed: " + e, "err");
    } finally {
      b.classList.remove("saving");
    }
    refresh();
  });
}

// ---------------- render ----------------
function setPill(id, text, cls) { const e = $(id); e.textContent = text; e.className = "pill " + (cls||""); }

let LAST_LOGS = {};

function renderPanel(name, fn) {
  try {
    fn();
  } catch (e) {
    console.error(`Render failed: ${name}`, e);
  }
}

function render(s) {
  SETTINGS = s.settings;
  const conn = s.connection || {};
  setPill("hdr-conn", conn.connected ? "connected" : "offline", conn.connected ? "ok":"bad");
  setPill("hdr-env", SETTINGS.environment, SETTINGS.environment==="production"?"prod":"");
  setPill("hdr-mode", SETTINGS.mode === "live_autonomous" ? "Live Autonomous" : "Dry Run",
    SETTINGS.mode==="live_autonomous"?"warnp":"");
  const live = SETTINGS.live_autonomous_armed && SETTINGS.allow_live_orders;
  setPill("hdr-live", live ? "ON" : "OFF", live ? "on" : "");
  setPill("hdr-kill", SETTINGS.kill_switch_engaged ? "ENGAGED" : "clear",
    SETTINGS.kill_switch_engaged ? "bad" : "ok");
  // Loop cycle pill: shows current effective interval + ATR%. Color tier
  // matches how reactive the loop is right now.
  const ls = s.loop_status || {};
  const intvl = ls.interval_seconds;
  const atr = ls.atr_pct;
  let cycleText = "—", cycleCls = "";
  if (intvl != null) {
    const intvlStr = intvl < 60 ? `${intvl}s` : `${Math.round(intvl/60)}m`;
    cycleText = atr != null ? `${intvlStr} · ATR ${(atr).toFixed(2)}%` : intvlStr;
    cycleCls = intvl <= 10 ? "hot" : intvl <= 60 ? "warm" : "cool";
  }
  setPill("hdr-cycle", cycleText, cycleCls);
  $("hdr-refresh").textContent = new Date(s.ts).toLocaleTimeString();

  // reflect settings into controls (only if not focused)
  syncControl("sel-timeframe", SETTINGS.timeframe);
  syncControl("sel-strategy", SETTINGS.strategy);
  syncControl("sel-mode", SETTINGS.mode);
  syncControl("sel-model", SETTINGS.model);
  if (document.activeElement !== $("inp-leverage")) $("inp-leverage").value = SETTINGS.leverage_target;
  $("chk-balance").checked = SETTINGS.use_actual_balance;
  $("chk-live").checked = SETTINGS.allow_live_orders;
  $("chk-opus").checked = SETTINGS.let_opus_decide;
  $("chk-auto").checked = SETTINGS.auto_submit_orders;
  $("chk-noask").checked = SETTINGS.do_not_ask_again;
  $("chk-trust").checked = SETTINGS.trust_strategy;
  $("chk-ecosystem").checked = !!SETTINGS.ecosystem_phase;
  $("chk-csd").checked = !!SETTINGS.csd_governor_enabled;
  const aggLevel = SETTINGS.aggressiveness || "conservative";
  document.querySelectorAll("button.agg").forEach(b => {
    const on = b.dataset.level === aggLevel;
    b.classList.toggle("active", on);
    b.setAttribute("aria-pressed", on ? "true" : "false");
  });
  const sizeLevel = SETTINGS.trade_size || "standard";
  document.querySelectorAll("button.size").forEach(b => {
    const on = b.dataset.level === sizeLevel;
    b.classList.toggle("active", on);
    b.setAttribute("aria-pressed", on ? "true" : "false");
  });

  renderPanel("account", () => renderAccount(s.account, s));
  renderPanel("market", () => renderMarket(s.market));
  renderPanel("ecosystem", () => renderEcosystem(s));
  renderPanel("csd", () => renderCsd(s));
  renderPanel("forager", () => renderForager(s));
  renderPanel("alpha", () => renderAlphaDecomp(s));
  renderPanel("fast_guard", () => renderFastGuard(s));
  renderPanel("decision", () => renderDecision(s.last_decision, s));
  renderPanel("risk", () => renderRisk(s.last_decision));
  renderPanel("tuning", () => renderTuning(s.tuning));
  renderPanel("llm_usage", () => renderLlmUsage(s.llm_usage));
  LAST_LOGS = s.logs || {};
  renderPanel("logs", () => renderLogs(LAST_LOGS));
}

function syncControl(id, val) { const e = $(id); if (document.activeElement !== e && val !== undefined) e.value = val; }

function kv(grid, rows) {
  $(grid).innerHTML = rows.map(([k, v, cls]) =>
    `<div class="k">${k}</div><div class="v ${cls||""}">${v}</div>`).join("");
}

function renderAccount(a, s) {
  a = a || {};
  const dayPnl = (a.equity!=null && s.day_start_equity!=null) ? a.equity - s.day_start_equity : null;
  const dd = (a.equity!=null && s.max_equity) ? (s.max_equity - a.equity)/s.max_equity*100 : null;
  const sgn = (n) => n==null?"":(n>=0?"pos":"neg");
  kv("account-grid", [
    ["Account Equity", usd(a.equity)],
    ["Available Balance", usd(a.available_balance)],
    ["Margin Used", usd(a.margin_used)],
    ["Position", `${fmt(a.position_contracts)} (${a.position_direction||"FLAT"})`],
    ["Notional Exposure", usd(a.notional_exposure)],
    ["Open Orders", a.open_orders_count ?? 0],
    ["Effective Leverage", fmt(a.effective_leverage) + "x"],
    ["Leverage Target", fmt(SETTINGS.leverage_target,1) + "x"],
    ["Realized P&L (pos fees)", usd(a.fees ? -a.fees : 0)],
    ["Unrealized P&L", usd(a.unrealized_pnl), sgn(a.unrealized_pnl)],
    ["Daily P&L", usd(dayPnl), sgn(dayPnl)],
    ["Current Drawdown", pct(dd)],
    ["Margin Safety", a.margin_safety || "—"],
    ["Liquidation Risk", a.liquidation_risk || "—"],
    ["Last Refresh", a.refreshed_ts ? new Date(a.refreshed_ts).toLocaleTimeString() : "—"],
  ]);
}

function renderMarket(m) {
  m = m || {}; const ob = m.orderbook || {}; const f = m.features || {};
  kv("market-grid", [
    ["BTC Perp Price", usd(m.price, 4)],
    ["~BTC Spot", usd(m.btc_spot_estimate, 0)],
    ["Bid", usd(ob.best_bid, 4)],
    ["Ask", usd(ob.best_ask, 4)],
    ["Spread", usd(ob.spread, 4)],
    ["Spread (bps)", fmt(ob.spread_bps, 1)],
    ["Orderbook Depth", fmt(ob.depth_total, 0)],
    ["ATR (volatility)", pct(f.atr_pct)],
    ["RSI", fmt(f.rsi, 1)],
    ["Regime", f.regime || "—"],
    ["Candle Size", m.timeframe || SETTINGS.timeframe],
    ["Cycle Refresh", m.refreshed_ts ? new Date(m.refreshed_ts).toLocaleTimeString() : "—"],
    ["Live Tick",   m.live_quote_ts ? new Date(m.live_quote_ts).toLocaleTimeString() : "—"],
  ]);
  const sig = (label, v) => {
    const cls = v==="bullish"?"bull":v==="bearish"?"bear":"neu";
    return `<span class="sig ${cls}">${label}: ${v||"—"}</span>`;
  };
  $("signals").innerHTML = sig("Trend", f.trend) + sig("Momentum", f.momentum)
    + sig("MeanRev", f.meanrev) + `<span class="sig neu">Regime: ${f.regime||"—"}</span>`;
  drawChart(m.candles || []);
}

function renderDecision(d, s) {
  d = d || {};
  const verb = d.verb || "—";
  const conf = (d.confidence||0)*100;
  $("decision-head").innerHTML =
    `<div class="verb ${verb}">${verb}</div>`
    + `<div class="conf-bar"><i style="width:${conf}%"></i></div>`
    + `<div class="mono">${fmt(conf,0)}%</div>`;
  const sz = d.sizing || {};
  kv("decision-body", [
    ["Model Confidence", pct(conf)],
    ["Expected Edge", pct(d.expected_edge_pct)],
    ["Expected Profit", d.expected_profit_usd!=null ? usd(d.expected_profit_usd) : "—"],
    ["Expected Risk", d.expected_risk_usd!=null ? usd(d.expected_risk_usd) : "—"],
    ["Proposed Δ Size", fmt(sz.delta_contracts) + " contracts"],
    ["Target Position", fmt(sz.target_contracts) + " contracts"],
    ["Proposed Notional", usd(sz.target_notional)],
    ["Proposed Limit Price", usd(sz.limit_price, 4)],
    ["Eff. Leverage After", fmt(sz.target_leverage) + "x"],
    ["Action Type", d.action || "—"],
    ["Strategy Proposal", d.proposal_action || "—"],
    ["Strategy Target", d.proposal_target_fraction!=null
        ? fmt(d.proposal_target_fraction, 3) : "—"],
    ["Base Target", d.proposal_target_fraction_base!=null
        ? fmt(d.proposal_target_fraction_base, 3) : "—"],
    ["Decided By", d.source ? (d.source + (d.model_note?` · ${d.model_note}`:"")) : "—"],
    ["Execution", d.execution ? (d.execution.submitted ? "SUBMITTED ✓"
        : (d.execution.note||"no order")) : "—"],
  ]);
  const li = (arr) => (arr||[]).map(x=>`<li>${x}</li>`).join("") || "<li>—</li>";
  const ecoNote = (d.ecology_note || "").trim();
  $("decision-reasons").innerHTML =
    `<h4>Reasons For</h4><ul class="reasons-for">${li(d.reasons_for)}</ul>`
    + `<h4>Reasons Against</h4><ul class="reasons-against">${li(d.reasons_against)}</ul>`
    + `<h4>Strategy Proposal</h4><ul class="reasons-for">${li(d.proposal_rationale_for)}</ul>`
    + `<h4>Why better than HOLD</h4><div>${d.why_better_than_hold||"—"}</div>`
    + (ecoNote ? `<h4>Ecology Note (from LLM overlay)</h4><div class="eco-note-line">${ecoNote}</div>` : "");

  // profit panel
  kv("profit-grid", [
    ["Selected Action", verb],
    ["Confidence-Weighted EV", pct((d.expected_edge_pct||0)*(d.confidence||0))],
    ["Expected Edge", pct(d.expected_edge_pct)],
    ["Risk-Adjusted Return", d.expected_profit_usd!=null && d.expected_risk_usd ?
        fmt(d.expected_profit_usd/Math.max(d.expected_risk_usd,1e-6),2) : "—"],
    ["Projected Notional", usd(sz.target_notional)],
    ["Why > HOLD", verb!=="HOLD" ? "positive expected edge" : "no positive edge now"],
  ]);
  const missed = d.missed_opportunity;
  const me = $("missed");
  if (missed) { me.textContent = "⚠ Missed-opportunity: " + missed; me.classList.add("show"); }
  else me.classList.remove("show");
}

function renderRisk(d) {
  const checks = (d && d.checks && d.checks.checks) || [];
  $("risk-grid").innerHTML = checks.map(c =>
    `<div class="row"><span><span class="dot ${c.status}"></span>${c.name}</span>`
    + `<span class="rdetail">${c.detail}</span></div>`).join("") || "<div class='rdetail'>No cycle run yet.</div>";
}

function renderLlmUsage(u) {
  u = u || {};
  const tot = u.totals || {};
  const h1 = u.last_1h || {}; const d1 = u.last_24h || {};
  // "Hourly rate" = cost actually billed in the trailing 60 min; matches what
  // you'd see on the Anthropic dashboard. Extrapolation to per-day uses the
  // 24h actual, not 24 * the hourly snapshot (less misleading on bursty load).
  kv("llm-grid", [
    ["Calls — last hour",  h1.calls ?? 0],
    ["Cost — last hour",   usd(h1.cost_usd, 3)],
    ["Calls — last 24h",   d1.calls ?? 0],
    ["Cost — last 24h",    usd(d1.cost_usd, 2)],
    ["Calls — lifetime",   tot.calls ?? 0],
    ["Cost — lifetime",    usd(tot.cost_usd, 2)],
    ["Tokens in — lifetime",  fmt(tot.tokens_input ?? 0, 0)],
    ["Tokens out — lifetime", fmt(tot.tokens_output ?? 0, 0)],
  ]);
  const by = tot.by_model || {};
  const rows = Object.entries(by)
    .sort((a, b) => (b[1].cost_usd || 0) - (a[1].cost_usd || 0))
    .map(([name, v]) => [name, `${v.calls} · ${usd(v.cost_usd, 2)}`]);
  $("llm-by-model").innerHTML = rows.length
    ? rows.map(([k, v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join("")
    : `<div class="rdetail">No calls yet.</div>`;
  $("llm-note").textContent = "Cost-gated: Opus/Sonnet only called on non-HOLD signals. HOLDs cost $0.";
}

function renderTuning(t) {
  t = t || {};
  kv("tuning-grid", [
    ["Status", t.status || "idle"],
    ["What", t.what || "—"],
    ["Tested / Total", `${t.tested||0} / ${t.total||0}`],
    ["Remaining", `${Math.max((t.total||0)-(t.tested||0),0)}`],
    ["Elapsed", t.elapsed_s!=null ? fmt(t.elapsed_s,0)+"s" : "—"],
    ["Est. Remaining", t.eta_s!=null ? fmt(t.eta_s,0)+"s" : "—"],
    ["Best Score", t.best_score!=null ? fmt(t.best_score,4) : "—"],
    ["Auto-applied to live", t.applied ? "YES ✓" : "no"],
  ]);
  const pctDone = t.total ? (t.tested/t.total*100) : 0;
  $("tune-bar").style.width = pctDone + "%";
  $("tuning-params").textContent = t.best_params ? JSON.stringify(t.best_params, null, 1) : "";
}

function renderLogs(logs) {
  logs = logs || {};
  const data = logs[CURRENT_TAB] || [];
  const fmtLine = (e) => {
    if (CURRENT_TAB === "events")
      return `<div class="line"><span class="t">${new Date(e.ts).toLocaleTimeString()}</span>`
        + `<span class="lvl ${e.level}">${e.level}</span><span>${e.kind}: ${e.message}</span></div>`;
    if (CURRENT_TAB === "decisions")
      return `<div class="line"><span class="t">${new Date(e.ts).toLocaleTimeString()}</span>`
        + `<span class="lvl info">${e.verb}</span><span>conf ${fmt((e.confidence||0)*100,0)}% · `
        + `edge ${pct(e.expected_edge_pct)} · ${e.execution&&e.execution.submitted?"SUBMITTED":"(no order)"}</span></div>`;
    if (CURRENT_TAB === "orders")
      return `<div class="line"><span class="t">${new Date(e.ts).toLocaleTimeString()}</span>`
        + `<span class="lvl info">${e.side}</span><span>${fmt(e.count)} @ ${usd(e.price,4)} `
        + `· filled ${fmt(e.fill_count)} · ${e.event||""}</span></div>`;
    if (CURRENT_TAB === "fills")
      return `<div class="line"><span class="t">${new Date(e.ts||Date.now()).toLocaleTimeString()}</span>`
        + `<span>${JSON.stringify(e)}</span></div>`;
    if (CURRENT_TAB === "pnl_history")
      return `<div class="line"><span class="t">${new Date(e.ts).toLocaleTimeString()}</span>`
        + `<span>equity ${usd(e.equity)} · uPnL ${usd(e.unrealized)}</span></div>`;
    return "";
  };
  $("log-body").innerHTML = data.map(fmtLine).join("") || "<div class='rdetail'>No entries.</div>";
}

// ---------------- Trophic Information Forager / Mycelial Network ----------
const PHASE_ORDER = ["producer","predator","exhaustion","scavenger","decomposer","churn"];
const PHASE_COLORS = {
  producer:   "#22c55e",
  predator:   "#ef4444",
  exhaustion: "#f59e0b",
  scavenger:  "#38bdf8",
  decomposer: "#a78bfa",
  churn:      "#94a3b8",
};
const ORG_LABEL = {
  predator:   "Predator",
  scavenger:  "Scavenger",
  decomposer: "Decomposer",
  mycelium:   "Mycelium",
  immune:     "Immune",
  producer:   "Producer",
};
// Fixed circular layout for food-web nodes so they don't jitter between cycles.
// The set of POSSIBLE nodes is fixed; nodes that don't exist this cycle are
// rendered semi-transparent. This keeps the visual stable as feeds come/go.
const WEB_NODES = ["live_mid","live_spread_bps","live_imb_top","live_imb_depth","live_depth_total",
                   "price","vol","range","volume","spread","oi_change",
                   "btc_spot","eth_spot","eth_perp","liq_proxy","funding_bn",
                   "cb_btc_depth_imbalance","cb_btc_spread_bps",
                   "deribit_perp_basis","deribit_term_basis","deribit_oi_change",
                   "deribit_funding_8h","deribit_option_skew",
                   "kraken_coinbase_basis_bps","kraken_spread_bps","kraken_eth_btc_lead",
                   "hl_alt_breadth","hl_total_oi_change","hl_btc_book_imbalance","hl_premium_z",
                   "crypto_total_mcap_change","btc_dominance_change","top10_breadth",
                   "risk_on_alt_breadth","mempool_congestion_z","btc_fee_pressure"];
const WEB_NODE_LABEL = {
  price:"BTC perp", vol:"vol", range:"range", volume:"volume",
  live_mid:"1s mid", live_spread_bps:"1s spread", live_imb_top:"1s top",
  live_imb_depth:"1s depth", live_depth_total:"1s liq",
  spread:"spread", oi_change:"OI", btc_spot:"BTC spot",
  eth_spot:"ETH spot", eth_perp:"ETH perp", liq_proxy:"liquidations",
  funding_bn:"funding", cb_btc_depth_imbalance:"CB depth",
  cb_btc_spread_bps:"CB spread", deribit_perp_basis:"DB perp",
  deribit_term_basis:"DB term", deribit_oi_change:"DB OI",
  deribit_funding_8h:"DB fund", deribit_option_skew:"DB skew",
  kraken_coinbase_basis_bps:"Kr basis", kraken_spread_bps:"Kr spread",
  kraken_eth_btc_lead:"Kr ETH/BTC", hl_alt_breadth:"HL breadth",
  hl_total_oi_change:"HL OI", hl_btc_book_imbalance:"HL book",
  hl_premium_z:"HL prem", crypto_total_mcap_change:"mcap",
  btc_dominance_change:"BTC dom", top10_breadth:"top10",
  risk_on_alt_breadth:"alt risk", mempool_congestion_z:"mempool",
  btc_fee_pressure:"BTC fees",
};
const WEB_NODE_SCALE = {
  live_mid:"seconds", live_spread_bps:"seconds", live_imb_top:"seconds",
  live_imb_depth:"seconds", live_depth_total:"seconds",
  cb_btc_depth_imbalance:"seconds", cb_btc_spread_bps:"seconds",
  cb_eth_depth_imbalance:"seconds", hl_btc_book_imbalance:"seconds",
  price:"minutes", vol:"minutes", range:"minutes", volume:"minutes",
  spread:"minutes", oi_change:"minutes", btc_spot:"minutes",
  eth_spot:"minutes", eth_perp:"minutes", liq_proxy:"minutes",
  funding_bn:"minutes", deribit_perp_basis:"minutes",
  deribit_term_basis:"minutes", deribit_oi_change:"minutes",
  kraken_coinbase_basis_bps:"minutes", kraken_spread_bps:"minutes",
  kraken_eth_btc_lead:"minutes", hl_alt_breadth:"minutes",
  hl_total_oi_change:"minutes", hl_premium_z:"minutes",
  deribit_funding_8h:"hours", deribit_option_skew:"hours",
  top10_breadth:"hours", risk_on_alt_breadth:"hours",
  mempool_congestion_z:"hours", btc_fee_pressure:"hours",
  crypto_total_mcap_change:"days", btc_dominance_change:"days",
};
const WEB_CENTER = {x: 210, y: 160};
const WEB_SCALE_RADIUS = {days:38, hours:78, minutes:118, seconds:150};
const WEB_SCALE_ANGLE_OFFSET = {
  days: -Math.PI / 2,
  hours: -Math.PI / 2 + 0.38,
  minutes: -Math.PI / 2 + 0.16,
  seconds: -Math.PI / 2 - 0.08,
};

function _scaleForNode(name, eco) {
  const sub = ((eco || {}).multiscale || {}).subnetworks || {};
  for (const scale of ["seconds","minutes","hours","days"]) {
    if (((sub[scale] || {}).nodes || []).includes(name)) return scale;
  }
  return WEB_NODE_SCALE[name] || "minutes";
}

function _webNodesForScale(scale, eco) {
  const active = new Set(((eco || {}).nodes || []));
  const known = WEB_NODES.filter(n => active.has(n) && _scaleForNode(n, eco) === scale);
  const extras = Array.from(active)
    .filter(n => _scaleForNode(n, eco) === scale && !known.includes(n))
    .sort();
  return known.concat(extras);
}

function _webNodePos(name, eco) {
  const scale = _scaleForNode(name, eco);
  const nodes = _webNodesForScale(scale, eco);
  const idx = nodes.indexOf(name);
  if (idx < 0) return null;
  const n = Math.max(1, nodes.length);
  const R = WEB_SCALE_RADIUS[scale] || 103;
  const offset = WEB_SCALE_ANGLE_OFFSET[scale] || -Math.PI / 2;
  const angle = offset + (idx / n) * Math.PI * 2;
  return {
    x: WEB_CENTER.x + R * Math.cos(angle),
    y: WEB_CENTER.y + R * Math.sin(angle),
    angle,
    scale,
  };
}

function renderEcosystem(s) {
  const liveEco = s.ecosystem || {};
  const decisionEco = (s.last_decision && s.last_decision.ecosystem) || {};
  const eco = Object.assign({}, liveEco, decisionEco);
  if (!eco.multiscale && liveEco.multiscale) eco.multiscale = liveEco.multiscale;
  if ((!eco.nodes || !eco.nodes.length) && liveEco.nodes) eco.nodes = liveEco.nodes;
  if ((!eco.edges || !eco.edges.length) && liveEco.edges) eco.edges = liveEco.edges;
  const phase = eco.phase || "producer";
  const mult = (typeof eco.size_mult === "number") ? eco.size_mult : 1.0;
  const applied = !!(s.last_decision && s.last_decision.ecosystem_applied);

  // header
  const badge = $("eco-phase-badge");
  badge.textContent = phase.toUpperCase();
  badge.className = "eco-badge " + phase;
  $("eco-mult").textContent = `size ×${mult.toFixed(2)}` +
    (applied ? " · applied" : " · observe-only");
  $("eco-mult").style.color = applied ? "" : "var(--muted)";
  const ksName = eco.keystone || "—";
  $("eco-keystone").textContent = "keystone: " + (WEB_NODE_LABEL[ksName] || ksName);
  const dist = (eco.drivers && eco.drivers.disturbance) || 0;
  const distEl = $("eco-disturbance");
  distEl.textContent = "disturbance " + dist.toFixed(2);
  distEl.style.color = dist > 1.5 ? "#ff9a9a" : dist > 0.8 ? "#f59e0b" : "var(--muted)";
  const nm = eco.network_metrics || {};
  const ascEl = $("eco-ascendancy");
  if (ascEl) {
    if (typeof nm.rel_ascendancy === "number") {
      ascEl.textContent = `A/C ${nm.rel_ascendancy.toFixed(2)} · R/C ${(nm.rel_reserve||0).toFixed(2)}`;
      // Highlight brittle (high rel_A + disturbance) and diffuse (low rel_A)
      if (nm.rel_ascendancy > 0.55 && dist > 0.7) ascEl.style.color = "#ff9a9a";
      else if (nm.rel_ascendancy < 0.20) ascEl.style.color = "#f59e0b";
      else ascEl.style.color = "var(--muted)";
    } else {
      ascEl.textContent = "A/C —";
      ascEl.style.color = "var(--muted)";
    }
  }
  const tstEl = $("eco-tst");
  if (tstEl) {
    tstEl.textContent = (typeof nm.TST === "number")
        ? `TST ${nm.TST.toFixed(2)} · edges ${nm.n_active_edges||0}`
        : "TST —";
  }

  // rationale
  $("eco-rationale").textContent = eco.rationale ? `phase: ${eco.rationale}` : "—";

  renderFoodWeb(eco);
  renderPhaseRing(phase);
  renderVitality(s);
  renderScaleNetworks(eco.multiscale || {}, eco);
  renderDrivers(eco.drivers || {});
  renderOrganisms(eco.organisms || {});
  renderFeeds(eco.multi_summary && eco.multi_summary.feeds_ok ? eco.multi_summary.feeds_ok :
              (s.multiasset && s.multiasset.sources) || {});
}

/* Foraging-cycle panel: enable toggle + a "how hungry" preset selector
   (instead of raw parameters) + a live "captured profit over time" chart. */
function wireForagerHunger() {
  document.querySelectorAll("#forager-hunger button.fhunger").forEach(b => {
    b.onclick = async () => {
      await patchSettings({forager_hunger: b.dataset.hunger});
      refresh();
    };
  });
  document.querySelectorAll("#autotomy-aggression button.aaggr").forEach(b => {
    b.onclick = async () => {
      await patchSettings({autotomy_aggression: b.dataset.aggression});
      refresh();
    };
  });
  const cd = $("inp-forager-cooldown");
  if (cd) cd.onchange = (e) => {
    const v = parseInt(e.target.value, 10);
    if (!isNaN(v) && v >= 0) patchSettings({forager_cooldown_seconds: v});
  };
  const acd = $("inp-autotomy-cooldown");
  if (acd) acd.onchange = (e) => {
    const v = parseInt(e.target.value, 10);
    if (!isNaN(v) && v >= 0) patchSettings({autotomy_cooldown_seconds: v});
  };
}

/* Novel "foraging cycle" diagram: the forager orbits a 5-phase ring
   (search -> work edge -> harvest -> rest -> re-enter), a marker sits at the
   current phase, and a central core grows with harvest pressure / colors with
   regime eagerness. Updates every refresh as the forager forages. */
const FORAGER_PHASES = [
  {key: "searching",  label: "search"},
  {key: "working",    label: "work edge"},
  {key: "harvesting", label: "harvest"},
  {key: "resting",    label: "rest"},
  {key: "reentering", label: "re-enter"},
];
function renderForagerOrbit(s) {
  const svg = $("forager-orbit");
  if (!svg) return;
  const f = s.forager_state || {};
  const on = !!(s.settings && s.settings.forager_enabled);
  const cx = 150, cy = 100, R = 70;
  const phase = on ? (f.phase || "searching") : "off";
  const eager = Math.max(0, Math.min(1, +f.eagerness || 0));
  const hp = +f.harvest_pressure || 0;
  const num = (x, d = 2) => (x === undefined || x === null) ? "—" : (+x).toFixed(d);
  const eColor = eager < 0.34 ? "#22c55e" : eager < 0.67 ? "#f59e0b" : "#ef4444";
  const pts = [];
  let nodes = "";
  FORAGER_PHASES.forEach((p, i) => {
    const a = (-90 + i * 72) * Math.PI / 180;
    const x = cx + R * Math.cos(a), y = cy + R * Math.sin(a);
    pts.push(x.toFixed(1) + "," + y.toFixed(1));
    const active = on && p.key === phase;
    nodes += `<circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${active ? 9 : 5}" `
      + `fill="${active ? eColor : "#334155"}" stroke="${active ? "#fff" : "#1f2937"}" stroke-width="${active ? 1.6 : 1}"/>`;
    const lx = cx + (R + 20) * Math.cos(a), ly = cy + (R + 20) * Math.sin(a);
    nodes += `<text x="${lx.toFixed(1)}" y="${(ly + 3).toFixed(1)}" fill="${active ? "#e5e7eb" : "#64748b"}" `
      + `font-size="9" text-anchor="middle" font-weight="${active ? "700" : "400"}">${p.label}</text>`;
  });
  const ring = `<polygon points="${pts.join(" ")}" fill="none" stroke="#1f2937" stroke-width="1"/>`;
  const coreR = 9 + 17 * Math.max(0, Math.min(1, (hp + 1) / 3));
  const core = `<circle cx="${cx}" cy="${cy}" r="${coreR.toFixed(1)}" fill="${eColor}" fill-opacity="0.18" stroke="${eColor}" stroke-width="1.5"/>`
    + `<text x="${cx}" y="${cy - 1}" fill="#e5e7eb" font-size="11" text-anchor="middle" font-weight="700">${on ? phase : "off"}</text>`
    + `<text x="${cx}" y="${cy + 12}" fill="#9ca3af" font-size="9" text-anchor="middle">pnl_R ${num(f.pnl_R)}</text>`;
  const sx = 296;
  const stats =
    `<text x="${sx}" y="38" fill="#9ca3af" font-size="10">hunger: <tspan fill="#e5e7eb" font-weight="700">${(s.settings && s.settings.forager_hunger) || "—"}</tspan></text>`
    + `<text x="${sx}" y="58" fill="#9ca3af" font-size="10">regime eagerness: <tspan fill="${eColor}" font-weight="700">${(eager * 100).toFixed(0)}%</tspan></text>`
    + `<text x="${sx}" y="78" fill="#9ca3af" font-size="10">harvest pressure: <tspan fill="#e5e7eb">${num(hp)}</tspan></text>`
    + `<text x="${sx}" y="98" fill="#9ca3af" font-size="10">cooldown: <tspan fill="#e5e7eb">${f.in_cooldown ? ((f.cooldown_remaining_s || 0) + "s") : "—"}</tspan></text>`
    + `<text x="${sx}" y="118" fill="#9ca3af" font-size="10">captured: <tspan fill="#22c55e">$${num(f.captured_cumulative)}</tspan></text>`
    + `<text x="${sx}" y="140" fill="#64748b" font-size="9">disturb ${num(f.disturbance_score)} · CSD ${num(f.csd_risk)}</text>`
    + `<text x="${sx}" y="155" fill="#64748b" font-size="9">reserve ${num(f.reserve)} · ascend ${num(f.rel_ascendancy)}</text>`;
  svg.innerHTML = ring + core + nodes + stats;
}

function renderForagerChart(harvests) {
  const svg = $("forager-chart");
  if (!svg) return;
  const note = $("forager-chart-note");
  const W = 480, H = 120, PADL = 46, PADR = 10, PADT = 10, PADB = 18;
  const chron = (harvests || []).slice().reverse();   // oldest -> newest
  if (!chron.length) {
    svg.innerHTML = `<text x="${W / 2}" y="${H / 2}" fill="#64748b" font-size="11" text-anchor="middle">no harvests yet — captured profit appears here as the forager banks winners</text>`;
    if (note) note.textContent = "Total captured: $0.00 · 0 harvests";
    return;
  }
  const ys = chron.map(h => +h.cumulative_usd || 0);
  const n = ys.length;
  const maxY = Math.max(...ys, 0.0001), minY = Math.min(...ys, 0);
  const X = (i) => PADL + (n <= 1 ? (W - PADL - PADR) / 2 : (i / (n - 1)) * (W - PADL - PADR));
  const Y = (v) => H - PADB - ((v - minY) / ((maxY - minY) || 1)) * (H - PADT - PADB);
  let d = `M ${X(0).toFixed(1)} ${Y(ys[0]).toFixed(1)}`;
  for (let i = 1; i < n; i++)
    d += ` L ${X(i).toFixed(1)} ${Y(ys[i - 1]).toFixed(1)} L ${X(i).toFixed(1)} ${Y(ys[i]).toFixed(1)}`;
  const area = d + ` L ${X(n - 1).toFixed(1)} ${(H - PADB).toFixed(1)} L ${X(0).toFixed(1)} ${(H - PADB).toFixed(1)} Z`;
  let dots = "";
  for (let i = 0; i < n; i++)
    dots += `<circle cx="${X(i).toFixed(1)}" cy="${Y(ys[i]).toFixed(1)}" r="2" fill="#22c55e"/>`;
  svg.innerHTML =
    `<path d="${area}" fill="#22c55e" fill-opacity="0.12"/>`
    + `<path d="${d}" fill="none" stroke="#22c55e" stroke-width="1.6"/>`
    + `<line x1="${PADL}" y1="${H - PADB}" x2="${W - PADR}" y2="${H - PADB}" stroke="#1f2937"/>`
    + `<text x="4" y="${(Y(maxY) + 4).toFixed(1)}" fill="#9ca3af" font-size="9">$${maxY.toFixed(2)}</text>`
    + `<text x="4" y="${(H - PADB).toFixed(1)}" fill="#9ca3af" font-size="9">$0</text>`
    + dots;
  if (note) note.textContent =
    `Total captured: $${ys[n - 1].toFixed(2)} · ${n} harvest${n > 1 ? "s" : ""}`;
}

function renderForager(s) {
  const f = s.forager_state || {};
  const a = s.autotomy_state || {};
  const on = !!(s.settings && s.settings.forager_enabled);
  const aon = !!(s.settings && s.settings.autotomy_enabled);
  if ($("chk-forager-enabled")) $("chk-forager-enabled").checked = on;
  if ($("chk-autotomy-enabled")) $("chk-autotomy-enabled").checked = aon;
  const hunger = (s.settings && s.settings.forager_hunger) || "ravenous";
  const aggression = (s.settings && s.settings.autotomy_aggression) || "ravenous";
  document.querySelectorAll("#forager-hunger button.fhunger").forEach(b => {
    const act = b.dataset.hunger === hunger;
    b.style.background = act ? "#1d4ed8" : "";
    b.style.color = act ? "#fff" : "";
    b.style.fontWeight = act ? "700" : "";
  });
  document.querySelectorAll("#autotomy-aggression button.aaggr").forEach(b => {
    const act = b.dataset.aggression === aggression;
    b.style.background = act ? "#7f1d1d" : "";
    b.style.color = act ? "#fff" : "";
    b.style.fontWeight = act ? "700" : "";
  });
  const num = (x, d = 2) => (x === undefined || x === null) ? "—" : (+x).toFixed(d);
  const badge = $("forager-badge");
  if (badge) {
    let txt = "OFF", cls = "churn";
    if (on) {
      if (f.harvested) { txt = "HARVEST"; cls = "stop"; }
      else if (f.in_cooldown) { txt = "RESTING"; cls = "caution"; }
      else { txt = "FORAGING"; cls = "ok"; }
    }
    badge.textContent = txt;
    badge.className = "eco-badge " + cls;
  }
  if ($("forager-pnlr")) $("forager-pnlr").textContent = "pnl_R: " + num(f.pnl_R);
  if ($("forager-hp")) $("forager-hp").textContent = "HP: " + num(f.harvest_pressure);
  if ($("forager-cooldown")) $("forager-cooldown").textContent =
    "cooldown: " + (f.in_cooldown ? ((f.cooldown_remaining_s || 0) + "s") : "—");
  if ($("forager-detail")) {
    $("forager-detail").textContent = !on
      ? "Forager off. Toggle on to harvest profit and rest during churn."
      : `[${hunger}] edge_decay ${num(f.edge_decay)} · decomposer ${num(f.decomposer)} · `
        + `disturbance ${num(f.disturbance_score)} · reserve ${num(f.reserve)}`
        + ` · HP d ${num(f.harvest_pressure_delta)} / imp ${num(f.harvest_pressure_impulse)}`
        + (f.harvested && f.harvest_reason ? `  — ${f.harvest_reason}` : "")
        + (f.blocked_entry ? "  — resting (entry blocked)" : "");
  }
  const abadge = $("autotomy-badge");
  if (abadge) {
    let txt = "AUTOTOMY OFF", cls = "churn";
    if (aon) {
      if (a.ejected) { txt = "EJECT"; cls = "predator"; }
      else if (a.in_cooldown) { txt = "LOSS COOLDOWN"; cls = "caution"; }
      else if ((+a.loss_R || 0) > 0) { txt = "WATCHING"; cls = "exhaustion"; }
      else { txt = "ARMED"; cls = "ok"; }
    }
    abadge.textContent = txt;
    abadge.className = "eco-badge " + cls;
  }
  if ($("autotomy-lossr")) $("autotomy-lossr").textContent = "loss_R: " + num(a.loss_R);
  if ($("autotomy-pressure")) $("autotomy-pressure").textContent =
    "AP: " + num(a.autotomy_pressure) + " / "
    + num(a.threshold !== undefined ? a.threshold : (s.settings && s.settings.autotomy_pressure_threshold));
  if ($("autotomy-cooldown")) $("autotomy-cooldown").textContent =
    "cooldown: " + (a.in_cooldown ? ((a.cooldown_remaining_s || 0) + "s") : "—");
  if ($("autotomy-detail")) {
    $("autotomy-detail").textContent = !aon
      ? "Autotomy off. It still watches for toxic losing positions."
      : `phase ${a.phase || "—"} · confirmations ${a.confirmations || 0}/${a.min_confirmations || 0} · `
        + `predator ${num(a.predator_score)} · cascade ${num(a.cascade_risk)} · `
        + `reserve collapse ${num(a.reserve_collapse)} · CSD ${num(a.csd_risk)}`
        + ` · AP d ${num(a.autotomy_pressure_delta)} / imp ${num(a.autotomy_pressure_impulse)}`
        + (a.ejected && a.eject_reason ? `  — ${a.eject_reason}` : "")
        + (a.blocked_entry ? "  — loss cooldown (entry blocked)" : "");
  }
  if (aon && $("autotomy-detail")) {
    $("autotomy-detail").textContent =
      `[${aggression}] ` + $("autotomy-detail").textContent
      + ` · AP projected ${num(a.autotomy_pressure_projected)}`;
  }
  syncControl("inp-forager-cooldown", s.settings ? s.settings.forager_cooldown_seconds : undefined);
  syncControl("inp-autotomy-cooldown", s.settings ? s.settings.autotomy_cooldown_seconds : undefined);
  renderForagerOrbit(s);
  renderForagerChart(s.forager_harvests);
}

function renderVisualReview(s) {
  const vr = s.visual_review || {};
  const enabled = !!(s.settings && s.settings.visual_review_enabled);
  const concern = (vr.concern || "OK").toUpperCase();
  const concernLower = concern.toLowerCase();
  const trend = vr.trend || "—";
  const note = vr.note || "";
  const lastTs = vr.last_ts; // unix seconds (server side)
  const intervalSec = (s.settings && s.settings.visual_review_interval_seconds) || 600;

  const badge = $("vr-concern-badge");
  if (!enabled) {
    badge.textContent = "DISABLED";
    badge.className = "eco-badge churn";
  } else if (!lastTs) {
    badge.textContent = "PENDING";
    badge.className = "eco-badge churn";
  } else {
    badge.textContent = concern;
    badge.className = "eco-badge " + concernLower;
  }
  $("vr-trend").textContent = "trend: " + trend;

  // Countdown to next review
  let cdText = "—";
  if (enabled && lastTs) {
    const nowSec = Date.now() / 1000;
    const elapsed = nowSec - lastTs;
    const remaining = Math.max(0, intervalSec - elapsed);
    if (remaining > 0) {
      const m = Math.floor(remaining / 60);
      const sec = Math.floor(remaining % 60);
      cdText = `next in ${m}m ${sec.toString().padStart(2,"0")}s`;
    } else {
      cdText = "next: due";
    }
  } else if (enabled) {
    cdText = "next: first cycle";
  } else {
    cdText = "disabled";
  }
  $("vr-cooldown").textContent = cdText;

  if (note) {
    $("vr-note").textContent = note;
    $("vr-note").style.color = concern === "STOP" ? "#ff9a9a" :
                                concern === "CAUTION" ? "#f59e0b" : "var(--muted)";
  } else if (!enabled) {
    $("vr-note").textContent = "Visual review is disabled. Toggle on to begin chart sanity checks.";
    $("vr-note").style.color = "var(--muted)";
  } else {
    $("vr-note").textContent = "No review yet. Will fire on the first cycle.";
    $("vr-note").style.color = "var(--muted)";
  }

  // Image: only show if a review has been generated
  const imgLink = $("vr-image-link");
  const img = $("vr-image");
  if (lastTs) {
    imgLink.hidden = false;
    img.src = "/api/visual_review/latest.png?t=" + Math.floor(lastTs);
  } else {
    imgLink.hidden = true;
  }
}

/* Resilience well diagram — the classic CSD potential landscape. A deep
   narrow well means strong mean-reversion (every shock falls back to 0).
   As risk rises the well flattens — the ball can drift further and escape.
   Coordinate system: x in [-1, +1] mapping to deviation, U(x) drawn upward.
   Visual coupling:
     - well depth scales with (1 - risk)^1.5 — recovery rate goes to 0 at risk=1
     - ball x-position = sign(skew_now) * sqrt(|skew_z|) scaled to fit
     - ball height = U(ball_x) so it sits in the basin
     - threshold marker is the rim height; if ball crosses, brittle styling
*/
/* Fast guard mini-diagram: a horizontal drift gauge.
   X axis: intra-cycle move in bps, symmetric ±threshold*1.5.
   - Green band:  |move| < threshold*0.5  (safe)
   - Amber band:  threshold*0.5 ≤ |move| < threshold  (warn)
   - Red band:    |move| ≥ threshold  (danger -> guard fires)
   Pointer: current intra-cycle drift, color by zone.
   A small blinking dot in the corner shows the guard is actively polling.
*/
function renderFastGuard(s) {
  const svg = $("fg-diag");
  if (!svg) return;
  const fg = s.fast_guard_state || {};
  const settings = s.settings || {};
  const enabled = !!settings.fast_guard_enabled;
  const running = !!fg.running;
  const threshold = (typeof settings.fast_guard_emergency_move_bps === "number")
                    ? settings.fast_guard_emergency_move_bps : 10.0;
  const poll = (typeof settings.fast_guard_poll_seconds === "number")
               ? settings.fast_guard_poll_seconds : 3;
  const move = (typeof fg.current_intra_cycle_move_bps === "number")
               ? fg.current_intra_cycle_move_bps : 0;
  const fires = fg.emergencies_triggered || 0;
  const polls = fg.polls || 0;

  // Pill in the header
  const pill = $("fg-pill");
  if (pill) {
    if (!enabled) {
      pill.textContent = "fast guard OFF";
      pill.style.color = "var(--muted)";
    } else if (!running) {
      pill.textContent = `fast guard armed · ${poll}s poll`;
      pill.style.color = "var(--muted)";
    } else {
      const moveStr = (move >= 0 ? "+" : "") + move.toFixed(1) + "bps";
      pill.textContent = `fast guard ${poll}s · ${polls} polls · ${fires} fires · drift ${moveStr}`;
      pill.style.color = Math.abs(move) >= threshold ? "#ff9a9a"
                       : Math.abs(move) >= threshold * 0.5 ? "#f59e0b"
                       : "#86efac";
    }
  }

  // Diagram
  const W = 360, H = 80;
  const PAD_L = 30, PAD_R = 30, PAD_T = 18, PAD_B = 22;
  const xL = PAD_L, xR = W - PAD_R, xC = (xL + xR) / 2;
  const yMid = (PAD_T + (H - PAD_B)) / 2;
  // X axis: -1.5*threshold .. +1.5*threshold
  const xMax = threshold * 1.5;
  const toX = (v) => xC + (Math.max(-xMax, Math.min(xMax, v)) / xMax)
                           * ((xR - xL) / 2);
  // Band stops
  const halfBand = toX(threshold * 0.5) - xC;
  const fullBand = toX(threshold) - xC;
  const bands = `
    <rect class="fg-band-safe"
      x="${(xC - halfBand).toFixed(1)}" y="${PAD_T}"
      width="${(2 * halfBand).toFixed(1)}" height="${H - PAD_T - PAD_B}"/>
    <rect class="fg-band-warn"
      x="${(xC - fullBand).toFixed(1)}" y="${PAD_T}"
      width="${(fullBand - halfBand).toFixed(1)}" height="${H - PAD_T - PAD_B}"/>
    <rect class="fg-band-warn"
      x="${(xC + halfBand).toFixed(1)}" y="${PAD_T}"
      width="${(fullBand - halfBand).toFixed(1)}" height="${H - PAD_T - PAD_B}"/>
    <rect class="fg-band-danger"
      x="${xL}" y="${PAD_T}"
      width="${(xC - fullBand - xL).toFixed(1)}" height="${H - PAD_T - PAD_B}"/>
    <rect class="fg-band-danger"
      x="${(xC + fullBand).toFixed(1)}" y="${PAD_T}"
      width="${(xR - (xC + fullBand)).toFixed(1)}" height="${H - PAD_T - PAD_B}"/>
  `;
  // Axis + threshold marks + ticks
  const axis = `
    <line x1="${xL}" y1="${(H - PAD_B).toFixed(1)}" x2="${xR}" y2="${(H - PAD_B).toFixed(1)}" class="fg-axis"/>
    <line x1="${xC}" y1="${PAD_T}" x2="${xC}" y2="${(H - PAD_B).toFixed(1)}" class="fg-tick"/>
    <line x1="${(xC - fullBand).toFixed(1)}" y1="${PAD_T}" x2="${(xC - fullBand).toFixed(1)}" y2="${(H - PAD_B).toFixed(1)}" class="fg-threshold"/>
    <line x1="${(xC + fullBand).toFixed(1)}" y1="${PAD_T}" x2="${(xC + fullBand).toFixed(1)}" y2="${(H - PAD_B).toFixed(1)}" class="fg-threshold"/>
  `;
  // Tick labels
  const ticks = `
    <text x="${xC}" y="${(H - 6)}" class="fg-label">0</text>
    <text x="${(xC - fullBand).toFixed(1)}" y="${(H - 6)}" class="fg-label">−${threshold.toFixed(0)}bps</text>
    <text x="${(xC + fullBand).toFixed(1)}" y="${(H - 6)}" class="fg-label">+${threshold.toFixed(0)}bps</text>
    <text x="${xC}" y="${PAD_T - 6}" class="fg-label">intra-cycle mid drift</text>
  `;
  // Pointer
  const px = toX(move);
  let pCls = "fg-pointer";
  if (Math.abs(move) >= threshold) pCls += " danger";
  else if (Math.abs(move) >= threshold * 0.5) pCls += " warn";
  const pointer = `
    <polygon points="${px.toFixed(1)},${(yMid - 7).toFixed(1)} ${(px - 6).toFixed(1)},${(yMid + 9).toFixed(1)} ${(px + 6).toFixed(1)},${(yMid + 9).toFixed(1)}"
      class="${pCls}"/>
    <text x="${px.toFixed(1)}" y="${(yMid - 11).toFixed(1)}" class="fg-label cur">${move >= 0 ? "+" : ""}${move.toFixed(1)}bps</text>
  `;
  // Active-poll blinker (top-left corner)
  const blink = running
      ? `<circle cx="${xL + 4}" cy="${PAD_T + 2}" r="3" class="fg-poll-dot"/>`
      : "";

  svg.innerHTML = bands + axis + ticks + pointer + blink;
}

/* Alpha decomposition diagram — horizontal stacked bar showing each
   component's signed contribution to the blended alpha aim. Components:
     MPC       (validated reversion alpha)
     spot_lead (Coinbase BTC 3-min log-return tilt)
     funding   (perp funding-rate fade)
     OI        (Hyperliquid OI delta × price direction)
   Bottom row shows the final blended aim with a gradient highlight.
*/
const ALPHA_COMPONENTS = [
  {key: "mpc",          label: "MPC alpha",   note: "validated reversion"},
  {key: "spot_lead",    label: "spot lead",   note: "Coinbase BTC 3m"},
  {key: "funding_fade", label: "funding",     note: "perp fade"},
  {key: "oi_pressure",  label: "OI press.",   note: "Hyperliquid"},
  {key: "ecology_flow", label: "ecology",     note: "network state"},
];

function renderAlphaDecomp(s) {
  const svg = $("alpha-decomp");
  if (!svg) return;
  const bs = s.blend_state || {};
  const weights = bs.weights || {};
  const raw = bs.raw || {};
  const parts = bs.parts || {};
  const blended = bs.blended || 0;
  const enabled = !!bs.enabled;

  // Layout. The previous version packed everything into 420x240 and the
  // component band ran into the BLENDED row. Now: wider canvas, larger pads,
  // band tightly bounding ONLY the component rows.
  const W = 480, H = 280;
  const PAD_L = 110, PAD_R = 78, PAD_T = 44, PAD_B = 28;
  const xL = PAD_L, xR = W - PAD_R;        // bar plotting region
  const xC = (xL + xR) / 2;
  const rowGap = 23;
  const compTop = PAD_T + 16;
  const compBottom = compTop + (ALPHA_COMPONENTS.length - 1) * rowGap + 10;
  const dividerY = compBottom + 10;
  const blendY = dividerY + 26;

  // Max absolute value across components+blended for scaling
  const allVals = [blended, ...Object.values(parts), ...Object.values(raw)];
  const maxAbs = Math.max(0.05, ...allVals.map(v => Math.abs(v || 0)));
  const valueToX = (v) => xC + (v / maxAbs) * ((xR - xL) / 2);

  // Defs: gradient for blended bar + arrowhead
  const defs = `<defs>
    <linearGradient id="alpha-blended-grad" x1="0" y1="0" x2="1" y2="0">
      <stop offset="0%"   stop-color="#fbbf24" stop-opacity="0.85"/>
      <stop offset="100%" stop-color="#f59e0b" stop-opacity="0.95"/>
    </linearGradient>
    <marker id="alpha-arrow-head" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="6" markerHeight="6" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" class="alpha-arrow-head"/>
    </marker>
  </defs>`;

  // Top header above the chart area
  const valueColX = xR + 16;
  const header = `
    <text x="${xL - 8}" y="${PAD_T - 24}" class="alpha-header" text-anchor="end">COMPONENT</text>
    <text x="${xC}" y="${PAD_T - 24}" class="alpha-header" text-anchor="middle">CONTRIBUTION → blended aim</text>
    <text x="${valueColX}" y="${PAD_T - 24}" class="alpha-header" text-anchor="start">VALUE</text>
  `;

  // Background band ONLY behind the component rows (not the blended row)
  const bandTop = compTop - 14;
  const bandHeight = (compBottom + 6) - bandTop;
  const band = `<rect class="alpha-band" x="${xL - 6}" y="${bandTop}"
    width="${xR - xL + 12}" height="${bandHeight}" rx="6"/>`;

  // Zero line spans both the component band AND the blended row
  const zeroTop = PAD_T - 6;
  const zeroBot = blendY + 14;
  let ticks = `<line x1="${xC}" y1="${zeroTop}" x2="${xC}" y2="${zeroBot}"
    class="alpha-zero"/>`;
  // Edge tick marks (SHORT / LONG)
  const tickVals = [-1, -0.5, 0.5, 1].map(t => t * maxAbs);
  for (const tv of tickVals) {
    const tx = valueToX(tv);
    ticks += `<line x1="${tx}" y1="${PAD_T - 6}" x2="${tx}" y2="${PAD_T}"
      class="alpha-axis-tick"/>`;
  }
  // Axis end labels above the chart
  const axisLabels = `
    <text x="${valueToX(-maxAbs).toFixed(1)}" y="${PAD_T - 8}" class="alpha-tick-label">SHORT</text>
    <text x="${xC}" y="${PAD_T - 8}" class="alpha-tick-label">0</text>
    <text x="${valueToX(maxAbs).toFixed(1)}" y="${PAD_T - 8}" class="alpha-tick-label">LONG</text>
  `;

  // Component rows
  let rows = "";
  ALPHA_COMPONENTS.forEach((c, i) => {
    const y = compTop + i * rowGap;
    const w = weights[c.key] || 0;
    const p = parts[c.key] || 0;
    const dim = (Math.abs(p) < 1e-5);
    const x0 = valueToX(0);
    const x1 = valueToX(p);
    const bx = Math.min(x0, x1), bw = Math.max(1, Math.abs(x1 - x0));
    const dimCls = dim ? " dim" : "";
    rows += `
      <text x="${xL - 12}" y="${y - 2}" class="alpha-row-label${dim ? ' dim' : ''}">${c.label}</text>
      <text x="${xL - 12}" y="${y + 9}" class="alpha-weight-tag" text-anchor="end">w=${w.toFixed(2)}</text>
      <rect x="${bx.toFixed(1)}" y="${(y - 8).toFixed(1)}" width="${bw.toFixed(1)}" height="14"
            rx="2.5" class="alpha-bar-${c.key} alpha-bar${dimCls}"/>
      <text x="${valueColX}" y="${y}" class="alpha-value">${p >= 0 ? '+' : ''}${p.toFixed(3)}</text>
    `;
  });

  // Divider line between components and blended row
  const divider = `<line x1="${xL - 12}" y1="${dividerY}" x2="${valueColX + 56}" y2="${dividerY}"
    stroke="#2a3554" stroke-width=".8" opacity=".75"/>`;

  // Blended row — thicker, glowing, OUTSIDE the band
  const bX0 = valueToX(0);
  const bX1 = valueToX(blended);
  const bbx = Math.min(bX0, bX1), bbw = Math.max(1, Math.abs(bX1 - bX0));
  const blendBar = `
    <text x="${xL - 12}" y="${blendY - 2}" class="alpha-row-label" style="fill:#fbbf24">BLENDED</text>
    <text x="${xL - 12}" y="${blendY + 9}" class="alpha-weight-tag" text-anchor="end">= Σ wᵢ αᵢ</text>
    <rect x="${bbx.toFixed(1)}" y="${(blendY - 12).toFixed(1)}" width="${bbw.toFixed(1)}" height="22"
      rx="3" class="alpha-blended-fill alpha-blended-stroke"/>
    <text x="${valueColX}" y="${blendY}" class="alpha-value" style="fill:#fbbf24;font-weight:700">
      ${blended >= 0 ? '+' : ''}${blended.toFixed(3)}</text>
  `;
  // Directional arrow from zero to blended endpoint
  const arrowDir = `<line x1="${xC}" y1="${blendY}" x2="${bX1.toFixed(1)}" y2="${blendY}"
    stroke="#fbbf24" stroke-width="2.8" opacity=".95" marker-end="url(#alpha-arrow-head)"/>`;

  svg.innerHTML = defs + band + header + ticks + axisLabels + rows + divider + blendBar + arrowDir;

  // Caption: what fired and what was dim
  let topContrib = null, topMag = 0;
  for (const c of ALPHA_COMPONENTS) {
    const v = Math.abs(parts[c.key] || 0);
    if (v > topMag) { topMag = v; topContrib = c; }
  }
  const diag = bs.diagnostics || {};
  const muted = (diag.muted_components || [])
    .filter(k => ALPHA_COMPONENTS.some(c => c.key === k))
    .map(k => (ALPHA_COMPONENTS.find(c => c.key === k) || {label:k}).label);
  let note = "";
  if (!enabled) {
    note = "Blend disabled — only MPC alpha drives the live signal. Toggle signal_blend_enabled to combine.";
  } else if (topContrib && topMag > 1e-5) {
    note = `Top contributor: ${topContrib.label} (${topContrib.note}). `
      + `Final aim ${blended >= 0 ? "LONG" : "SHORT"} ${Math.abs(blended).toFixed(3)}.`;
  } else {
    note = "All components near zero — flat aim this cycle.";
  }
  if (muted.length) note += ` Muted raw signal: ${muted.join(", ")}.`;
  const ctl = bs.controller || {};
  if (ctl && ctl.target_after != null) {
    const aim = Number(ctl.aim_capped || 0);
    const target = Number(ctl.target_after || 0);
    const band = Number(ctl.band || 0);
    const phase = ctl.phase || diag.phase || "unknown";
    note += ` Controller target ${target >= 0 ? "LONG" : "SHORT"} ${Math.abs(target).toFixed(3)} `
      + `after ${phase} band ${band.toFixed(3)} from aim ${aim >= 0 ? "+" : ""}${aim.toFixed(3)}.`;
    if (ctl.live_anchor_applied) note += " Live-position anchored.";
    if (ctl.controller_move === "hold_band") note += " Held inside no-trade band.";
  }
  $("alpha-decomp-note").textContent = note;
}

/* Window of Vitality — Ulanowicz 2D regime map.
   X: rel_ascendancy (0..1)
   Y: disturbance (0..2+, clamped)
   Four colored regions:
     top-right    = brittle    (one pathway locked under stress)
     top-left     = diffuse    (no clean driver, stressed)
     bottom-right = organized  (structured but calm)
     bottom-left  = quiet      (no edge, calm)
   Center band   = healthy adaptive churn
   Current point glows + pulses; trailing breadcrumbs fade.
*/
const VIT_TRAIL = [];   // rolling history of {a, d}
const VIT_TRAIL_MAX = 30;

function renderVitality(s) {
  const svg = $("eco-vitality");
  if (!svg) return;
  const eco = s.ecosystem || (s.last_decision && s.last_decision.ecosystem) || {};
  const nm = eco.network_metrics || {};
  const dr = eco.drivers || {};
  const relA = (typeof nm.rel_ascendancy === "number") ? nm.rel_ascendancy : 0;
  const dist = (typeof dr.disturbance === "number") ? dr.disturbance : 0;

  // Append to trail (deduplicate consecutive identical points).
  const last = VIT_TRAIL[VIT_TRAIL.length - 1];
  if (!last || Math.abs(last.a - relA) > 0.001 || Math.abs(last.d - dist) > 0.005) {
    VIT_TRAIL.push({a: relA, d: dist});
    if (VIT_TRAIL.length > VIT_TRAIL_MAX) VIT_TRAIL.shift();
  }

  const W = 240, H = 220;
  const PAD_L = 28, PAD_R = 14, PAD_T = 22, PAD_B = 26;
  const xL = PAD_L, xR = W - PAD_R;
  const yT = PAD_T, yB = H - PAD_B;
  // Map rel_A in [0, 1] to [xL, xR]; disturbance in [0, 2] to [yB, yT] (inverted)
  const D_MAX = 2.0;
  const toPx = (a, d) => ({
    x: xL + Math.max(0, Math.min(1, a)) * (xR - xL),
    y: yB - Math.max(0, Math.min(D_MAX, d)) / D_MAX * (yB - yT),
  });

  // Quadrant boundaries (rel_A=0.5 horizontally; disturbance=0.7 vertically
  // ~ matches the brittle/diffuse thresholds 0.55/0.20 used in the classifier
  // but rounded to 0.5 for the visual quadrant split).
  const xMid = toPx(0.5, 0).x;
  const yMid = toPx(0, 0.7).y;
  // Quadrant fills
  const quads = `
    <rect class="vit-quad-quiet"     x="${xL}" y="${yMid}" width="${xMid - xL}" height="${yB - yMid}"/>
    <rect class="vit-quad-organized" x="${xMid}" y="${yMid}" width="${xR - xMid}" height="${yB - yMid}"/>
    <rect class="vit-quad-diffuse"   x="${xL}" y="${yT}" width="${xMid - xL}" height="${yMid - yT}"/>
    <rect class="vit-quad-brittle"   x="${xMid}" y="${yT}" width="${xR - xMid}" height="${yMid - yT}"/>
  `;

  // Axes + grid
  let grid = `
    <line x1="${xL}" y1="${yB}" x2="${xR}" y2="${yB}" class="vit-axis"/>
    <line x1="${xL}" y1="${yT}" x2="${xL}" y2="${yB}" class="vit-axis"/>
    <line x1="${xMid}" y1="${yT}" x2="${xMid}" y2="${yB}" class="vit-threshold"/>
    <line x1="${xL}" y1="${yMid}" x2="${xR}" y2="${yMid}" class="vit-threshold"/>
  `;
  // Tick gridlines
  for (let v of [0.25, 0.75]) {
    const p = toPx(v, 0);
    grid += `<line x1="${p.x}" y1="${yT}" x2="${p.x}" y2="${yB}" class="vit-grid"/>`;
  }
  for (let v of [0.5, 1.0, 1.5]) {
    const p = toPx(0, v);
    grid += `<line x1="${xL}" y1="${p.y}" x2="${xR}" y2="${p.y}" class="vit-grid"/>`;
  }

  // Axis labels
  const axLabels = `
    <text x="${(xL + xR) / 2}" y="${H - 6}" class="vit-label">rel ascendancy A/C</text>
    <text x="${xL - 14}" y="${(yT + yB) / 2}" class="vit-label"
          transform="rotate(-90, ${xL - 14}, ${(yT + yB) / 2})">disturbance</text>
    <text x="${xL - 4}" y="${yB + 11}" class="vit-label" text-anchor="end">0</text>
    <text x="${xR}" y="${yB + 11}" class="vit-label" text-anchor="end">1</text>
    <text x="${xL - 4}" y="${yT + 4}" class="vit-label" text-anchor="end">${D_MAX}</text>
  `;

  // Quadrant labels (inset)
  const quadLabels = `
    <text x="${xL + 6}" y="${yT + 14}" class="vit-quad-label diffuse" text-anchor="start">DIFFUSE</text>
    <text x="${xR - 6}" y="${yT + 14}" class="vit-quad-label brittle" text-anchor="end">BRITTLE</text>
    <text x="${xL + 6}" y="${yB - 6}" class="vit-quad-label quiet" text-anchor="start">QUIET</text>
    <text x="${xR - 6}" y="${yB - 6}" class="vit-quad-label organized" text-anchor="end">ORGANIZED</text>
  `;

  // Trail: older points smaller + more transparent
  let trail = "";
  for (let i = 0; i < VIT_TRAIL.length - 1; i++) {
    const p = toPx(VIT_TRAIL[i].a, VIT_TRAIL[i].d);
    const age = (VIT_TRAIL.length - 1 - i) / VIT_TRAIL_MAX;
    const r = 1.5 + (1 - age) * 1.5;
    const op = (1 - age) * 0.5;
    trail += `<circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${r.toFixed(1)}"
      class="vit-trail" opacity="${op.toFixed(2)}"/>`;
  }
  // Trail line connecting points
  if (VIT_TRAIL.length >= 2) {
    const pts = VIT_TRAIL.map(p => {
      const q = toPx(p.a, p.d);
      return q.x.toFixed(1) + "," + q.y.toFixed(1);
    }).join(" ");
    trail += `<polyline points="${pts}" fill="none" stroke="#38bdf8"
      stroke-opacity=".25" stroke-width="1"/>`;
  }

  // Current point — class by quadrant
  const cur = toPx(relA, dist);
  let pointCls = "vit-point";
  if (relA >= 0.5 && dist >= 0.7) pointCls += " brittle";
  else if (relA < 0.5 && dist >= 0.7) pointCls += " diffuse";
  else if (relA >= 0.5 && dist < 0.7) pointCls += " organized";
  else pointCls += " quiet";
  const point = `<circle cx="${cur.x.toFixed(1)}" cy="${cur.y.toFixed(1)}"
    r="6" class="${pointCls}"/>
    <text x="${cur.x.toFixed(1)}" y="${(cur.y + 14).toFixed(1)}" class="vit-label"
      font-weight="700">${relA.toFixed(2)} · ${dist.toFixed(2)}</text>`;

  svg.innerHTML = `<rect class="vit-bg" x="0" y="0" width="${W}" height="${H}"/>` +
    quads + grid + quadLabels + axLabels + trail + point;
}

function renderCsdWell(s) {
  const svg = $("csd-well");
  if (!svg) return;
  const csd = s.csd_state || {};
  const risk = (typeof csd.risk === "number") ? csd.risk : 0;
  const threshold = (typeof csd.threshold === "number") ? csd.threshold : 0.95;
  const skewNow = (typeof csd.skew_now === "number") ? csd.skew_now : 0;
  const history = (csd.skew_history && csd.skew_history.length) ? csd.skew_history : [];

  // SVG coordinate frame: 360 wide x 170 tall, y inverted (0=top).
  const W = 360, H = 170, PAD_X = 24, PAD_TOP = 14, PAD_BOT = 24;
  const xL = PAD_X, xR = W - PAD_X;
  const yFloor = H - PAD_BOT;
  const xC = (xL + xR) / 2;
  const halfW = (xR - xL) / 2;

  // Well shape: U(x) = a * x^2, where a scales with (1 - risk)^1.5.
  // a=0.85 at risk=0 (deep), a=0.10 at risk=1 (almost flat).
  const a = 0.10 + 0.85 * Math.pow(Math.max(0, 1 - risk), 1.5);
  // Convert U-units to pixels: U(±1) -> wellTopY (above floor)
  const wellPxMax = yFloor - PAD_TOP - 8;
  const uToPx = wellPxMax / (a * 1.0);  // a * 1.0 is U at x=±1 normalized

  // Compute the well curve polyline
  const N = 80;
  let path = "";
  for (let i = 0; i <= N; i++) {
    const t = i / N;
    const x = -1 + 2 * t;
    const u = a * x * x;
    const px = xL + t * (xR - xL);
    const py = yFloor - Math.min(u * uToPx, wellPxMax);
    path += (i === 0 ? "M" : "L") + px.toFixed(1) + "," + py.toFixed(1) + " ";
  }

  // Floor line + ground hatching (subtle grid below)
  const grid = `
    <line x1="${xL}" y1="${yFloor}" x2="${xR}" y2="${yFloor}" class="csd-well-floor"/>
    <line x1="${xC}" y1="${yFloor}" x2="${xC}" y2="${PAD_TOP}" class="csd-well-axis"/>
    <text x="${xC}" y="${H - 7}" class="csd-well-label center">deviation x_t</text>
    <text x="${xL + 6}" y="${PAD_TOP + 8}" class="csd-well-label" text-anchor="start">U(x)</text>
    <text x="${xL - 2}" y="${yFloor + 12}" class="csd-well-label" text-anchor="start">-σ</text>
    <text x="${xR + 2}" y="${yFloor + 12}" class="csd-well-label" text-anchor="end">+σ</text>
  `;

  // Threshold rim — horizontal line at U corresponding to z=threshold-sigmoid-inverse.
  // Visualize the gate: a horizontal dashed line at a level that the ball would
  // need to reach to "escape" the basin. We render it at 75% of well max so it's
  // visible across all risk levels but clearly above the ball most of the time.
  const rimPy = yFloor - 0.75 * wellPxMax;
  const rim = `<line x1="${xL + 14}" y1="${rimPy}" x2="${xR - 14}" y2="${rimPy}"
    class="csd-well-threshold"/>
    <text x="${xR - 14}" y="${rimPy - 4}" class="csd-well-label" text-anchor="end">gate</text>`;

  // Ball position. Use the z-score of skew_now vs trailing history to
  // determine how far up the well wall the ball has climbed.
  let z = 0;
  if (history.length >= 8) {
    const mean = history.reduce((a, b) => a + b, 0) / history.length;
    const variance = history.reduce((s, v) => s + (v - mean) ** 2, 0) / (history.length - 1);
    const std = Math.sqrt(variance);
    if (std > 0) z = (skewNow - mean) / std;
  }
  // Map z to ball x in [-1, +1]. z=±3 saturates to ±1.
  const ballX = Math.max(-1, Math.min(1, z / 3));
  // Ball y follows U(ballX) so it sits in the basin
  const ballU = a * ballX * ballX;
  const ballPx = xL + (ballX + 1) / 2 * (xR - xL);
  const ballPy = yFloor - Math.min(ballU * uToPx, wellPxMax) - 6;  // slightly above well surface
  // Visual classification
  let curveCls = "csd-well-curve", ballCls = "csd-well-ball";
  if (risk > threshold) {
    curveCls += " brittle"; ballCls += " brittle";
  } else if (risk > 0.7 * threshold) {
    curveCls += " stressed"; ballCls += " stressed";
  }

  // Ground texture below floor — subtle grid
  let groundGrid = "";
  for (let i = 1; i < 8; i++) {
    const px = xL + i / 8 * (xR - xL);
    groundGrid += `<line x1="${px}" y1="${yFloor}" x2="${px - 4}" y2="${yFloor + 6}" class="csd-well-grid"/>`;
  }

  svg.innerHTML = grid + groundGrid + rim +
    `<path d="${path}" class="${curveCls}"/>` +
    `<circle cx="${ballPx.toFixed(1)}" cy="${ballPy.toFixed(1)}" r="5.5" class="${ballCls}"/>`;
}

function renderCsd(s) {
  const csd = s.csd_state || {};
  const dec = s.last_decision || {};
  const enabled = !!(s.settings && s.settings.csd_governor_enabled);
  const risk = (typeof csd.risk === "number") ? csd.risk : 0;
  const riskNow = (typeof csd.risk_now === "number") ? csd.risk_now : risk;
  const riskProjected = (typeof csd.risk_projected === "number") ? csd.risk_projected : risk;
  const threshold = (typeof csd.threshold === "number") ? csd.threshold :
                    (s.settings && s.settings.csd_governor_threshold) || 0.95;
  const skewNow = (typeof csd.skew_now === "number") ? csd.skew_now : null;
  const gatedThisCycle = !!(dec && dec.proposal && dec.proposal.csd_gated);
  const pred = csd.predictive_enabled
    ? ` now ${riskNow.toFixed(3)} -> projected ${riskProjected.toFixed(3)}`
      + (csd.time_to_threshold_s != null ? `, TTT ${Number(csd.time_to_threshold_s).toFixed(1)}s` : "")
    : "";

  const fill = $("csd-bar-fill");
  if (fill) {
    fill.style.width = (Math.max(0, Math.min(1, risk)) * 100).toFixed(1) + "%";
  }
  const mark = $("csd-bar-mark");
  if (mark) {
    mark.style.left = (Math.max(0, Math.min(1, threshold)) * 100).toFixed(1) + "%";
  }
  $("csd-risk-val").textContent = risk.toFixed(3);
  $("csd-risk-val").style.color = risk > threshold ? "#ff9a9a" :
                                  risk > 0.7 * threshold ? "#f59e0b" : "var(--muted)";
  $("csd-skew-val").textContent = (skewNow === null || skewNow === undefined) ? "—" : skewNow.toFixed(4);
  $("csd-threshold-val").textContent = threshold.toFixed(2);

  const badge = $("csd-state-badge");
  if (gatedThisCycle) {
    badge.textContent = "GATED";
    badge.className = "eco-badge gated";
  } else if (enabled) {
    badge.textContent = "ACTIVE";
    badge.className = "eco-badge active";
  } else {
    badge.textContent = "DISABLED";
    badge.className = "eco-badge churn";
  }

  // descriptive line
  let note = "";
  if (!enabled) {
    note = "Governor off. abs(skew) of log-deviation is being observed but not gating size.";
  } else if (gatedThisCycle) {
    note = `Gated this cycle: risk ${risk.toFixed(3)} > threshold ${threshold.toFixed(2)} — position zeroed.`;
  } else if (risk > 0.7 * threshold) {
    note = `Elevated: ${(risk * 100).toFixed(1)}% of saturation. Within ${((threshold - risk) * 100).toFixed(1)}pp of gate.`;
  } else {
    note = `Calm regime: skew z-score modest, no gating expected.`;
  }
  $("csd-note").textContent = note + pred;
  renderCsdWell(s);
}

function _selectWebEdges(eco) {
  const active = new Set((eco.nodes || []));
  const candidates = (eco.edges || [])
    .filter(e => active.has(e.from) && active.has(e.to) && Math.abs(e.weight) >= 0.08)
    .map(e => ({
      ...e,
      fromScale: _scaleForNode(e.from, eco),
      toScale: _scaleForNode(e.to, eco),
    }))
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));

  const intra = [];
  const cross = [];
  const intraCounts = {};
  const crossCounts = {};
  for (const e of candidates) {
    if (e.fromScale === e.toScale) {
      const key = e.fromScale;
      intraCounts[key] = intraCounts[key] || 0;
      if (intraCounts[key] < 4) {
        intra.push(e);
        intraCounts[key] += 1;
      }
    } else {
      const key = `${e.fromScale}->${e.toScale}`;
      crossCounts[key] = crossCounts[key] || 0;
      if (crossCounts[key] < 2) {
        cross.push(e);
        crossCounts[key] += 1;
      }
    }
  }
  return intra.concat(cross.slice(0, 12))
    .sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight))
    .slice(0, 28);
}

function renderFoodWeb(eco) {
  const svg = $("eco-web");
  if (!svg) return;
  const active = new Set((eco.nodes || []));
  const centrality = eco.centrality || {};
  const keystone = eco.keystone;
  const edges = _selectWebEdges(eco);
  // Top-K edges so the web doesn't become a hairball — keep ~24 strongest.
  edges.sort((a, b) => Math.abs(b.weight) - Math.abs(a.weight));
  const top = edges.slice(0, 28);

  const defs = `<defs>
    <marker id="eco-arrow" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#3b82f6" opacity="0.65"/>
    </marker>
    <marker id="eco-arrow-pos" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#22c55e" opacity="0.75"/>
    </marker>
    <marker id="eco-arrow-neg" viewBox="0 0 8 8" refX="7" refY="4"
            markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#ef4444" opacity="0.75"/>
    </marker>
  </defs>`;

  // edges
  let edgePaths = "";
  for (const e of top) {
    const a = _webNodePos(e.from, eco), b = _webNodePos(e.to, eco);
    if (!a || !b) continue;
    // Use a slight curve so opposing arrows don't overlap
    const dx = b.x - a.x, dy = b.y - a.y;
    const mx = (a.x + b.x) / 2, my = (a.y + b.y) / 2;
    // perpendicular offset for curvature
    const len = Math.hypot(dx, dy) || 1;
    const sameScale = a.scale === b.scale;
    let cx, cy;
    if (sameScale) {
      const rx = mx - WEB_CENTER.x;
      const ry = my - WEB_CENTER.y;
      const rl = Math.hypot(rx, ry) || 1;
      cx = mx + rx / rl * 22;
      cy = my + ry / rl * 22;
    } else {
      const ox = -dy / len * 10;
      const oy = dx / len * 10;
      cx = mx + ox;
      cy = my + oy;
    }
    // shorten endpoints so arrows don't hide inside nodes
    const ang1 = Math.atan2(cy - a.y, cx - a.x);
    const ang2 = Math.atan2(b.y - cy, b.x - cx);
    const r = 15;
    const x1 = a.x + Math.cos(ang1) * r, y1 = a.y + Math.sin(ang1) * r;
    const x2 = b.x - Math.cos(ang2) * r, y2 = b.y - Math.sin(ang2) * r;
    const cls = e.weight > 0 ? "pos" : "neg";
    const marker = e.weight > 0 ? "eco-arrow-pos" : "eco-arrow-neg";
    const w = Math.min(2.7, 0.5 + Math.abs(e.weight) * 3.0);
    const edgeKind = sameScale ? "intra" : "cross";
    edgePaths += `<path d="M${x1.toFixed(1)},${y1.toFixed(1)} Q${cx.toFixed(1)},${cy.toFixed(1)} ${x2.toFixed(1)},${y2.toFixed(1)}"
      class="eco-edge ${cls} ${edgeKind}" stroke-width="${w.toFixed(2)}"
      marker-end="url(#${marker})" opacity="${(0.28 + Math.abs(e.weight) * 0.55).toFixed(2)}"/>`;
  }

  // nodes
  let nodeMarkup = "";
  const scaleRings = ["days","hours","minutes","seconds"].map(scale => {
    const r = WEB_SCALE_RADIUS[scale];
    const label = scale.toUpperCase();
    return `<circle cx="${WEB_CENTER.x}" cy="${WEB_CENTER.y}" r="${r}" class="eco-scale-ring ${scale}"/>
      <text x="${WEB_CENTER.x + r + 5}" y="${WEB_CENTER.y + 1}" class="eco-scale-label">${label}</text>`;
  }).join("");
  const renderNodes = []
    .concat(_webNodesForScale("days", eco))
    .concat(_webNodesForScale("hours", eco))
    .concat(_webNodesForScale("minutes", eco))
    .concat(_webNodesForScale("seconds", eco));
  for (const name of renderNodes) {
    const p = _webNodePos(name, eco);
    if (!p) continue;
    const isActive = active.has(name);
    const isKey = name === keystone;
    const c = centrality[name] || 0;
    const radius = 9 + c * 5;
    const opacity = isActive ? 1.0 : 0.25;
    const lx = p.x + Math.cos(p.angle) * (radius + 8);
    const ly = p.y + Math.sin(p.angle) * (radius + 8);
    const anchor = Math.cos(p.angle) > 0.25 ? "start"
      : Math.cos(p.angle) < -0.25 ? "end" : "middle";
    nodeMarkup += `<g>
      <title>${WEB_NODE_LABEL[name] || name} (${p.scale})</title>
      <circle cx="${p.x.toFixed(1)}" cy="${p.y.toFixed(1)}" r="${radius.toFixed(1)}"
        class="eco-node ${isKey ? 'keystone' : ''}" opacity="${opacity}"/>
      <text x="${lx.toFixed(1)}" y="${(ly + 3).toFixed(1)}"
        class="eco-node-label ${isKey ? 'keystone' : ''}" opacity="${opacity}"
        style="text-anchor:${anchor}">${WEB_NODE_LABEL[name] || name}</text>
    </g>`;
  }

  svg.innerHTML = defs + scaleRings + edgePaths + nodeMarkup;
}

function renderPhaseRing(activePhase) {
  const svg = $("eco-ring");
  if (!svg) return;
  // 6 phases around a circle, churn drawn in the middle as the "rest state"
  const ring = PHASE_ORDER.filter(p => p !== "churn");  // 5 around the perimeter
  const cx = 120, cy = 120, R = 80;
  const defs = `<defs>
    <marker id="eco-ring-arrow-head" viewBox="0 0 8 8" refX="6" refY="4"
            markerWidth="5" markerHeight="5" orient="auto">
      <path d="M0,0 L8,4 L0,8 Z" fill="#3b82f6" opacity="0.55"/>
    </marker>
  </defs>`;
  // Succession arrows between consecutive ring positions
  let arrows = "";
  let nodes = "";
  let labels = "";
  for (let i = 0; i < ring.length; i++) {
    const ph = ring[i];
    const t = (i / ring.length) * Math.PI * 2 - Math.PI / 2;
    const x = cx + R * Math.cos(t);
    const y = cy + R * Math.sin(t);
    const isActive = ph === activePhase;
    const fillCol = isActive ? PHASE_COLORS[ph] + "33" : "#141925";
    nodes += `<g class="eco-ring-node ${isActive ? 'active ' + ph : ''}">
      <circle cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="${isActive ? 26 : 22}"
              style="fill:${fillCol}"/>
    </g>`;
    labels += `<text x="${x.toFixed(1)}" y="${(y + 3).toFixed(1)}"
      class="eco-ring-label" style="${isActive ? `fill:${PHASE_COLORS[ph]}` : ''}">${ph.slice(0,5).toUpperCase()}</text>`;
    // succession arrow to next
    const j = (i + 1) % ring.length;
    const t2 = (j / ring.length) * Math.PI * 2 - Math.PI / 2;
    const x2 = cx + R * Math.cos(t2);
    const y2 = cy + R * Math.sin(t2);
    // shorten so arrow doesn't hide inside node
    const ax = Math.atan2(y2 - y, x2 - x);
    const ox1 = x + Math.cos(ax) * 26;
    const oy1 = y + Math.sin(ax) * 26;
    const ox2 = x2 - Math.cos(ax) * 28;
    const oy2 = y2 - Math.sin(ax) * 28;
    // gentle outward arc
    const mxa = (ox1 + ox2) / 2, mya = (oy1 + oy2) / 2;
    const dvx = mxa - cx, dvy = mya - cy;
    const dvl = Math.hypot(dvx, dvy) || 1;
    const cxa = mxa + dvx / dvl * 16, cya = mya + dvy / dvl * 16;
    arrows += `<path d="M${ox1.toFixed(1)},${oy1.toFixed(1)} Q${cxa.toFixed(1)},${cya.toFixed(1)} ${ox2.toFixed(1)},${oy2.toFixed(1)}"
      class="eco-ring-arrow"/>`;
  }
  // central CHURN node
  const churnActive = activePhase === "churn";
  const churnFill = churnActive ? PHASE_COLORS.churn + "55" : "#1a2238";
  const churn = `<g class="eco-ring-node ${churnActive ? 'active churn' : ''}">
    <circle cx="${cx}" cy="${cy}" r="${churnActive ? 30 : 26}"
            style="fill:${churnFill};stroke:${churnActive ? PHASE_COLORS.churn : '#2a3554'};stroke-width:${churnActive ? 3 : 1.5}"/>
  </g>
  <text x="${cx}" y="${cy - 2}" class="eco-ring-center">CHURN</text>
  <text x="${cx}" y="${cy + 10}" class="eco-ring-center sub">(rest state)</text>`;
  svg.innerHTML = defs + arrows + nodes + labels + churn;
}

function renderScaleNetworks(ms, eco) {
  const el = $("eco-scales");
  if (!el) return;
  const hasMs = !!(ms && ms.subnetworks);
  const sub = hasMs ? (ms.subnetworks || {}) : {};
  const order = ms.scale_order || ["seconds","minutes","hours","days"];
  const cards = order.map(scale => {
    const sn = sub[scale] || {};
    const nm = sn.network_metrics || {};
    const fallbackNodes = ((eco && eco.nodes) || []).filter(n => _scaleForNode(n, eco) === scale);
    const scaleNodes = (sn.nodes && sn.nodes.length) ? sn.nodes : fallbackNodes;
    const key = sn.keystone || "—";
    const nodes = scaleNodes.length;
    const keyRaw = sn.keystone || scaleNodes[0] || "-";
    const keyLabel = WEB_NODE_LABEL[keyRaw] || keyRaw;
    if (!sn.keystone && key && keyLabel !== "-") WEB_NODE_LABEL[key] = keyLabel;
    const reserve = typeof nm.rel_reserve === "number" ? nm.rel_reserve.toFixed(2) : "—";
    const asc = typeof nm.rel_ascendancy === "number" ? nm.rel_ascendancy.toFixed(2) : "—";
    return `<span class="eco-feed on" title="${scale} subnetwork: ${nodes} nodes">
      ${scale}: ${WEB_NODE_LABEL[key] || key} · A ${asc}/R ${reserve}
    </span>`;
  }).join("");
  const cross = (ms.cross_scale_edges || []).slice(0, 3).map(e =>
    `<span class="eco-feed" title="${e.n_edges || 0} cross-scale edges">
      ${e.from_scale}→${e.to_scale} ${Number(e.weight || 0).toFixed(2)}
    </span>`).join("");
  el.innerHTML = cards + cross;
}

function renderDrivers(d) {
  const items = [
    ["vol_z", d.vol_z, 4],
    ["spread_z", d.spread_z, 4],
    ["stretch_z", d.stretch_z, 5],
    ["liq_z", d.liq_proxy_z, 4],
    ["oi_z", d.oi_change_z, 4],
    ["disturbance", d.disturbance, 4],
    ["sec_dist", d.seconds_disturbance, 4],
    ["proj_dist", d.disturbance_projected, 4],
  ];
  $("eco-drivers").innerHTML = items.map(([name, val, max]) => {
    const v = (typeof val === "number") ? val : 0;
    const pct = Math.min(100, Math.abs(v) / max * 100);
    const cls = v > 0 ? "pos" : v < 0 ? "neg" : "";
    const left = name === "disturbance" ? 0 : 50;  // signed bars center, disturbance left
    const barI = name === "disturbance"
      ? `<i class="${cls}" style="left:0;width:${pct}%"></i>`
      : (v >= 0
          ? `<i class="${cls}" style="left:50%;width:${(pct/2).toFixed(1)}%"></i>`
          : `<i class="${cls}" style="left:${(50-pct/2).toFixed(1)}%;width:${(pct/2).toFixed(1)}%"></i>`);
    return `<div class="eco-driver">
      <div class="k">${name}</div>
      <div class="v">${v.toFixed(2)}<div class="bar">${barI}</div></div>
    </div>`;
  }).join("");
}

function renderOrganisms(o) {
  const scores = o.scores || {};
  const alloc = o.allocation || {};
  const names = ["scavenger","decomposer","producer","mycelium","predator","immune"];
  const topByAlloc = Object.entries(alloc).sort((a,b)=>b[1]-a[1])[0];
  const topName = topByAlloc ? topByAlloc[0] : null;
  $("eco-organisms").innerHTML = names.map(n => {
    const sc = scores[n] || 0;
    const al = alloc[n] || 0;
    return `<div class="eco-organism ${n===topName?'top':''}">
      <span class="name">${ORG_LABEL[n] || n}</span>
      <div class="bar"><i style="width:${(sc*100).toFixed(1)}%"></i></div>
      <span class="score">${sc.toFixed(2)}</span>
      <span class="alloc">${(al*100).toFixed(0)}%</span>
    </div>`;
  }).join("");
}

function renderFeeds(feeds) {
  feeds = feeds || {};
  const expected = [
    ["coinbase", "Coinbase spot"],
    ["coinbase_book", "Coinbase book"],
    ["kalshi_eth", "Kalshi ETH"],
    ["hyperliquid", "Hyperliquid OI/funding"],
    ["hyperliquid_breadth", "HL breadth/book"],
    ["deribit", "Deribit surface"],
    ["kraken", "Kraken dispersion"],
    ["crypto_breadth", "Crypto breadth"],
    ["mempool", "Mempool"],
  ];
  $("eco-feeds").innerHTML = expected.map(([k, label]) => {
    const on = feeds[k];
    const cls = on === true ? "on" : on === false ? "off" : "";
    return `<span class="eco-feed ${cls}">${label}: ${on === true ? "OK" : on === false ? "down" : "—"}</span>`;
  }).join("");
}

function drawChart(candles) {
  const cv = $("chart"); const ctx = cv.getContext("2d");
  const w = cv.width = cv.clientWidth * (window.devicePixelRatio||1);
  const h = cv.height = 140 * (window.devicePixelRatio||1);
  ctx.clearRect(0,0,w,h);
  if (!candles.length) return;
  const lows = candles.map(c=>c.low), highs = candles.map(c=>c.high);
  const min = Math.min(...lows), max = Math.max(...highs); const range = (max-min)||1;
  const n = candles.length; const cw = w/n;
  const y = (p) => h - ((p-min)/range)*(h-10) - 5;
  candles.forEach((c,i) => {
    const x = i*cw + cw/2; const up = c.close >= c.open;
    ctx.strokeStyle = up ? "#22c55e" : "#ef4444"; ctx.fillStyle = ctx.strokeStyle;
    ctx.lineWidth = 1*(window.devicePixelRatio||1);
    ctx.beginPath(); ctx.moveTo(x, y(c.high)); ctx.lineTo(x, y(c.low)); ctx.stroke();
    const bw = Math.max(cw*0.6, 1);
    const yo = y(c.open), yc = y(c.close);
    ctx.fillRect(x-bw/2, Math.min(yo,yc), bw, Math.max(Math.abs(yc-yo),1));
  });
}

// ---------------- loop ----------------
// Adaptive refresh: as fast as 1s when the trading loop is "hot" (sub-10s
// cycles), down to 5s when quiet. We chain setTimeout so the cadence updates
// the instant the server reports a new interval. Two reasons to clamp at 1s:
// avoid hammering the local HTTP server, and keep the UI thread breathing.
const REFRESH_MIN_MS = 1000, REFRESH_MAX_MS = 5000;
let _refreshTimer = null;
function _nextRefreshMs(s) {
  const intvl = (s && s.loop_status && s.loop_status.interval_seconds) || null;
  if (!intvl) return 3000;
  // Refresh ~3x per loop cycle so we catch updates between cycles without
  // bombarding the server when the loop is slow.
  const ms = Math.round((intvl * 1000) / 3);
  return Math.max(REFRESH_MIN_MS, Math.min(REFRESH_MAX_MS, ms));
}
async function refresh() {
  let next = 3000;
  try {
    const s = await api("/api/state");
    render(s);
    next = _nextRefreshMs(s);
  } catch (e) { /* network blip — retry on default cadence */ }
  if (_refreshTimer) clearTimeout(_refreshTimer);
  _refreshTimer = setTimeout(refresh, next);
}
wire();
refresh();
