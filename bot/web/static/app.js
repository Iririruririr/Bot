/* FX bot dashboard - vanilla JS, no dependencies.
 *
 * Talks to the stdlib server in bot/web/server.py over a tiny stateless JSON
 * API:
 *   POST /api/run     run a backtest, returns the whole report
 *   POST /api/replay  run a paper session, returns a bar-by-bar trace
 *
 * There is no polling and no job id: the server does all the work in one
 * request, which is what lets the same code run on a per-request host.  The
 * "watch it live" experience is playback of the trace on the client, so pausing,
 * scrubbing and changing speed are all local.
 */

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

/* ------------------------------------------------------------------ playback */

const playback = {
  frames: [],      // one entry per bar, from /api/replay
  trades: [],      // every closed trade, in order
  events: [],      // every event, in order
  stats: null,
  meta: null,
  i: 0,            // current bar
  playing: false,
  speed: 1,        // bars per tick multiplier
  timer: null,
};

const BASE_DELAY_MS = 60;   // one bar at 1x

/* ------------------------------------------------------------------- config */

function readConfig() {
  const num = (id, fallback) => {
    const raw = parseFloat($(id).value);
    return Number.isFinite(raw) ? raw : fallback;
  };
  const list = (id) =>
    $(id).value.split(/[\s,]+/).map(Number).filter(Number.isFinite);

  const mode = $("scale_mode").value;
  const scaleOutR = list("scale_out_r");
  const scaleOutFrac = list("scale_out_frac");

  return {
    strategy: $("strategy").value,
    bars: num("bars", 2500),
    seed: num("seed", 7),
    vol_pips: num("vol_pips", 12),
    scaling: {
      mode,
      tranches: mode === "off" ? 1 : num("tranches", 3),
      add_step_r: num("add_step_r", 1.0),
      add_size_mult: 1.0,
      scale_out: mode === "off" ? [] : scaleOutR.map((r, i) => [r, scaleOutFrac[i] ?? 0]),
      breakeven_at_r: mode === "off" ? null : (num("breakeven_at_r", 1) || null),
      trail_atr_mult: mode === "off" ? null : (num("trail_atr", 3) || null),
      trail_start_r: 1.0,
    },
    risk: {
      risk_per_trade_pct: num("risk_pct", 1.0),
      max_daily_loss_pct: num("max_daily_loss", 3.0),
      max_drawdown_pct: num("max_drawdown", 15.0),
      max_open_exposure_pct: num("max_exposure", 100),
    },
    broker: {
      starting_cash: 100000,
      spread_pips: num("spread_pips", 1.0),
      slippage_pips: num("slippage_pips", 0.2),
      commission_per_million: 0,
      lot_size: 1000,
    },
  };
}

function writeConfig(cfg) {
  $("spread_pips").value = cfg.broker.spread_pips;
  $("slippage_pips").value = cfg.broker.slippage_pips;
  $("risk_pct").value = cfg.risk.risk_per_trade_pct;
  $("max_daily_loss").value = cfg.risk.max_daily_loss_pct;
  $("max_drawdown").value = cfg.risk.max_drawdown_pct;
  $("max_exposure").value = cfg.risk.max_open_exposure_pct;
}

/* ------------------------------------------------------------------- utils */

const fmtMoney = (v) =>
  (v < 0 ? "-$" : "$") + Math.abs(v).toLocaleString(undefined, { maximumFractionDigits: 0 });
const fmtPct = (v) => `${v >= 0 ? "+" : ""}${v.toFixed(2)}%`;
const fmtNum = (v, d = 2) =>
  v === null || v === undefined ? "—" : Number(v).toLocaleString(undefined, { minimumFractionDigits: d, maximumFractionDigits: d });
const fmtTime = (iso) => {
  const d = new Date(iso);
  return Number.isNaN(d.getTime()) ? "—" : d.toISOString().slice(0, 16).replace("T", " ");
};

function setStatus(state, text) {
  const pill = $("conn");
  pill.className = `pill pill-${state}`;
  pill.textContent = text;
}

/* ------------------------------------------------------------------- chart */

let lastCurve = [];   // kept so the chart can redraw on window resize

function drawChart(curve) {
  lastCurve = curve || [];
  const host = $("chart");
  host.innerHTML = "";
  $("empty-state").style.display = lastCurve.length ? "none" : "flex";
  if (!lastCurve.length) return;

  const W = host.clientWidth || 800;
  const H = host.clientHeight || 300;
  const pad = { top: 12, right: 12, bottom: 22, left: 62 };
  const iw = W - pad.left - pad.right;
  const ih = H - pad.top - pad.bottom;

  const values = lastCurve.map((p) => p.equity);
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
  const padY = (hi - lo) * 0.08;
  lo -= padY; hi += padY;

  const x = (i) => pad.left + (i / Math.max(lastCurve.length - 1, 1)) * iw;
  const y = (v) => pad.top + (1 - (v - lo) / (hi - lo)) * ih;

  const NS = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(NS, "svg");
  svg.setAttribute("viewBox", `0 0 ${W} ${H}`);

  const add = (tag, attrs) => {
    const node = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    svg.appendChild(node);
    return node;
  };

  const ticks = 5;
  for (let i = 0; i <= ticks; i++) {
    const v = lo + ((hi - lo) * i) / ticks;
    const yy = y(v);
    add("line", { x1: pad.left, x2: W - pad.right, y1: yy, y2: yy, stroke: "#232c3d", "stroke-width": 1 });
    const label = add("text", {
      x: pad.left - 8, y: yy + 4, fill: "#7d8aa3", "font-size": 10,
      "text-anchor": "end", "font-family": "ui-monospace, monospace",
    });
    label.textContent = fmtMoney(v);
  }

  const line = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  add("path", {
    d: `${line} L${x(values.length - 1)},${pad.top + ih} L${x(0)},${pad.top + ih} Z`,
    fill: "rgba(76,141,255,.10)",
  });
  add("path", { d: line, fill: "none", stroke: "#4c8dff", "stroke-width": 1.6 });

  // a marker on the bar being played back, if we are mid-replay
  if (lastCurve.marker !== undefined) {
    const at = Math.min(lastCurve.marker, values.length - 1);
    const mx = x(at);
    const my = y(values[at]);
    add("line", { x1: mx, x2: mx, y1: pad.top, y2: pad.top + ih,
                  stroke: "#ffb648", "stroke-width": 1, "stroke-dasharray": "2 3" });
    add("circle", { cx: mx, cy: my, r: 3.5, fill: "#ffb648" });
  }

  const start = values[0];
  add("line", {
    x1: pad.left, x2: W - pad.right, y1: y(start), y2: y(start),
    stroke: "#7d8aa3", "stroke-width": 1, "stroke-dasharray": "3 4",
  });

  const first = add("text", { x: pad.left, y: H - 6, fill: "#7d8aa3", "font-size": 10, "font-family": "ui-monospace, monospace" });
  first.textContent = fmtTime(lastCurve[0].time);
  const last = add("text", { x: W - pad.right, y: H - 6, fill: "#7d8aa3", "font-size": 10, "text-anchor": "end", "font-family": "ui-monospace, monospace" });
  last.textContent = fmtTime(lastCurve[lastCurve.length - 1].time);

  host.appendChild(svg);

  $("chart-legend").innerHTML =
    `<span><i style="background:#4c8dff"></i>equity</span>` +
    `<span><i style="background:#7d8aa3"></i>start ${fmtMoney(start)}</span>` +
    (lastCurve.marker !== undefined ? `<span><i style="background:#ffb648"></i>bar ${lastCurve.marker}</span>` : "");
}

/* ------------------------------------------------------------------- stats */

function renderStats(rows) {
  const host = $("stats");
  host.innerHTML = "";
  for (const [k, v, cls] of rows) {
    const box = el("div", "stat");
    box.appendChild(el("div", "k", k));
    box.appendChild(el("div", `v ${cls}`.trim(), v));
    host.appendChild(box);
  }
}

function statsFromReport(stats) {
  return [
    ["Net P&L", fmtMoney(stats.net_pnl), stats.net_pnl >= 0 ? "pos" : "neg"],
    ["Return", fmtPct(stats.total_return_pct), stats.total_return_pct >= 0 ? "pos" : "neg"],
    ["Final equity", fmtMoney(stats.final_equity), ""],
    ["Max drawdown", `${fmtNum(stats.max_drawdown_pct)}%`, "neg"],
    ["Sharpe", fmtNum(stats.sharpe), stats.sharpe >= 0 ? "pos" : "neg"],
    ["Sortino", fmtNum(stats.sortino), stats.sortino >= 0 ? "pos" : "neg"],
    ["Profit factor", stats.profit_factor === null ? "∞" : fmtNum(stats.profit_factor), ""],
    ["Trades", String(stats.trades), ""],
    ["Win rate", `${fmtNum(stats.win_rate, 1)}%`, ""],
    ["Wins / losses", `${stats.wins} / ${stats.losses}`, ""],
    ["Expectancy", fmtMoney(stats.expectancy), stats.expectancy >= 0 ? "pos" : "neg"],
    ["Expectancy R", `${stats.expectancy_r >= 0 ? "+" : ""}${fmtNum(stats.expectancy_r)}R`, stats.expectancy_r >= 0 ? "pos" : "neg"],
    ["Avg win / loss", `${fmtMoney(stats.avg_win)} / ${fmtMoney(stats.avg_loss)}`, ""],
    ["Exposure", `${fmtNum(stats.exposure_pct, 1)}%`, ""],
  ];
}

function statsFromFrame(frame) {
  const open = frame.positions ?? [];
  const rows = [
    ["Bar", `${frame.bar} / ${frame.bars}`, ""],
    ["Equity", fmtMoney(frame.equity), frame.equity >= 100000 ? "pos" : "neg"],
    ["Cash", fmtMoney(frame.cash), ""],
    ["Unrealised", fmtMoney(frame.unrealized), frame.unrealized >= 0 ? "pos" : "neg"],
    ["Open positions", String(open.length), ""],
    ["Closed trades", String(frame.trades_closed), ""],
  ];
  if (open.length) {
    const p = open[0];
    rows.push([`${p.side} ${p.symbol}`, `${Math.round(p.qty).toLocaleString()} @ ${p.avg_price.toFixed(5)}`, p.side === "BUY" ? "pos" : "neg"]);
    rows.push(["Tranches", `${p.tranches} (+${p.adds} adds)`, ""]);
    rows.push(["Stop", p.stop_loss ? p.stop_loss.toFixed(5) : "—", ""]);
    rows.push(["Position P&L", fmtMoney(p.unrealized), p.unrealized >= 0 ? "pos" : "neg"]);
  }
  return rows;
}

/* ----------------------------------------------------------------- scaling */

function renderScaling(scalingStats) {
  const host = $("scaling");
  host.innerHTML = "";
  if (!scalingStats) return;
  const chips = [
    ["Scale-in adds", scalingStats.adds ?? 0, ""],
    ["Partial exits", scalingStats.scale_outs ?? 0, ""],
    ["Breakeven moves", scalingStats.breakevens ?? 0, ""],
    ["Trail updates", scalingStats.trail_updates ?? 0, ""],
    ["Adds blocked", scalingStats.skipped_adds ?? 0, (scalingStats.skipped_adds ?? 0) > 0 ? "warn" : ""],
  ];
  for (const [label, value, cls] of chips) {
    const chip = el("div", `chip ${cls}`.trim());
    chip.appendChild(el("b", "", String(value)));
    chip.appendChild(el("span", "", label));
    host.appendChild(chip);
  }
}

/* ------------------------------------------------------------------ tables */

function renderTrades(trades) {
  const body = $("trades-body");
  body.innerHTML = "";
  $("trade-count").textContent = trades.length ? `${trades.length} closed` : "";
  for (const t of trades.slice(-60).reverse()) {
    const tr = el("tr");
    const cells = [
      [fmtTime(t.exit_time), ""],
      [t.side, t.side === "BUY" ? "side-buy" : "side-sell"],
      [t.qty.toLocaleString(undefined, { maximumFractionDigits: 0 }), "num"],
      [t.entry_price.toFixed(5), "num"],
      [t.exit_price.toFixed(5), "num"],
      [fmtMoney(t.net_pnl), `num ${t.net_pnl >= 0 ? "pos" : "neg"}`],
      [`${t.max_r >= 0 ? "+" : ""}${fmtNum(t.max_r, 1)}R`, "num"],
      [t.exit_reason, "reason"],
    ];
    for (const [text, cls] of cells) tr.appendChild(el("td", cls, text));
    body.appendChild(tr);
  }
}

const EVENT_KINDS = [
  "entry", "attach", "scale_in", "add_skipped", "scale_out",
  "breakeven", "trail", "time_stop", "risk_rebalanced",
  "exit_flip", "risk_block", "size_zero", "trade_closed",
];

function renderEvents(events) {
  const body = $("events-body");
  body.innerHTML = "";
  for (const e of events.filter((x) => EVENT_KINDS.includes(x.kind)).slice(-50).reverse()) {
    const tr = el("tr");
    const kindTd = el("td");
    kindTd.appendChild(el("span", `ev-kind ev-${e.kind}`, e.kind));
    tr.appendChild(kindTd);
    tr.appendChild(el("td", "reason", eventDetail(e)));
    body.appendChild(tr);
  }
}

function eventDetail(e) {
  if (e.kind === "scale_in") return `tranche ${e.tranche} +${Math.round(e.qty).toLocaleString()} @ ${Number(e.price).toFixed(5)} (${fmtNum(e.progress_r, 1)}R)`;
  if (e.kind === "scale_out") return `${e.r}R · close ${Math.round(e.qty).toLocaleString()} @ ${Number(e.price).toFixed(5)}`;
  if (e.kind === "entry") return `${e.side} ${Math.round(e.qty).toLocaleString()} @ ${Number(e.price).toFixed(5)} · ${e.reason ?? ""}`;
  if (e.kind === "exit_flip") return `closed · ${e.reason ?? ""}`;
  if (e.kind === "trade_closed") return `${Math.round(e.qty).toLocaleString()} closed · ${fmtMoney(e.pnl)} · ${e.reason ?? ""}`;
  if (e.kind === "add_skipped") return `${e.reason ?? "blocked"}${e.price !== undefined ? ` @ ${Number(e.price).toFixed(5)}` : ""}`;
  if (e.kind === "attach") return `entry ${Number(e.entry).toFixed(5)} · stop ${Number(e.stop).toFixed(5)}`;
  if (e.kind === "risk_block" || e.kind === "size_zero") return e.reason ?? "";
  if (e.stop !== undefined) return `stop → ${Number(e.stop).toFixed(5)}`;
  if (e.bars !== undefined) return `after ${e.bars} bars`;
  if (e.price !== undefined) return `@ ${Number(e.price).toFixed(5)}`;
  return "";
}

/* -------------------------------------------------------------- rendering */

function renderBacktest(result) {
  stopPlayback();
  clearReplayControls();
  drawChart(result.equity_curve);
  renderStats(statsFromReport(result.stats));
  renderScaling(result.scaling_stats);
  renderTrades(result.trades);
  renderEvents(result.events);
  $("equity-value").textContent = fmtMoney(result.stats.final_equity);
}

function renderFrameAt(i) {
  const frame = playback.frames[i];
  if (!frame) return;
  playback.i = i;

  // the equity curve grows as the replay advances, so the chart "draws" itself
  const curve = playback.frames.slice(0, i + 1).map((f) => ({ time: null, equity: f.equity }));
  curve.marker = i;
  drawChart(curve);

  renderStats(statsFromFrame(frame));
  renderTrades(playback.trades.slice(0, frame.trades_closed));
  renderEvents(playback.events.slice(0, frame.events_seen));

  $("scrub").value = String(i);
  $("scrub").max = String(playback.frames.length - 1);
  $("bar-readout").textContent = `bar ${i} / ${playback.frames.length - 1}`;
  $("equity-value").textContent = fmtMoney(frame.equity);
}

/* --------------------------------------------------------------- playback */

function play() {
  if (playback.playing || !playback.frames.length) return;
  if (playback.i >= playback.frames.length - 1) playback.i = 0;   // replay from the top
  playback.playing = true;
  $("play-btn").textContent = "Pause";
  setStatus("running", "replaying…");

  const tick = () => {
    if (!playback.playing) return;
    if (playback.i >= playback.frames.length - 1) {
      finishPlayback();
      return;
    }
    renderFrameAt(playback.i + 1);
    playback.timer = setTimeout(tick, BASE_DELAY_MS / playback.speed);
  };
  playback.timer = setTimeout(tick, BASE_DELAY_MS / playback.speed);
}

function pause() {
  playback.playing = false;
  clearTimeout(playback.timer);
  $("play-btn").textContent = "Play";
  setStatus("idle", "paused");
}

function finishPlayback() {
  playback.playing = false;
  clearTimeout(playback.timer);
  $("play-btn").textContent = "Replay";
  renderScaling(playback.stats);
  renderTrades(playback.trades);
  renderEvents(playback.events);
  setStatus("done", "replay complete");
}

function stopPlayback() {
  playback.playing = false;
  clearTimeout(playback.timer);
  if ($("play-btn")) $("play-btn").textContent = "Play";
}

function clearReplayControls() {
  $("scrub").value = "0";
  $("scrub").max = "0";
  $("bar-readout").textContent = "bar —";
}

/* ------------------------------------------------------------------ actions */

async function runBacktest() {
  stopPlayback();
  setStatus("running", "running backtest…");
  setBusy(true);
  try {
    const res = await fetch("/api/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readConfig()),
    });
    const data = await res.json();
    if (!res.ok) {
      setStatus("error", data.error || "failed");
      console.error(data.error);
      return;
    }
    renderBacktest(data);
    setStatus("done", `${data.stats.trades} trades`);
  } catch (err) {
    setStatus("error", "request failed");
    console.error(err);
  } finally {
    setBusy(false);
  }
}

async function startReplay() {
  stopPlayback();
  setStatus("running", "computing replay…");
  setBusy(true);
  try {
    const res = await fetch("/api/replay", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(readConfig()),
    });
    const data = await res.json();
    if (!res.ok) {
      setStatus("error", data.error || "failed");
      console.error(data.error);
      return;
    }
    playback.frames = data.frames;
    playback.trades = data.trades;
    playback.events = data.events;
    playback.stats = data.scaling_stats;
    playback.meta = data.meta;
    playback.i = 0;
    // the transport only makes sense once there is something to play
    $("play-btn").disabled = false;
    $("scrub").disabled = false;
    $("speed").disabled = false;
    renderFrameAt(0);
    renderScaling(playback.stats);
    setStatus("idle", "replay ready");
    play();
  } catch (err) {
    setStatus("error", "request failed");
    console.error(err);
  } finally {
    setBusy(false);
  }
}

function setBusy(busy) {
  $("run-btn").disabled = busy;
  $("replay-btn").disabled = busy;
}

/* ------------------------------------------------------------------- wiring */

async function init() {
  $("vol_pips").addEventListener("input", (e) => {
    $("vol-label").textContent = e.target.value;
  });
  $("run-btn").addEventListener("click", runBacktest);
  $("replay-btn").addEventListener("click", startReplay);
  $("play-btn").addEventListener("click", () => (playback.playing ? pause() : play()));
  $("scrub").addEventListener("input", (e) => {
    pause();
    renderFrameAt(Number(e.target.value));
  });
  $("speed").addEventListener("change", (e) => {
    playback.speed = Number(e.target.value);
    if (playback.playing) { pause(); play(); }
  });

  // space toggles playback, but not while typing in a field
  document.addEventListener("keydown", (e) => {
    if (e.code !== "Space" || /INPUT|SELECT|TEXTAREA/.test(document.activeElement.tagName)) return;
    e.preventDefault();
    if (playback.frames.length) playback.playing ? pause() : play();
  });

  // mobile browsers fire resize while the URL bar collapses - debounce it
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      if (lastCurve.length) drawChart(lastCurve);
    }, 150);
  });

  try {
    const res = await fetch("/api/config");
    const cfg = await res.json();
    writeConfig(cfg);
    const select = $("strategy");
    for (const name of Object.keys(cfg.strategies ?? {})) {
      const opt = el("option", "", name);
      opt.value = name;
      if (name === "ma_crossover") opt.selected = true;
      select.appendChild(opt);
    }
    setStatus("idle", "ready");
  } catch (err) {
    setStatus("error", "server unreachable");
    console.error(err);
  }
}

document.addEventListener("DOMContentLoaded", init);
