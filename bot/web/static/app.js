/* FX bot dashboard - vanilla JS, no dependencies.
 *
 * Talks to the stdlib server in bot/web/server.py over a tiny JSON API:
 *   POST /api/run    start a backtest        -> { job_id }
 *   POST /api/paper  start a live replay      -> { job_id }
 *   GET  /api/jobs/:id  poll job state        -> snapshot
 *   POST /api/jobs/:id/stop
 */

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

let pollTimer = null;
let activeJob = null;
let lastCurve = [];   // kept so the chart can redraw on window resize

/* ------------------------------------------------------------------ config */

function readConfig() {
  const num = (id, fallback) => {
    const raw = parseFloat($(id).value);
    return Number.isFinite(raw) ? raw : fallback;
  };
  const list = (id) =>
    $(id).value.split(/[\s,]+/).map(Number).filter(Number.isFinite);

  const scaleOutR = list("scale_out_r");
  const scaleOutFrac = list("scale_out_frac");
  const mode = $("scale_mode").value;

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

function drawChart(curve) {
  lastCurve = curve || [];
  const host = $("chart");
  host.innerHTML = "";
  $("empty-state").style.display = curve.length ? "none" : "flex";
  if (!curve.length) return;

  const W = host.clientWidth || 800;
  const H = host.clientHeight || 300;
  const pad = { top: 12, right: 12, bottom: 22, left: 62 };
  const iw = W - pad.left - pad.right;
  const ih = H - pad.top - pad.bottom;

  const values = curve.map((p) => p.equity);
  let lo = Math.min(...values);
  let hi = Math.max(...values);
  if (hi - lo < 1e-9) { lo -= 1; hi += 1; }
  const padY = (hi - lo) * 0.08;
  lo -= padY; hi += padY;

  const x = (i) => pad.left + (i / Math.max(curve.length - 1, 1)) * iw;
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

  // gridlines + y labels
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

  // area under the curve
  const line = values.map((v, i) => `${i ? "L" : "M"}${x(i).toFixed(1)},${y(v).toFixed(1)}`).join("");
  add("path", {
    d: `${line} L${x(values.length - 1)},${pad.top + ih} L${x(0)},${pad.top + ih} Z`,
    fill: "rgba(76,141,255,.10)",
  });
  add("path", { d: line, fill: "none", stroke: "#4c8dff", "stroke-width": 1.6 });

  // starting-cash reference line
  const start = values[0];
  add("line", {
    x1: pad.left, x2: W - pad.right, y1: y(start), y2: y(start),
    stroke: "#7d8aa3", "stroke-width": 1, "stroke-dasharray": "3 4",
  });

  // first / last time labels
  const first = add("text", { x: pad.left, y: H - 6, fill: "#7d8aa3", "font-size": 10, "font-family": "ui-monospace, monospace" });
  first.textContent = fmtTime(curve[0].time);
  const last = add("text", { x: W - pad.right, y: H - 6, fill: "#7d8aa3", "font-size": 10, "text-anchor": "end", "font-family": "ui-monospace, monospace" });
  last.textContent = fmtTime(curve[curve.length - 1].time);

  host.appendChild(svg);

  $("chart-legend").innerHTML =
    `<span><i style="background:#4c8dff"></i>equity</span>` +
    `<span><i style="background:#7d8aa3"></i>start ${fmtMoney(start)}</span>`;
}

/* ------------------------------------------------------------------- stats */

function renderStats(stats) {
  const host = $("stats");
  host.innerHTML = "";
  const rows = [
    ["Net P&L", fmtMoney(stats.net_pnl), stats.net_pnl >= 0 ? "pos" : "neg"],
    ["Return", fmtPct(stats.total_return_pct), stats.total_return_pct >= 0 ? "pos" : "neg"],
    ["Final equity", fmtMoney(stats.final_equity), ""],
    ["Max drawdown", `${fmtNum(stats.max_drawdown_pct)}%`, "neg"],
    ["Sharpe", fmtNum(stats.sharpe), stats.sharpe >= 0 ? "pos" : "neg"],
    ["Profit factor", stats.profit_factor === null ? "∞" : fmtNum(stats.profit_factor), ""],
    ["Trades", String(stats.trades), ""],
    ["Win rate", `${fmtNum(stats.win_rate, 1)}%`, ""],
    ["Wins / losses", `${stats.wins} / ${stats.losses}`, ""],
    ["Expectancy", fmtMoney(stats.expectancy), stats.expectancy >= 0 ? "pos" : "neg"],
    ["Expectancy R", `${stats.expectancy_r >= 0 ? "+" : ""}${fmtNum(stats.expectancy_r)}R`, stats.expectancy_r >= 0 ? "pos" : "neg"],
    ["Avg win / loss", `${fmtMoney(stats.avg_win)} / ${fmtMoney(stats.avg_loss)}`, ""],
    ["Exposure", `${fmtNum(stats.exposure_pct, 1)}%`, ""],
  ];
  for (const [k, v, cls] of rows) {
    const box = el("div", "stat");
    box.appendChild(el("div", "k", k));
    box.appendChild(el("div", `v ${cls}`.trim(), v));
    host.appendChild(box);
  }
}

/* ----------------------------------------------------------------- scaling */

function renderScaling(scalingStats, events) {
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
  const shown = trades.slice(-60).reverse();
  for (const t of shown) {
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
    for (const [text, cls] of cells) {
      const td = el("td", cls, text);
      tr.appendChild(td);
    }
    body.appendChild(tr);
  }
}

function renderEvents(events) {
  const body = $("events-body");
  body.innerHTML = "";
  const KINDS = [
    "entry", "attach", "scale_in", "add_skipped", "scale_out",
    "breakeven", "trail", "time_stop", "risk_rebalanced",
    "exit_flip", "risk_block", "size_zero", "trade_closed",
  ];
  const interesting = events.filter((e) => KINDS.includes(e.kind));
  const shown = interesting.slice(-50).reverse();
  for (const e of shown) {
    const tr = el("tr");
    const kindTd = el("td");
    kindTd.appendChild(el("span", `ev-kind ev-${e.kind}`, e.kind));
    tr.appendChild(kindTd);
    let detail = "";
    if (e.kind === "scale_in") detail = `tranche ${e.tranche} +${Math.round(e.qty).toLocaleString()} @ ${Number(e.price).toFixed(5)} (${fmtNum(e.progress_r, 1)}R)`;
    else if (e.kind === "scale_out") detail = `${e.r}R · close ${Math.round(e.qty).toLocaleString()} @ ${Number(e.price).toFixed(5)}`;
    else if (e.kind === "entry") detail = `${e.side} ${Math.round(e.qty).toLocaleString()} @ ${Number(e.price).toFixed(5)} · ${e.reason ?? ""}`;
    else if (e.kind === "exit_flip") detail = `closed · ${e.reason ?? ""}`;
    else if (e.kind === "risk_block") detail = e.reason ?? "";
    else if (e.kind === "size_zero") detail = e.reason ?? "";
    else if (e.kind === "add_skipped") detail = `${e.reason ?? "blocked"}${e.price !== undefined ? ` @ ${Number(e.price).toFixed(5)}` : ""}`;
    else if (e.kind === "trade_closed") detail = `${Math.round(e.qty).toLocaleString()} closed · ${fmtMoney(e.pnl)} · ${e.reason ?? ""}`;
    else if (e.kind === "attach") detail = `entry ${Number(e.entry).toFixed(5)} · stop ${Number(e.stop).toFixed(5)}`;
    else if (e.stop !== undefined) detail = `stop → ${Number(e.stop).toFixed(5)}`;
    else if (e.bars !== undefined) detail = `after ${e.bars} bars`;
    else if (e.price !== undefined) detail = `@ ${Number(e.price).toFixed(5)}`;
    tr.appendChild(el("td", "reason", detail));
    body.appendChild(tr);
  }
}

/* -------------------------------------------------------------------- jobs */

async function startJob(endpoint) {
  const payload = readConfig();
  if (endpoint === "/api/paper") payload.delay_ms = 45;

  setButtons(false);
  setStatus("running", "starting…");
  $("progress").classList.add("on");
  const bar = $("progress").firstElementChild;
  bar.style.width = "0%";

  const res = await fetch(endpoint, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });
  const data = await res.json();
  if (!res.ok) {
    setStatus("error", data.error || "failed");
    setButtons(true);
    return;
  }
  activeJob = data.job_id;
  poll();
}

function poll() {
  clearTimeout(pollTimer);
  if (!activeJob) return;
  fetch(`/api/jobs/${activeJob}`)
    .then((r) => r.json())
    .then((job) => {
      const bar = $("progress").firstElementChild;
      bar.style.width = `${Math.round((job.progress ?? 0) * 100)}%`;

      if (job.status === "running") {
        setStatus("running", job.message || "running…");
        if (job.result) renderLive(job.result);
        pollTimer = setTimeout(poll, 220);
        return;
      }

      $("progress").classList.remove("on");
      if (job.status === "error") {
        setStatus("error", "error");
        console.error(job.message);
      } else {
        setStatus("done", job.status === "stopped" ? "stopped" : "done");
      }
      if (job.result) {
        job.result.stats ? renderBacktest(job.result) : renderLive(job.result);
      }
      setButtons(true);
      activeJob = null;
    })
    .catch((err) => {
      setStatus("error", "connection lost");
      console.error(err);
      setButtons(true);
      activeJob = null;
    });
}

function renderBacktest(result) {
  drawChart(result.equity_curve);
  renderStats(result.stats);
  renderScaling(result.scaling_stats, result.events);
  renderTrades(result.trades);
  renderEvents(result.events);
  $("equity-value").textContent = fmtMoney(result.stats.final_equity);
}

function renderLive(result) {
  drawChart(result.equity_curve);
  $("equity-value").textContent = fmtMoney(result.equity);
  renderScaling(result.scaling_stats, result.events);
  renderTrades(result.trades);
  renderEvents(result.events);

  // live positions get their own compact readout in the stats panel
  const host = $("stats");
  host.innerHTML = "";
  const open = result.positions ?? [];
  const rows = [
    ["Bar", `${result.bar} / ${result.bars}`, ""],
    ["Equity", fmtMoney(result.equity), result.equity >= 100000 ? "pos" : "neg"],
    ["Cash", fmtMoney(result.cash), ""],
    ["Unrealised", fmtMoney(result.unrealized), result.unrealized >= 0 ? "pos" : "neg"],
    ["Open positions", String(open.length), ""],
    ["Closed trades", String((result.trades ?? []).length), ""],
  ];
  if (open.length) {
    const p = open[0];
    rows.push([`${p.side} ${p.symbol}`, `${Math.round(p.qty).toLocaleString()} @ ${p.avg_price.toFixed(5)}`, p.side === "BUY" ? "pos" : "neg"]);
    rows.push(["Tranches", `${p.tranches} (+${p.adds} adds)`, ""]);
    rows.push(["Stop", p.stop_loss ? p.stop_loss.toFixed(5) : "—", ""]);
    rows.push(["Position P&L", fmtMoney(p.unrealized), p.unrealized >= 0 ? "pos" : "neg"]);
  }
  for (const [k, v, cls] of rows) {
    const box = el("div", "stat");
    box.appendChild(el("div", "k", k));
    box.appendChild(el("div", `v ${cls}`.trim(), v));
    host.appendChild(box);
  }
}

/* ------------------------------------------------------------------ wiring */

function setButtons(enabled) {
  $("run-btn").disabled = !enabled;
  $("paper-btn").disabled = !enabled;
  $("stop-btn").disabled = enabled;
}

async function stopJob() {
  if (!activeJob) return;
  await fetch(`/api/jobs/${activeJob}/stop`, { method: "POST" });
  setStatus("running", "stopping…");
}

async function init() {
  $("vol_pips").addEventListener("input", (e) => {
    $("vol-label").textContent = e.target.value;
  });
  $("run-btn").addEventListener("click", () => startJob("/api/run"));
  $("paper-btn").addEventListener("click", () => startJob("/api/paper"));
  $("stop-btn").addEventListener("click", stopJob);
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
    for (const [name, info] of Object.entries(cfg.strategies ?? {})) {
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
