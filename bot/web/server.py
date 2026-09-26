"""Web dashboard for the FX trading bot and scaling bot.

Stdlib-only HTTP server (``http.server``) that serves a single-page dashboard
plus a small JSON API, so running the site adds no dependencies.

    python -m bot web --port 8000

The dashboard can backtest any strategy/scaling configuration, and it can run a
**live paper session** that replays bars in real time so you can watch the
scaling bot add to winners and scale out at each R level as it happens.
"""
from __future__ import annotations

import json
import threading
import time
import traceback
import uuid
from dataclasses import asdict, dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional
from urllib.parse import urlparse

STATIC_DIR = Path(__file__).parent / "static"


# --------------------------------------------------------------------------- #
# job management
# --------------------------------------------------------------------------- #


@dataclass
class Job:
    id: str
    kind: str                      # "backtest" | "paper"
    status: str = "running"        # running | done | error | stopped
    created_at: float = field(default_factory=time.time)
    progress: float = 0.0
    message: str = ""
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    stop_requested: bool = False

    def snapshot(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "error": self.error,
            "result": self.result,
        }


class JobManager:
    """Runs long work off the request thread and keeps the latest state."""

    def __init__(self) -> None:
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()

    def submit(self, kind: str, fn: Callable[[Job], None]) -> str:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind)
        with self._lock:
            self._jobs[job.id] = job

        def runner() -> None:
            try:
                fn(job)
                if job.status == "running":
                    job.status = "done"
            except Exception as exc:  # pragma: no cover - defensive
                job.status = "error"
                job.error = f"{type(exc).__name__}: {exc}"
                job.message = traceback.format_exc(limit=6)

        thread = threading.Thread(target=runner, daemon=True)
        thread.start()
        return job.id

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def all(self) -> List[Job]:
        with self._lock:
            return list(self._jobs.values())

    def stop(self, job_id: str) -> bool:
        job = self.get(job_id)
        if job is None:
            return False
        job.stop_requested = True
        return True


# --------------------------------------------------------------------------- #
# the work itself
# --------------------------------------------------------------------------- #


def run_backtest_job(job: Job, payload: Dict[str, Any]) -> None:
    """Execute one backtest and stash the report payload on the job."""
    from bot.backtest.runner import run_backtest
    from bot.brokers.paper import BrokerConfig
    from bot.core.risk import RiskConfig
    from bot.data.feed import SimConfig, SimulatedFeed
    from bot.scaling.engine import ScalingConfig
    from bot.strategies import build

    strategy_name = payload.get("strategy", "ma_crossover")
    strategy_params = payload.get("strategy_params") or {}
    bars = int(payload.get("bars", 2000))
    seed = int(payload.get("seed", 7))
    symbol = payload.get("symbol", "EUR/USD")

    scaling_cfg = ScalingConfig.from_dict(payload.get("scaling") or {})
    risk_cfg = RiskConfig(**{k: v for k, v in (payload.get("risk") or {}).items()
                             if k in RiskConfig.__dataclass_fields__})
    broker_cfg = BrokerConfig(**{k: v for k, v in (payload.get("broker") or {}).items()
                                 if k in BrokerConfig.__dataclass_fields__})

    strategy = build(strategy_name, **strategy_params)
    feed = SimulatedFeed(
        SimConfig(symbol=symbol, bars=bars, seed=seed, start_price=1.0800,
                  vol_pips=float(payload.get("vol_pips", 12.0)))
    )

    job.message = "running backtest"
    result = run_backtest(
        feed=feed,
        strategy=strategy,
        scaling=scaling_cfg,
        risk=risk_cfg,
        broker=broker_cfg,
    )

    job.progress = 1.0
    job.result = result_payload(result)
    job.message = f"{result.stats.trades} trades"
    job.status = "done"


def result_payload(result) -> Dict[str, Any]:
    """Turn a BacktestResult into JSON the dashboard can render."""
    from bot.backtest.stats import Stats

    stats: Stats = result.stats
    return {
        "meta": result.meta,
        "stats": {
            "starting_equity": stats.starting_equity,
            "net_pnl": stats.net_pnl,
            "total_return_pct": stats.total_return_pct,
            "final_equity": stats.final_equity,
            "max_drawdown_pct": stats.max_drawdown_pct,
            "max_drawdown_amount": stats.max_drawdown_amount,
            "sharpe": stats.sharpe,
            "sortino": stats.sortino,
            "profit_factor": stats.profit_factor if stats.profit_factor != float("inf") else None,
            "trades": stats.trades,
            "wins": stats.wins,
            "losses": stats.losses,
            "win_rate": stats.win_rate,
            "expectancy": stats.expectancy,
            "expectancy_r": stats.expectancy_r,
            "avg_win": stats.avg_win,
            "avg_loss": stats.avg_loss,
            "largest_win": stats.largest_win,
            "largest_loss": stats.largest_loss,
            "total_commission": stats.total_commission,
            "exposure_pct": stats.exposure_pct,
        },
        "equity_curve": [
            {"time": when.isoformat(), "equity": equity}
            for when, equity in result.equity_curve
        ],
        "trades": [
            {
                "symbol": t.symbol,
                "side": t.side.value,
                "qty": t.qty,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "entry_time": t.entry_time.isoformat(),
                "exit_time": t.exit_time.isoformat(),
                "pnl": t.pnl,
                "commission": t.commission,
                "net_pnl": t.net_pnl,
                "exit_reason": t.exit_reason,
                "max_r": t.max_r,
            }
            for t in result.trades
        ],
        "scaling_stats": result.scaling_stats,
        "events": [
            {k: v for k, v in event.items() if isinstance(v, (str, int, float, bool, type(None)))}
            for event in result.events[-400:]
        ],
    }


def run_paper_job(job: Job, payload: Dict[str, Any]) -> None:
    """Replay a simulated feed bar-by-bar so the dashboard can watch it live."""
    from bot.brokers.paper import BrokerConfig, PaperBroker
    from bot.core.engine import Engine, EngineConfig
    from bot.core.risk import RiskConfig, RiskManager
    from bot.data.feed import SimConfig, SimulatedFeed
    from bot.scaling.engine import ScalingConfig, ScalingEngine
    from bot.strategies import build

    bars = int(payload.get("bars", 600))
    seed = int(payload.get("seed", 7))
    delay = max(float(payload.get("delay_ms", 40)) / 1000.0, 0.0)
    symbol = payload.get("symbol", "EUR/USD")

    broker = PaperBroker(BrokerConfig(**{k: v for k, v in (payload.get("broker") or {}).items()
                                         if k in BrokerConfig.__dataclass_fields__}))
    risk = RiskManager(RiskConfig(**{k: v for k, v in (payload.get("risk") or {}).items()
                                     if k in RiskConfig.__dataclass_fields__}),
                       broker.config.starting_cash)
    scaling = ScalingEngine(ScalingConfig.from_dict(payload.get("scaling") or {}), broker, risk=risk)
    strategy = build(payload.get("strategy", "ma_crossover"), **(payload.get("strategy_params") or {}))
    strategy.warmup = min(strategy.warmup, 40)
    engine = Engine(
        broker=broker,
        strategies=[strategy],
        risk=risk,
        scaling=scaling,
        config=EngineConfig(record_equity=True),
    )
    scaling.on_event = engine.emit

    feed = SimulatedFeed(SimConfig(symbol=symbol, bars=bars, seed=seed, start_price=1.0800))

    index = 0
    for index, candle in enumerate(feed):
        if job.stop_requested:
            job.status = "stopped"
            break
        engine.on_candle(candle)
        job.progress = (index + 1) / bars
        job.result = live_snapshot(engine, broker, scaling, index, bars)
        if delay:
            time.sleep(delay)

    finished_on = bars if job.status == "running" else index
    if job.status == "running":
        job.status = "done"
    # report the bar we actually finished on, not the requested total
    job.result = live_snapshot(engine, broker, scaling, finished_on, bars)


def live_snapshot(engine, broker, scaling, index: int, total: int) -> Dict[str, Any]:
    """Compact live state for the dashboard."""
    curve = engine.equity_curve
    # downsample the curve so the payload stays small on long runs
    step = max(len(curve) // 400, 1)
    return {
        "bar": index,
        "bars": total,
        "equity": broker.equity(),
        "cash": broker.cash(),
        "unrealized": broker.unrealized_pnl(),
        "positions": [
            {
                "symbol": p.symbol,
                "side": p.side.value,
                "qty": p.qty,
                "avg_price": p.avg_price,
                "stop_loss": p.stop_loss,
                "adds": p.adds,
                "tranches": len(p.tranches),
                "unrealized": p.unrealized(
                    broker.mid(p.symbol) or p.avg_price, broker.spec(p.symbol), broker.config.account_ccy
                ),
            }
            for p in broker.positions().values()
        ],
        "equity_curve": [
            {"time": when.isoformat(), "equity": equity}
            for when, equity in curve[::step] + ([] if step == 1 else curve[-1:])
        ],
        "trades": [
            {
                "symbol": t.symbol,
                "side": t.side.value,
                "qty": t.qty,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "exit_time": t.exit_time.isoformat(),
                "net_pnl": t.net_pnl,
                "exit_reason": t.exit_reason,
                "max_r": t.max_r,
            }
            for t in broker.closed_trades()[-40:]
        ],
        "scaling_stats": dict(scaling.stats),
        "events": [
            {k: v for k, v in e.items() if isinstance(v, (str, int, float, bool, type(None)))}
            for e in engine.events[-60:]
        ],
    }


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "BotDashboard/1.0"
    jobs: JobManager = None  # set by serve()

    # ------------------------------------------------------------- utilities --
    def _send_json(self, payload: Any, status: int = 200) -> None:
        body = json.dumps(payload, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _send_file(self, path: Path, content_type: str) -> None:
        try:
            body = path.read_bytes()
        except OSError:
            self._send_json({"error": "not found"}, status=404)
            return
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> Dict[str, Any]:
        length = int(self.headers.get("Content-Length") or 0)
        if length <= 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw.decode() or "{}")
        except json.JSONDecodeError:
            return {}

    def log_message(self, fmt, *args):  # pragma: no cover - quieter logs
        pass

    # ---------------------------------------------------------------- routes --
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path

        if path in ("/", "/index.html"):
            return self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if path == "/style.css":
            return self._send_file(STATIC_DIR / "style.css", "text/css; charset=utf-8")
        if path == "/app.js":
            return self._send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")

        if path == "/api/health":
            return self._send_json({"ok": True, "version": _version()})

        if path == "/api/strategies":
            return self._send_json(strategies_payload())

        if path == "/api/config":
            return self._send_json(default_config_payload())

        if path.startswith("/api/jobs/"):
            job_id = path.rsplit("/", 1)[-1]
            job = self.jobs.get(job_id)
            if job is None:
                return self._send_json({"error": "unknown job"}, status=404)
            return self._send_json(job.snapshot())

        if path == "/api/jobs":
            return self._send_json([j.snapshot() for j in self.jobs.all()])

        return self._send_json({"error": f"no route for {path}"}, status=404)

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        path = parsed.path
        payload = self._read_body()

        if path == "/api/run":
            job_id = self.jobs.submit("backtest", lambda job: run_backtest_job(job, payload))
            return self._send_json({"job_id": job_id}, status=202)

        if path == "/api/paper":
            job_id = self.jobs.submit("paper", lambda job: run_paper_job(job, payload))
            return self._send_json({"job_id": job_id}, status=202)

        if path.startswith("/api/jobs/") and path.endswith("/stop"):
            job_id = path.split("/")[3]
            if self.jobs.stop(job_id):
                return self._send_json({"stopped": job_id})
            return self._send_json({"error": "unknown job"}, status=404)

        return self._send_json({"error": f"no route for {path}"}, status=404)


def _version() -> str:
    from bot import __version__

    return __version__


def strategies_payload() -> Dict[str, Any]:
    from bot.strategies import available, build

    strategies = {}
    for name in available():
        strategy = build(name)
        strategies[name] = {
            "name": name,
            "warmup": strategy.warmup,
            "params": dict(strategy.params),
        }
    return {"strategies": strategies}


def default_config_payload() -> Dict[str, Any]:
    from bot.config import BotConfig

    config = BotConfig()
    return {
        "broker": asdict(config.broker),
        "risk": asdict(config.risk),
        "scaling": config.scaling.to_dict(),
        "engine": asdict(config.engine),
        "strategies": strategies_payload()["strategies"],
    }


def serve(host: str = "0.0.0.0", port: int = 8000) -> ThreadingHTTPServer:
    """Start the dashboard server (blocking)."""
    handler = type("BoundDashboardHandler", (DashboardHandler,), {"jobs": JobManager()})
    httpd = ThreadingHTTPServer((host, port), handler)
    return httpd


def main(argv: Optional[List[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="FX bot web dashboard")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)

    httpd = serve(args.host, args.port)
    host, port = httpd.server_address[:2]
    print(f"Bot dashboard running on http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        httpd.server_close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
