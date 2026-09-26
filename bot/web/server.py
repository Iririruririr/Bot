"""Web dashboard for the FX trading bot and scaling bot.

A stdlib-only HTTP server (``http.server``) that serves a single-page dashboard
plus a small JSON API, so running the site adds no dependencies.

    python -m bot web --port 8000

Everything is **stateless**: each request runs its work to completion and
returns the whole answer in one response.  That is deliberate - it is what lets
the same code run unchanged on a platform that invokes a function per request
(Vercel, Lambda, Cloud Functions) with no long-lived process and no shared
memory.

The "watch it live" experience is achieved by having the server compute a full
bar-by-bar trace and the *browser* play it back, so the pacing - and the
ability to pause, scrub and change speed - lives on the client where it belongs.
"""
from __future__ import annotations

import json
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

STATIC_DIR = Path(__file__).parent / "static"

# The dashboard sends one payload per watched bar, so a long replay is a biggish
# JSON document.  Cap it to keep responses sane; the UI picks the length.
MAX_REPLAY_BARS = 2500


# --------------------------------------------------------------------------- #
# payload plumbing
# --------------------------------------------------------------------------- #


def _pick(data: Dict[str, Any], cls: type) -> Dict[str, Any]:
    """Keep only the keys a dataclass actually accepts.

    Lets the UI send extra/renamed fields without breaking a config object.
    """
    return {k: v for k, v in (data or {}).items() if k in cls.__dataclass_fields__}


def _json_safe(event: Dict[str, Any]) -> Dict[str, Any]:
    """Drop anything that is not a plain JSON scalar."""
    return {k: v for k, v in event.items()
            if isinstance(v, (str, int, float, bool, type(None)))}


# --------------------------------------------------------------------------- #
# the work itself
# --------------------------------------------------------------------------- #


def _build_stack(payload: Dict[str, Any]):
    """Assemble broker/risk/scaling/strategy/engine/feed from a request."""
    from bot.brokers.paper import BrokerConfig, PaperBroker
    from bot.core.engine import Engine, EngineConfig
    from bot.core.risk import RiskConfig, RiskManager
    from bot.data.feed import SimConfig, SimulatedFeed
    from bot.scaling.engine import ScalingConfig, ScalingEngine
    from bot.strategies import build

    bars = int(payload.get("bars", 2000))
    bars = max(1, min(bars, MAX_REPLAY_BARS))
    seed = int(payload.get("seed", 7))
    symbol = payload.get("symbol", "EUR/USD")
    start_price = float(payload.get("start_price", 1.0800))

    broker_cfg = BrokerConfig(**_pick(payload.get("broker"), BrokerConfig))
    risk_cfg = RiskConfig(**_pick(payload.get("risk"), RiskConfig))

    broker = PaperBroker(broker_cfg)
    risk = RiskManager(risk_cfg, broker_cfg.starting_cash)
    scaling = ScalingEngine(ScalingConfig.from_dict(payload.get("scaling") or {}),
                            broker, account_ccy=broker_cfg.account_ccy, risk=risk)

    strategy = build(payload.get("strategy", "ma_crossover"),
                     **(payload.get("strategy_params") or {}))
    engine = Engine(broker=broker, strategies=[strategy], risk=risk, scaling=scaling,
                    config=EngineConfig(record_equity=True))
    # the scaling bot drives the engine's event log so the dashboard sees adds
    scaling.on_event = engine.emit

    feed = SimulatedFeed(SimConfig(symbol=symbol, bars=bars, seed=seed,
                                   start_price=start_price,
                                   vol_pips=float(payload.get("vol_pips", 12.0))))
    return broker, risk, scaling, strategy, engine, feed, bars


def run_backtest(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run one backtest and return the whole report in a single response."""
    from bot.backtest.runner import run_backtest as _run

    broker, risk, scaling, strategy, engine, feed, bars = _build_stack(payload)
    result = _run(feed=feed, strategy=strategy,
                  scaling=scaling.config, risk=risk.config,
                  broker=broker.config)
    return result_payload(result)


def run_replay(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Run a paper session to completion and return a bar-by-bar trace.

    The trace is deliberately a *list of frames* rather than a list of full
    snapshots: trades and events are sent once, in order, and each frame only
    records how many of them had happened by that bar.  The browser slices them
    back together as it plays.
    """
    broker, risk, scaling, strategy, engine, feed, bars = _build_stack(payload)

    # a short warmup means the first watched bars already show activity
    strategy.warmup = min(strategy.warmup, 40)

    frames: List[Dict[str, Any]] = []
    index = 0
    for index, candle in enumerate(feed):
        engine.on_candle(candle)
        frames.append(_frame(engine, broker, scaling, index, bars))

    return {
        "meta": {
            "strategy": strategy.name,
            "bars": bars,
            "seed": int(payload.get("seed", 7)),
            "scaling": scaling.config.to_dict(),
            "starting_equity": broker.config.starting_cash,
        },
        "frames": frames,
        "trades": [_trade_payload(t) for t in broker.closed_trades()],
        "events": [_json_safe(e) for e in engine.events],
        "scaling_stats": dict(scaling.stats),
    }


def _frame(engine, broker, scaling, index: int, total: int) -> Dict[str, Any]:
    """One bar of the replay: account state plus open-position detail."""
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
                    broker.mid(p.symbol) or p.avg_price,
                    broker.spec(p.symbol),
                    broker.config.account_ccy,
                ),
            }
            for p in broker.positions().values()
        ],
        "trades_closed": len(broker.closed_trades()),
        "events_seen": len(engine.events),
    }


# --------------------------------------------------------------------------- #
# report payloads
# --------------------------------------------------------------------------- #


def _trade_payload(t) -> Dict[str, Any]:
    return {
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
        "trades": [_trade_payload(t) for t in result.trades],
        "scaling_stats": result.scaling_stats,
        "events": [_json_safe(e) for e in result.events[-400:]],
    }


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


# --------------------------------------------------------------------------- #
# HTTP layer
# --------------------------------------------------------------------------- #


class DashboardHandler(BaseHTTPRequestHandler):
    server_version = "BotDashboard/2.0"

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
            return self._send_json({"error": "not found"}, status=404)
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

    # ---------------------------------------------------------------- routing --
    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path in ("/", "/index.html"):
            return self._send_file(STATIC_DIR / "index.html", "text/html; charset=utf-8")
        if path == "/style.css":
            return self._send_file(STATIC_DIR / "style.css", "text/css; charset=utf-8")
        if path == "/app.js":
            return self._send_file(STATIC_DIR / "app.js", "application/javascript; charset=utf-8")

        try:
            status, payload = self._route_get(path)
        except Exception as exc:  # pragma: no cover - defensive
            status, payload = 500, {"error": f"{type(exc).__name__}: {exc}"}
        return self._send_json(payload, status)

    def _route_get(self, path: str) -> Tuple[int, Dict[str, Any]]:
        if path == "/api/health":
            return 200, {"ok": True, "version": _version(), "stateless": True}
        if path == "/api/strategies":
            return 200, strategies_payload()
        if path == "/api/config":
            return 200, default_config_payload()
        return 404, {"error": f"no route for {path}"}

    def do_POST(self) -> None:
        path = urlparse(self.path).path
        payload = self._read_body()

        try:
            status, body = self._route_post(path, payload)
        except Exception as exc:  # pragma: no cover - defensive
            status, body = 400, {"error": f"{type(exc).__name__}: {exc}"}
        return self._send_json(body, status)

    def _route_post(self, path: str, payload: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        if path == "/api/run":
            return 200, run_backtest(payload)
        if path == "/api/replay":
            return 200, run_replay(payload)
        return 404, {"error": f"no route for {path}"}


def _version() -> str:
    from bot import __version__

    return __version__


def serve(host: str = "0.0.0.0", port: int = 8000) -> ThreadingHTTPServer:
    """Bind the dashboard server (blocking once ``serve_forever`` is called)."""
    return ThreadingHTTPServer((host, port), DashboardHandler)


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
