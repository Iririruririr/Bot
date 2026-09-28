"""Tests for the web dashboard: the stateless JSON API and the HTTP layer.

Every test spins up a real :class:`ThreadingHTTPServer` on an ephemeral port and
talks to it with ``urllib``, so these are end-to-end tests rather than unit tests
of helper functions.

The API is deliberately **stateless**: each request runs its work to completion
and returns the whole answer in one response.  That is what lets the same code
run on a platform that invokes a function per request, and it is what these
tests pin down.
"""
from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request

import pytest

from bot.web import server as web
from bot.web.server import (
    MAX_REPLAY_BARS,
    _pick,
    run_backtest,
    run_replay,
    strategies_payload,
)


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #


class LiveServer:
    """A dashboard server running on 127.0.0.1 with a throwaway port."""

    def __init__(self):
        self.httpd = web.serve("127.0.0.1", 0)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=120) as response:
            return response.status, json.loads(response.read().decode())

    def get_raw(self, path):
        with urllib.request.urlopen(self.base + path, timeout=120) as response:
            return response.status, response.headers, response.read()

    def post(self, path, body):
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=300) as response:
            return response.status, json.loads(response.read().decode())

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def server():
    live = LiveServer()
    yield live
    live.close()


def payload(**overrides):
    """A small but realistic dashboard request."""
    body = {
        "strategy": "ma_crossover",
        "bars": 300,
        "seed": 7,
        "vol_pips": 12.0,
        "scaling": {
            "mode": "pyramid",
            "tranches": 3,
            "add_step_r": 1.0,
            "add_size_mult": 1.0,
            "scale_out": [[1.0, 0.5], [2.0, 0.25], [3.0, 0.25]],
            "breakeven_at_r": 1.0,
            "trail_atr_mult": 3.0,
            "trail_start_r": 1.0,
        },
        "risk": {
            "risk_per_trade_pct": 1.0,
            "max_daily_loss_pct": 3.0,
            "max_drawdown_pct": 15.0,
            "max_open_exposure_pct": 100,
        },
        "broker": {
            "starting_cash": 100_000.0,
            "spread_pips": 1.0,
            "slippage_pips": 0.2,
            "commission_per_million": 0.0,
            "lot_size": 1000.0,
        },
    }
    body.update(overrides)
    return body


SCALING_OFF = {"mode": "off", "tranches": 1, "scale_out": []}


# --------------------------------------------------------------------------- #
# _pick
# --------------------------------------------------------------------------- #


def test_pick_drops_unknown_keys_and_keeps_known_ones():
    from bot.core.risk import RiskConfig

    assert _pick({"risk_per_trade_pct": 2.0, "nonsense": 1}, RiskConfig) == {
        "risk_per_trade_pct": 2.0
    }
    assert _pick({}, RiskConfig) == {}
    assert _pick(None, RiskConfig) == {}


# --------------------------------------------------------------------------- #
# static assets + read-only API
# --------------------------------------------------------------------------- #


def test_index_is_served_as_html(server):
    status, headers, body = server.get_raw("/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"FX Bot" in body
    assert b"/style.css" in body
    assert b"/app.js" in body


def test_assets_are_served(server):
    for path, expected in (("/style.css", "text/css"), ("/app.js", "application/javascript")):
        status, headers, body = server.get_raw(path)
        assert status == 200, path
        assert headers["Content-Type"].startswith(expected), path
        assert body.strip(), f"{path} is empty"


def test_app_js_uses_relative_urls_and_polls_nothing(server):
    """The dashboard is proxied under a preview host and must stay stateless."""
    text = server.get_raw("/app.js")[2].decode()
    assert 'fetch("/api/' in text
    assert "http://localhost" not in text
    assert "127.0.0.1" not in text
    # no polling, no job ids, no /api/jobs
    assert "setInterval" not in text
    assert "/api/jobs" not in text


def test_health(server):
    status, body = server.get("/api/health")
    assert status == 200
    assert body["ok"] is True
    assert body["version"]
    assert body["stateless"] is True


def test_strategies_lists_every_registered_strategy(server):
    _, body = server.get("/api/strategies")
    assert set(body["strategies"]) == {"ma_crossover", "rsi_reversion", "breakout"}
    for name, info in body["strategies"].items():
        assert info["name"] == name
        assert info["warmup"] > 0
        assert isinstance(info["params"], dict)


def test_config_payload_matches_bot_config(server):
    _, body = server.get("/api/config")
    assert body["broker"]["starting_cash"] > 0
    assert body["risk"]["risk_per_trade_pct"] > 0
    assert body["engine"]["record_equity"] is True
    assert set(body["scaling"]) >= {"mode", "tranches", "add_step_r"}
    assert set(body["strategies"]) == {"ma_crossover", "rsi_reversion", "breakout"}


def test_unknown_get_route_is_a_json_404(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        server.get("/api/nope")
    assert excinfo.value.code == 404
    assert "error" in json.loads(excinfo.value.read().decode())


def test_unknown_post_route_is_a_json_404(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        server.post("/api/nope", {})
    assert excinfo.value.code == 404


def test_job_endpoints_are_gone(server):
    """The polling model was removed; those paths must 404, not silently work."""
    for path in ("/api/jobs", "/api/jobs/abc123"):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            server.get(path)
        assert excinfo.value.code == 404, path
    for path in ("/api/paper", "/api/jobs/abc123/stop"):
        with pytest.raises(urllib.error.HTTPError) as excinfo:
            server.post(path, {})
        assert excinfo.value.code == 404, path


# --------------------------------------------------------------------------- #
# POST /api/run
# --------------------------------------------------------------------------- #


def test_run_returns_the_whole_report_in_one_response(server):
    status, body = server.post("/api/run", payload(bars=250))

    assert status == 200
    # no job id - the answer is already complete
    assert "job_id" not in body
    assert body["meta"]["strategy"] == "ma_crossover"
    assert body["meta"]["bars"] == 250
    assert body["meta"]["scaling"]["mode"] == "pyramid"

    stats = body["stats"]
    for key in (
        "starting_equity", "net_pnl", "total_return_pct", "final_equity",
        "max_drawdown_pct", "max_drawdown_amount", "sharpe", "sortino",
        "profit_factor", "trades", "wins", "losses", "win_rate",
        "expectancy", "expectancy_r", "avg_win", "avg_loss",
        "largest_win", "largest_loss", "total_commission", "exposure_pct",
    ):
        assert key in stats, key
    assert stats["trades"] == len(body["trades"])
    assert stats["wins"] + stats["losses"] <= stats["trades"]
    assert stats["starting_equity"] == 100_000.0
    assert len(body["equity_curve"]) == 250      # one point per bar

    trade = body["trades"][0]
    for key in ("symbol", "side", "qty", "entry_price", "exit_price",
                "entry_time", "exit_time", "pnl", "commission", "net_pnl",
                "exit_reason", "max_r"):
        assert key in trade, key
    assert trade["side"] in ("BUY", "SELL")


def test_run_reports_scaling_activity(server):
    _, body = server.post("/api/run", payload())
    scaling = body["scaling_stats"]
    for key in ("adds", "scale_outs", "breakevens", "trail_updates", "skipped_adds"):
        assert key in scaling, key
    assert scaling["adds"] > 0
    assert scaling["scale_outs"] > 0
    assert len(body["events"]) > 0
    assert all("kind" in e for e in body["events"])


def test_run_scaling_off_still_works(server):
    """The plain trading bot must be runnable from the dashboard."""
    _, body = server.post("/api/run", payload(scaling=SCALING_OFF))
    assert body["meta"]["scaling"]["mode"] == "off"
    assert body["scaling_stats"]["adds"] == 0
    assert body["scaling_stats"]["scale_outs"] == 0


def test_run_is_deterministic_for_a_fixed_seed(server):
    _, first = server.post("/api/run", payload(bars=200))
    _, second = server.post("/api/run", payload(bars=200))

    assert first["stats"]["net_pnl"] == second["stats"]["net_pnl"]
    assert first["stats"]["trades"] == second["stats"]["trades"]
    assert first["equity_curve"] == second["equity_curve"]
    assert first["trades"] == second["trades"]


def test_run_with_an_unknown_strategy_is_a_json_error(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        server.post("/api/run", payload(strategy="does_not_exist"))
    assert excinfo.value.code == 400
    assert "error" in json.loads(excinfo.value.read().decode())


def test_run_with_bad_scaling_config_is_a_json_error(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        server.post("/api/run", payload(scaling={"mode": "sideways"}))
    assert excinfo.value.code == 400
    assert "error" in json.loads(excinfo.value.read().decode())


def test_run_ignores_unknown_config_keys(server):
    """Extra keys from a future UI must not break the dataclass constructors."""
    _, body = server.post(
        "/api/run",
        payload(bars=120, risk={"risk_per_trade_pct": 1.0, "bogus": 42},
                broker={"starting_cash": 50_000.0, "bogus": "x"}),
    )
    assert body["stats"]["starting_equity"] == 50_000.0


def test_run_caps_bars_at_the_replay_limit(server):
    """A huge request must not be able to pin the process for ever."""
    _, body = server.post("/api/run", payload(bars=10 * MAX_REPLAY_BARS))
    assert body["meta"]["bars"] == MAX_REPLAY_BARS


# --------------------------------------------------------------------------- #
# POST /api/replay
# --------------------------------------------------------------------------- #


def test_replay_returns_a_frame_per_bar(server):
    _, body = server.post("/api/replay", payload(bars=250))

    assert body["meta"]["bars"] == 250
    assert len(body["frames"]) == 250
    # frames are in bar order
    assert [f["bar"] for f in body["frames"]] == list(range(250))
    for frame in body["frames"]:
        for key in ("bar", "bars", "equity", "cash", "unrealized",
                    "positions", "trades_closed", "events_seen"):
            assert key in frame, key
        assert frame["equity"] > 0


def test_replay_frames_are_consistent(server):
    """A frame is only useful if its trade/event counts slice the lists back."""
    _, body = server.post("/api/replay", payload(bars=300))
    frames = body["frames"]

    for frame in frames:
        assert 0 <= frame["trades_closed"] <= len(body["trades"])
        assert 0 <= frame["events_seen"] <= len(body["events"])

    # monotonic, and the last frame has seen everything
    counts = [f["trades_closed"] for f in frames]
    assert counts == sorted(counts)
    assert counts[-1] == len(body["trades"])
    assert frames[-1]["events_seen"] == len(body["events"])


def test_replay_final_frame_matches_a_plain_backtest(server):
    """The replay is the same engine - it must agree with a backtest."""
    _, replay = server.post("/api/replay", payload(bars=300, scaling=SCALING_OFF))
    _, back = server.post("/api/run", payload(bars=300, scaling=SCALING_OFF))

    assert replay["frames"][-1]["equity"] == pytest.approx(
        back["stats"]["final_equity"], rel=1e-9)
    assert len(replay["trades"]) == back["stats"]["trades"]
    assert replay["scaling_stats"]["adds"] == 0


def test_replay_shows_scaling_activity(server):
    _, body = server.post("/api/replay", payload(bars=400))
    stats = body["scaling_stats"]
    assert stats["adds"] > 0
    assert stats["scale_outs"] > 0

    kinds = {e["kind"] for e in body["events"]}
    assert "scale_in" in kinds
    assert "scale_out" in kinds
    assert "entry" in kinds


def test_replay_positions_carry_the_tranche_count(server):
    """The whole point of the view: you can see the scaling bot adding."""
    _, body = server.post("/api/replay", payload(bars=400))
    saw_position = False
    saw_multiple_tranches = False

    for frame in body["frames"]:
        for position in frame["positions"]:
            saw_position = True
            for key in ("symbol", "side", "qty", "avg_price", "stop_loss",
                        "adds", "tranches", "unrealized"):
                assert key in position, key
            if position["tranches"] > 1:
                saw_multiple_tranches = True

    assert saw_position
    assert saw_multiple_tranches, "expected the scaling bot to add at least once"


def test_replay_is_deterministic(server):
    _, first = server.post("/api/replay", payload(bars=200))
    _, second = server.post("/api/replay", payload(bars=200))
    assert first["frames"] == second["frames"]
    assert first["trades"] == second["trades"]


def test_replay_events_are_json_safe(server):
    """Events must survive a json.dumps round trip with no custom encoder."""
    _, body = server.post("/api/replay", payload(bars=300))
    assert body["events"]
    assert json.loads(json.dumps(body)) == body


def test_replay_clamps_warmup_so_something_happens(monkeypatch):
    """The live view shortens warmup so the first watched bars already show activity.

    A stub strategy whose warmup is longer than the whole replay would never
    signal at all if the clamp were missing - so any position proves it.
    """
    import bot.strategies as strategies

    real_build = strategies.build

    def stub(name, **params):
        strategy = real_build(name, **params)
        strategy.warmup = 10_000          # far longer than the replay below
        return strategy

    monkeypatch.setattr(strategies, "build", stub)
    result = run_replay(payload(bars=200, scaling=SCALING_OFF))

    assert any(frame["positions"] for frame in result["frames"]) or result["trades"]
    # the shared strategy class must not be left mutated for anyone else
    assert real_build("ma_crossover").warmup == 60


def test_replay_with_an_unknown_strategy_is_a_json_error(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        server.post("/api/replay", payload(strategy="does_not_exist"))
    assert excinfo.value.code == 400


# --------------------------------------------------------------------------- #
# direct exercise of the job bodies (no HTTP)
# --------------------------------------------------------------------------- #


def test_run_backtest_returns_a_payload():
    result = run_backtest(payload(bars=150))
    assert result["stats"]["trades"] >= 0
    assert len(result["equity_curve"]) == 150


def test_run_replay_returns_frames_and_lists():
    result = run_replay(payload(bars=150))
    assert len(result["frames"]) == 150
    assert isinstance(result["trades"], list)
    assert isinstance(result["events"], list)
    assert result["meta"]["bars"] == 150


def test_frame_shape_from_a_real_run():
    result = run_replay(payload(bars=60))
    first = result["frames"][0]
    assert first["bar"] == 0
    assert first["bars"] == 60
    assert first["trades_closed"] == 0
    assert first["events_seen"] == 0
    # the last frame has seen everything the run produced
    last = result["frames"][-1]
    assert last["bar"] == 59
    assert last["trades_closed"] == len(result["trades"])
    assert last["events_seen"] == len(result["events"])


# --------------------------------------------------------------------------- #
# server plumbing
# --------------------------------------------------------------------------- #


def test_serve_binds_to_every_interface():
    httpd = web.serve("0.0.0.0", 0)
    try:
        assert httpd.server_address[0] == "0.0.0.0"
        assert httpd.server_address[1] > 0
    finally:
        httpd.server_close()


def test_serve_is_threaded():
    """Concurrent requests must not serialise behind one long replay."""
    httpd = web.serve("127.0.0.1", 0)
    try:
        assert isinstance(httpd, __import__("http.server", fromlist=["ThreadingHTTPServer"])
                          .ThreadingHTTPServer)
    finally:
        httpd.server_close()


def test_strategies_payload_direct():
    body = strategies_payload()
    assert set(body["strategies"]) == {"ma_crossover", "rsi_reversion", "breakout"}


# --------------------------------------------------------------------------- #
# PORT handling (process hosts inject it)
# --------------------------------------------------------------------------- #


def test_default_port_reads_the_environment(monkeypatch):
    """Railway/Render/Heroku/Fly all inject $PORT - it must win over 8000."""
    monkeypatch.setenv("PORT", "9773")
    assert web._default_port() == 9773


def test_default_port_ignores_a_missing_or_blank_port(monkeypatch):
    monkeypatch.delenv("PORT", raising=False)
    assert web._default_port() == 8000
    monkeypatch.setenv("PORT", "   ")
    assert web._default_port() == 8000


@pytest.mark.parametrize("bad", ["not-a-port", "0", "-1", "70000", "8o00"])
def test_default_port_falls_back_on_garbage(monkeypatch, bad):
    monkeypatch.setenv("PORT", bad)
    assert web._default_port() == 8000


def test_explicit_port_beats_the_environment(monkeypatch):
    """An explicit --port is the strongest signal of the three."""
    monkeypatch.setenv("PORT", "9773")
    started = {}

    class FakeServer:
        def __init__(self, host, port):
            started["port"] = port
            self.server_address = (host, port)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr(web, "serve", lambda host, port: FakeServer(host, port))
    assert web.main(["--host", "127.0.0.1", "--port", "9774"]) == 0
    assert started["port"] == 9774


def test_environment_port_is_used_when_no_flag_is_given(monkeypatch):
    monkeypatch.setenv("PORT", "9775")
    started = {}

    class FakeServer:
        def __init__(self, host, port):
            started["port"] = port
            self.server_address = (host, port)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr(web, "serve", lambda host, port: FakeServer(host, port))
    assert web.main(["--host", "127.0.0.1"]) == 0
    assert started["port"] == 9775


def test_host_falls_back_to_the_environment(monkeypatch):
    monkeypatch.setenv("HOST", "127.0.0.1")
    started = {}

    class FakeServer:
        def __init__(self, host, port):
            started["host"] = host
            self.server_address = (host, port)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            pass

    monkeypatch.setattr(web, "serve", lambda host, port: FakeServer(host, port))
    assert web.main([]) == 0
    assert started["host"] == "127.0.0.1"


def test_main_starts_and_stops(monkeypatch):
    started = {}
    stopped = []

    class FakeServer:
        def __init__(self, host, port):
            started["host"], started["port"] = host, port
            self.server_address = (host, port)

        def serve_forever(self):
            raise KeyboardInterrupt

        def server_close(self):
            stopped.append(True)

    monkeypatch.setattr(web, "serve", lambda host, port: FakeServer(host, port))
    assert web.main(["--host", "0.0.0.0", "--port", "8123"]) == 0
    assert started == {"host": "0.0.0.0", "port": 8123}
    assert stopped == [True]
