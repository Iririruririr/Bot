"""Tests for the web dashboard: job manager, JSON API, and the HTTP layer.

Every test spins up a real :class:`ThreadingHTTPServer` on an ephemeral port and
talks to it with ``urllib``, so these are end-to-end tests rather than unit tests
of helper functions.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest

from bot.web import server as web
from bot.web.server import Job, JobManager, run_backtest_job, run_paper_job


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
        with urllib.request.urlopen(self.base + path, timeout=30) as response:
            return response.status, json.loads(response.read().decode())

    def get_raw(self, path):
        with urllib.request.urlopen(self.base + path, timeout=30) as response:
            return response.status, response.headers, response.read()

    def post(self, path, body):
        request = urllib.request.Request(
            self.base + path,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read().decode())

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture()
def server():
    live = LiveServer()
    yield live
    live.close()


def backtest_payload(**overrides):
    """A small but realistic backtest request."""
    payload = {
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
    payload.update(overrides)
    return payload


def wait_for(job_id, server, statuses=("done",), timeout=60.0):
    deadline = time.time() + timeout
    job = None
    while time.time() < deadline:
        _, job = server.get(f"/api/jobs/{job_id}")
        if job["status"] in statuses:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} never left {job['status'] if job else 'unknown'}")


# --------------------------------------------------------------------------- #
# JobManager
# --------------------------------------------------------------------------- #


def test_job_manager_runs_a_job_and_marks_it_done():
    manager = JobManager()
    job_id = manager.submit("backtest", lambda job: job.__setattr__("progress", 1.0))

    deadline = time.time() + 10
    while time.time() < deadline and manager.get(job_id).status == "running":
        time.sleep(0.02)

    job = manager.get(job_id)
    assert job.status == "done"
    assert job.progress == 1.0
    assert job.error is None
    assert job.id in [j.id for j in manager.all()]


def test_job_manager_captures_exceptions_as_errors():
    def boom(job):
        raise RuntimeError("kaboom")

    manager = JobManager()
    job_id = manager.submit("backtest", boom)

    deadline = time.time() + 10
    while time.time() < deadline and manager.get(job_id).status == "running":
        time.sleep(0.02)

    job = manager.get(job_id)
    assert job.status == "error"
    assert "kaboom" in job.error
    assert "RuntimeError" in job.error


def test_job_manager_stop_is_idempotent_and_ignores_unknown_ids():
    manager = JobManager()
    job_id = manager.submit("backtest", lambda job: time.sleep(0.05))
    assert manager.stop(job_id) is True
    assert manager.get(job_id).stop_requested is True
    assert manager.stop("nope") is False


def test_job_snapshot_round_trips_over_json():
    job = Job(id="abc123", kind="paper")
    job.progress = 0.5
    snapshot = json.loads(json.dumps(job.snapshot()))
    assert snapshot["id"] == "abc123"
    assert snapshot["kind"] == "paper"
    assert snapshot["status"] == "running"
    assert snapshot["progress"] == 0.5
    assert snapshot["result"] is None


# --------------------------------------------------------------------------- #
# static assets + read-only API
# --------------------------------------------------------------------------- #


def test_index_is_served_as_html(server):
    status, headers, body = server.get_raw("/")
    assert status == 200
    assert headers["Content-Type"].startswith("text/html")
    assert b"FX Bot" in body
    # the page must reference the assets it needs
    assert b"/style.css" in body
    assert b"/app.js" in body


def test_assets_are_served(server):
    for path, expected in (("/style.css", "text/css"), ("/app.js", "application/javascript")):
        status, headers, body = server.get_raw(path)
        assert status == 200, path
        assert headers["Content-Type"].startswith(expected), path
        assert body.strip(), f"{path} is empty"


def test_app_js_uses_relative_urls(server):
    """The dashboard is proxied under a preview host, so no absolute origins."""
    _, _, body = server.get_raw("/app.js")
    text = body.decode()
    assert "fetch(\"/api/" in text
    assert "http://localhost" not in text
    assert "127.0.0.1" not in text


def test_health(server):
    status, payload = server.get("/api/health")
    assert status == 200
    assert payload["ok"] is True
    assert payload["version"]


def test_strategies_lists_every_registered_strategy(server):
    _, payload = server.get("/api/strategies")
    names = set(payload["strategies"])
    assert names == {"ma_crossover", "rsi_reversion", "breakout"}
    for name, info in payload["strategies"].items():
        assert info["name"] == name
        assert info["warmup"] > 0
        assert isinstance(info["params"], dict)


def test_config_payload_matches_bot_config(server):
    _, payload = server.get("/api/config")
    assert payload["broker"]["starting_cash"] > 0
    assert payload["risk"]["risk_per_trade_pct"] > 0
    assert payload["engine"]["record_equity"] is True
    assert set(payload["scaling"]) >= {"mode", "tranches", "add_step_r"}
    assert set(payload["strategies"]) == {"ma_crossover", "rsi_reversion", "breakout"}


def test_unknown_get_route_is_a_json_404(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        server.get("/api/nope")
    assert excinfo.value.code == 404
    body = json.loads(excinfo.value.read().decode())
    assert "error" in body


def test_unknown_post_route_is_a_json_404(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        server.post("/api/nope", {})
    assert excinfo.value.code == 404


def test_unknown_job_is_a_json_404(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        server.get("/api/jobs/does-not-exist")
    assert excinfo.value.code == 404


def test_stopping_an_unknown_job_is_a_json_404(server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        server.post("/api/jobs/does-not-exist/stop", {})
    assert excinfo.value.code == 404


# --------------------------------------------------------------------------- #
# POST /api/run
# --------------------------------------------------------------------------- #


def test_run_returns_a_job_id(server):
    status, payload = server.post("/api/run", backtest_payload(bars=120))
    assert status == 202
    assert payload["job_id"]
    job = wait_for(payload["job_id"], server)
    assert job["status"] == "done"


def test_run_produces_a_full_report(server):
    _, started = server.post("/api/run", backtest_payload())
    job = wait_for(started["job_id"], server)
    result = job["result"]

    assert result["meta"]["strategy"] == "ma_crossover"
    assert result["meta"]["bars"] == 300
    assert result["meta"]["scaling"]["mode"] == "pyramid"

    stats = result["stats"]
    for key in (
        "net_pnl", "total_return_pct", "final_equity", "max_drawdown_pct",
        "sharpe", "sortino", "profit_factor", "trades", "wins", "losses",
        "win_rate", "expectancy", "expectancy_r", "exposure_pct",
    ):
        assert key in stats, key
    assert stats["trades"] == len(result["trades"])
    assert stats["wins"] + stats["losses"] <= stats["trades"]
    # one equity point per bar
    assert len(result["equity_curve"]) == 300

    trade = result["trades"][0]
    for key in ("symbol", "side", "qty", "entry_price", "exit_price",
                "entry_time", "exit_time", "pnl", "commission", "net_pnl",
                "exit_reason", "max_r"):
        assert key in trade, key
    assert trade["side"] in ("BUY", "SELL")


def test_run_scaling_bot_reports_its_activity(server):
    _, started = server.post("/api/run", backtest_payload())
    job = wait_for(started["job_id"], server)
    scaling = job["result"]["scaling_stats"]

    for key in ("adds", "scale_outs", "breakevens", "trail_updates", "skipped_adds"):
        assert key in scaling, key
    assert scaling["adds"] > 0
    assert scaling["scale_outs"] > 0


def test_run_scaling_off_still_works(server):
    """The trading bot on its own must be runnable from the dashboard."""
    payload = backtest_payload()
    payload["scaling"] = {"mode": "off", "tranches": 1, "scale_out": []}
    _, started = server.post("/api/run", payload)
    job = wait_for(started["job_id"], server)

    result = job["result"]
    assert result["meta"]["scaling"]["mode"] == "off"
    assert result["scaling_stats"]["adds"] == 0
    assert result["scaling_stats"]["scale_outs"] == 0


def test_run_is_deterministic_for_a_fixed_seed(server):
    _, first = server.post("/api/run", backtest_payload(bars=200))
    _, second = server.post("/api/run", backtest_payload(bars=200))

    a = wait_for(first["job_id"], server)["result"]
    b = wait_for(second["job_id"], server)["result"]

    assert a["stats"]["net_pnl"] == b["stats"]["net_pnl"]
    assert a["stats"]["trades"] == b["stats"]["trades"]
    assert a["equity_curve"] == b["equity_curve"]


def test_run_with_an_unknown_strategy_reports_an_error(server):
    _, started = server.post("/api/run", backtest_payload(strategy="does_not_exist"))
    job = wait_for(started["job_id"], server, statuses=("error",))
    assert job["status"] == "error"
    assert job["error"]
    assert job["result"] is None


def test_run_ignores_unknown_config_keys(server):
    """Extra keys from a future UI must not break the dataclass constructors."""
    _, started = server.post(
        "/api/run",
        backtest_payload(bars=120, risk={"risk_per_trade_pct": 1.0, "bogus": 42},
                         broker={"starting_cash": 50_000.0, "bogus": "x"}),
    )
    job = wait_for(started["job_id"], server)
    assert job["status"] == "done"
    assert job["result"]["stats"]["starting_equity"] == 50_000.0


def test_jobs_endpoint_lists_every_job(server):
    _, started = server.post("/api/run", backtest_payload(bars=120))
    wait_for(started["job_id"], server)

    status, jobs = server.get("/api/jobs")
    assert status == 200
    assert isinstance(jobs, list)
    assert any(job["id"] == started["job_id"] for job in jobs)


# --------------------------------------------------------------------------- #
# POST /api/paper
# --------------------------------------------------------------------------- #


def test_paper_reports_incremental_progress(server):
    _, started = server.post(
        "/api/paper",
        backtest_payload(bars=400, delay_ms=0, scaling={"mode": "pyramid", "tranches": 3}),
    )

    # poll until the job has moved past the warmup
    seen_bar = 0
    deadline = time.time() + 60
    while time.time() < deadline:
        _, job = server.get(f"/api/jobs/{started['job_id']}")
        if job["result"]:
            seen_bar = max(seen_bar, job["result"]["bar"])
            if seen_bar > 60:
                break
        time.sleep(0.02)

    _, job = server.get(f"/api/jobs/{started['job_id']}")
    assert job["result"]["bar"] >= 60
    assert job["result"]["bars"] == 400
    assert 0.0 < job["progress"] < 1.0

    final = wait_for(started["job_id"], server)
    assert final["result"]["bar"] == 400
    assert final["progress"] == 1.0


def test_paper_snapshot_shape(server):
    _, started = server.post(
        "/api/paper",
        backtest_payload(bars=250, delay_ms=0, scaling={"mode": "pyramid", "tranches": 3}),
    )
    job = wait_for(started["job_id"], server)
    snap = job["result"]

    for key in ("bar", "bars", "equity", "cash", "unrealized",
                "positions", "equity_curve", "trades", "scaling_stats", "events"):
        assert key in snap, key
    assert isinstance(snap["positions"], list)
    assert isinstance(snap["events"], list)
    assert len(snap["equity_curve"]) >= 1

    for point in snap["equity_curve"]:
        assert set(point) == {"time", "equity"}
        assert point["equity"] > 0

    # every event is JSON-serialisable and carries a kind
    for event in snap["events"]:
        assert "kind" in event
        json.dumps(event)


def test_paper_exposes_live_position_state(server):
    _, started = server.post(
        "/api/paper",
        backtest_payload(bars=400, delay_ms=0, scaling={"mode": "pyramid", "tranches": 3}),
    )
    saw_position = False
    deadline = time.time() + 60
    while time.time() < deadline:
        _, job = server.get(f"/api/jobs/{started['job_id']}")
        positions = (job["result"] or {}).get("positions") or []
        if positions:
            position = positions[0]
            for key in ("symbol", "side", "qty", "avg_price", "stop_loss",
                        "adds", "tranches", "unrealized"):
                assert key in position, key
            assert position["tranches"] >= 1
            saw_position = True
            break
        time.sleep(0.02)

    wait_for(started["job_id"], server)
    assert saw_position, "expected at least one open position during the replay"


def test_paper_can_be_stopped_early(server):
    _, started = server.post("/api/paper", backtest_payload(bars=5000, delay_ms=5))

    # wait for the replay to get going, then stop it
    deadline = time.time() + 60
    while time.time() < deadline:
        _, job = server.get(f"/api/jobs/{started['job_id']}")
        if (job["result"] or {}).get("bar", 0) > 20:
            break
        time.sleep(0.02)

    status, payload = server.post(f"/api/jobs/{started['job_id']}/stop", {})
    assert status == 200
    assert payload["stopped"] == started["job_id"]

    job = wait_for(started["job_id"], server, statuses=("stopped",), timeout=20)
    assert job["result"]["bar"] < 5000
    assert job["result"]["bars"] == 5000


def test_paper_matches_a_plain_backtest_on_the_same_seed(server):
    """The live replay is the same engine - it must agree with a backtest."""
    _, paper_id = server.post(
        "/api/paper", backtest_payload(bars=300, delay_ms=0, scaling={"mode": "off"})
    )
    paper = wait_for(paper_id["job_id"], server)["result"]

    _, back_id = server.post("/api/run", backtest_payload(bars=300, scaling={"mode": "off"}))
    back = wait_for(back_id["job_id"], server)["result"]

    assert paper["equity"] == pytest.approx(back["stats"]["final_equity"], rel=1e-9)
    assert len(paper["trades"]) == back["stats"]["trades"]


# --------------------------------------------------------------------------- #
# direct exercise of the job bodies (no HTTP)
# --------------------------------------------------------------------------- #


def test_run_backtest_job_populates_the_job():
    job = Job(id="direct", kind="backtest")
    run_backtest_job(job, backtest_payload(bars=150))

    assert job.status == "done"
    assert job.progress == 1.0
    assert job.result["stats"]["trades"] >= 0


def test_run_paper_job_respects_stop_requested():
    job = Job(id="direct-paper", kind="paper")
    payload = backtest_payload(bars=500, delay_ms=0, scaling={"mode": "off"})

    # stop after ~30 bars by hooking the snapshot
    original = web.live_snapshot
    calls = {"n": 0}

    def counting(engine, broker, scaling, index, total):
        calls["n"] += 1
        if calls["n"] > 30:
            job.stop_requested = True
        return original(engine, broker, scaling, index, total)

    web.live_snapshot = counting
    try:
        run_paper_job(job, payload)
    finally:
        web.live_snapshot = original

    assert job.status == "stopped"
    assert job.result["bar"] < 500


def test_run_paper_job_does_not_leak_warmup_into_a_backtest():
    """The live view shortens warmup so something happens quickly; the shared
    strategy classes must not be left in that state for other runs."""
    job = Job(id="warmup", kind="paper")
    run_paper_job(job, backtest_payload(bars=120, delay_ms=0))

    from bot.strategies import build

    fresh = build("ma_crossover")
    assert fresh.warmup == 60
    run_backtest_job(Job(id="bt", kind="backtest"), backtest_payload(bars=120))
    assert build("ma_crossover").warmup == 60


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
