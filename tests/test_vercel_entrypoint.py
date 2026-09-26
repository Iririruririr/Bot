"""Tests for the Vercel entrypoint.

Vercel's Python runtime finds a function by looking for a **top-level class
named ``handler``** in each ``api/*.py`` file, using an AST-level check.  A
refactor that turns that class into an assignment (``handler = something``)
silently produces zero functions and fails the build with:

    The pattern "api/index.py" defined in `functions` doesn't match any
    Serverless Functions inside the api directory.

These tests pin the contract down so that regression cannot happen quietly, and
then drive the entrypoint over a real socket to prove it actually serves.
"""
from __future__ import annotations

import ast
import importlib.util
import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ENTRYPOINT = REPO_ROOT / "api" / "index.py"


# --------------------------------------------------------------------------- #
# the entrypoint contract Vercel's AST analyser enforces
# --------------------------------------------------------------------------- #


def test_entrypoint_exists():
    assert ENTRYPOINT.is_file(), f"missing {ENTRYPOINT}"


def test_entrypoint_defines_a_top_level_class_named_handler():
    """Vercel only recognises a class *definition*, never an assignment."""
    tree = ast.parse(ENTRYPOINT.read_text())

    classes = [
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "handler"
    ]
    assert len(classes) == 1, (
        "api/index.py must define exactly one top-level class named 'handler'"
    )

    # and it must not also be shadowed by an assignment of the same name
    assignments = [
        node for node in tree.body
        if isinstance(node, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "handler" for t in node.targets)
    ]
    assert not assignments, "'handler' must be a class definition, not an assignment"


def test_handler_inherits_from_base_http_request_handler():
    """Vercel requires the handler to be a BaseHTTPRequestHandler subclass."""
    spec = importlib.util.spec_from_file_location("vercel_entrypoint", ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    handler = module.handler
    assert isinstance(handler, type)
    assert issubclass(handler, BaseHTTPRequestHandler)


# --------------------------------------------------------------------------- #
# the entrypoint actually serving
# --------------------------------------------------------------------------- #


def _load_entrypoint():
    spec = importlib.util.spec_from_file_location("vercel_entrypoint_serving", ENTRYPOINT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def vercel_server():
    """Run the Vercel entrypoint on a throwaway port, as Vercel would."""
    module = _load_entrypoint()
    httpd = HTTPServer(("127.0.0.1", 0), module.handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    httpd.shutdown()
    httpd.server_close()


def test_entrypoint_serves_the_api(vercel_server):
    with urllib.request.urlopen(vercel_server + "/api/health", timeout=60) as response:
        body = json.loads(response.read().decode())
    assert response.status == 200
    assert body["ok"] is True


def test_entrypoint_serves_a_backtest(vercel_server):
    request = urllib.request.Request(
        vercel_server + "/api/run",
        data=json.dumps({
            "strategy": "ma_crossover",
            "bars": 200,
            "scaling": {"mode": "off", "tranches": 1, "scale_out": []},
        }).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.loads(response.read().decode())
    assert response.status == 200
    assert body["stats"]["trades"] >= 0
    assert len(body["equity_curve"]) == 200


def test_entrypoint_serves_static_assets(vercel_server):
    for path, expected in (("/", "text/html"), ("/style.css", "text/css"),
                           ("/app.js", "application/javascript")):
        with urllib.request.urlopen(vercel_server + path, timeout=60) as response:
            assert response.status == 200, path
            assert response.headers["Content-Type"].startswith(expected), path
            assert response.read().strip(), f"{path} is empty"


def test_entrypoint_404s_unknown_routes(vercel_server):
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        urllib.request.urlopen(vercel_server + "/api/nope", timeout=60)
    assert excinfo.value.code == 404


def test_entrypoint_is_the_same_code_as_the_local_server():
    """The deployed function must not be a second implementation."""
    module = _load_entrypoint()
    from bot.web import server

    assert module.handler is not server.DashboardHandler      # it is a subclass
    assert issubclass(module.handler, server.DashboardHandler)
    assert module.handler.do_GET is server.DashboardHandler.do_GET
    assert module.handler.do_POST is server.DashboardHandler.do_POST


# --------------------------------------------------------------------------- #
# vercel.json
# --------------------------------------------------------------------------- #


def test_vercel_json_is_valid_and_routes_the_api():
    config = json.loads((REPO_ROOT / "vercel.json").read_text())

    assert config["outputDirectory"] == "bot/web/static"

    # the static assets must exist where outputDirectory points
    for asset in ("index.html", "style.css", "app.js"):
        assert (REPO_ROOT / config["outputDirectory"] / asset).is_file(), asset

    rewrites = config.get("rewrites", [])
    api_routes = [r for r in rewrites if r["source"].startswith("/api/")]
    assert api_routes, "vercel.json must route /api/* to the function"
    assert api_routes[0]["destination"] == "/api/index.py"

    # no catch-all rewrite that would shadow the API
    assert not any(r["source"] in ("/(.*)", "/") for r in rewrites)


def test_vercel_json_gives_the_function_a_realistic_budget():
    config = json.loads((REPO_ROOT / "vercel.json").read_text())
    fn = config["functions"]["api/index.py"]
    assert fn["maxDuration"] >= 60, "a 2500-bar backtest needs more than a minute"
    assert fn["maxDuration"] <= 300, "Hobby tops out at 300s"
    assert fn["memory"] >= 512
    # the function must carry the bot package with it
    assert "bot/**" in fn["includeFiles"]


def test_python_version_pinned():
    """Vercel reads the Python version from .python-version at the project root.

    The file must be a bare version string - pyenv and Vercel both choke on
    comments or blank lines.
    """
    raw = (REPO_ROOT / ".python-version").read_text()
    # one line, optionally terminated by a single newline - nothing else
    assert raw == raw.strip() + "\n" or raw == raw.strip(), "stray whitespace"
    text = raw.strip()
    assert "\n" not in text, "exactly one line, no comments"
    assert not text.startswith("#"), "no comments in .python-version"
    assert text and text[0].isdigit()
    major, minor = (int(p) for p in text.split(".")[:2])
    assert (major, minor) >= (3, 9), "Vercel's Python runtime needs 3.9+"


def test_requirements_exist_for_the_build():
    """Vercel installs requirements.txt before building the function."""
    assert (REPO_ROOT / "requirements.txt").is_file()
