"""The /reset endpoint — headless "New Chat" for the monothink trainer.

The trainer's --chat appends forever to one conversation, so Monolith goes warm
(models the examiner) and stops failing. /reset starts a fresh/cold chat surface
the way the UI's New Chat does. These tests cover the handler->callback seam
(no Qt needed) using the same live-HTTP round-trip pattern as
test_agent_server_sp2.test_http_monothink_and_rating_roundtrip:

  * wired on_reset is invoked and its receipt echoes back, ok:true
  * unwired (no UI) falls back to a headless working_memory clear, ok:true
  * an on_reset that raises never 500s — returns ok:false with an error

The actual PageChat._start_new_session thread-hop is NOT exercised here (it
needs a running Qt app); the ConnectionsPage wiring that emits the queued
signal is verified by reading, and confirmed live only after a Monolith restart.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request

import pytest

import engine.agent_server as ags


@pytest.fixture(autouse=True)
def _protect_bearing(monkeypatch):
    """Never let these tests clobber the live config/bearing.json. Stub the
    store's clear_bearing with a counter so the cold-reset can be asserted
    WITHOUT touching disk. Autouse → every test in this file is protected."""
    calls = {"n": 0}
    monkeypatch.setattr(
        "addons.system.bearing.store.clear_bearing",
        lambda: calls.__setitem__("n", calls["n"] + 1),
    )
    return calls


def _post(port: int, path: str, body: dict | None = None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode("utf-8"))


def test_reset_invokes_wired_callback_and_echoes_receipt():
    calls = {"n": 0}

    def on_reset():
        calls["n"] += 1
        return {"dispatched": True, "surface": "new_chat"}

    server = ags.AgentServer()
    server.on_reset = on_reset
    server.start(0)
    try:
        _host, port = server._httpd.server_address
        st, j = _post(port, "/reset", {})
        assert st == 200
        assert j["ok"] is True
        assert j["dispatched"] is True
        assert j["surface"] == "new_chat"
        assert calls["n"] == 1  # the wired callback actually fired
    finally:
        server.stop()


def test_reset_tolerates_empty_body():
    """No request body is required — a bare POST must still reset, not hang."""
    calls = {"n": 0}
    server = ags.AgentServer()
    server.on_reset = lambda: calls.__setitem__("n", calls["n"] + 1) or {"dispatched": True}
    server.start(0)
    try:
        _host, port = server._httpd.server_address
        st, j = _post(port, "/reset", None)  # no Content-Length / no body
        assert st == 200
        assert j["ok"] is True
        assert calls["n"] == 1
    finally:
        server.stop()


def test_reset_headless_fallback_clears_working_memory(monkeypatch):
    """With no UI wired (on_reset is None), fall back to a module-level
    working_memory clear and report honestly that conversation_history was NOT
    cleared (it lives in the UI-side engine, unreachable from the server)."""
    cleared = {"n": 0}
    monkeypatch.setattr(
        "core.continuity.clear_working_memory",
        lambda: cleared.__setitem__("n", cleared["n"] + 1),
    )
    server = ags.AgentServer()
    assert server.on_reset is None  # default: no UI wired
    server.start(0)
    try:
        _host, port = server._httpd.server_address
        st, j = _post(port, "/reset", {})
        assert st == 200
        assert j["ok"] is True
        assert j["headless"] is True
        assert j["working_memory_cleared"] is True
        assert j["conversation_history_cleared"] is False
        assert cleared["n"] == 1
    finally:
        server.stop()


def test_reset_never_500s_when_callback_raises():
    """A failing on_reset must surface ok:false, not a 500."""
    def boom():
        raise RuntimeError("kaboom")

    server = ags.AgentServer()
    server.on_reset = boom
    server.start(0)
    try:
        _host, port = server._httpd.server_address
        st, j = _post(port, "/reset", {})
        assert st == 200  # handler caught it; not a 500
        assert j["ok"] is False
        assert "kaboom" in j["error"]
    finally:
        server.stop()


def test_reset_clears_bearing_for_cold_surface(_protect_bearing):
    """/reset must also clear the cross-session BEARING. The stale [BEARING]
    block (frozen on a prior session, re-injected every turn) is what keeps
    Monolith warm ACROSS a conversation reset — the UI's New Chat keeps bearing
    by design, so the trainer's cold /reset has to clear it explicitly."""
    server = ags.AgentServer()
    server.on_reset = lambda: {"dispatched": True, "surface": "new_chat"}
    server.start(0)
    try:
        _host, port = server._httpd.server_address
        st, j = _post(port, "/reset", {})
        assert st == 200
        assert j["ok"] is True
        assert j["bearing_cleared"] is True
        assert _protect_bearing["n"] == 1  # the store clear actually fired
    finally:
        server.stop()
