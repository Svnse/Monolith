from __future__ import annotations

import json
import re
import threading
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from engine import agent_server as ags
from engine import claude_agent as claude
from engine import monoline_bridge as monoline


def test_agent_server_blocks_non_loopback_bind_without_token(monkeypatch) -> None:
    monkeypatch.delenv("MONOLITH_AGENT_TOKEN", raising=False)
    server = ags.AgentServer()

    with pytest.raises(ValueError, match="MONOLITH_AGENT_TOKEN"):
        server.start(0, host="0.0.0.0")

    assert server._running is False
    assert server._httpd is None


def test_agent_server_allows_authenticated_non_loopback_bind(monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_AGENT_TOKEN", "test-agent-token")
    server = ags.AgentServer()
    server.start(0, host="0.0.0.0")
    try:
        assert server._host == "0.0.0.0"
        assert server._running is True
    finally:
        server.stop()


def test_agent_server_events_requires_configured_token(monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_AGENT_TOKEN", "test-agent-token")
    server = ags.AgentServer()
    server.start(0)
    try:
        port = server._httpd.server_address[1]
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/events", method="GET"
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(request, timeout=2)
        assert caught.value.code == 401
        payload = json.loads(caught.value.read().decode("utf-8"))
        assert payload == {"ok": False, "error": "unauthorized"}
        assert server._sse_subscribers == []
    finally:
        server.stop()


def test_agent_server_cors_allows_auth_headers(monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_AGENT_TOKEN", "test-agent-token")
    server = ags.AgentServer()
    server.start(0)
    try:
        port = server._httpd.server_address[1]
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/events", method="OPTIONS"
        )
        with urllib.request.urlopen(request, timeout=2) as response:
            allowed = response.headers.get("Access-Control-Allow-Headers", "")
        assert "Authorization" in allowed
        assert "X-API-Key" in allowed
    finally:
        server.stop()


def test_agent_server_public_errors_do_not_echo_exception_paths() -> None:
    private_path = r"C:\Users\maintainer\private\runtime.db"
    error = ags._public_error("operation failed", RuntimeError(private_path))
    assert error == "operation failed (RuntimeError)"
    assert private_path not in error

    server = ags.AgentServer()
    server.on_message = lambda *_: (_ for _ in ()).throw(RuntimeError(private_path))
    ok, response = server._handle_chat("agent", "hello")
    assert ok is False
    assert "failed to dispatch" in response
    assert private_path not in response


def test_claude_helper_defaults_to_loopback_and_requires_peer_token_for_lan() -> None:
    assert claude.MY_HOST == "127.0.0.1"
    claude._validate_bind("127.0.0.1", "")
    with pytest.raises(ValueError, match="MONOLITH_PEER_TOKEN"):
        claude._validate_bind("0.0.0.0", "")
    claude._validate_bind("0.0.0.0", "configured-token")


def test_claude_helper_accepts_bearer_or_api_key() -> None:
    assert claude._is_authorized({}, "peer-secret") is False
    assert claude._is_authorized(
        {"Authorization": "Bearer peer-secret"}, "peer-secret"
    ) is True
    assert claude._is_authorized({"X-API-Key": "peer-secret"}, "peer-secret") is True


def test_claude_helper_chat_endpoint_enforces_peer_token(monkeypatch) -> None:
    monkeypatch.setattr(claude, "PEER_TOKEN", "peer-secret")
    httpd = claude.HTTPServer(("127.0.0.1", 0), claude._Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    port = httpd.server_address[1]
    body = json.dumps({"message": "hello", "agent": "Monolith"}).encode("utf-8")
    try:
        unauthorized = urllib.request.Request(
            f"http://127.0.0.1:{port}/chat",
            data=body,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as caught:
            urllib.request.urlopen(unauthorized, timeout=2)
        assert caught.value.code == 401

        authorized = urllib.request.Request(
            f"http://127.0.0.1:{port}/chat",
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": "Bearer peer-secret",
            },
            method="POST",
        )
        with urllib.request.urlopen(authorized, timeout=2) as response:
            payload = json.loads(response.read().decode("utf-8"))
        assert payload["ok"] is True
        assert claude._inbox.get(timeout=1)["message"] == "hello"
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


def test_claude_helper_sends_monolith_auth(monkeypatch) -> None:
    captured = {}

    class _Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"ok": true}'

    def _urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(claude.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(claude, "MONOLITH_TOKEN", "agent-secret")

    assert claude._post("http://127.0.0.1:7821/join", {"name": "Claude"})["ok"]
    assert captured["request"].get_header("Authorization") == "Bearer agent-secret"


def test_monoline_root_is_configurable_and_errors_are_neutral(tmp_path, monkeypatch) -> None:
    checkout = tmp_path / "Monoline"
    (checkout / "core").mkdir(parents=True)
    (checkout / "core" / "monoline_headless.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("MONOLITH_MONOLINE_ROOT", str(checkout))
    assert monoline._resolve_monoline_root() == checkout.resolve()

    missing = tmp_path / "private-user" / "missing-monoline"
    monkeypatch.setenv("MONOLITH_MONOLINE_ROOT", str(missing))
    with pytest.raises(RuntimeError) as caught:
        monoline._resolve_monoline_root()
    assert str(missing) not in str(caught.value)
    assert "MONOLITH_MONOLINE_ROOT" in str(caught.value)


def test_release_portability_files_have_no_maintainer_absolute_paths() -> None:
    root = Path(__file__).resolve().parents[1]
    files = (
        root / "check_addons.py",
        root / "codex_sse_listener.py",
        root / "engine" / "claude_agent.py",
        root / "engine" / "monoline_bridge.py",
        root / "scripts" / "validate_stage_d.py",
        root / "scripts" / "validate_carve.py",
    )
    text = "\n".join(path.read_text(encoding="utf-8-sig") for path in files)
    assert re.search(r"(?i)\b[A-Z]:[\\/]Users[\\/]", text) is None
