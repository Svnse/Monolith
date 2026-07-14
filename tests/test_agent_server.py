from __future__ import annotations

import threading
import urllib.request

from core.version import APP_VERSION
from engine.agent_server import (
    EVENT_THINKING,
    EVENT_TOKEN,
    AgentServer,
    _ThinkingStreamParser,
    _clean_agent_response,
    _extract_thinking_text,
)


def test_mcp_initialize_reports_application_version() -> None:
    responses: list[dict] = []

    AgentServer()._dispatch_mcp(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}},
        responses.append,
    )

    assert responses[0]["result"]["serverInfo"] == {
        "name": "monolith",
        "version": APP_VERSION,
    }


def test_handle_chat_callback_exception_releases_gate() -> None:
    server = AgentServer()
    server.on_message = lambda _agent, _msg: (_ for _ in ()).throw(RuntimeError("boom"))

    ok1, msg1 = server._handle_chat("agent", "hello")
    ok2, msg2 = server._handle_chat("agent", "hello again")

    assert ok1 is False
    assert ok2 is False
    assert "failed to dispatch" in msg1.lower()
    assert "failed to dispatch" in msg2.lower()
    assert server._busy is False


def test_stop_shuts_down_http_thread() -> None:
    server = AgentServer()
    server.start(0)

    thread = server._http_thread
    assert thread is not None
    assert thread.is_alive()

    server.stop()
    assert not thread.is_alive()


def test_webhook_delivery_failure_is_logged(monkeypatch) -> None:
    server = AgentServer()
    logs: list[str] = []
    server.on_log = logs.append

    def _raise(*_args, **_kwargs):
        raise RuntimeError("webhook down")

    monkeypatch.setattr(urllib.request, "urlopen", _raise)
    server._fire_webhook("http://example.invalid/hook", {"event": "token", "data": {}, "ts": 0})

    assert any("webhook delivery failed" in line for line in logs)


def test_register_webhook_rejects_invalid_url() -> None:
    server = AgentServer()
    result = server.register_webhook("ftp://example.com/hook", ["token"])
    assert result["ok"] is False
    assert "invalid webhook url" in result["error"]


def test_register_webhook_enforces_limit(monkeypatch) -> None:
    server = AgentServer()
    monkeypatch.setattr(server, "MAX_WEBHOOKS", 1)
    ok = server.register_webhook("http://example.com/hook", ["token"])
    blocked = server.register_webhook("http://example.com/hook2", ["token"])

    assert ok["ok"] is True
    assert blocked["ok"] is False
    assert "too many webhooks" in blocked["error"]


def test_broadcast_event_skips_when_webhook_inflight_limit_reached() -> None:
    server = AgentServer()
    logs: list[str] = []
    server.on_log = logs.append
    server.register_webhook("http://example.com/hook", ["token"])
    server._webhook_inflight = threading.BoundedSemaphore(1)
    # Exhaust the semaphore so dispatch must skip.
    assert server._webhook_inflight.acquire(blocking=False)

    try:
        server.broadcast_event("token", {"text": "hello"})
    finally:
        server._webhook_inflight.release()

    assert any("in-flight limit reached" in line for line in logs)


def test_handle_chat_rejects_large_messages() -> None:
    server = AgentServer()
    over_limit = "x" * (server.MAX_MESSAGE_CHARS + 1)
    ok, msg = server._handle_chat("agent", over_limit)
    assert ok is False
    assert "message too large" in msg


def test_handle_chat_busy_wait_uses_short_window(monkeypatch) -> None:
    server = AgentServer()
    monkeypatch.setattr(server, "BUSY_WAIT_TIMEOUT", 0.01)

    with server._gate:
        server._busy = True

    ok, msg = server._handle_chat("agent", "hello")
    assert ok is False
    assert "busy" in msg.lower()


def test_start_binds_to_loopback_by_default() -> None:
    server = AgentServer()
    server.start(0)
    try:
        assert server._host == "127.0.0.1"
    finally:
        server.stop()


def test_subscribe_sse_enforces_limit(monkeypatch) -> None:
    server = AgentServer()
    monkeypatch.setattr(server, "MAX_SSE_SUBSCRIBERS", 1)

    first = server._subscribe_sse()
    second = server._subscribe_sse()

    assert first is not None
    assert second is None

    server._unsubscribe_sse(first)
    third = server._subscribe_sse()
    assert third is not None


def test_agent_server_auth_token_validation(monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_AGENT_TOKEN", "secret-123")
    server = AgentServer()

    assert server._is_authorized({}) is False
    assert server._is_authorized({"Authorization": "Bearer wrong"}) is False
    assert server._is_authorized({"Authorization": "Bearer secret-123"}) is True
    assert server._is_authorized({"X-API-Key": "secret-123"}) is True


def test_extract_thinking_text_returns_only_internal_reasoning() -> None:
    raw = "<think>step one</think>Hello<analysis>step two</analysis> world"

    assert _extract_thinking_text(raw) == "step one\n\nstep two"
    assert _clean_agent_response(raw) == "Hello world"


def test_clean_agent_response_strips_private_curiosity_block() -> None:
    raw = (
        "Visible answer\n"
        "<curiosity>\n"
        "monolith | is curious_about | why precision feels load-bearing\n"
        "</curiosity>"
    )

    clean = _clean_agent_response(raw)

    assert clean == "Visible answer"
    assert "CURIOSITY" not in clean
    assert "precision feels load-bearing" not in clean


def test_thinking_stream_parser_splits_tags_across_chunks() -> None:
    parser = _ThinkingStreamParser()

    out1 = parser.feed("Hello <thi")
    out2 = parser.feed("nk>plan")
    out3 = parser.feed(" A</think> world")
    out4 = parser.flush()

    assert out1 == [(EVENT_TOKEN, "Hello ")]
    assert out2 == [(EVENT_THINKING, "plan")]
    assert out3 == [(EVENT_THINKING, " A"), (EVENT_TOKEN, " world")]
    assert out4 == []


def test_handle_chat_result_returns_raw_and_clean_response() -> None:
    server = AgentServer()

    def _respond(_agent: str, _msg: str) -> None:
        server.push_token("<think>private plan</think>Visible answer")
        server.push_done()

    server.on_message = _respond
    result = server._handle_chat_result("agent", "hello")

    assert result["ok"] is True
    assert result["response"] == "Visible answer"
    assert result["raw_response"] == "<think>private plan</think>Visible answer"
