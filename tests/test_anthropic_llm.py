"""Native Anthropic backend adapter: speaks /v1/messages internally, exposes the OpenAICompatLLM
interface (a create_chat_completion generator yielding {"choices":[{"delta":{"content":...}}]}).
All HTTP is mocked — no real API call in unit tests."""
from __future__ import annotations

import io
import json
import urllib.error
import urllib.request

import pytest

from engine.anthropic_llm import AnthropicLLM, is_anthropic_url


class _FakeResp:
    def __init__(self, lines=None, body=None):
        self._lines = lines or []
        self._body = body
        self.closed = False

    def __iter__(self):
        return iter(self._lines)

    def read(self):
        return self._body

    def close(self):
        self.closed = True


def _patch_urlopen(monkeypatch, captured, resp):
    def _fake(req, timeout=None):
        captured["req"] = req
        return resp
    monkeypatch.setattr("engine.anthropic_llm._http_open", _fake)


# ── detection + factory ──────────────────────────────────────────────


def test_is_anthropic_url():
    assert is_anthropic_url("https://api.anthropic.com")
    assert is_anthropic_url("https://api.anthropic.com/v1")
    assert not is_anthropic_url("https://api.deepseek.com")
    assert not is_anthropic_url("https://api.openai.com/v1")
    assert not is_anthropic_url("")


def test_factory_routes_by_url():
    from engine.llm import OpenAICompatLLM, make_cloud_llm
    assert isinstance(make_cloud_llm("https://api.anthropic.com", "k", "claude-opus-4-8"), AnthropicLLM)
    assert isinstance(make_cloud_llm("https://api.deepseek.com", "k", "deepseek-v4-pro"), OpenAICompatLLM)


# ── headers ──────────────────────────────────────────────────────────


def test_headers_use_x_api_key_and_version_not_bearer():
    h = AnthropicLLM("https://api.anthropic.com", "sk-ant-xyz", "claude-opus-4-8")._headers()
    assert h["x-api-key"] == "sk-ant-xyz"
    assert h["anthropic-version"]
    assert "Authorization" not in h


# ── request translation ──────────────────────────────────────────────


def test_request_translation(monkeypatch):
    captured: dict = {}
    _patch_urlopen(monkeypatch, captured, _FakeResp(
        body=json.dumps({"content": [{"type": "text", "text": "Paris"}]}).encode("utf-8")))
    llm = AnthropicLLM("https://api.anthropic.com", "k", "claude-opus-4-8")
    list(llm.create_chat_completion(
        messages=[{"role": "system", "content": "be terse"},
                  {"role": "user", "content": "capital of France?"}],
        max_tokens=123, temperature=0.5, stream=False, enable_thinking=False))
    req = captured["req"]
    assert req.full_url.endswith("/v1/messages")
    body = json.loads(req.data)
    assert body["model"] == "claude-opus-4-8"
    assert body["system"] == "be terse"                                   # system extracted top-level
    assert body["messages"] == [{"role": "user", "content": "capital of France?"}]
    assert body["max_tokens"] == 123
    assert "enable_thinking" not in body                                  # Monolith-only kwarg dropped


def test_non_stream_yields_one_content_chunk(monkeypatch):
    captured: dict = {}
    _patch_urlopen(monkeypatch, captured, _FakeResp(
        body=json.dumps({"content": [{"type": "text", "text": "Paris"}]}).encode("utf-8")))
    llm = AnthropicLLM("https://api.anthropic.com", "k", "claude-opus-4-8")
    chunks = list(llm.create_chat_completion(messages=[{"role": "user", "content": "q"}], stream=False))
    assert chunks == [{"choices": [{"delta": {"content": "Paris"}}]}]


# ── SSE streaming translation ────────────────────────────────────────


def test_streaming_translates_anthropic_sse_to_openai_chunks(monkeypatch):
    captured: dict = {}
    lines = [
        b'event: content_block_delta\n',
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hel"}}\n',
        b'\n',
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"lo"}}\n',
        b'\n',
        b'data: {"type":"message_stop"}\n',
        b'\n',
    ]
    _patch_urlopen(monkeypatch, captured, _FakeResp(lines=lines))
    llm = AnthropicLLM("https://api.anthropic.com", "k", "claude-opus-4-8")
    out = list(llm.create_chat_completion(messages=[{"role": "user", "content": "q"}], stream=True))
    texts = [c["choices"][0]["delta"].get("content") for c in out if "choices" in c]
    assert "".join(t for t in texts if t) == "Hello"


# ── message normalization (Anthropic requires alternation, start user) ─


def test_messages_normalized_start_user_and_merge_consecutive(monkeypatch):
    captured: dict = {}
    _patch_urlopen(monkeypatch, captured, _FakeResp(body=json.dumps({"content": []}).encode("utf-8")))
    llm = AnthropicLLM("https://api.anthropic.com", "k", "claude-opus-4-8")
    list(llm.create_chat_completion(messages=[
        {"role": "system", "content": "S"},
        {"role": "assistant", "content": "A1"},   # starts assistant → a user must precede
        {"role": "user", "content": "U1"},
        {"role": "user", "content": "U2"},         # consecutive user → merge
    ], stream=False))
    roles = [m["role"] for m in json.loads(captured["req"].data)["messages"]]
    assert roles[0] == "user"
    assert all(roles[i] != roles[i + 1] for i in range(len(roles) - 1))


# ── deprecated-param retry (Opus 4.8 rejects temperature) ────────────


def test_retries_stripping_deprecated_param(monkeypatch):
    calls = []

    def _fake(req, timeout=None):
        payload = json.loads(req.data)
        calls.append(payload)
        if "temperature" in payload:
            raise urllib.error.HTTPError(
                req.full_url, 400, "Bad Request", {},
                io.BytesIO(b'{"error":{"message":"`temperature` is deprecated for this model."}}'))
        return _FakeResp(body=json.dumps({"content": [{"type": "text", "text": "ok"}]}).encode("utf-8"))

    monkeypatch.setattr("engine.anthropic_llm._http_open", _fake)
    llm = AnthropicLLM("https://api.anthropic.com", "k", "claude-opus-4-8")
    chunks = list(llm.create_chat_completion(
        messages=[{"role": "user", "content": "q"}], temperature=0.5, enable_thinking=False, stream=False))
    assert chunks == [{"choices": [{"delta": {"content": "ok"}}]}]
    assert len(calls) == 2                  # retried once
    assert "temperature" in calls[0]        # first attempt carried it
    assert "temperature" not in calls[1]    # stripped the deprecated param, then succeeded


# ── v2: extended thinking → reasoning_content lane ───────────────────


def test_thinking_enabled_adds_param_and_drops_sampling(monkeypatch):
    captured: dict = {}
    _patch_urlopen(monkeypatch, captured, _FakeResp(
        body=json.dumps({"content": [{"type": "text", "text": "ok"}]}).encode("utf-8")))
    llm = AnthropicLLM("https://api.anthropic.com", "k", "claude-opus-4-8")
    list(llm.create_chat_completion(
        messages=[{"role": "user", "content": "q"}],
        max_tokens=65536, temperature=0.7, top_p=0.9, enable_thinking=True, stream=False))
    body = json.loads(captured["req"].data)
    assert body["thinking"]["type"] == "adaptive"
    assert body["thinking"]["display"] == "summarized"   # Opus 4.8 defaults to omitted; opt in
    assert body["output_config"]["effort"]               # effort guidance present
    assert "temperature" not in body and "top_p" not in body and "top_k" not in body


def test_thinking_on_by_default_for_thinking_model(monkeypatch):
    # a thinking model thinks unless suppressed -> absent enable_thinking => thinking present
    captured: dict = {}
    _patch_urlopen(monkeypatch, captured, _FakeResp(
        body=json.dumps({"content": [{"type": "text", "text": "ok"}]}).encode("utf-8")))
    llm = AnthropicLLM("https://api.anthropic.com", "k", "claude-opus-4-8")
    list(llm.create_chat_completion(messages=[{"role": "user", "content": "q"}], stream=False))
    assert json.loads(captured["req"].data).get("thinking", {}).get("type") == "adaptive"


def test_thinking_suppressed_when_explicitly_false(monkeypatch):
    captured: dict = {}
    _patch_urlopen(monkeypatch, captured, _FakeResp(
        body=json.dumps({"content": [{"type": "text", "text": "ok"}]}).encode("utf-8")))
    llm = AnthropicLLM("https://api.anthropic.com", "k", "claude-opus-4-8")
    list(llm.create_chat_completion(
        messages=[{"role": "user", "content": "q"}], enable_thinking=False, stream=False))
    assert "thinking" not in json.loads(captured["req"].data)


def test_thinking_bumps_max_tokens_when_too_small(monkeypatch):
    captured: dict = {}
    _patch_urlopen(monkeypatch, captured, _FakeResp(
        body=json.dumps({"content": [{"type": "text", "text": "ok"}]}).encode("utf-8")))
    llm = AnthropicLLM("https://api.anthropic.com", "k", "claude-opus-4-8")
    list(llm.create_chat_completion(
        messages=[{"role": "user", "content": "q"}], max_tokens=1024, enable_thinking=True, stream=False))
    body = json.loads(captured["req"].data)
    assert body["max_tokens"] >= 8192   # bumped to the floor so adaptive thinking has room


def test_thinking_streams_as_reasoning_content(monkeypatch):
    captured: dict = {}
    lines = [
        b'data: {"type":"content_block_delta","delta":{"type":"thinking_delta","thinking":"weigh it"}}\n',
        b'\n',
        b'data: {"type":"content_block_delta","delta":{"type":"signature_delta","signature":"sig"}}\n',
        b'\n',
        b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Answer"}}\n',
        b'\n',
        b'data: {"type":"message_stop"}\n',
        b'\n',
    ]
    _patch_urlopen(monkeypatch, captured, _FakeResp(lines=lines))
    llm = AnthropicLLM("https://api.anthropic.com", "k", "claude-opus-4-8")
    out = list(llm.create_chat_completion(
        messages=[{"role": "user", "content": "q"}], enable_thinking=True, stream=True))
    reasoning = "".join(c["choices"][0]["delta"].get("reasoning_content", "") for c in out if "choices" in c)
    content = "".join(c["choices"][0]["delta"].get("content", "") for c in out if "choices" in c)
    assert reasoning == "weigh it"     # thinking_delta -> reasoning_content lane
    assert content == "Answer"          # signature_delta ignored, text_delta -> content


# ── IPv4-forced connect (dodge the dead IPv6 AAAA that blocks ~21s before fallback) ─


def test_connection_forces_ipv4(monkeypatch):
    import socket
    from engine.anthropic_llm import _IPv4HTTPSConnection
    calls: dict = {}

    def fake_getaddrinfo(host, port, family=0, *a, **k):
        calls["family"] = family            # must be AF_INET, never the OS-default (IPv6-first) order
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("160.79.104.10", 443))]

    class _FakeSock:
        def settimeout(self, *a): pass
        def bind(self, *a): pass
        def setsockopt(self, *a): pass
        def connect(self, sa): calls["connected"] = sa
        def close(self): pass

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    monkeypatch.setattr(socket, "socket", lambda *a, **k: _FakeSock())
    conn = _IPv4HTTPSConnection("api.anthropic.com", 443)
    conn._context = type("Ctx", (), {"wrap_socket": staticmethod(lambda s, server_hostname=None: s)})()
    conn.connect()

    assert calls["family"] == socket.AF_INET                  # resolves IPv4 only
    assert calls["connected"] == ("160.79.104.10", 443)       # connects straight to the working route
