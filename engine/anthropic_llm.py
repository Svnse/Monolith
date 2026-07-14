"""Native Anthropic backend adapter.

Speaks Anthropic's /v1/messages wire format (x-api-key + anthropic-version headers, system as a
top-level field, max_tokens required, its own SSE event schema) but exposes the SAME interface as
engine.llm.OpenAICompatLLM — a create_chat_completion(**kwargs) generator yielding
{"choices":[{"delta":{"content": ...}}]} chunks — so nothing downstream changes. Selected by
engine.llm.make_cloud_llm when the api_base URL is an Anthropic host.

v1 scope: chat + streaming text. Extended thinking and native tool-use are deferred — Monolith's loop
is text-based (tool calls come through the model's text per the system prompt).
"""
from __future__ import annotations

import http.client
import json
import re
import socket
import urllib.error
import urllib.request

ANTHROPIC_VERSION = "2023-06-01"
_DEFAULT_MAX_TOKENS = 4096
_THINKING_MIN_MAX_TOKENS = 8192  # floor so adaptive thinking leaves room for the answer
_THINKING_EFFORT = "high"        # output_config.effort that elicits real thinking (Opus 4.8)
_STOP = object()  # sentinel: end the stream
_STRIP_HINT = re.compile(r"`([a-zA-Z_][a-zA-Z0-9_]*)`")  # backtick-quoted field in an error body


# ── IPv4-forced transport ────────────────────────────────────────────────────
# api.anthropic.com advertises an IPv6 (AAAA) address. On a host with no working IPv6 route,
# urllib tries IPv6 FIRST and blocks the full ~21s TCP connect timeout before falling back to the
# IPv4 that connects in ~30ms — a fixed ~22s tax on EVERY call (measured; independent of model,
# thinking, prompt size). Force IPv4 while preserving TLS + SNI + cert verification.


class _IPv4HTTPSConnection(http.client.HTTPSConnection):
    def connect(self):
        if self._tunnel_host:                       # proxy CONNECT tunnel: defer to stdlib path
            return super().connect()
        infos = socket.getaddrinfo(self.host, self.port, socket.AF_INET, socket.SOCK_STREAM)
        af, socktype, proto, _canon, sa = infos[0]
        sock = socket.socket(af, socktype, proto)
        try:
            if self.timeout is not None and self.timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                sock.settimeout(self.timeout)
            if self.source_address:
                sock.bind(self.source_address)
            sock.connect(sa)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except Exception:
            sock.close()
            raise
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


class _IPv4HTTPSHandler(urllib.request.HTTPSHandler):
    def https_open(self, req):
        return self.do_open(_IPv4HTTPSConnection, req)


_OPENER = urllib.request.build_opener(_IPv4HTTPSHandler())


def _http_open(req, timeout):
    """Open req through the IPv4-forced opener. Single transport seam (patched in unit tests)."""
    return _OPENER.open(req, timeout=timeout)


def is_anthropic_url(base_url: str) -> bool:
    """True if base_url points at Anthropic (so the native adapter is used)."""
    b = (base_url or "").strip().lower()
    return "anthropic.com" in b or b.startswith("anthropic") or "://anthropic" in b


def _messages_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if base.endswith("/v1/messages"):
        return base
    if base.endswith("/v1"):
        return f"{base}/messages"
    return f"{base}/v1/messages"


def _models_url(base_url: str) -> str:
    base = (base_url or "").strip().rstrip("/")
    if base.endswith("/models"):
        return base
    if base.endswith("/v1"):
        return f"{base}/models"
    return f"{base}/v1/models"


def _split_system_and_messages(messages):
    """OpenAI-shaped messages -> (system_str, [user/assistant only, normalized])."""
    system_parts = []
    convo = []
    for m in messages or []:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).lower()
        content = m.get("content", "")
        content = "" if content is None else str(content)
        if role == "system":
            if content:
                system_parts.append(content)
        else:
            convo.append({"role": "assistant" if role == "assistant" else "user",
                          "content": content})
    return ("\n\n".join(system_parts), _normalize_alternation(convo))


def _normalize_alternation(convo):
    """Anthropic requires messages to alternate user/assistant and start with user. Merge
    consecutive same-role turns; prepend a user turn if it would otherwise start with assistant."""
    out: list[dict] = []
    for m in convo:
        if out and out[-1]["role"] == m["role"]:
            out[-1] = {"role": m["role"], "content": f"{out[-1]['content']}\n\n{m['content']}"}
        else:
            out.append(dict(m))
    if out and out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": "(continue)"})
    return out


class AnthropicLLM:
    """Same shape as OpenAICompatLLM, native Anthropic underneath."""

    def __init__(self, base_url: str, api_key: str | None, model: str, timeout: int = 300):
        self.base_url = (base_url or "").strip().rstrip("/")
        self.api_key = api_key or ""
        self.model = model
        self.timeout = timeout
        self._active_response = None

    def cancel(self) -> None:
        resp = self._active_response
        self._active_response = None
        if resp is None:
            return
        try:
            resp.close()
        except Exception:
            pass

    def _headers(self) -> dict:
        h = {"content-type": "application/json", "anthropic-version": ANTHROPIC_VERSION}
        if self.api_key:
            h["x-api-key"] = self.api_key
        return h

    def list_models(self) -> dict:
        try:
            req = urllib.request.Request(_models_url(self.base_url), headers=self._headers(), method="GET")
            with _http_open(req, self.timeout) as resp:
                return json.loads(resp.read())
        except Exception:
            return {}

    def _build_payload(self, kwargs: dict) -> dict:
        system, convo = _split_system_and_messages(kwargs.get("messages") or [])
        try:
            max_tokens = int(kwargs.get("max_tokens") or _DEFAULT_MAX_TOKENS)
        except (TypeError, ValueError):
            max_tokens = _DEFAULT_MAX_TOKENS
        payload: dict = {"model": self.model, "messages": convo, "max_tokens": max(1, max_tokens)}
        if system:
            payload["system"] = system
        for k in ("temperature", "top_p", "top_k"):
            v = kwargs.get(k)
            if v is not None:
                payload[k] = v
        stop = kwargs.get("stop")
        if stop:
            payload["stop_sequences"] = stop if isinstance(stop, list) else [str(stop)]
        # Extended thinking. `enable_thinking` is a SUPPRESSION knob: a thinking model thinks by
        # default; enable_thinking=False removes it. So think unless it's explicitly False. The
        # budget scales with max_tokens (small auxiliary calls don't balloon), and Anthropic requires
        # max_tokens > budget_tokens, so bump max_tokens to leave headroom for the answer.
        if kwargs.get("enable_thinking", True) is not False:
            # Opus 4.8 uses ADAPTIVE thinking (model picks its own depth), and CRUCIALLY defaults
            # thinking.display to "omitted" (signature only, empty thinking text) — a silent change
            # from 4.6. display="summarized" restores the readable thinking summary as thinking_delta.
            # output_config.effort guides how much it thinks (low/medium/high/xhigh/max; high=default).
            payload["thinking"] = {"type": "adaptive", "display": "summarized"}
            payload["output_config"] = {"effort": _THINKING_EFFORT}
            if payload["max_tokens"] < _THINKING_MIN_MAX_TOKENS:
                payload["max_tokens"] = _THINKING_MIN_MAX_TOKENS
            for k in ("temperature", "top_p", "top_k"):   # incompatible with thinking
                payload.pop(k, None)
        payload["stream"] = bool(kwargs.get("stream", False))
        return payload

    def _open_request(self, payload: dict):
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(_messages_url(self.base_url), data=data,
                                     headers=self._headers(), method="POST")
        return _http_open(req, self.timeout)

    def _open_with_retry(self, payload: dict):
        """POST; on a 400 that names a deprecated/unsupported param (e.g. Opus 4.8 rejects
        `temperature`), strip that param and retry — bounded, never stripping a required field."""
        attempts = 0
        while True:
            try:
                return self._open_request(payload)
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="ignore")
                low = body.lower()
                if (exc.code == 400 and attempts < 4 and any(
                        w in low for w in ("deprecated", "unsupported", "not supported", "unexpected"))):
                    stripped = False
                    for field in _STRIP_HINT.findall(body):
                        if field in payload and field not in ("model", "messages", "max_tokens", "system"):
                            payload.pop(field, None)
                            stripped = True
                    if stripped:
                        attempts += 1
                        continue
                raise RuntimeError(f"HTTP {exc.code}: {body or exc.reason}") from exc

    def create_chat_completion(self, **kwargs):
        payload = self._build_payload(kwargs)
        resp = self._open_with_retry(payload)
        self._active_response = resp

        if not payload["stream"]:
            try:
                body = resp.read()
            finally:
                try:
                    resp.close()
                except Exception:
                    pass
                self._active_response = None
            try:
                parsed = json.loads(body)
            except (json.JSONDecodeError, TypeError):
                parsed = {}
            blocks = parsed.get("content") or []
            reasoning = "".join(str(b.get("thinking", "")) for b in blocks
                                if isinstance(b, dict) and b.get("type") == "thinking")
            text = "".join(str(b.get("text", "")) for b in blocks
                           if isinstance(b, dict) and b.get("type") == "text")
            if reasoning:
                yield {"choices": [{"delta": {"reasoning_content": reasoning}}]}
            if text:
                yield {"choices": [{"delta": {"content": text}}]}
            return

        try:
            data_buf = ""
            try:
                line_iter = iter(resp)
            except Exception:
                line_iter = iter([])
            while True:
                try:
                    raw_line = next(line_iter)
                except StopIteration:
                    break
                except (urllib.error.URLError, ValueError, OSError, AttributeError):
                    break  # cancel() / socket death -> graceful end of stream
                line = raw_line.decode("utf-8", errors="ignore").rstrip("\r\n")
                if not line:
                    if data_buf:
                        chunk = self._translate_event(data_buf)
                        data_buf = ""
                        if chunk is _STOP:
                            break
                        if chunk is not None:
                            yield chunk
                    continue
                if line.startswith("data:"):
                    part = line[len("data:"):].lstrip()
                    data_buf = part if not data_buf else f"{data_buf}\n{part}"
                # event:/id:/retry:/comment lines are ignored
            if data_buf:
                chunk = self._translate_event(data_buf)
                if chunk is not None and chunk is not _STOP:
                    yield chunk
        finally:
            try:
                resp.close()
            except Exception:
                pass
            self._active_response = None

    def _translate_event(self, data_buf: str):
        """Anthropic SSE data json -> an OpenAI content chunk, _STOP, or None (ignore)."""
        try:
            obj = json.loads(data_buf)
        except json.JSONDecodeError:
            return None
        etype = obj.get("type")
        if etype == "content_block_delta":
            delta = obj.get("delta") or {}
            dt = delta.get("type")
            if dt == "text_delta":
                text = str(delta.get("text", ""))
                if text:
                    return {"choices": [{"delta": {"content": text}}]}
            elif dt == "thinking_delta":   # extended thinking -> the reasoning lane
                thinking = str(delta.get("thinking", ""))
                if thinking:
                    return {"choices": [{"delta": {"reasoning_content": thinking}}]}
            # signature_delta and others: ignore
            return None
        if etype in ("message_stop", "error"):
            return _STOP
        return None
