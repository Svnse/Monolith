"""Tests for core/acatalepsy/llm_sidecar.py and runtime.py.

Sidecar tests use a fake OpenAI client (no real network) to verify
the auditor-tuned defaults reach the call site: temp=0, bounded
max_tokens, the right model name, system+user message shape.

Runtime tests verify the register/get/deregister singleton lifecycle.
"""
from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


# ── Sidecar — config validation ──────────────────────────────────────


def test_make_auditor_llm_rejects_missing_api_base() -> None:
    from core.acatalepsy.llm_sidecar import make_auditor_llm, SidecarConfigError
    with pytest.raises(SidecarConfigError, match="api_base"):
        make_auditor_llm(config_override={
            "backend": "gguf_api", "api_base": "", "api_model": "x", "api_key": "k",
        })


def test_make_auditor_llm_rejects_missing_api_model() -> None:
    from core.acatalepsy.llm_sidecar import make_auditor_llm, SidecarConfigError
    with pytest.raises(SidecarConfigError, match="api_model"):
        make_auditor_llm(config_override={
            "backend": "gguf_api", "api_base": "http://localhost:8000",
            "api_model": "", "api_key": "k",
        })


def test_make_auditor_llm_rejects_gguf_without_api_base() -> None:
    from core.acatalepsy.llm_sidecar import make_auditor_llm, SidecarUnsupportedBackend
    with pytest.raises(SidecarUnsupportedBackend, match="gguf"):
        make_auditor_llm(config_override={
            "backend": "gguf", "api_base": "", "api_model": "x", "api_key": "",
        })


def test_make_auditor_llm_accepts_gguf_with_api_base() -> None:
    """gguf is OK as long as an api_base is provided (front-server pattern)."""
    from core.acatalepsy import llm_sidecar
    # Inject a fake openai module so make_auditor_llm can build the callable.
    fake_module = _install_fake_openai_module()
    try:
        callable_ = llm_sidecar.make_auditor_llm(config_override={
            "backend": "gguf", "api_base": "http://localhost:8000",
            "api_model": "local-model", "api_key": "",
        })
        assert callable_ is not None
    finally:
        _uninstall_fake_openai_module(fake_module)


# ── Sidecar — call shape (with fake openai client) ───────────────────


def _install_fake_openai_module() -> types.ModuleType | None:
    """Inject a fake `openai` module into sys.modules. Returns the
    previous module so we can restore it after the test.

    The fake exposes `OpenAI` as a MagicMock; calls to
    `client.chat.completions.create(...)` return an object whose
    .choices[0].message.content is the canned response.
    """
    previous = sys.modules.get("openai")

    fake = types.ModuleType("openai")

    def _make_response(content: str):
        # Build the openai-SDK-like return shape
        msg = types.SimpleNamespace(content=content)
        choice = types.SimpleNamespace(message=msg)
        return types.SimpleNamespace(choices=[choice])

    def OpenAI(*args, **kwargs):
        client = MagicMock(name="FakeOpenAIClient")
        client._init_args = args
        client._init_kwargs = kwargs
        # Default response: empty candidates list
        client.chat.completions.create.return_value = _make_response(
            '{"candidates": []}'
        )
        return client

    fake.OpenAI = OpenAI
    sys.modules["openai"] = fake
    return previous


def _uninstall_fake_openai_module(previous: types.ModuleType | None) -> None:
    if previous is not None:
        sys.modules["openai"] = previous
    else:
        sys.modules.pop("openai", None)


def test_sidecar_callable_invokes_client_with_auditor_defaults() -> None:
    from core.acatalepsy import llm_sidecar
    previous = _install_fake_openai_module()
    try:
        # Build callable
        callable_ = llm_sidecar.make_auditor_llm(
            config_override={
                "backend": "gguf_api",
                "api_base": "http://localhost:8000",
                "api_model": "test-model-v1",
                "api_key": "sk-test",
            },
            temperature=0.0,
            timeout_secs=42.0,
            max_tokens=4096,
        )

        # Get a handle on the underlying client by re-calling OpenAI
        # (the factory captured the same one). We patched OpenAI to be
        # MagicMock-backed, so each .OpenAI() returns a new MagicMock —
        # the actual client is held inside the callable's closure. We
        # exercise the callable then assert via the response shape.
        result = callable_(system_prompt="SYSTEM", user_content="USER")
        assert result == '{"candidates": []}'

        # Inspect the call. The callable already invoked client.chat.completions.create
        # — re-create a sentinel call to compare? Simpler: spy via a captured
        # client. Use a more elaborate fake.
    finally:
        _uninstall_fake_openai_module(previous)


def test_sidecar_callable_passes_correct_kwargs() -> None:
    """Deeper introspection: verify the callable invokes the client with
    temp / max_tokens / model / message shape we expect."""
    from core.acatalepsy import llm_sidecar

    captured_calls: list[dict] = []
    previous = sys.modules.get("openai")
    fake = types.ModuleType("openai")

    class CapturingClient:
        def __init__(self, *args, **kwargs):
            self.init_kwargs = kwargs
            self.chat = types.SimpleNamespace(
                completions=types.SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs):
            captured_calls.append(kwargs)
            msg = types.SimpleNamespace(content='{"candidates": []}')
            choice = types.SimpleNamespace(message=msg)
            return types.SimpleNamespace(choices=[choice])

    fake.OpenAI = CapturingClient
    sys.modules["openai"] = fake
    try:
        callable_ = llm_sidecar.make_auditor_llm(
            config_override={
                "backend": "gguf_api",
                "api_base": "http://localhost:8000",
                "api_model": "test-model-v1",
                "api_key": "sk-test",
            },
            temperature=0.0,
            timeout_secs=42.0,
            max_tokens=4096,
        )
        callable_(system_prompt="SYSTEM PROMPT", user_content="USER CONTENT")

        assert len(captured_calls) == 1
        call_kwargs = captured_calls[0]
        assert call_kwargs["model"] == "test-model-v1"
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 4096
        assert call_kwargs["messages"] == [
            {"role": "system", "content": "SYSTEM PROMPT"},
            {"role": "user", "content": "USER CONTENT"},
        ]
    finally:
        if previous is not None:
            sys.modules["openai"] = previous
        else:
            sys.modules.pop("openai", None)


# ── Runtime singleton ────────────────────────────────────────────────


def test_runtime_register_get_deregister() -> None:
    from core.acatalepsy import runtime
    # Start clean
    runtime.deregister_worker()
    assert runtime.get_active_worker() is None

    fake_worker = MagicMock(name="FakeWorker")
    runtime.register_worker(fake_worker)
    assert runtime.get_active_worker() is fake_worker

    runtime.deregister_worker()
    assert runtime.get_active_worker() is None


def test_runtime_register_replaces() -> None:
    from core.acatalepsy import runtime
    runtime.deregister_worker()
    a, b = MagicMock(name="a"), MagicMock(name="b")
    runtime.register_worker(a)
    runtime.register_worker(b)
    assert runtime.get_active_worker() is b
    runtime.deregister_worker()


def test_runtime_deregister_matches_only_same_worker() -> None:
    """If a new worker is registered between stop+deregister, the older
    deregister call should not clear it."""
    from core.acatalepsy import runtime
    runtime.deregister_worker()
    old, new = MagicMock(name="old"), MagicMock(name="new")
    runtime.register_worker(old)
    # Operator registers new without deregistering old
    runtime.register_worker(new)
    # Old worker's shutdown handler calls deregister with itself
    runtime.deregister_worker(old)
    # New should still be the active one
    assert runtime.get_active_worker() is new
    runtime.deregister_worker()
