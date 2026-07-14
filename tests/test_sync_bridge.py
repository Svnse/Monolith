"""Tests for ``engine.sync_bridge`` — the headless LLM-call path used by
monothink (and any future caller that needs a one-shot completion outside
the streaming chat plumbing).

The critical invariant under test: thinking-mode backends that emit output
via ``delta.reasoning_content`` instead of ``delta.content`` must surface
*something* to the caller. Two consecutive monothink /rating attempts on a
DeepSeek V4 backend journaled "empty_response" because the original parser
only read ``delta.content``; the reasoning fallback was added so the headless
sync path mirrors the streaming chat parser at engine/llm.py:392-396.
"""
from __future__ import annotations

from typing import Any

from engine import sync_bridge as sb


class _StreamLLM:
    """Minimal stand-in for OpenAICompatLLM that returns whatever stream the
    test wants. Avoids hitting a real API."""

    def __init__(self, stream: list[dict]) -> None:
        self._stream = stream

    def create_chat_completion(self, **kwargs) -> Any:  # noqa: ARG002
        return iter(self._stream)


def _delta(content: str = "", reasoning: str = "") -> dict:
    delta: dict = {}
    if content:
        delta["content"] = content
    if reasoning:
        delta["reasoning_content"] = reasoning
    return {"choices": [{"delta": delta}]}


def test_extract_parts_returns_content_when_present() -> None:
    chunk = _delta(content="hello")
    content, reasoning = sb._extract_parts_from_chunk(chunk)
    assert content == "hello"
    assert reasoning == ""


def test_extract_parts_returns_reasoning_when_present() -> None:
    chunk = _delta(reasoning="thinking out loud")
    content, reasoning = sb._extract_parts_from_chunk(chunk)
    assert content == ""
    assert reasoning == "thinking out loud"


def test_extract_parts_returns_both_when_both_present() -> None:
    chunk = _delta(content="answer", reasoning="thought")
    content, reasoning = sb._extract_parts_from_chunk(chunk)
    assert content == "answer"
    assert reasoning == "thought"


def test_extract_parts_falls_back_to_reasoning_key() -> None:
    """Some OpenAI-compat shims use ``reasoning`` instead of ``reasoning_content``."""
    chunk = {"choices": [{"delta": {"reasoning": "thinking"}}]}
    content, reasoning = sb._extract_parts_from_chunk(chunk)
    assert content == ""
    assert reasoning == "thinking"


def test_extract_parts_reads_message_form_for_non_streaming() -> None:
    """A non-streaming response uses ``choices[0].message.content`` (and
    ``message.reasoning_content`` on thinking backends)."""
    chunk = {
        "choices": [{
            "message": {
                "content": "final answer",
                "reasoning_content": "internal thought",
            },
        }],
    }
    content, reasoning = sb._extract_parts_from_chunk(chunk)
    assert content == "final answer"
    assert reasoning == "internal thought"


def test_generate_sync_prefers_content_when_both_streamed() -> None:
    """When the model emits reasoning chunks AND a final answer chunk,
    callers should see the answer, not the reasoning."""
    stream = [
        _delta(reasoning="step 1"),
        _delta(reasoning="step 2"),
        _delta(content="NO_CHANGE"),
    ]
    result = sb.generate_sync(_StreamLLM(stream), [{"role": "user", "content": "x"}], {})
    assert result == "NO_CHANGE"


def test_generate_sync_falls_back_to_reasoning_when_content_never_emitted() -> None:
    """The empty_response failure mode: thinking-mode model consumed its whole
    token budget on reasoning, never emitted a content chunk. Without this
    fallback, monothink journals empty_response and nothing else; with it,
    the reasoning surfaces and the caller can decide what to do."""
    stream = [
        _delta(reasoning="step 1 — analyze rating signal"),
        _delta(reasoning="step 2 — propose NO_CHANGE because rating was 91"),
    ]
    result = sb.generate_sync(_StreamLLM(stream), [{"role": "user", "content": "x"}], {})
    assert "NO_CHANGE" in result  # the model's reasoning conclusion is visible


def test_generate_sync_returns_empty_string_when_nothing_emitted() -> None:
    """No content AND no reasoning → genuinely empty result (the old
    behavior, preserved)."""
    stream = [{"choices": [{"delta": {}}]}, {"choices": [{"delta": {}}]}]
    result = sb.generate_sync(_StreamLLM(stream), [{"role": "user", "content": "x"}], {})
    assert result == ""


class _CountingStreamLLM:
    """A stream whose generator records how many chunks were actually pulled,
    so a test can prove an early break stopped consuming the stream."""

    def __init__(self, n: int, consumed: dict) -> None:
        self._n = n
        self._consumed = consumed

    def create_chat_completion(self, **kwargs):  # noqa: ARG002
        def _gen():
            for i in range(self._n):
                self._consumed["n"] += 1
                yield _delta(reasoning=f"tok{i}")
        return _gen()


def test_generate_sync_parts_breaks_stream_on_cancel() -> None:
    # Enforce-stop: a should_cancel that flips True mid-stream must break the chunk loop
    # promptly instead of draining the whole generation (the workshop "takes a while" cause).
    consumed = {"n": 0}
    llm = _CountingStreamLLM(100, consumed)
    should_cancel = lambda: consumed["n"] >= 3  # cancel observed once 3 chunks have arrived

    content, reasoning = sb.generate_sync_parts(
        llm, [{"role": "user", "content": "x"}], {}, should_cancel=should_cancel)

    assert consumed["n"] <= 5            # broke early — did NOT consume all 100 chunks
    assert "tok0" in reasoning           # partial output up to the break is returned
    assert "tok9" not in reasoning       # later chunks never processed


def test_generate_sync_parts_consumes_all_when_cancel_never_fires() -> None:
    # Backward-compat: a should_cancel that never returns True must not short-circuit.
    consumed = {"n": 0}
    llm = _CountingStreamLLM(10, consumed)

    _content, reasoning = sb.generate_sync_parts(
        llm, [{"role": "user", "content": "x"}], {}, should_cancel=lambda: False)

    assert consumed["n"] == 10
    assert "tok9" in reasoning


def test_generate_sync_non_streaming_dict_response() -> None:
    """If the backend returns a single dict (non-streaming), the same
    extraction logic applies."""
    response = {
        "choices": [{"message": {"content": "the answer"}}],
    }

    class _OneShot:
        def create_chat_completion(self, **kwargs):  # noqa: ARG002
            return response

    result = sb.generate_sync(_OneShot(), [{"role": "user", "content": "x"}], {})
    assert result == "the answer"
