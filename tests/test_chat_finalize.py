"""Tests for core/chat_finalize.py — READY-time pipeline emit + verifier.

Regression for defect #2 (audit 2026-05-14): TurnReadyEvent must publish to
the pipeline regardless of MONOLITH_VERIFIER_V1. The verifier flag gates only
the synchronous verify_response call; output_sanitizer and verifier_bridge
have their own kill switches and need the event delivered to skip-or-fire on
their own.
"""
from __future__ import annotations

import pytest

from core.chat_finalize import finalize_assistant_turn


def test_emit_fires_when_verifier_disabled(monkeypatch):
    """With MONOLITH_VERIFIER_V1=0, the pipeline emit must STILL fire.

    Pre-fix, chat.py:_run_response_verifier early-returned at the flag check
    before the _emit_pipeline_turn_ready call — so output_sanitizer and
    verifier_bridge silently never received TurnReadyEvent.
    """
    monkeypatch.setenv("MONOLITH_VERIFIER_V1", "0")

    emitted = []
    verdicts = []

    finalize_assistant_turn(
        raw="hello world",
        public="hello world",
        config={},
        emit_pipeline_ready=lambda raw, public, tools_used: emitted.append((raw, public, tools_used)),
        record_verdict=verdicts.append,
    )

    assert len(emitted) == 1, "Pipeline emit must fire even when verifier is disabled"
    assert emitted[0][0] == "hello world"
    assert emitted[0][1] == "hello world"
    assert emitted[0][2] == ()
    assert verdicts == [], "No verdict should be recorded when verifier is disabled"


def test_emit_and_verify_when_verifier_enabled(monkeypatch):
    """With MONOLITH_VERIFIER_V1=1, both the pipeline emit and verdict-recording fire."""
    monkeypatch.setenv("MONOLITH_VERIFIER_V1", "1")

    emitted = []
    verdicts = []

    finalize_assistant_turn(
        raw="hello world",
        public="hello world",
        config={},
        emit_pipeline_ready=lambda raw, public, tools_used: emitted.append((raw, public, tools_used)),
        record_verdict=verdicts.append,
    )

    assert len(emitted) == 1
    assert len(verdicts) == 1
    # Sanity: verdict payload has the documented shape
    assert "verdict" in verdicts[0]
    assert "duration_ms" in verdicts[0]


def test_skip_when_raw_is_empty(monkeypatch):
    """Empty raw input — neither emit nor verify should fire."""
    monkeypatch.setenv("MONOLITH_VERIFIER_V1", "1")

    emitted = []
    verdicts = []

    finalize_assistant_turn(
        raw="",
        public="",
        config={},
        emit_pipeline_ready=lambda raw, public, tools_used: emitted.append(1),
        record_verdict=verdicts.append,
    )

    assert emitted == []
    assert verdicts == []


def test_skip_when_raw_is_whitespace_only(monkeypatch):
    """Whitespace-only raw — same as empty: no emit, no verify."""
    monkeypatch.setenv("MONOLITH_VERIFIER_V1", "1")

    emitted = []
    verdicts = []

    finalize_assistant_turn(
        raw="   \n\t  ",
        public="",
        config={},
        emit_pipeline_ready=lambda raw, public, tools_used: emitted.append(1),
        record_verdict=verdicts.append,
    )

    assert emitted == []
    assert verdicts == []


def test_tools_used_propagates_to_emit(monkeypatch):
    """Caller-supplied tools_used flows to the pipeline emit so the
    verifier_bridge policy can read it off TurnReadyEvent. Previously
    hardcoded to () inside finalize_assistant_turn — pinning that to
    the function's body kept the tool-evidence verifier check dormant
    even when the caller knew the tool list."""
    monkeypatch.setenv("MONOLITH_VERIFIER_V1", "0")  # skip verifier path

    emitted = []
    finalize_assistant_turn(
        raw="answer text",
        public="answer text",
        config={},
        emit_pipeline_ready=lambda raw, public, tools_used: emitted.append(tools_used),
        record_verdict=lambda _p: None,
        tools_used=("read_file", "grep"),
    )

    assert emitted == [("read_file", "grep")]


def test_tools_used_propagates_to_verifier(monkeypatch):
    """Caller-supplied tools_used also flows to the in-process
    verify_response call so the dormant tool-evidence check has the
    list it needs. Previously hardcoded to [] inside the function."""
    monkeypatch.setenv("MONOLITH_VERIFIER_V1", "1")

    captured = []
    import core.chat_finalize as cf

    def fake_verify(**kwargs):
        captured.append(kwargs.get("tools_used"))
        return type("R", (), {
            "verdict": "pass",
            "findings": [],
            "to_payload": lambda self: {"verdict": "pass", "duration_ms": 0},
        })()

    monkeypatch.setattr(cf, "verify_response", fake_verify)

    finalize_assistant_turn(
        raw="answer",
        public="answer",
        config={},
        emit_pipeline_ready=lambda *a: None,
        record_verdict=lambda _p: None,
        tools_used=("read_file",),
    )

    assert captured == [["read_file"]]


def test_tools_used_defaults_to_empty(monkeypatch):
    """Callers that don't pass tools_used see the prior behavior:
    empty tuple emitted, empty list at the verifier callsite."""
    monkeypatch.setenv("MONOLITH_VERIFIER_V1", "0")

    emitted = []
    finalize_assistant_turn(
        raw="hi",
        public="hi",
        config={},
        emit_pipeline_ready=lambda raw, public, tools_used: emitted.append(tools_used),
        record_verdict=lambda _p: None,
    )

    assert emitted == [()]


# ── apply_terminal_correction (M1 consumption-seam apply step) ───────────────

from core.chat_finalize import apply_terminal_correction


class _FakeWidget:
    def __init__(self):
        self.applied = None

    def update_main_text(self, text):
        self.applied = text


def test_apply_correction_rerenders_widget_when_corrected():
    w = _FakeWidget()
    logs = []
    result = apply_terminal_correction(
        corrected="clean answer",
        public="clean answer <think>leak</think>",
        get_widget=lambda: w,
        on_debug=logs.append,
    )
    assert result is True
    assert w.applied == "clean answer"
    assert any("re-rendered answer" in m for m in logs)


def test_apply_correction_noop_when_no_correction():
    w = _FakeWidget()
    result = apply_terminal_correction(
        corrected=None,
        public="unchanged",
        get_widget=lambda: w,
    )
    assert result is False
    assert w.applied is None


def test_apply_correction_noop_when_corrected_equals_public():
    w = _FakeWidget()
    result = apply_terminal_correction(
        corrected="same",
        public="same",
        get_widget=lambda: w,
    )
    assert result is False
    assert w.applied is None


def test_apply_correction_logs_observable_noop_when_widget_missing():
    # The load-bearing case: a correction exists but the widget can't be found.
    # Must be LOGGED, never silent — else M1 ships dark.
    logs = []
    result = apply_terminal_correction(
        corrected="fixed",
        public="broken <think>x</think>",
        get_widget=lambda: None,
        on_debug=logs.append,
    )
    assert result is False
    assert any("no widget" in m for m in logs)


def test_apply_correction_survives_widget_lookup_exception():
    def _boom():
        raise RuntimeError("surface gone")

    logs = []
    result = apply_terminal_correction(
        corrected="fixed",
        public="broken",
        get_widget=_boom,
        on_debug=logs.append,
    )
    assert result is False
    assert any("no widget" in m for m in logs)
