"""Channel-boundary fixes (2026-06-11).

Three confirmed conflicts from turn_trace frame evidence on the DeepSeek
cloud backend (which merges consecutive user-role messages server-side):

  1. The GENERATING user turn was sent untagged — history turns carry
     [CHANNEL: ...] (added by build_engine_history at replay) but the live
     turn appended in LLMEngine.generate had no header, so the model saw
     injected ephemeral lanes and the user's words as one undifferentiated
     blob.
  2. monothink_interceptor injected the raw scaffold with no envelope —
     the only ephemeral lane without a bracket header, so it read as the
     user's own document after the server-side merge.
  3. build_engine_history rewrote command_block receipts (role=system,
     kind=command_block) as [UI_EVENT] user messages, despite
     _emit_command_block's documented contract that they are filtered out
     of model history.
"""
from __future__ import annotations

import sys
from types import SimpleNamespace

from PySide6.QtWidgets import QApplication

from core.state import AppState
from engine.llm import LLMEngine
from ui.pages.chat_session import ChatSessionManager


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _make_engine(monkeypatch) -> LLMEngine:
    import engine.llm as ellm
    import core.turn_trace as tt
    monkeypatch.setattr(ellm, "apply_interceptors", lambda messages, config: messages)
    monkeypatch.setattr(tt, "record_frame", lambda *a, **k: None)
    engine = LLMEngine(AppState())
    engine.backend = "gguf"
    engine.model_loaded = True
    engine.conversation_history = [
        {"role": "system", "content": "You are Monolith."},
    ]
    return engine


# ── fix 1: generating turn carries a [CHANNEL: USER, ...] tag ────────────


def test_generating_turn_is_channel_tagged(monkeypatch):
    _app()
    engine = _make_engine(monkeypatch)
    try:
        captured: list = []
        engine.sig_gguf_generate.connect(lambda payload: captured.append(payload))
        engine.generate({
            "prompt": "what is the issue?",
            "config": {"system_prompt": "You are Monolith."},
        })
        assert captured, "generate() did not dispatch a payload"
        last_user = [m for m in captured[-1]["messages"] if m["role"] == "user"][-1]
        assert last_user["content"].startswith("[CHANNEL: USER"), last_user["content"]
        assert "what is the issue?" in last_user["content"]
    finally:
        engine.shutdown()


def test_pretagged_connect_prompt_is_not_double_tagged(monkeypatch):
    _app()
    engine = _make_engine(monkeypatch)
    try:
        captured: list = []
        engine.sig_gguf_generate.connect(lambda payload: captured.append(payload))
        prompt = "[CHANNEL: connect/Codex, mcp send_message]\n\nping"
        engine.generate({
            "prompt": prompt,
            "config": {"system_prompt": "You are Monolith."},
        })
        last_user = [m for m in captured[-1]["messages"] if m["role"] == "user"][-1]
        assert last_user["content"] == prompt
        assert last_user["content"].count("[CHANNEL:") == 1
    finally:
        engine.shutdown()


def test_duplicate_send_guard_survives_tagging(monkeypatch):
    """Double-click resend: the stored turn now carries a tag the raw
    prompt lacks — the dup guard must compare tag-stripped bodies."""
    _app()
    engine = _make_engine(monkeypatch)
    try:
        captured: list = []
        engine.sig_gguf_generate.connect(lambda payload: captured.append(payload))
        for _ in range(2):
            engine.generate({
                "prompt": "same question",
                "config": {"system_prompt": "You are Monolith."},
            })
        user_msgs = [
            m for m in captured[-1]["messages"]
            if m["role"] == "user" and "same question" in m["content"]
        ]
        assert len(user_msgs) == 1, user_msgs
    finally:
        engine.shutdown()


def test_ephemeral_followup_is_not_tagged(monkeypatch):
    """Tool-followup messages are scaffold, not user turns — no tag."""
    _app()
    engine = _make_engine(monkeypatch)
    try:
        engine.conversation_history.append({"role": "user", "content": "outer"})
        captured: list = []
        engine.sig_gguf_generate.connect(lambda payload: captured.append(payload))
        engine.generate({
            "prompt": "[TOOL RESULT:run_command]\nexit_code=0",
            "ephemeral": True,
            "config": {"system_prompt": "You are Monolith."},
        })
        tool_msg = next(
            m for m in captured[-1]["messages"]
            if str(m.get("content", "")).startswith("[TOOL RESULT:run_command]")
        )
        assert not tool_msg["content"].startswith("[CHANNEL:")
    finally:
        engine.shutdown()


# ── fix 2: monothink scaffold injected with an envelope header ───────────


def test_monothink_injection_carries_envelope(monkeypatch):
    import core.monothink as mt
    monkeypatch.setattr(mt, "read_scaffold", lambda: "# MonoThink — origin 0\n\nbody")
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hello"},
    ]
    result = mt.monothink_interceptor(messages, {"force_monothink": True})
    assert result is not None
    injected = next(m for m in result if m.get("source") == "monothink")
    first_line = injected["content"].splitlines()[0]
    assert first_line.startswith("[MONOTHINK]"), first_line
    assert "# MonoThink — origin 0" in injected["content"]
    # role/keys contract unchanged
    assert injected["role"] == "user"
    assert injected["ephemeral"] is True


def test_monothink_injection_dedup_unchanged(monkeypatch):
    import core.monothink as mt
    monkeypatch.setattr(mt, "read_scaffold", lambda: "body")
    messages = [
        {"role": "user", "content": "x", "source": "monothink"},
        {"role": "user", "content": "hello"},
    ]
    assert mt.monothink_interceptor(messages, {"force_monothink": True}) is None


# ── fix 3: command_block receipts stay out of engine history ─────────────


def test_command_block_receipts_excluded_from_engine_history():
    sessions = ChatSessionManager("MASTER")
    sessions.set_current(
        sessions.create_session(
            messages=[
                {"role": "user", "text": "Hello", "time": "2026-01-01T00:00:00"},
                {
                    "role": "system",
                    "text": "/rating  →  87  (no #tag — not trained)",
                    "kind": "command_block",
                    "time": "2026-01-01T00:00:01",
                },
                {"role": "user", "text": "Continue", "time": "2026-01-01T00:00:02"},
            ]
        )
    )
    history = sessions.build_engine_history()
    assert not any("/rating" in item["content"] for item in history)


def test_generic_system_messages_still_become_ui_events():
    sessions = ChatSessionManager("MASTER")
    sessions.set_current(
        sessions.create_session(
            messages=[
                {"role": "user", "text": "Hello", "time": "2026-01-01T00:00:00"},
                {"role": "system", "text": "Session reset", "time": "2026-01-01T00:00:01"},
            ]
        )
    )
    history = sessions.build_engine_history()
    assert {"role": "user", "content": "[UI_EVENT]\nSession reset"} in history


# ── fix 4: replayed history must NOT carry generating-turn mode fields ───
# Regression for the 2026-06-19 "E." duplicate-greeting bug: build_engine_history
# tagged its LAST history user turn with include_modes=True, but the live turn is
# tagged separately by LLMEngine.generate. So two user turns carried
# `monothink=on` (the "this is the turn to answer" marker), and right after a
# one-exchange history the mis-tagged one was the *previous* message — the model
# re-answered it. channel_tag.py's contract: modes appear ONLY on the generating
# turn; replayed history shows the role token only.


def test_replayed_history_user_turns_carry_no_mode_fields():
    import core.channel_tag as ct
    orig = ct._peek_monothink
    ct._peek_monothink = lambda: True  # force a mode field to be available
    try:
        sessions = ChatSessionManager("MASTER")
        sessions.set_current(
            sessions.create_session(
                messages=[
                    {"role": "user", "text": "I greet monolith", "time": "2026-01-01T00:00:00"},
                    {"role": "assistant", "text": "E.", "time": "2026-01-01T00:00:01"},
                    {"role": "user", "text": "I read your think block", "time": "2026-01-01T00:00:02"},
                ]
            )
        )
        history = sessions.build_engine_history()
        offenders = [
            h["content"][:70]
            for h in history
            if h.get("role") == "user"
            and str(h.get("content", "")).lstrip().startswith("[CHANNEL: USER")
            and ("monothink=" in h["content"] or "prompts=" in h["content"])
        ]
        assert not offenders, f"replayed history user turn carries generating-turn modes: {offenders}"
    finally:
        ct._peek_monothink = orig
