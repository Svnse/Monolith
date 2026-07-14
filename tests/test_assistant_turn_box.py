from __future__ import annotations

from ui.pages.assistant_turn_box import AssistantTurnBox
from ui.pages.chat_session import ChatSessionManager


def _make_box(messages: list[dict] | None = None) -> tuple[AssistantTurnBox, ChatSessionManager]:
    sessions = ChatSessionManager("MASTER", now_iso=lambda: "2026-01-01T00:00:00+00:00")
    sessions.set_current(sessions.create_session(messages=messages or []))
    return AssistantTurnBox(sessions), sessions


def test_start_new_stream_and_append_token_updates_current_assistant() -> None:
    box, sessions = _make_box()

    index = box.start_new_stream()
    box.append_token("Hel")
    box.append_token("lo")

    assert index == 0
    assert box.active_assistant_index == 0
    assert sessions.current["messages"][0]["role"] == "assistant"
    assert sessions.current["messages"][0]["text"] == "Hello"
    assert box.active_assistant_token_count == 2


def test_rewrite_stream_appends_into_existing_assistant() -> None:
    box, sessions = _make_box(
        messages=[
            {"i": 1, "role": "user", "text": "Say hi", "time": "2026-01-01T00:00:00+00:00"},
            {"i": 2, "role": "assistant", "text": "Hi", "time": "2026-01-01T00:00:00+00:00"},
        ]
    )

    box.start_rewrite_stream(1)
    box.append_token(" there")

    assert sessions.current["messages"][1]["text"] == "Hi there"
    assert box.stream_target_index() == 1


def test_edit_from_index_returns_text_without_mutating_session() -> None:
    box, sessions = _make_box(
        messages=[
            {"i": 1, "role": "user", "text": "first", "time": "2026-01-01T00:00:00+00:00"},
            {"i": 2, "role": "assistant", "text": "reply", "time": "2026-01-01T00:00:00+00:00"},
            {"i": 3, "role": "user", "text": "second", "time": "2026-01-01T00:00:00+00:00"},
            {"i": 4, "role": "assistant", "text": "later", "time": "2026-01-01T00:00:00+00:00"},
        ]
    )

    text = box.edit_from_index(2)

    assert text == "second"
    assert [msg["text"] for msg in sessions.current["messages"]] == ["first", "reply", "second", "later"]


def test_commit_edit_from_index_updates_user_message_and_truncates_tail() -> None:
    box, sessions = _make_box(
        messages=[
            {"i": 1, "role": "user", "text": "first", "time": "2026-01-01T00:00:00+00:00"},
            {"i": 2, "role": "assistant", "text": "reply", "time": "2026-01-01T00:00:00+00:00"},
            {"i": 3, "role": "user", "text": "second", "time": "2026-01-01T00:00:00+00:00"},
            {"i": 4, "role": "assistant", "text": "later", "time": "2026-01-01T00:00:00+00:00"},
        ]
    )

    text = box.commit_edit_from_index(2, "second revised")

    assert text == "second revised"
    assert [msg["text"] for msg in sessions.current["messages"]] == ["first", "reply", "second revised"]
    assert [msg["i"] for msg in sessions.current["messages"]] == [1, 2, 3]


def test_regen_last_assistant_returns_latest_user_prompt() -> None:
    box, sessions = _make_box(
        messages=[
            {"i": 1, "role": "user", "text": "Explain it", "time": "2026-01-01T00:00:00+00:00"},
            {"i": 2, "role": "assistant", "text": "Draft", "time": "2026-01-01T00:00:00+00:00"},
        ]
    )

    prompt = box.regen_last_assistant()

    assert prompt == "Explain it"
    assert len(sessions.current["messages"]) == 1
    assert sessions.current["messages"][0]["role"] == "user"


def test_regen_from_index_targets_clicked_assistant() -> None:
    box, sessions = _make_box(
        messages=[
            {"i": 1, "role": "user", "text": "first question", "time": "2026-01-01T00:00:00+00:00"},
            {"i": 2, "role": "assistant", "text": "first reply", "time": "2026-01-01T00:00:00+00:00"},
            {"i": 3, "role": "user", "text": "second question", "time": "2026-01-01T00:00:00+00:00"},
            {"i": 4, "role": "assistant", "text": "second reply", "time": "2026-01-01T00:00:00+00:00"},
        ]
    )

    prompt = box.regen_from_index(1)

    assert prompt == "first question"
    assert [msg["text"] for msg in sessions.current["messages"]] == ["first question"]


def test_cleanup_empty_assistant_removes_placeholder_and_resets_runtime() -> None:
    box, sessions = _make_box()
    box.start_new_stream()

    removed = box.cleanup_empty_assistant_if_needed()

    assert removed is True
    assert sessions.current["messages"] == []
    assert box.active_assistant_index is None
    assert box.rewrite_assistant_index is None
    assert box.active_assistant_token_count == 0


def test_display_stream_splits_thinking_from_answer_across_chunks() -> None:
    box, _sessions = _make_box()
    box.start_new_stream()

    first = box.consume_display_chunk("<thi")
    second = box.consume_display_chunk("nk>plan")
    third = box.consume_display_chunk("</think>Answer")
    final = box.finalize_display_stream()

    assert first.has_changes() is False
    assert second.thinking_opened is True
    assert second.thinking_text == "plan"
    assert third.thinking_closed is True
    assert third.answer_text == "Answer"
    assert final.has_changes() is False


def test_build_display_snapshot_parses_history_text() -> None:
    box, _sessions = _make_box()

    snapshot = box.build_display_snapshot("<think>reason</think>Hello", close_open=True)

    assert snapshot.answer_text == "Hello"
    assert snapshot.thinking_text == "reason"
    assert snapshot.thinking_seen is True
    assert snapshot.thinking_active is False


# ── acatalepsy tag extraction ──────────────────────────────────────────

from ui.pages.assistant_turn_box import AssistantStreamNormalizer


def test_acu_tag_extracted_from_stream() -> None:
    norm = AssistantStreamNormalizer()
    u1 = norm.consume("Hello <acatalepsy>user likes python</acatalepsy> world")
    final = norm.finalize()

    assert norm.answer_text == "Hello  world"
    assert norm.acu_text == "user likes python"
    assert norm.acu_seen is True
    assert norm.acu_active is False


def test_acu_tag_streamed_char_by_char() -> None:
    norm = AssistantStreamNormalizer()
    text = "Hi<acatalepsy>fact</acatalepsy>Bye"
    updates = [norm.consume(ch) for ch in text]
    final = norm.finalize()

    assert norm.answer_text == "HiBye"
    assert norm.acu_text == "fact"
    assert norm.acu_seen is True


def test_acu_and_think_tags_coexist() -> None:
    norm = AssistantStreamNormalizer()
    norm.consume("<think>reasoning</think>Answer<acatalepsy>claim</acatalepsy> done")
    norm.finalize()

    assert norm.thinking_text == "reasoning"
    assert norm.thinking_seen is True
    assert norm.answer_text == "Answer done"
    assert norm.acu_text == "claim"
    assert norm.acu_seen is True


def test_acu_snapshot_includes_acu_fields() -> None:
    norm = AssistantStreamNormalizer.from_text(
        "<think>t</think>answer<acatalepsy>mem</acatalepsy>"
    )
    snap = norm.snapshot()

    assert snap.answer_text == "answer"
    assert snap.thinking_text == "t"
    assert snap.acu_text == "mem"
    assert snap.acu_seen is True


def test_curiosity_tag_hidden_from_answer_but_retained_in_lane() -> None:
    norm = AssistantStreamNormalizer()
    norm.consume("Answer<curiosity>monolith | values | precision</curiosity> tail")
    norm.finalize()

    assert norm.answer_text == "Answer tail"
    assert norm.curiosity_text == "monolith | values | precision"
    assert norm.curiosity_seen is True


def test_unclosed_acu_finalized() -> None:
    norm = AssistantStreamNormalizer()
    norm.consume("text<acatalepsy>partial claim")
    final = norm.finalize(close_open=True)

    assert norm.acu_text == "partial claim"
    assert norm.acu_active is False
    assert final.acu_closed is True


def test_acu_display_update_fields() -> None:
    box, _sessions = _make_box()
    box.start_new_stream()

    u1 = box.consume_display_chunk("<acatalepsy>")
    assert u1.acu_opened is True
    u2 = box.consume_display_chunk("data")
    assert u2.acu_text == "data"
    u3 = box.consume_display_chunk("</acatalepsy>")
    assert u3.acu_closed is True


def test_tool_loop_done_stripped_when_whole_in_one_chunk() -> None:
    norm = AssistantStreamNormalizer()
    norm.consume("Here it is.[TOOL_LOOP_DONE] Final answer.")
    norm.finalize()
    assert "TOOL_LOOP" not in norm.answer_text
    assert norm.answer_text == "Here it is. Final answer."


def test_tool_loop_done_split_across_chunks_never_displays() -> None:
    # The sentinel arrives split across streamed token boundaries — the case
    # the per-chunk replace() missed, leaking the marker to the screen until
    # the post-stream rerender removed it.
    norm = AssistantStreamNormalizer()
    for chunk in ("Done.", " [TOOL_", "LOOP_", "DONE]", " Tail."):
        norm.consume(chunk)
    norm.finalize()
    assert "TOOL_LOOP" not in norm.answer_text
    assert "[TOOL_" not in norm.answer_text
    assert "DONE]" not in norm.answer_text
    # Sentinel sat between two spaces, so both survive (display strip is exact,
    # not whitespace-collapsing) — the point is no marker fragment leaks.
    assert norm.answer_text == "Done.  Tail."


def test_tool_loop_done_split_char_by_char_never_displays() -> None:
    norm = AssistantStreamNormalizer()
    for ch in "ok[TOOL_LOOP_DONE]done":
        norm.consume(ch)
    norm.finalize()
    assert "TOOL_LOOP" not in norm.answer_text
    assert norm.answer_text == "okdone"


def test_trailing_bracket_prefix_flushed_at_finalize() -> None:
    # A trailing run that merely *looks* like the start of the sentinel but
    # never completes is genuine content and must survive finalize.
    norm = AssistantStreamNormalizer()
    norm.consume("price was [TOOL")
    norm.finalize()
    assert norm.answer_text == "price was [TOOL"


def test_sentinel_near_miss_is_not_swallowed() -> None:
    norm = AssistantStreamNormalizer()
    for chunk in ("see [TOOL_", "LOOX] here"):  # diverges at X -> not the sentinel
        norm.consume(chunk)
    norm.finalize()
    assert norm.answer_text == "see [TOOL_LOOX] here"
