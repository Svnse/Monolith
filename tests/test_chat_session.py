from __future__ import annotations

from ui.pages.chat_session import ChatSessionManager


def test_build_engine_history_skips_tool_calls_and_maps_tool_results() -> None:
    sessions = ChatSessionManager("MASTER")
    sessions.set_current(
        sessions.create_session(
            messages=[
                {"role": "user", "text": "Read the file", "time": "2026-01-01T00:00:00"},
                {"role": "tool_call", "text": '{"tool":"read_file","path":"C:/tmp/a.txt"}', "time": "2026-01-01T00:00:01"},
                {"role": "tool_result", "text": '{"tool":"read_file","result":"[read_file: a.txt]\\nhello"}', "time": "2026-01-01T00:00:02"},
                {"role": "assistant", "text": "The file says hello.", "time": "2026-01-01T00:00:03"},
            ]
        )
    )

    history = sessions.build_engine_history()

    assert history[0] == {"role": "system", "content": "MASTER"}
    # User messages now carry [CHANNEL: USER] prefix
    user_msgs = [item for item in history if item["role"] == "user"]
    assert any("Read the file" in item["content"] for item in user_msgs)
    assert any("[CHANNEL: USER]" in item["content"] for item in user_msgs)
    # Assistant messages carry [CHANNEL: ASSISTANT] prefix
    asst_msgs = [item for item in history if item["role"] == "assistant"]
    assert any("The file says hello." in item["content"] for item in asst_msgs)
    # tool_call rows are excluded
    assert all(item["role"] != "tool_call" for item in history)
    # tool_result rows mapped to [TOOL RESULT:...]
    assert any(item["content"].startswith("[TOOL RESULT:read_file]") for item in history if item["role"] == "user")


def test_build_engine_history_rewrites_system_messages_as_ui_event() -> None:
    sessions = ChatSessionManager("MASTER")
    sessions.set_current(
        sessions.create_session(
            messages=[
                {"role": "user", "text": "Hello", "time": "2026-01-01T00:00:00"},
                {"role": "system", "text": "Session reset", "time": "2026-01-01T00:00:01"},
                {"role": "user", "text": "Continue", "time": "2026-01-01T00:00:02"},
            ]
        )
    )

    history = sessions.build_engine_history()

    assert {"role": "user", "content": "[UI_EVENT]\nSession reset"} in history
    user_contents = [item["content"] for item in history if item["role"] == "user"]
    assert not any(content.startswith("[TOOL RESULT]") for content in user_contents)


def test_build_engine_history_strips_think_blocks_from_assistant_replay() -> None:
    sessions = ChatSessionManager("MASTER")
    sessions.set_current(
        sessions.create_session(
            messages=[
                {"role": "user", "text": "What is 2+2?", "time": "2026-01-01T00:00:00"},
                {
                    "role": "assistant",
                    "text": "<think>computing 2+2 = 4</think>4",
                    "time": "2026-01-01T00:00:01",
                },
                {"role": "user", "text": "And 3+3?", "time": "2026-01-01T00:00:02"},
                {
                    "role": "assistant",
                    "text": "<think>\nmulti-line\nreasoning\n</think>\n6",
                    "time": "2026-01-01T00:00:03"
                },
            ]
        )
    )

    history = sessions.build_engine_history()

    assistant_contents = [item["content"] for item in history if item["role"] == "assistant"]
    # Channel tags are prepended but think blocks are stripped from the actual content
    for c in assistant_contents:
        assert "<think>" not in c and "</think>" not in c
    # The actual answer text should be present after the channel tag
    assert any("4" in c for c in assistant_contents)
    assert any("6" in c for c in assistant_contents)


def test_build_engine_history_keeps_workshop_trace_attachment_for_the_model() -> None:
    # Feature 1 load-bearing check: a workshop answer is stored with a hidden
    # [ATTACHED: workshop trace] block. The DISPLAY path strips it (split_attached), but
    # build_engine_history must NOT — the whole point is that the next turn's model context
    # carries the per-block trace. Guards against the silent no-op where it gets stripped
    # symmetrically with display (then "added to CTX" would do nothing).
    stored = (
        "FINAL ANSWER\n\n"
        "[ATTACHED: workshop trace (42 chars, trace)]\n"
        "- Axioms: extracted the hard constraints\n"
        "[/ATTACHED]"
    )
    sessions = ChatSessionManager("MASTER")
    sessions.set_current(sessions.create_session(messages=[
        {"role": "user", "text": "run the forge", "time": "2026-01-01T00:00:00"},
        {"role": "assistant", "text": stored, "origin": "pipeline", "time": "2026-01-01T00:00:01"},
    ]))

    history = sessions.build_engine_history()
    asst = next(item for item in history if item["role"] == "assistant")

    assert "FINAL ANSWER" in asst["content"]
    assert "[ATTACHED: workshop trace" in asst["content"]                  # block survives into history
    assert "- Axioms: extracted the hard constraints" in asst["content"]  # ...with the per-block trace
