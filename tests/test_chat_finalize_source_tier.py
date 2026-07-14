import pytest

from core import turn_trace as tt
import core.chat_finalize as cf


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = tmp_path / "turn_trace.sqlite3"
    tt.set_db_path(db)
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "1")
    monkeypatch.setenv("MONOLITH_SOURCE_TIER_V1", "1")
    monkeypatch.setenv("MONOLITH_VERIFIER_V1", "0")  # skip the real verifier
    yield
    tt.set_db_path(None)


def _frame(turn_id):
    tt.record_frame(tt.FrameTraceRecord(
        turn_id=turn_id, parent_turn_id=None, captured_at="2026-06-07T00:00:00+00:00",
        backend="gguf", engine_key="llm", gen_id=1, final_messages=(),
        system_prompt_chars=0, user_prompt_chars=0, total_chars=0, metadata={},
    ))


def _finalize(raw, public, turn_id):
    cf.finalize_assistant_turn(
        raw=raw, public=public, config={"_turn_id": turn_id},
        emit_pipeline_ready=lambda r, p, t: None,
        record_verdict=lambda payload: None,
    )


def test_finalize_stamps_generation_tier_on_bare_answer(env):
    _frame("t-gen")
    _finalize("just an assertion", "just an assertion", "t-gen")
    assert tt.get_turn_trace("t-gen").frame.metadata["source_tier"] == "generation"


def test_finalize_stamps_tool_tier_when_tool_tag_present(env):
    _frame("t-tool")
    _finalize("<tool_call>{}</tool_call> done", "done", "t-tool")
    assert tt.get_turn_trace("t-tool").frame.metadata["source_tier"] == "tool"


def test_finalize_does_not_stamp_when_flag_off(env, monkeypatch):
    _frame("t-off")
    monkeypatch.setenv("MONOLITH_SOURCE_TIER_V1", "0")
    _finalize("<tool_call>{}</tool_call>", "", "t-off")
    assert "source_tier" not in tt.get_turn_trace("t-off").frame.metadata


def test_finalize_no_turn_id_is_safe(env):
    # missing _turn_id → best-effort skip, no exception
    cf.finalize_assistant_turn(
        raw="answer", public="answer", config={},
        emit_pipeline_ready=lambda r, p, t: None, record_verdict=lambda payload: None,
    )


def test_finalize_uses_config_exchange_tools_when_raw_has_no_tag(env):
    # The LIVE path: a terminal synthesis turn has NO <tool_call> tag, but the
    # exchange used tools — chat.py stashes that on config["_source_tier_tools"].
    # Without this, every real tool-using exchange would mislabel as generation.
    _frame("t-exch")
    cf.finalize_assistant_turn(
        raw="here is my synthesis of the results", public="here is my synthesis of the results",
        config={"_turn_id": "t-exch", "_source_tier_tools": ("tool",)},
        emit_pipeline_ready=lambda r, p, t: None, record_verdict=lambda payload: None,
    )
    assert tt.get_turn_trace("t-exch").frame.metadata["source_tier"] == "tool"
