import pytest

from core import turn_trace as tt


@pytest.fixture
def trace_db(tmp_path, monkeypatch):
    """Redirect turn_trace to a temp DB; feature flag on. Mirrors test_turn_trace.py."""
    db = tmp_path / "turn_trace.sqlite3"
    tt.set_db_path(db)
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "1")
    monkeypatch.setenv("MONOLITH_SOURCE_TIER_V1", "1")
    yield db
    tt.set_db_path(None)


def _write_frame(turn_id):
    tt.record_frame(tt.FrameTraceRecord(
        turn_id=turn_id, parent_turn_id=None, captured_at="2026-06-07T00:00:00+00:00",
        backend="gguf", engine_key="llm", gen_id=1, final_messages=(),
        system_prompt_chars=0, user_prompt_chars=0, total_chars=0,
        metadata={"model_loaded": True},
    ))


def test_record_source_tier_updates_existing_frame_metadata(trace_db):
    _write_frame("turn-1")
    tt.record_source_tier("turn-1", "generation", {"answer": "generation", "trace": "generation"})
    joined = tt.get_turn_trace("turn-1")
    assert joined.frame.metadata["source_tier"] == "generation"
    assert joined.frame.metadata["region_tiers"]["answer"] == "generation"
    # existing metadata preserved
    assert joined.frame.metadata["model_loaded"] is True


def test_record_source_tier_noop_when_feature_flag_off(trace_db, monkeypatch):
    _write_frame("turn-2")
    monkeypatch.setenv("MONOLITH_SOURCE_TIER_V1", "0")
    tt.record_source_tier("turn-2", "tool", {"answer": "tool"})
    joined = tt.get_turn_trace("turn-2")
    assert "source_tier" not in joined.frame.metadata


def test_record_source_tier_noop_when_frame_absent(trace_db):
    # no frame written for this id → best-effort skip, no exception
    tt.record_source_tier("missing-turn", "tool", {"answer": "tool"})
    assert tt.get_turn_trace("missing-turn") is None
