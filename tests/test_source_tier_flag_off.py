import pytest

from core import turn_trace as tt
import core.chat_finalize as cf
import core.monosearch.adapters.turn_trace as ad


@pytest.fixture
def trace_db(tmp_path, monkeypatch):
    db = tmp_path / "turn_trace.sqlite3"
    tt.set_db_path(db)
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "1")
    monkeypatch.delenv("MONOLITH_SOURCE_TIER_V1", raising=False)  # default OFF
    monkeypatch.setenv("MONOLITH_VERIFIER_V1", "0")
    yield db
    tt.set_db_path(None)


def test_flag_off_finalize_writes_no_tier(trace_db):
    tt.record_frame(tt.FrameTraceRecord(
        turn_id="off-1", parent_turn_id=None, captured_at="2026-06-07T00:00:00+00:00",
        backend="gguf", engine_key="llm", gen_id=1, final_messages=(),
        system_prompt_chars=0, user_prompt_chars=0, total_chars=0, metadata={"model_loaded": True},
    ))
    cf.finalize_assistant_turn(
        raw="<tool_call>{}</tool_call>", public="", config={"_turn_id": "off-1"},
        emit_pipeline_ready=lambda r, p, t: None, record_verdict=lambda p: None,
    )
    meta = tt.get_turn_trace("off-1").frame.metadata
    assert "source_tier" not in meta and "region_tiers" not in meta
    assert meta["model_loaded"] is True  # untouched


def test_flag_off_adapter_omits_tier(monkeypatch):
    monkeypatch.delenv("MONOLITH_SOURCE_TIER_V1", raising=False)
    frame = tt.FrameTraceRecord(
        turn_id="off-2", parent_turn_id=None, captured_at="2026-06-07T00:00:00+00:00",
        backend="gguf", engine_key="llm", gen_id=1, final_messages=(),
        system_prompt_chars=0, user_prompt_chars=0, total_chars=0,
        metadata={"source_tier": "generation"},  # stale data from a prior on-period
    )
    joined = tt.TurnTraceJoined(
        turn_id="off-2", parent_turn_id=None, stages=(), frame=frame, outcomes=(), summary={},
    )
    rec = ad.TurnTraceAdapter()._to_record(joined)
    assert "source_tier" not in rec.metadata  # flag off → not surfaced even if persisted
