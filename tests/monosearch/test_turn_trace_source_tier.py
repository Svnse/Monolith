import core.turn_trace as _tt
import core.monosearch.adapters.turn_trace as ad


def _make_joined(metadata):
    frame = _tt.FrameTraceRecord(
        turn_id="t-1", parent_turn_id=None, captured_at="2026-06-07T00:00:00+00:00",
        backend="gguf", engine_key="llm", gen_id=1, final_messages=(),
        system_prompt_chars=0, user_prompt_chars=0, total_chars=0, metadata=metadata,
    )
    return _tt.TurnTraceJoined(
        turn_id="t-1", parent_turn_id=None, stages=(), frame=frame, outcomes=(), summary={},
    )


def test_adapter_surfaces_source_tier_when_flag_on(monkeypatch):
    monkeypatch.setenv("MONOLITH_SOURCE_TIER_V1", "1")
    rec = ad.TurnTraceAdapter()._to_record(
        _make_joined({"source_tier": "generation", "region_tiers": {"answer": "generation"}})
    )
    assert rec.metadata["source_tier"] == "generation"
    assert rec.metadata["region_tiers"]["answer"] == "generation"


def test_adapter_omits_source_tier_when_flag_off(monkeypatch):
    monkeypatch.setenv("MONOLITH_SOURCE_TIER_V1", "0")
    rec = ad.TurnTraceAdapter()._to_record(_make_joined({"source_tier": "generation"}))
    assert "source_tier" not in rec.metadata
