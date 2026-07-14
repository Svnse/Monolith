import pytest

from core import turn_trace as tt
import core.chat_finalize as cf


@pytest.fixture
def env(tmp_path, monkeypatch):
    db = tmp_path / "turn_trace.sqlite3"
    tt.set_db_path(db)
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "1")
    monkeypatch.setenv("MONOLITH_SOURCE_TIER_V1", "1")
    monkeypatch.setenv("MONOLITH_VERIFIER_V1", "0")
    yield
    tt.set_db_path(None)


CASES = [
    ("t-1", "<tool_call>{}</tool_call> ok", "ok", "tool"),
    ("t-2", "<think>plan</think> bare answer", "bare answer", "generation"),
    ("t-3", "no tags here", "no tags here", "generation"),
    ("t-4", "<think>p</think><tool_call>{}</tool_call> a", "a", "tool"),
]


def _run_all():
    for turn_id, raw, public, _ in CASES:
        tt.record_frame(tt.FrameTraceRecord(
            turn_id=turn_id, parent_turn_id=None, captured_at="2026-06-07T00:00:00+00:00",
            backend="gguf", engine_key="llm", gen_id=1, final_messages=(),
            system_prompt_chars=0, user_prompt_chars=0, total_chars=0, metadata={},
        ))
        cf.finalize_assistant_turn(
            raw=raw, public=public, config={"_turn_id": turn_id},
            emit_pipeline_ready=lambda r, p, t: None, record_verdict=lambda p: None,
        )


def test_e2e_each_case_persists_expected_tier(env):
    _run_all()
    tiers = {tid: tt.get_turn_trace(tid).frame.metadata.get("source_tier") for tid, *_ in CASES}
    assert tiers == {"t-1": "tool", "t-2": "generation", "t-3": "generation", "t-4": "tool"}


def test_e2e_distribution_invariant_tool_implies_had_tool(env):
    # Every persisted 'tool' tier must have a tool region recorded — the
    # internal-consistency invariant the offline report checks on live data.
    _run_all()
    for turn_id, *_ in CASES:
        meta = tt.get_turn_trace(turn_id).frame.metadata
        if meta.get("source_tier") == "tool":
            assert meta.get("region_tiers", {}).get("tool") == "tool"
