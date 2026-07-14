from __future__ import annotations

import json
import time

import pytest

from core import turn_trace as tt
from core import message_interceptors as mi


@pytest.fixture
def trace_db(tmp_path, monkeypatch):
    """Redirect turn_trace store to a temp file for each test."""
    db = tmp_path / "turn_trace.sqlite3"
    tt.set_db_path(db)
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "1")
    yield db
    tt.set_db_path(None)


# ── primitives ──────────────────────────────────────────────────────


def test_hash_is_deterministic_and_truncated() -> None:
    h1 = tt._hash("hello world")
    h2 = tt._hash("hello world")
    assert h1 == h2
    assert len(h1) == 12


def test_preview_truncates_and_normalizes_newlines() -> None:
    long = "abc\ndef\rghi" * 200
    p = tt._preview(long)
    assert len(p) <= 500
    assert "\n" not in p
    assert "\r" not in p


def test_metadata_4kb_cap_replaces_with_marker() -> None:
    big = {"x": "y" * 5000}
    encoded = tt._bound_metadata(big)
    parsed = json.loads(encoded)
    assert parsed.get("_truncated") is True


def test_metadata_cap_preserves_small_fields_drops_oversized_think_block() -> None:
    """Regression: a rating's metadata is {failure_tags, surface_note, think_block}.
    The think_block alone blows the 4096 cap. The OLD behavior nuked the WHOLE
    dict to {_truncated, _orig_bytes}, losing the audit-critical failure_tags +
    surface_note (so trainer-driven ratings showed tagless in outcome_traces).
    Now only the oversized field is dropped; the small ones survive."""
    meta = {
        "failure_tags": ["premise_unchecked"],
        "surface_note": "conceded under one push",
        "think_block": "x" * 8000,
    }
    parsed = json.loads(tt._bound_metadata(meta))
    assert parsed["failure_tags"] == ["premise_unchecked"]        # preserved
    assert parsed["surface_note"] == "conceded under one push"    # preserved
    assert str(parsed["think_block"]).startswith("[dropped:")     # dropped + marked
    assert "think_block" in parsed.get("_truncated_fields", [])
    assert parsed.get("_truncated") is True                       # flag still set


# ── outcome reads (support for the monosearch outcome_traces adapter) ─

def _stub_monothink(monkeypatch) -> None:
    """record_outcome fires the monothink evolution hook on a tagged rating;
    stub it so these read-helper tests stay hermetic (no DeepSeek call)."""
    monkeypatch.setattr(
        "core.monothink.maybe_evolve_after_rating",
        lambda *a, **k: None,
        raising=False,
    )


def test_read_recent_outcomes_newest_first_with_id(trace_db, monkeypatch) -> None:
    _stub_monothink(monkeypatch)
    for i in range(3):
        tt.record_outcome(tt.OutcomeTraceRecord(
            turn_id=f"t{i}", recorded_at=f"2026-06-0{i + 1}T00:00:00+00:00", kind="copy",
        ))
    rows = tt.read_recent_outcomes(10)
    assert [r.turn_id for r in rows] == ["t2", "t1", "t0"]   # newest-first (id DESC)
    assert all(isinstance(r.id, int) and r.id > 0 for r in rows)


def test_read_recent_outcomes_keyword_matches_failure_tag(trace_db, monkeypatch) -> None:
    _stub_monothink(monkeypatch)
    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id="t_bad", recorded_at="2026-06-05T00:00:00+00:00", kind="rating",
        rating_value=48,
        reason="Reasoning-failure(s) flagged — [premise_unchecked] a premise was used "
               "without being compared against the evidence present in the turn.",
        metadata={"failure_tags": ["premise_unchecked"]},
    ))
    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id="t_good", recorded_at="2026-06-05T01:00:00+00:00", kind="rating",
        rating_value=88, reason="", metadata={},
    ))
    hits = tt.read_recent_outcomes(10, keyword="premise")
    assert [r.turn_id for r in hits] == ["t_bad"]            # substring on the raw stored strings
    assert hits[0].metadata["failure_tags"] == ["premise_unchecked"]


def test_read_outcome_by_id_roundtrip(trace_db, monkeypatch) -> None:
    _stub_monothink(monkeypatch)
    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id="t1", recorded_at="2026-06-05T00:00:00+00:00", kind="rating",
        rating_value=50, reason="r", metadata={"failure_tags": ["premise_unchecked"]},
    ))
    one = tt.read_recent_outcomes(1)[0]
    got = tt.read_outcome(one.id)
    assert got is not None
    assert got.id == one.id and got.turn_id == "t1"
    assert tt.read_outcome(999999) is None                   # missing id -> None


def test_stage_record_invariants_kept_with_reason_raises() -> None:
    with pytest.raises(ValueError):
        tt.StageTraceRecord(
            turn_id="t", seq=0, stage_name="x", stage_kind="interceptor",
            entered_at="2026-05-10T00:00:00Z", exited_at="2026-05-10T00:00:01Z",
            outcome="ran", outcome_reason="should not be here",
            messages_in=1, messages_out=1,
        )


def test_stage_record_invariants_dropped_without_reason_raises() -> None:
    with pytest.raises(ValueError):
        tt.StageTraceRecord(
            turn_id="t", seq=0, stage_name="x", stage_kind="interceptor",
            entered_at="2026-05-10T00:00:00Z", exited_at="2026-05-10T00:00:01Z",
            outcome="errored", outcome_reason=None,
            messages_in=1, messages_out=1,
        )


def test_stage_record_time_ordering_enforced() -> None:
    with pytest.raises(ValueError):
        tt.StageTraceRecord(
            turn_id="t", seq=0, stage_name="x", stage_kind="interceptor",
            entered_at="2026-05-10T00:00:01Z", exited_at="2026-05-10T00:00:00Z",
            outcome="ran", outcome_reason=None, messages_in=1, messages_out=1,
        )


def test_drop_reason_serializes_to_string() -> None:
    assert tt.DropReason.DUPLICATE.value == "duplicate"
    # str enum lets json render it as the value
    assert json.dumps(tt.DropReason.DUPLICATE) == '"duplicate"'


# ── store: write + read ────────────────────────────────────────────


def _make_stage(turn_id: str, seq: int, name: str = "demo", outcome: str = "ran") -> tt.StageTraceRecord:
    now = "2026-05-10T00:00:00Z"
    return tt.StageTraceRecord(
        turn_id=turn_id, seq=seq, stage_name=name, stage_kind="interceptor",
        entered_at=now, exited_at=now,
        outcome=outcome,
        outcome_reason=None if outcome == "ran" else "test reason",
        messages_in=1, messages_out=1,
    )


def _make_frame(turn_id: str, parent: str | None = None) -> tt.FrameTraceRecord:
    return tt.FrameTraceRecord(
        turn_id=turn_id, parent_turn_id=parent,
        captured_at="2026-05-10T00:00:00Z",
        backend="gguf_api", engine_key="llm_test", gen_id=1,
        final_messages=(
            tt.FrameMessage.from_message({"role": "system", "content": "sys"}),
            tt.FrameMessage.from_message({"role": "user", "content": "hello"}),
        ),
        system_prompt_chars=3, user_prompt_chars=5, total_chars=8,
        config_snapshot={"max_tokens": 1024, "temp": 1.0},
    )


def test_record_and_get_turn_trace_roundtrip(trace_db) -> None:
    tt.record_stage(_make_stage("turn-A", 0))
    tt.record_stage(_make_stage("turn-A", 1, name="effort"))
    tt.record_frame(_make_frame("turn-A"))

    joined = tt.get_turn_trace("turn-A")
    assert joined is not None
    assert joined.turn_id == "turn-A"
    assert len(joined.stages) == 2
    assert [s.seq for s in joined.stages] == [0, 1]
    assert joined.frame is not None
    assert joined.frame.backend == "gguf_api"
    assert joined.summary["frame_present"] is True
    assert joined.summary["errored_stage_count"] == 0


def test_get_turn_trace_returns_none_for_unknown(trace_db) -> None:
    assert tt.get_turn_trace("does-not-exist") is None


def test_record_frame_replaces_on_duplicate_turn_id(trace_db) -> None:
    f1 = _make_frame("turn-X")
    tt.record_frame(f1)
    # Replace
    f2 = tt.FrameTraceRecord(
        turn_id="turn-X", parent_turn_id=None, captured_at="2026-05-10T00:00:01Z",
        backend="openai", engine_key="llm_test", gen_id=2,
        final_messages=(), system_prompt_chars=0, user_prompt_chars=0, total_chars=0,
    )
    tt.record_frame(f2)
    joined = tt.get_turn_trace("turn-X")
    assert joined is not None
    assert joined.frame.backend == "openai"


def test_list_recent_turns_orders_newest_first(trace_db) -> None:
    tt.record_frame(tt.FrameTraceRecord(
        turn_id="t1", parent_turn_id=None, captured_at="2026-05-10T00:00:01Z",
        backend="x", engine_key="e", gen_id=1, final_messages=(),
        system_prompt_chars=0, user_prompt_chars=0, total_chars=0,
    ))
    tt.record_frame(tt.FrameTraceRecord(
        turn_id="t2", parent_turn_id=None, captured_at="2026-05-10T00:00:02Z",
        backend="x", engine_key="e", gen_id=2, final_messages=(),
        system_prompt_chars=0, user_prompt_chars=0, total_chars=0,
    ))
    tt.record_frame(tt.FrameTraceRecord(
        turn_id="t3", parent_turn_id=None, captured_at="2026-05-10T00:00:03Z",
        backend="x", engine_key="e", gen_id=3, final_messages=(),
        system_prompt_chars=0, user_prompt_chars=0, total_chars=0,
    ))
    rows = tt.list_recent_turns(limit=10)
    assert [r.turn_id for r in rows] == ["t3", "t2", "t1"]


def test_recent_and_search_order_by_insertion_not_wall_clock(trace_db) -> None:
    """When-plane fix: recent/search ordering uses the monotonic autoincrement
    id, not wall-clock captured_at. A turn recorded after a backward clock step
    (earlier captured_at but higher id) must still sort as the most recent."""
    # Inserted FIRST, stamped LATER:
    tt.record_frame(tt.FrameTraceRecord(
        turn_id="t_old", parent_turn_id=None, captured_at="2026-05-10T12:00:00Z",
        backend="x", engine_key="e", gen_id=1, final_messages=(),
        system_prompt_chars=0, user_prompt_chars=0, total_chars=0,
    ))
    # Inserted SECOND (higher id), stamped EARLIER (clock stepped backward):
    tt.record_frame(tt.FrameTraceRecord(
        turn_id="t_new", parent_turn_id=None, captured_at="2026-05-10T11:00:00Z",
        backend="x", engine_key="e", gen_id=2, final_messages=(),
        system_prompt_chars=0, user_prompt_chars=0, total_chars=0,
    ))
    assert [r.turn_id for r in tt.list_recent_turns(limit=10)][0] == "t_new"
    assert [r.turn_id for r in tt.search_turns(limit=10)][0] == "t_new"


def test_search_turns_has_errored_filter(trace_db) -> None:
    tt.record_stage(_make_stage("good", 0))
    tt.record_frame(_make_frame("good"))
    tt.record_stage(_make_stage("bad", 0, outcome="errored"))
    tt.record_frame(_make_frame("bad"))

    errored = tt.search_turns(has_errored_stage=True, limit=10)
    assert [r.turn_id for r in errored] == ["bad"]
    clean = tt.search_turns(has_errored_stage=False, limit=10)
    assert [r.turn_id for r in clean] == ["good"]


def test_cleanup_old_records_drops_old_keeps_new(trace_db, monkeypatch) -> None:
    real_datetime = tt.datetime

    class FixedDateTime(real_datetime):
        @classmethod
        def now(cls, tz=None):
            fixed = real_datetime(2026, 5, 20, tzinfo=tt.timezone.utc)
            return fixed.astimezone(tz) if tz is not None else fixed.replace(tzinfo=None)

    monkeypatch.setattr(tt, "datetime", FixedDateTime)

    # Both frame and stage from "ancient" carry the same old timestamp.
    tt.record_frame(tt.FrameTraceRecord(
        turn_id="ancient", parent_turn_id=None,
        captured_at="2020-01-01T00:00:00Z",
        backend="x", engine_key="e", gen_id=1, final_messages=(),
        system_prompt_chars=0, user_prompt_chars=0, total_chars=0,
    ))
    tt.record_stage(tt.StageTraceRecord(
        turn_id="ancient", seq=0, stage_name="x", stage_kind="interceptor",
        entered_at="2020-01-01T00:00:00Z", exited_at="2020-01-01T00:00:01Z",
        outcome="ran", outcome_reason=None, messages_in=1, messages_out=1,
    ))
    tt.record_frame(_make_frame("recent"))
    tt.record_stage(_make_stage("recent", 0))

    counts = tt.cleanup_old_records(ttl_days=30)
    assert counts["frame"] >= 1
    assert counts["stage"] >= 1
    assert tt.get_turn_trace("ancient") is None
    assert tt.get_turn_trace("recent") is not None


# ── flag off ───────────────────────────────────────────────────────


def test_record_stage_no_op_when_flag_off(monkeypatch, trace_db) -> None:
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "0")
    tt.record_stage(_make_stage("turn-Z", 0))
    tt.record_frame(_make_frame("turn-Z"))
    assert tt.get_turn_trace("turn-Z") is None


# ── apply_interceptors emits stage records ─────────────────────────


def _identity_interceptor(messages, config):
    return None  # skipped


def _injecting_interceptor(messages, config):
    out = list(messages)
    out.insert(0, {"role": "user", "content": "[INJECTED]", "ephemeral": True, "source": "test"})
    return out


def _erroring_interceptor(messages, config):
    raise RuntimeError("kaboom")


def test_apply_interceptors_emits_skipped_record(trace_db) -> None:
    mi.clear_interceptors()
    mi.register_interceptor(_identity_interceptor)
    config = {"_turn_id": "turn-skip"}
    msgs = [{"role": "user", "content": "hi"}]
    out = mi.apply_interceptors(msgs, config)
    assert out == msgs
    joined = tt.get_turn_trace("turn-skip")
    assert joined is not None
    assert len(joined.stages) == 1
    assert joined.stages[0].outcome == "skipped"
    assert joined.stages[0].outcome_reason == "returned None"
    mi.clear_interceptors()


def test_apply_interceptors_emits_ran_record_with_added_items(trace_db) -> None:
    mi.clear_interceptors()
    mi.register_interceptor(_injecting_interceptor)
    config = {"_turn_id": "turn-inject"}
    msgs = [{"role": "user", "content": "hi"}]
    out = mi.apply_interceptors(msgs, config)
    assert len(out) == 2  # injected + original
    joined = tt.get_turn_trace("turn-inject")
    assert joined is not None
    s = joined.stages[0]
    assert s.outcome == "ran"
    assert s.messages_in == 1
    assert s.messages_out == 2
    assert len(s.items_added) == 1
    assert s.items_added[0].source == "test"
    mi.clear_interceptors()


def test_apply_interceptors_records_errored_does_not_break(trace_db) -> None:
    mi.clear_interceptors()
    mi.register_interceptor(_erroring_interceptor)
    mi.register_interceptor(_injecting_interceptor)
    config = {"_turn_id": "turn-err"}
    msgs = [{"role": "user", "content": "hi"}]
    out = mi.apply_interceptors(msgs, config)
    # Erroring interceptor was skipped; injector still ran.
    assert len(out) == 2
    joined = tt.get_turn_trace("turn-err")
    assert joined is not None
    assert joined.stages[0].outcome == "errored"
    assert "kaboom" in (joined.stages[0].outcome_reason or "")
    assert joined.stages[1].outcome == "ran"
    mi.clear_interceptors()


def test_apply_interceptors_no_trace_when_no_turn_id(trace_db) -> None:
    """Without _turn_id in config, interceptors run normally but no records emit."""
    mi.clear_interceptors()
    mi.register_interceptor(_injecting_interceptor)
    msgs = [{"role": "user", "content": "hi"}]
    out = mi.apply_interceptors(msgs, {})  # no _turn_id
    assert len(out) == 2
    # No turn was recorded.
    rows = tt.list_recent_turns(limit=5)
    assert rows == []
    mi.clear_interceptors()


def test_parent_turn_id_propagation(trace_db) -> None:
    mi.clear_interceptors()
    mi.register_interceptor(_injecting_interceptor)
    config = {"_turn_id": "child", "_parent_turn_id": "parent"}
    mi.apply_interceptors([{"role": "user", "content": "x"}], config)
    joined = tt.get_turn_trace("child")
    assert joined is not None
    assert joined.stages[0].parent_turn_id == "parent"
    mi.clear_interceptors()


# ── effort_tier on FrameTraceRecord ────────────────────────────────


def test_frame_records_effort_tier_roundtrip(trace_db) -> None:
    """effort_tier persists through write + read."""
    f = tt.FrameTraceRecord(
        turn_id="turn-eff", parent_turn_id=None,
        captured_at="2026-05-10T00:00:00Z",
        backend="gguf_api", engine_key="llm_test", gen_id=1,
        final_messages=(),
        system_prompt_chars=0, user_prompt_chars=0, total_chars=0,
        effort_tier="ultimate",
    )
    tt.record_frame(f)
    joined = tt.get_turn_trace("turn-eff")
    assert joined is not None
    assert joined.frame is not None
    assert joined.frame.effort_tier == "ultimate"
    assert joined.summary["effort_tier"] == "ultimate"


def test_frame_effort_tier_defaults_to_none(trace_db) -> None:
    """A frame without explicit effort_tier records None and survives roundtrip."""
    tt.record_frame(_make_frame("turn-no-eff"))
    joined = tt.get_turn_trace("turn-no-eff")
    assert joined is not None
    assert joined.frame is not None
    assert joined.frame.effort_tier is None
    assert joined.summary["effort_tier"] is None


# ── Layer D: OutcomeTraceRecord ────────────────────────────────────


def test_outcome_invariants_unknown_kind_raises() -> None:
    with pytest.raises(ValueError):
        tt.OutcomeTraceRecord(
            turn_id="t", recorded_at="2026-05-10T00:00:00Z",
            kind="unknown_kind",
        )


def test_outcome_invariants_rating_requires_value() -> None:
    with pytest.raises(ValueError):
        tt.OutcomeTraceRecord(
            turn_id="t", recorded_at="2026-05-10T00:00:00Z",
            kind="rating",
        )


def test_outcome_invariants_rating_value_range() -> None:
    with pytest.raises(ValueError):
        tt.OutcomeTraceRecord(
            turn_id="t", recorded_at="2026-05-10T00:00:00Z",
            kind="rating", rating_value=150,
        )
    with pytest.raises(ValueError):
        tt.OutcomeTraceRecord(
            turn_id="t", recorded_at="2026-05-10T00:00:00Z",
            kind="rating", rating_value=-1,
        )


def test_outcome_invariants_non_rating_must_not_carry_value() -> None:
    with pytest.raises(ValueError):
        tt.OutcomeTraceRecord(
            turn_id="t", recorded_at="2026-05-10T00:00:00Z",
            kind="thumbs_up", rating_value=80,
        )


def test_outcome_kind_serializes_to_string() -> None:
    assert tt.OutcomeKind.THUMBS_UP.value == "thumbs_up"
    assert tt.OutcomeKind.RATING.value == "rating"


def test_record_outcome_and_get_turn_trace_includes_outcomes(trace_db) -> None:
    tt.record_frame(_make_frame("turn-out"))
    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id="turn-out", recorded_at="2026-05-10T00:00:01Z",
        kind="thumbs_up",
    ))
    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id="turn-out", recorded_at="2026-05-10T00:00:02Z",
        kind="rating", rating_value=85, reason="missed an edge case",
    ))
    joined = tt.get_turn_trace("turn-out")
    assert joined is not None
    assert len(joined.outcomes) == 2
    assert joined.outcomes[0].kind == "thumbs_up"
    assert joined.outcomes[1].kind == "rating"
    assert joined.outcomes[1].rating_value == 85
    assert joined.outcomes[1].reason == "missed an edge case"
    assert joined.summary["outcome_count"] == 2
    assert joined.summary["last_rating"] == 85


def test_record_outcome_works_without_frame(trace_db) -> None:
    """An outcome can land before the frame is persisted (race window)."""
    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id="turn-orphan", recorded_at="2026-05-10T00:00:00Z",
        kind="copy",
    ))
    joined = tt.get_turn_trace("turn-orphan")
    assert joined is not None
    assert joined.frame is None
    assert len(joined.outcomes) == 1
    assert joined.outcomes[0].kind == "copy"


def test_list_outcomes_for_turn_returns_in_order(trace_db) -> None:
    tt.record_frame(_make_frame("turn-multi"))
    for i, kind in enumerate(["thumbs_up", "copy", "regen", "thumbs_down"]):
        tt.record_outcome(tt.OutcomeTraceRecord(
            turn_id="turn-multi",
            recorded_at=f"2026-05-10T00:00:0{i}Z",
            kind=kind,
        ))
    outs = tt.list_outcomes_for_turn("turn-multi")
    assert [o.kind for o in outs] == ["thumbs_up", "copy", "regen", "thumbs_down"]


def test_outcome_record_no_op_when_flag_off(monkeypatch, trace_db) -> None:
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "0")
    tt.record_frame(_make_frame("turn-flag"))
    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id="turn-flag", recorded_at="2026-05-10T00:00:00Z",
        kind="thumbs_up",
    ))
    # Frame won't have been written either with flag off
    assert tt.get_turn_trace("turn-flag") is None


def test_cleanup_old_records_drops_outcomes(trace_db) -> None:
    tt.record_frame(tt.FrameTraceRecord(
        turn_id="turn-old", parent_turn_id=None,
        captured_at="2020-01-01T00:00:00Z",
        backend="x", engine_key="e", gen_id=1, final_messages=(),
        system_prompt_chars=0, user_prompt_chars=0, total_chars=0,
    ))
    tt.record_outcome(tt.OutcomeTraceRecord(
        turn_id="turn-old", recorded_at="2020-01-01T00:00:00Z",
        kind="thumbs_down",
    ))
    counts = tt.cleanup_old_records(ttl_days=30)
    assert counts["outcome"] >= 1
    assert tt.get_turn_trace("turn-old") is None
