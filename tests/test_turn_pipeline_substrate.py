"""Phase 1 substrate tests for the Turn Pipeline.

Scope: the bus skeleton, event dataclasses, Layer E persistence in
turn_trace.sqlite3, registry validation. No policies, no producers, no
real stream processing — those land in Phase 2+.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from core import pipeline_registry as reg
from core import turn_pipeline_events as ev
from core import turn_trace as tt
from monokernel import turn_pipeline as tp


@pytest.fixture
def pipeline_db(tmp_path, monkeypatch):
    """Isolated turn_trace store + fresh pipeline singleton per test."""
    db = tmp_path / "turn_trace.sqlite3"
    tt.set_db_path(db)
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "1")
    tp.reset_for_tests()
    yield db
    tt.set_db_path(None)
    tp.reset_for_tests()


# ── event taxonomy ──────────────────────────────────────────────────


def test_stamp_event_assigns_identity() -> None:
    raw = ev.StreamChunkReceivedEvent(text="hello", lane_hint="answer")
    stamped = ev.stamp_event(
        raw,
        turn_id="t1", parent_turn_id=None, seq=7,
        emitted_at="2026-05-13T00:00:00+00:00",
        source_kind="kernel", source_name="test",
    )
    assert stamped.turn_id == "t1"
    assert stamped.seq == 7
    assert stamped.source_kind == "kernel"
    assert stamped.text == "hello"  # subclass field survives
    assert stamped.kind == "StreamChunkReceivedEvent"


def test_event_to_payload_includes_kind_and_subclass_fields() -> None:
    e = ev.stamp_event(
        ev.OutputSanitizedEvent(lane="answer", before="hi ' there", after="hi  there", rule_fired="stray_quote"),
        turn_id="t1", parent_turn_id=None, seq=3,
        emitted_at="x", source_kind="policy", source_name="sanitizer",
    )
    p = e.to_payload()
    assert p["kind"] == "OutputSanitizedEvent"
    assert p["rule_fired"] == "stray_quote"
    assert p["source_name"] == "sanitizer"


def test_fault_record_validates_severity_pairing() -> None:
    with pytest.raises(ValueError):
        tt.FaultTraceRecord(
            turn_id="t", parent_turn_id=None, seq=0, emitted_at="x",
            event_kind="X", source_kind="kernel", source_name="k",
            fault_kind="abc",  # severity missing — must raise
        )
    with pytest.raises(ValueError):
        tt.FaultTraceRecord(
            turn_id="t", parent_turn_id=None, seq=0, emitted_at="x",
            event_kind="X", source_kind="kernel", source_name="k",
            severity="warn",  # fault_kind missing — must raise
        )


def test_fault_record_validates_source_kind() -> None:
    with pytest.raises(ValueError):
        tt.FaultTraceRecord(
            turn_id="t", parent_turn_id=None, seq=0, emitted_at="x",
            event_kind="X", source_kind="bogus", source_name="k",
        )


# ── bus mechanics ───────────────────────────────────────────────────


def test_bootstrap_pipeline_with_empty_registry(pipeline_db) -> None:
    pipeline = tp.bootstrap_pipeline()
    assert isinstance(pipeline, tp.TurnPipeline)
    # Calling again returns the same singleton via get_pipeline.
    assert tp.get_pipeline() is pipeline


def test_run_turn_emits_bracketing_lifecycle_events(pipeline_db) -> None:
    pipeline = tp.bootstrap_pipeline()
    result = pipeline.run_turn(turn_id="bracket-1")
    assert result["outcome"] == "ok"
    rows = tt.list_pipeline_events("bracket-1")
    assert [r.event_kind for r in rows] == [
        "TurnStreamStartedEvent",
        "TurnCompleteEvent",
    ]
    assert all(r.source_kind == "kernel" for r in rows)
    assert all(r.source_name == "turn_pipeline" for r in rows)


def test_publish_assigns_monotonic_seq(pipeline_db) -> None:
    pipeline = tp.bootstrap_pipeline()
    ctx = tp.TurnContext(turn_id="seq-1", started_at=0.0)
    pipeline.publish(ev.StreamChunkReceivedEvent(text="a"), ctx, source_kind="producer", source_name="test")
    pipeline.publish(ev.StreamChunkReceivedEvent(text="b"), ctx, source_kind="producer", source_name="test")
    pipeline.publish(ev.StreamChunkReceivedEvent(text="c"), ctx, source_kind="producer", source_name="test")
    rows = tt.list_pipeline_events("seq-1")
    assert [r.seq for r in rows] == [0, 1, 2]


def test_seq_stays_monotonic_across_multiple_ctx_for_same_turn(pipeline_db) -> None:
    """Regression: chat.py creates a fresh TurnContext per emit site. The
    kernel must scope seq by turn_id (not by ctx) so emissions across
    contexts still produce a unique, monotonic sequence."""
    pipeline = tp.bootstrap_pipeline()
    for i in range(5):
        ctx = tp.TurnContext(turn_id="multi-ctx", started_at=0.0)
        pipeline.publish(
            ev.FaultDetectedEvent(fault_kind=f"k{i}", severity="warn"),
            ctx, source_kind="kernel", source_name="test",
        )
    seqs = [r.seq for r in tt.list_pipeline_events("multi-ctx")]
    assert seqs == [0, 1, 2, 3, 4]


def test_seq_is_independent_across_turns(pipeline_db) -> None:
    pipeline = tp.bootstrap_pipeline()
    for tid in ("A", "B"):
        ctx = tp.TurnContext(turn_id=tid, started_at=0.0)
        pipeline.publish(
            ev.StreamChunkReceivedEvent(text="x"),
            ctx, source_kind="producer", source_name="t",
        )
    assert [r.seq for r in tt.list_pipeline_events("A")] == [0]
    assert [r.seq for r in tt.list_pipeline_events("B")] == [0]


def test_fault_event_lifts_fields_to_columns(pipeline_db) -> None:
    pipeline = tp.bootstrap_pipeline()
    ctx = tp.TurnContext(turn_id="fault-1", started_at=0.0)
    pipeline.publish(
        ev.FaultDetectedEvent(fault_kind="tool_no_fire", severity="warn", detail={"retry": 1}),
        ctx, source_kind="policy", source_name="tool_loop_continuation",
    )
    rows = tt.list_pipeline_events("fault-1")
    assert len(rows) == 1
    assert rows[0].fault_kind == "tool_no_fire"
    assert rows[0].severity == "warn"
    assert rows[0].payload["detail"] == {"retry": 1}
    assert ctx.fault_count == 1  # kernel bumps the counter on publish


def test_turn_ready_event_has_default_turn_phase() -> None:
    """Default turn_phase is 'initial' so callers that don't pass it get
    backward-compatible behavior. Round-trips through payload_fields()."""
    e = ev.TurnReadyEvent(raw_answer="a", public_answer="b")
    assert e.turn_phase == "initial"
    payload = e.payload_fields()
    assert payload["turn_phase"] == "initial"


def test_turn_ready_event_carries_tool_followup_phase() -> None:
    """Callers that emit a post-tool continuation set turn_phase='tool_followup'
    so consumers can query 'did the answer regress after tool execution?'."""
    e = ev.TurnReadyEvent(
        raw_answer="a", public_answer="b",
        turn_phase="tool_followup",
    )
    assert e.turn_phase == "tool_followup"
    assert e.payload_fields()["turn_phase"] == "tool_followup"


def test_turn_ready_event_phase_persists_through_kernel(pipeline_db) -> None:
    """Phase round-trips from publish() through fault_traces persistence
    so downstream queries can filter by phase."""
    pipeline = tp.bootstrap_pipeline()
    ctx = tp.TurnContext(turn_id="phase-1", started_at=0.0)
    pipeline.publish(
        ev.TurnReadyEvent(raw_answer="x", public_answer="y", turn_phase="tool_followup"),
        ctx, source_kind="kernel", source_name="test",
    )
    rows = tt.list_pipeline_events("phase-1")
    ready_rows = [r for r in rows if r.event_kind == "TurnReadyEvent"]
    assert len(ready_rows) == 1
    assert ready_rows[0].payload.get("turn_phase") == "tool_followup"


def test_fault_row_carries_payload_schema_version(pipeline_db) -> None:
    """Every new fault_traces row records the payload-format version it was
    written under, so future readers can branch on schema changes without
    backfill. Kernel writes FAULT_TRACES_PAYLOAD_SCHEMA_VERSION (currently 1)
    on every publish."""
    pipeline = tp.bootstrap_pipeline()
    ctx = tp.TurnContext(turn_id="schema-version-1", started_at=0.0)
    pipeline.publish(
        ev.FaultDetectedEvent(fault_kind="x", severity="warn"),
        ctx, source_kind="policy", source_name="t",
    )
    rows = tt.list_pipeline_events("schema-version-1")
    assert len(rows) == 1
    assert rows[0].payload_schema_version == tt.FAULT_TRACES_PAYLOAD_SCHEMA_VERSION
    assert rows[0].payload_schema_version >= 1


def test_fault_row_schema_version_round_trip_via_list_faults(pipeline_db) -> None:
    """list_faults_since also returns the version, so consumers reading via
    that path get the same shape as list_pipeline_events."""
    pipeline = tp.bootstrap_pipeline()
    ctx = tp.TurnContext(turn_id="schema-version-2", started_at=0.0)
    pipeline.publish(
        ev.FaultDetectedEvent(fault_kind="x", severity="warn"),
        ctx, source_kind="policy", source_name="t",
    )
    faults = tt.list_faults_since("2020-01-01T00:00:00+00:00")
    assert any(f.payload_schema_version >= 1 for f in faults)


def test_list_faults_since_filters_to_fault_rows_only(pipeline_db) -> None:
    pipeline = tp.bootstrap_pipeline()
    ctx = tp.TurnContext(turn_id="filter-1", started_at=0.0)
    pipeline.publish(ev.StreamChunkReceivedEvent(text="x"), ctx, source_kind="producer", source_name="p")
    pipeline.publish(
        ev.FaultDetectedEvent(fault_kind="kindA", severity="warn"),
        ctx, source_kind="policy", source_name="p",
    )
    pipeline.publish(
        ev.FaultDetectedEvent(fault_kind="kindB", severity="hard"),
        ctx, source_kind="policy", source_name="p",
    )
    all_events = tt.list_pipeline_events("filter-1")
    assert len(all_events) == 3
    faults_only = tt.list_faults_since("2020-01-01T00:00:00+00:00")
    kinds = {f.fault_kind for f in faults_only}
    assert kinds == {"kindA", "kindB"}
    a_only = tt.list_faults_since("2020-01-01T00:00:00+00:00", fault_kind="kindA")
    assert {f.fault_kind for f in a_only} == {"kindA"}


def test_cleanup_old_records_includes_fault_key(pipeline_db) -> None:
    pipeline = tp.bootstrap_pipeline()
    pipeline.run_turn(turn_id="cleanup-1")
    counts = tt.cleanup_old_records(ttl_days=30)
    assert "fault" in counts
    assert counts["fault"] == 0  # records are fresh — nothing deleted


# ── registry validation ─────────────────────────────────────────────


def test_validate_against_filesystem_passes_for_real_dir() -> None:
    # The actual core/pipeline_policies/ dir matches the declared POLICIES.
    # This is the production-boot assertion.
    from pathlib import Path
    here = Path(__file__).resolve()
    policies_dir = here.parent.parent / "core" / "pipeline_policies"
    reg.validate_against_filesystem(policies_dir)


def test_validate_against_filesystem_fails_on_empty_dir_when_policies_declared(tmp_path) -> None:
    # POLICIES is non-empty post-Phase-2 but tmp_path is empty —
    # the validator must report every declared policy as missing-on-disk.
    with pytest.raises(RuntimeError) as exc:
        reg.validate_against_filesystem(tmp_path)
    msg = str(exc.value)
    # At least one declared policy name appears in the error.
    assert "output_sanitizer" in msg or "verifier_bridge" in msg


def test_validate_against_filesystem_fails_on_orphan_file(tmp_path, monkeypatch) -> None:
    # Stub POLICIES to empty so we can isolate the orphan-file failure mode.
    monkeypatch.setattr(reg, "POLICIES", ())
    pdir = tmp_path / "policies"
    pdir.mkdir()
    (pdir / "rogue.py").write_text("# unregistered policy\n", encoding="utf-8")
    on_disk = reg.discovered_policy_modules(pdir)
    assert on_disk == {"core.pipeline_policies.rogue"}
    with pytest.raises(RuntimeError) as exc:
        reg.validate_against_filesystem(pdir)
    assert "rogue" in str(exc.value)


def test_policy_registration_rejects_mutation_without_kill_switch() -> None:
    with pytest.raises(ValueError) as exc:
        reg.PolicyRegistration(
            name="bad",
            module_path="core.pipeline_policies.bad",
            subscribes_to=(),
            depends_on=(),
            authority_tier=ev.AuthorityTier.MUTATION,
            kill_switch_env_flag="",  # missing!
        )
    assert "kill_switch_env_flag" in str(exc.value)


def test_policy_registration_rejects_retry_budget_on_non_dispatch() -> None:
    with pytest.raises(ValueError):
        reg.PolicyRegistration(
            name="bad",
            module_path="core.pipeline_policies.bad",
            subscribes_to=(),
            depends_on=(),
            authority_tier=ev.AuthorityTier.OBSERVATION,
            kill_switch_env_flag="",
            retry_budget=3,  # only dispatch tier may declare this
        )


def test_topo_sort_handles_dependencies_and_detects_cycles() -> None:
    A = reg.PolicyRegistration(
        name="A", module_path="core.pipeline_policies.a",
        subscribes_to=("E",), depends_on=(),
        authority_tier=ev.AuthorityTier.OBSERVATION, kill_switch_env_flag="",
    )
    B = reg.PolicyRegistration(
        name="B", module_path="core.pipeline_policies.b",
        subscribes_to=("E",), depends_on=("A",),
        authority_tier=ev.AuthorityTier.OBSERVATION, kill_switch_env_flag="",
    )
    C = reg.PolicyRegistration(
        name="C", module_path="core.pipeline_policies.c",
        subscribes_to=("E",), depends_on=("B",),
        authority_tier=ev.AuthorityTier.OBSERVATION, kill_switch_env_flag="",
    )
    ordered = reg.topo_sort([C, B, A])
    assert [p.name for p in ordered] == ["A", "B", "C"]

    # Introduce a cycle.
    A2 = reg.PolicyRegistration(
        name="A", module_path="core.pipeline_policies.a",
        subscribes_to=("E",), depends_on=("C",),  # cycle!
        authority_tier=ev.AuthorityTier.OBSERVATION, kill_switch_env_flag="",
    )
    with pytest.raises(ValueError) as exc:
        reg.topo_sort([A2, B, C])
    assert "cycle" in str(exc.value)


# ── independence ────────────────────────────────────────────────────


def test_pipeline_import_graph_does_not_pull_engine_or_acu() -> None:
    import sys
    before = dict(sys.modules)
    import core.turn_pipeline_events  # noqa: F401
    import core.pipeline_registry  # noqa: F401
    import monokernel.turn_pipeline  # noqa: F401
    added = set(sys.modules) - set(before)
    forbidden = [m for m in added if m.startswith("engine.") or "acu" in m or "acatalepsy" in m]
    assert forbidden == [], f"pipeline transitively loaded forbidden modules: {forbidden}"
