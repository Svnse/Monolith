"""Phase 2 policy-behavior tests.

Each policy gets at least one test exercising its decision rule and one
verifying it persists the expected events to fault_traces. Policies are
loaded via the real bootstrap path (not stubbed), so this also serves as
an integration smoke for pipeline_registry.POLICIES.
"""
from __future__ import annotations

import pytest

from core import turn_pipeline_events as ev
from core import turn_trace as tt
from monokernel import turn_pipeline as tp


@pytest.fixture
def booted_pipeline(tmp_path, monkeypatch):
    db = tmp_path / "turn_trace.sqlite3"
    tt.set_db_path(db)
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "1")
    monkeypatch.setenv("MONOLITH_VERIFIER_V1", "1")
    tp.reset_for_tests()
    pipeline = tp.bootstrap_pipeline()
    yield pipeline
    tt.set_db_path(None)
    tp.reset_for_tests()


def _events(turn_id: str) -> list[tt.FaultTraceRecord]:
    return tt.list_pipeline_events(turn_id)


def _publish(pipeline, ctx, event, *, kind="producer", name="test"):
    pipeline.publish(event, ctx, source_kind=kind, source_name=name)


# ── output_sanitizer ────────────────────────────────────────────────


def test_output_sanitizer_flags_internal_tag_leak_in_live_chunk(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="san-1", started_at=0.0)
    _publish(
        booted_pipeline, ctx,
        ev.TagRoutedEvent(lane="answer", delta_text="ok <think>oops</think> next", tag_state="normal"),
    )
    kinds = {r.event_kind for r in _events("san-1")}
    assert "FaultDetectedEvent" in kinds
    faults = [r for r in _events("san-1") if r.event_kind == "FaultDetectedEvent"]
    assert any(r.fault_kind == "internal_tag_leak" for r in faults)


def test_output_sanitizer_ignores_clean_live_chunk(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="san-2", started_at=0.0)
    _publish(
        booted_pipeline, ctx,
        ev.TagRoutedEvent(lane="answer", delta_text="plain prose, no tags", tag_state="normal"),
    )
    faults = [r for r in _events("san-2") if r.event_kind == "FaultDetectedEvent"]
    assert faults == []


def test_output_sanitizer_terminal_fires_fence_imbalance(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="san-3", started_at=0.0)
    _publish(
        booted_pipeline, ctx,
        ev.TurnReadyEvent(raw_answer="```python\nprint(1)\n", public_answer="```python\nprint(1)\n"),
        kind="kernel", name="turn_pipeline",
    )
    rules = [
        r.payload.get("rule_fired")
        for r in _events("san-3")
        if r.event_kind == "OutputSanitizedEvent"
    ]
    assert "fence_imbalance" in rules
    assert ctx.mutation_count >= 1


def test_output_sanitizer_terminal_writes_sanitized_text_on_leak(booted_pipeline) -> None:
    # The consumption seam: on an internal-tag leak the MUTATION handler writes
    # the corrected answer onto ctx.sanitized_text, and it SURVIVES publish()
    # (publish only writes back the four canonical scalars), so the finalize
    # site can read it back and re-commit.
    ctx = tp.TurnContext(turn_id="san-mut-1", started_at=0.0)
    leaked = "The answer is 42. <think>but actually I'm unsure</think> Final."
    _publish(
        booted_pipeline, ctx,
        ev.TurnReadyEvent(raw_answer=leaked, public_answer=leaked),
        kind="kernel", name="turn_pipeline",
    )
    assert ctx.sanitized_text is not None
    assert "<think>" not in ctx.sanitized_text
    assert "but actually" not in ctx.sanitized_text
    assert "The answer is 42." in ctx.sanitized_text
    assert "Final." in ctx.sanitized_text


def test_output_sanitizer_terminal_leaves_sanitized_text_none_on_clean(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="san-mut-2", started_at=0.0)
    clean = "Just a normal answer with no internal tags."
    _publish(
        booted_pipeline, ctx,
        ev.TurnReadyEvent(raw_answer=clean, public_answer=clean),
        kind="kernel", name="turn_pipeline",
    )
    assert ctx.sanitized_text is None


# ── tool_failure_classifier ─────────────────────────────────────────


def test_classifier_recoverable_with_hint_for_missing_field(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="cls-1", started_at=0.0)
    _publish(
        booted_pipeline, ctx,
        ev.ToolFailedEvent(
            call_id="c1",
            tool_name="read_file",
            envelope_summary={"error": "missing required field 'path'"},
        ),
    )
    cls_events = [r for r in _events("cls-1") if r.event_kind == "ToolFailureClassifiedEvent"]
    assert len(cls_events) == 1
    assert cls_events[0].payload["classification"] == ev.ToolFailureKind.RECOVERABLE_WITH_HINT.value
    hint_events = [r for r in _events("cls-1") if r.event_kind == "HintInjectionRequestedEvent"]
    assert len(hint_events) == 1


def test_classifier_hard_failure_for_sandbox_violation(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="cls-2", started_at=0.0)
    _publish(
        booted_pipeline, ctx,
        ev.ToolFailedEvent(
            call_id="c2",
            tool_name="run_command",
            envelope_summary={"error": "sandbox violation: write outside allowed paths"},
        ),
    )
    cls_events = [r for r in _events("cls-2") if r.event_kind == "ToolFailureClassifiedEvent"]
    assert cls_events[0].payload["classification"] == ev.ToolFailureKind.HARD_FAILURE.value
    # no hint event for hard failure
    hint_events = [r for r in _events("cls-2") if r.event_kind == "HintInjectionRequestedEvent"]
    assert hint_events == []


def test_classifier_informational_for_empty_result(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="cls-3", started_at=0.0)
    _publish(
        booted_pipeline, ctx,
        ev.ToolFailedEvent(
            call_id="c3",
            tool_name="grep",
            envelope_summary={"error": "no matches"},
        ),
    )
    cls_events = [r for r in _events("cls-3") if r.event_kind == "ToolFailureClassifiedEvent"]
    assert cls_events[0].payload["classification"] == ev.ToolFailureKind.INFORMATIONAL.value


# ── tool_loop_continuation ──────────────────────────────────────────


def test_continuation_detects_no_fire_and_requests_requeue(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="cont-1", started_at=0.0)
    _publish(
        booted_pipeline, ctx,
        ev.TurnStreamEndedEvent(
            closed_lanes=("answer", "tool_call"),
            had_tool_call=True,
            had_continuation=False,
        ),
        kind="kernel", name="turn_pipeline",
    )
    kinds = [r.event_kind for r in _events("cont-1")]
    assert "FaultDetectedEvent" in kinds
    assert "LoopContinuationRequestedEvent" in kinds
    fault = next(r for r in _events("cont-1") if r.event_kind == "FaultDetectedEvent")
    assert fault.fault_kind == "tool_no_fire"
    assert ctx.requeue_count == 1


def test_continuation_suppresses_after_budget_exhausted(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="cont-2", started_at=0.0)
    # Pre-load the retry counter to the budget so the next no-fire fails closed.
    ctx.retry_budget_used["tool_loop_continuation"] = 2  # default budget
    _publish(
        booted_pipeline, ctx,
        ev.TurnStreamEndedEvent(closed_lanes=("answer",), had_tool_call=True, had_continuation=False),
        kind="kernel", name="turn_pipeline",
    )
    kinds = [r.event_kind for r in _events("cont-2")]
    assert "LoopContinuationSuppressedEvent" in kinds
    assert "LoopContinuationRequestedEvent" not in kinds


def test_continuation_no_op_when_loop_closed_cleanly(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="cont-3", started_at=0.0)
    _publish(
        booted_pipeline, ctx,
        ev.TurnStreamEndedEvent(
            closed_lanes=("answer",), had_tool_call=False, had_continuation=False,
        ),
        kind="kernel", name="turn_pipeline",
    )
    rows = _events("cont-3")
    assert [r.event_kind for r in rows] == ["TurnStreamEndedEvent"]  # only the input
    assert ctx.requeue_count == 0


def test_classifier_suppression_propagates_through_continuation(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="cont-4", started_at=0.0)
    # First — classify a hard failure. The classifier emits
    # ToolFailureClassifiedEvent, which tool_loop_continuation consumes and
    # publishes LoopContinuationSuppressedEvent + flips ctx.suppressed_continuation.
    _publish(
        booted_pipeline, ctx,
        ev.ToolFailedEvent(
            call_id="c1",
            tool_name="run_command",
            envelope_summary={"error": "permission denied"},
        ),
    )
    assert ctx.suppressed_continuation
    # Now — a no-fire stream end. tool_loop_continuation must NOT issue a
    # requeue request because suppression already flagged this turn.
    _publish(
        booted_pipeline, ctx,
        ev.TurnStreamEndedEvent(
            closed_lanes=("tool_call",), had_tool_call=True, had_continuation=False,
        ),
        kind="kernel", name="turn_pipeline",
    )
    kinds = [r.event_kind for r in _events("cont-4")]
    # Allowed: suppression + the input events. Not allowed: a new requeue request.
    assert "LoopContinuationRequestedEvent" not in kinds


# ── parse_retry ─────────────────────────────────────────────────────


def test_parse_retry_requests_continuation_within_budget(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="parse-1", started_at=0.0)
    _publish(
        booted_pipeline, ctx,
        ev.ToolParseFailedEvent(raw='{"name": "x"', error="invalid json", attempt=1),
    )
    kinds = [r.event_kind for r in _events("parse-1")]
    assert "LoopContinuationRequestedEvent" in kinds
    assert ctx.retry_budget_used["parse_retry"] == 1


def test_parse_retry_gives_up_after_budget(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="parse-2", started_at=0.0)
    ctx.retry_budget_used["parse_retry"] = 2  # budget
    _publish(
        booted_pipeline, ctx,
        ev.ToolParseFailedEvent(raw='{bad', error="invalid json", attempt=3),
    )
    kinds = [r.event_kind for r in _events("parse-2")]
    assert "LoopContinuationSuppressedEvent" in kinds
    # Hard fault recorded.
    faults = [r for r in _events("parse-2") if r.event_kind == "FaultDetectedEvent"]
    assert any(r.fault_kind == "tool_parse_unrecoverable" for r in faults)
    assert ctx.suppressed_continuation


# ── verifier_bridge ─────────────────────────────────────────────────


def test_verifier_bridge_publishes_verdict_and_faults_on_internal_tag(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="ver-1", started_at=0.0)
    # Build a TurnReadyEvent whose public_answer leaks <think> — the
    # response_verifier hard-fails this with code="raw_internal_tag".
    _publish(
        booted_pipeline, ctx,
        ev.TurnReadyEvent(
            raw_answer="<think>internal</think>hello",
            public_answer="hello <think>leak</think> world",
            tools_used=(),
        ),
        kind="kernel", name="turn_pipeline",
    )
    kinds = [r.event_kind for r in _events("ver-1")]
    assert "VerifierVerdictEvent" in kinds
    verdict = next(r for r in _events("ver-1") if r.event_kind == "VerifierVerdictEvent")
    assert verdict.payload["verdict"] == "hard_fail"
    # At least one FaultDetectedEvent with fault_kind starting "verifier:"
    fault_kinds = [
        r.fault_kind for r in _events("ver-1")
        if r.event_kind == "FaultDetectedEvent" and r.fault_kind
    ]
    assert any(fk.startswith("verifier:") for fk in fault_kinds)


def test_verifier_bridge_pass_on_clean_answer(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="ver-2", started_at=0.0)
    _publish(
        booted_pipeline, ctx,
        ev.TurnReadyEvent(
            raw_answer="hi there",
            public_answer="hi there",
            tools_used=(),
        ),
        kind="kernel", name="turn_pipeline",
    )
    # Verifier returns either pass or warn (weak_completion_signal) — never
    # hard_fail on a clean answer with no completion claim. We only check
    # that no verifier hard-fault was raised.
    verifier_hard_faults = [
        r for r in _events("ver-2")
        if r.event_kind == "FaultDetectedEvent"
        and (r.fault_kind or "").startswith("verifier:")
        and r.severity == "hard"
    ]
    assert verifier_hard_faults == []


# ── tool_repetition_detector ────────────────────────────────────────


def test_repetition_detector_fires_on_consecutive_identical_calls(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="rep-1", started_at=0.0)
    args = {"path": "C:/foo.txt"}
    _publish(
        booted_pipeline, ctx,
        ev.ToolCallParsedEvent(call_id="c1", tool_name="read_file", payload=args),
    )
    _publish(
        booted_pipeline, ctx,
        ev.ToolCallParsedEvent(call_id="c2", tool_name="read_file", payload=args),
    )
    kinds = [r.event_kind for r in _events("rep-1")]
    assert "FaultDetectedEvent" in kinds
    fault = next(
        r for r in _events("rep-1")
        if r.event_kind == "FaultDetectedEvent" and r.fault_kind == "tool_call_repeated"
    )
    assert fault.payload["detail"]["consecutive_count"] == 2
    assert "LoopContinuationSuppressedEvent" in kinds
    assert ctx.suppressed_continuation


def test_repetition_detector_ignores_distinct_calls(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="rep-2", started_at=0.0)
    _publish(
        booted_pipeline, ctx,
        ev.ToolCallParsedEvent(call_id="c1", tool_name="read_file", payload={"path": "a.txt"}),
    )
    _publish(
        booted_pipeline, ctx,
        ev.ToolCallParsedEvent(call_id="c2", tool_name="read_file", payload={"path": "b.txt"}),
    )
    repeat_faults = [
        r for r in _events("rep-2")
        if r.event_kind == "FaultDetectedEvent" and r.fault_kind == "tool_call_repeated"
    ]
    assert repeat_faults == []
    assert not ctx.suppressed_continuation


def test_repetition_detector_does_not_double_fire_within_run(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="rep-3", started_at=0.0)
    args = {"path": "C:/foo.txt"}
    for cid in ("c1", "c2", "c3", "c4"):
        _publish(
            booted_pipeline, ctx,
            ev.ToolCallParsedEvent(call_id=cid, tool_name="read_file", payload=args),
        )
    repeat_faults = [
        r for r in _events("rep-3")
        if r.event_kind == "FaultDetectedEvent" and r.fault_kind == "tool_call_repeated"
    ]
    # Threshold trips on the 2nd consecutive call; subsequent identical
    # calls must not re-emit the fault until a different call breaks the run.
    assert len(repeat_faults) == 1


def test_repetition_detector_resets_after_distinct_call(booted_pipeline) -> None:
    ctx = tp.TurnContext(turn_id="rep-4", started_at=0.0)
    args_a = {"path": "a.txt"}
    args_b = {"path": "b.txt"}
    # First repeat-pair on args_a → suppression fires.
    _publish(booted_pipeline, ctx, ev.ToolCallParsedEvent(call_id="c1", tool_name="read_file", payload=args_a))
    _publish(booted_pipeline, ctx, ev.ToolCallParsedEvent(call_id="c2", tool_name="read_file", payload=args_a))
    # A distinct call breaks the run; suppression must re-arm.
    _publish(booted_pipeline, ctx, ev.ToolCallParsedEvent(call_id="c3", tool_name="read_file", payload=args_b))
    # Now a fresh repeat-pair on args_b → second fault expected.
    _publish(booted_pipeline, ctx, ev.ToolCallParsedEvent(call_id="c4", tool_name="read_file", payload=args_b))
    repeat_faults = [
        r for r in _events("rep-4")
        if r.event_kind == "FaultDetectedEvent" and r.fault_kind == "tool_call_repeated"
    ]
    assert len(repeat_faults) == 2


# ── registry integration ────────────────────────────────────────────


def test_all_registered_policies(booted_pipeline) -> None:
    """Exact-set assertion on registered policy names.

    When a new policy is added (e.g. subordinate_clause_detector landed after
    the original five), include it in the expected set. Function renamed from
    `test_all_five_policies_are_registered` once the count exceeded five —
    keeping the count in the name made it lie about reality.
    """
    from core import pipeline_registry as reg
    names = {p.name for p in reg.iter_policies()}
    assert names == {
        "output_sanitizer",
        "tool_failure_classifier",
        "tool_loop_continuation",
        "tool_repetition_detector",
        "parse_retry",
        "verifier_bridge",
        "subordinate_clause_detector",
        "commitment_detector",
    }


def test_kill_switch_disables_mutation_tier(booted_pipeline, monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_PIPELINE_SANITIZER_V1", "0")
    ctx = tp.TurnContext(turn_id="kill-1", started_at=0.0)
    _publish(
        booted_pipeline, ctx,
        ev.TagRoutedEvent(lane="answer", delta_text="bad <think>leak</think>", tag_state="normal"),
    )
    # With sanitizer disabled, no FaultDetectedEvent should appear from it.
    fault_rows = [r for r in _events("kill-1") if r.event_kind == "FaultDetectedEvent"]
    assert fault_rows == []


# ── cross-emit per-turn state (regression) ──────────────────────────


def test_dispatch_state_persists_across_separate_ctx_for_same_turn(booted_pipeline) -> None:
    """Regression: chat.py constructs a fresh TurnContext per emit site
    (see ui/pages/chat.py:2463 + :2791). Dispatch-tier policies that track
    ctx.retry_budget_used and ctx.suppressed_continuation must see canonical
    per-turn state — not a reset accumulator on every emit — or the budget
    threshold never trips in production.

    Witness: tool_loop_continuation has DEFAULT_RETRY_BUDGET=2. After three
    no-fire emits within the same turn, the third must be SUPPRESSED. With
    per-emit ctx state, each emit reads an empty retry_budget_used dict,
    bumps to 1, never reaches budget, and the suppression never fires.
    """
    pipeline = booted_pipeline
    turn_id = "cross-ctx-budget"

    for i in range(3):
        # Fresh ctx every emit, just like chat.py does.
        ctx = tp.TurnContext(turn_id=turn_id, started_at=0.0)
        _publish(
            pipeline, ctx,
            ev.TurnStreamEndedEvent(
                closed_lanes=("answer",),
                had_tool_call=True,
                had_continuation=False,
            ),
            kind="kernel", name="turn_pipeline",
        )

    kinds = [r.event_kind for r in _events(turn_id)]
    # Two requeue requests (emits 1 and 2 within budget=2), one suppressed
    # (emit 3 over budget).
    assert kinds.count("LoopContinuationRequestedEvent") == 2
    assert kinds.count("LoopContinuationSuppressedEvent") == 1


def test_fault_count_accumulates_across_separate_ctx_for_same_turn(booted_pipeline) -> None:
    """Companion regression: kernel-side ctx.fault_count auto-bump on
    FaultDetectedEvent must accumulate across emits within a turn so the
    TurnCompleteEvent summary reports the real fault total, not just the
    last-emit count."""
    pipeline = booted_pipeline
    turn_id = "cross-ctx-faults"

    for i in range(4):
        ctx = tp.TurnContext(turn_id=turn_id, started_at=0.0)
        _publish(
            pipeline, ctx,
            ev.FaultDetectedEvent(fault_kind=f"k{i}", severity="warn"),
            kind="kernel", name="test",
        )
        # Last emit must see the total, not 1.
        if i == 3:
            assert ctx.fault_count == 4
