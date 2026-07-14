"""Tests for the Self-Check Loop subsystem.

Closes Monolith's keystone open loop: every turn the verifier produces a
verdict (pass/warn/hard_fail) that lands in fault_traces as a
VerifierVerdictEvent row, but the model never reads it back. These tests
cover the three wires:

  Wire 1 — turn_trace.get_last_verification_result(): read the latest
           persisted verdict (Seed A — the verdict is ALREADY durable in
           fault_traces; this is a read-API, not a persistence seam).
  Wire 2 — core.fault_telemetry: the [SELF-CHECK] coalescer contributor.
  Wire 3 — confidence_trajectory verdict penalty.
"""
from __future__ import annotations

import os

import pytest

import core.turn_trace as tt


# ── fixtures ────────────────────────────────────────────────────────────────


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Each test gets its own turn_trace store with writes enabled."""
    db = tmp_path / "test_self_check.sqlite3"
    tt.set_db_path(db)
    os.environ["MONOLITH_TURN_TRACE_V1"] = "1"
    yield db
    tt.set_db_path(None)


def _seed_verdict(turn_id, verdict, *, findings=None, emitted_at="2026-06-06T00:00:00+00:00", seq=1):
    """Seed a VerifierVerdictEvent row the way the kernel persists it
    (monokernel/turn_pipeline.py -> _record_event_to_fault_traces -> record_fault)."""
    tt.record_fault(
        tt.FaultTraceRecord(
            turn_id=turn_id,
            parent_turn_id=None,
            seq=seq,
            emitted_at=emitted_at,
            event_kind="VerifierVerdictEvent",
            source_kind="policy",
            source_name="verifier_bridge",
            payload={"verdict": verdict, "findings": findings or []},
        )
    )


# ── Wire 1: get_last_verification_result ─────────────────────────────────────


def test_get_last_verification_result_returns_latest_verdict():
    _seed_verdict("turn-1", "hard_fail", findings=[{"code": "weak_completion_signal"}])
    res = tt.get_last_verification_result()
    assert res is not None
    assert res["verdict"] == "hard_fail"
    assert res["turn_id"] == "turn-1"
    assert res["findings"] == [{"code": "weak_completion_signal"}]


def test_get_last_verification_result_returns_none_when_empty():
    assert tt.get_last_verification_result() is None


def test_get_last_verification_result_picks_newest_of_several():
    _seed_verdict("turn-1", "pass", emitted_at="2026-06-06T00:00:00+00:00", seq=1)
    _seed_verdict("turn-2", "warn", emitted_at="2026-06-06T00:01:00+00:00", seq=2)
    _seed_verdict("turn-3", "hard_fail", emitted_at="2026-06-06T00:02:00+00:00", seq=3)
    res = tt.get_last_verification_result()
    assert res["turn_id"] == "turn-3"
    assert res["verdict"] == "hard_fail"


def test_get_last_verification_result_ignores_non_verdict_events():
    # A plain fault row (FaultDetectedEvent) must not be mistaken for a verdict.
    tt.record_fault(
        tt.FaultTraceRecord(
            turn_id="turn-1", parent_turn_id=None, seq=1,
            emitted_at="2026-06-06T00:00:00+00:00",
            event_kind="FaultDetectedEvent", source_kind="policy",
            source_name="commitment_detector",
            fault_kind="tool_no_fire", severity="warn",
            payload={"detail": "no tool fired"},
        )
    )
    assert tt.get_last_verification_result() is None


def test_get_last_verification_result_none_when_flag_off():
    _seed_verdict("turn-1", "hard_fail")
    os.environ["MONOLITH_TURN_TRACE_V1"] = "0"
    try:
        assert tt.get_last_verification_result() is None
    finally:
        os.environ["MONOLITH_TURN_TRACE_V1"] = "1"


# ── Wire 2: the [SELF-CHECK] coalescer contributor ───────────────────────────


@pytest.fixture
def self_check_on():
    os.environ["MONOLITH_FAULT_TELEMETRY_V1"] = "1"
    yield
    os.environ.pop("MONOLITH_FAULT_TELEMETRY_V1", None)


def test_render_block_shows_verdict_and_finding_codes(self_check_on):
    import core.fault_telemetry as ft
    _seed_verdict(
        "turn-1", "hard_fail",
        findings=[{"code": "dangling_tool_evidence", "message": "tool evidence block opened but not closed"}],
    )
    block = ft.render_self_check_block()
    assert block is not None
    assert "[SELF-CHECK]" in block
    assert "hard_fail" in block
    assert "dangling_tool_evidence" in block


def test_render_block_quiet_on_clean_pass(self_check_on):
    import core.fault_telemetry as ft
    _seed_verdict("turn-1", "pass", findings=[])
    assert ft.render_self_check_block() is None


def test_contribute_section_returns_section_on_failure(self_check_on):
    import core.fault_telemetry as ft
    _seed_verdict("turn-1", "warn", findings=[{"code": "empty_public_answer"}])
    sec = ft.contribute_section([{"role": "user", "content": "hi"}], {})
    assert sec is not None
    assert sec.name == "self_check"
    assert "warn" in sec.text


def test_contribute_section_none_when_flag_off():
    # flag intentionally not set (defaults OFF — ships dark)
    import core.fault_telemetry as ft
    _seed_verdict("turn-1", "hard_fail")
    assert ft.contribute_section([{"role": "user", "content": "hi"}], {}) is None


def test_contribute_section_suppressed_on_connect_peer_turn(self_check_on):
    import core.fault_telemetry as ft
    _seed_verdict("turn-1", "hard_fail")
    peer_msgs = [{"role": "user", "content": "[CHANNEL: connect/examiner] probe"}]
    assert ft.contribute_section(peer_msgs, {}) is None


def test_coalescer_registers_self_check_in_both_places():
    # The contributor name must appear in BOTH _contributors() and _DROP_ORDER,
    # keyed by the same string, or apply_budget can't prioritize it.
    from core import ephemeral_coalescer as ec
    names = [n for n, _ in ec._contributors()]
    assert "self_check" in names, names
    assert "self_check" in ec._DROP_ORDER, ec._DROP_ORDER
    # It must outrank low-priority telemetry under budget pressure (keystone signal).
    order = list(ec._DROP_ORDER)
    assert order.index("self_check") < order.index("rating_telemetry")


# ── Wire 3: confidence penalty chain ─────────────────────────────────────────


def test_verdict_penalty_values():
    from core import confidence_trajectory as ct
    assert ct._verdict_penalty("warn") == 0.8
    assert ct._verdict_penalty("hard_fail") == 0.6
    assert ct._verdict_penalty("pass") == 1.0
    assert ct._verdict_penalty(None) == 1.0
    assert ct._verdict_penalty("garbage") == 1.0


def test_confidence_block_discounted_on_hard_fail(self_check_on, monkeypatch, tmp_path):
    from core import confidence_trajectory as ct
    monkeypatch.setattr(ct, "_LOG_PATH", tmp_path / "confidence_log.jsonl")
    ct.record_confidence(80, "claim", "premise", "model-x")  # mean = 80
    _seed_verdict("turn-1", "hard_fail")
    block = ct.render_confidence_block()
    assert "mean: 48" in block  # 80 * 0.6
    assert "hard_fail" in block


def test_confidence_block_not_discounted_when_self_check_off(monkeypatch, tmp_path):
    from core import confidence_trajectory as ct
    monkeypatch.delenv("MONOLITH_FAULT_TELEMETRY_V1", raising=False)  # loop off
    monkeypatch.setattr(ct, "_LOG_PATH", tmp_path / "confidence_log.jsonl")
    ct.record_confidence(80, "claim", "premise", "model-x")
    _seed_verdict("turn-1", "hard_fail")
    block = ct.render_confidence_block()
    assert "mean: 80" in block  # raw, no penalty applied


# ── Wire 2b: C1.1 recurring non-verifier faults ──────────────────────────────


def _seed_fault(turn_id, fault_kind, *, seq=1, source_name="commitment_detector",
                emitted_at="2026-06-06T00:00:00+00:00"):
    """Seed a plain FaultDetectedEvent row (the non-verifier fault category)."""
    tt.record_fault(
        tt.FaultTraceRecord(
            turn_id=turn_id, parent_turn_id=None, seq=seq,
            emitted_at=emitted_at, event_kind="FaultDetectedEvent",
            source_kind="policy", source_name=source_name,
            fault_kind=fault_kind, severity="warn", payload={},
        )
    )


def test_recurrence_counts_only_repeats_sorted_by_count():
    import core.fault_telemetry as ft
    for i in range(3):
        _seed_fault(f"tl-{i}", "think_leak")
    for i in range(2):
        _seed_fault(f"tnf-{i}", "tool_no_fire")
    _seed_fault("one-off", "commitment_unfulfilled:inspection")  # count 1 → excluded
    rec = ft._recent_recurring_faults()
    assert rec == [("think_leak", 3), ("tool_no_fire", 2)]  # one-off dropped, count desc


def test_recurrence_excludes_environmental_kinds():
    import core.fault_telemetry as ft
    for i in range(5):
        _seed_fault(f"sd-{i}", "spawn_denied")  # environmental → not the model's to fix
    assert ft._recent_recurring_faults() == []


def test_recurrence_excludes_verifier_prefixed_kinds():
    import core.fault_telemetry as ft
    for i in range(3):
        _seed_fault(f"v-{i}", "verifier:weak_completion")
    assert ft._recent_recurring_faults() == []


def test_recurrence_capped_at_max_kinds():
    import core.fault_telemetry as ft
    for kind in ("a_fault", "b_fault", "c_fault", "d_fault"):
        for i in range(2):
            _seed_fault(f"{kind}-{i}", kind)
    rec = ft._recent_recurring_faults(max_kinds=3)
    assert len(rec) == 3


def test_recurring_line_has_monosearch_pointer():
    import core.fault_telemetry as ft
    for i in range(2):
        _seed_fault(f"tl-{i}", "think_leak")
    line = ft._recurring_line()
    assert line is not None
    assert "think_leak" in line and "×2" in line
    assert "MonoSearch" in line


def test_recurring_line_none_when_no_recurrence():
    import core.fault_telemetry as ft
    _seed_fault("solo", "think_leak")  # one-off only
    assert ft._recurring_line() is None


def test_recurrence_shows_even_on_clean_pass(self_check_on):
    # The key C1.1 behavior: a recurring fault surfaces even when the LAST turn's
    # verdict was a clean pass — recurrence is a standing signal, not last-turn.
    import core.fault_telemetry as ft
    _seed_verdict("turn-1", "pass", findings=[])
    for i in range(2):
        _seed_fault(f"tl-{i}", "think_leak")
    block = ft.render_self_check_block()
    assert block is not None
    assert "[SELF-CHECK]" in block
    assert "think_leak ×2" in block


def test_block_silent_when_clean_pass_and_no_recurrence(self_check_on):
    # Preserve the existing "silent on a quiet turn" behavior.
    import core.fault_telemetry as ft
    _seed_verdict("turn-1", "pass", findings=[])
    _seed_fault("solo", "think_leak")  # one-off, not recurrence
    assert ft.render_self_check_block() is None


def test_verdict_and_recurrence_both_render(self_check_on):
    import core.fault_telemetry as ft
    _seed_verdict("turn-1", "warn", findings=[{"code": "empty_public_answer"}])
    for i in range(2):
        _seed_fault(f"tnf-{i}", "tool_no_fire")
    block = ft.render_self_check_block()
    assert "warn" in block
    assert "empty_public_answer" in block
    assert "tool_no_fire ×2" in block
