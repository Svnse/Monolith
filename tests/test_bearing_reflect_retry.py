"""End-to-end test of the reflect-and-retry mechanism:

  reject → next-turn [BEARING_UPDATE_REJECTED] injected →
  repair succeeds (clears) OR third rejection escalates (emit_fault).
"""
from __future__ import annotations

import pytest

from addons.system.bearing import audit
from addons.system.bearing import compiler
from addons.system.bearing import schema as bs
from addons.system.bearing import store
from addons.system.bearing import updater


@pytest.fixture
def bearing_tmp(monkeypatch, tmp_path):
    monkeypatch.setattr(store, "_STORE_PATH", tmp_path / "bearing.json")
    monkeypatch.setattr(audit, "_AUDIT_PATH", tmp_path / "bearing.audit.jsonl")
    yield tmp_path


@pytest.fixture
def fake_emit(monkeypatch):
    captured: list[dict] = []

    def _emit(turn_id, fault_kind, detector_name, evidence, metadata=None):
        captured.append({
            "turn_id": turn_id,
            "fault_kind": fault_kind,
            "detector_name": detector_name,
            "evidence": evidence,
        })
        return 1

    import core.fault_response as fr
    monkeypatch.setattr(fr, "emit_fault", _emit)
    yield captured


# ── reject → next-turn injection → repair succeeds ───────────────────


def test_reject_then_repair_clears_state(bearing_tmp, fake_emit) -> None:
    # Turn 1: submit a bad envelope (no reason → D1 fails)
    bad = '<bearing_update>{"active_goal": {"new": "x"}}</bearing_update>'
    r1 = updater.process_turn_output("t1", bad)
    assert r1.structural_verdict is not None and r1.structural_verdict.ok is False

    # State after t1: pending_rejection set, streak=1, bearing unchanged
    assert store.get_pending_rejection() is not None
    assert store.get_rejection_streak() == 1
    assert store.get_bearing().active_goal == ""

    # Turn 2 begins: compiler renders the BEARING block with the rejection
    messages = [{"role": "user", "content": "next turn"}]
    rendered = compiler.bearing_interceptor(messages, {})
    assert rendered is not None
    bearing_block = rendered[0]["content"]
    assert "[BEARING_UPDATE_REJECTED]" in bearing_block
    assert "D1" in bearing_block
    assert "t1" in bearing_block  # prior_turn_id surfaced

    # Turn 2: model emits a corrected envelope
    good = '<bearing_update>{"active_goal": {"new": "x", "reason": "user asked"}}</bearing_update>'
    r2 = updater.process_turn_output("t2", good, model_id="m1")
    assert r2.bearing_changed is True

    # State after t2: pending_rejection cleared, streak=0, bearing updated
    assert store.get_pending_rejection() is None
    assert store.get_rejection_streak() == 0
    assert store.get_bearing().active_goal == "x"

    # No escalation should have fired
    assert fake_emit == []


# ── three rejections in succession → escalation ──────────────────────


def test_third_reject_escalates_via_emit_fault(bearing_tmp, fake_emit) -> None:
    bad = '<bearing_update>{"active_goal": {"new": "x"}}</bearing_update>'  # missing reason
    updater.process_turn_output("t1", bad)
    updater.process_turn_output("t2", bad)
    r3 = updater.process_turn_output("t3", bad)
    assert r3.escalated is True

    # emit_fault was called with the bearing fault kind
    assert len(fake_emit) == 1
    assert fake_emit[0]["fault_kind"] == "bearing_structural_unrecoverable"
    assert fake_emit[0]["detector_name"] == "bearing_addon"

    # Bearing remains unchanged through all three rejections
    assert store.get_bearing().active_goal == ""


# ── escalation does NOT use a new event kind ─────────────────────────


def test_escalation_uses_existing_FaultDetectedEvent_kind() -> None:
    """The Bearing plan forbids adding a new PipelineEvent subclass.
    Verify no BearingTransitionEvent has snuck into the kernel."""
    from core import turn_pipeline_events as tpe
    # The kernel must NOT have a BearingTransitionEvent class.
    assert not hasattr(tpe, "BearingTransitionEvent"), (
        "Plan violation: BearingTransitionEvent must NOT exist in core.turn_pipeline_events"
    )
    # Bearing fault kinds must be registered in fault_response.KNOWN_KINDS.
    from core import fault_response as fr
    assert "bearing_structural_unrecoverable" in fr.KNOWN_KINDS
    assert "bearing_grounding_failed" in fr.KNOWN_KINDS


# ── interceptor sees the rejection block across turn boundary ─────────


def test_pending_rejection_survives_session_cycle(bearing_tmp, fake_emit) -> None:
    """A pending_rejection set in one turn is visible to the compiler in
    the next turn — even if no successful commit happens in between."""
    bad = '<bearing_update>{"active_goal": {"new": "x"}}</bearing_update>'
    updater.process_turn_output("t1", bad)
    # Simulate next turn opening — compiler reads from store fresh.
    messages = [{"role": "user", "content": "another user message"}]
    rendered = compiler.bearing_interceptor(messages, {})
    assert rendered is not None
    assert "[BEARING_UPDATE_REJECTED]" in rendered[0]["content"]


# ── empty envelope is no-op success (resets streak) ──────────────────


def test_empty_envelope_clears_prior_rejection(bearing_tmp, fake_emit) -> None:
    # First a rejection
    bad = '<bearing_update>{"active_goal": {"new": "x"}}</bearing_update>'
    updater.process_turn_output("t1", bad)
    assert store.get_rejection_streak() == 1

    # Then an empty envelope — counts as a successful no-op turn
    empty = '<bearing_update>{}</bearing_update>'
    r = updater.process_turn_output("t2", empty)
    assert r.structural_verdict is not None and r.structural_verdict.ok is True
    assert store.get_rejection_streak() == 0
    assert store.get_pending_rejection() is None


# ── no envelope present has no impact on rejection state ────────────


def test_turn_without_envelope_does_not_change_rejection_state(bearing_tmp, fake_emit) -> None:
    bad = '<bearing_update>{"active_goal": {"new": "x"}}</bearing_update>'
    updater.process_turn_output("t1", bad)
    assert store.get_rejection_streak() == 1
    # Next turn the model emits NO envelope.
    r = updater.process_turn_output("t2", "just a normal response with no envelope")
    assert r.found_envelope is False
    # Pending rejection still present — model didn't repair, didn't make it worse
    assert store.get_pending_rejection() is not None
    assert store.get_rejection_streak() == 1
