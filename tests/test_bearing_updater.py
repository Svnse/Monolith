from __future__ import annotations

import pytest

from addons.system.bearing import audit
from addons.system.bearing import schema as bs
from addons.system.bearing import store
from addons.system.bearing import updater


@pytest.fixture
def tmp_bearing(monkeypatch, tmp_path):
    """Redirect both the bearing store and the audit jsonl to tmp."""
    monkeypatch.setattr(store, "_STORE_PATH", tmp_path / "bearing.json")
    monkeypatch.setattr(audit, "_AUDIT_PATH", tmp_path / "bearing.audit.jsonl")
    yield tmp_path


@pytest.fixture
def fake_emit(monkeypatch):
    """Stub core.fault_response.emit_fault — capture calls without touching SQLite."""
    captured: list[dict] = []

    def _emit(turn_id, fault_kind, detector_name, evidence, metadata=None):
        captured.append({
            "turn_id": turn_id,
            "fault_kind": fault_kind,
            "detector_name": detector_name,
            "evidence": evidence,
            "metadata": metadata,
        })
        return 42  # any positive int = "emitted"

    import core.fault_response as fr
    monkeypatch.setattr(fr, "emit_fault", _emit)
    yield captured


# ── envelope extraction ─────────────────────────────────────────────


def test_no_envelope_in_response() -> None:
    envelope, parse_failed = updater.extract_envelope("plain text, no tags")
    assert envelope is None
    assert parse_failed is False


def test_empty_envelope_body() -> None:
    envelope, parse_failed = updater.extract_envelope("<bearing_update></bearing_update>")
    assert envelope == {}
    assert parse_failed is False


def test_valid_envelope_extracted() -> None:
    text = '<bearing_update>{"active_goal": {"new": "x", "reason": "r"}}</bearing_update>'
    envelope, parse_failed = updater.extract_envelope(text)
    assert isinstance(envelope, dict)
    assert envelope["active_goal"]["new"] == "x"
    assert parse_failed is False


def test_malformed_json_envelope_returns_parse_failed() -> None:
    text = '<bearing_update>{this is not json}</bearing_update>'
    envelope, parse_failed = updater.extract_envelope(text)
    assert envelope is None
    assert parse_failed is True


def test_non_object_json_envelope_returns_parse_failed() -> None:
    text = '<bearing_update>[1, 2, 3]</bearing_update>'
    envelope, parse_failed = updater.extract_envelope(text)
    assert envelope is None
    assert parse_failed is True


def test_envelope_spans_multiple_lines() -> None:
    text = (
        '<bearing_update>\n'
        '  {\n'
        '    "next_move": {"new": "x", "reason": "r"}\n'
        '  }\n'
        '</bearing_update>'
    )
    envelope, parse_failed = updater.extract_envelope(text)
    assert isinstance(envelope, dict)
    assert envelope["next_move"]["new"] == "x"


# ── apply_envelope ──────────────────────────────────────────────────


def test_apply_envelope_sets_primitive_slot() -> None:
    old = bs.Bearing()
    envelope = {"active_goal": {"new": "x", "reason": "r"}}
    new = updater.apply_envelope(old, envelope, turn_id="t1", model_id="m1")
    assert new.active_goal == "x"
    assert new.last_writer_model_id == "m1"
    assert new.updated_at_turn == "t1"


def test_apply_envelope_preserves_unchanged_slots() -> None:
    old = bs.Bearing(active_goal="keep", current_frame="frame")
    envelope = {"next_move": {"new": "nm", "reason": "r"}}
    new = updater.apply_envelope(old, envelope, "t1", "m1")
    assert new.active_goal == "keep"
    assert new.current_frame == "frame"
    assert new.next_move == "nm"


def test_apply_envelope_adds_tension() -> None:
    envelope = {"open_tensions": {"add": [
        {"text": "t1-text", "opened_at_turn": "t1"}
    ]}}
    new = updater.apply_envelope(bs.Bearing(), envelope, "t1", "m1")
    assert len(new.open_tensions) == 1
    assert new.open_tensions[0].text == "t1-text"


def test_apply_envelope_resolves_tension() -> None:
    old = bs.Bearing(open_tensions=(
        bs.Tension(text="keep", opened_at_turn="t0"),
        bs.Tension(text="resolve_me", opened_at_turn="t0"),
    ))
    envelope = {"open_tensions": {"resolve": [
        {"index": 1, "reason": "addressed", "grounding": "tool_result_1"}
    ]}}
    new = updater.apply_envelope(old, envelope, "t2", "m1")
    assert len(new.open_tensions) == 1
    assert new.open_tensions[0].text == "keep"


def test_apply_envelope_modal_branch_transition() -> None:
    old = bs.Bearing(modal_branches=(
        bs.ModalBranch(text="b1", status="active", reason="r1", last_touched_turn="t0"),
    ))
    envelope = {"modal_branches": {"transition": [
        {"index": 0, "from": "active", "to": "dormant", "reason": "tabled"}
    ]}}
    new = updater.apply_envelope(old, envelope, "t1", "m1")
    assert new.modal_branches[0].status == "dormant"
    assert new.modal_branches[0].last_touched_turn == "t1"


def test_apply_envelope_adds_referent() -> None:
    envelope = {"referents": {"add": [
        {"name": "x.py", "kind": "file", "status": "observed",
         "grounded_at_turn": "t1", "grounding": "tool_result_1", "reason": "r"}
    ]}}
    new = updater.apply_envelope(bs.Bearing(), envelope, "t1", "m1")
    assert len(new.referents) == 1
    assert new.referents[0].name == "x.py"


# ── updated_at_turn_n stamping ──────────────────────────────────────


def test_apply_envelope_stamps_turn_n() -> None:
    new = updater.apply_envelope(
        bs.Bearing(), {"active_goal": {"new": "x", "reason": "r"}},
        turn_id="t1", model_id="m1", turn_n=312,
    )
    assert new.updated_at_turn_n == 312


def test_apply_envelope_turn_n_zero_preserves_old() -> None:
    """turn_n=0 (feature off / not threaded) must not clobber a prior count."""
    old = bs.Bearing(active_goal="g", updated_at_turn_n=99)
    new = updater.apply_envelope(
        old, {"next_move": {"new": "nm", "reason": "r"}}, "t2", "m1", turn_n=0,
    )
    assert new.updated_at_turn_n == 99


def test_process_turn_output_threads_turn_n(tmp_bearing) -> None:
    text = '<bearing_update>{"active_goal": {"new": "shipped", "reason": "r"}}</bearing_update>'
    updater.process_turn_output("t1", text, model_id="m1", turn_n=312)
    assert store.get_bearing().updated_at_turn_n == 312


def test_turn_n_survives_grounding_downgrade(tmp_bearing, fake_emit) -> None:
    """The grounding-failed path rebuilds the Bearing field by field; the count
    must survive, or a grounding failure silently zeroes the readable age."""
    text = (
        '<bearing_update>{"referents": {"add": ['
        '{"name": "x.py", "kind": "file", "status": "observed", '
        '"grounded_at_turn": "t1", "grounding": "tool_result_1", "reason": "r"}'
        ']}}</bearing_update>'
    )
    # tool_result set does NOT contain the cited id → G1 fails → downgrade path.
    updater.process_turn_output(
        "t1", text, model_id="m1", turn_n=312,
        tool_result_ids={"tool_result_999"},
    )
    persisted = store.get_bearing()
    assert persisted.referents[0].status == "unverified"  # downgrade ran
    assert persisted.updated_at_turn_n == 312             # ...count survived


# ── process_turn_output: structural-pass path ───────────────────────


def test_process_turn_no_envelope_returns_not_found(tmp_bearing) -> None:
    result = updater.process_turn_output("t1", "plain text")
    assert result.found_envelope is False
    assert result.bearing_changed is False


def test_process_turn_structural_pass_persists_bearing(tmp_bearing) -> None:
    text = '<bearing_update>{"active_goal": {"new": "shipped", "reason": "user said so"}}</bearing_update>'
    result = updater.process_turn_output("t1", text, model_id="m1")
    assert result.found_envelope is True
    assert result.structural_verdict is not None
    assert result.structural_verdict.ok is True
    assert result.bearing_changed is True
    # Persisted
    persisted = store.get_bearing()
    assert persisted.active_goal == "shipped"
    assert persisted.last_writer_model_id == "m1"
    assert persisted.updated_at_turn == "t1"


def test_process_turn_pass_clears_pending_rejection_and_resets_streak(tmp_bearing) -> None:
    # Pre-populate prior-rejection state.
    store.set_pending_rejection(["D1"], turn_id="t0", ts="prev")
    store.increment_rejection_streak()
    store.increment_rejection_streak()

    text = '<bearing_update>{"active_goal": {"new": "x", "reason": "r"}}</bearing_update>'
    result = updater.process_turn_output("t1", text, model_id="m1")
    assert result.bearing_changed is True
    assert store.get_pending_rejection() is None
    assert store.get_rejection_streak() == 0


def test_process_turn_logs_applied_in_audit(tmp_bearing) -> None:
    text = '<bearing_update>{"active_goal": {"new": "x", "reason": "r"}}</bearing_update>'
    updater.process_turn_output("t1", text, model_id="m1")
    rows = audit.read_recent()
    assert any(r["kind"] == "applied" and r["turn_id"] == "t1" for r in rows)


# ── process_turn_output: structural-reject path ─────────────────────


def test_process_turn_structural_reject_does_not_change_bearing(tmp_bearing, fake_emit) -> None:
    store.set_bearing(bs.Bearing(active_goal="keep"))
    # Missing "reason" → D1 fails.
    text = '<bearing_update>{"active_goal": {"new": "lost"}}</bearing_update>'
    result = updater.process_turn_output("t1", text)
    assert result.bearing_changed is False
    assert result.structural_verdict is not None
    assert result.structural_verdict.ok is False
    # Bearing unchanged.
    assert store.get_bearing().active_goal == "keep"


def test_process_turn_reject_sets_pending_rejection(tmp_bearing, fake_emit) -> None:
    text = '<bearing_update>{"active_goal": {"new": "lost"}}</bearing_update>'
    updater.process_turn_output("t1", text)
    pending = store.get_pending_rejection()
    assert pending is not None
    assert pending["turn_id"] == "t1"
    assert "D1" in pending["failed_rules"]


def test_process_turn_reject_increments_streak(tmp_bearing, fake_emit) -> None:
    text = '<bearing_update>{"active_goal": {"new": "lost"}}</bearing_update>'
    r1 = updater.process_turn_output("t1", text)
    r2 = updater.process_turn_output("t2", text)
    assert r1.streak_after == 1
    assert r2.streak_after == 2
    assert r1.escalated is False
    assert r2.escalated is False


# ── escalation at N=3 ───────────────────────────────────────────────


def test_process_turn_third_reject_escalates(tmp_bearing, fake_emit) -> None:
    text = '<bearing_update>{"active_goal": {"new": "x"}}</bearing_update>'  # no reason
    updater.process_turn_output("t1", text)
    updater.process_turn_output("t2", text)
    r3 = updater.process_turn_output("t3", text)
    assert r3.streak_after == 3
    assert r3.escalated is True
    # emit_fault was called with bearing_structural_unrecoverable
    assert len(fake_emit) == 1
    assert fake_emit[0]["fault_kind"] == "bearing_structural_unrecoverable"
    assert fake_emit[0]["detector_name"] == "bearing_addon"
    # Audit has the escalated row
    rows = audit.read_recent()
    assert any(r["kind"] == "escalated" for r in rows)


def test_process_turn_recovery_after_escalation_resets_streak(tmp_bearing, fake_emit) -> None:
    bad = '<bearing_update>{"active_goal": {"new": "x"}}</bearing_update>'
    updater.process_turn_output("t1", bad)
    updater.process_turn_output("t2", bad)
    updater.process_turn_output("t3", bad)
    # Streak is now 3, fault was emitted
    good = '<bearing_update>{"active_goal": {"new": "x", "reason": "r"}}</bearing_update>'
    r = updater.process_turn_output("t4", good)
    assert r.bearing_changed is True
    assert store.get_rejection_streak() == 0
    assert store.get_pending_rejection() is None


# ── parse_failed path ───────────────────────────────────────────────


def test_process_turn_parse_failed_treated_as_rejection(tmp_bearing, fake_emit) -> None:
    text = '<bearing_update>{not valid}</bearing_update>'
    result = updater.process_turn_output("t1", text)
    assert result.parse_failed is True
    assert result.streak_after == 1
    # Pending rejection set with parse_error
    pending = store.get_pending_rejection()
    assert pending is not None
    assert "parse_error" in pending["failed_rules"]


# ── escalation threshold tunable ────────────────────────────────────


def test_escalation_threshold_env_override(monkeypatch, tmp_bearing, fake_emit) -> None:
    monkeypatch.setenv("MONOLITH_BEARING_ESCALATION_N", "2")
    text = '<bearing_update>{"active_goal": {"new": "x"}}</bearing_update>'
    updater.process_turn_output("t1", text)
    r2 = updater.process_turn_output("t2", text)
    assert r2.escalated is True
