from __future__ import annotations

from addons.system.bearing import schema as bs
from addons.system.bearing.structural_verifier import verify_structural


# ── empty / no-op / not-a-dict ───────────────────────────────────────


def test_empty_envelope_passes() -> None:
    v = verify_structural(bs.Bearing(), {})
    assert v.ok is True


def test_not_a_dict_fails_d0() -> None:
    v = verify_structural(bs.Bearing(), "string")  # type: ignore[arg-type]
    assert v.ok is False
    assert "D0" in v.failed_rules


# ── D1: reason required ──────────────────────────────────────────────


def test_d1_primitive_slot_without_reason_fails() -> None:
    v = verify_structural(bs.Bearing(), {"active_goal": {"new": "x"}})
    assert v.ok is False
    assert "D1" in v.failed_rules


def test_d1_primitive_slot_with_reason_passes_d1() -> None:
    proposed = {"active_goal": {"new": "x", "reason": "user asked"}}
    v = verify_structural(bs.Bearing(), proposed)
    assert "D1" not in v.failed_rules


def test_d1_resolve_without_reason_fails() -> None:
    old = bs.Bearing(open_tensions=(bs.Tension(text="t", opened_at_turn="u1"),))
    proposed = {"open_tensions": {"resolve": [{"index": 0, "grounding": "tool_result_1"}]}}
    v = verify_structural(old, proposed)
    assert "D1" in v.failed_rules


def test_d1_open_tensions_add_requires_text() -> None:
    proposed = {"open_tensions": {"add": [{"opened_at_turn": "u1"}]}}
    v = verify_structural(bs.Bearing(), proposed)
    assert "D1" in v.failed_rules


# ── D2: current_frame provenance ─────────────────────────────────────


def test_d2_current_frame_missing_previous_fails() -> None:
    proposed = {"current_frame": {"new": "x", "reason": "r", "trigger": "tool_result_1"}}
    v = verify_structural(bs.Bearing(), proposed)
    assert "D2" in v.failed_rules


def test_d2_current_frame_missing_trigger_fails() -> None:
    proposed = {"current_frame": {"new": "x", "reason": "r", "previous": ""}}
    v = verify_structural(bs.Bearing(), proposed)
    assert "D2" in v.failed_rules


def test_d2_current_frame_complete_passes() -> None:
    proposed = {"current_frame": {
        "new": "x", "previous": "y", "trigger": "tool_result_1", "reason": "r",
    }}
    v = verify_structural(bs.Bearing(), proposed)
    assert v.ok is True


# ── D3: index in range ───────────────────────────────────────────────


def test_d3_resolve_index_out_of_range_fails() -> None:
    old = bs.Bearing(open_tensions=(bs.Tension(text="t", opened_at_turn="u1"),))
    proposed = {"open_tensions": {"resolve": [{"index": 5, "reason": "r"}]}}
    v = verify_structural(old, proposed)
    assert "D3" in v.failed_rules


def test_d3_resolve_index_negative_fails() -> None:
    old = bs.Bearing(open_tensions=(bs.Tension(text="t", opened_at_turn="u1"),))
    proposed = {"open_tensions": {"resolve": [{"index": -1, "reason": "r"}]}}
    v = verify_structural(old, proposed)
    assert "D3" in v.failed_rules


def test_d3_modal_transition_out_of_range_fails() -> None:
    old = bs.Bearing()
    proposed = {"modal_branches": {"transition": [
        {"index": 0, "from": "active", "to": "closed", "reason": "r"}
    ]}}
    v = verify_structural(old, proposed)
    assert "D3" in v.failed_rules


# ── D4: enum values ──────────────────────────────────────────────────


def test_d4_modal_add_bad_status_fails() -> None:
    proposed = {"modal_branches": {"add": [
        {"text": "b", "status": "frozen", "reason": "r"}
    ]}}
    v = verify_structural(bs.Bearing(), proposed)
    assert "D4" in v.failed_rules


def test_d4_referent_bad_kind_fails() -> None:
    proposed = {"referents": {"add": [
        {"name": "x", "kind": "starship", "status": "observed", "grounded_at_turn": "u1", "reason": "r"}
    ]}}
    v = verify_structural(bs.Bearing(), proposed)
    assert "D4" in v.failed_rules


def test_d4_user_model_bad_register_fails() -> None:
    proposed = {"user_model": {"intent_read": "x", "register": "direct", "confidence": 0.5}}
    v = verify_structural(bs.Bearing(), proposed)
    assert "D4" in v.failed_rules


def test_d4_stakes_bad_reversibility_fails() -> None:
    proposed = {"stakes": {"reversibility": "permanent", "urgency": "low", "cost_if_wrong": "x"}}
    v = verify_structural(bs.Bearing(), proposed)
    assert "D4" in v.failed_rules


# ── D5: character limits ─────────────────────────────────────────────


def test_d5_active_goal_over_limit_fails() -> None:
    over = "x" * (bs.MAX_ACTIVE_GOAL + 1)
    proposed = {"active_goal": {"new": over, "reason": "r"}}
    v = verify_structural(bs.Bearing(), proposed)
    assert "D5" in v.failed_rules


def test_d5_tension_text_over_limit_fails() -> None:
    over = "x" * (bs.MAX_TENSION + 1)
    proposed = {"open_tensions": {"add": [{"text": over, "opened_at_turn": "u1"}]}}
    v = verify_structural(bs.Bearing(), proposed)
    assert "D5" in v.failed_rules


# ── D6: schema version ───────────────────────────────────────────────


def test_d6_wrong_schema_version_fails() -> None:
    proposed = {"schema_version": 99}
    v = verify_structural(bs.Bearing(), proposed)
    assert "D6" in v.failed_rules


def test_d6_correct_schema_version_passes() -> None:
    proposed = {"schema_version": bs.SCHEMA_VERSION}
    v = verify_structural(bs.Bearing(), proposed)
    assert v.ok is True


def test_d6_no_schema_version_passes() -> None:
    proposed = {"active_goal": {"new": "x", "reason": "r"}}
    v = verify_structural(bs.Bearing(), proposed)
    assert "D6" not in v.failed_rules


# ── D7: count limits ─────────────────────────────────────────────────


def test_d7_open_tensions_overflow_fails() -> None:
    # MAX_TENSIONS = 5; start with 4, try to add 2 = 6, fails
    existing = tuple(
        bs.Tension(text=f"t{i}", opened_at_turn=f"u{i}") for i in range(4)
    )
    old = bs.Bearing(open_tensions=existing)
    proposed = {"open_tensions": {"add": [
        {"text": "new1", "opened_at_turn": "u-new1"},
        {"text": "new2", "opened_at_turn": "u-new2"},
    ]}}
    v = verify_structural(old, proposed)
    assert "D7" in v.failed_rules


def test_d7_open_tensions_resolve_offsets_add_in_count() -> None:
    """Adding 1 + resolving 1 = net zero; should not trigger D7."""
    existing = tuple(
        bs.Tension(text=f"t{i}", opened_at_turn=f"u{i}") for i in range(bs.MAX_TENSIONS)
    )
    old = bs.Bearing(open_tensions=existing)
    proposed = {
        "open_tensions": {
            "add": [{"text": "new", "opened_at_turn": "u-new"}],
            "resolve": [{"index": 0, "reason": "r", "grounding": "tool_result_1"}],
        },
    }
    v = verify_structural(old, proposed)
    assert "D7" not in v.failed_rules


def test_d7_referents_overflow_fails() -> None:
    existing = tuple(
        bs.Referent(name=f"r{i}", kind="entity", status="observed", grounded_at_turn=f"u{i}")
        for i in range(bs.MAX_REFERENTS)
    )
    old = bs.Bearing(referents=existing)
    proposed = {"referents": {"add": [
        {"name": "r-new", "kind": "entity", "status": "observed", "grounded_at_turn": "u-new", "reason": "r"}
    ]}}
    v = verify_structural(old, proposed)
    assert "D7" in v.failed_rules


def test_d7_modal_branches_overflow_fails() -> None:
    existing = tuple(
        bs.ModalBranch(text=f"b{i}", status="active", reason=f"r{i}", last_touched_turn=f"u{i}")
        for i in range(bs.MAX_BRANCHES)
    )
    old = bs.Bearing(modal_branches=existing)
    proposed = {"modal_branches": {"add": [
        {"text": "b-new", "status": "active", "reason": "r-new"}
    ]}}
    v = verify_structural(old, proposed)
    assert "D7" in v.failed_rules


# ── combined failure modes ───────────────────────────────────────────


def test_multiple_rules_can_fail_together() -> None:
    proposed = {
        "active_goal": {"new": "x" * (bs.MAX_ACTIVE_GOAL + 10)},  # D1 (no reason) + D5
        "modal_branches": {"add": [{"text": "b", "status": "bogus", "reason": "r"}]},  # D4
    }
    v = verify_structural(bs.Bearing(), proposed)
    assert v.ok is False
    failed = set(v.failed_rules)
    assert "D1" in failed
    assert "D4" in failed
    assert "D5" in failed
