from __future__ import annotations

import pytest

from addons.system.bearing import schema as bs


# ── empty Bearing ────────────────────────────────────────────────────


def test_empty_bearing_constructs() -> None:
    b = bs.Bearing()
    assert b.schema_version == bs.SCHEMA_VERSION
    assert b.current_frame == ""
    assert b.active_goal == ""
    assert b.open_tensions == ()
    assert b.referents == ()
    assert b.modal_branches == ()
    assert b.stakes is None
    assert b.user_model is None


def test_empty_bearing_is_empty() -> None:
    assert bs.Bearing().is_empty() is True


def test_non_empty_bearing_not_empty() -> None:
    b = bs.Bearing(active_goal="x")
    assert b.is_empty() is False


def test_bearing_with_only_a_tension_not_empty() -> None:
    b = bs.Bearing(open_tensions=(bs.Tension(text="t", opened_at_turn="t1"),))
    assert b.is_empty() is False


# ── frozen guarantees ────────────────────────────────────────────────


def test_bearing_is_frozen() -> None:
    b = bs.Bearing()
    with pytest.raises(Exception):
        b.current_frame = "mutated"  # type: ignore[misc]


def test_tension_is_frozen() -> None:
    t = bs.Tension(text="x", opened_at_turn="y")
    with pytest.raises(Exception):
        t.text = "mutated"  # type: ignore[misc]


def test_referent_is_frozen() -> None:
    r = bs.Referent(name="x", kind="file", status="observed", grounded_at_turn="t1")
    with pytest.raises(Exception):
        r.status = "unverified"  # type: ignore[misc]


def test_modal_branch_is_frozen() -> None:
    mb = bs.ModalBranch(text="x", status="active", reason="y", last_touched_turn="t1")
    with pytest.raises(Exception):
        mb.status = "closed"  # type: ignore[misc]


# ── enum value sets ──────────────────────────────────────────────────


def test_valid_branch_status_values() -> None:
    assert "active" in bs.VALID_BRANCH_STATUS
    assert "dormant" in bs.VALID_BRANCH_STATUS
    assert "closed" in bs.VALID_BRANCH_STATUS
    assert "rejected" in bs.VALID_BRANCH_STATUS
    assert "superseded" in bs.VALID_BRANCH_STATUS


def test_valid_referent_status_values() -> None:
    assert {"observed", "inferred", "predicted", "unverified"} == set(bs.VALID_REFERENT_STATUS)


def test_valid_register_values() -> None:
    assert {"literal", "performative", "ironic", "exploratory"} == set(bs.VALID_REGISTER)


# ── character limit constants ────────────────────────────────────────


def test_character_limit_constants_positive() -> None:
    for name in (
        "MAX_CURRENT_FRAME",
        "MAX_ACTIVE_GOAL",
        "MAX_TRAJECTORY",
        "MAX_TENSION",
        "MAX_REFERENT_NAME",
        "MAX_BRANCH_TEXT",
        "MAX_BRANCH_REASON",
        "MAX_USER_INTENT_READ",
        "MAX_NEXT_MOVE",
        "MAX_STAKES_COST_IF_WRONG",
    ):
        value = getattr(bs, name)
        assert isinstance(value, int)
        assert value > 0


def test_count_limit_constants() -> None:
    assert bs.MAX_TENSIONS == 5
    assert bs.MAX_REFERENTS == 8
    assert bs.MAX_BRANCHES == 6


# ── to_dict / from_dict roundtrip ────────────────────────────────────


def test_empty_bearing_roundtrip() -> None:
    b = bs.Bearing()
    d = b.to_dict()
    b2 = bs.Bearing.from_dict(d)
    assert b == b2


def test_full_bearing_roundtrip() -> None:
    b = bs.Bearing(
        current_frame="reviewing the design doc",
        active_goal="ship Bearing V0",
        trajectory="design → verification → plan → impl",
        open_tensions=(
            bs.Tension(text="reflect-and-retry is novel", opened_at_turn="t1"),
            bs.Tension(text="layer accounting unaddressed", opened_at_turn="t2"),
        ),
        referents=(
            bs.Referent(
                name="continuity.py", kind="file", status="observed", grounded_at_turn="t1"
            ),
        ),
        modal_branches=(
            bs.ModalBranch(
                text="route to Turn Pipeline", status="rejected",
                reason="violates constraint #3", last_touched_turn="t3",
            ),
        ),
        stakes=bs.Stakes(reversibility="hard", urgency="medium", cost_if_wrong="time"),
        user_model=bs.UserModel(intent_read="contract-level review", register="literal", confidence=0.9),
        next_move="write playful-chasing-penguin.md",
        last_writer_model_id="claude-opus-4-7",
        updated_at_turn="t4",
    )
    d = b.to_dict()
    b2 = bs.Bearing.from_dict(d)
    assert b == b2


def test_from_dict_with_bad_input_returns_empty() -> None:
    assert bs.Bearing.from_dict(None) == bs.Bearing()  # type: ignore[arg-type]
    assert bs.Bearing.from_dict("not a dict") == bs.Bearing()  # type: ignore[arg-type]
    assert bs.Bearing.from_dict([]) == bs.Bearing()  # type: ignore[arg-type]


def test_from_dict_ignores_malformed_nested_entries() -> None:
    # Garbage inside the list should not cause a crash; malformed entries skip.
    d = {
        "open_tensions": ["not a dict", {"text": "ok", "opened_at_turn": "t1"}],
        "referents": [123, {"name": "x", "kind": "file", "status": "observed", "grounded_at_turn": "t1"}],
    }
    b = bs.Bearing.from_dict(d)
    assert len(b.open_tensions) == 1
    assert b.open_tensions[0].text == "ok"
    assert len(b.referents) == 1
    assert b.referents[0].name == "x"


# ── updated_at_turn_n (readable turn-count metadata) ─────────────────


def test_updated_at_turn_n_round_trips() -> None:
    """The integer turn-count must survive to_dict/from_dict alongside the
    UUID updated_at_turn (which is kept for trace-join)."""
    b = bs.Bearing(current_frame="f", updated_at_turn="abc123", updated_at_turn_n=312)
    d = b.to_dict()
    assert d["updated_at_turn_n"] == 312
    b2 = bs.Bearing.from_dict(d)
    assert b2.updated_at_turn_n == 312
    assert b == b2


def test_updated_at_turn_n_defaults_zero() -> None:
    assert bs.Bearing().updated_at_turn_n == 0


def test_from_dict_missing_turn_n_defaults_zero() -> None:
    """A legacy bearing.json written before this field exists must load with 0,
    not crash — backward compatibility."""
    b = bs.Bearing.from_dict({"current_frame": "f", "updated_at_turn": "x"})
    assert b.updated_at_turn_n == 0


def test_from_dict_coerces_bad_turn_n_to_zero() -> None:
    b = bs.Bearing.from_dict({"updated_at_turn_n": "not an int"})
    assert b.updated_at_turn_n == 0


def test_from_dict_user_model_coerces_bad_confidence() -> None:
    d = {"user_model": {"intent_read": "x", "register": "literal", "confidence": "not-a-float"}}
    b = bs.Bearing.from_dict(d)
    assert b.user_model is not None
    assert b.user_model.confidence == 0.0
