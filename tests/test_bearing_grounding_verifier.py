from __future__ import annotations

from pathlib import Path

from addons.system.bearing import schema as bs
from addons.system.bearing.grounding_verifier import verify_grounding


# ── G1: observed referent.add must cite known tool_result ────────────


def test_g1_observed_referent_with_known_grounding_passes() -> None:
    envelope = {"referents": {"add": [
        {"name": "x", "kind": "entity", "status": "observed",
         "grounded_at_turn": "t1", "grounding": "tool_result_1", "reason": "r"}
    ]}}
    applied = bs.Bearing(referents=(
        bs.Referent(name="x", kind="entity", status="observed", grounded_at_turn="t1"),
    ))
    v = verify_grounding("t1", envelope, applied, tool_result_ids={"tool_result_1"})
    assert v.ok is True


def test_g1_observed_referent_unknown_grounding_fails() -> None:
    envelope = {"referents": {"add": [
        {"name": "x", "kind": "entity", "status": "observed",
         "grounding": "tool_result_99", "reason": "r"}
    ]}}
    applied = bs.Bearing(referents=(
        bs.Referent(name="x", kind="entity", status="observed", grounded_at_turn="t1"),
    ))
    v = verify_grounding("t1", envelope, applied, tool_result_ids={"tool_result_1"})
    assert v.ok is False
    assert "G1" in v.failed_rules
    assert 0 in v.downgrade_referent_indices


def test_g1_skipped_when_tool_result_ids_unknown() -> None:
    """Best-effort: caller passes no tool_result_ids → G1 is skipped."""
    envelope = {"referents": {"add": [
        {"name": "x", "kind": "entity", "status": "observed",
         "grounding": "tool_result_99", "reason": "r"}
    ]}}
    applied = bs.Bearing()
    v = verify_grounding("t1", envelope, applied, tool_result_ids=None)
    assert v.ok is True


def test_g1_unobserved_referent_skipped() -> None:
    envelope = {"referents": {"add": [
        {"name": "x", "kind": "entity", "status": "inferred",
         "grounding": "tool_result_99", "reason": "r"}
    ]}}
    applied = bs.Bearing()
    v = verify_grounding("t1", envelope, applied, tool_result_ids={"tool_result_1"})
    assert v.ok is True  # "inferred" doesn't trigger G1


# ── G2: open_tensions.resolve grounding ──────────────────────────────


def test_g2_resolve_with_known_grounding_passes() -> None:
    envelope = {"open_tensions": {"resolve": [
        {"index": 0, "reason": "r", "grounding": "tool_result_3"}
    ]}}
    applied = bs.Bearing()
    v = verify_grounding("t1", envelope, applied, tool_result_ids={"tool_result_3"})
    assert v.ok is True


def test_g2_resolve_unknown_grounding_fails() -> None:
    envelope = {"open_tensions": {"resolve": [
        {"index": 0, "reason": "r", "grounding": "tool_result_99"}
    ]}}
    applied = bs.Bearing()
    v = verify_grounding("t1", envelope, applied, tool_result_ids={"tool_result_3"})
    assert v.ok is False
    assert "G2" in v.failed_rules


# ── G3: file referent path existence ─────────────────────────────────


def test_g3_existing_file_passes(tmp_path) -> None:
    real_file = tmp_path / "real.txt"
    real_file.write_text("hello", encoding="utf-8")
    envelope = {"referents": {"add": [
        {"name": str(real_file), "kind": "file", "status": "observed",
         "grounded_at_turn": "t1", "reason": "r"}
    ]}}
    applied = bs.Bearing(referents=(
        bs.Referent(name=str(real_file), kind="file", status="observed", grounded_at_turn="t1"),
    ))
    v = verify_grounding("t1", envelope, applied)
    assert v.ok is True


def test_g3_missing_file_fails(tmp_path) -> None:
    missing = tmp_path / "does-not-exist.txt"
    envelope = {"referents": {"add": [
        {"name": str(missing), "kind": "file", "status": "observed",
         "grounded_at_turn": "t1", "reason": "r"}
    ]}}
    applied = bs.Bearing(referents=(
        bs.Referent(name=str(missing), kind="file", status="observed", grounded_at_turn="t1"),
    ))
    v = verify_grounding("t1", envelope, applied)
    assert v.ok is False
    assert "G3" in v.failed_rules
    assert 0 in v.downgrade_referent_indices


# ── G4: peer referent membership ─────────────────────────────────────


def test_g4_known_peer_passes() -> None:
    envelope = {"referents": {"add": [
        {"name": "claude-peer", "kind": "peer", "status": "observed",
         "grounded_at_turn": "t1", "reason": "r"}
    ]}}
    applied = bs.Bearing(referents=(
        bs.Referent(name="claude-peer", kind="peer", status="observed", grounded_at_turn="t1"),
    ))
    v = verify_grounding("t1", envelope, applied, connected_peers={"claude-peer"})
    assert v.ok is True


def test_g4_unknown_peer_fails() -> None:
    envelope = {"referents": {"add": [
        {"name": "ghost-peer", "kind": "peer", "status": "observed",
         "grounded_at_turn": "t1", "reason": "r"}
    ]}}
    applied = bs.Bearing(referents=(
        bs.Referent(name="ghost-peer", kind="peer", status="observed", grounded_at_turn="t1"),
    ))
    v = verify_grounding("t1", envelope, applied, connected_peers={"claude-peer"})
    assert v.ok is False
    assert "G4" in v.failed_rules


def test_g4_skipped_when_no_peer_context() -> None:
    envelope = {"referents": {"add": [
        {"name": "anyone", "kind": "peer", "status": "observed",
         "grounded_at_turn": "t1", "reason": "r"}
    ]}}
    applied = bs.Bearing()
    v = verify_grounding("t1", envelope, applied)
    assert v.ok is True


# ── empty envelope ───────────────────────────────────────────────────


def test_empty_envelope_passes() -> None:
    assert verify_grounding("t1", {}, bs.Bearing()).ok is True


def test_no_referents_passes() -> None:
    envelope = {"active_goal": {"new": "x", "reason": "r"}}
    assert verify_grounding("t1", envelope, bs.Bearing()).ok is True
