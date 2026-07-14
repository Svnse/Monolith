"""Recall handles — the turn-scoped handle->ACU map that lets the grounded verdict
resolve a cited handle (R1) back to an Authority via compute_authority. The paired
half of the cite mechanism: render_recall_lane assigns + registers; the verdict
resolves."""
from datetime import datetime

from core import recall_handles
from core import runtime_state_projection as rsp
from core.acatalepsy.authority import compute_authority, AU_LOCKED


def test_resolve_returns_authority_for_registered_handle():
    recall_handles.reset()
    recall_handles.register("R1", {"locked": 1})
    assert recall_handles.resolve("R1") == AU_LOCKED


def test_unregistered_or_fabricated_handle_resolves_to_none():
    recall_handles.reset()
    recall_handles.register("R1", {"locked": 1})
    assert recall_handles.resolve("R9") is None   # never shown this turn -> no ground


def test_reset_clears_the_map():
    recall_handles.register("R1", {"locked": 1})
    recall_handles.reset()
    assert recall_handles.resolve("R1") is None


def test_render_recall_lane_emits_handles_and_registers_them(monkeypatch):
    from core import acu_retrieval
    rows = [
        {"canonical": "a | r | b", "locked": 1},
        {"canonical": "c | r | d", "l_level": "L1", "provenance": "self"},
    ]
    monkeypatch.setattr(acu_retrieval, "retrieve_relevant_acus", lambda prompt: list(rows))
    monkeypatch.setattr(acu_retrieval, "_write_recall_hit", lambda acus: None)

    out = rsp.render_recall_lane([{"role": "user", "content": "tell me"}])

    assert "[R1]" in out and "[R2]" in out
    # label+canonical stay contiguous (the existing recall test's invariant)
    assert "[LOCKED] a | r | b" in out
    # the handle resolves to exactly the authority the lane rendered
    assert recall_handles.resolve("R1") == compute_authority(rows[0])
    assert recall_handles.resolve("R2") == compute_authority(rows[1])


def test_render_resets_stale_handles_from_a_prior_turn(monkeypatch):
    recall_handles.register("R1", {"locked": 1})   # stale from "last turn"
    from core import acu_retrieval
    monkeypatch.setattr(acu_retrieval, "retrieve_relevant_acus", lambda prompt: [])
    monkeypatch.setattr(acu_retrieval, "_write_recall_hit", lambda acus: None)

    rsp.render_recall_lane([{"role": "user", "content": "tell me"}])

    assert recall_handles.resolve("R1") is None     # reset even when this turn recalls nothing
