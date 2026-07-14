"""The composition seam: render_recall_lane assigns + registers handles -> a model
cites one -> parse_cites extracts it -> select resolves it through the LIVE
recall_handles map -> the cited handle wins at the rendered belief's authority.

Every piece is unit-tested in isolation; this is the one test that runs them
COMPOSED, so an ordinal-drift / map-coherence bug between render and resolve can't
hide in the seam. Also locks the documented scheme gap (un-handled grounds leashed).
"""
from datetime import datetime

from core import runtime_state_projection as rsp
from core import recall_handles, grounded_verdict
from core.acatalepsy.authority import compute_authority


def _seed(monkeypatch, rows):
    from core import acu_retrieval
    monkeypatch.setattr(acu_retrieval, "retrieve_relevant_acus", lambda p: [dict(r) for r in rows])
    monkeypatch.setattr(acu_retrieval, "_write_recall_hit", lambda a: None)


def test_render_to_cite_to_resolve_seam(monkeypatch):
    rows = [
        {"canonical": "channel switch | is | substantive", "locked": 1},          # -> AU_LOCKED
        {"canonical": "a weaker self belief | is | provisional", "l_level": "L1", "provenance": "self"},
    ]
    _seed(monkeypatch, rows)

    # 1. lane renders + registers R1, R2 against the live map
    lane = rsp.render_recall_lane([{"role": "user", "content": "q"}])
    assert "[R1]" in lane and "[R2]" in lane

    # 2. two model-style candidates citing different handles (R1 = the stronger ground)
    a = grounded_verdict.candidate_from_text("A", "answer A, grounded in [cite: R2]")
    b = grounded_verdict.candidate_from_text("B", "answer B, grounded in [cite: R1]")

    # 3. select resolves the cites through recall_handles.resolve (the live map)
    winner = grounded_verdict.select([a, b], recall_handles.resolve)

    # 4. the cite resolving to higher authority wins, AT the rendered belief's actual authority
    assert winner.candidate.id == "B"
    assert winner.winning_cite == "R1"
    assert winner.authority == compute_authority(rows[0])


def test_un_handled_ground_is_leashed_below_a_cited_handle(monkeypatch):
    # The documented scheme gap, locked as intended behavior: a conclusion grounded in
    # this-turn reasoning/observation (no recalled handle to cite) is leashed below one
    # that cites a recalled handle. Correct-by-design — the verdict prefers
    # recall/tool-grounded answers over pure-reasoned ones.
    _seed(monkeypatch, [{"canonical": "x | is | y", "locked": 1}])
    rsp.render_recall_lane([{"role": "user", "content": "q"}])

    reasoned = grounded_verdict.candidate_from_text("reasoned", "I reason this directly; no recalled ground applies.")
    cited = grounded_verdict.candidate_from_text("cited", "grounded answer [cite: R1]")
    winner = grounded_verdict.select([reasoned, cited], recall_handles.resolve)
    assert winner.candidate.id == "cited"   # handle-grounded wins; the reasoned one is leashed
