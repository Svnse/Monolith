"""V1 single-answer grounded verdict — the independent error signal at finalize.

`verdict_for_turn` parses the answer's cites, resolves each against THIS turn's
recall handles, and splits grounded vs FABRICATED (cited-but-unresolvable = the
laundering fault). The load-bearing distinction is fabricated vs no-ground: both
mean "no resolved ground", but fabricated is a fault (claimed a ground that does
not exist) and no-ground is honest behavior. Only fabricated feeds Self-Check.
"""
from datetime import datetime, timezone

import pytest

from core import grounded_verdict as gv
from core import recall_handles


def _resolver(mapping):
    return lambda h: mapping.get(h)


def _fresh() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── the four buckets ─────────────────────────────────────────────────────────


def test_grounded_when_a_cite_resolves():
    v = gv.verdict_for_turn("X holds. [cite: R1]", _resolver({"R1": 3}))
    assert v.grounded is True
    assert v.authority == 3
    assert v.winning_cite == "R1"
    assert v.fabricated == ()
    assert v.no_ground is False


def test_authority_is_best_of_multiple_grounds():
    v = gv.verdict_for_turn("[cite: R1] [cite: R2]", _resolver({"R1": 2, "R2": 4}))
    assert v.authority == 4
    assert v.winning_cite == "R2"
    assert v.grounded is True


def test_ungrounded_when_no_cites_and_no_token():
    v = gv.verdict_for_turn("just an answer, no citation", _resolver({}))
    assert v.grounded is False
    assert v.fabricated == ()
    assert v.no_ground is False


def test_honest_no_ground_token():
    v = gv.verdict_for_turn("nothing recalled supports this. [no-ground]", _resolver({}))
    assert v.no_ground is True
    assert v.fabricated == ()
    assert v.grounded is False


# ── the load-bearing boundary: fabricated vs no-ground ───────────────────────


def test_fabricated_when_cite_resolves_to_none():
    # Cited a handle never shown this turn -> resolves to None -> FAULT.
    v = gv.verdict_for_turn("the answer is X [cite: R7]", _resolver({"R1": 3}))
    assert v.fabricated == ("R7",)
    assert v.grounded is False
    # CRITICAL: a fabricated cite is NOT honest no-ground.
    assert v.no_ground is False


def test_fabricated_is_distinct_from_no_ground():
    # Adjacent failure modes — both have grounded=False, but only one is a fault.
    # If these collapse, the prize bucket dies silently (fault read as honesty).
    fabricated = gv.verdict_for_turn("X [cite: R9]", _resolver({}))
    honest = gv.verdict_for_turn("X [no-ground]", _resolver({}))
    assert fabricated.fabricated == ("R9",) and fabricated.no_ground is False
    assert honest.fabricated == () and honest.no_ground is True
    assert fabricated.fabricated != honest.fabricated
    assert fabricated.no_ground != honest.no_ground


def test_mixed_grounded_and_fabricated():
    v = gv.verdict_for_turn("X [cite: R1] and Y [cite: R7]", _resolver({"R1": 4}))
    assert v.grounded is True and v.authority == 4
    assert v.fabricated == ("R7",)


def test_real_recall_resolver_gives_none_for_unshown_handle():
    # End-to-end with the PROD resolver (not a mock): the whole fault signal rests
    # on recall_handles.resolve returning None (not falsy / not throwing) for a
    # handle never registered this turn.
    recall_handles.reset()
    recall_handles.register("R1", {"truth": "confirmed",
                                   "truth_checked_at": _fresh(), "state": "active"})  # AU3
    v = gv.verdict_for_turn("grounded [cite: R1], bad [cite: R7]", recall_handles.resolve)
    assert v.winning_cite == "R1" and v.authority == 3
    assert v.fabricated == ("R7",)   # never shown -> None -> fabricated, not no-ground


# ── flag (dark by default) ───────────────────────────────────────────────────


def test_flag_off_by_default(monkeypatch):
    monkeypatch.delenv("MONOLITH_GROUNDED_VERDICT_V1", raising=False)
    assert gv.grounded_verdict_enabled() is False


def test_flag_on(monkeypatch):
    monkeypatch.setenv("MONOLITH_GROUNDED_VERDICT_V1", "1")
    assert gv.grounded_verdict_enabled() is True


# ── KNOWN_KINDS includes the fabricated-cite fault (else emit_fault drops it) ──


def test_fabricated_cite_is_a_known_fault_kind():
    from core.fault_response import KNOWN_KINDS
    assert "fabricated_cite" in KNOWN_KINDS


# ── born-alive consumer: ONLY fabricated-cite feeds the fault loop ────────────


def test_hook_feeds_only_fabricated_cite_to_fault_loop(monkeypatch):
    monkeypatch.setenv("MONOLITH_GROUNDED_VERDICT_V1", "1")
    import core.chat_finalize as cf
    import core.fault_response as fr
    import core.turn_trace as tt

    calls = []
    monkeypatch.setattr(fr, "emit_fault", lambda *a, **k: (calls.append((a, k)), 1)[1])
    monkeypatch.setattr(tt, "record_grounded_verdict", lambda *a, **k: None)
    recall_handles.reset()
    recall_handles.register("R1", {"truth": "confirmed",
                                   "truth_checked_at": _fresh(), "state": "active"})  # AU3

    cf._record_grounded_verdict_for_turn("ans [cite: R1]", {"_turn_id": "t1"}, None)   # grounded
    cf._record_grounded_verdict_for_turn("ans [no-ground]", {"_turn_id": "t2"}, None)  # honest
    cf._record_grounded_verdict_for_turn("ans no cite", {"_turn_id": "t3"}, None)      # ungrounded
    assert calls == []   # none of the three is a fault

    cf._record_grounded_verdict_for_turn("ans [cite: R7]", {"_turn_id": "t4"}, None)   # fabricated
    assert len(calls) == 1
    args, _kwargs = calls[0]
    assert args[0] == "t4"
    assert args[1] == "fabricated_cite"


def test_hook_noop_when_flag_off(monkeypatch):
    monkeypatch.delenv("MONOLITH_GROUNDED_VERDICT_V1", raising=False)
    import core.chat_finalize as cf
    import core.fault_response as fr

    calls = []
    monkeypatch.setattr(fr, "emit_fault", lambda *a, **k: (calls.append(a), 1)[1])
    recall_handles.reset()
    cf._record_grounded_verdict_for_turn("ans [cite: R7]", {"_turn_id": "t9"}, None)
    assert calls == []   # flag off -> nothing fires


def test_finalize_assistant_turn_invokes_the_hook(monkeypatch):
    # Wiring proof: the hook must be CALLED from finalize_assistant_turn (the live
    # entry), not merely callable in isolation. A fabricated cite passed through
    # the real entry must reach emit_fault.
    monkeypatch.setenv("MONOLITH_GROUNDED_VERDICT_V1", "1")
    from core.chat_finalize import finalize_assistant_turn
    import core.fault_response as fr
    import core.turn_trace as tt

    calls = []
    monkeypatch.setattr(fr, "emit_fault", lambda *a, **k: (calls.append(a), 1)[1])
    monkeypatch.setattr(tt, "record_grounded_verdict", lambda *a, **k: None)
    recall_handles.reset()
    recall_handles.register("R1", {"truth": "confirmed",
                                   "truth_checked_at": _fresh(), "state": "active"})

    finalize_assistant_turn(
        raw="the answer [cite: R1]", public="the answer",
        config={"_turn_id": "wt1"},
        emit_pipeline_ready=lambda *a: None, record_verdict=lambda *a: None,
    )
    assert calls == []   # grounded -> no fault

    finalize_assistant_turn(
        raw="the answer [cite: R7]", public="the answer",
        config={"_turn_id": "wt2"},
        emit_pipeline_ready=lambda *a: None, record_verdict=lambda *a: None,
    )
    assert len(calls) == 1 and calls[0][1] == "fabricated_cite"   # wired + fault fired


# ── real-store landing: frame stamp + a real fabricated_cite fault row ───────


@pytest.fixture
def trace_db(tmp_path, monkeypatch):
    from core import turn_trace as tt
    db = tmp_path / "turn_trace.sqlite3"
    tt.set_db_path(db)
    monkeypatch.setenv("MONOLITH_TURN_TRACE_V1", "1")
    monkeypatch.setenv("MONOLITH_GROUNDED_VERDICT_V1", "1")
    yield db
    tt.set_db_path(None)


def _write_frame(turn_id):
    from core import turn_trace as tt
    tt.record_frame(tt.FrameTraceRecord(
        turn_id=turn_id, parent_turn_id=None, captured_at="2026-06-07T00:00:00+00:00",
        backend="gguf", engine_key="llm", gen_id=1, final_messages=(),
        system_prompt_chars=0, user_prompt_chars=0, total_chars=0,
        metadata={"model_loaded": True},
    ))


def test_record_grounded_verdict_stamps_frame_additively(trace_db):
    from core import turn_trace as tt
    _write_frame("gv-1")
    tt.record_grounded_verdict("gv-1", {"grounded": True, "authority": 3, "winning_cite": "R1",
                                        "cited": ["R1"], "fabricated": [], "no_ground": False})
    meta = tt.get_turn_trace("gv-1").frame.metadata
    assert meta["grounded_verdict"]["authority"] == 3
    assert meta["grounded_verdict"]["winning_cite"] == "R1"
    assert meta["model_loaded"] is True   # existing frame metadata preserved


def test_full_chain_fabricated_writes_real_fault_row(trace_db):
    # Born-alive proof on a REAL store: fabricated cite -> a real fabricated_cite
    # row in fault_traces (the table Self-Check drains) + frame stamp, with the
    # fabricated/no-ground distinction intact (not coerced, not swallowed).
    import core.chat_finalize as cf
    from core import turn_trace as tt
    from core.fault_response import read_by_kind
    _write_frame("gv-fab")
    recall_handles.reset()
    recall_handles.register("R1", {"truth": "confirmed",
                                   "truth_checked_at": _fresh(), "state": "active"})

    cf._record_grounded_verdict_for_turn("answer [cite: R7]", {"_turn_id": "gv-fab"}, None)

    meta = tt.get_turn_trace("gv-fab").frame.metadata["grounded_verdict"]
    assert meta["fabricated"] == ["R7"] and meta["no_ground"] is False and meta["grounded"] is False
    assert any(f.turn_id == "gv-fab" for f in read_by_kind("fabricated_cite"))


def test_full_chain_grounded_records_verdict_no_fault(trace_db):
    import core.chat_finalize as cf
    from core import turn_trace as tt
    from core.fault_response import read_by_kind
    _write_frame("gv-ok")
    recall_handles.reset()
    recall_handles.register("R1", {"truth": "confirmed",
                                   "truth_checked_at": _fresh(), "state": "active"})

    cf._record_grounded_verdict_for_turn("answer [cite: R1]", {"_turn_id": "gv-ok"}, None)

    meta = tt.get_turn_trace("gv-ok").frame.metadata["grounded_verdict"]
    assert meta["grounded"] is True and meta["authority"] == 3 and meta["fabricated"] == []
    assert not any(f.turn_id == "gv-ok" for f in read_by_kind("fabricated_cite"))
