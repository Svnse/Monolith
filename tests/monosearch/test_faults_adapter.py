from unittest.mock import patch
from core import fault_response as fr
from core.monosearch.adapters.faults import FaultAdapter
from core.monosearch.record import EvidenceTier, Provenance

_REC = fr.FaultRecord(
    id=7, turn_id="t1", fault_kind="think_leak", detected_at="2026-06-02T18:00:00+00:00",
    detector_name="detect_think_leak", evidence="<think> leaked", metadata={"open": 1}, severity="warn",
)


def test_to_record_shape():
    a = FaultAdapter()
    with patch.object(fr, "read_recent", return_value=[_REC]):
        recs = a.search("", {}, 10)
    r = recs[0]
    assert r.namespaced_id == "fault:7"
    assert r.source == "fault_traces"
    assert r.provenance is Provenance.SELF
    assert r.evidence_tier == EvidenceTier.LITERAL
    assert r.ts == 1780423200.0  # 2026-06-02T18:00:00Z -> epoch
    assert r.metadata["fault_kind"] == "think_leak"


def test_recurrence_key_is_kind_level():
    a = FaultAdapter()
    assert a._recurrence_key(_REC) == "think_leak"  # kind-level, NOT evidence-hashed


def test_recurrence_key_aggregates_by_kind_across_different_evidence():
    # The discriminating test: same kind, DIFFERENT evidence (the prod reality) ->
    # SAME key, so `failing` aggregates "I keep tripping think_leak" instead of a
    # pile of count-1 entries.
    a = FaultAdapter()
    r1 = fr.FaultRecord(id=1, turn_id="t", fault_kind="think_leak", detected_at="x",
                        detector_name="d", evidence="<think> snippet A")
    r2 = fr.FaultRecord(id=2, turn_id="t", fault_kind="think_leak", detected_at="x",
                        detector_name="d", evidence="a totally different <think> snippet B")
    assert a._recurrence_key(r1) == a._recurrence_key(r2) == "think_leak"


def test_search_finds_matches_beyond_the_recent_window():
    # Seed matching faults FIRST (oldest), then many newer non-matching ones, then
    # search with a small limit. Pre-fix, read_recent(limit) fetched only the recent
    # N before filtering and missed the old matches.
    for i in range(3):
        fr.emit_fault(turn_id=f"old{i}", fault_kind="tool_no_fire",
                      detector_name="detect_tool_no_fire", evidence="stated a tool, none fired")
    for i in range(30):
        fr.emit_fault(turn_id=f"new{i}", fault_kind="think_leak",
                      detector_name="detect_think_leak", evidence="<think> leak")
    a = FaultAdapter()
    recs = a.search("tool", {}, 5)  # small limit; the tool faults are old (beyond recent 5)
    assert recs, "keyword search missed matches beyond the recent window"
    assert all(r.metadata["fault_kind"] == "tool_no_fire" for r in recs)


def test_get_parses_namespaced_id():
    a = FaultAdapter()
    with patch.object(fr, "read_one", return_value=_REC) as m:
        r = a.get("fault:7")
    m.assert_called_once_with(7)
    assert r.namespaced_id == "fault:7"
    assert a.get("acu:7") is None  # wrong namespace
