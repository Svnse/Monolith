"""bearing adapter (spec §5). Maps addons.system.bearing.audit JSONL rows to
Records. The store's only columns are {ts (ISO), turn_id, kind} plus per-kind
extra fields (failed_rules on rejected/grounding_failed, slots_changed on
applied, ...). Recurrence is the KIND — "rejected ×N" aggregates the bearing-cycle
signal; every row with a kind is salience-eligible (a kindless row -> None).

The unit tests hand-build rows; test_real_full_file_read seeds the REAL store via
the audit module's own _AUDIT_PATH monkeypatch (the pattern test_bearing_audit.py
uses) and drives the real read path end-to-end.
"""
from datetime import datetime, timezone

import pytest

from addons.system.bearing import audit
from core.monosearch.adapters.bearing import BearingAdapter
from core.monosearch.record import EvidenceTier, Provenance


def _iso(s: str) -> str:
    return s


# A rejected row -> recurrence_key is the kind ("rejected"), aggregating the
# bearing-cycle signal regardless of which rules failed.
_REJECTED = {
    "ts": "2026-06-02T18:00:00+00:00",
    "turn_id": "t1",
    "kind": "rejected",
    "failed_rules": ["D1", "D3"],
    "_seq": 0,
}
# An applied row also has a kind -> recurrence_key is "applied" (every row with
# a kind is salience-eligible now; the kind is the recurrence unit).
_APPLIED = {
    "ts": "2026-06-02T18:01:00+00:00",
    "turn_id": "t2",
    "kind": "applied",
    "slots_changed": ["current_frame"],
    "_seq": 1,
}


def test_to_record_shape():
    a = BearingAdapter()
    r = a._to_record(_REJECTED)
    assert r.namespaced_id == "bearing_audit:t1:0"
    assert r.source == "bearing"
    assert r.provenance is Provenance.SELF
    assert r.evidence_tier == EvidenceTier.DERIVED
    # ts ISO -> epoch
    assert r.ts == datetime(2026, 6, 2, 18, 0, 0, tzinfo=timezone.utc).timestamp()
    assert r.metadata["kind"] == "rejected"
    assert r.metadata["turn_id"] == "t1"
    assert r.metadata["failed_rules"] == ["D1", "D3"]


def test_recurrence_key_is_the_kind():
    a = BearingAdapter()
    # Aggregate on kind alone — "rejected ×N" is the bearing-cycle signal; the
    # exact failed-rule set no longer fragments it.
    assert a._recurrence_key(_REJECTED) == "rejected"


def test_recurrence_key_ignores_failed_rule_set():
    # Same kind, DIFFERENT rule sets -> SAME key, so "rejected" aggregates
    # instead of splitting into count-1 buckets per rule set.
    a = BearingAdapter()
    r1 = dict(_REJECTED, failed_rules=["D1", "D3"])
    r2 = dict(_REJECTED, failed_rules=["D7"])
    assert a._recurrence_key(r1) == a._recurrence_key(r2) == "rejected"


def test_recurrence_key_none_for_success_kinds():
    a = BearingAdapter()
    # applied/cleared are SUCCESSES — they must NOT recur into the signal.
    assert a._recurrence_key(_APPLIED) is None
    assert a._recurrence_key(dict(_APPLIED, kind="cleared")) is None


def test_recurrence_key_kind_survives_empty_failed_rules():
    a = BearingAdapter()
    # An empty failed_rules list no longer matters — the kind is the key.
    assert a._recurrence_key(dict(_REJECTED, failed_rules=[])) == "rejected"


def test_recurrence_key_none_when_no_kind():
    a = BearingAdapter()
    assert a._recurrence_key({"turn_id": "x", "_seq": 0}) is None


def test_id_uses_synthesized_line_seq():
    a = BearingAdapter()
    assert a._to_record(_APPLIED).namespaced_id == "bearing_audit:t2:1"


def test_real_full_file_read(monkeypatch, tmp_path):
    # Real end-to-end: write rows through the REAL audit.append path, read them
    # back through the REAL read_all, and map them through the adapter.
    monkeypatch.setattr(audit, "_AUDIT_PATH", tmp_path / "bearing.audit.jsonl")
    audit.append("rejected", turn_id="ta", failed_rules=["D1"])
    audit.append("applied", turn_id="tb", slots_changed=["next_move"])
    audit.append("grounding_failed", turn_id="tc", failed_rules=["G2"])

    a = BearingAdapter()
    recs = a.list({}, 100)
    assert len(recs) == 3
    ids = [r.namespaced_id for r in recs]
    assert ids == ["bearing_audit:ta:0", "bearing_audit:tb:1", "bearing_audit:tc:2"]

    # Recurrence eligibility: PROBLEM kinds key into the signal; the `applied`
    # success does NOT (recurrence_key None), so it can't dilute `recurring`.
    by_id = {r.namespaced_id: r for r in recs}
    assert by_id["bearing_audit:ta:0"].recurrence_key == "rejected"
    assert by_id["bearing_audit:tb:1"].recurrence_key is None  # applied = success
    assert by_id["bearing_audit:tc:2"].recurrence_key == "grounding_failed"

    # ts parsed off the real ISO timestamps.
    assert all(r.ts is not None for r in recs)

    # get() round-trips by namespaced id.
    g = a.get("bearing_audit:ta:0")
    assert g is not None and g.metadata["turn_id"] == "ta"
    assert a.get("fault:1") is None  # wrong namespace


def test_search_filters_by_query_and_since(monkeypatch, tmp_path):
    monkeypatch.setattr(audit, "_AUDIT_PATH", tmp_path / "bearing.audit.jsonl")
    audit.append("rejected", turn_id="ta", failed_rules=["D1"])
    audit.append("grounding_failed", turn_id="tb", failed_rules=["G2"])

    a = BearingAdapter()
    # keyword on kind/turn_id/rules
    hits = a.search("grounding", {}, 10)
    assert [r.metadata["turn_id"] for r in hits] == ["tb"]
    hits = a.search("D1", {}, 10)
    assert [r.metadata["turn_id"] for r in hits] == ["ta"]
