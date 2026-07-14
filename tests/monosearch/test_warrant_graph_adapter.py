from __future__ import annotations

import json

import pytest

from core.monosearch.adapters.warrant_graph import WarrantGraphAdapter
from core.monosearch.record import EvidenceTier, Provenance


def _acu_row(**over):
    base = {
        "node_kind": "acu",
        "acu_id": 42,
        "claim_text": "sky | appears | blue",
        "acu_provenance": "user",
        "acu_source": "model",
        "acu_kind": "claim",
        "l_level": "L2",
        "acu_state": "active",
        "truth": "confirmed",
        "truth_confidence": 0.91,
        "truth_method": "tavily",
        "truth_checked_at": "2026-06-02T18:10:00+00:00",
        "evidence_url": "https://example.test/sky",
        "evidence_json": "[]",
        "acu_evidence_spans": "[]",
        "acu_source_event": 7,
        "reinforcement": 3,
        "cid": "CID-1",
        "eqid": None,
        "locked": 0,
        "lock_reason": None,
        "acu_created_at": "2026-06-02T18:00:00+00:00",
        "acu_last_seen": "2026-06-02T18:05:00+00:00",
        "candidate_id": 9,
        "candidate_claim": "sky | appears | blue",
        "evidence_log_id": 7,
        "evidence_char_start": 4,
        "evidence_char_end": 19,
        "evidence_span": "the sky looked blue",
        "candidate_source": "auditor_monolith",
        "warrant_text": "User made an observable claim.",
        "candidate_reinforcement_count": 1,
        "contradicts_acu_id": None,
        "candidate_state": "accepted",
        "candidate_created_at": "2026-06-02T18:01:00+00:00",
        "auditor_run_id": 3,
        "decision_id": 11,
        "decision": "accept",
        "decided_by": "user_e",
        "decided_at": "2026-06-02T18:02:00+00:00",
        "reject_reason": None,
        "edited_form": None,
        "decision_note": "good extraction",
        "resulting_acu_id": 42,
        "evidence_event_id": 7,
        "evidence_event_kind": "user_message",
        "evidence_event_ts": 1780423200.0,
        "evidence_session_id": "s1",
        "relation_count": 1,
        "relation_summary": "out:contradicts:acu:43:sky | appears | green",
    }
    base.update(over)
    return base


def _candidate_row(**over):
    base = _acu_row(
        node_kind="candidate",
        acu_id=None,
        claim_text="rain | causes | wet streets",
        acu_provenance=None,
        acu_source=None,
        acu_kind=None,
        l_level=None,
        acu_state=None,
        truth=None,
        truth_confidence=None,
        truth_method=None,
        evidence_url=None,
        acu_source_event=None,
        reinforcement=None,
        cid=None,
        eqid=None,
        candidate_id=15,
        candidate_claim="rain | causes | wet streets",
        candidate_state="pending",
        decision_id=None,
        decision=None,
        decided_by=None,
        decided_at=None,
        decision_note=None,
        resulting_acu_id=None,
        relation_count=0,
        relation_summary=None,
    )
    base.update(over)
    return base


def test_to_record_renders_acu_warrant_chain():
    adapter = WarrantGraphAdapter()
    rec = adapter._to_record(_acu_row())

    assert rec.namespaced_id == "warrant:acu:42"
    assert rec.source == "acatalepsy-warrants"
    assert rec.provenance is Provenance.USER
    assert rec.evidence_tier == EvidenceTier.DERIVED
    assert rec.recurrence_key is None
    assert "claim: sky | appears | blue" in rec.text
    assert "evidence: canonical_log:7 chars=4-19 span=the sky looked blue" in rec.text
    assert "warrant: User made an observable claim." in rec.text
    assert "decision: accept by=user_e" in rec.text
    assert "truth_evidence: method=tavily confidence=0.91" in rec.text
    assert "defeaters: relation=contradicts" in rec.text
    assert "- out:contradicts:acu:43:sky | appears | green" in rec.text
    assert rec.metadata["has_defeater"] is True
    assert rec.metadata["truth"] == "confirmed"


def test_to_record_renders_candidate_only_warrant_chain():
    adapter = WarrantGraphAdapter()
    rec = adapter._to_record(_candidate_row())

    assert rec.namespaced_id == "warrant:candidate:15"
    assert rec.provenance is Provenance.SELF
    assert "status: candidate_state=pending" in rec.text
    assert "decision: unresolved" in rec.text
    assert rec.metadata["node_kind"] == "candidate"
    assert rec.metadata["candidate_state"] == "pending"
    assert rec.metadata["acu_id"] is None


@pytest.fixture()
def seeded(tmp_path, monkeypatch):
    from core import db_connect as _dbc

    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acu.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")

    from core.acatalepsy import canonical_log, intake, schema

    schema.migrate()
    for mod in (canonical_log, intake):
        tl = getattr(mod, "_tl", None)
        if tl is not None:
            for attr in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(tl, attr):
                    delattr(tl, attr)

    conn = _dbc.connect_acatalepsy(role="migration")
    try:
        cur = conn.execute(
            "INSERT INTO canonical_log(ts, kind, session_id, acu_id, payload) VALUES(?,?,?,?,?)",
            (1780423200.0, "user_message", "s1", None, json.dumps({"text": "the sky looked blue"})),
        )
        evidence_event_id = int(cur.lastrowid)
        cur = conn.execute(
            """
            INSERT INTO acu_candidates(
                canonical_form, evidence_log_id, evidence_char_start, evidence_char_end,
                evidence_span, source, reason, reinforcement_count, contradicts_acu_id,
                state, created_at, auditor_run_id
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "sky | appears | blue",
                evidence_event_id,
                4,
                19,
                "the sky looked blue",
                "auditor_monolith",
                "User made an observable claim.",
                1,
                None,
                "accepted",
                "2026-06-02T18:01:00+00:00",
                1,
            ),
        )
        candidate_id = int(cur.lastrowid)
        cur = conn.execute(
            """
            INSERT INTO acus(
                canonical, source, provenance, kind, l_level, reinforcement, state, truth,
                truth_confidence, truth_method, evidence_spans, created_at, last_seen,
                last_touched_ts, candidate_id, locked
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "sky | appears | blue",
                "self",
                "user",
                "claim",
                "L2",
                3,
                "active",
                "confirmed",
                0.91,
                "tavily",
                "[]",
                "2026-06-02T18:00:00+00:00",
                "2026-06-02T18:05:00+00:00",
                "2026-06-02T18:05:00+00:00",
                candidate_id,
                0,
            ),
        )
        acu_id = int(cur.lastrowid)
        cur = conn.execute(
            """
            INSERT INTO acu_decisions(
                candidate_id, decision, decided_by, decided_at, reject_reason,
                edited_form, note, resulting_acu_id
            ) VALUES(?,?,?,?,?,?,?,?)
            """,
            (candidate_id, "accept", "user_e", "2026-06-02T18:02:00+00:00", None, None, "good", acu_id),
        )
        decision_id = int(cur.lastrowid)
        conn.execute("UPDATE acus SET decision_id=? WHERE id=?", (decision_id, acu_id))

        cur = conn.execute(
            """
            INSERT INTO acus(
                canonical, source, provenance, kind, l_level, reinforcement, state,
                evidence_spans, created_at, last_seen, last_touched_ts, locked
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "sky | appears | green",
                "self",
                "self",
                "claim",
                "L1",
                1,
                "active",
                "[]",
                "2026-06-02T18:03:00+00:00",
                "2026-06-02T18:03:00+00:00",
                "2026-06-02T18:03:00+00:00",
                0,
            ),
        )
        other_acu_id = int(cur.lastrowid)
        conn.execute(
            """
            INSERT INTO acu_relations(source_id, target_id, relation, score, created_at, updated_at)
            VALUES(?,?,?,?,?,?)
            """,
            (acu_id, other_acu_id, "contradicts", 0.8, "2026-06-02T18:04:00+00:00", None),
        )

        cur = conn.execute(
            """
            INSERT INTO acu_candidates(
                canonical_form, evidence_log_id, evidence_char_start, evidence_char_end,
                evidence_span, source, reason, reinforcement_count, contradicts_acu_id,
                state, created_at, auditor_run_id
            ) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                "rain | causes | wet streets",
                evidence_event_id,
                0,
                8,
                "rain fell",
                "auditor_monolith",
                "Candidate is useful but not reviewed yet.",
                1,
                acu_id,
                "pending",
                "2026-06-02T18:06:00+00:00",
                2,
            ),
        )
        pending_candidate_id = int(cur.lastrowid)
        conn.commit()
    finally:
        conn.close()

    return {
        "acu": acu_id,
        "other_acu": other_acu_id,
        "candidate": candidate_id,
        "pending_candidate": pending_candidate_id,
    }


def test_e2e_get_acu_warrant_chain(seeded):
    adapter = WarrantGraphAdapter()
    rec = adapter.get(f"warrant:acu:{seeded['acu']}")

    assert rec is not None
    assert rec.namespaced_id == f"warrant:acu:{seeded['acu']}"
    assert "claim: sky | appears | blue" in rec.text
    assert "evidence: canonical_log:1 chars=4-19 span=the sky looked blue" in rec.text
    assert "warrant: User made an observable claim." in rec.text
    assert "decision: accept by=user_e" in rec.text
    assert f"out:contradicts:acu:{seeded['other_acu']}:sky | appears | green" in rec.text
    assert rec.metadata["relation_count"] == 1
    assert rec.metadata["has_defeater"] is True


def test_e2e_get_candidate_warrant_chain(seeded):
    adapter = WarrantGraphAdapter()
    rec = adapter.get(f"warrant:candidate:{seeded['pending_candidate']}")

    assert rec is not None
    assert rec.namespaced_id == f"warrant:candidate:{seeded['pending_candidate']}"
    assert "claim: rain | causes | wet streets" in rec.text
    assert "status: candidate_state=pending" in rec.text
    assert "decision: unresolved" in rec.text
    assert rec.metadata["has_defeater"] is True
    assert rec.metadata["contradicts_acu_id"] == seeded["acu"]


def test_e2e_search_and_filters(seeded):
    adapter = WarrantGraphAdapter()

    hits = adapter.search("observable claim", {}, 10)
    assert any(r.namespaced_id == f"warrant:acu:{seeded['acu']}" for r in hits)

    confirmed = adapter.search("", {"truth": "confirmed", "node_kind": "acu"}, 10)
    assert {r.namespaced_id for r in confirmed} == {f"warrant:acu:{seeded['acu']}"}

    pending = adapter.search("", {"state": "pending", "node_kind": "candidate"}, 10)
    assert {r.namespaced_id for r in pending} == {f"warrant:candidate:{seeded['pending_candidate']}"}

    contradicts = adapter.search("", {"relation": "contradicts"}, 10)
    assert {r.namespaced_id for r in contradicts} == {
        f"warrant:acu:{seeded['acu']}",
        f"warrant:acu:{seeded['other_acu']}",
    }


def test_get_rejects_wrong_ids(seeded):
    adapter = WarrantGraphAdapter()

    assert adapter.get("warrant:bad:1") is None
    assert adapter.get("warrant:acu:not-int") is None
    assert adapter.get("acu:1") is None
