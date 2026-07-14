"""Tests for L3 TRUSTED promotion (Phase 6) — the retrieval-grade maturity tier.

L2 (crystallized, structurally valid) -> L3 (trusted / load-bearing) requires the
claim to have been externally `confirmed` at high confidence AND reinforced (seen
more than once). CID is unchanged: trust is earned, identity is not re-minted.
Promotion is wired into the verifier (post-confirm) and intake MATCH (post-reinforce).
"""
from __future__ import annotations

import json


def _setup(tmp_path, monkeypatch):
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "l3.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core.acatalepsy import schema, canonical_log
    schema.migrate()
    tl = getattr(canonical_log, "_tl", None)
    if tl is not None:
        for a in ("writer_conn", "reader_conn"):
            if hasattr(tl, a):
                delattr(tl, a)
    return _dbc


def _insert(dbc, *, l_level="L2", truth=None, truth_confidence=None, reinforcement=1,
            cid="cid:sha256:x", canonical="alpha | relates_to | beta", kind="world-fact",
            provenance="world"):
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        cur = conn.execute(
            "INSERT INTO acus(canonical, source, provenance, kind, l_level, reinforcement, "
            "cid, truth, truth_confidence, evidence_spans, state, cf_version, created_at, last_seen) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (canonical, provenance, provenance, kind, l_level, reinforcement,
             cid, truth, truth_confidence, "[]", "active", 1, "now", "now"))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _level(dbc, acu_id):
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        return conn.execute(
            "SELECT l_level FROM acus WHERE id=?", (acu_id,)).fetchone()["l_level"]
    finally:
        conn.close()


def _promote(dbc, acu_id):
    from core.acatalepsy.crystallize import maybe_promote_l3
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        ok = maybe_promote_l3(acu_id, conn)
        conn.commit()
        return ok
    finally:
        conn.close()


# ── unit: the promotion gate ───────────────────────────────────────────

def test_confirmed_reinforced_l2_promotes_to_l3(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert(dbc, truth="confirmed", truth_confidence=0.95, reinforcement=2)
    assert _promote(dbc, aid) is True
    assert _level(dbc, aid) == "L3"


def test_unconfirmed_does_not_promote(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert(dbc, truth=None, reinforcement=5)
    assert _promote(dbc, aid) is False
    assert _level(dbc, aid) == "L2"


def test_low_reinforcement_does_not_promote(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert(dbc, truth="confirmed", truth_confidence=0.95, reinforcement=1)
    assert _promote(dbc, aid) is False
    assert _level(dbc, aid) == "L2"


def test_low_confidence_does_not_promote(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert(dbc, truth="confirmed", truth_confidence=0.4, reinforcement=5)
    assert _promote(dbc, aid) is False
    assert _level(dbc, aid) == "L2"


def test_contested_does_not_promote(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert(dbc, truth="contested", truth_confidence=0.95, reinforcement=5)
    assert _promote(dbc, aid) is False


def test_archived_row_never_promotes(tmp_path, monkeypatch):
    # A merged/archived loser that is still l_level='L2'+confirmed must NOT be
    # promoted to a trusted tier — only active rows earn trust. Guard the
    # shared chokepoint, not just the call sites.
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert(dbc, truth="confirmed", truth_confidence=0.95, reinforcement=5)
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        conn.execute("UPDATE acus SET state='archived', merged_into=1 WHERE id=?", (aid,))
        conn.commit()
    finally:
        conn.close()
    assert _promote(dbc, aid) is False
    assert _level(dbc, aid) == "L2"


def test_l1_stub_never_promotes(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert(dbc, l_level="L1", cid=None, truth="confirmed",
                  truth_confidence=0.9, reinforcement=5)
    assert _promote(dbc, aid) is False
    assert _level(dbc, aid) == "L1"


def test_promotion_is_idempotent(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert(dbc, truth="confirmed", truth_confidence=0.95, reinforcement=3)
    assert _promote(dbc, aid) is True
    assert _promote(dbc, aid) is False   # already L3 -> no-op
    assert _level(dbc, aid) == "L3"


# ── integration: the verifier promotes on confirm ──────────────────────

def test_verifier_confirm_promotes_reinforced_claim_to_l3(tmp_path, monkeypatch):
    from core.acatalepsy.grounding import Evidence
    from core.acatalepsy import verifier
    dbc = _setup(tmp_path, monkeypatch)
    # reinforced (seen twice) but not yet truth-checked
    aid = _insert(dbc, truth=None, reinforcement=2, canonical="paris | capital_of | france")
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        verifier.verify_acu(
            aid, conn=conn,
            search_fn=lambda q: [Evidence("http://x", "Paris is the capital of France", 0.95)],
            judge_fn=lambda c, e: ("confirmed", 0.95))
        conn.commit()
    finally:
        conn.close()
    assert _level(dbc, aid) == "L3"   # confirmed + reinforced -> trusted


def test_verifier_confirm_does_not_promote_unreinforced(tmp_path, monkeypatch):
    from core.acatalepsy.grounding import Evidence
    from core.acatalepsy import verifier
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert(dbc, truth=None, reinforcement=1, canonical="berlin | capital_of | germany")
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        verifier.verify_acu(
            aid, conn=conn,
            search_fn=lambda q: [Evidence("http://x", "Berlin is the capital", 0.95)],
            judge_fn=lambda c, e: ("confirmed", 0.95))
        conn.commit()
    finally:
        conn.close()
    assert _level(dbc, aid) == "L2"   # confirmed but only seen once -> stays L2
