"""Recall ranks by Authority (closes A2) — AU3 behavior-shaping outranks AU2
recall-eligible, and AU1 stored-only (L1 stubs / -inf falsehoods) is excluded.
"""
from __future__ import annotations


def _setup(tmp_path, monkeypatch):
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "r.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    monkeypatch.setenv("MONOLITH_ACU_RECALL_V1", "1")
    from core.acatalepsy import schema
    schema.migrate()
    return _dbc


def _insert(dbc, canonical, *, kind, l_level, provenance, reinforcement=1, truth=None, state="active"):
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        conn.execute(
            "INSERT INTO acus(canonical, source, provenance, kind, l_level, reinforcement, "
            "truth, state, evidence_spans, cf_version, created_at, last_seen, last_touched_ts) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (canonical, provenance, provenance, kind, l_level, reinforcement, truth, state,
             "[]", 1, "now", "now", "now"))
        conn.commit()
    finally:
        conn.close()


def test_recall_ranks_by_authority_and_excludes_au1(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    # AU3: confirmed world-fact (behavior-shaping)
    _insert(dbc, "paris | capital_of | france", kind="world-fact", l_level="L2",
            provenance="world", reinforcement=1, truth="confirmed")
    # AU2: a kind=self fact that is USER-sourced (externally grounded) stays
    # recall-eligible — capped at AU_RECALL by the authority seal, not suppressed by
    # the recall seal (which only drops self+self). This is the "user-taught self
    # fact stays recallable" behaviour.
    _insert(dbc, "paris | is | beautiful", kind="self", l_level="L2",
            provenance="user", reinforcement=1)
    # AU1: L1 stub (stored-only, not recall-grade)
    _insert(dbc, "paris | has | cafes", kind="self", l_level="L1", provenance="user")

    from core.acu_retrieval import retrieve_relevant_acus
    results = retrieve_relevant_acus(
        "tell me about paris and france and beautiful cafes please")
    cans = [r["canonical"] for r in results]

    assert "paris | capital_of | france" in cans     # AU3 surfaces
    assert "paris | is | beautiful" in cans           # AU2 (self+user) surfaces
    assert "paris | has | cafes" not in cans          # AU1 L1 stub excluded
    assert cans[0] == "paris | capital_of | france"   # AU3 ranked above AU2


def test_self_self_excluded_from_world_answer_recall(tmp_path, monkeypatch):
    # SEAL: pure self-reinforcement (kind=self AND provenance=self) is identity
    # memory, not a world-answer ground — it must not surface in the recall lane.
    dbc = _setup(tmp_path, monkeypatch)
    _insert(dbc, "paris | is | beautiful", kind="self", l_level="L2",
            provenance="self", reinforcement=3)
    from core.acu_retrieval import retrieve_relevant_acus
    results = retrieve_relevant_acus("tell me whether paris is beautiful or not really")
    assert all("beautiful" not in r["canonical"] for r in results)


def test_self_user_fact_stays_recallable(tmp_path, monkeypatch):
    # Counterpart: the SAME claim, user-sourced, IS recallable (only self+self sealed).
    dbc = _setup(tmp_path, monkeypatch)
    _insert(dbc, "monolith | runs_on | opus", kind="self", l_level="L2",
            provenance="user", reinforcement=3)
    from core.acu_retrieval import retrieve_relevant_acus
    results = retrieve_relevant_acus("which model does monolith run on, is it opus")
    assert any("opus" in r["canonical"] for r in results)


def test_inf_falsehood_never_recalled(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    _insert(dbc, "trump | president_of | usa | 1800", kind="world-fact", l_level="L2",
            provenance="world", truth="contradicted", state="-inf")
    from core.acu_retrieval import retrieve_relevant_acus
    results = retrieve_relevant_acus("was trump president of the usa in 1800 or not")
    assert all("1800" not in r["canonical"] for r in results)


def test_score_acu_decay_differentiates_within_recency_bucket(monkeypatch):
    """When-plane decay: with MONOLITH_ACU_DECAY_V1 on, _score_acu gives a
    continuous recency gradient where the binary 7-day _recency_bonus is flat.
    A 30-day-old and a 300-day-old confirmed claim must NOT score equally (both
    are >7d, so the old binary bonus gave both 0.0)."""
    from datetime import datetime, timezone, timedelta
    from core import acu_retrieval
    monkeypatch.setenv("MONOLITH_ACU_DECAY_V1", "1")
    now = datetime(2026, 6, 2, tzinfo=timezone.utc)

    def _row(days: int) -> dict:
        return {
            "canonical": "widgets are blue",
            "reinforcement": 8, "provenance": "self",
            "state": "active", "truth": "confirmed", "l_level": "L2",
            "last_touched_ts": (now - timedelta(days=days)).isoformat(),
        }

    toks = acu_retrieval._tokenize("tell me about widgets")
    recent = acu_retrieval._score_acu(_row(30), toks, now=now)
    ancient = acu_retrieval._score_acu(_row(300), toks, now=now)
    assert recent > ancient
