"""Tests for the Truth-branch Verifier + Tavily grounding client (Phase 3).

The verifier grounds world-fact/causal claims (deterministic internal
contradiction first, then Tavily + an injected judge), records a verdict, writes
a typed `contradicts` CCG edge, and flips confirmed falsehoods to state='-inf'.
search_fn/judge_fn are injected here; production wires Tavily + an LLM judge.
"""
from __future__ import annotations

import json

import pytest

from core.acatalepsy.grounding import Evidence, GroundingUnavailable, get_api_key, tavily_search


# ── grounding key resolution ──────────────────────────────────────────

def test_get_api_key_from_env(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "sk-test")
    assert get_api_key() == "sk-test"


def test_get_api_key_none_when_absent(monkeypatch, tmp_path):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    from core import paths
    monkeypatch.setattr(paths, "CONFIG_DIR", tmp_path, raising=False)
    assert get_api_key() is None


def test_tavily_search_raises_without_key(monkeypatch, tmp_path):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    from core import paths
    monkeypatch.setattr(paths, "CONFIG_DIR", tmp_path, raising=False)
    with pytest.raises(GroundingUnavailable):
        tavily_search("anything")


# ── verifier harness ──────────────────────────────────────────────────

def _setup(tmp_path, monkeypatch):
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "v.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core.acatalepsy import schema, canonical_log
    schema.migrate()
    tl = getattr(canonical_log, "_tl", None)
    if tl is not None:
        for a in ("writer_conn", "reader_conn"):
            if hasattr(tl, a):
                delattr(tl, a)
    return _dbc


def _insert(dbc, canonical, kind, provenance="world"):
    from core.acatalepsy.normalize import parse_triple
    t = parse_triple(canonical)
    ct = json.dumps({"entity_a": t.entity_a, "relation": t.relation,
                     "entity_b": t.entity_b, "qualifiers": t.qualifiers}) if t else None
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        cur = conn.execute(
            "INSERT INTO acus(canonical, source, provenance, kind, l_level, reinforcement, "
            "evidence_spans, canonical_triple, state, cf_version, created_at, last_seen) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
            (canonical, provenance, provenance, kind, "L2", 1, "[]", ct, "active", 1, "now", "now"))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def _verify(dbc, acu_id, search_fn, judge_fn):
    from core.acatalepsy import verifier
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        v = verifier.verify_acu(acu_id, conn=conn, search_fn=search_fn, judge_fn=judge_fn)
        conn.commit()
        return v
    finally:
        conn.close()


def _row(dbc, acu_id):
    conn = dbc.connect_acatalepsy(role="migration")
    try:
        return conn.execute("SELECT * FROM acus WHERE id=?", (acu_id,)).fetchone()
    finally:
        conn.close()


def _boom(q):
    raise AssertionError("grounding should not be called")


# ── verdicts ──────────────────────────────────────────────────────────

def test_world_fact_confirmed(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert(dbc, "paris | capital_of | france", "world-fact")
    v = _verify(dbc, aid,
                lambda q: [Evidence("http://x", "Paris is the capital of France", 0.9)],
                lambda c, e: ("confirmed", 0.95))
    assert v.verdict == "confirmed" and v.method == "tavily"
    row = _row(dbc, aid)
    assert row["truth"] == "confirmed"
    assert row["truth_confidence"] == 0.95
    assert row["evidence_url"] == "http://x"
    assert row["state"] == "active"


def test_world_fact_contradicted_flips_to_inf(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert(dbc, "trump | president_of | usa | 2005", "world-fact")
    v = _verify(dbc, aid,
                lambda q: [Evidence("u", "Trump was president 2017-2021", 0.9)],
                lambda c, e: ("contradicted", 0.9))
    assert v.verdict == "contradicted"
    row = _row(dbc, aid)
    assert row["truth"] == "contradicted"
    assert row["state"] == "-inf"   # confirmed falsehood excluded from recall


def test_internal_contradiction_flags_contested_no_grounding(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    existing = _insert(dbc, "trump | president_of | usa", "world-fact")
    aid = _insert(dbc, "trump | president_of | russia", "world-fact")
    v = _verify(dbc, aid, _boom, lambda c, e: ("confirmed", 1.0))
    # An internal conflict flags contested (not a unilateral -inf for the newcomer).
    assert v.verdict == "contested" and v.method == "internal"
    row = _row(dbc, aid)
    assert row["truth"] == "contested"
    assert row["state"] == "active"   # NOT flipped to -inf without external grounding
    conn = dbc.connect_acatalepsy(role="migration")
    edge = conn.execute(
        "SELECT target_id FROM acu_relations WHERE relation='contradicts' AND source_id=?",
        (aid,)).fetchone()
    conn.close()
    assert edge["target_id"] == existing


def test_verify_is_idempotent_no_duplicate_edge(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    _insert(dbc, "trump | president_of | usa", "world-fact")
    aid = _insert(dbc, "trump | president_of | russia", "world-fact")
    _verify(dbc, aid, _boom, lambda c, e: ("confirmed", 1.0))
    _verify(dbc, aid, _boom, lambda c, e: ("confirmed", 1.0))   # re-run
    conn = dbc.connect_acatalepsy(role="migration")
    n = conn.execute(
        "SELECT COUNT(*) FROM acu_relations WHERE relation='contradicts' AND source_id=?",
        (aid,)).fetchone()[0]
    conn.close()
    assert n == 1   # idempotent — no duplicate edge


def test_confirmed_requires_evidence(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert(dbc, "vague | claim | here", "world-fact")
    # judge says confirmed but there's NO evidence -> downgraded to unverifiable.
    v = _verify(dbc, aid, lambda q: [], lambda c, e: ("confirmed", 0.9))
    assert v.verdict == "unverifiable"


def test_unverifiable_when_grounding_unavailable(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert(dbc, "obscure | claim_about | thing", "world-fact")

    def unavailable(q):
        raise GroundingUnavailable("no key")

    v = _verify(dbc, aid, unavailable, lambda c, e: ("confirmed", 1.0))
    assert v.verdict == "unverifiable"
    assert _row(dbc, aid)["truth"] == "unverifiable"


def test_self_kind_is_not_truth_checked(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert(dbc, "monolith | uses | seven tiers", "self")
    v = _verify(dbc, aid, _boom, lambda c, e: ("confirmed", 1.0))
    assert v is None
    assert _row(dbc, aid)["truth"] is None


def test_causal_capped_at_contested(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    aid = _insert(dbc, "dislike | causes | preference", "causal")
    v = _verify(dbc, aid,
                lambda q: [Evidence("u", "some evidence", 0.5)],
                lambda c, e: ("confirmed", 0.9))
    assert v.verdict == "contested"   # causal confirmed -> contested ceiling
    assert _row(dbc, aid)["truth"] == "contested"


# ── batch + worker ────────────────────────────────────────────────────

def test_run_verifier_batch_only_checkable_kinds(tmp_path, monkeypatch):
    dbc = _setup(tmp_path, monkeypatch)
    wf = _insert(dbc, "paris | capital_of | france", "world-fact")
    _insert(dbc, "monolith | uses | tiers", "self")   # not checkable -> skipped
    from core.acatalepsy.verifier import run_verifier
    counts = run_verifier(
        limit=10,
        search_fn=lambda q: [Evidence("u", "ev", 0.9)],
        judge_fn=lambda c, e: ("confirmed", 0.9))
    assert counts["confirmed"] == 1
    assert _row(dbc, wf)["truth"] == "confirmed"


def test_verifier_worker_runs_then_stops(tmp_path, monkeypatch):
    import time
    dbc = _setup(tmp_path, monkeypatch)
    wf = _insert(dbc, "berlin | capital_of | germany", "world-fact")
    from core.acatalepsy.verifier import VerifierWorker
    w = VerifierWorker(
        poll_interval_secs=0.05, batch_limit=10,
        search_fn=lambda q: [Evidence("u", "ev", 0.9)],
        judge_fn=lambda c, e: ("confirmed", 0.95))
    w.start()
    try:
        verdict = None
        deadline = time.time() + 5.0
        while time.time() < deadline:
            verdict = _row(dbc, wf)["truth"]
            if verdict is not None:
                break
            time.sleep(0.05)
    finally:
        w.stop(timeout=2.0)
    assert verdict == "confirmed"
