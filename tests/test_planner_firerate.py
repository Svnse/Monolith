"""DONE-GATE for M1 (spec §4). M1 is done when a realistic goal decomposes into
a valid, acyclic, persisted plan whose dependency frontier walks in order, and
propose_candidates() draws goals from the real sources (bearing + pulls).
"""
from __future__ import annotations

import pytest

from core import planner, plans

_REALISTIC = (
    "PLAN: add a retry policy to the HTTP client\n"
    "STEP: read | core/http_client.py | depends: none\n"
    "STEP: add | retry-with-backoff decorator | depends: 1\n"
    "STEP: write | tests/test_retry.py | depends: 2\n"
    "STEP: run | pytest tests/test_retry.py | depends: 3\n"
)


@pytest.fixture
def store(monkeypatch, tmp_path):
    plans.set_db_path(tmp_path / "turn_trace.sqlite3")
    from core import db_connect as _dbc
    monkeypatch.setattr(_dbc, "DB_PATH", tmp_path / "acatalepsy.sqlite3", raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    from core.acatalepsy import schema, canonical_log
    schema.migrate()
    tl = getattr(canonical_log, "_tl", None)
    if tl is not None:
        for a in ("writer_conn", "reader_conn"):
            if hasattr(tl, a):
                delattr(tl, a)
    yield
    plans.set_db_path(None)


def test_realistic_goal_decomposes_and_frontier_walks_in_order(store, monkeypatch) -> None:
    monkeypatch.setattr(planner, "_call_llm", lambda prompt: _REALISTIC)
    plan = planner.decompose("add a retry policy to the HTTP client", source="explicit")

    assert plan is not None
    assert len(plan["steps"]) == 4
    # acyclic: every dependency references an EARLIER step only
    for s in plan["steps"]:
        assert all(d < s["seq"] for d in s["depends_on"])

    uid = plan["plan_uid"]
    walked = []
    for _ in range(10):  # bounded
        ready = plans.next_ready_steps(uid)
        if not ready:
            break
        nxt = ready[0]
        walked.append(nxt["seq"])
        plans.mark_step(uid, nxt["seq"], "done")
    assert walked == [1, 2, 3, 4]  # frontier respected dependency order

    plans.set_plan_status(uid, "done")
    assert plans.get_active_plan() is None  # nothing open once done


def test_candidates_draw_from_bearing_and_pulls(store, monkeypatch) -> None:
    monkeypatch.setattr(planner, "_bearing_goal", lambda: "finish the M-arc")
    monkeypatch.setattr(planner, "_curiosity_pulls", lambda: [
        {"canonical": "monolith | values | precision", "pull_strength": 0.3},
        {"canonical": "monolith | holds | continuity", "pull_strength": 0.25},
    ])
    cands = planner.propose_candidates()
    assert any(c["source"] == "bearing" and "M-arc" in c["goal"] for c in cands)
    assert sum(1 for c in cands if c["source"] == "curiosity") == 2
