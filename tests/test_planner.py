from __future__ import annotations

import pytest

from core import planner, plans


@pytest.fixture
def store(monkeypatch, tmp_path):
    plans.set_db_path(tmp_path / "turn_trace.sqlite3")
    # isolate canonical_log so plan_proposed emits land in a tmp DB
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


_LLM_OK = (
    "PLAN: build the widget\n"
    "STEP: read | spec.md | depends: none\n"
    "STEP: draft | widget.py | depends: 1\n"
    "STEP: test | widget.py | depends: 2\n"
)


def test_propose_candidates_from_bearing_and_pulls(monkeypatch) -> None:
    monkeypatch.setattr(planner, "_bearing_goal", lambda: "ship V0")
    monkeypatch.setattr(planner, "_curiosity_pulls", lambda: [
        {"canonical": "monolith | values | precision", "pull_strength": 0.33},
    ])
    cands = planner.propose_candidates()
    sources = {c["source"] for c in cands}
    goals = " ".join(c["goal"] for c in cands)
    assert "bearing" in sources and "curiosity" in sources
    assert "ship V0" in goals
    assert "precision" in goals


def test_decompose_creates_valid_ordered_plan(store, monkeypatch) -> None:
    monkeypatch.setattr(planner, "_call_llm", lambda prompt: _LLM_OK)
    plan = planner.decompose("build the widget", source="explicit")
    assert plan is not None
    seqs = [s["seq"] for s in plan["steps"]]
    assert seqs == [1, 2, 3]
    assert plan["steps"][0]["verb"] == "read"
    assert plan["steps"][1]["depends_on"] == [1]
    assert plan["steps"][2]["depends_on"] == [2]
    # persisted + retrievable as the active plan
    assert plans.get_active_plan()["goal"] == "build the widget"


def test_decompose_emits_canonical_log(store, monkeypatch) -> None:
    monkeypatch.setattr(planner, "_call_llm", lambda prompt: _LLM_OK)
    planner.decompose("build the widget", source="curiosity")
    from core.acatalepsy import canonical_log
    kinds = [e.kind for e in canonical_log.read_since(0, limit=500)]
    assert "plan_proposed" in kinds


def test_decompose_rejects_empty_plan(store, monkeypatch) -> None:
    monkeypatch.setattr(planner, "_call_llm", lambda prompt: "PLAN: nothing\n(no steps)\n")
    assert planner.decompose("vague goal", source="explicit") is None
    assert plans.get_active_plan() is None


def test_decompose_drops_forward_dependencies(store, monkeypatch) -> None:
    bad = (
        "PLAN: x\n"
        "STEP: a | t1 | depends: 2\n"   # step 1 depends on LATER step 2 -> dropped
        "STEP: b | t2 | depends: 1\n"
    )
    monkeypatch.setattr(planner, "_call_llm", lambda prompt: bad)
    plan = planner.decompose("x", source="explicit")
    assert plan["steps"][0]["depends_on"] == []   # forward dep dropped
    assert plan["steps"][1]["depends_on"] == [1]
