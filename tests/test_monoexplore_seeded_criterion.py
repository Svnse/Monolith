# tests/test_monoexplore_seeded_criterion.py
import pytest
from core import plans, monoexplore


@pytest.fixture
def db(tmp_path):
    plans.set_db_path(tmp_path / "tt.sqlite3")
    yield
    plans.set_db_path(None)


def test_expedition_seeds_one_criterion_when_gate_on(db, monkeypatch):
    monkeypatch.setenv("MONOLITH_PLAN_DONE_GATE_V1", "1")
    # stub decompose so no LLM call is made
    monkeypatch.setattr(
        "core.planner.decompose",
        lambda goal, source="explicit", **kw: plans.get_plan(
            plans.create_plan(goal=goal, source=source,
                              steps=[{"verb": "look", "target": "x", "depends_on": []}])),
    )
    res = monoexplore.start_expedition("explore X", force=True)
    crits = plans.get_criteria(res["plan_uid"])
    assert len(crits) == 1
    assert "grounded finding" in crits[0]["criterion"]


def test_expedition_no_criteria_when_gate_off(db, monkeypatch):
    monkeypatch.delenv("MONOLITH_PLAN_DONE_GATE_V1", raising=False)
    monkeypatch.setattr(
        "core.planner.decompose",
        lambda goal, source="explicit", **kw: plans.get_plan(
            plans.create_plan(goal=goal, source=source,
                              steps=[{"verb": "look", "target": "x", "depends_on": []}])),
    )
    res = monoexplore.start_expedition("explore X", force=True)
    assert plans.get_criteria(res["plan_uid"]) == []  # byte-identical
