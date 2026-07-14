from __future__ import annotations

import pytest

_STEPS = [
    {"verb": "read", "target": "spec.md", "depends_on": []},
    {"verb": "draft", "target": "widget.py", "depends_on": [1]},
]


@pytest.fixture
def plan_store(monkeypatch, tmp_path):
    from core import plans
    plans.set_db_path(tmp_path / "turn_trace.sqlite3")
    yield plans
    plans.set_db_path(None)


def test_observer_surfaces_active_plan_when_flag_on(plan_store, monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_PLANNER_V1", "1")
    plan_store.create_plan(goal="ship the widget", source="explicit", steps=_STEPS)
    from addons.system.observer import runtime
    block = runtime.build_observer_snapshot(turn_id="t-plan").get("block", "")
    assert "active plan" in block.lower()
    assert "ship the widget" in block
    assert "read" in block  # the dependency-ready next step


def test_observer_silent_on_plan_when_flag_off(plan_store, monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_PLANNER_V1", "0")
    plan_store.create_plan(goal="ship the widget", source="explicit", steps=_STEPS)
    from addons.system.observer import runtime
    block = runtime.build_observer_snapshot(turn_id="t-plan2").get("block", "")
    assert "active plan" not in block.lower()
