import pytest
from core import plans
from skills.plan import executor


@pytest.fixture
def db(tmp_path):
    plans.set_db_path(tmp_path / "tt.sqlite3")
    yield
    plans.set_db_path(None)


def _active_plan():
    uid = plans.create_plan(
        goal="g", source="explicit",
        steps=[{"verb": "v", "target": "t", "depends_on": []}],
    )
    plans.set_plan_status(uid, "active")
    return uid


def test_mark_no_longer_auto_dones_when_gate_on(db, monkeypatch):
    monkeypatch.setenv("MONOLITH_PLAN_DONE_GATE_V1", "1")
    uid = _active_plan()
    executor.run({"op": "mark", "step": 1, "status": "done"}, None)
    assert plans.get_plan(uid)["status"] == "active"  # gate governs, not steps


def test_mark_still_auto_dones_when_gate_off(db, monkeypatch):
    monkeypatch.delenv("MONOLITH_PLAN_DONE_GATE_V1", raising=False)
    uid = _active_plan()
    executor.run({"op": "mark", "step": 1, "status": "done"}, None)
    assert plans.get_plan(uid)["status"] == "done"  # byte-identical old behavior


def test_full_explicit_completion_flow(db, monkeypatch):
    monkeypatch.setenv("MONOLITH_PLAN_DONE_GATE_V1", "1")
    uid = _active_plan()
    assert "1 success criteria" in executor.run(
        {"op": "criteria", "criteria": "the thing is verified"}, None)
    out = executor.run({"op": "ground", "ground": "ran it: works"}, None)
    assert "[cite: obs:" in out
    obs_id = out.split("obs:")[1].rstrip("]] ")
    executor.run({"op": "mark", "step": 1, "status": "done"}, None)
    assert "→ met" in executor.run(
        {"op": "attest", "seq": 1, "evidence": f"[cite: obs:{obs_id}]"}, None)
    assert "COMPLETE" in executor.run({"op": "complete"}, None)
    assert plans.get_plan(uid)["status"] == "done"


def test_complete_refuses_with_message(db, monkeypatch):
    monkeypatch.setenv("MONOLITH_PLAN_DONE_GATE_V1", "1")
    uid = _active_plan()
    executor.run({"op": "criteria", "criteria": "c1"}, None)
    executor.run({"op": "mark", "step": 1, "status": "done"}, None)
    out = executor.run({"op": "complete"}, None)
    assert "NOT complete" in out and "criteria not met" in out


def test_show_surfaces_citable_obs_handles(db, monkeypatch):
    monkeypatch.setenv("MONOLITH_PLAN_DONE_GATE_V1", "1")
    uid = _active_plan()
    executor.run({"op": "ground", "ground": "verified the output"}, None)
    out = executor.run({"op": "show"}, None)
    assert "obs:" in out and "verified the output" in out  # discoverable for attest
