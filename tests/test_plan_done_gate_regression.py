# tests/test_plan_done_gate_regression.py
import pytest
from core import plans
from skills.plan import executor
from addons.system.bearing import compiler
from addons.system.bearing.schema import Bearing


@pytest.fixture
def db(tmp_path):
    plans.set_db_path(tmp_path / "tt.sqlite3")
    yield
    plans.set_db_path(None)
    compiler._PLAN_VIEW_CACHE.clear()


def test_flag_off_mark_completes_and_bearing_uses_fields(db, monkeypatch):
    monkeypatch.delenv("MONOLITH_PLAN_DONE_GATE_V1", raising=False)
    uid = plans.create_plan(goal="g", source="explicit",
                            steps=[{"verb": "v", "target": "t", "depends_on": []}])
    plans.set_plan_status(uid, "active")
    executor.run({"op": "mark", "step": 1, "status": "done"}, None)
    assert plans.get_plan(uid)["status"] == "done"            # old auto-done intact
    b = Bearing(trajectory="legacy traj", next_move="legacy nm")
    out = compiler.format_bearing_block(b, None, None, compiler._resolve_plan_view({"_turn_id": "x"}))
    assert "trajectory: legacy traj" in out                   # bearing fields, not plan
