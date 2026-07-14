# tests/test_bearing_plan_view.py
import pytest
from core import plans
from addons.system.bearing import compiler
from addons.system.bearing.schema import Bearing


@pytest.fixture
def db(tmp_path):
    plans.set_db_path(tmp_path / "tt.sqlite3")
    yield
    plans.set_db_path(None)
    compiler._PLAN_VIEW_CACHE.clear()


def test_format_flag_off_byte_identical():
    # plan_view=None (flag off) → renders the model-authored bearing fields verbatim
    b = Bearing(current_frame="cf", active_goal="ag", trajectory="model traj", next_move="model nm")
    out = compiler.format_bearing_block(b, None, None, None)
    assert "trajectory: model traj" in out
    assert "next_move: model nm" in out


def test_format_plan_view_overrides():
    b = Bearing(active_goal="ag", trajectory="model traj", next_move="model nm")
    pv = {"trajectory": "PLAN traj", "next_move": "PLAN nm"}
    out = compiler.format_bearing_block(b, None, None, pv)
    assert "trajectory: PLAN traj" in out and "model traj" not in out
    assert "next_move: PLAN nm" in out and "model nm" not in out


def test_format_plan_view_empty_omits_lines():
    b = Bearing(active_goal="ag", trajectory="model traj", next_move="model nm")
    pv = {"trajectory": "", "next_move": ""}
    out = compiler.format_bearing_block(b, None, None, pv)
    assert "trajectory:" not in out and "next_move:" not in out
    assert "active_goal: ag" in out  # the model-authored seed still renders


def test_resolve_plan_view_off_returns_none(db, monkeypatch):
    monkeypatch.delenv("MONOLITH_PLAN_DONE_GATE_V1", raising=False)
    assert compiler._resolve_plan_view({"_turn_id": "t1"}) is None


def test_resolve_plan_view_derives_and_freezes(db, monkeypatch):
    monkeypatch.setenv("MONOLITH_PLAN_DONE_GATE_V1", "1")
    uid = plans.create_plan(goal="G", source="explicit",
                            steps=[{"verb": "write", "target": "code", "depends_on": []}])
    plans.set_plan_status(uid, "active")
    cfg = {"_turn_id": "t1"}
    v1 = compiler._resolve_plan_view(cfg)
    assert "G" in v1["trajectory"] and "write code" in v1["trajectory"]
    assert v1["next_move"].startswith("step 1: write code")
    # freeze: marking a step does NOT change the view within the same outer turn
    plans.mark_step(uid, 1, "done")
    assert compiler._resolve_plan_view(cfg) == v1
    # new outer turn recomputes
    v2 = compiler._resolve_plan_view({"_turn_id": "t2"})
    assert "[done] write code" in v2["trajectory"]
