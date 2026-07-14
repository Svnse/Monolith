# tests/test_plan_grounding.py
import pytest
from core import plans, plan_grounding


@pytest.fixture
def db(tmp_path):
    plans.set_db_path(tmp_path / "tt.sqlite3")
    yield
    plans.set_db_path(None)


def _make_plan():
    return plans.create_plan(
        goal="g", source="explicit",
        steps=[{"verb": "v", "target": "t", "depends_on": []}],
    )


def test_resolver_resolves_real_obs(db):
    uid = _make_plan()
    obs_id = plans.record_finding(uid, "grounded fact")
    resolve = plan_grounding.make_plan_resolver(uid)
    assert resolve(f"obs:{obs_id}") == 1


def test_resolver_rejects_unknown_and_malformed(db):
    uid = _make_plan()
    resolve = plan_grounding.make_plan_resolver(uid)
    assert resolve("obs:99999") is None      # no such obs
    assert resolve("R3") is None              # not an obs handle
    assert resolve("obs:abc") is None         # malformed
    assert resolve("") is None


def test_resolver_is_plan_scoped(db):
    uid_a = _make_plan()
    uid_b = _make_plan()
    obs_id = plans.record_finding(uid_a, "fact in A")
    assert plan_grounding.make_plan_resolver(uid_a)(f"obs:{obs_id}") == 1
    assert plan_grounding.make_plan_resolver(uid_b)(f"obs:{obs_id}") is None


def test_resolver_rejects_visited_breadcrumbs(db):
    uid = _make_plan()
    fid = plans.record_finding(uid, "real substantive evidence")  # obs id 1, kind=finding
    plans.record_observations(uid, "t1", visited=["list_files src/"], findings=[])  # obs id 2, kind=visited
    resolve = plan_grounding.make_plan_resolver(uid)
    assert resolve(f"obs:{fid}") == 1     # a real finding still resolves
    assert resolve("obs:2") is None       # a visited breadcrumb must NOT resolve (fail closed)
