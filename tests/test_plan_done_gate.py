# tests/test_plan_done_gate.py
from core import plans


def test_done_gate_default_off(monkeypatch):
    monkeypatch.delenv("MONOLITH_PLAN_DONE_GATE_V1", raising=False)
    assert plans.done_gate_enabled() is False


def test_done_gate_on(monkeypatch):
    monkeypatch.setenv("MONOLITH_PLAN_DONE_GATE_V1", "1")
    assert plans.done_gate_enabled() is True


import pytest
from core import plans


@pytest.fixture
def db(tmp_path):
    plans.set_db_path(tmp_path / "tt.sqlite3")
    yield
    plans.set_db_path(None)


def _make_plan():
    return plans.create_plan(
        goal="ship feature X", source="explicit",
        steps=[{"verb": "write", "target": "code", "depends_on": []}],
    )


def test_set_and_get_criteria(db):
    uid = _make_plan()
    n = plans.set_criteria(uid, ["tests pass", "docs updated", "  "])
    assert n == 2  # blank dropped
    crits = plans.get_criteria(uid)
    assert [c["criterion"] for c in crits] == ["tests pass", "docs updated"]
    assert all(c["status"] == "open" for c in crits)
    assert [c["seq"] for c in crits] == [1, 2]


def test_set_criteria_replaces(db):
    uid = _make_plan()
    plans.set_criteria(uid, ["a", "b"])
    plans.set_criteria(uid, ["c"])
    assert [c["criterion"] for c in plans.get_criteria(uid)] == ["c"]


def test_get_criteria_unknown_plan(db):
    assert plans.get_criteria("nope") == []


def test_completed_at_column_exists(db):
    uid = _make_plan()
    p = plans.get_plan(uid)  # must not raise after migration
    assert p is not None


def test_record_finding_returns_citable_id(db):
    uid = _make_plan()
    obs_id = plans.record_finding(uid, "ran pytest: 12 passed")
    assert isinstance(obs_id, int) and obs_id > 0
    row = plans.get_observation(uid, obs_id)
    assert row["content"] == "ran pytest: 12 passed"
    assert row["kind"] == "finding"


def test_record_finding_blank_and_unknown(db):
    uid = _make_plan()
    assert plans.record_finding(uid, "   ") is None
    assert plans.record_finding("nope", "x") is None


def test_get_observation_cross_plan_isolation(db):
    uid_a = _make_plan()
    uid_b = _make_plan()
    obs_id = plans.record_finding(uid_a, "finding in A")
    assert plans.get_observation(uid_a, obs_id) is not None
    assert plans.get_observation(uid_b, obs_id) is None  # belongs to A only


def test_list_findings_surfaces_citable_ids(db):
    uid = _make_plan()
    a = plans.record_finding(uid, "first")
    b = plans.record_finding(uid, "second")
    found = plans.list_findings(uid)
    assert [f["id"] for f in found] == [a, b]   # chronological, with ids
    assert found[0]["content"] == "first"
    assert plans.list_findings("nope") == []


# ── Task 4: attest_criterion ───────────────────────────────────────────────────
from core import plan_grounding


def test_attest_grounded_marks_met(db):
    uid = _make_plan()
    plans.set_criteria(uid, ["did the thing"])
    obs_id = plans.record_finding(uid, "evidence the thing happened")
    resolve = plan_grounding.make_plan_resolver(uid)
    crit = plans.attest_criterion(uid, 1, f"done — see [cite: obs:{obs_id}]", resolve)
    assert crit["status"] == "met"
    assert crit["cite_handle"] == f"obs:{obs_id}"
    assert crit["ground_kind"] == "obs"


def test_attest_fabricated_marks_failed(db):
    uid = _make_plan()
    plans.set_criteria(uid, ["did the thing"])
    resolve = plan_grounding.make_plan_resolver(uid)
    crit = plans.attest_criterion(uid, 1, "trust me [cite: obs:9999]", resolve)
    assert crit["status"] == "failed"
    assert crit["cite_handle"] is None


def test_attest_no_ground_stays_open(db):
    uid = _make_plan()
    plans.set_criteria(uid, ["did the thing"])
    resolve = plan_grounding.make_plan_resolver(uid)
    crit = plans.attest_criterion(uid, 1, "no evidence yet [no-ground]", resolve)
    assert crit["status"] == "open"


def test_attest_unknown_seq(db):
    uid = _make_plan()
    plans.set_criteria(uid, ["c1"])
    resolve = plan_grounding.make_plan_resolver(uid)
    assert plans.attest_criterion(uid, 9, "x [cite: obs:1]", resolve) == {}


# ── Task 5: complete_plan ──────────────────────────────────────────────────────
def _grounded_met(uid, seq):
    obs_id = plans.record_finding(uid, f"evidence for {seq}")
    plans.attest_criterion(uid, seq, f"[cite: obs:{obs_id}]", plan_grounding.make_plan_resolver(uid))


def test_complete_refuses_incomplete_steps(db):
    uid = _make_plan()
    plans.set_criteria(uid, ["c1"])
    _grounded_met(uid, 1)
    audit = plans.complete_plan(uid)
    assert audit["ok"] is False and audit["reason"] == "steps_incomplete"
    assert plans.get_plan(uid)["status"] != "done"


def test_complete_refuses_zero_criteria(db):
    uid = _make_plan()
    plans.mark_step(uid, 1, "done")
    audit = plans.complete_plan(uid)
    assert audit["ok"] is False and audit["reason"] == "no_criteria"


def test_complete_refuses_unmet_criteria(db):
    uid = _make_plan()
    plans.mark_step(uid, 1, "done")
    plans.set_criteria(uid, ["c1"])  # never attested → open
    audit = plans.complete_plan(uid)
    assert audit["ok"] is False and audit["reason"] == "criteria_unmet"
    assert audit["criteria_unmet"] == [1]


def test_complete_success(db):
    uid = _make_plan()
    plans.mark_step(uid, 1, "done")
    plans.set_criteria(uid, ["c1"])
    _grounded_met(uid, 1)
    audit = plans.complete_plan(uid)
    assert audit["ok"] is True
    p = plans.get_plan(uid)
    assert p["status"] == "done" and p["completed_at"]


def test_complete_idempotent_when_already_done(db):
    uid = _make_plan()
    plans.mark_step(uid, 1, "done")
    plans.set_criteria(uid, ["c1"])
    _grounded_met(uid, 1)
    assert plans.complete_plan(uid)["ok"] is True
    again = plans.complete_plan(uid)
    assert again["ok"] is True and again.get("already_done") is True


def test_complete_refuses_abandoned(db):
    uid = _make_plan()
    plans.set_plan_status(uid, "abandoned")
    plans.mark_step(uid, 1, "done")
    plans.set_criteria(uid, ["c1"])
    _grounded_met(uid, 1)
    audit = plans.complete_plan(uid)
    assert audit["ok"] is False and audit["reason"] == "abandoned"
