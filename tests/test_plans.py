from __future__ import annotations

import pytest

from core import plans


@pytest.fixture
def store(monkeypatch, tmp_path):
    plans.set_db_path(tmp_path / "turn_trace.sqlite3")
    yield
    plans.set_db_path(None)


_STEPS = [
    {"verb": "read", "target": "spec.md", "depends_on": []},
    {"verb": "draft", "target": "module", "depends_on": [1]},
    {"verb": "test", "target": "module", "depends_on": [2]},
]


def test_create_and_get_plan(store) -> None:
    uid = plans.create_plan(goal="ship the thing", source="explicit", steps=_STEPS)
    p = plans.get_plan(uid)
    assert p is not None
    assert p["goal"] == "ship the thing"
    assert p["source"] == "explicit"
    assert p["status"] == "proposed"
    assert [s["seq"] for s in p["steps"]] == [1, 2, 3]
    assert p["steps"][1]["verb"] == "draft"
    assert p["steps"][1]["depends_on"] == [1]
    assert all(s["status"] == "pending" for s in p["steps"])


def test_get_active_plan_returns_latest_open(store) -> None:
    plans.create_plan(goal="old", source="explicit", steps=_STEPS)
    uid2 = plans.create_plan(goal="new", source="bearing", steps=_STEPS)
    active = plans.get_active_plan()
    assert active is not None and active["plan_uid"] == uid2


def test_mark_step_updates_status(store) -> None:
    uid = plans.create_plan(goal="g", source="explicit", steps=_STEPS)
    plans.mark_step(uid, 1, "done")
    p = plans.get_plan(uid)
    assert p["steps"][0]["status"] == "done"


def test_next_ready_steps_respects_dependencies(store) -> None:
    uid = plans.create_plan(goal="g", source="explicit", steps=_STEPS)
    ready = plans.next_ready_steps(uid)
    assert [s["seq"] for s in ready] == [1]  # only step 1 (no deps)
    plans.mark_step(uid, 1, "done")
    ready = plans.next_ready_steps(uid)
    assert [s["seq"] for s in ready] == [2]  # step 2 unblocked


def test_set_plan_status(store) -> None:
    uid = plans.create_plan(goal="g", source="curiosity", steps=_STEPS)
    plans.set_plan_status(uid, "done")
    assert plans.get_plan(uid)["status"] == "done"
    # a done plan is no longer the active plan
    assert plans.get_active_plan() is None


def test_list_plans_newest_first(store) -> None:
    plans.create_plan(goal="a", source="explicit", steps=_STEPS)
    plans.create_plan(goal="b", source="explicit", steps=_STEPS)
    items = plans.list_plans(limit=10)
    assert [p["goal"] for p in items[:2]] == ["b", "a"]


def test_plan_survives_connection_reopen(tmp_path) -> None:
    """Durability — the whole reason a SQLite table was chosen over Bearing.
    A plan written under one connection must be readable after the connection
    is closed and a fresh one opened against the same file."""
    path = tmp_path / "turn_trace.sqlite3"
    plans.set_db_path(path)
    try:
        uid = plans.create_plan(goal="durable goal", source="explicit", steps=_STEPS)
        plans.set_db_path(path)  # closes the cached _conn → next call opens a fresh one
        p = plans.get_plan(uid)
        assert p is not None
        assert p["goal"] == "durable goal"
        assert len(p["steps"]) == 3
    finally:
        plans.set_db_path(None)


def test_record_and_get_observations(store) -> None:
    uid = plans.create_plan(goal="g", source="monoexplore", steps=_STEPS)
    n = plans.record_observations(
        uid, "exp_000001",
        visited=["list_files /root", "read_file a.py"],
        findings=["a | rel | b"],
    )
    assert n == 3
    obs = plans.get_observations(uid)
    assert obs["visited"] == ["list_files /root", "read_file a.py"]
    assert obs["findings"] == ["a | rel | b"]


def test_record_observations_dedupes_visited(store) -> None:
    uid = plans.create_plan(goal="g", source="monoexplore", steps=_STEPS)
    plans.record_observations(uid, "exp_000001", visited=["list_files /root"], findings=[])
    # /root already seen → only the new /sub row is written
    n = plans.record_observations(
        uid, "exp_000002", visited=["list_files /root", "list_files /sub"], findings=[])
    assert n == 1
    obs = plans.get_observations(uid)
    assert obs["visited"] == ["list_files /root", "list_files /sub"]


def test_get_observations_respects_caps_keeping_most_recent(store) -> None:
    uid = plans.create_plan(goal="g", source="monoexplore", steps=_STEPS)
    visited = [f"list_files /d{i}" for i in range(30)]
    findings = [f"f{i} | rel | x" for i in range(20)]
    plans.record_observations(uid, "exp_000001", visited=visited, findings=findings)
    obs = plans.get_observations(uid, max_visited=25, max_findings=12)
    assert obs["visited"] == visited[-25:]      # most-recent kept, chronological order
    assert obs["findings"] == findings[-12:]


def test_observations_scoped_by_plan(store) -> None:
    uid1 = plans.create_plan(goal="g1", source="monoexplore", steps=_STEPS)
    uid2 = plans.create_plan(goal="g2", source="monoexplore", steps=_STEPS)
    plans.record_observations(uid1, "exp_a", visited=["list_files /one"], findings=["one | r | x"])
    plans.record_observations(uid2, "exp_b", visited=["list_files /two"], findings=["two | r | x"])
    assert plans.get_observations(uid1) == {"visited": ["list_files /one"], "findings": ["one | r | x"]}
    assert plans.get_observations(uid2) == {"visited": ["list_files /two"], "findings": ["two | r | x"]}


def test_record_observations_unknown_plan_is_noop(store) -> None:
    assert plans.record_observations("nonexistent", "exp_x", visited=["x"], findings=[]) == 0
    assert plans.get_observations("nonexistent") == {"visited": [], "findings": []}


def test_observations_survive_connection_reopen(tmp_path) -> None:
    """Durability — the ledger is the expedition's cross-wake memory; it must
    survive the daemon restarting (a fresh connection to the same file)."""
    path = tmp_path / "turn_trace.sqlite3"
    plans.set_db_path(path)
    try:
        uid = plans.create_plan(goal="durable", source="monoexplore", steps=_STEPS)
        plans.record_observations(uid, "exp_1", visited=["list_files /root"], findings=["a | r | b"])
        plans.set_db_path(path)  # closes cached _conn → next call opens fresh
        obs = plans.get_observations(uid)
        assert obs["visited"] == ["list_files /root"]
        assert obs["findings"] == ["a | r | b"]
    finally:
        plans.set_db_path(None)


def test_plans_coexist_with_turn_trace_in_one_file(tmp_path) -> None:
    """plans + turn_trace each hold their own connection to the SAME
    turn_trace.sqlite3 (E's storage choice). Both must create their tables and
    write/read without clobbering each other."""
    from core import turn_trace
    shared = tmp_path / "turn_trace.sqlite3"
    turn_trace.set_db_path(shared)
    plans.set_db_path(shared)
    try:
        assert turn_trace._get_conn() is not None  # materializes turn_trace's tables
        uid = plans.create_plan(goal="coexist", source="explicit", steps=_STEPS)
        assert plans.get_plan(uid)["goal"] == "coexist"
        conn = plans._get_conn()
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        assert {"plans", "plan_steps"}.issubset(tables)        # M1's tables
        assert "stage_traces" in tables                         # turn_trace's table — same file
    finally:
        plans.set_db_path(None)
        turn_trace.set_db_path(None)
