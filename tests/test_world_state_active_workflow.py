from __future__ import annotations

from pathlib import Path

from core.world_state import WorldStateStore


def _store(tmp_path: Path) -> WorldStateStore:
    return WorldStateStore(path=Path(tmp_path) / "world_state.json")  # isolate persistence


def test_default_active_workflow_is_empty(tmp_path):
    s = _store(tmp_path)
    assert s.get_active_workflow() == ""  # '' == Genesis


def test_set_and_get_active_workflow(tmp_path):
    s = _store(tmp_path)
    s.set_active_workflow("alpha")
    assert s.get_active_workflow() == "alpha"


def test_set_none_clears_active_workflow(tmp_path):
    s = _store(tmp_path)
    s.set_active_workflow("alpha")
    s.set_active_workflow(None)
    assert s.get_active_workflow() == ""  # absence == Genesis


def test_set_empty_clears_active_workflow(tmp_path):
    s = _store(tmp_path)
    s.set_active_workflow("alpha")
    s.set_active_workflow("")
    assert s.get_active_workflow() == ""


def test_active_workflow_survives_snapshot(tmp_path):
    s = _store(tmp_path)
    s.set_active_workflow("beta")
    assert s.snapshot().get("active_workflow_id") == "beta"
