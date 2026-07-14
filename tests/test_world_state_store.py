from __future__ import annotations

import json

import pytest

from core.world_state import WorldStateStore


def test_snapshot_is_deep_copy(tmp_path) -> None:
    store = WorldStateStore(path=tmp_path / "world_state.json")
    store.set_engine_meta("demo", config={"temperature": 0.2})

    snap = store.snapshot()
    snap["engines"]["demo"]["config"]["temperature"] = 0.9

    assert store.state["engines"]["demo"]["config"]["temperature"] == 0.2


def test_save_is_atomic_and_preserves_existing_file_on_replace_failure(monkeypatch, tmp_path) -> None:
    path = tmp_path / "world_state.json"
    original_payload = {"updated_at": "2026-01-01T00:00:00+00:00", "engines": {"demo": {"status": "READY"}}}
    path.write_text(json.dumps(original_payload), encoding="utf-8")

    store = WorldStateStore(path=path)
    store.set_engine_status("demo", "RUNNING")

    def _fail_replace(_src, _dst):
        raise OSError("replace failed")

    monkeypatch.setattr("core.world_state.os.replace", _fail_replace)

    with pytest.raises(OSError):
        store.save()

    on_disk = json.loads(path.read_text(encoding="utf-8"))
    assert on_disk == original_payload
    assert not (tmp_path / "world_state.json.tmp").exists()


def test_invalid_engine_key_is_ignored(tmp_path) -> None:
    store = WorldStateStore(path=tmp_path / "world_state.json")
    store.set_engine_status("bad key with spaces", "RUNNING")
    assert store.state["engines"] == {}


def test_world_state_sanitizes_non_json_values(tmp_path) -> None:
    store = WorldStateStore(path=tmp_path / "world_state.json")
    store.set_engine_meta("demo", callback=object(), nested={"x": object()})
    store.set_pending_action({"type": "approve", "payload": {"raw": object()}})
    store.append_action_log({"data": object()})
    store.save()

    loaded = json.loads((tmp_path / "world_state.json").read_text(encoding="utf-8"))
    assert isinstance(loaded["engines"]["demo"]["callback"], str)
    assert isinstance(loaded["engines"]["demo"]["nested"]["x"], str)
    assert isinstance(loaded["session"]["pending_action"]["payload"]["raw"], str)
    assert isinstance(loaded["action_log"][-1]["data"], str)
