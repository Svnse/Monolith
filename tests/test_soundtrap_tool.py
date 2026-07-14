from __future__ import annotations

from pathlib import Path

from core import skill_runtime
from core.skill_registry import clear_skill_cache, get_tool
from core.soundtrap import SoundtrapStore, execute_soundtrap_command


def test_soundtrap_tool_is_discoverable() -> None:
    clear_skill_cache()
    spec = get_tool("soundtrap")

    assert spec is not None
    assert spec.name == "soundtrap"
    assert spec.json_schema is not None


def test_soundtrap_executor_uses_live_callback(tmp_path: Path) -> None:
    seen: list[dict] = []
    ctx = skill_runtime.ToolExecutionContext(
        archive_dir=tmp_path,
        on_soundtrap=lambda cmd: seen.append(dict(cmd)) or "[soundtrap: state projects=0 clips=0]",
    )

    env = skill_runtime.execute_tool_call_enveloped({"tool": "soundtrap", "op": "state"}, ctx)

    assert env.ok is True
    assert env.tool == "soundtrap"
    assert seen and seen[0]["op"] == "state"
    assert "projects=0" in env.text


def test_soundtrap_is_denied_below_l1(tmp_path: Path) -> None:
    ctx = skill_runtime.ToolExecutionContext(
        archive_dir=tmp_path,
        level=2,
        allowed_tools=skill_runtime.L2_WORKER_TOOLS,
    )

    env = skill_runtime.execute_tool_call_enveloped({"tool": "soundtrap", "op": "state"}, ctx)

    assert env.ok is False
    assert env.data.get("denied") is True
    assert env.data.get("reason") == "capability"


def test_soundtrap_command_adds_and_lists_clip(tmp_path: Path) -> None:
    audio = tmp_path / "hat.wav"
    audio.write_bytes(b"RIFFfake")
    store = SoundtrapStore(tmp_path / "soundtrap")

    created = execute_soundtrap_command({"op": "create_project", "name": "Beat"}, store=store)
    added = execute_soundtrap_command({"op": "add_clip", "path": str(audio), "name": "Hat"}, store=store)
    listed = execute_soundtrap_command({"op": "list_clips"}, store=store)

    assert "project created" in created
    assert "clip added" in added
    assert "Hat" in listed


def test_soundtrap_command_supports_daw_arrangement_ops(tmp_path: Path) -> None:
    audio = tmp_path / "pad.wav"
    audio.write_bytes(b"RIFFfake")
    store = SoundtrapStore(tmp_path / "soundtrap")

    execute_soundtrap_command({"op": "create_project", "name": "Song", "bpm": 118}, store=store)
    track_added = execute_soundtrap_command({"op": "add_track", "name": "pads"}, store=store)
    execute_soundtrap_command({"op": "add_clip", "path": str(audio), "name": "Pad"}, store=store)
    clip_id = store.snapshot()["clips"][0]["id"]
    placed = execute_soundtrap_command(
        {"op": "place_clip", "clip_id": clip_id, "track": "pads", "start_beat": 2},
        store=store,
    )
    placement_id = store.snapshot()["projects"][0]["tracks"][0]["placements"][0]["id"]
    moved = execute_soundtrap_command(
        {"op": "move_placement", "placement_id": placement_id, "track": "pads", "start_beat": 6},
        store=store,
    )

    assert "track added" in track_added
    assert "placed" in placed
    assert "moved" in moved
    assert store.snapshot()["projects"][0]["tracks"][0]["placements"][0]["start_beat"] == 6.0
