from __future__ import annotations

from pathlib import Path

import pytest

from core.soundtrap import SoundtrapStore


def _audio_file(tmp_path: Path, name: str = "kick.wav") -> Path:
    path = tmp_path / name
    path.write_bytes(b"RIFFfake-wave")
    return path


def test_soundtrap_store_creates_project_imports_clip_and_places_it(tmp_path: Path) -> None:
    store = SoundtrapStore(tmp_path / "soundtrap")
    project = store.create_project("Night Loop", bpm=128)
    clip = store.import_clip(_audio_file(tmp_path), name="Kick", project_id=project["id"])
    placement = store.place_clip(clip["id"], track="drums", start_beat=4, project_id=project["id"])

    snapshot = store.snapshot()

    assert snapshot["active_project_id"] == project["id"]
    assert len(snapshot["projects"]) == 1
    assert len(snapshot["clips"]) == 1
    assert Path(snapshot["clips"][0]["path"]).exists()
    assert placement["track"] == "drums"
    assert snapshot["projects"][0]["tracks"][0]["placements"][0]["clip_id"] == clip["id"]


def test_soundtrap_store_rejects_unsupported_audio_extension(tmp_path: Path) -> None:
    store = SoundtrapStore(tmp_path / "soundtrap")
    bad = tmp_path / "note.txt"
    bad.write_text("not audio", encoding="utf-8")

    with pytest.raises(ValueError, match="unsupported audio extension"):
        store.import_clip(bad)


def test_soundtrap_remove_clip_removes_placements(tmp_path: Path) -> None:
    store = SoundtrapStore(tmp_path / "soundtrap")
    project = store.create_project("Loop")
    clip = store.import_clip(_audio_file(tmp_path), project_id=project["id"])
    store.place_clip(clip["id"], project_id=project["id"], track="drums")

    removed = store.remove_clip(clip["id"])
    snapshot = store.snapshot()

    assert removed["id"] == clip["id"]
    assert snapshot["clips"] == []
    assert snapshot["projects"][0]["tracks"][0]["placements"] == []


def test_soundtrap_store_adds_track_and_moves_placement(tmp_path: Path) -> None:
    store = SoundtrapStore(tmp_path / "soundtrap")
    project = store.create_project("Arrangement", bpm=100)
    track = store.add_track("drums", project_id=project["id"])
    clip = store.import_clip(_audio_file(tmp_path), project_id=project["id"])
    placement = store.place_clip(clip["id"], project_id=project["id"], track=track["name"])

    moved = store.move_placement(
        placement["id"],
        project_id=project["id"],
        track="bass",
        start_beat=8,
        length_beats=2,
    )
    snapshot = store.snapshot()

    assert moved["track"] == "bass"
    assert moved["start_beat"] == 8.0
    assert moved["length_beats"] == 2.0
    assert snapshot["projects"][0]["tracks"][0]["placements"] == []
    assert snapshot["projects"][0]["tracks"][1]["placements"][0]["id"] == placement["id"]


def test_soundtrap_store_sets_bpm_and_removes_placement(tmp_path: Path) -> None:
    store = SoundtrapStore(tmp_path / "soundtrap")
    project = store.create_project("Tempo")
    clip = store.import_clip(_audio_file(tmp_path), project_id=project["id"])
    placement = store.place_clip(clip["id"], project_id=project["id"], track="keys")

    updated = store.set_project_bpm(project["id"], 132)
    removed = store.remove_placement(placement["id"], project_id=project["id"])
    snapshot = store.snapshot()

    assert updated["bpm"] == 132.0
    assert removed["id"] == placement["id"]
    assert snapshot["projects"][0]["tracks"][0]["placements"] == []
