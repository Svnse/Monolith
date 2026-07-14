from __future__ import annotations

import json
import os
import re
import shutil
import uuid
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from core.paths import ARTIFACTS_DIR


SCHEMA_VERSION = 1
SOUNDTRAP_DIR = ARTIFACTS_DIR / "soundtrap"
LIBRARY_FILENAME = "library.json"
CLIPS_DIRNAME = "clips"
SUPPORTED_AUDIO_EXTS = {".wav", ".mp3", ".flac"}


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _clean_name(value: object, default: str) -> str:
    text = str(value or "").strip()
    return text or default


def _safe_slug(value: object, default: str = "clip") -> str:
    text = str(value or "").strip().lower()
    text = re.sub(r"[^a-z0-9_.-]+", "-", text).strip("-._")
    return text[:64] or default


def _coerce_float(value: object, default: float, minimum: float | None = None) -> float:
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        parsed = default
    if minimum is not None:
        parsed = max(minimum, parsed)
    return parsed


def _default_library() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "active_project_id": None,
        "projects": [],
        "clips": [],
    }


class SoundtrapStore:
    """Small persisted foundation for loop projects and audio clips.

    V1 owns metadata and copied clip files. It deliberately does not implement
    a mix engine; placements are persisted so a future arranger/exporter has a
    stable target to consume.
    """

    def __init__(self, root: Path | None = None) -> None:
        self.root = Path(root or SOUNDTRAP_DIR)
        self.library_path = self.root / LIBRARY_FILENAME
        self.clips_dir = self.root / CLIPS_DIRNAME
        self.root.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)

    def snapshot(self) -> dict[str, Any]:
        return deepcopy(self._load())

    def state_summary(self) -> str:
        data = self._load()
        projects = data.get("projects") or []
        clips = data.get("clips") or []
        active = self._find_project(data, data.get("active_project_id"))
        active_name = active.get("name") if active else "none"
        placements = sum(
            len(track.get("placements") or [])
            for project in projects
            for track in (project.get("tracks") or [])
        )
        return (
            f"[soundtrap: state projects={len(projects)} clips={len(clips)} "
            f"placements={placements} active={active_name}]"
        )

    def list_projects(self) -> list[dict[str, Any]]:
        return list(self._load().get("projects") or [])

    def list_clips(self, *, project_id: str | None = None) -> list[dict[str, Any]]:
        data = self._load()
        clips = list(data.get("clips") or [])
        if not project_id:
            return clips
        used_ids: set[str] = set()
        project = self._require_project(data, project_id)
        for track in project.get("tracks") or []:
            for placement in track.get("placements") or []:
                clip_id = str(placement.get("clip_id") or "")
                if clip_id:
                    used_ids.add(clip_id)
        return [clip for clip in clips if clip.get("id") in used_ids]

    def active_project(self) -> dict[str, Any] | None:
        data = self._load()
        return self._find_project(data, data.get("active_project_id"))

    def create_project(self, name: str, *, bpm: float = 120.0) -> dict[str, Any]:
        data = self._load()
        project = {
            "id": _new_id("project"),
            "name": _clean_name(name, "Untitled Loop"),
            "bpm": _coerce_float(bpm, 120.0, 1.0),
            "created_at": _now(),
            "updated_at": _now(),
            "tracks": [],
        }
        data.setdefault("projects", []).append(project)
        data["active_project_id"] = project["id"]
        self._save(data)
        return deepcopy(project)

    def set_active_project(self, project_id: str) -> dict[str, Any]:
        data = self._load()
        project = self._require_project(data, project_id)
        data["active_project_id"] = project["id"]
        self._save(data)
        return deepcopy(project)

    def set_project_bpm(self, project_id: str | None, bpm: float) -> dict[str, Any]:
        data = self._load()
        project = self._resolve_project(data, project_id)
        project["bpm"] = _coerce_float(bpm, 120.0, 1.0)
        project["updated_at"] = _now()
        data["active_project_id"] = project["id"]
        self._save(data)
        return deepcopy(project)

    def add_track(self, name: str, *, project_id: str | None = None) -> dict[str, Any]:
        data = self._load()
        project = self._resolve_project(data, project_id)
        track = self._ensure_track(project, name)
        project["updated_at"] = _now()
        data["active_project_id"] = project["id"]
        self._save(data)
        out = deepcopy(track)
        out["project_id"] = project["id"]
        return out

    def import_clip(
        self,
        path: str | Path,
        *,
        name: str | None = None,
        project_id: str | None = None,
        source: str = "import",
        prompt: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        source_path = Path(path).expanduser()
        if not source_path.is_absolute():
            source_path = source_path.resolve()
        else:
            source_path = source_path.resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"audio file does not exist: {source_path}")
        if not source_path.is_file():
            raise ValueError(f"audio path is not a file: {source_path}")
        ext = source_path.suffix.lower()
        if ext not in SUPPORTED_AUDIO_EXTS:
            supported = ", ".join(sorted(SUPPORTED_AUDIO_EXTS))
            raise ValueError(f"unsupported audio extension {ext!r}; supported: {supported}")

        data = self._load()
        if project_id:
            self._require_project(data, project_id)

        clip_id = _new_id("clip")
        base_name = _safe_slug(name or source_path.stem, "clip")
        dest = self.clips_dir / f"{clip_id}_{base_name}{ext}"
        if source_path != dest:
            shutil.copy2(source_path, dest)

        clip = {
            "id": clip_id,
            "name": _clean_name(name, source_path.stem),
            "path": str(dest),
            "source_path": str(source_path),
            "source": source,
            "prompt": str(prompt or ""),
            "metadata": dict(metadata or {}),
            "created_at": _now(),
            "project_id": project_id,
        }
        data.setdefault("clips", []).append(clip)
        self._save(data)
        return deepcopy(clip)

    def remove_clip(self, clip_id: str, *, delete_file: bool = False) -> dict[str, Any]:
        data = self._load()
        clip = self._require_clip(data, clip_id)
        data["clips"] = [item for item in data.get("clips") or [] if item.get("id") != clip_id]
        for project in data.get("projects") or []:
            for track in project.get("tracks") or []:
                track["placements"] = [
                    placement for placement in track.get("placements") or []
                    if placement.get("clip_id") != clip_id
                ]
        self._save(data)
        if delete_file:
            try:
                path = Path(str(clip.get("path") or ""))
                if path.exists() and path.is_file():
                    path.unlink()
            except Exception:
                pass
        return deepcopy(clip)

    def place_clip(
        self,
        clip_id: str,
        *,
        track: str = "Track 1",
        start_beat: float = 0.0,
        project_id: str | None = None,
        length_beats: float | None = None,
        gain: float = 1.0,
        muted: bool = False,
    ) -> dict[str, Any]:
        data = self._load()
        self._require_clip(data, clip_id)
        project = self._resolve_project(data, project_id)
        track_obj = self._ensure_track(project, track)
        placement = {
            "id": _new_id("placement"),
            "clip_id": clip_id,
            "start_beat": _coerce_float(start_beat, 0.0, 0.0),
            "length_beats": None if length_beats is None else _coerce_float(length_beats, 0.0, 0.0),
            "gain": _coerce_float(gain, 1.0, 0.0),
            "muted": bool(muted),
            "created_at": _now(),
        }
        track_obj.setdefault("placements", []).append(placement)
        project["updated_at"] = _now()
        data["active_project_id"] = project["id"]
        self._save(data)
        out = deepcopy(placement)
        out["project_id"] = project["id"]
        out["track_id"] = track_obj["id"]
        out["track"] = track_obj["name"]
        return out

    def move_placement(
        self,
        placement_id: str,
        *,
        project_id: str | None = None,
        track: str | None = None,
        start_beat: float | None = None,
        length_beats: float | None = None,
    ) -> dict[str, Any]:
        data = self._load()
        project = self._resolve_project(data, project_id)
        found: dict[str, Any] | None = None
        source_track: dict[str, Any] | None = None
        for track_obj in project.get("tracks") or []:
            for placement in track_obj.get("placements") or []:
                if placement.get("id") == placement_id:
                    found = placement
                    source_track = track_obj
                    break
            if found is not None:
                break
        if found is None or source_track is None:
            raise KeyError(f"placement not found: {placement_id}")

        target_track = source_track if track is None else self._ensure_track(project, track)
        if start_beat is not None:
            found["start_beat"] = _coerce_float(start_beat, 0.0, 0.0)
        if length_beats is not None:
            found["length_beats"] = _coerce_float(length_beats, 0.0, 0.0)
        if target_track is not source_track:
            source_track["placements"] = [
                item for item in source_track.get("placements") or []
                if item.get("id") != placement_id
            ]
            target_track.setdefault("placements", []).append(found)
        project["updated_at"] = _now()
        data["active_project_id"] = project["id"]
        self._save(data)
        out = deepcopy(found)
        out["project_id"] = project["id"]
        out["track_id"] = target_track["id"]
        out["track"] = target_track["name"]
        return out

    def remove_placement(self, placement_id: str, *, project_id: str | None = None) -> dict[str, Any]:
        data = self._load()
        projects = [self._resolve_project(data, project_id)] if project_id else list(data.get("projects") or [])
        for project in projects:
            for track in project.get("tracks") or []:
                placements = track.get("placements") or []
                for placement in placements:
                    if placement.get("id") == placement_id:
                        track["placements"] = [
                            item for item in placements if item.get("id") != placement_id
                        ]
                        project["updated_at"] = _now()
                        data["active_project_id"] = project["id"]
                        self._save(data)
                        out = deepcopy(placement)
                        out["project_id"] = project["id"]
                        out["track_id"] = track["id"]
                        out["track"] = track["name"]
                        return out
        raise KeyError(f"placement not found: {placement_id}")

    def create_pending_clip(
        self,
        *,
        prompt: str,
        name: str | None = None,
        project_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = self._load()
        if project_id:
            self._require_project(data, project_id)
        clip = {
            "id": _new_id("clip"),
            "name": _clean_name(name, "Generated Clip"),
            "path": "",
            "source_path": "",
            "source": "generated",
            "prompt": str(prompt or ""),
            "metadata": dict(metadata or {}),
            "pending": True,
            "created_at": _now(),
            "project_id": project_id,
        }
        data.setdefault("clips", []).append(clip)
        self._save(data)
        return deepcopy(clip)

    def fulfill_pending_clip(
        self,
        pending_clip_id: str,
        audio_path: str | Path,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        data = self._load()
        pending = self._require_clip(data, pending_clip_id)
        imported = self.import_clip(
            audio_path,
            name=pending.get("name") or None,
            project_id=pending.get("project_id") or None,
            source="generated",
            prompt=pending.get("prompt") or "",
            metadata={**dict(pending.get("metadata") or {}), **dict(metadata or {})},
        )
        data = self._load()
        data["clips"] = [item for item in data.get("clips") or [] if item.get("id") != pending_clip_id]
        self._save(data)
        return imported

    def _load(self) -> dict[str, Any]:
        if not self.library_path.exists():
            data = _default_library()
            self._save(data)
            return data
        try:
            raw = json.loads(self.library_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            raw = {}
        data = _default_library()
        if isinstance(raw, dict):
            data.update(raw)
        if not isinstance(data.get("projects"), list):
            data["projects"] = []
        if not isinstance(data.get("clips"), list):
            data["clips"] = []
        data["schema_version"] = SCHEMA_VERSION
        return data

    def _save(self, data: dict[str, Any]) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        data = deepcopy(data)
        data["schema_version"] = SCHEMA_VERSION
        tmp = self.library_path.with_suffix(self.library_path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, self.library_path)

    def _find_project(self, data: dict[str, Any], project_id: object) -> dict[str, Any] | None:
        pid = str(project_id or "")
        for project in data.get("projects") or []:
            if project.get("id") == pid:
                return project
        return None

    def _require_project(self, data: dict[str, Any], project_id: str) -> dict[str, Any]:
        project = self._find_project(data, project_id)
        if project is None:
            raise KeyError(f"project not found: {project_id}")
        return project

    def _resolve_project(self, data: dict[str, Any], project_id: str | None) -> dict[str, Any]:
        if project_id:
            return self._require_project(data, project_id)
        active_id = str(data.get("active_project_id") or "")
        if active_id:
            project = self._find_project(data, active_id)
            if project is not None:
                return project
        project = {
            "id": _new_id("project"),
            "name": "Untitled Loop",
            "bpm": 120.0,
            "created_at": _now(),
            "updated_at": _now(),
            "tracks": [],
        }
        data.setdefault("projects", []).append(project)
        data["active_project_id"] = project["id"]
        return project

    def _require_clip(self, data: dict[str, Any], clip_id: str) -> dict[str, Any]:
        for clip in data.get("clips") or []:
            if clip.get("id") == clip_id:
                return clip
        raise KeyError(f"clip not found: {clip_id}")

    def _ensure_track(self, project: dict[str, Any], name: str) -> dict[str, Any]:
        clean = _clean_name(name, "Track 1")
        tracks = project.setdefault("tracks", [])
        for track in tracks:
            if str(track.get("name") or "").lower() == clean.lower():
                return track
        track = {
            "id": _new_id("track"),
            "name": clean,
            "volume": 1.0,
            "muted": False,
            "placements": [],
            "created_at": _now(),
        }
        tracks.append(track)
        return track


def format_projects(projects: list[dict[str, Any]]) -> str:
    if not projects:
        return "[soundtrap: no projects]"
    lines = [f"[soundtrap: {len(projects)} project(s)]"]
    for project in projects:
        tracks = project.get("tracks") or []
        placements = sum(len(track.get("placements") or []) for track in tracks)
        lines.append(
            f"- {project.get('id')} | {project.get('name')} | "
            f"bpm={project.get('bpm')} | tracks={len(tracks)} | placements={placements}"
        )
    return "\n".join(lines)


def format_clips(clips: list[dict[str, Any]]) -> str:
    if not clips:
        return "[soundtrap: no clips]"
    lines = [f"[soundtrap: {len(clips)} clip(s)]"]
    for clip in clips:
        status = "pending" if clip.get("pending") else Path(str(clip.get("path") or "")).name
        lines.append(
            f"- {clip.get('id')} | {clip.get('name')} | {clip.get('source')} | {status}"
        )
    return "\n".join(lines)


def execute_soundtrap_command(
    cmd: dict[str, Any],
    *,
    store: SoundtrapStore | None = None,
    generate_audio: Callable[[dict[str, Any]], object] | None = None,
) -> str:
    store = store or SoundtrapStore()
    op = str(cmd.get("op") or cmd.get("verb") or "state").strip().lower().replace("-", "_")
    try:
        if op == "state":
            return store.state_summary()
        if op == "list_projects":
            return format_projects(store.list_projects())
        if op == "list_clips":
            project_id = str(cmd.get("project_id") or "").strip() or None
            return format_clips(store.list_clips(project_id=project_id))
        if op == "create_project":
            project = store.create_project(
                str(cmd.get("name") or "Untitled Loop"),
                bpm=_coerce_float(cmd.get("bpm"), 120.0, 1.0),
            )
            return f"[soundtrap: project created {project['id']} | {project['name']}]"
        if op in {"set_bpm", "set_project_bpm"}:
            project = store.set_project_bpm(
                str(cmd.get("project_id") or "").strip() or None,
                _coerce_float(cmd.get("bpm"), 120.0, 1.0),
            )
            return f"[soundtrap: bpm set {project['id']} | {project['bpm']}]"
        if op == "set_active_project":
            project_id = str(cmd.get("project_id") or cmd.get("id") or "").strip()
            if not project_id:
                return "[soundtrap: project_id required]"
            project = store.set_active_project(project_id)
            return f"[soundtrap: active project {project['id']} | {project['name']}]"
        if op == "add_track":
            track = store.add_track(
                str(cmd.get("name") or cmd.get("track") or "Track 1"),
                project_id=str(cmd.get("project_id") or "").strip() or None,
            )
            return f"[soundtrap: track added {track['id']} | {track['name']}]"
        if op in {"add_clip", "import_clip"}:
            path = str(cmd.get("path") or "").strip()
            if not path:
                return "[soundtrap: path required]"
            clip = store.import_clip(
                path,
                name=str(cmd.get("name") or "").strip() or None,
                project_id=str(cmd.get("project_id") or "").strip() or None,
                source="import",
            )
            return f"[soundtrap: clip added {clip['id']} | {clip['name']}]"
        if op == "remove_clip":
            clip_id = str(cmd.get("clip_id") or cmd.get("id") or "").strip()
            if not clip_id:
                return "[soundtrap: clip_id required]"
            clip = store.remove_clip(clip_id, delete_file=bool(cmd.get("delete_file", False)))
            return f"[soundtrap: clip removed {clip['id']} | {clip['name']}]"
        if op == "place_clip":
            clip_id = str(cmd.get("clip_id") or "").strip()
            if not clip_id:
                return "[soundtrap: clip_id required]"
            placement = store.place_clip(
                clip_id,
                project_id=str(cmd.get("project_id") or "").strip() or None,
                track=str(cmd.get("track") or "Track 1"),
                start_beat=_coerce_float(cmd.get("start_beat"), 0.0, 0.0),
                length_beats=(
                    None
                    if cmd.get("length_beats") is None
                    else _coerce_float(cmd.get("length_beats"), 0.0, 0.0)
                ),
                gain=_coerce_float(cmd.get("gain"), 1.0, 0.0),
                muted=bool(cmd.get("muted", False)),
            )
            return (
                f"[soundtrap: placed {clip_id} on {placement['track']} "
                f"at beat {placement['start_beat']}]"
            )
        if op == "move_placement":
            placement_id = str(cmd.get("placement_id") or cmd.get("id") or "").strip()
            if not placement_id:
                return "[soundtrap: placement_id required]"
            placement = store.move_placement(
                placement_id,
                project_id=str(cmd.get("project_id") or "").strip() or None,
                track=(str(cmd.get("track") or "").strip() or None),
                start_beat=(
                    None
                    if cmd.get("start_beat") is None
                    else _coerce_float(cmd.get("start_beat"), 0.0, 0.0)
                ),
                length_beats=(
                    None
                    if cmd.get("length_beats") is None
                    else _coerce_float(cmd.get("length_beats"), 0.0, 0.0)
                ),
            )
            return (
                f"[soundtrap: moved {placement['id']} to {placement['track']} "
                f"at beat {placement.get('start_beat')}]"
            )
        if op == "remove_placement":
            placement_id = str(cmd.get("placement_id") or cmd.get("id") or "").strip()
            if not placement_id:
                return "[soundtrap: placement_id required]"
            placement = store.remove_placement(
                placement_id,
                project_id=str(cmd.get("project_id") or "").strip() or None,
            )
            return f"[soundtrap: placement removed {placement['id']} | {placement['track']}]"
        if op == "generate_clip":
            prompt = str(cmd.get("prompt") or "").strip()
            if not prompt:
                return "[soundtrap: prompt required]"
            if generate_audio is None:
                return "[soundtrap: audio engine unavailable for generate_clip]"
            pending = store.create_pending_clip(
                prompt=prompt,
                name=str(cmd.get("name") or "").strip() or None,
                project_id=str(cmd.get("project_id") or "").strip() or None,
                metadata={
                    "duration": cmd.get("duration"),
                    "sample_rate": cmd.get("sample_rate"),
                },
            )
            payload = {
                "prompt": prompt,
                "duration": cmd.get("duration", 5.0),
                "sample_rate": cmd.get("sample_rate", 32000),
                "soundtrap_pending_clip_id": pending["id"],
                "soundtrap_clip_name": pending["name"],
            }
            result = generate_audio(payload)
            if isinstance(result, str) and result and "generation started" not in result.lower():
                store.remove_clip(pending["id"])
                return f"[soundtrap: generate_clip failed - {result}]"
            suffix = f" | {result}" if isinstance(result, str) and result else ""
            return f"[soundtrap: generation started {pending['id']} | {pending['name']}{suffix}]"
    except Exception as exc:
        return f"[soundtrap: error - {exc}]"
    return (
        "[soundtrap: unknown op. Use state, create_project, set_active_project, set_bpm, "
        "list_projects, list_clips, add_track, add_clip, place_clip, move_placement, "
        "remove_placement, remove_clip, generate_clip]"
    )
