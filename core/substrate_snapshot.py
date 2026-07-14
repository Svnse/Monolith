from __future__ import annotations

import json
import sqlite3
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from core.paths import ARTIFACTS_DIR, CONFIG_DIR, LOG_DIR, SKILLS_DIR


SNAPSHOT_DIR = ARTIFACTS_DIR / "snapshots"
_SECRET_KEYS = {"api_key", "token", "secret", "password", "authorization"}


@dataclass(frozen=True)
class SnapshotEntry:
    kind: str
    path: str
    included: bool
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SnapshotPlan:
    created_at: str
    entries: tuple[SnapshotEntry, ...]
    redacted: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "created_at": self.created_at,
            "redacted": self.redacted,
            "entries": [entry.to_dict() for entry in self.entries],
        }


def _now_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        out = {}
        for key, item in value.items():
            if any(secret in str(key).lower() for secret in _SECRET_KEYS):
                out[key] = "<redacted>"
            else:
                out[key] = _redact(item)
        return out
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value


def build_snapshot_plan(*, include_secrets: bool = False) -> SnapshotPlan:
    entries: list[SnapshotEntry] = []
    config_path = CONFIG_DIR / "config.yaml"
    entries.append(SnapshotEntry("config", str(config_path), config_path.exists(), "main config"))
    for skill in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        entries.append(SnapshotEntry("skill", str(skill), True, "skill entrypoint"))
    for db_name in ("turn_trace.sqlite3", "acatalepsy.sqlite3"):
        path = LOG_DIR / db_name
        entries.append(SnapshotEntry("sqlite_summary", str(path), path.exists(), "summary only"))
    return SnapshotPlan(
        created_at=datetime.now(timezone.utc).isoformat(),
        entries=tuple(entries),
        redacted=not include_secrets,
    )


def _sqlite_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False}
    summary: dict[str, Any] = {"exists": True, "tables": {}}
    try:
        conn = sqlite3.connect(str(path))
        try:
            tables = [
                row[0]
                for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()
            ]
            for table in tables:
                try:
                    count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                except Exception:
                    count = None
                summary["tables"][table] = {"count": count}
        finally:
            conn.close()
    except Exception as exc:
        summary["error"] = str(exc)
    return summary


def export_snapshot(
    *,
    output_path: Path | None = None,
    include_secrets: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    plan = build_snapshot_plan(include_secrets=include_secrets)
    if dry_run:
        return {"dry_run": True, "plan": plan.to_dict(), "output_path": str(output_path or "")}

    out = output_path or (SNAPSHOT_DIR / f"restore-substrate-{_now_slug()}.zip")
    out.parent.mkdir(parents=True, exist_ok=True)
    manifest = plan.to_dict()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        config_path = CONFIG_DIR / "config.yaml"
        if config_path.exists():
            try:
                config_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
            except Exception:
                config_data = {}
            if not include_secrets:
                config_data = _redact(config_data)
            zf.writestr("config/config.yaml", yaml.safe_dump(config_data, sort_keys=False))
        for skill in sorted(SKILLS_DIR.glob("*/SKILL.md")):
            zf.write(skill, f"skills/{skill.parent.name}/SKILL.md")
        for db_name in ("turn_trace.sqlite3", "acatalepsy.sqlite3"):
            db_path = LOG_DIR / db_name
            zf.writestr(
                f"summaries/{db_name}.summary.json",
                json.dumps(_sqlite_summary(db_path), ensure_ascii=False, indent=2),
            )
    return {"dry_run": False, "output_path": str(out), "plan": manifest}


def restore_snapshot_dry_run(archive_path: Path) -> dict[str, Any]:
    path = Path(archive_path)
    if not path.exists():
        return {"ok": False, "error": "archive not found", "changes": []}
    changes: list[dict[str, Any]] = []
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = set(zf.namelist())
            if "config/config.yaml" in names:
                changes.append({"kind": "config", "target": str(CONFIG_DIR / "config.yaml"), "action": "would_replace"})
            for name in sorted(n for n in names if n.startswith("skills/") and n.endswith("/SKILL.md")):
                parts = name.split("/")
                if len(parts) >= 3:
                    changes.append({"kind": "skill", "target": str(SKILLS_DIR / parts[1] / "SKILL.md"), "action": "would_replace"})
    except Exception as exc:
        return {"ok": False, "error": str(exc), "changes": []}
    return {"ok": True, "archive": str(path), "changes": changes}
