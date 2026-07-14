from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import ARTIFACTS_DIR, SKILLS_DIR
from core.skill_registry import list_tools


SKILL_HYGIENE_PATH = ARTIFACTS_DIR / "skill_hygiene.json"


@dataclass(frozen=True)
class SkillHygieneRecord:
    name: str
    path: str
    description: str
    last_used_at: str | None = None
    test_status: str = "unknown"
    retrieval_tags: tuple[str, ...] = ()
    catalog_enabled: bool = True
    prompt_weight: float = 1.0
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["retrieval_tags"] = list(self.retrieval_tags)
        data["notes"] = list(self.notes)
        return data


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_meta(path: Path | None = None) -> dict[str, dict[str, Any]]:
    meta_path = path or SKILL_HYGIENE_PATH
    if not meta_path.exists():
        return {}
    try:
        data = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _save_meta(data: dict[str, dict[str, Any]], path: Path | None = None) -> Path:
    meta_path = path or SKILL_HYGIENE_PATH
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta_path


def list_skill_hygiene(*, metadata_path: Path | None = None) -> list[SkillHygieneRecord]:
    meta = _load_meta(metadata_path)
    records: list[SkillHygieneRecord] = []
    for spec in list_tools():
        saved = meta.get(spec.name, {}) if isinstance(meta.get(spec.name), dict) else {}
        records.append(
            SkillHygieneRecord(
                name=spec.name,
                path=str(spec.path),
                description=spec.description,
                last_used_at=saved.get("last_used_at"),
                test_status=str(saved.get("test_status") or "unknown"),
                retrieval_tags=tuple(str(v) for v in saved.get("retrieval_tags", []) if str(v).strip()),
                catalog_enabled=bool(saved.get("catalog_enabled", True)),
                prompt_weight=float(saved.get("prompt_weight", 1.0) or 1.0),
                notes=tuple(str(v) for v in saved.get("notes", []) if str(v).strip()),
            )
        )
    return records


def mark_skill_used(name: str, *, metadata_path: Path | None = None) -> None:
    key = str(name or "").strip()
    if not key:
        return
    meta = _load_meta(metadata_path)
    row = dict(meta.get(key, {})) if isinstance(meta.get(key), dict) else {}
    row["last_used_at"] = _now_iso()
    meta[key] = row
    _save_meta(meta, metadata_path)


def update_skill_hygiene(
    name: str,
    *,
    retrieval_tags: list[str] | None = None,
    catalog_enabled: bool | None = None,
    prompt_weight: float | None = None,
    notes: list[str] | None = None,
    test_status: str | None = None,
    metadata_path: Path | None = None,
) -> None:
    key = str(name or "").strip()
    if not key:
        raise ValueError("skill name is required")
    meta = _load_meta(metadata_path)
    row = dict(meta.get(key, {})) if isinstance(meta.get(key), dict) else {}
    if retrieval_tags is not None:
        row["retrieval_tags"] = [str(v).strip() for v in retrieval_tags if str(v).strip()]
    if catalog_enabled is not None:
        row["catalog_enabled"] = bool(catalog_enabled)
    if prompt_weight is not None:
        row["prompt_weight"] = max(0.0, min(10.0, float(prompt_weight)))
    if notes is not None:
        row["notes"] = [str(v).strip() for v in notes if str(v).strip()]
    if test_status is not None:
        row["test_status"] = str(test_status)
    meta[key] = row
    _save_meta(meta, metadata_path)


def audit_skills(*, metadata_path: Path | None = None) -> dict[str, Any]:
    records = list_skill_hygiene(metadata_path=metadata_path)
    seen_paths: set[str] = set()
    findings: list[dict[str, Any]] = []
    for record in records:
        path = Path(record.path)
        if not path.exists():
            findings.append({"skill": record.name, "status": "fail", "message": "SKILL.md is missing"})
        elif path.suffix.lower() != ".md":
            findings.append({"skill": record.name, "status": "warn", "message": "skill entry is not markdown"})
        if record.path in seen_paths:
            findings.append({"skill": record.name, "status": "warn", "message": "duplicate skill path"})
        seen_paths.add(record.path)
        if record.catalog_enabled and record.prompt_weight <= 0:
            findings.append({"skill": record.name, "status": "warn", "message": "catalog-enabled skill has zero prompt weight"})
        if not record.retrieval_tags:
            findings.append({"skill": record.name, "status": "warn", "message": "missing retrieval tags"})
    existing_dirs = {p.name for p in SKILLS_DIR.glob("*") if p.is_dir()}
    registered = {record.name for record in records}
    for dirname in sorted(existing_dirs - registered):
        findings.append({"skill": dirname, "status": "warn", "message": "directory is not registered"})
    status = "ok" if not any(f["status"] == "fail" for f in findings) else "fail"
    return {
        "status": status,
        "count": len(records),
        "findings": findings,
        "records": [record.to_dict() for record in records],
    }
