from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from core.paths import ARTIFACTS_DIR


INVESTIGATION_DIR = ARTIFACTS_DIR / "investigations"


@dataclass(frozen=True)
class SourceRef:
    kind: str
    ref: str
    title: str = ""
    note: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class InvestigationRun:
    run_id: str
    goal: str
    created_at: str
    updated_at: str
    source_refs: tuple[SourceRef, ...] = ()
    linked_plan_uids: tuple[str, ...] = ()
    linked_acu_ids: tuple[int, ...] = ()
    synthesis_markdown: str = ""
    status: str = "active"

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "goal": self.goal,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "source_refs": [ref.to_dict() for ref in self.source_refs],
            "linked_plan_uids": list(self.linked_plan_uids),
            "linked_acu_ids": list(self.linked_acu_ids),
            "synthesis_markdown": self.synthesis_markdown,
            "status": self.status,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _slug(text: str) -> str:
    value = re.sub(r"[^a-z0-9]+", "-", str(text or "").lower()).strip("-")
    return value[:48] or "investigation"


def _run_path(run_id: str, *, root: Path | None = None) -> Path:
    return (root or INVESTIGATION_DIR) / f"{run_id}.json"


def _markdown_path(run: InvestigationRun, *, root: Path | None = None) -> Path:
    return (root or INVESTIGATION_DIR) / f"{run.run_id}-{_slug(run.goal)}.md"


def create_investigation(goal: str, *, root: Path | None = None) -> InvestigationRun:
    goal_s = str(goal or "").strip()
    if not goal_s:
        raise ValueError("investigation requires a goal")
    now = _now_iso()
    run = InvestigationRun(run_id=uuid.uuid4().hex, goal=goal_s, created_at=now, updated_at=now)
    save_investigation(run, root=root)
    return run


def save_investigation(run: InvestigationRun, *, root: Path | None = None) -> tuple[Path, Path]:
    base = root or INVESTIGATION_DIR
    base.mkdir(parents=True, exist_ok=True)
    json_path = _run_path(run.run_id, root=base)
    md_path = _markdown_path(run, root=base)
    json_path.write_text(json.dumps(run.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    md = render_markdown(run)
    md_path.write_text(md, encoding="utf-8")
    return json_path, md_path


def load_investigation(run_id: str, *, root: Path | None = None) -> InvestigationRun | None:
    path = _run_path(run_id, root=root)
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return InvestigationRun(
        run_id=str(data["run_id"]),
        goal=str(data["goal"]),
        created_at=str(data["created_at"]),
        updated_at=str(data["updated_at"]),
        source_refs=tuple(SourceRef(**item) for item in data.get("source_refs", [])),
        linked_plan_uids=tuple(str(v) for v in data.get("linked_plan_uids", [])),
        linked_acu_ids=tuple(int(v) for v in data.get("linked_acu_ids", [])),
        synthesis_markdown=str(data.get("synthesis_markdown", "")),
        status=str(data.get("status", "active")),
    )


def list_investigations(*, limit: int = 50, root: Path | None = None) -> list[InvestigationRun]:
    base = root or INVESTIGATION_DIR
    if not base.exists():
        return []
    runs: list[InvestigationRun] = []
    for path in sorted(base.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if len(runs) >= max(1, int(limit or 50)):
            break
        run = load_investigation(path.stem, root=base)
        if run is not None:
            runs.append(run)
    runs.sort(key=lambda run: run.updated_at, reverse=True)
    return runs[: max(1, int(limit or 50))]


def add_source_ref(
    run_id: str,
    *,
    kind: str,
    ref: str,
    title: str = "",
    note: str = "",
    root: Path | None = None,
) -> InvestigationRun:
    run = load_investigation(run_id, root=root)
    if run is None:
        raise ValueError(f"investigation not found: {run_id}")
    src = SourceRef(kind=str(kind or "source"), ref=str(ref or ""), title=str(title or ""), note=str(note or ""))
    updated = InvestigationRun(
        run_id=run.run_id,
        goal=run.goal,
        created_at=run.created_at,
        updated_at=_now_iso(),
        source_refs=tuple(list(run.source_refs) + [src]),
        linked_plan_uids=run.linked_plan_uids,
        linked_acu_ids=run.linked_acu_ids,
        synthesis_markdown=run.synthesis_markdown,
        status=run.status,
    )
    save_investigation(updated, root=root)
    return updated


def update_synthesis(
    run_id: str,
    synthesis_markdown: str,
    *,
    linked_plan_uids: list[str] | None = None,
    linked_acu_ids: list[int] | None = None,
    status: str | None = None,
    root: Path | None = None,
) -> InvestigationRun:
    run = load_investigation(run_id, root=root)
    if run is None:
        raise ValueError(f"investigation not found: {run_id}")
    updated = InvestigationRun(
        run_id=run.run_id,
        goal=run.goal,
        created_at=run.created_at,
        updated_at=_now_iso(),
        source_refs=run.source_refs,
        linked_plan_uids=tuple(linked_plan_uids) if linked_plan_uids is not None else run.linked_plan_uids,
        linked_acu_ids=tuple(int(v) for v in linked_acu_ids) if linked_acu_ids is not None else run.linked_acu_ids,
        synthesis_markdown=str(synthesis_markdown or ""),
        status=str(status or run.status),
    )
    save_investigation(updated, root=root)
    return updated


def render_markdown(run: InvestigationRun) -> str:
    lines = [f"# Investigation: {run.goal}", "", f"- run_id: `{run.run_id}`", f"- status: `{run.status}`", ""]
    if run.source_refs:
        lines.append("## Sources")
        for ref in run.source_refs:
            label = ref.title or ref.ref
            extra = f" - {ref.note}" if ref.note else ""
            lines.append(f"- `{ref.kind}` {label}{extra}")
        lines.append("")
    if run.linked_plan_uids:
        lines.append("## Plans")
        lines.extend(f"- `{uid}`" for uid in run.linked_plan_uids)
        lines.append("")
    if run.linked_acu_ids:
        lines.append("## ACUs")
        lines.extend(f"- `{acu_id}`" for acu_id in run.linked_acu_ids)
        lines.append("")
    lines.append("## Synthesis")
    lines.append(run.synthesis_markdown.strip() or "_No synthesis yet._")
    lines.append("")
    return "\n".join(lines)
