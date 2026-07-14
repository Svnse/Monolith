from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from core.paths import MONOLITH_ROOT  # Monolith's own paths.py

GENESIS_ID = "genesis"  # reserved sentinel; never a .monoline file
WORKFLOWS_DIR = MONOLITH_ROOT / "monoline" / "worlds"  # shared %APPDATA%/Monolith tree


@dataclass(frozen=True)
class Workflow:
    id: str
    name: str
    description: str
    kind: str                        # "native" (Genesis only) | "monoline"
    source_path: Path | None = None  # None for Genesis; the .monoline file otherwise


GENESIS = Workflow(
    id=GENESIS_ID, name="Genesis",
    description="The full native Monolith brain (turn pipeline + acatalepsy + bearing + "
                "monothink + identity + continuity + effort + planes + kernel). Default chat flow.",
    kind="native", source_path=None,
)


class WorkflowRegistry:
    """Lists Genesis (sentinel) + every saved Monoline blueprint on disk.
    Reads .monoline JSON DIRECTLY (no Monoline import -> avoids the core/ collision
    AND keeps the listing path independent of whether the plugin is installed)."""

    def __init__(self, workflows_dir: Path = WORKFLOWS_DIR) -> None:
        self._dir = workflows_dir

    @property
    def workflows_dir(self) -> Path:
        """The directory globbed for .monoline blueprints (for a fs watcher to monitor)."""
        return self._dir

    def list_workflows(self) -> list[Workflow]:
        items = [GENESIS]  # always first card, always present
        if self._dir.exists():
            seen: set[str] = set()
            for path in sorted(self._dir.glob("*.monoline"),
                               key=lambda p: p.stat().st_mtime, reverse=True):
                try:
                    data = json.loads(path.read_text(encoding="utf-8"))
                except Exception:
                    continue  # tolerate partial/corrupt files
                wid = str(data.get("id") or "").strip()
                if not wid or wid == GENESIS_ID or wid in seen:
                    continue  # GENESIS_ID is reserved; never shadowed
                seen.add(wid)
                items.append(Workflow(
                    id=wid, name=str(data.get("name") or "Untitled"),
                    description=str(data.get("description") or ""),
                    kind="monoline", source_path=path))
        return items

    def get(self, workflow_id: str) -> Workflow | None:
        if (workflow_id or GENESIS_ID) == GENESIS_ID:
            return GENESIS  # unset/empty/'genesis' -> Genesis
        for wf in self.list_workflows():
            if wf.id == workflow_id:
                return wf
        return None  # unknown -> caller falls back to Genesis

    def bind_world_state(self, world_state) -> None:
        """Optional: bind a WorldStateStore so set_active/active_id persist the flag."""
        self._ws = world_state

    def set_active(self, workflow_id: str) -> None:
        ws = getattr(self, "_ws", None)
        if ws is not None:
            ws.set_active_workflow(workflow_id or None)

    def active_id(self) -> str:
        ws = getattr(self, "_ws", None)
        return ws.get_active_workflow() if ws is not None else ""
