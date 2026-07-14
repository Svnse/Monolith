from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class NoteRecord:
    note_id: str
    title: str
    safe_title: str
    path: Path
    relative_path: str
    size: int
    mtime_ns: int
    sha256: str


@dataclass(frozen=True)
class NoteTextWindow:
    text: str
    start: int
    end: int
    truncated: bool


@dataclass(frozen=True)
class NoteLink:
    source_note_id: str
    source_title: str
    target_ref: str
    target_title: str
    target_note_id: str | None
    kind: str
    label: str


@dataclass(frozen=True)
class NoteBacklink:
    source_note_id: str
    source_title: str
    source_path: str
    kind: str
    label: str


@dataclass(frozen=True)
class NoteGraphNode:
    note_id: str
    title: str
    safe_title: str
    path: str
    size: int
    degree: int
    tags: tuple[str, ...] = ()
    resolved: bool = True
    focused: bool = False


@dataclass(frozen=True)
class NoteGraphEdge:
    source: str
    target: str
    kind: str
    label: str = ""


@dataclass(frozen=True)
class NoteGraph:
    nodes: tuple[NoteGraphNode, ...]
    edges: tuple[NoteGraphEdge, ...]
    focus_note_id: str | None = None
