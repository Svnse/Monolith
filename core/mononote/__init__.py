"""MonoNote local-first note substrate.

Markdown files remain the source of truth. SQLite/search/canvas layers can
cache metadata later, but ACU evidence starts at canonical_log provenance.
"""
from __future__ import annotations

from .model import (
    NoteBacklink,
    NoteGraph,
    NoteGraphEdge,
    NoteGraphNode,
    NoteLink,
    NoteRecord,
    NoteTextWindow,
)
from .store import (
    load_note,
    list_notes,
    note_path_for_title,
    safe_title,
    slice_note_text,
    write_note,
)
from . import index

__all__ = (
    "NoteRecord",
    "NoteTextWindow",
    "NoteLink",
    "NoteBacklink",
    "NoteGraph",
    "NoteGraphEdge",
    "NoteGraphNode",
    "load_note",
    "list_notes",
    "note_path_for_title",
    "safe_title",
    "slice_note_text",
    "write_note",
    "index",
)
