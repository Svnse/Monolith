from __future__ import annotations

import sqlite3
import time
from pathlib import Path
import re

from core import paths
from core.paths import ensure_safe_local_path

from .model import NoteBacklink, NoteGraph, NoteGraphEdge, NoteGraphNode, NoteRecord
from . import store


_WIKILINK_RE = re.compile(r"(?<!!)\[\[(?P<target>[^\]\n|#]+)(?:#[^\]\n|]+)?(?:\|(?P<label>[^\]\n]+))?\]\]")
_MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[(?P<label>[^\]\n]+)\]\((?P<target>[^)\s]+)(?:\s+\"[^\"]*\")?\)")
_INLINE_TAG_RE = re.compile(r"(?<![\w/])#(?P<tag>[A-Za-z][A-Za-z0-9_-]{1,63})\b")
_FRONTMATTER_TAGS_RE = re.compile(r"^tags\s*:\s*(?P<value>.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def db_path(path: str | Path | None = None) -> Path:
    return ensure_safe_local_path(Path(path) if path is not None else paths.LOG_DIR / "mononote.sqlite3")


def connect(path: str | Path | None = None) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path(path))
    conn.row_factory = sqlite3.Row
    ensure_schema(conn)
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS note_index (
            note_id       TEXT PRIMARY KEY,
            title         TEXT NOT NULL,
            safe_title    TEXT NOT NULL,
            path          TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            sha256        TEXT NOT NULL,
            size          INTEGER NOT NULL,
            mtime_ns      INTEGER NOT NULL,
            indexed_at    REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS canvas_nodes (
            note_id    TEXT PRIMARY KEY,
            x          REAL NOT NULL,
            y          REAL NOT NULL,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS note_links (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            source_note_id TEXT NOT NULL,
            target_note_id TEXT,
            target_ref     TEXT NOT NULL,
            target_title   TEXT NOT NULL,
            kind           TEXT NOT NULL,
            label          TEXT NOT NULL,
            indexed_at     REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS note_tags (
            note_id    TEXT NOT NULL,
            tag        TEXT NOT NULL,
            source     TEXT NOT NULL,
            indexed_at REAL NOT NULL,
            PRIMARY KEY(note_id, tag, source)
        );
        CREATE INDEX IF NOT EXISTS idx_note_index_safe_title ON note_index(safe_title);
        CREATE INDEX IF NOT EXISTS idx_note_links_source ON note_links(source_note_id);
        CREATE INDEX IF NOT EXISTS idx_note_links_target ON note_links(target_note_id);
        CREATE INDEX IF NOT EXISTS idx_note_tags_note ON note_tags(note_id);
        """
    )
    conn.commit()


def upsert_note(conn: sqlite3.Connection, note: NoteRecord) -> None:
    conn.execute(
        """
        INSERT INTO note_index(
            note_id, title, safe_title, path, relative_path, sha256, size, mtime_ns, indexed_at
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(note_id) DO UPDATE SET
            title=excluded.title,
            safe_title=excluded.safe_title,
            path=excluded.path,
            relative_path=excluded.relative_path,
            sha256=excluded.sha256,
            size=excluded.size,
            mtime_ns=excluded.mtime_ns,
            indexed_at=excluded.indexed_at
        """,
        (
            note.note_id,
            note.title,
            note.safe_title,
            str(note.path),
            note.relative_path,
            note.sha256,
            int(note.size),
            int(note.mtime_ns),
            time.time(),
        ),
    )


def _target_key(value: str) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    if not raw:
        return ""
    raw = raw.split("#", 1)[0].split("?", 1)[0].strip()
    if "/" in raw:
        raw = Path(raw).stem or raw.rsplit("/", 1)[-1]
    elif raw.lower().endswith(".md"):
        raw = Path(raw).stem
    return store.safe_title(raw).lower()


def _display_target(value: str) -> str:
    raw = str(value or "").strip().replace("\\", "/")
    raw = raw.split("#", 1)[0].split("?", 1)[0].strip()
    if "/" in raw:
        raw = Path(raw).stem or raw.rsplit("/", 1)[-1]
    elif raw.lower().endswith(".md"):
        raw = Path(raw).stem
    return store.safe_title(raw)


def _is_external_link(target: str) -> bool:
    raw = str(target or "").strip().lower()
    return raw.startswith(("http://", "https://", "mailto:", "app://", "file://"))


def _frontmatter_block(body: str) -> str:
    text = str(body or "")
    if not text.startswith("---"):
        return ""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return ""
    block: list[str] = []
    for line in lines[1:]:
        if line.strip() == "---":
            return "\n".join(block)
        block.append(line)
    return ""


def _parse_frontmatter_tags(body: str) -> set[str]:
    block = _frontmatter_block(body)
    if not block:
        return set()
    tags: set[str] = set()
    match = _FRONTMATTER_TAGS_RE.search(block)
    if match is not None:
        value = match.group("value").strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        for part in re.split(r"[, ]+", value):
            token = part.strip().strip("'\"")
            if token:
                tags.add(store.safe_title(token).lower())
    lines = block.splitlines()
    for index, line in enumerate(lines):
        if not re.match(r"^tags\s*:\s*$", line, re.IGNORECASE):
            continue
        for child in lines[index + 1:]:
            if not child.startswith((" ", "\t", "-")):
                break
            token = child.strip().lstrip("-").strip().strip("'\"")
            if token:
                tags.add(store.safe_title(token).lower())
    return {tag for tag in tags if tag}


def _parse_inline_tags(body: str) -> set[str]:
    return {store.safe_title(match.group("tag")).lower() for match in _INLINE_TAG_RE.finditer(str(body or ""))}


def _parse_note_links(body: str, note_lookup: dict[str, str]) -> list[tuple[str | None, str, str, str, str]]:
    text = str(body or "")
    links: list[tuple[str | None, str, str, str, str]] = []
    seen: set[tuple[str, str, str]] = set()

    def add(kind: str, target: str, label: str = "") -> None:
        if not str(target or "").strip() or _is_external_link(target):
            return
        key = _target_key(target)
        if not key:
            return
        target_title = _display_target(target)
        link_key = (kind, key, str(label or "").strip())
        if link_key in seen:
            return
        seen.add(link_key)
        links.append((note_lookup.get(key), target, target_title, kind, str(label or "").strip()))

    for match in _WIKILINK_RE.finditer(text):
        target = match.group("target")
        add("wikilink", target, match.group("label") or _display_target(target))
    for match in _MARKDOWN_LINK_RE.finditer(text):
        target = match.group("target")
        if target.startswith("#"):
            continue
        if target.lower().endswith(".md") or "/" in target:
            add("markdown", target, match.group("label") or _display_target(target))
    return links


def _note_lookup(notes: list[NoteRecord]) -> dict[str, str]:
    lookup: dict[str, str] = {}
    for note in notes:
        keys = {
            note.note_id.lower(),
            note.safe_title.lower(),
            store.safe_title(note.title).lower(),
            _target_key(note.relative_path),
            _target_key(note.path.name),
        }
        for key in keys:
            if key:
                lookup[key] = note.note_id
    return lookup


def _refresh_note_edges(conn: sqlite3.Connection, note: NoteRecord, body: str, note_lookup: dict[str, str]) -> None:
    now = time.time()
    conn.execute("DELETE FROM note_links WHERE source_note_id = ?", (note.note_id,))
    conn.execute("DELETE FROM note_tags WHERE note_id = ?", (note.note_id,))
    for target_note_id, target_ref, target_title, kind, label in _parse_note_links(body, note_lookup):
        conn.execute(
            """
            INSERT INTO note_links(
                source_note_id, target_note_id, target_ref, target_title, kind, label, indexed_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (note.note_id, target_note_id, target_ref, target_title, kind, label, now),
        )
    for tag in sorted(_parse_frontmatter_tags(body) | _parse_inline_tags(body)):
        conn.execute(
            """
            INSERT OR REPLACE INTO note_tags(note_id, tag, source, indexed_at)
            VALUES (?, ?, ?, ?)
            """,
            (note.note_id, tag, "markdown", now),
        )


def refresh_notes(*, vault: str | Path | None = None, path: str | Path | None = None) -> list[NoteRecord]:
    notes = store.list_notes(max_results=200, vault=vault)
    conn = connect(path)
    try:
        seen = {note.note_id for note in notes}
        for note in notes:
            upsert_note(conn, note)
        if seen:
            placeholders = ",".join("?" for _ in seen)
            conn.execute(f"DELETE FROM note_index WHERE note_id NOT IN ({placeholders})", tuple(seen))
            conn.execute(f"DELETE FROM note_links WHERE source_note_id NOT IN ({placeholders})", tuple(seen))
            conn.execute(f"DELETE FROM note_tags WHERE note_id NOT IN ({placeholders})", tuple(seen))
        else:
            conn.execute("DELETE FROM note_index")
            conn.execute("DELETE FROM note_links")
            conn.execute("DELETE FROM note_tags")
        lookup = _note_lookup(notes)
        for note in notes:
            try:
                body = note.path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                body = ""
            _refresh_note_edges(conn, note, body, lookup)
        conn.commit()
    finally:
        conn.close()
    return notes


def list_indexed_notes(*, path: str | Path | None = None, limit: int = 200) -> list[dict]:
    conn = connect(path)
    try:
        rows = conn.execute(
            "SELECT * FROM note_index ORDER BY safe_title ASC LIMIT ?",
            (max(1, min(200, int(limit or 200))),),
        ).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def _row_to_note_key(row: sqlite3.Row | dict) -> str:
    return str(row["note_id"])


def _resolve_note_row(conn: sqlite3.Connection, ref: str) -> sqlite3.Row | None:
    raw = str(ref or "").strip()
    if not raw:
        return None
    row = conn.execute("SELECT * FROM note_index WHERE note_id = ?", (raw,)).fetchone()
    if row is not None:
        return row
    safe = store.safe_title(raw)
    row = conn.execute(
        "SELECT * FROM note_index WHERE lower(safe_title) = lower(?) OR lower(title) = lower(?)",
        (safe, raw),
    ).fetchone()
    if row is not None:
        return row
    key = _target_key(raw)
    if not key:
        return None
    return conn.execute(
        "SELECT * FROM note_index WHERE lower(safe_title) = ? OR lower(relative_path) = ?",
        (key, raw.lower().replace("\\", "/")),
    ).fetchone()


def resolve_note_ref(ref: str, *, path: str | Path | None = None) -> dict | None:
    conn = connect(path)
    try:
        row = _resolve_note_row(conn, ref)
        return dict(row) if row is not None else None
    finally:
        conn.close()


def read_backlinks(note_id: str, *, path: str | Path | None = None) -> list[NoteBacklink]:
    conn = connect(path)
    try:
        target = _resolve_note_row(conn, note_id)
        if target is None:
            return []
        rows = conn.execute(
            """
            SELECT l.kind, l.label, n.note_id, n.safe_title, n.path
            FROM note_links l
            JOIN note_index n ON n.note_id = l.source_note_id
            WHERE l.target_note_id = ?
            ORDER BY n.safe_title ASC, l.kind ASC
            """,
            (_row_to_note_key(target),),
        ).fetchall()
        return tuple(
            NoteBacklink(
                source_note_id=str(row["note_id"]),
                source_title=str(row["safe_title"]),
                source_path=str(row["path"]),
                kind=str(row["kind"]),
                label=str(row["label"] or ""),
            )
            for row in rows
        )
    finally:
        conn.close()


def _tags_by_note(conn: sqlite3.Connection) -> dict[str, tuple[str, ...]]:
    rows = conn.execute("SELECT note_id, tag FROM note_tags ORDER BY tag ASC").fetchall()
    tags: dict[str, list[str]] = {}
    for row in rows:
        tags.setdefault(str(row["note_id"]), []).append(str(row["tag"]))
    return {note_id: tuple(values) for note_id, values in tags.items()}


def read_note_graph(
    focus_note_id: str | None = None,
    *,
    depth: int = 2,
    path: str | Path | None = None,
) -> NoteGraph:
    conn = connect(path)
    try:
        note_rows = conn.execute("SELECT * FROM note_index ORDER BY safe_title ASC").fetchall()
        note_ids = {str(row["note_id"]) for row in note_rows}
        focus_row = _resolve_note_row(conn, focus_note_id or "") if focus_note_id else None
        focus_id = str(focus_row["note_id"]) if focus_row is not None else None
        link_rows = conn.execute(
            """
            SELECT source_note_id, target_note_id, target_ref, target_title, kind, label
            FROM note_links
            ORDER BY source_note_id ASC, target_title ASC
            """
        ).fetchall()

        included = set(note_ids)
        if focus_id:
            max_depth = max(0, min(6, int(depth or 0)))
            adjacency: dict[str, set[str]] = {note_id: set() for note_id in note_ids}
            for row in link_rows:
                source = str(row["source_note_id"])
                target = row["target_note_id"]
                if target:
                    target = str(target)
                    adjacency.setdefault(source, set()).add(target)
                    adjacency.setdefault(target, set()).add(source)
            included = {focus_id}
            frontier = {focus_id}
            for _ in range(max_depth):
                next_frontier: set[str] = set()
                for note_id in frontier:
                    next_frontier.update(adjacency.get(note_id, set()) - included)
                if not next_frontier:
                    break
                included.update(next_frontier)
                frontier = next_frontier

        tags = _tags_by_note(conn)
        degree: dict[str, int] = {note_id: 0 for note_id in included}
        edges: list[NoteGraphEdge] = []
        unresolved_nodes: dict[str, NoteGraphNode] = {}
        for row in link_rows:
            source = str(row["source_note_id"])
            target = str(row["target_note_id"] or "")
            if source not in included:
                continue
            if target and target not in included:
                continue
            if not target:
                target = f"unresolved:{_target_key(str(row['target_ref'])) or str(row['target_title']).lower()}"
                unresolved_nodes.setdefault(
                    target,
                    NoteGraphNode(
                        note_id=target,
                        title=str(row["target_title"]),
                        safe_title=str(row["target_title"]),
                        path=str(row["target_ref"]),
                        size=0,
                        degree=0,
                        resolved=False,
                    ),
                )
            edges.append(
                NoteGraphEdge(
                    source=source,
                    target=target,
                    kind=str(row["kind"]),
                    label=str(row["label"] or ""),
                )
            )
            degree[source] = degree.get(source, 0) + 1
            degree[target] = degree.get(target, 0) + 1

        nodes: list[NoteGraphNode] = []
        for row in note_rows:
            note_id = str(row["note_id"])
            if note_id not in included:
                continue
            nodes.append(
                NoteGraphNode(
                    note_id=note_id,
                    title=str(row["title"]),
                    safe_title=str(row["safe_title"]),
                    path=str(row["path"]),
                    size=int(row["size"]),
                    degree=int(degree.get(note_id, 0)),
                    tags=tags.get(note_id, ()),
                    focused=bool(focus_id and note_id == focus_id),
                )
            )
        for node_id, node in unresolved_nodes.items():
            nodes.append(
                NoteGraphNode(
                    note_id=node.note_id,
                    title=node.title,
                    safe_title=node.safe_title,
                    path=node.path,
                    size=node.size,
                    degree=int(degree.get(node_id, 0)),
                    tags=node.tags,
                    resolved=False,
                    focused=False,
                )
            )
        nodes.sort(key=lambda node: (not node.focused, not node.resolved, node.safe_title.lower()))
        return NoteGraph(nodes=tuple(nodes), edges=tuple(edges), focus_note_id=focus_id)
    finally:
        conn.close()


def read_canvas_position(note_id: str, *, path: str | Path | None = None) -> tuple[float, float] | None:
    conn = connect(path)
    try:
        row = conn.execute(
            "SELECT x, y FROM canvas_nodes WHERE note_id = ?",
            (str(note_id),),
        ).fetchone()
        if row is None:
            return None
        return float(row["x"]), float(row["y"])
    finally:
        conn.close()


def write_canvas_position(
    note_id: str,
    x: float,
    y: float,
    *,
    path: str | Path | None = None,
) -> None:
    conn = connect(path)
    try:
        conn.execute(
            """
            INSERT INTO canvas_nodes(note_id, x, y, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(note_id) DO UPDATE SET
                x=excluded.x,
                y=excluded.y,
                updated_at=excluded.updated_at
            """,
            (str(note_id), float(x), float(y), time.time()),
        )
        conn.commit()
    finally:
        conn.close()
