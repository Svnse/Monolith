from __future__ import annotations

import hashlib
import re
from typing import Any

from core.acatalepsy import canonical_log

from .model import NoteRecord, NoteTextWindow


NOTE_READ_KIND = "mononote_note_read"
NOTE_READ_SCHEMA = "MONONOTE_NOTE_READ_V1"
NOTE_SOURCE = "mononote"
NOTE_CANDIDATE_SOURCE = "mononote_note"

_PROMPT_AUTH_WORDS = {
    "note",
    "notes",
    "mononote",
    "monomem",
    "scratchpad",
    "vault",
}
_PROMPT_DENY_PHRASES = (
    "do not read",
    "don't read",
    "dont read",
    "without reading",
    "no note",
    "no notes",
)


def _ctx_level(ctx: Any) -> int:
    try:
        return int(getattr(ctx, "level", 1))
    except (TypeError, ValueError):
        return 1


def _world_state_snapshot(ctx: Any) -> dict[str, Any]:
    world_state = getattr(ctx, "world_state", None)
    if world_state is None:
        return {}
    try:
        if hasattr(world_state, "snapshot"):
            snap = world_state.snapshot()
        else:
            snap = getattr(world_state, "state", {})
    except Exception:
        return {}
    return snap if isinstance(snap, dict) else {}


def authorization_text_from_context(ctx: Any) -> str:
    snapshot = _world_state_snapshot(ctx)
    session = snapshot.get("session")
    if not isinstance(session, dict):
        return ""
    text = str(session.get("last_user_prompt") or "")
    return text[:1000]


def is_skill_note_read_authorized(title: str, ctx: Any) -> bool:
    """Conservative guard for model-initiated note reads.

    L1 is the only runtime level allowed to read notes. If the host exposes the
    last user prompt, require either note intent words or title tokens. If the
    prompt is unavailable, do not block direct/test contexts; the provenance
    payload still records the empty authorization text.
    """
    if _ctx_level(ctx) > 1:
        return False
    prompt = authorization_text_from_context(ctx).strip().lower()
    if not prompt:
        return True
    if any(phrase in prompt for phrase in _PROMPT_DENY_PHRASES):
        return False
    if any(word in prompt for word in _PROMPT_AUTH_WORDS):
        return True
    title_tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", str(title or "").lower())
        if len(token) >= 3
    ]
    return any(token in prompt for token in title_tokens)


def _parent_turn_id(ctx: Any) -> str:
    return str(getattr(ctx, "parent_turn_id", "") or "")[:128]


def record_note_read(
    note: NoteRecord,
    window: NoteTextWindow,
    *,
    ctx: Any = None,
    read_mode: str = "skill",
    session_id: str | None = None,
) -> int:
    requested_by = "ui" if str(read_mode).startswith("ui_") else "skill"
    parent_turn_id = _parent_turn_id(ctx)
    payload = {
        "schema": NOTE_READ_SCHEMA,
        "source": NOTE_SOURCE,
        "read_mode": str(read_mode or "skill"),
        "requested_by": requested_by,
        "note_id": note.note_id,
        "title": note.title,
        "safe_title": note.safe_title,
        "path": str(note.path),
        "relative_path": note.relative_path,
        "sha256": note.sha256,
        "size": note.size,
        "mtime_ns": note.mtime_ns,
        "selection_start": int(window.start),
        "selection_end": int(window.end),
        "excerpt_sha256": hashlib.sha256(window.text.encode("utf-8")).hexdigest(),
        "truncated": bool(window.truncated),
        "turn_id": parent_turn_id,
        "authorization_text": authorization_text_from_context(ctx),
    }
    return canonical_log.append(
        NOTE_READ_KIND,
        payload=payload,
        session_id=session_id or parent_turn_id or None,
        sentinel_reason="mononote:note_read",
    )
