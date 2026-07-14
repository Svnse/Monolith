"""Read a saved MonoNote note by title and emit note-read provenance."""
from __future__ import annotations

from core.mononote import provenance, store


def _safe_missing_name(title: str) -> str:
    try:
        return store.safe_title(title)
    except Exception:
        return "note"


def run(cmd: dict, ctx) -> str:
    title = str(cmd.get("title", "")).strip()
    if not title:
        return "[read_note: no title provided]"

    if not provenance.is_skill_note_read_authorized(title, ctx):
        return "[read_note: denied - user did not ask to read notes this turn]"

    try:
        note, body = store.load_note(title)
    except FileNotFoundError:
        return f"[read_note: no note named '{_safe_missing_name(title)}']"
    except Exception as exc:
        return f"[read_note: error - {exc}]"

    window = store.slice_note_text(
        body,
        offset=cmd.get("offset", 0),
        max_chars=cmd.get("max_chars", 8000),
        selection_start=cmd.get("selection_start"),
        selection_end=cmd.get("selection_end"),
    )
    try:
        event_id = provenance.record_note_read(note, window, ctx=ctx, read_mode="skill")
    except Exception as exc:
        return f"[read_note: provenance error - {exc}]"

    truncated = "true" if window.truncated else "false"
    header = (
        f'[NOTE_READ event_id={event_id} note_id={note.note_id} '
        f'title="{note.title}" hash={note.sha256} source=mononote '
        f"range={window.start}:{window.end} truncated={truncated}]"
    )
    return f"{header}\n{window.text}"
