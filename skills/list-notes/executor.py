"""List saved MonoNote notes by title."""
from __future__ import annotations

from core.mononote import store


def run(cmd: dict, _ctx) -> str:
    pattern = str(cmd.get("pattern", "")).strip().lower()
    try:
        max_results = int(cmd.get("max_results", 50))
    except (TypeError, ValueError):
        max_results = 50

    try:
        records = store.list_notes(pattern, max_results=max_results)
    except Exception as exc:
        return f"[list_notes: error - {exc}]"

    if not records:
        return "[list_notes: no notes match]" if pattern else "[list_notes: no notes saved]"
    return f"[list_notes: {len(records)} note(s)]\n" + "\n".join(
        f"- {record.safe_title}" for record in records
    )
