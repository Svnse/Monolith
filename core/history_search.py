"""
core/history_search.py — Search engine for chat archive JSON files.

Used by:
  - UI history tab (live filtering as user types)
  - LLM tool (search_history tool_call)
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class SearchResult:
    title: str
    date: str
    role: str
    snippet: str       # matched message text excerpt
    path: Path
    message_index: int # which message matched (0-based)

    def to_context_block(self) -> str:
        """Format for injecting into LLM context."""
        return (
            f"[{self.date}] {self.title}\n"
            f"  {self.role}: {self.snippet}"
        )


def search_archives(
    query: str,
    archive_dir: Path,
    max_results: int = 8,
    roles: tuple[str, ...] = ("user", "assistant"),
) -> list[SearchResult]:
    """
    Full-text search across all saved chat archives.
    Matches are ranked by number of query terms found per message.
    Returns up to max_results results.
    """
    if not query or not query.strip():
        return []

    terms = [t.lower() for t in query.strip().split() if t.strip()]
    if not terms:
        return []

    scored: list[tuple[int, SearchResult]] = []

    for path in sorted(archive_dir.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        meta = data.get("meta", {})
        title = meta.get("title", path.stem)
        # File-level updated_at is the whole-conversation last-edit time; use it
        # only as a fallback. Each match is dated by its OWN message `time` so an
        # old message in a recently-touched archive isn't mislabeled as recent
        # (when-plane fix).
        file_date = (meta.get("updated_at") or "")[:10]  # YYYY-MM-DD

        for idx, msg in enumerate(data.get("messages", [])):
            role = msg.get("role", "user")
            if role not in roles:
                continue
            date = (str(msg.get("time") or "")[:10]) or file_date
            text = msg.get("text", "")
            lower_text = text.lower()

            hits = sum(1 for t in terms if t in lower_text)
            if hits == 0:
                continue

            # Build snippet: find first matching term position
            pos = len(text)
            for t in terms:
                p = lower_text.find(t)
                if p != -1 and p < pos:
                    pos = p
            start = max(0, pos - 40)
            end = min(len(text), pos + 60)
            raw = text[start:end].replace("\n", " ").strip()
            snippet = ("…" if start > 0 else "") + raw + ("…" if end < len(text) else "")

            scored.append((
                hits,
                SearchResult(
                    title=title,
                    date=date,
                    role=role,
                    snippet=snippet,
                    path=path,
                    message_index=idx,
                )
            ))

    # Sort by hit count descending, then by date descending
    scored.sort(key=lambda x: (x[0], x[1].date), reverse=True)
    return [r for _, r in scored[:max_results]]


def search_archives_by_title(
    query: str,
    archive_dir: Path,
) -> list[Path]:
    """
    Lightweight title-only filter — used by the UI list to show/hide items.
    Returns paths of archives whose title matches the query.
    """
    if not query.strip():
        return []
    q = query.strip().lower()
    matched: list[Path] = []
    for path in archive_dir.glob("*.json"):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            title = data.get("meta", {}).get("title", path.stem)
        except Exception:
            title = path.stem
        if q in title.lower():
            matched.append(path)
    return matched
