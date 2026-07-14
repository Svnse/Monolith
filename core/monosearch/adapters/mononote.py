from __future__ import annotations

from datetime import datetime, timezone

from core.mononote import store
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record

_ID_PREFIX = "mononote:"
_SNIPPET_CHARS = 360


def _now_epoch() -> float:
    return datetime.now(timezone.utc).timestamp()


def _snippet(body: str, query: str, *, chars: int = _SNIPPET_CHARS) -> str:
    text = str(body or "")
    if not text:
        return ""
    needle = str(query or "").strip().lower()
    if not needle:
        return text[:chars]
    lower = text.lower()
    pos = lower.find(needle)
    if pos < 0:
        for token in needle.split():
            pos = lower.find(token)
            if pos >= 0:
                break
    if pos < 0:
        return text[:chars]
    start = max(0, pos - chars // 3)
    end = min(len(text), start + chars)
    prefix = "..." if start > 0 else ""
    suffix = "..." if end < len(text) else ""
    return f"{prefix}{text[start:end]}{suffix}"


def _metadata(record, *, snippet: str = "") -> dict:
    return {
        "kind": "mononote_note",
        "note_id": record.note_id,
        "title": record.title,
        "safe_title": record.safe_title,
        "path": str(record.path),
        "relative_path": record.relative_path,
        "sha256": record.sha256,
        "size": record.size,
        "mtime_ns": record.mtime_ns,
        "snippet": snippet,
        "acu_evidence": False,
        "evidence_instruction": "Call read_note or send the note to chat to create mononote_note_read provenance.",
    }


def _text(record, snippet: str, *, detail: bool = False) -> str:
    lines = [
        "[MONONOTE]",
        f"title: {record.safe_title}",
        f"path: {record.path}",
        f"sha256: {record.sha256}",
        "acu_evidence: false",
        "evidence_instruction: search/index snippets are not ACU evidence; call read_note or send the note to chat to log mononote_note_read.",
    ]
    if snippet:
        label = "content" if detail else "snippet"
        lines.append(f"{label}:")
        lines.append(snippet)
    return "\n".join(lines)


class MonoNoteAdapter(SourceAdapter):
    name = "mononote"
    evidence_tier = EvidenceTier.LITERAL

    def _to_record(self, note, snippet: str = "", *, detail: bool = False) -> Record:
        return Record(
            namespaced_id=f"{_ID_PREFIX}{note.note_id}",
            source=self.name,
            provenance=Provenance.USER,
            recurrence_key=None,
            text=_text(note, snippet, detail=detail),
            metadata=_metadata(note, snippet=snippet),
            ts=None,
            evidence_tier=self.evidence_tier,
        )

    def search(self, query: str, filters: dict, limit: int) -> list[Record]:
        q = str(query or "").strip().lower()
        max_results = max(1, min(200, int(limit or 20)))
        scored: list[tuple[int, object, str]] = []
        for note in store.list_notes(max_results=200):
            try:
                _loaded, body = store.load_note(note.safe_title)
            except Exception:
                continue
            haystack = " ".join((note.safe_title, note.title, body)).lower()
            score = 1 if not q else 0
            if q and q in haystack:
                score += 100
            for token in q.split():
                if token and token in haystack:
                    score += 8
            if score <= 0:
                continue
            scored.append((score, note, _snippet(body, q)))
        scored.sort(key=lambda item: (-item[0], item[1].safe_title.lower()))
        return [self._to_record(note, snippet) for _, note, snippet in scored[:max_results]]

    def get(self, namespaced_id: str) -> Record | None:
        raw = str(namespaced_id or "").strip()
        if raw.startswith(_ID_PREFIX):
            target = raw[len(_ID_PREFIX):]
            for note in store.list_notes(max_results=200):
                if note.note_id == target:
                    try:
                        _loaded, body = store.load_note(note.safe_title)
                    except Exception:
                        body = ""
                    return self._to_record(note, body[:2000], detail=True)
            return None
        if raw.startswith("note:"):
            title = raw[len("note:"):]
            try:
                note, body = store.load_note(title)
            except Exception:
                return None
            return self._to_record(note, body[:2000], detail=True)
        return None

    def list(self, filters: dict, limit: int) -> list[Record]:
        records = store.list_notes(max_results=max(1, min(200, int(limit or 20))))
        return [self._to_record(record, "") for record in records]
