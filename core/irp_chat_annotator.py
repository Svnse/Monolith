"""IRP annotation for assistant chat emissions.

Lightweight scanner that runs when assistant text is logged to
canonical_log. Identifies claims that overlap with existing ACUs and
annotates the log payload with IRP label metadata. This gives the
Auditor richer input — it can see which assistant statements reinforce
or touch existing knowledge.

Not user-visible. Does not modify the assistant's displayed text.
Adds an ``irp_annotations`` field to the canonical_log payload only.
"""
from __future__ import annotations

import re
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9_./]+")
_MIN_TEXT_CHARS = 80
_MAX_ANNOTATIONS = 5


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def annotate_assistant_payload(
    text: str,
    payload: dict[str, Any],
    *,
    max_acus: int = 200,
) -> dict[str, Any]:
    """Add ``irp_annotations`` to a canonical_log assistant_message payload.

    Returns the (possibly enriched) payload. Never raises — returns
    the original payload unchanged on any error.
    """
    if not text or len(text.strip()) < _MIN_TEXT_CHARS:
        return payload

    try:
        from core.acu_store import ACUStore
        from core.irp import label_for_claim

        store = ACUStore()
        try:
            rows = store.retrieve(limit=max_acus)
        finally:
            store.close()

        if not rows:
            return payload

        text_tokens = _tokenize(text)
        if not text_tokens:
            return payload

        hits: list[dict[str, Any]] = []
        for row in rows:
            canonical = str(row.get("canonical", "")).lower()
            acu_tokens = _tokenize(canonical)
            if not acu_tokens:
                continue
            overlap = len(text_tokens & acu_tokens)
            if overlap < 2:
                continue
            label = label_for_claim(row)
            hits.append({
                "acu_id": row.get("id"),
                "label": label,
                "overlap": overlap,
                "canonical_preview": str(row.get("canonical", ""))[:120],
            })

        if not hits:
            return payload

        hits.sort(key=lambda h: h["overlap"], reverse=True)
        payload["irp_annotations"] = hits[:_MAX_ANNOTATIONS]
        return payload

    except Exception:
        return payload
