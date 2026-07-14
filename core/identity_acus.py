"""Origin 0 identity loader for ACU substrate.

V0 treats Origin 0 as immutable seed material. It is loaded into ACUs as
LOCKED claims so downstream readers can include identity facts without making
the model infer them from a persona prompt.
"""
from __future__ import annotations

from core.acu_store import ACUStore
from core.identity import load_identity


SOURCE = "identity_origin_0"
LOCK_REASON = "origin_0"


def extract_origin0_claims(identity_text: str | None = None) -> list[str]:
    """Extract compact, deterministic claims from the Origin 0 markdown."""

    from core.identity_regions import split_regions

    raw = load_identity() if identity_text is None else str(identity_text or "")
    # Region-aware: only the frozen Origin-0 region becomes LOCKED claims.
    # Emergent prose (below EMERGENT:BEGIN) must never be locked as Origin-0.
    text, _emergent = split_regions(raw)
    claims: list[str] = []
    section = "Identity"
    paragraph: list[str] = []

    def flush() -> None:
        if not paragraph:
            return
        body = " ".join(part.strip() for part in paragraph if part.strip()).strip()
        paragraph.clear()
        if body:
            claims.append(f"Origin 0 / {section}: {body}")

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            flush()
            continue
        if line.startswith("#"):
            flush()
            section = line.lstrip("#").strip() or section
            continue
        paragraph.append(line)
    flush()

    deduped: list[str] = []
    seen: set[str] = set()
    for claim in claims:
        if claim not in seen:
            seen.add(claim)
            deduped.append(claim)
    return deduped


def ensure_origin0_acus_loaded() -> list[int]:
    """Load Origin 0 markdown paragraphs into the ACU store as LOCKED."""

    claims = extract_origin0_claims()
    if not claims:
        return []
    store = ACUStore()
    try:
        return [
            store.ingest_locked(
                claim,
                source=SOURCE,
                lock_reason=LOCK_REASON,
                confidentity=1.0,
            )
            for claim in claims
        ]
    finally:
        store.close()
