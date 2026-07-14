"""EQID v1 — deterministic equivalence grouping via inverse-relation rewrites.

EQID groups CIDs that are deterministic rewrites of one another (an inverse-
relation pair, or a symmetric relation with swapped entities) so retrieval can
treat them as one assertion. It GROUPS content-addressed claims; it never
collapses them, never mints or reassigns a CID, and never participates in
identity. Membership is decided purely by a frozen, versioned rewrite map applied
to parse_triple output — no embeddings, no LLM, fully deterministic (so replay
re-derives it).
"""
from __future__ import annotations

import hashlib

from core.acatalepsy.normalize import normalize_canonical, parse_triple

__all__ = (
    "EQID_MAP_VERSION",
    "equivalence_key",
    "compute_eqid",
    "compute_eqid_for_form",
    "backfill_eqids",
)

# Bump when the rewrite map changes — like cf_version, it is part of the eqid
# hash so a map change never silently re-buckets existing groups.
EQID_MAP_VERSION = 1

# Hand-curated, conservative inverse pairs. (a, b) means "X a Y" == "Y b X".
# Add only well-established inverses.
_INVERSE_PAIRS: tuple[tuple[str, str], ...] = (
    ("capital_of", "has_capital"),
    ("parent_of", "child_of"),
    ("author_of", "written_by"),
    ("preferred_over", "dispreferred_to"),
    ("part_of", "has_part"),
    ("causes", "caused_by"),
    ("precedes", "follows"),
    ("owns", "owned_by"),
)

# Relations equal to their own inverse: "X r Y" == "Y r X".
_SYMMETRIC: frozenset[str] = frozenset({
    "sibling_of", "related_to", "married_to", "adjacent_to", "equivalent_to",
})

_INVERSE: dict[str, str] = {}
for _a, _b in _INVERSE_PAIRS:
    _INVERSE[_a] = _b
    _INVERSE[_b] = _a


def equivalence_key(triple) -> tuple[str, str, str, str] | None:
    """Canonical orientation of a triple for equivalence grouping.

    A claim and its inverse rewrite yield the SAME key:
      ("paris","capital_of","france") and ("france","has_capital","paris")
      both -> ("capital_of","paris","france", <quals>).
    Symmetric relations are entity-order-independent. A relation with no known
    inverse returns its own (rel, a, b) — a singleton key (it only matches an
    identical canonical form, which already shares a CID).
    """
    if triple is None:
        return None
    a = str(triple.entity_a)
    b = str(triple.entity_b)
    rel = str(triple.relation)
    quals = str(triple.qualifiers or "")
    if rel in _SYMMETRIC:
        lo, hi = sorted((a, b))
        return (rel, lo, hi, quals)
    inv = _INVERSE.get(rel)
    if inv is not None:
        canonical_rel = min(rel, inv)
        if rel == canonical_rel:
            return (canonical_rel, a, b, quals)
        return (canonical_rel, b, a, quals)   # rewrite to canonical orientation
    return (rel, a, b, quals)


def compute_eqid(triple) -> str | None:
    """Content hash over the equivalence key + map version. None for an
    unparseable triple. Deterministic — replay re-derives the same value."""
    key = equivalence_key(triple)
    if key is None:
        return None
    payload = f"{EQID_MAP_VERSION}\x1f" + "\x1f".join(key)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"eqid:sha256:{digest}"


def compute_eqid_for_form(canonical_form: str) -> str | None:
    """Normalize a raw/stored canonical form, then compute its eqid. The single
    entry point — always normalizes first so callers can't skip it."""
    return compute_eqid(parse_triple(normalize_canonical(canonical_form)))


def backfill_eqids(conn) -> int:
    """Assign eqids to crystallized rows (cid set) that lack one. L1 stubs
    (cid IS NULL) are skipped — EQID groups CIDs, not uncrystallized stubs.
    Idempotent. Returns the number of rows updated. Caller owns the transaction."""
    rows = conn.execute(
        "SELECT id, canonical FROM acus WHERE cid IS NOT NULL AND eqid IS NULL"
    ).fetchall()
    n = 0
    for r in rows:
        value = compute_eqid_for_form(r["canonical"])
        if value is not None:
            conn.execute("UPDATE acus SET eqid=? WHERE id=?", (value, int(r["id"])))
            n += 1
    return n
