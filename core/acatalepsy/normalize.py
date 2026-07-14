"""Canonical-form normalizer for the Acatalepsy substrate spine.

Pure, deterministic, no I/O. Produces the syntactic normal form that the
content hash (CID = hash(canonical_form + cf_version)) is computed over,
so this module is retroactively load-bearing for *identity*.

CRITICAL: the R1-R7 rules below are frozen as part of ``CF_VERSION``. Any
change to their semantics is a ``CF_VERSION`` bump plus an explicit
re-crystallization migration — NEVER an in-place edit — otherwise the same
claim normalizes differently, mints a different CID, and forks identity
(which, per spec, never re-merges).

This module only *shapes* the form. The atomicity gate
(``core.acatalepsy.atomicity.is_atomic``) decides whether a form is a
valid single subject-predicate assertion.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass

__all__ = ("CF_VERSION", "CanonicalTriple", "normalize_canonical", "parse_triple")

# Frozen normalizer version. Bump only alongside a re-crystallization
# migration; it is hashed into every CID.
CF_VERSION: int = 1

_WS_RE = re.compile(r"\s+")

# R7: smart quotes / dashes → ASCII so visually-identical claims share a form.
_SMART_MAP = {
    "‘": "'", "’": "'", "‚": "'", "‛": "'",
    "“": '"', "”": '"', "„": '"', "‟": '"',
    "‒": "-", "–": "-", "—": "-", "―": "-",
}
_SMART_TABLE = {ord(k): v for k, v in _SMART_MAP.items()}

# R6: trailing punctuation stripped per part (internal punctuation is content).
_TRAILING_PUNCT = ".,;:"


def normalize_canonical(raw: object) -> str:
    """Return the canonical normal form of ``raw``.

    Rules, applied in order:
      R1  Unicode NFC normalization
      R7  smart quotes / em-dashes → ASCII
      R2  strip outer whitespace
      R3  casefold (full-unicode lowercase)
      R4  collapse internal whitespace runs → single space
      R5  pipe spacing: split on '|', strip each part, rejoin as ' | '
      R6  strip trailing .,;: from each part (internal punctuation kept)

    Idempotent: ``normalize_canonical(normalize_canonical(x)) ==
    normalize_canonical(x)``.
    """
    if raw is None:
        return ""
    s = unicodedata.normalize("NFC", str(raw))   # R1
    s = s.translate(_SMART_TABLE)                 # R7
    s = s.strip()                                 # R2
    s = s.casefold()                              # R3
    s = _WS_RE.sub(" ", s)                        # R4
    if "|" in s:                                  # R5 + R6
        parts = [p.strip().rstrip(_TRAILING_PUNCT).strip() for p in s.split("|")]
        return " | ".join(parts)
    return s.rstrip(_TRAILING_PUNCT).strip()


@dataclass(frozen=True)
class CanonicalTriple:
    entity_a: str
    relation: str
    entity_b: str
    qualifiers: str | None


def parse_triple(normalized: str) -> CanonicalTriple | None:
    """Parse a (already-normalized) canonical form into its triple.

    Returns ``None`` unless the form has 3 or 4 non-empty pipe parts —
    mirroring ``atomicity._MIN_PARTS`` / ``_MAX_PARTS``. The 4th part, if
    present, is the qualifiers slot.
    """
    parts = [p.strip() for p in str(normalized or "").split("|")]
    if len(parts) < 3 or len(parts) > 4:
        return None
    if any(not p for p in parts):
        return None
    qualifiers = parts[3] if len(parts) == 4 else None
    return CanonicalTriple(parts[0], parts[1], parts[2], qualifiers)
