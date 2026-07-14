"""MonoFrame — content-word divergence (the one surviving v1 primitive).

`token_divergence` is now used only by MonoFrame v2's nearest-card retrieval
(correction_store.nearest_human_card) to rank a stored correction against the
current ask. The v1 numeric second-opinion loop (assess/Divergence + the
per-turn observe ledger) was superseded by the CorrectionCard pipeline and
removed. Pure: no I/O, no model calls, no env reads.
"""
from __future__ import annotations

from .drift import _content_tokens


def token_divergence(frame_a: str, frame_b: str) -> float:
    """Symmetric content-word divergence: 0.0 (identical) .. 1.0 (disjoint).

    1 - Jaccard over content tokens (stopwords removed, case-folded — the same
    tokenizer drift.py uses, so the two detectors speak one vocabulary). Two
    empty frames are identical (0.0); one empty is full divergence (1.0).
    """
    a = _content_tokens(frame_a or "")
    b = _content_tokens(frame_b or "")
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return 1.0 - (len(a & b) / len(union))
