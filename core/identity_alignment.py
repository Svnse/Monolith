"""Identity-alignment scoring — confidentity = confidence × identity_alignment.

M2 V0, the load-bearing M0 prerequisite. Self-derived ACUs carry no confidentity
by default (only Origin-0 locked claims do). This module computes how aligned a
claim is with the *current identity corpus* (Origin-0 + accepted Emergent), and
folds it with a provenance-weighted confidence into a single confidentity score.

Backend seam (default chosen empirically by the fire-rate test, not by ideology):
  * ``lexical`` (default) — deterministic overlap coefficient over the claim's
    *content* tokens vs the corpus. Tuned for short triples (normalize by the
    claim's tokens, not raw count) + stopword filtering. Zero deps, cheap.
  * ``embed`` (opt-in via MONOLITH_IDENTITY_ALIGN_EMBED=1) — sentence-embedding
    cosine similarity. ``transformers``+``torch`` are already declared deps.
    Lazily loaded; ANY failure falls back to lexical so scoring never breaks.

Rationale for lexical-default: dependency/operational minimalism for a
propose-only V0 + the human gate makes false positives cheap (a bad proposal is
a reject, not identity corruption). NOTE: this is NOT a replay-determinism
argument — confidentity is computed read-time and not part of canonical_log
replay. The real risk is the opposite (silence/false-negatives); the fire-rate
acceptance test (tests/test_identity_emergence_firerate.py) is the guard.
"""
from __future__ import annotations

import os
import re

_TOKEN_RE = re.compile(r"[a-z0-9_./]+")  # mirrors core/acu_retrieval._TOKEN_RE

# Provenance gradient — the Mad-Cow ceiling lives here: self-provenance claims
# top out at 0.5 confidentity, so a claim cannot self-promote to identity-grade
# without independent user/world reinforcement (defense-in-depth behind the
# propose-only human gate).
_PROV_WEIGHT = {"user": 1.0, "world": 0.8, "tool": 0.8, "self": 0.5}

_EMBED_FLAG = "MONOLITH_IDENTITY_ALIGN_EMBED"

# Function words that would inflate overlap without carrying identity content.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "not", "no", "of", "to", "in", "on",
    "over", "for", "with", "as", "is", "am", "are", "be", "been", "being",
    "i", "me", "my", "we", "us", "our", "you", "your", "it", "its", "this",
    "that", "these", "those", "do", "does", "did", "done", "so", "than",
    "then", "by", "at", "from", "into", "out", "up", "down", "what", "how",
    "when", "where", "who", "which", "whose", "if", "else", "while", "because",
    "therefore", "about", "only", "also", "can", "will", "have", "has", "had",
})


def _content_tokens(text: str) -> set[str]:
    # Strip edge ./ so sentence punctuation ("fluency.") doesn't fork a token
    # while interior dots/slashes ("file.py", "a/b") survive.
    toks = (t.strip("./") for t in _TOKEN_RE.findall(str(text or "").lower()))
    return {t for t in toks if t and t not in _STOPWORDS}


# ── alignment backends ────────────────────────────────────────────────

def _align_lexical(text: str, corpus: str) -> float:
    """Overlap coefficient: |claim_tokens ∩ corpus_tokens| / |claim_tokens|.

    Normalized by the CLAIM's token count (not raw overlap) so a short triple
    that fully lands in the corpus scores 1.0 rather than being lost in a large
    corpus's denominator.
    """
    claim = _content_tokens(text)
    if not claim:
        return 0.0
    corpus_tokens = _content_tokens(corpus)
    if not corpus_tokens:
        return 0.0
    overlap = len(claim & corpus_tokens)
    return overlap / len(claim)


def _align_embed(text: str, corpus: str) -> float:
    """Cosine similarity of sentence embeddings, scaled to [0, 1].

    Lazy, thread-safe singleton model load. Raises on any failure; the public
    ``compute_identity_alignment`` catches and falls back to lexical.
    """
    model = _get_embed_model()
    import numpy as np  # transformers pulls numpy
    vecs = model.encode([str(text or ""), str(corpus or "")])
    a, b = vecs[0], vecs[1]
    denom = float((np.linalg.norm(a) * np.linalg.norm(b)) or 1.0)
    cos = float(np.dot(a, b) / denom)
    # cosine is [-1, 1]; map to [0, 1] and clamp.
    return max(0.0, min(1.0, (cos + 1.0) / 2.0))


_EMBED_MODEL = None
_EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # pinned for reproducibility


def _get_embed_model():
    global _EMBED_MODEL
    if _EMBED_MODEL is None:
        from sentence_transformers import SentenceTransformer  # optional dep
        _EMBED_MODEL = SentenceTransformer(_EMBED_MODEL_NAME)
    return _EMBED_MODEL


# ── public API ────────────────────────────────────────────────────────

def resolve_backend(backend: str | None) -> str:
    """Resolve the alignment backend. Explicit arg wins; else the env flag;
    else lexical (the default)."""
    if backend in ("lexical", "embed"):
        return backend
    raw = str(os.environ.get(_EMBED_FLAG, "0")).strip().lower()
    return "embed" if raw in {"1", "true", "yes", "on"} else "lexical"


def compute_identity_alignment(text: str, corpus: str, *, backend: str | None = None) -> float:
    """Identity alignment of a claim against the identity corpus, in [0, 1]."""
    resolved = resolve_backend(backend)
    if resolved == "embed":
        try:
            return _align_embed(text, corpus)
        except Exception:
            return _align_lexical(text, corpus)
    return _align_lexical(text, corpus)


def _confidence(acu_row: dict) -> float:
    """Provenance-weighted confidence in [0, 1] (the Mad-Cow gradient).

    Reinforcement is deliberately NOT a factor here: under propose-only,
    *surfacing* a claim for human review is not *promoting* it, so a fresh but
    highly-aligned claim must be able to cross the surfacing threshold (else the
    loop is silent — the dominant V0 risk). Reinforcement is used downstream as
    a candidate tiebreaker, not as a confidentity gate.
    """
    prov = str(acu_row.get("provenance") or "self").strip().lower()
    return _PROV_WEIGHT.get(prov, _PROV_WEIGHT["self"])


def score_confidentity(acu_row: dict, corpus: str, *, backend: str | None = None) -> float:
    """confidentity = confidence(row) × identity_alignment(row, corpus), clamped."""
    alignment = compute_identity_alignment(acu_row.get("canonical", ""), corpus, backend=backend)
    c = _confidence(acu_row) * alignment
    return max(0.0, min(1.0, c))


# ── stability (M3 unify: the disposition discriminator) ────────────────
#
# One identity signal, two dispositions split by stability:
#   identity-aligned + stable      -> EMERGENCE (consolidate into identity, M2)
#   identity-aligned + NOT stable  -> CURIOSITY (a fresh pull to explore, M3)
# A claim has exactly one disposition, so the two features never contradict.
# Stability rises as a claim is reinforced / crystallized / truth-confirmed —
# this is also the Mad-Cow gate: a fresh self-claim can't be consolidated until
# it has earned reinforcement.
STABILITY_THRESHOLD = 0.5

_L_LEVEL_STABILITY = {"L1": 0.2, "L2": 0.6, "L3": 1.0}


def stability_score(acu_row: dict) -> float:
    """How settled/integrated a claim is, in [0, 1]. Reinforcement-dominant,
    l-level-weighted, with a small truth-confirmed bonus."""
    reinforcement = float(acu_row.get("reinforcement", 1) or 1)
    reinf_comp = min(reinforcement, 5.0) / 5.0
    l_comp = _L_LEVEL_STABILITY.get(str(acu_row.get("l_level") or "L1").strip().upper(), 0.2)
    truth_bonus = 0.1 if str(acu_row.get("truth") or "").strip().lower() == "confirmed" else 0.0
    return max(0.0, min(1.0, 0.6 * reinf_comp + 0.4 * l_comp + truth_bonus))
