"""friction_differ — the load-bearing component of the Friction Organ.

Settles a prior-turn intent prediction by scoring the FRICTION in the user's
verbatim next message: how much re-mesh work the conversation had to do. This
is NOT a correctness verdict (register is profound; the user can be wrong about
their own intent) — it is a measure of observable repair, over three channels:

  1. repair MARKERS   — closed lexicon over the verbatim reply (high precision,
                        low recall: many real misses carry no marker)
  2. content OVERLAP  — did the reply build on the prior answer's specifics
                        (uptake) or pivot away (topic_drift)? This is the
                        load-bearing channel: it converts *some* silent
                        divergence into signal, so "low friction" and "no
                        signal" stop being the same observation.
  3. RE-ASK           — did the reply re-raise something the prior answer
                        claimed to cover?

Pure code, stdlib only, deterministic, NO LLM call (settlement independence:
the settler must not be the model judging itself). Reads only observable text
shapes — never infers what the user *feels* or *wants* (that is the rejected
operator_model / psyche-analyzer pattern).

friction_type is a CLOSED enum (FRICTION_TYPES). friction_score in [0, 1].
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# ── closed enum ──────────────────────────────────────────────────────
FRICTION_TYPES: tuple[str, ...] = (
    "uptake",          # positive: reply built on the prior answer's specifics
    "unresolved",      # no signal either way (NOT confirmation — see spec §5)
    "correction",      # explicit "no / not quite / I meant ..."
    "reframe",         # restating the ask in different terms / "the right view is"
    "clarify",         # re-explaining what was meant
    "register_shift",  # literal<->exploratory<->ironic move flagged in text
    "meta",            # comment on the model itself ("you're collapsing/drifting")
    "reask",           # re-raising something the prior answer claimed to cover
    "topic_drift",     # low overlap, new direction, no markers (silent-divergence proxy)
    "abandon",         # reply drops the thread entirely
)

# ── marker lexicon (channel 1) ───────────────────────────────────────
# Each marker maps to a regex over the verbatim reply.
# NOTE: do NOT wrap multi-word/punctuation alternatives in an outer \b(...)\b —
# a trailing \b after punctuation (e.g. "no,") or a leading \b before a space
# never anchors, silently killing the alternative. Each pattern is standalone.
_MARKER_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("correction", re.compile(
        r"(?i)(?:\bno\b[,.\s]|\bnope\b|that'?s (?:not|wrong)|\bnot quite\b|"
        r"\bincorrect\b|actually,? not|\bi meant\b|i didn'?t mean|that isn'?t|"
        r"isn'?t (?:right|correct)|\bwrong\b)")),
    ("meta", re.compile(
        r"(?i)you'?re (?:collapsing|drifting|sprawling|overcomplicat\w*|hedging|"
        r"repeating|missing|going in circles)|you (?:keep|always)\b|"
        r"stop (?:hedging|sprawling)|too (?:long|verbose|abstract)|you said this")),
    ("reframe", re.compile(
        r"(?i)the (?:real|right|better) (?:view|question|frame|axis|way)|"
        r"isn'?t (?:really )?the right|\breframe\b|rather than|instead of (?:think|fram)|"
        r"the (?:point|issue) is|i'?d (?:rather|frame)")),
    ("clarify", re.compile(
        r"(?i)what i mean(?:t)? (?:is|was)|to (?:be )?clarif|let me (?:re)?explain|"
        r"i should (?:have )?(?:said|been clear)|to put it (?:another|differently)")),
    ("register_shift", re.compile(
        r"(?i)i was (?:joking|kidding|being (?:sarcastic|ironic))|seriously[,. ]|"
        r"to be serious|jokes aside|\bliterally\b|not (?:literally|seriously)")),
)

# Re-ask: an interrogative cue. Combined with overlap (see _score) to fire.
_INTERROGATIVE_RE = re.compile(
    r"(\?|\b(?:why|how|what|when|where|which|who|didn'?t you|did you|can you|"
    r"could you|isn'?t (?:it|there)|again)\b)", re.I)

# ── content-overlap (channel 2) ──────────────────────────────────────
_WORD_RE = re.compile(r"[a-z0-9_]+", re.I)

_STOPWORDS: frozenset[str] = frozenset("""
a an the and or but if then else of to in on at for with from by as is are was
were be been being this that these those it its it's i you he she we they them
me my your our their his her do does did doing done have has had having will
would can could should may might must shall not no yes so than too very just
about into over under again more most some any all each one two also which who
whom whose what when where why how here there now then up down out off only own
same such only s t re ve ll d m o
""".split())

# tuning constants (the calibration step sets/validates these)
_UPTAKE_OVERLAP = 0.30     # >= this  -> reply builds on prior answer's content
_DRIFT_OVERLAP = 0.10      # <= this  -> reply pivoted away
_MIN_TOKENS_FOR_OVERLAP = 3  # below this, overlap is unreliable

_MARKER_SCORE: dict[str, float] = {
    "correction": 0.95,
    "meta": 0.92,
    "reframe": 0.85,
    "register_shift": 0.78,
    "clarify": 0.70,
}
_REASK_SCORE = 0.80
_STRUCT_SCORE: dict[str, float] = {
    "uptake": 0.10,
    "unresolved": 0.45,
    "topic_drift": 0.70,
    "abandon": 0.65,
}
# dominant-type priority: strongest evidence wins the label
_TYPE_PRIORITY: tuple[str, ...] = (
    "correction", "meta", "reframe", "register_shift", "clarify",
    "reask", "topic_drift", "uptake", "abandon", "unresolved",
)


@dataclass(frozen=True)
class FrictionResult:
    friction_score: float          # 0..1
    friction_type: str             # one of FRICTION_TYPES
    channel_json: dict             # raw per-channel evidence (auditable)


def _content_tokens(text: str) -> set[str]:
    toks = {w.lower() for w in _WORD_RE.findall(text or "")}
    return {t for t in toks if t not in _STOPWORDS and len(t) > 1}


def _content_overlap(prior_answer: str, user_msg: str) -> float:
    """Overlap coefficient: |A ∩ U| / min(|A|, |U|) over content tokens.

    Measures what fraction of the user's content words also appear in the prior
    answer — high = same subject (uptake or reask), low = pivot (topic_drift).
    Returns -1.0 when either side is too short to judge (caller treats as
    'unresolved', never as drift)."""
    a = _content_tokens(prior_answer)
    u = _content_tokens(user_msg)
    if len(u) < _MIN_TOKENS_FOR_OVERLAP or len(a) < _MIN_TOKENS_FOR_OVERLAP:
        return -1.0
    inter = len(a & u)
    denom = min(len(a), len(u))
    return inter / denom if denom else -1.0


def _detect_markers(user_msg: str) -> list[str]:
    return [name for name, pat in _MARKER_PATTERNS if pat.search(user_msg or "")]


def _is_abandon(user_msg: str, overlap: float) -> bool:
    """Very short reply that doesn't engage the prior answer's content."""
    toks = _content_tokens(user_msg)
    return len(toks) <= 2 and (overlap < 0 or overlap <= _DRIFT_OVERLAP)


def score(prior_answer: str, user_msg: str, prediction: dict | None = None) -> FrictionResult:
    """Score the friction of `user_msg` as a reply to `prior_answer`.

    `prediction` (claim/falsifier) is accepted for future falsifier-aware
    scoring; v1 uses the observable channels. Pure + deterministic."""
    markers = _detect_markers(user_msg)
    overlap = _content_overlap(prior_answer, user_msg)
    interrogative = bool(_INTERROGATIVE_RE.search(user_msg or ""))

    # re-ask: interrogative that re-raises covered content (needs real overlap)
    reask = bool(interrogative and overlap >= _UPTAKE_OVERLAP and not markers)

    # structural read from overlap (channel 2)
    if overlap < 0:
        struct_type = "unresolved"
    elif _is_abandon(user_msg, overlap):
        struct_type = "abandon"
    elif overlap >= _UPTAKE_OVERLAP:
        struct_type = "uptake"
    elif overlap <= _DRIFT_OVERLAP:
        struct_type = "topic_drift"
    else:
        struct_type = "unresolved"

    # candidate (type, score) contributions
    candidates: list[tuple[str, float]] = [(struct_type, _STRUCT_SCORE.get(struct_type, 0.45))]
    for m in markers:
        candidates.append((m, _MARKER_SCORE.get(m, 0.6)))
    if reask:
        candidates.append(("reask", _REASK_SCORE))

    # friction_score = strongest evidence of re-mesh work (mirror, not minimizer)
    friction_score = round(max(s for _, s in candidates), 3)

    # dominant label by priority among the candidate types present
    present = {t for t, _ in candidates}
    friction_type = next((t for t in _TYPE_PRIORITY if t in present), "unresolved")

    channel_json = {
        "markers": markers,
        "answer_overlap": round(overlap, 3),
        "interrogative": interrogative,
        "reask": reask,
        "struct_type": struct_type,
        "user_token_count": len(_content_tokens(user_msg)),
    }
    return FrictionResult(friction_score, friction_type, channel_json)
