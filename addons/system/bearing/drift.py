"""Semantic-drift detector for bearing.current_frame — the world-model grounder.

Pure (no I/O, no env). Flags when the model-authored current_frame has gone
STALE relative to what the user is now asking — the confident-wrong-on-stale-
frame case the offline replay (tools/frame_drift) measured at ~40% of outer
turns. The model OBEYS its frame, so a stale frame steers it confidently wrong;
this is the one genuinely-unbuilt piece of the cognitive world model.

Heuristic, deliberately cheap (runs inside the prompt-assembly interceptor):
content-word overlap between current_frame and the recent user asks, gated by
frame age (turns since current_frame was last updated). The compiler builds a
StalenessSignal from a drift observation and plugs it into the existing staleness
spine (staleness.py) as a third detector — observe-first: log the observation,
inject the nudge only behind a second flag.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Minimal stopword set — only the highest-frequency function words. Kept small on
# purpose: content overlap is the signal, so over-filtering would hide real
# topic matches and inflate the drift rate.
_STOPWORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "if", "of", "to", "in", "on", "for",
    "with", "at", "by", "from", "as", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "i", "you", "we",
    "they", "he", "she", "me", "my", "your", "our", "their", "what", "which",
    "who", "how", "when", "where", "why", "do", "does", "did", "can", "could",
    "would", "should", "will", "shall", "may", "might", "not", "no", "yes", "so",
    "than", "then", "there", "here", "one", "up", "out", "about", "into", "over",
    "just", "like", "get", "got", "make", "made", "want", "need",
})

_WORD_RE = re.compile(r"[a-z][a-z0-9_]{2,}")


def _content_tokens(text: str) -> set[str]:
    """Lowercase content words (len>=3, non-stopword) of *text*."""
    return {w for w in _WORD_RE.findall(text.lower()) if w not in _STOPWORDS}


def _lexical_overlap(frame_tokens: set[str], ask_tokens: set[str]) -> float:
    """Fraction of the frame's content words present in the asks. 0 if either empty."""
    if not frame_tokens or not ask_tokens:
        return 0.0
    return len(frame_tokens & ask_tokens) / len(frame_tokens)


# Message-block markers that are NOT a real user ask (injected blocks / tool turns).
_NON_ASK_PREFIXES = (
    "Tool results", "[TOOL", "[SUBAGENT_RESULT", "[BEARING", "[RUNTIME STATE",
    "[OBSERVER", "[SELF-CHECK", "[LAST TURN", "[REVIEW QUEUE",
)
_CHANNEL_RE = re.compile(r"^\[CHANNEL:[^\]]*\]\s*")


def _msg_text(m: dict) -> str:
    """Text of a live message dict — content may be a str or a list of blocks."""
    c = m.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(
            b.get("text", "") for b in c
            if isinstance(b, dict) and b.get("type") == "text"
        )
    return ""


def recent_asks(messages: list[dict], k: int = 3) -> list[str]:
    """The last *k* real user asks from a live message list (newest last).

    Skips ephemeral coalescer blocks, tool-result turns, and injected blocks —
    the same filter the offline extractor used, adapted to live message dicts.
    Pure; never raises on odd shapes.
    """
    out: list[str] = []
    for m in reversed(messages or ()):
        if not isinstance(m, dict) or m.get("role") != "user":
            continue
        if m.get("ephemeral"):
            continue
        src = str(m.get("source") or "")
        if src and src not in ("user", "chat"):
            continue
        text = _msg_text(m).strip()
        if not text or any(text.lstrip().startswith(p) for p in _NON_ASK_PREFIXES):
            continue
        out.append(_CHANNEL_RE.sub("", text))
        if len(out) >= k:
            break
    return list(reversed(out))


def overlap_of(current_frame: str, asks: list[str]) -> tuple[float, int, int]:
    """Raw drift signal for the observe ledger.

    Returns (overlap, frame_token_count, ask_token_count). overlap is the fraction
    of the frame's content words present in the union of *asks*.
    """
    ft = _content_tokens(current_frame or "")
    at: set[str] = set()
    for a in asks or ():
        at |= _content_tokens(a or "")
    return _lexical_overlap(ft, at), len(ft), len(at)


@dataclass(frozen=True)
class DriftObservation:
    is_drift: bool
    overlap: float        # |frame ∩ asks| / |frame|, on content words
    frame_age: int        # turns since current_frame was last updated
    frame_tokens: int     # content-word count of the frame (0 => unjudgeable)
    detail: str


# Defaults: don't flag a frame younger than 2 turns; flag when < 18% of the
# frame's content words appear in the recent asks. Observe-first exists precisely
# to calibrate these against the live ledger before any nudge fires.
_AGE_THRESHOLD = 2
_OVERLAP_THRESHOLD = 0.18


def detect_drift(
    current_frame: str,
    recent_asks: list[str],
    frame_age: int,
    *,
    age_threshold: int = _AGE_THRESHOLD,
    overlap_threshold: float = _OVERLAP_THRESHOLD,
) -> DriftObservation:
    """Judge whether *current_frame* has gone stale vs *recent_asks*.

    Drift iff: the frame has content words, has persisted >= age_threshold turns,
    there is at least one ask to compare against, AND the fraction of the frame's
    content words present in the union of recent asks is below overlap_threshold.
    Never raises; a frame with no content words is unjudgeable (is_drift False).
    """
    frame_tokens = _content_tokens(current_frame or "")
    ask_tokens: set[str] = set()
    for a in recent_asks or ():
        ask_tokens |= _content_tokens(a or "")

    if not frame_tokens:
        return DriftObservation(False, 0.0, frame_age, 0, "frame has no content words")
    if not ask_tokens:
        return DriftObservation(False, 0.0, frame_age, len(frame_tokens), "no recent asks to compare")

    overlap = _lexical_overlap(frame_tokens, ask_tokens)
    aged = frame_age >= age_threshold
    is_drift = aged and overlap < overlap_threshold
    if is_drift:
        detail = f"frame stale {frame_age} turns, content overlap {overlap:.2f} < {overlap_threshold}"
    elif not aged:
        detail = f"frame young ({frame_age} < {age_threshold} turns) — not judged stale"
    else:
        detail = f"frame still on-topic (overlap {overlap:.2f})"
    return DriftObservation(is_drift, overlap, frame_age, len(frame_tokens), detail)
