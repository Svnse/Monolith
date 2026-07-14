"""ACU retrieval — recall relevant learned claims into the generation frame.

Closes the write→read gap: the auditor pipeline extracts and stores ACUs,
this module retrieves the ones relevant to the current turn and injects
them as an ephemeral coalescer section so the model sees what it learned.

Scoring (STOPGAP): token-overlap between the user prompt and each ACU's
canonical form + behavioural signal (reinforcement) + recency. `veracity` is
dead. Deterministic, no embeddings, no new deps. The final ranking is
Authority/Truth-driven once the branch phases land.

Flag: MONOLITH_ACU_RECALL_V1 (default OFF — ships dark for first observation).
"""
from __future__ import annotations

import os
import re
from typing import Any

_FLAG_ENV = "MONOLITH_ACU_RECALL_V1"
_MAX_BLOCK_CHARS = 600
_MIN_PROMPT_CHARS = 40

_TOKEN_RE = re.compile(r"[a-z0-9_./]+")


def _flag_enabled() -> bool:
    raw = str(os.environ.get(_FLAG_ENV, "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _tokenize(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _recency_bonus(acu: dict) -> float:
    ts = acu.get("last_touched_ts") or acu.get("last_seen")
    if not ts:
        return 0.0
    try:
        from datetime import datetime, timezone
        dt = datetime.fromisoformat(str(ts))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(timezone.utc) - dt).total_seconds() / 86400.0
        return 0.5 if age_days <= 7 else 0.0
    except Exception:
        return 0.0


def _score_acu(acu: dict, prompt_tokens: set[str], *, now=None) -> float:
    # Ranking: relevance (token overlap) gates; AUTHORITY dominates (gated by
    # Kind/Truth); reinforcement + recency break ties within a level. AU1
    # (stored-only: L1 stubs, -inf falsehoods) is NOT recall-eligible. `veracity`
    # is dead and never consulted.
    # Lever 3 — exclude the Origin-0 identity SEED from the task-recall pool. The 12
    # locked seed claims are AU4 and overlap nearly every prompt via stopwords, so
    # with authority-first ranking they filled all 5 slots and buried every AU3/AU2
    # belief. Identity reaches the model via the {identity_block} anchor, not recall.
    from core.identity_acus import SOURCE as _SEED_SRC, LOCK_REASON as _SEED_LOCK
    if (str(acu.get("source", "") or "") == _SEED_SRC
            or str(acu.get("lock_reason", "") or "") == _SEED_LOCK):
        return 0.0
    # SEAL: pure self-reinforcement (kind=self AND provenance=self) is excluded from
    # the WORLD-ANSWER recall lane — and therefore from grounded-citation grounds,
    # since recall_handles only registers rows returned here. Such claims are identity
    # memory; they reach the model via the identity projection channel, not as a
    # recalled belief that could ground a world answer. user/world-sourced self-facts
    # (kind=self, provenance in {user,world}) STAY recallable (rendered PROVISIONAL).
    if (str(acu.get("kind", "") or "").strip().lower() == "self"
            and str(acu.get("provenance", "") or "").strip().lower() == "self"):
        return 0.0
    canonical = str(acu.get("canonical", "")).lower()
    acu_tokens = _tokenize(canonical)
    if not acu_tokens:
        return 0.0
    # Lever 1 — relevance must include a CONTENT token. Stopword-only overlap
    # ("on"/"is"/"a") is not relevance; counting it made every >AU1 claim "relevant"
    # to every prompt. Reuse the curated identity-alignment stopword set.
    from core.identity_alignment import _STOPWORDS
    meaningful = (prompt_tokens & acu_tokens) - _STOPWORDS
    if not meaningful:
        return 0.0
    overlap = len(meaningful)
    from core.acatalepsy.authority import compute_authority, AU_STORED
    authority = compute_authority(acu)
    if authority <= AU_STORED:
        return 0.0
    from core.acatalepsy import decay
    if decay.decay_enabled():
        # Decay folds time-recency INTO the reinforcement weight, replacing both
        # the raw-reinforcement term and the binary 7-day recency bonus with one
        # continuous, monotone term — no cliff, no saturate-forever (when-plane
        # decay primitive). Locked/AU4 and user/world facts resist per decay.py.
        weight = min(decay.effective_reinforcement(acu, now=now), 10.0) / 10.0
        # Authority FIRST: AU4 > AU3 > AU2 regardless of token overlap, so a locked
        # belief outranks a high-overlap L1 stub. Overlap + decay break ties WITHIN
        # a tier. (Was overlap-dominant — relevance beat authority, defeating deference.)
        return float(authority) * 1000.0 + overlap + weight
    reinforcement = float(acu.get("reinforcement", 1) or 1)
    return float(authority) * 1000.0 + overlap + min(reinforcement, 10.0) / 10.0 + _recency_bonus(acu)


def retrieve_relevant_acus(
    user_prompt: str,
    *,
    limit: int = 5,
    min_score: float = 1.0,
    max_candidates: int = 200,
) -> list[dict[str, Any]]:
    """Return up to *limit* ACUs relevant to *user_prompt*, scored by
    token overlap + veracity weight. Empty list when flag is off, prompt
    is too short, or no ACUs score above *min_score*.
    """
    if not _flag_enabled():
        return []
    prompt = str(user_prompt or "").strip()
    if len(prompt) < _MIN_PROMPT_CHARS:
        return []

    prompt_tokens = _tokenize(prompt)
    if not prompt_tokens:
        return []

    try:
        from core.acu_store import ACUStore
        store = ACUStore()
        try:
            rows = store.retrieve(limit=max_candidates)
        finally:
            close_fn = getattr(store, "close", None)
            if callable(close_fn):
                close_fn()
    except Exception:
        return []

    # One clock for the whole ranking pass so every candidate decays against the
    # same instant (and the TurnClock can thread a per-turn clock here later).
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    scored: list[tuple[float, dict]] = []
    for row in rows:
        s = _score_acu(row, prompt_tokens, now=now)
        if s >= min_score:
            scored.append((s, row))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [row for _, row in scored[:limit]]


def _deference_label(authority: int) -> str:
    from core.acatalepsy.authority import AU_BEHAVIOR, AU_LOCKED
    if authority >= AU_LOCKED:
        return "LOCKED"
    if authority >= AU_BEHAVIOR:
        return "VERIFIED"
    return "PROVISIONAL"


def format_recall_block(acus: list[dict]) -> str:
    """Format retrieved ACUs as a GRADED [RECALLED] block (the deference seam).

    ESTABLISHED beliefs (AU4 locked / AU3 truth-confirmed-fresh) carry an override
    contract; ADVISORY beliefs (AU2) stay background. The model is told to DEFER to
    established claims over its in-context reconstruction unless it has direct
    contradicting evidence THIS turn — and the "unless" keeps it from over-deferring.
    Label is derived from Authority (not the dead `veracity`), so confirmed facts
    actually render [VERIFIED].
    """
    if not acus:
        return ""
    from core.acatalepsy.authority import compute_authority, AU_BEHAVIOR

    graded = [(compute_authority(a), a) for a in acus]
    established = [(au, a) for au, a in graded if au >= AU_BEHAVIOR]
    advisory = [(au, a) for au, a in graded if au < AU_BEHAVIOR]

    header = (
        "[RECALLED] — your own prior beliefs. ESTABLISHED claims (locked/verified) "
        "OVERRIDE your in-context guess unless you have direct contradicting evidence "
        "THIS turn. ADVISORY claims are background — verify before acting."
    )
    lines = [header]
    total = len(header)
    idx = 0
    for group_name, group in (("established", established), ("advisory", advisory)):
        if not group:
            continue
        sub = f"  {group_name}:"
        if total + len(sub) + 1 <= _MAX_BLOCK_CHARS:
            lines.append(sub)
            total += len(sub) + 1
        for au, acu in group:
            idx += 1
            canonical = str(acu.get("canonical", "")).strip()
            line = f"    {idx}. [{_deference_label(au)}] {canonical}"
            if total + len(line) + 1 > _MAX_BLOCK_CHARS:
                break
            lines.append(line)
            total += len(line) + 1
    lines.append("[/RECALLED]")
    return "\n".join(lines)


def recall_hit_summary(acus: list[dict]) -> dict:
    """Summarize a recall hit for the deference hit-log: count, authority
    histogram, and the established (override-tier) canonicals."""
    from core.acatalepsy.authority import compute_authority, AU_BEHAVIOR
    by_authority: dict[int, int] = {}
    established: list[str] = []
    for a in acus:
        au = int(compute_authority(a))
        by_authority[au] = by_authority.get(au, 0) + 1
        if au >= AU_BEHAVIOR:
            established.append(str(a.get("canonical", "")).strip())
    return {"n": len(acus), "by_authority": by_authority, "established": established}


def _write_recall_hit(acus: list[dict]) -> None:
    """Best-effort append to the deference hit-log so we can SEE what was recalled
    at what authority — you cannot verify deference fires without looking."""
    try:
        import json
        from datetime import datetime, timezone
        from core.paths import LOG_DIR
        rec = {"ts": datetime.now(timezone.utc).isoformat(), **recall_hit_summary(acus)}
        path = LOG_DIR / "recall_hits.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def contribute_section(messages: list[dict], config: dict):
    """Ephemeral coalescer section contributor."""
    from core.ephemeral_coalescer import SectionResult
    if not _flag_enabled():
        return None

    prompt = ""
    for msg in reversed(messages):
        if isinstance(msg, dict) and msg.get("role") == "user" and not msg.get("ephemeral"):
            prompt = str(msg.get("content", ""))
            break
    if not prompt:
        return None

    acus = retrieve_relevant_acus(prompt)
    if not acus:
        return None

    block = format_recall_block(acus)
    if not block:
        return None
    _write_recall_hit(acus)
    return SectionResult(name="acu_recall", text=block)
