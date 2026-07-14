"""Fault Telemetry — the [SELF-CHECK] contributor (Self-Check Loop, wire 2).

Closes Monolith's keystone open loop: every turn the verifier produces a
verdict (pass / warn / hard_fail) and the fault detectors fire, but the model
never reads any of it back. This contributor reads the *last* turn's verdict
(``turn_trace.get_last_verification_result`` — the verdict is already durable
in fault_traces; see that function) and, at the start of the next turn,
injects a compact ``[SELF-CHECK]`` block so the model can see how its previous
answer was judged and self-correct.

Same shape as the other standalone-compatible coalescer contributors
(``rating_telemetry``):
  * gated on a flag
  * read state (here: turn_trace verdict)
  * project to a string
  * return a SectionResult to the ephemeral_coalescer

Discipline: this is *observation*, not a rule. The block presents the verdict
and finding codes; the model decides what to do. It stays SILENT on a clean
``pass`` turn so it costs nothing when there's nothing to report (~92% of
turns are pass in practice).

Suppressed on CONNECT peer turns (an examiner / trainer LLM) for the same
observer-effect reason as rating_telemetry: a peer must not see Monolith's own
grades and perform to them.

Flag: MONOLITH_FAULT_TELEMETRY_V1 (default OFF — ships dark; flip to "1" to
enable). Reversible: with the flag off this contributor returns None and the
loop is inert.
"""
from __future__ import annotations

import os
from typing import Any

from core import turn_trace as _tt

_FLAG_ENV = "MONOLITH_FAULT_TELEMETRY_V1"
_TAG = "[SELF-CHECK]"
_MAX_FINDINGS_SHOWN = 4
_MAX_MESSAGE_CHARS = 90

# C1.1 — recurring non-verifier faults ("what I keep failing at"). Bounded so the
# block never floods: recurrence only (count >= _RECUR_MIN_COUNT), capped kinds,
# environmental/non-cognitive kinds excluded (a spawn_denied the model can't act on
# would otherwise pin the same line every turn — the stale-sticky shape it'd create).
_RECUR_FETCH = 60          # recent fault rows to scan
_RECUR_MIN_COUNT = 2       # count >= this = recurrence; one-offs excluded
_MAX_RECURRING_KINDS = 3   # cap distinct kinds shown
_ENV_FAULT_KINDS = frozenset({"spawn_denied"})  # environmental, not cognitive-actionable


def _flag_enabled() -> bool:
    # Default OFF: the loop ships dark and is opt-in / reversible.
    raw = str(os.environ.get(_FLAG_ENV, "0")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _is_peer_turn(messages: list[dict]) -> bool:
    """True when this turn arrived on a CONNECT peer channel (examiner / trainer
    LLM), identified by the ``[CHANNEL: connect/...]`` tag the agent-server
    prepends. Self-check is suppressed on these turns so the peer doesn't see
    Monolith's own grades and perform to them (observer effect). Mirrors
    ``rating_telemetry._is_peer_turn``."""
    for i in range(len(messages) - 1, -1, -1):
        msg = messages[i]
        if msg.get("role") == "user" and not msg.get("ephemeral"):
            return "[CHANNEL: connect/" in str(msg.get("content") or "")
    return False


def _trim(text: Any) -> str | None:
    if not text:
        return None
    s = str(text).strip()
    if not s:
        return None
    if len(s) <= _MAX_MESSAGE_CHARS:
        return s
    return s[: _MAX_MESSAGE_CHARS - 1].rstrip() + "…"


def _recent_recurring_faults(fetch: int = _RECUR_FETCH,
                             max_kinds: int = _MAX_RECURRING_KINDS) -> list[tuple[str, int]]:
    """Top recurring NON-verifier, cognitively-actionable fault kinds in the recent
    window: (fault_kind, count), sorted by count desc then recency. Recurrence only
    (count >= _RECUR_MIN_COUNT); one-offs, environmental, and verifier-prefixed kinds
    excluded. Reads fault_response DIRECTLY (never depends on MonoSearch being on).
    Same kind-level recurrence key MonoSearch uses, so the two tiers agree."""
    from core import fault_response
    try:
        rows = fault_response.read_recent(fetch)  # newest-first
    except Exception:
        return []
    counts: dict[str, int] = {}
    recency: dict[str, int] = {}  # lowest index = most recent occurrence
    for idx, r in enumerate(rows):
        kind = (getattr(r, "fault_kind", "") or "").strip()
        if not kind or kind.startswith("verifier") or kind in _ENV_FAULT_KINDS:
            continue
        counts[kind] = counts.get(kind, 0) + 1
        recency.setdefault(kind, idx)
    recurring = [(k, c) for k, c in counts.items() if c >= _RECUR_MIN_COUNT]
    recurring.sort(key=lambda kc: (-kc[1], recency[kc[0]]))
    return recurring[:max_kinds]


def _recurring_line() -> str | None:
    """One bounded line of recurring faults, or None. Points to MonoSearch for bulk."""
    rec = _recent_recurring_faults()
    if not rec:
        return None
    parts = ", ".join(f"{k} ×{c}" for k, c in rec)
    return f"- recurring faults: {parts} (MonoSearch `failing` for the full list)"


def render_self_check_block() -> str | None:
    """Build the ``[SELF-CHECK]`` block: last verdict (when not a clean pass) plus
    C1.1's recurring-fault line (when any kind recurs). Returns None only when there
    is nothing worth showing — no actionable verdict AND no recurrence."""
    verdict = _tt.get_last_verification_result()
    recurring = _recurring_line()

    tier = ""
    findings: Any = []
    if verdict is not None:
        tier = str(verdict.get("verdict") or "").strip()
        findings = verdict.get("findings") or []
    verdict_worth_showing = bool(tier) and not (tier == "pass" and not findings)

    if not verdict_worth_showing and recurring is None:
        return None  # quiet turn — nothing to report

    lines = [_TAG]
    if verdict_worth_showing:
        lines.append(f"- last turn verdict: {tier}")
        shown = findings[:_MAX_FINDINGS_SHOWN] if isinstance(findings, list) else []
        for f in shown:
            if not isinstance(f, dict):
                continue
            code = str(f.get("code") or "finding").strip()
            msg = _trim(f.get("message"))
            line = f"- {code}"
            if msg:
                line += f": {msg}"
            lines.append(line)
        extra = len(findings) - len(shown) if isinstance(findings, list) else 0
        if extra > 0:
            lines.append(f"- (+{extra} more)")
    if recurring is not None:
        lines.append(recurring)
    return "\n".join(lines)


def contribute_section(messages: list[dict], config: dict):
    """Section-contributor for the ephemeral_coalescer.

    Returns None when the flag is off, the turn is a CONNECT peer turn, or there
    is nothing to report (clean pass / no verdict yet)."""
    from core.ephemeral_coalescer import SectionResult
    if not _flag_enabled() or _is_peer_turn(messages):
        return None
    block = render_self_check_block()
    if block is None:
        return None
    return SectionResult(name="self_check", text=block)
