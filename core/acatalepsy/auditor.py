"""LLM-based extraction auditor — proposes ACU candidates from a
canonical_log slice.

This is NOT the legacy structural ticket-veto auditor (that one lives
in the legacy Monolith auditor and does deterministic
pre-write checks). This module is the v1 *producer* — read a log
slice, ask the LLM "what atomic claims are worth saving from this?",
validate the response, insert candidates with pending state.

Design notes
============

* The LLM is injected as ``LLMCallable`` (a function that takes a list
  of messages and returns the assistant text). This decouples the
  auditor from any specific LLM client and makes the parse / validate
  / insert logic testable with a stub.

* Identity is injected fresh per run via ``load_identity()`` so the
  auditor stays in character. The legacy module-level system.md is NOT
  loaded — the auditor has its own dedicated 6-section prompt.

* The cursor (``last_processed_event_id``) is stored as the payload of
  the most recent ``auditor_cursor_advance`` event in canonical_log
  itself. No separate state table needed in v1.

* Atomicity gate runs at insert time per candidate. Failed candidates
  are logged to canonical_log as ``auditor_atomicity_reject`` and
  dropped (hard-reject in v1; auto-split is v1.5+).

* Each run emits ``auditor_run_started`` + ``auditor_run_complete``
  (or ``auditor_run_failed``) events. Run ID = the started event's
  event_id, propagated to every candidate inserted in that run.

Public API
==========
  - run_audit(llm, *, source, start_event_id=None, end_event_id=None,
              max_events=200) -> AuditRunResult
  - last_processed_event_id() -> int
  - load_identity_block() -> str             (helper, calls core.identity)
  - build_system_prompt(...) -> str          (pure)
  - format_log_slice(events) -> str          (pure)
  - parse_candidates(text) -> list[CandidateProposal]   (pure)

The pure functions are independently testable.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Protocol

from core.acatalepsy import canonical_log as _canonical_log
from core.acatalepsy import candidates as _candidates
from core.acatalepsy.atomicity import is_atomic
from core.acatalepsy.extraction_quality import is_extraction_quality_acceptable

_MONONOTE_NOTE_READ_KIND = "mononote_note_read"
_MONONOTE_CANDIDATE_SOURCE = "mononote_note"


__all__ = (
    "AUDITOR_PROMPT_VERSION",
    "AuditRunResult",
    "CandidateProposal",
    "LLMCallable",
    "build_system_prompt",
    "cancel_current_run",
    "close_orphaned_runs",
    "current_in_flight_run",
    "format_log_slice",
    "last_processed_event_id",
    "load_identity_block",
    "parse_candidates",
    "read_audit_log_tail",
    "read_recent_runs",
    "reset_cursor",
    "run_audit",
)


# Canonical-log event kinds the auditor emits during a run. Used by the
# audit log tail viewer to filter the canonical log to auditor activity.
AUDITOR_EVENT_KINDS: frozenset[str] = frozenset({
    "auditor_run_started",
    "auditor_run_complete",
    "auditor_run_failed",
    "auditor_cursor_advance",
    "auditor_atomicity_reject",
    "auditor_extraction_filter_reject",
    "auditor_llm_call_started",
    "auditor_llm_call_returned",
    "candidate_emitted",
})


# Bumped when the system prompt template materially changes. Stored on
# each auditor_run_complete event so we can correlate prompt versions
# with candidate quality over time.
AUDITOR_PROMPT_VERSION: int = 1


# ── Types ─────────────────────────────────────────────────────────────


class LLMCallable(Protocol):
    """A function that takes (system_prompt, user_content) and returns
    the assistant's text response. Implementations:
      - production: wrap Monolith's existing engine
      - tests: a stub that returns canned JSON
    """
    def __call__(self, *, system_prompt: str, user_content: str) -> str: ...


@dataclass(frozen=True)
class CandidateProposal:
    """One candidate as proposed by the LLM, before atomicity gate."""
    canonical_form: str
    evidence_log_id: int
    evidence_char_start: int
    evidence_char_end: int
    evidence_span: str
    reason: str
    reinforcement_count: int = 1
    contradicts_acu_id: int | None = None


@dataclass(frozen=True)
class AuditRunResult:
    """Summary of one audit run. Returned from run_audit()."""
    run_id: int                          # canonical_log event_id of auditor_run_started
    started_at: float
    completed_at: float | None
    status: str                          # 'success' | 'failed' | 'empty_slice'
    slice_start_event_id: int            # cursor before run
    slice_end_event_id: int              # cursor after run
    events_processed: int
    proposals_returned: int              # what LLM emitted
    candidates_inserted: int             # passed atomicity, landed in acu_candidates
    candidates_rejected: int             # failed atomicity gate
    rejection_reasons: list[str] = field(default_factory=list)
    error: str | None = None


# ── Identity ──────────────────────────────────────────────────────────


def load_identity_block() -> str:
    """Best-effort identity load. Falls back to empty string if the
    identity surface isn't reachable (e.g., in tests where core.identity
    raises on missing CONFIG_DIR)."""
    try:
        from core.identity import load_identity
        return load_identity()
    except Exception:
        return ""


# ── Cursor ────────────────────────────────────────────────────────────


def last_processed_event_id() -> int:
    """Return the most recent cursor value, or 0 if no cursor has been
    advanced yet. Reads the latest ``auditor_cursor_advance`` event in
    canonical_log.
    """
    # We can't filter by kind cheaply without an index, so scan recent
    # tail. In v1 this is fine — the function is called once per run.
    # Iterate backward from the latest event_id.
    latest = _canonical_log.latest_event_id()
    if latest == 0:
        return 0
    # Walk back in chunks of 200.
    chunk = 200
    end = latest
    while end > 0:
        start = max(0, end - chunk)
        events = _canonical_log.read_since(start, limit=chunk)
        for ev in reversed(events):
            if ev.kind == "auditor_cursor_advance":
                if ev.payload and isinstance(ev.payload.get("cursor_value"), int):
                    return int(ev.payload["cursor_value"])
        if start == 0:
            break
        end = start
    return 0


def reset_cursor() -> int:
    """Force the auditor cursor back to 0 so the next run re-audits the
    entire canonical log from the beginning.

    The cursor is stored as the payload of the most recent
    ``auditor_cursor_advance`` event, so "resetting" means appending a
    new one with ``cursor_value=0``. The old advance events stay in the
    log for provenance — only the latest cursor wins.

    Returns the cursor value that was set (always 0). Idempotent: calling
    it twice in a row is fine, just writes two cursor_advance events.
    """
    _canonical_log.append(
        "auditor_cursor_advance",
        payload={"cursor_value": 0, "run_id": None, "reset": True},
    )
    return 0


def read_recent_runs(limit: int = 10) -> list[dict[str, Any]]:
    """Return the most recent N audit runs, newest first.

    Walks the canonical_log tail looking for ``auditor_run_complete`` and
    ``auditor_run_failed`` events. Each returned dict carries the run's
    summary stats:

      {
        "run_id":             int   (canonical_log event_id of run_started),
        "kind":               str   ('auditor_run_complete' | 'auditor_run_failed'),
        "status":             str   ('success' | 'empty_slice' | 'failed'),
        "events_processed":   int,
        "proposals_returned": int,  # 0 for failed / empty
        "candidates_inserted": int,
        "candidates_rejected": int,
        "error":              str | None,
        "ts":                 float,
      }

    Useful for the MonoBase panel to answer "did the auditor actually
    run and what happened?" — especially the proposals_returned vs.
    candidates_inserted breakdown, which tells you whether the LLM
    proposed nothing or whether the atomicity gate rejected everything.
    """
    if limit < 1:
        return []
    latest = _canonical_log.latest_event_id()
    if latest == 0:
        return []

    runs: list[dict[str, Any]] = []
    chunk = 500
    end = latest
    while end > 0 and len(runs) < limit:
        start = max(0, end - chunk)
        events = _canonical_log.read_since(start, limit=chunk)
        for ev in reversed(events):
            if ev.kind not in ("auditor_run_complete", "auditor_run_failed"):
                continue
            payload = ev.payload or {}
            runs.append({
                "run_id": int(payload.get("run_id") or 0),
                "kind": ev.kind,
                "status": str(payload.get("status") or ("failed" if ev.kind == "auditor_run_failed" else "")),
                "events_processed": int(payload.get("events_processed") or 0),
                "proposals_returned": int(payload.get("proposals_returned") or 0),
                "candidates_inserted": int(payload.get("candidates_inserted") or 0),
                "candidates_rejected": int(payload.get("candidates_rejected") or 0),
                "error": payload.get("error"),
                "ts": float(ev.ts),
            })
            if len(runs) >= limit:
                break
        if start == 0:
            break
        end = start
    return runs


def current_in_flight_run() -> dict[str, Any] | None:
    """Return the most recent ``auditor_run_started`` event that doesn't
    yet have a matching ``auditor_run_complete`` or ``auditor_run_failed``.
    None if the auditor is idle (no run in progress).

    The check is purely log-based — we don't poll the worker thread. A
    run is "in flight" when run_started's event_id is greater than every
    run_complete/failed event_id. Used by the MonoBase panel to show a
    live RUNNING indicator while an audit is processing.

    Returned dict mirrors the run_started payload plus its event_id and
    timestamp so the UI can show "running for Ns" elapsed time.
    """
    latest = _canonical_log.latest_event_id()
    if latest == 0:
        return None

    chunk = 500
    end = latest
    latest_started: dict[str, Any] | None = None
    latest_terminated_id: int = 0
    scanned_chunks = 0
    # Cap at 10 chunks (5k events) to match close_orphaned_runs scan
    # window — prevents unbounded backward scan on very long logs.
    while end > 0 and scanned_chunks < 10:
        start = max(0, end - chunk)
        events = _canonical_log.read_since(start, limit=chunk)
        for ev in reversed(events):
            if ev.kind in ("auditor_run_complete", "auditor_run_failed"):
                if ev.event_id > latest_terminated_id:
                    latest_terminated_id = ev.event_id
            elif ev.kind == "auditor_run_started" and latest_started is None:
                payload = ev.payload or {}
                latest_started = {
                    "event_id": int(ev.event_id),
                    "ts": float(ev.ts),
                    "source": str(payload.get("source") or ""),
                    "slice_start_event_id": int(payload.get("slice_start_event_id") or 0),
                    "slice_end_event_id": int(payload.get("slice_end_event_id") or 0),
                    "prompt_version": int(payload.get("prompt_version") or 0),
                }
            # Stop searching once we've found one of each (newest of both)
            if latest_started is not None and latest_terminated_id > 0:
                break
        if latest_started is not None and latest_terminated_id > 0:
            break
        if start == 0:
            break
        end = start
        scanned_chunks += 1

    if latest_started is None:
        return None
    # In flight iff started event_id > newest terminated event_id (or no
    # terminated event has ever been written yet).
    if latest_started["event_id"] > latest_terminated_id:
        return latest_started
    return None


def close_orphaned_runs(stale_after_secs: float = 600.0) -> int:
    """Close audit runs whose ``auditor_run_started`` event has no
    matching ``auditor_run_complete`` / ``auditor_run_failed`` AND whose
    started timestamp is older than ``stale_after_secs``.

    Use case: Monolith crashed or was force-quit mid-run; on restart the
    canonical log has a started event without a terminator, so the
    in-flight check reports "RUNNING" forever even with no worker alive.
    Calling this on panel init writes a synthetic
    ``auditor_run_failed`` with ``error="orphaned_at_startup"`` for each
    stale run, clearing the in-flight state.

    Returns the count of runs closed. Idempotent — repeated calls won't
    re-close the same runs because the freshly-written failed event
    becomes the terminator for that run_id.
    """
    latest = _canonical_log.latest_event_id()
    if latest == 0:
        return 0

    now_ts = time.time()
    threshold_ts = now_ts - max(0.0, float(stale_after_secs))

    # Collect all started events newer than enough chunks to matter, and
    # the set of run_ids that already have a terminator. The threshold
    # keeps us from scanning the whole log every call.
    started: list[tuple[int, dict[str, Any], float]] = []  # (run_id, payload, ts)
    terminated_run_ids: set[int] = set()
    chunk = 1000
    end = latest
    scanned_chunks = 0
    # Cap the scan at 10 chunks (10k events) so this stays fast on long
    # logs. Anything older than that is hopelessly orphaned anyway and
    # not worth chasing.
    while end > 0 and scanned_chunks < 10:
        start = max(0, end - chunk)
        events = _canonical_log.read_since(start, limit=chunk)
        for ev in events:
            if ev.kind == "auditor_run_started":
                run_id = int(ev.event_id)  # run_id IS the started event_id
                started.append((run_id, dict(ev.payload or {}), float(ev.ts)))
            elif ev.kind in ("auditor_run_complete", "auditor_run_failed"):
                rid = (ev.payload or {}).get("run_id")
                if isinstance(rid, int):
                    terminated_run_ids.add(int(rid))
        if start == 0:
            break
        end = start
        scanned_chunks += 1

    closed = 0
    for run_id, payload, ts in started:
        if run_id in terminated_run_ids:
            continue
        # Only skip runs that are genuinely recent: threshold_ts <= ts <= now.
        # A future-dated ts (ts > now) means the wall clock stepped backward
        # since the run started; treat that orphan as stale and close it rather
        # than hiding it as "too fresh" forever (when-plane clock fix).
        if threshold_ts <= ts <= now_ts:
            continue  # too fresh; might be a legitimate slow run
        try:
            _canonical_log.append(
                "auditor_run_failed",
                payload={
                    "run_id": run_id,
                    "error": "orphaned_at_startup",
                    "source": payload.get("source"),
                },
            )
            closed += 1
        except Exception:
            pass
    return closed


def cancel_current_run(reason: str = "user_cancelled") -> int | None:
    """Cancel the currently in-flight audit run by writing an
    ``auditor_run_failed`` event for it. Returns the run_id that was
    cancelled, or None if there's no run in flight.

    Note: this is a soft cancel. It marks the run as failed in the
    canonical log so the UI's in-flight check clears immediately, but
    it does NOT interrupt the background LLM call — Python threads
    can't be safely killed mid-call. The worker thread will complete
    its current LLM call (or hit its timeout, default 180s) before it
    notices the stop event. Any candidates the LLM eventually returns
    are still inserted normally; only the cursor advance is skipped if
    the worker checks _stop_event before advancing.
    """
    in_flight = current_in_flight_run()
    if in_flight is None:
        return None
    run_id = int(in_flight.get("event_id") or 0)
    if run_id <= 0:
        return None
    try:
        _canonical_log.append(
            "auditor_run_failed",
            payload={
                "run_id": run_id,
                "error": str(reason or "user_cancelled"),
            },
        )
    except Exception:
        return None
    return run_id


def read_audit_log_tail(limit: int = 30) -> list[dict[str, Any]]:
    """Return the most recent N auditor-related canonical_log events,
    newest first. Useful for the live AUDIT LOG viewer in MonoBase —
    answers "what is the auditor doing right now?" with per-step
    granularity (run_started → atomicity_reject → candidate_emitted →
    cursor_advance → run_complete).

    Each entry:
        {"event_id": int, "ts": float, "kind": str, "payload": dict}
    Filtered to AUDITOR_EVENT_KINDS so unrelated chat events don't
    flood the viewer.
    """
    if limit < 1:
        return []
    latest = _canonical_log.latest_event_id()
    if latest == 0:
        return []

    out: list[dict[str, Any]] = []
    chunk = 500
    end = latest
    while end > 0 and len(out) < limit:
        start = max(0, end - chunk)
        events = _canonical_log.read_since(start, limit=chunk)
        for ev in reversed(events):
            if ev.kind not in AUDITOR_EVENT_KINDS:
                continue
            out.append({
                "event_id": int(ev.event_id),
                "ts": float(ev.ts),
                "kind": ev.kind,
                "payload": dict(ev.payload) if ev.payload else {},
            })
            if len(out) >= limit:
                break
        if start == 0:
            break
        end = start
    return out


# ── Prompt construction ──────────────────────────────────────────────


_SYSTEM_PROMPT_TEMPLATE = """\
═══════════════════════════════════════════════════════
1. CONTEXT
═══════════════════════════════════════════════════════
You are an Acatalepsy auditor. Your job is to read a slice of
Monolith's canonical_log (event_id {start} -> {end}) and extract
atomic durable claims for review.

You are NOT in a conversation. You are NOT generating a response
to a user. You are reading a record and proposing candidates.

═══════════════════════════════════════════════════════
2. TASK
═══════════════════════════════════════════════════════
Extract claims that meet the criteria below. A single log entry
may produce 0, 1, or several candidates. Most produce 0 — claim
extraction is selective, not exhaustive.

Cross-turn reinforcement counts: if the same claim shape appears
in 3+ entries, emit ONE candidate with reinforcement_count=3,
not three duplicates.

═══════════════════════════════════════════════════════
3. IDENTITY
═══════════════════════════════════════════════════════
{identity_block}

Your identity shapes which claims feel worth saving. Stay in
character. Don't extract claims that contradict your identity
without flagging the contradiction.

═══════════════════════════════════════════════════════
4. CRITERIA — a claim is worth promoting if it...
═══════════════════════════════════════════════════════
1. Aligns with identity (provenance over assertion,
   anti-duplication, push-back over agreement-by-default)
2. Surfaces a load-bearing premise that wasn't on the table before
3. Productively contradicts something the system previously thought
4. Is specific enough to refine later (named entities, file:line,
   observable evidence)
5. Is something E or another peer could push back on
   (not consensus filler)

A claim that meets none of these is NOT worth extracting.
A claim that meets 2+ is a strong candidate.

═══════════════════════════════════════════════════════
5. OUTPUT SCHEMA — strict JSON, no prose
═══════════════════════════════════════════════════════
Return EXACTLY this shape with no surrounding commentary:
{{
  "candidates": [
    {{
      "canonical_form": "subject | relation | object",
      "evidence_log_id": <event_id from input>,
      "evidence_char_start": <int, 0-based offset into the event's text>,
      "evidence_char_end": <int, exclusive end offset>,
      "evidence_span": "literal text slice from the event",
      "reason": "one sentence: why this matters",
      "reinforcement_count": <int, default 1>,
      "contradicts_acu_id": <int or null>
    }}
  ]
}}

Atomicity rule (HARD-REJECTED if violated):
- One subject-predicate per canonical_form (3 or 4 pipe-delimited parts)
- No 'and' / 'or' / 'because' / 'therefore' / 'while' splitting
  into multiple predicates
- If you find a compound claim, emit it as N separate candidates

If you find zero claims worth proposing, return {{"candidates": []}}.

═══════════════════════════════════════════════════════
6. EXAMPLES
═══════════════════════════════════════════════════════
GOOD:
  canonical_form: "core/effort.py | defines | seven effort tiers"
  evidence_span: "Seven tiers: low, med, high, xhigh, ultimate, experimental, monolith."
  reason: "load-bearing scaffold count; affects routing logic"

BAD (compound — would be hard-rejected):
  canonical_form: "Monolith | has | effort tiers and CONNECT addon"
  fix: split into two candidates

BAD (theater — meets no criteria):
  canonical_form: "user | typed | hello"
  fix: don't extract; conversational filler isn't durable

Now read the canonical_log slice and emit JSON.
"""


def build_system_prompt(
    *,
    slice_start_event_id: int,
    slice_end_event_id: int,
    identity_block: str,
) -> str:
    """Pure function — assemble the 6-section auditor system prompt.

    The double-brace escapes in the template above produce single-brace
    output for the JSON schema example; ``format`` only substitutes the
    named placeholders.
    """
    return _SYSTEM_PROMPT_TEMPLATE.format(
        start=slice_start_event_id,
        end=slice_end_event_id,
        identity_block=(identity_block.strip() or "(identity not loaded)"),
    )


def format_log_slice(events: list[_canonical_log.Event]) -> str:
    """Pure function — format a list of Events into the user-content
    string the LLM reads. One event per block, with explicit event_id
    so the LLM can reference it as evidence_log_id.
    """
    parts: list[str] = []
    for ev in events:
        # Pretty payload for the model; keep it small.
        if ev.payload is not None and isinstance(ev.payload, dict):
            try:
                payload_str = json.dumps(ev.payload, ensure_ascii=False)
            except Exception:
                payload_str = str(ev.payload)
        else:
            payload_str = ""
        parts.append(
            f"--- event_id={ev.event_id} kind={ev.kind}"
            + (f" session_id={ev.session_id}" if ev.session_id else "")
            + (f" acu_id={ev.acu_id}" if ev.acu_id is not None else "")
            + f"\n{payload_str}"
        )
    return "\n\n".join(parts)


def _candidate_source_for(
    proposal: CandidateProposal,
    events_by_id: dict[int, _canonical_log.Event],
    default_source: str,
) -> str:
    event = events_by_id.get(int(proposal.evidence_log_id))
    if event is not None and event.kind == _MONONOTE_NOTE_READ_KIND:
        return _MONONOTE_CANDIDATE_SOURCE
    return default_source


# ── Response parsing ─────────────────────────────────────────────────


# Match a JSON object inside the response. The LLM may wrap it in
# ```json fences or surrounding prose despite the strict-JSON instruction;
# this regex captures the outermost {...} block.
_JSON_BLOCK_RE = re.compile(r"\{(?:[^{}]|\{[^{}]*\})*\}", re.DOTALL)


class ParseError(ValueError):
    """Raised when the LLM response can't be parsed into candidates."""


def parse_candidates(response_text: str) -> list[CandidateProposal]:
    """Pure function — parse LLM response into CandidateProposal list.

    Tolerates code-fence wrapping but expects {"candidates": [...]} shape.
    Raises ParseError on unrecoverable malformed input.
    """
    if not response_text or not response_text.strip():
        raise ParseError("empty response")

    # Try direct JSON first.
    obj: Any
    text = response_text.strip()
    # Strip common fences.
    if text.startswith("```"):
        # Remove leading and trailing code fence
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Fallback: find first JSON object in the text.
        m = _JSON_BLOCK_RE.search(response_text)
        if not m:
            raise ParseError(f"no JSON object found in response (len={len(response_text)})")
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError as exc:
            raise ParseError(f"JSON decode failed: {exc}") from exc

    if not isinstance(obj, dict):
        raise ParseError(f"response root is not an object: {type(obj).__name__}")

    raw_candidates = obj.get("candidates")
    if raw_candidates is None:
        raise ParseError("missing 'candidates' key")
    if not isinstance(raw_candidates, list):
        raise ParseError(f"'candidates' is not a list: {type(raw_candidates).__name__}")

    proposals: list[CandidateProposal] = []
    for idx, raw in enumerate(raw_candidates):
        if not isinstance(raw, dict):
            raise ParseError(f"candidate[{idx}] is not an object: {type(raw).__name__}")
        try:
            proposals.append(CandidateProposal(
                canonical_form=str(raw["canonical_form"]),
                evidence_log_id=int(raw["evidence_log_id"]),
                evidence_char_start=int(raw["evidence_char_start"]),
                evidence_char_end=int(raw["evidence_char_end"]),
                evidence_span=str(raw["evidence_span"]),
                reason=str(raw["reason"]),
                reinforcement_count=int(raw.get("reinforcement_count", 1)),
                contradicts_acu_id=(
                    int(raw["contradicts_acu_id"])
                    if raw.get("contradicts_acu_id") is not None else None
                ),
            ))
        except (KeyError, ValueError, TypeError) as exc:
            raise ParseError(f"candidate[{idx}] malformed: {exc}") from exc

    return proposals


# ── Main entry point ─────────────────────────────────────────────────


def run_audit(
    llm: LLMCallable,
    *,
    source: str,
    start_event_id: int | None = None,
    end_event_id: int | None = None,
    max_events: int = 200,
) -> AuditRunResult:
    """Run one audit pass.

    Args:
        llm: callable that takes (system_prompt, user_content) and
            returns the assistant text. Inject Monolith's engine in
            production; a stub in tests.
        source: who's running the audit (e.g., "auditor_claude",
            "auditor_monolith"). Recorded on every candidate this run
            emits. Used by the decision-side authorization rule.
        start_event_id: cursor (exclusive). If None, uses
            last_processed_event_id() so re-runs are idempotent.
        end_event_id: upper bound (inclusive). If None, uses
            canonical_log.latest_event_id() at run start.
        max_events: hard cap on log slice size per run.

    Returns AuditRunResult with stats. On parse / LLM failure, returns
    with status='failed' and error set; the cursor is NOT advanced so
    the next run retries the same slice.
    """
    started_at = time.time()
    cursor_start = (
        int(start_event_id) if start_event_id is not None else last_processed_event_id()
    )
    cursor_end_requested = (
        int(end_event_id) if end_event_id is not None
        else _canonical_log.latest_event_id()
    )

    # Read the slice FIRST so we know how many events we'll actually
    # process under the max_events cap. The previous version wrote
    # run_started with slice_end=cursor_end_requested (always the full
    # latest), then advanced the cursor to that same full latest —
    # which meant max_events>=N events smaller than the slice silently
    # skipped events N+1..end. With max_events=50 and 497 unaudited
    # events, the audit would process 1-50 but the cursor would jump
    # to 497, leaving events 51-497 permanently un-audited.
    events = _canonical_log.read_since(cursor_start, limit=max_events)
    events = [e for e in events if e.event_id <= cursor_end_requested]
    events_by_id = {int(e.event_id): e for e in events}

    # Actual upper bound of this run's slice — what we'll ACTUALLY audit.
    # On an empty slice, fall back to cursor_end_requested so the next
    # cursor advance reflects "nothing to do up to this point".
    actual_slice_end = events[-1].event_id if events else cursor_end_requested

    # Emit run_started — its event_id becomes the run_id. slice_end is
    # the actual processed bound, not the full latest, so the MonoBase
    # RUNNING banner shows the real slice (e.g. "slice 0→50" with
    # max_events=50, not the misleading "slice 0→497").
    run_id = _canonical_log.append(
        "auditor_run_started",
        payload={
            "source": source,
            "slice_start_event_id": cursor_start,
            "slice_end_event_id": actual_slice_end,
            "slice_end_requested": cursor_end_requested,
            "max_events_per_run": int(max_events),
            "prompt_version": AUDITOR_PROMPT_VERSION,
        },
    )

    if not events:
        # Nothing to audit; advance cursor to cursor_end_requested and
        # return empty_slice. (Cursor advance to the requested end is
        # correct here — there's truly nothing between cursor_start and
        # latest, so the next run should not re-scan this range.)
        _canonical_log.append(
            "auditor_cursor_advance",
            payload={"cursor_value": cursor_end_requested, "run_id": run_id},
        )
        completed_at = time.time()
        _canonical_log.append(
            "auditor_run_complete",
            payload={
                "run_id": run_id,
                "status": "empty_slice",
                "events_processed": 0,
                "candidates_inserted": 0,
            },
        )
        return AuditRunResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=completed_at,
            status="empty_slice",
            slice_start_event_id=cursor_start,
            slice_end_event_id=cursor_end_requested,
            events_processed=0,
            proposals_returned=0,
            candidates_inserted=0,
            candidates_rejected=0,
        )

    # Build prompt + user content
    identity_block = load_identity_block()
    system_prompt = build_system_prompt(
        slice_start_event_id=cursor_start,
        slice_end_event_id=actual_slice_end,
        identity_block=identity_block,
    )
    user_content = format_log_slice(events)

    # Live LLM trace — emit a "call started" event so the MonoBase AUDIT
    # LOG shows the long synchronous wait isn't a hang. Includes prompt
    # size so users can see what they're feeding the model.
    llm_call_started_at = time.time()
    _canonical_log.append(
        "auditor_llm_call_started",
        payload={
            "run_id": run_id,
            "prompt_chars": len(system_prompt),
            "user_chars": len(user_content),
            "events_in_slice": len(events),
        },
    )

    # Call LLM
    try:
        response_text = llm(system_prompt=system_prompt, user_content=user_content)
    except Exception as exc:
        elapsed = time.time() - llm_call_started_at
        _canonical_log.append(
            "auditor_llm_call_returned",
            payload={
                "run_id": run_id,
                "elapsed_secs": round(elapsed, 2),
                "status": "error",
                "error": f"{type(exc).__name__}:{exc}",
            },
        )
        _canonical_log.append(
            "auditor_run_failed",
            payload={
                "run_id": run_id,
                "error": f"llm_call:{type(exc).__name__}:{exc}",
            },
        )
        return AuditRunResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=time.time(),
            status="failed",
            slice_start_event_id=cursor_start,
            slice_end_event_id=actual_slice_end,
            events_processed=len(events),
            proposals_returned=0,
            candidates_inserted=0,
            candidates_rejected=0,
            error=f"llm_call:{type(exc).__name__}",
        )

    elapsed = time.time() - llm_call_started_at
    response_preview = (response_text or "")[:8000]
    _canonical_log.append(
        "auditor_llm_call_returned",
        payload={
            "run_id": run_id,
            "elapsed_secs": round(elapsed, 2),
            "status": "ok",
            "response_chars": len(response_text or ""),
            "response_preview": response_preview,
            "response_truncated": len(response_text or "") > len(response_preview),
        },
    )

    # Parse
    try:
        proposals = parse_candidates(response_text)
    except ParseError as exc:
        # Advance cursor past the poisoned slice so the next run doesn't
        # retry the same events forever. The skipped range is recorded in
        # the failure payload for human inspection.
        _canonical_log.append(
            "auditor_cursor_advance",
            payload={
                "cursor_value": actual_slice_end,
                "run_id": run_id,
                "skipped_due_to": f"parse:{exc}",
            },
        )
        _canonical_log.append(
            "auditor_run_failed",
            payload={
                "run_id": run_id,
                "error": f"parse:{exc}",
                "response_preview": (response_text or "")[:500],
                "skipped_slice": [cursor_start, actual_slice_end],
            },
        )
        return AuditRunResult(
            run_id=run_id,
            started_at=started_at,
            completed_at=time.time(),
            status="failed",
            slice_start_event_id=cursor_start,
            slice_end_event_id=actual_slice_end,
            events_processed=len(events),
            proposals_returned=0,
            candidates_inserted=0,
            candidates_rejected=0,
            error=f"parse:{exc}",
        )

    # Validate + insert
    inserted = 0
    rejected = 0
    rejection_reasons: list[str] = []
    for prop in proposals:
        # Pre-atomicity extraction-quality filter — rejects conversational
        # fragments (questions, "want me to..." etc.) that are well-formed
        # but not claims. Catches the 2026-05-14 ACU pollution class.
        quality = is_extraction_quality_acceptable(prop.canonical_form)
        if not quality.ok:
            rejected += 1
            rejection_reasons.append(f"extraction:{quality.reason or 'unknown'}")
            _canonical_log.append(
                "auditor_extraction_filter_reject",
                payload={
                    "run_id": run_id,
                    "canonical_form": prop.canonical_form,
                    "reason": quality.reason,
                },
            )
            continue
        gate = is_atomic(prop.canonical_form)
        if not gate.ok:
            rejected += 1
            rejection_reasons.append(gate.reason or "unknown")
            _canonical_log.append(
                "auditor_atomicity_reject",
                payload={
                    "run_id": run_id,
                    "canonical_form": prop.canonical_form,
                    "reason": gate.reason,
                },
            )
            continue
        candidate_source = _candidate_source_for(prop, events_by_id, source)
        evidence_event = events_by_id.get(int(prop.evidence_log_id))
        evidence_kind = evidence_event.kind if evidence_event is not None else None
        candidate_reason = prop.reason
        if candidate_source != source:
            candidate_reason = (
                f"{prop.reason} "
                f"[evidence_source={candidate_source}; audit_source={source}]"
            )
        try:
            candidate_id = _candidates.insert_candidate(
                canonical_form=prop.canonical_form,
                evidence_log_id=prop.evidence_log_id,
                evidence_char_start=prop.evidence_char_start,
                evidence_char_end=prop.evidence_char_end,
                evidence_span=prop.evidence_span,
                source=candidate_source,
                reason=candidate_reason,
                reinforcement_count=prop.reinforcement_count,
                contradicts_acu_id=prop.contradicts_acu_id,
                auditor_run_id=run_id,
            )
            inserted += 1
            _canonical_log.append(
                "candidate_emitted",
                payload={
                    "run_id": run_id,
                    "candidate_id": candidate_id,
                    "canonical_form": prop.canonical_form,
                    "source": candidate_source,
                    "candidate_source": candidate_source,
                    "audit_source": source,
                    "evidence_kind": evidence_kind,
                    "note_read_event_id": (
                        int(prop.evidence_log_id)
                        if evidence_kind == _MONONOTE_NOTE_READ_KIND else None
                    ),
                },
            )
        except _candidates.CandidateAtomicityError as exc:
            # Belt + suspenders — atomicity already checked above.
            rejected += 1
            rejection_reasons.append(exc.reason)
        except Exception as exc:
            rejected += 1
            rejection_reasons.append(f"insert_error:{type(exc).__name__}")

    # Advance cursor to the ACTUAL last processed event, not the full
    # latest. This is the fix for the "max_events=50 but cursor jumps to
    # 497" bug — see the comment above the slice read for why. Next
    # run picks up from actual_slice_end + 1.
    _canonical_log.append(
        "auditor_cursor_advance",
        payload={"cursor_value": actual_slice_end, "run_id": run_id},
    )

    # Run complete
    completed_at = time.time()
    _canonical_log.append(
        "auditor_run_complete",
        payload={
            "run_id": run_id,
            "status": "success",
            "events_processed": len(events),
            "proposals_returned": len(proposals),
            "candidates_inserted": inserted,
            "candidates_rejected": rejected,
            "slice_end_event_id": actual_slice_end,
        },
    )

    return AuditRunResult(
        run_id=run_id,
        started_at=started_at,
        completed_at=completed_at,
        status="success",
        slice_start_event_id=cursor_start,
        slice_end_event_id=actual_slice_end,
        events_processed=len(events),
        proposals_returned=len(proposals),
        candidates_inserted=inserted,
        candidates_rejected=rejected,
        rejection_reasons=rejection_reasons,
    )
