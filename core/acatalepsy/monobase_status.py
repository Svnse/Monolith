"""Read model for the MonoBase companion pane.

The auditor and the decision layer have different write semantics:
audits produce pending candidates, while accept/edit decisions produce
actual ACU rows. This module keeps that distinction in one testable
place so the UI can render truthful status instead of inferring it from
ad hoc label text.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Iterable

from core.db_connect import connect_acatalepsy
from core.acatalepsy import auditor as _auditor
from core.acatalepsy import candidates as _candidates
from core.acatalepsy import canonical_log as _canonical_log
from core.acatalepsy import runtime as _runtime


__all__ = (
    "AcuWrite",
    "AcuRecord",
    "AuditorLlmOutput",
    "MonobaseSnapshot",
    "WorkerStatus",
    "build_monobase_snapshot",
    "format_elapsed",
    "format_recent_acu_write",
    "read_acu",
    "read_latest_llm_output",
    "read_recent_acus",
    "read_worker_status",
)


_ACU_WRITE_KINDS = frozenset({"candidate_accepted", "candidate_edited"})


@dataclass(frozen=True)
class WorkerStatus:
    registered: bool
    thread_alive: bool
    stop_requested: bool
    queue_size: int | None
    size_threshold: int | None
    max_events_per_run: int | None


@dataclass(frozen=True)
class AcuWrite:
    event_id: int
    ts: float
    kind: str
    candidate_id: int | None
    decision_id: int | None
    acu_id: int | None
    decided_by: str
    canonical_form: str | None


@dataclass(frozen=True)
class AcuRecord:
    id: int
    canonical: str
    source: str
    created_at: str
    last_seen: str
    veracity: float
    reinforcement: int
    candidate_id: int | None
    decision_id: int | None


@dataclass(frozen=True)
class AuditorLlmOutput:
    event_id: int
    ts: float
    run_id: int | None
    status: str
    elapsed_secs: float | None
    response_chars: int | None
    response_preview: str
    response_truncated: bool
    error: str | None


@dataclass(frozen=True)
class MonobaseSnapshot:
    now: float
    phase: str
    phase_title: str
    phase_detail: str
    phase_tone: str
    cursor: int
    latest_event_id: int
    pending_log_events: int
    candidate_counts: dict[str, int]
    pending_candidate_count: int
    worker: WorkerStatus
    in_flight_run: dict[str, Any] | None
    run_elapsed_secs: int | None
    llm_elapsed_secs: int | None
    recent_runs: tuple[dict[str, Any], ...]
    recent_acu_writes: tuple[AcuWrite, ...]


def format_elapsed(seconds: float | int | None) -> str:
    if seconds is None:
        return "--"
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def format_recent_acu_write(write: AcuWrite, *, now: float | None = None) -> str:
    age = format_elapsed((time.time() if now is None else now) - write.ts)
    acu = f"ACU #{write.acu_id}" if write.acu_id is not None else "ACU ?"
    candidate = (
        f"candidate #{write.candidate_id}"
        if write.candidate_id is not None
        else "unknown candidate"
    )
    decision = (
        f"decision #{write.decision_id}"
        if write.decision_id is not None
        else "decision ?"
    )
    return f"{acu} from {candidate} via {decision} ({age} ago)"


def read_worker_status(worker: Any | None = None) -> WorkerStatus:
    if worker is None:
        worker = _runtime.get_active_worker()
    if worker is None:
        return WorkerStatus(
            registered=False,
            thread_alive=False,
            stop_requested=False,
            queue_size=None,
            size_threshold=None,
            max_events_per_run=None,
        )

    thread = getattr(worker, "_thread", None)
    stop_event = getattr(worker, "_stop_event", None)
    queue_handle = getattr(worker, "queue_handle", None)
    queue_size: int | None = None
    if queue_handle is not None:
        try:
            queue_size = int(queue_handle.size())
        except Exception:
            queue_size = None

    return WorkerStatus(
        registered=True,
        thread_alive=bool(thread is not None and thread.is_alive()),
        stop_requested=bool(stop_event is not None and stop_event.is_set()),
        queue_size=queue_size,
        size_threshold=_int_or_none(getattr(worker, "_size_threshold", None)),
        max_events_per_run=_int_or_none(getattr(worker, "_max_events", None)),
    )


def read_recent_acus(limit: int = 500) -> list[AcuRecord]:
    if limit < 1:
        return []
    conn = connect_acatalepsy(role="reader")
    try:
        cur = conn.execute(
            "SELECT id, canonical, source, created_at, last_seen, veracity, reinforcement, "
            "candidate_id, decision_id "
            "FROM acus ORDER BY id DESC LIMIT ?",
            (int(limit),),
        )
        return [_row_to_acu(row) for row in cur.fetchall()]
    finally:
        conn.close()


def read_acu(acu_id: int) -> AcuRecord | None:
    conn = connect_acatalepsy(role="reader")
    try:
        cur = conn.execute(
            "SELECT id, canonical, source, created_at, last_seen, veracity, reinforcement, "
            "candidate_id, decision_id "
            "FROM acus WHERE id=?",
            (int(acu_id),),
        )
        row = cur.fetchone()
        return _row_to_acu(row) if row is not None else None
    finally:
        conn.close()


def _row_to_acu(row: Any) -> AcuRecord:
    return AcuRecord(
        id=int(row["id"]),
        canonical=str(row["canonical"]),
        source=str(row["source"]),
        created_at=str(row["created_at"]),
        last_seen=str(row["last_seen"]),
        veracity=float(row["veracity"]),
        reinforcement=int(row["reinforcement"]),
        candidate_id=(
            int(row["candidate_id"]) if row["candidate_id"] is not None else None
        ),
        decision_id=(
            int(row["decision_id"]) if row["decision_id"] is not None else None
        ),
    )


def read_latest_llm_output() -> AuditorLlmOutput | None:
    for ev in _auditor.read_audit_log_tail(limit=80):
        if ev.get("kind") != "auditor_llm_call_returned":
            continue
        payload = ev.get("payload") or {}
        preview = str(payload.get("response_preview") or "")
        return AuditorLlmOutput(
            event_id=int(ev.get("event_id") or 0),
            ts=float(ev.get("ts") or 0),
            run_id=_int_or_none(payload.get("run_id")),
            status=str(payload.get("status") or ""),
            elapsed_secs=_float_or_none(payload.get("elapsed_secs")),
            response_chars=_int_or_none(payload.get("response_chars")),
            response_preview=preview,
            response_truncated=bool(payload.get("response_truncated")),
            error=str(payload.get("error")) if payload.get("error") else None,
        )
    return None


def build_monobase_snapshot(*, now: float | None = None) -> MonobaseSnapshot:
    current_time = time.time() if now is None else float(now)
    cursor = int(_auditor.last_processed_event_id())
    latest = int(_canonical_log.latest_event_id())
    pending_log_events = max(0, latest - cursor)
    counts = dict(_candidates.count_by_state())
    pending_candidates = int(counts.get("pending", 0))
    worker = read_worker_status()
    in_flight = _auditor.current_in_flight_run()
    audit_tail = tuple(_auditor.read_audit_log_tail(limit=80))
    run_events = _events_for_current_run(in_flight, audit_tail)
    run_elapsed = _elapsed_from_event_ts(in_flight, current_time)
    llm_elapsed = _llm_elapsed(run_events, current_time)
    recent_runs = tuple(_auditor.read_recent_runs(limit=10))
    recent_acu_writes = tuple(_read_recent_acu_writes(limit=6))
    phase, title, detail, tone = _derive_phase(
        worker=worker,
        in_flight=in_flight,
        run_events=run_events,
        pending_log_events=pending_log_events,
        pending_candidates=pending_candidates,
        run_elapsed=run_elapsed,
        llm_elapsed=llm_elapsed,
    )

    return MonobaseSnapshot(
        now=current_time,
        phase=phase,
        phase_title=title,
        phase_detail=detail,
        phase_tone=tone,
        cursor=cursor,
        latest_event_id=latest,
        pending_log_events=pending_log_events,
        candidate_counts=counts,
        pending_candidate_count=pending_candidates,
        worker=worker,
        in_flight_run=in_flight,
        run_elapsed_secs=run_elapsed,
        llm_elapsed_secs=llm_elapsed,
        recent_runs=recent_runs,
        recent_acu_writes=recent_acu_writes,
    )


def _derive_phase(
    *,
    worker: WorkerStatus,
    in_flight: dict[str, Any] | None,
    run_events: tuple[dict[str, Any], ...],
    pending_log_events: int,
    pending_candidates: int,
    run_elapsed: int | None,
    llm_elapsed: int | None,
) -> tuple[str, str, str, str]:
    if worker.stop_requested and worker.thread_alive:
        return (
            "stopping",
            "Stopping auditor",
            "Waiting for the worker to return from its current call.",
            "warn",
        )

    if in_flight is not None:
        run_id = _int_or_none(in_flight.get("event_id"))
        slice_start = _int_or_none(in_flight.get("slice_start_event_id")) or 0
        slice_end = _int_or_none(in_flight.get("slice_end_event_id")) or 0
        if _has_llm_started(run_events) and not _has_llm_returned(run_events):
            return (
                "calling_llm",
                "Calling auditor LLM",
                (
                    f"Run #{run_id or '?'} slice {slice_start}-{slice_end}; "
                    f"LLM elapsed {format_elapsed(llm_elapsed)}."
                ),
                "active",
            )
        if _has_llm_returned(run_events):
            return (
                "updating_candidates",
                "Updating candidates",
                (
                    "LLM returned; parsing and saving pending candidates. "
                    "ACUs are written only on Accept/Edit."
                ),
                "active",
            )
        return (
            "auditing_log",
            "Auditing log",
            (
                f"Run #{run_id or '?'} slice {slice_start}-{slice_end}; "
                f"run elapsed {format_elapsed(run_elapsed)}."
            ),
            "active",
        )

    if worker.thread_alive:
        if worker.queue_size and worker.queue_size > 0:
            return (
                "queued",
                "Audit queued",
                f"{worker.queue_size} trigger(s) waiting for the auditor worker.",
                "active",
            )
        if pending_log_events > 0:
            threshold = worker.size_threshold or 0
            suffix = f" Threshold is {threshold}." if threshold else ""
            return (
                "watching",
                "Watching log",
                f"{pending_log_events} event(s) behind cursor.{suffix}",
                "idle",
            )
        return (
            "caught_up",
            "Caught up",
            "Worker is on; no unaudited canonical-log events right now.",
            "idle",
        )

    if worker.registered:
        return (
            "paused",
            "Auditor paused",
            "Worker exists but is not running; manual Audit can queue work.",
            "idle",
        )

    if pending_log_events > 0:
        return (
            "needs_audit",
            "Audit available",
            f"{pending_log_events} event(s) behind cursor; start the auditor to inspect them.",
            "warn",
        )

    if pending_candidates > 0:
        return (
            "candidates_pending",
            "Candidates pending",
            "Audit is idle; pending candidates need Accept/Reject decisions.",
            "warn",
        )

    return (
        "off",
        "Auditor off",
        "No worker is registered. Audits create candidates, not ACUs.",
        "idle",
    )


def _events_for_current_run(
    in_flight: dict[str, Any] | None,
    audit_tail: Iterable[dict[str, Any]],
) -> tuple[dict[str, Any], ...]:
    if in_flight is None:
        return ()
    run_id = _int_or_none(in_flight.get("event_id"))
    if run_id is None:
        return ()
    events: list[dict[str, Any]] = []
    for ev in audit_tail:
        payload = ev.get("payload") or {}
        payload_run_id = _int_or_none(payload.get("run_id"))
        event_id = _int_or_none(ev.get("event_id"))
        if payload_run_id == run_id or (
            ev.get("kind") == "auditor_run_started" and event_id == run_id
        ):
            events.append(ev)
    events.sort(key=lambda item: int(item.get("event_id") or 0))
    return tuple(events)


def _has_llm_started(events: Iterable[dict[str, Any]]) -> bool:
    return any(ev.get("kind") == "auditor_llm_call_started" for ev in events)


def _has_llm_returned(events: Iterable[dict[str, Any]]) -> bool:
    return any(ev.get("kind") == "auditor_llm_call_returned" for ev in events)


def _llm_elapsed(events: tuple[dict[str, Any], ...], now: float) -> int | None:
    started = next(
        (ev for ev in reversed(events) if ev.get("kind") == "auditor_llm_call_started"),
        None,
    )
    if started is None:
        return None
    returned = next(
        (ev for ev in reversed(events) if ev.get("kind") == "auditor_llm_call_returned"),
        None,
    )
    if returned is not None:
        payload = returned.get("payload") or {}
        elapsed = payload.get("elapsed_secs")
        try:
            return max(0, int(float(elapsed)))
        except (TypeError, ValueError):
            return _elapsed_from_event_ts(started, float(returned.get("ts") or now))
    return _elapsed_from_event_ts(started, now)


def _elapsed_from_event_ts(event: dict[str, Any] | None, now: float) -> int | None:
    if event is None:
        return None
    try:
        ts = float(event.get("ts") or 0)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    return max(0, int(now - ts))


def _read_recent_acu_writes(limit: int = 6) -> list[AcuWrite]:
    events = _read_recent_events(_ACU_WRITE_KINDS, limit=limit, scan_limit=5000)
    out: list[AcuWrite] = []
    for ev in events:
        payload = ev.payload or {}
        candidate_id = _int_or_none(payload.get("candidate_id"))
        candidate = _candidates.read_one(candidate_id) if candidate_id is not None else None
        out.append(
            AcuWrite(
                event_id=int(ev.event_id),
                ts=float(ev.ts),
                kind=ev.kind,
                candidate_id=candidate_id,
                decision_id=_int_or_none(payload.get("decision_id")),
                acu_id=(
                    _int_or_none(payload.get("resulting_acu_id"))
                    if payload.get("resulting_acu_id") is not None
                    else ev.acu_id
                ),
                decided_by=str(payload.get("decided_by") or ""),
                canonical_form=getattr(candidate, "canonical_form", None),
            )
        )
    return out


def _read_recent_events(
    kinds: frozenset[str],
    *,
    limit: int,
    scan_limit: int,
) -> list[_canonical_log.Event]:
    if limit < 1:
        return []
    latest = _canonical_log.latest_event_id()
    if latest == 0:
        return []
    out: list[_canonical_log.Event] = []
    chunk = 500
    end = latest
    scanned = 0
    while end > 0 and len(out) < limit and scanned < scan_limit:
        start = max(0, end - chunk)
        events = _canonical_log.read_since(start, limit=chunk)
        scanned += len(events)
        for ev in reversed(events):
            if ev.event_id > end:
                continue
            if ev.kind in kinds:
                out.append(ev)
                if len(out) >= limit:
                    break
        if start == 0:
            break
        end = start
    return out


def _int_or_none(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_or_none(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
