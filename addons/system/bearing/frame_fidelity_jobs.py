"""MonoFrame v2 — the fidelity job ledger (exact-once delivery).

The reliability layer for the constraint. Every committed frame SYNCHRONOUSLY
queues a fidelity job; the async judge completes it later, but the job cannot
disappear silently — a turn without a verdict has a visible job state (queued /
running / failed / skipped / expired). On restart, incomplete jobs are retried.
Duplicates (same frame_record_hash + answer_digest, which determine the job_id)
are rejected.

Append-only event log: the current state of a job is its latest row. Flag
MONOLITH_MONOFRAME_V1. Never raises into the chat path.

  Recorder   = commitment.
  Judge      = observational constraint.
  Job ledger = reliability of the constraint (this module).
"""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

from core.paths import CONFIG_DIR

from . import frame_fidelity, frame_selection

_FLAG_ENV = "MONOLITH_MONOFRAME_V1"
_TRUTHY = {"1", "true", "yes", "on"}

STORAGE_SURFACE = "frame_fidelity_jobs.jsonl"
_STORE = CONFIG_DIR / STORAGE_SURFACE

_INCOMPLETE = {"queued", "running"}


def enabled() -> bool:
    return os.environ.get(_FLAG_ENV, "0").strip().lower() in _TRUTHY


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def job_id_for(frame_record_hash: str, answer_digest: str) -> str:
    """Deterministic id from the dedup key — same commitment+answer => same job."""
    return frame_selection.digest(f"{frame_record_hash}|{answer_digest}")[:16]


# ── append-only log + current state ─────────────────────────────────


def _append(rec: dict[str, Any]) -> None:
    _STORE.parent.mkdir(parents=True, exist_ok=True)
    with _STORE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _all_rows() -> list[dict[str, Any]]:
    if not _STORE.exists():
        return []
    out: list[dict[str, Any]] = []
    try:
        for line in _STORE.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except Exception:
                    pass
    except Exception:
        return []
    return out


def current_jobs() -> dict[str, dict[str, Any]]:
    """job_id -> latest snapshot (append order = chronological)."""
    latest: dict[str, dict[str, Any]] = {}
    for r in _all_rows():
        jid = r.get("job_id")
        if jid:
            latest[jid] = r
    return latest


def get_job(job_id: str) -> dict[str, Any] | None:
    return current_jobs().get(job_id)


def incomplete_jobs() -> list[dict[str, Any]]:
    return [j for j in current_jobs().values() if j.get("status") in _INCOMPLETE]


# ── enqueue + transitions ───────────────────────────────────────────


def enqueue_job(
    *,
    turn_id: str,
    frame_record_hash: str,
    answer_digest: str,
    judge_version: str,
    answer: str = "",
    now: str | None = None,
) -> tuple[str, bool]:
    """Synchronously queue a job for a committed frame. Returns (job_id, created).
    A duplicate (same frame_record_hash + answer_digest) is rejected -> created False."""
    jid = job_id_for(frame_record_hash, answer_digest)
    if get_job(jid) is not None:
        return (jid, False)  # already tracked — exact-once
    rec = {
        "job_id": jid,
        "turn_id": turn_id,
        "frame_record_hash": frame_record_hash,
        "answer_digest": answer_digest,
        "status": "queued",
        "queued_at_utc": now or _now(),
        "started_at_utc": "",
        "completed_at_utc": "",
        "retry_count": 0,
        "error": "",
        "judge_version": judge_version,
        "output_fidelity_record_hash": "",
        "answer": answer,   # retained so a crashed job can be retried on recovery
        "verdict_possible": True,
        "reason": "",
    }
    _append(rec)
    return (jid, True)


def enqueue_recorder_failure(
    *, turn_id: str, answer_digest: str, now: str | None = None
) -> tuple[str, bool]:
    """An ANSWERING turn produced NO frame_selection block. That is a RECORDER
    FAILURE, not a skip — the answer exists but could not be judged because the
    commitment was never recorded. A visible ``failed`` job, so the absence is an
    artifact, not a silent gap. Returns (job_id, created)."""
    jid = frame_selection.digest(f"{turn_id}|norecord|{answer_digest}")[:16]
    if get_job(jid) is not None:
        return (jid, False)
    ts = now or _now()
    _append({
        "job_id": jid, "turn_id": turn_id, "frame_record_hash": "",
        "answer_digest": answer_digest, "status": "failed",
        "queued_at_utc": ts, "started_at_utc": "", "completed_at_utc": ts,
        "retry_count": 0, "error": "no_frame_selection_block", "judge_version": "",
        "output_fidelity_record_hash": "", "answer": "",
        "verdict_possible": False, "reason": "recorder_failed_before_judge",
    })
    return (jid, True)


def enqueue_skipped(
    *, turn_id: str, reason: str, answer_digest: str = "", now: str | None = None
) -> tuple[str, bool]:
    """A LEGITIMATE non-answer turn: no assistant final, tool-only internal turn,
    aborted generation, or safety/system interruption before an answer. Visible
    ``skipped`` job; no verdict is possible. Returns (job_id, created)."""
    jid = frame_selection.digest(f"{turn_id}|skipped|{answer_digest}")[:16]
    if get_job(jid) is not None:
        return (jid, False)
    ts = now or _now()
    _append({
        "job_id": jid, "turn_id": turn_id, "frame_record_hash": "",
        "answer_digest": answer_digest, "status": "skipped",
        "queued_at_utc": ts, "started_at_utc": "", "completed_at_utc": ts,
        "retry_count": 0, "error": str(reason), "judge_version": "",
        "output_fidelity_record_hash": "", "answer": "",
        "verdict_possible": False, "reason": str(reason),
    })
    return (jid, True)


def _update(job_id: str, **changes: Any) -> None:
    cur = get_job(job_id)
    if cur is None:
        return
    _append({**cur, **changes})


def mark_running(job_id: str, now: str | None = None) -> None:
    _update(job_id, status="running", started_at_utc=now or _now())


def mark_complete(job_id: str, *, output_fidelity_record_hash: str, now: str | None = None) -> None:
    _update(job_id, status="complete", completed_at_utc=now or _now(),
            output_fidelity_record_hash=output_fidelity_record_hash)


def mark_failed(job_id: str, *, error: str, now: str | None = None) -> None:
    _update(job_id, status="failed", error=str(error), completed_at_utc=now or _now())


def mark_skipped(job_id: str, *, reason: str, now: str | None = None) -> None:
    _update(job_id, status="skipped", error=str(reason), completed_at_utc=now or _now())


def mark_expired(job_id: str, *, reason: str, now: str | None = None) -> None:
    _update(job_id, status="expired", error=str(reason), completed_at_utc=now or _now())


# ── run + recover ───────────────────────────────────────────────────


def run_job(
    job_id: str,
    *,
    frame_record: dict[str, Any],
    answer: str,
    base_config: dict[str, Any] | None = None,
    generate=None,
) -> bool:
    """Mark running -> judge -> write fidelity record -> mark complete (or failed).
    Returns whether the job completed."""
    mark_running(job_id)
    try:
        rec = frame_fidelity.build_fidelity_record(
            frame_record=frame_record, answer=answer,
            base_config=base_config, generate=generate,
        )
        frame_fidelity.record_fidelity(rec)
        out_hash = frame_selection.digest(
            json.dumps(rec.to_dict(), sort_keys=True, ensure_ascii=False)
        )
        mark_complete(job_id, output_fidelity_record_hash=out_hash)
        return True
    except Exception as exc:
        mark_failed(job_id, error=str(exc))
        return False


def run_job_async(
    job_id: str,
    *,
    frame_record: dict[str, Any],
    answer: str,
    base_config: dict[str, Any] | None = None,
    generate=None,
) -> threading.Thread:
    def _work() -> None:
        try:
            run_job(job_id, frame_record=frame_record, answer=answer,
                    base_config=base_config, generate=generate)
        except Exception:
            pass

    t = threading.Thread(target=_work, name=f"fidelity-job-{job_id}", daemon=True)
    t.start()
    return t


def _frame_record_by_hash(frame_record_hash: str) -> dict[str, Any] | None:
    for r in frame_selection.read_recent(limit=10_000):
        if r.get("artifact_hash") == frame_record_hash:
            return r
    return None


_recovered = False


def recover_once(*, base_config: dict[str, Any] | None = None, generate=None) -> int:
    """Once-per-process crash recovery (call on the first turn after startup).
    Dispatches incomplete jobs ASYNC so the triggering turn is not blocked. Returns
    the number dispatched, or -1 if recovery already ran this process."""
    global _recovered
    if _recovered:
        return -1
    _recovered = True
    n = 0
    for job in incomplete_jobs():
        jid = job["job_id"]
        fr = _frame_record_by_hash(job.get("frame_record_hash", ""))
        ans = job.get("answer", "")
        if fr is None or not ans:
            mark_expired(jid, reason="inputs unrecoverable on restart")
            continue
        _update(jid, retry_count=int(job.get("retry_count", 0)) + 1)
        run_job_async(jid, frame_record=fr, answer=ans, base_config=base_config, generate=generate)
        n += 1
    return n


def recover_incomplete(*, base_config: dict[str, Any] | None = None, generate=None) -> int:
    """Re-dispatch incomplete (queued/running) jobs, e.g. after a crash/restart.
    Re-fetches the committed frame by hash and the retained answer; a job whose
    inputs are no longer recoverable is marked ``expired`` (visible, not silent).
    Returns how many jobs were re-run to completion."""
    done = 0
    for job in incomplete_jobs():
        jid = job["job_id"]
        fr = _frame_record_by_hash(job.get("frame_record_hash", ""))
        ans = job.get("answer", "")
        if fr is None or not ans:
            mark_expired(jid, reason="inputs unrecoverable on restart")
            continue
        _update(jid, retry_count=int(job.get("retry_count", 0)) + 1)
        if run_job(jid, frame_record=fr, answer=ans, base_config=base_config, generate=generate):
            done += 1
    return done
