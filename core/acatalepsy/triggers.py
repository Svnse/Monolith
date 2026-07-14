"""Trigger queue + async worker for the Acatalepsy auditor.

A1 ships only the size-threshold and manual triggers. Time + session-close
land in A2.

Design
======

* A single ``TriggerQueue`` (thread-safe) holds pending trigger events.
* A single ``AuditorWorker`` daemon thread blocks on the queue with a
  poll timeout. On wakeup it processes any queued trigger AND polls
  the size threshold (cheap canonical_log COUNT).
* The worker invokes ``auditor.run_audit(llm, source=...)``. The result
  is captured for observability but not surfaced anywhere mandatory in
  v1 — call sites can subscribe via the optional ``on_result`` callback.
* Errors from the audit pipeline are caught and logged to
  canonical_log as ``auditor_run_failed`` (already handled inside
  run_audit) — the worker thread doesn't die.

Public API
==========
  - TriggerQueue.enqueue(kind, note=None)
  - TriggerQueue.size()
  - AuditorWorker(llm, source, queue=None, ...)
  - AuditorWorker.start()
  - AuditorWorker.stop(timeout=None)

The dev panel calls ``queue.enqueue("manual")`` when the Audit-now
button is clicked. The chat dispatch (or a periodic timer) doesn't
need to enqueue size triggers — the worker polls them itself.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Callable

from core.acatalepsy import auditor as _auditor
from core.acatalepsy import canonical_log as _canonical_log


__all__ = (
    "DEFAULT_POLL_INTERVAL_SECS",
    "DEFAULT_SIZE_THRESHOLD",
    "Trigger",
    "TriggerQueue",
    "AuditorWorker",
)


DEFAULT_SIZE_THRESHOLD: int = 50
DEFAULT_POLL_INTERVAL_SECS: float = 30.0
# Default max events per audit run. Smaller than run_audit's intrinsic
# 200 because cloud LLMs handle 50-event slices better — less context to
# hallucinate over, faster per-run latency, and the user can crank it
# higher per session via the MonoBase max-events spinbox. The auditor
# advances the cursor by this much per run, so multiple runs cover a
# long log if needed.
DEFAULT_MAX_EVENTS_PER_RUN: int = 50


_VALID_TRIGGER_KINDS: frozenset[str] = frozenset({"manual", "size"})


@dataclass(frozen=True)
class Trigger:
    kind: str                           # "manual" | "size"
    enqueued_at: float
    note: str | None = None
    enqueued_by: str | None = None


class TriggerQueue:
    """Thread-safe queue of trigger events.

    The queue is unbounded but de-duplicates back-to-back identical
    triggers when they arrive within a short window (1 sec) — keeps
    UI button-mashing from queuing 10 manual runs.
    """

    def __init__(self) -> None:
        self._q: "queue.Queue[Trigger]" = queue.Queue()
        self._lock = threading.Lock()
        self._last_enqueue_ts: dict[str, float] = {}

    def enqueue(
        self,
        kind: str,
        note: str | None = None,
        enqueued_by: str | None = None,
        *,
        bypass_dedup: bool = False,
    ) -> bool:
        """Add a trigger to the queue. Returns True if enqueued, False
        if deduplicated (same kind within 1 sec window).

        ``bypass_dedup=True`` skips the dedup window — used by
        AuditorWorker.stop() to guarantee the wake-up token lands
        even if a manual trigger was enqueued moments earlier.
        """
        if kind not in _VALID_TRIGGER_KINDS:
            raise ValueError(
                f"invalid trigger kind {kind!r}; valid: {sorted(_VALID_TRIGGER_KINDS)}"
            )
        now = time.time()
        if not bypass_dedup:
            with self._lock:
                last = self._last_enqueue_ts.get(kind, 0.0)
                if now - last < 1.0:
                    return False
                self._last_enqueue_ts[kind] = now
        self._q.put(Trigger(kind=kind, enqueued_at=now, note=note, enqueued_by=enqueued_by))
        return True

    def size(self) -> int:
        return self._q.qsize()

    def get(self, timeout: float | None = None) -> Trigger | None:
        """Blocking get with timeout. Returns None on timeout."""
        try:
            return self._q.get(timeout=timeout)
        except queue.Empty:
            return None


class AuditorWorker:
    """Background thread that drives the auditor.

    Lifecycle:
      worker = AuditorWorker(llm=my_llm, source="auditor_monolith")
      worker.start()
      ...
      worker.stop()

    Behavior per wakeup:
      1. If a Trigger is on the queue (manual or size), pull and process.
      2. On timeout, check size threshold; if (latest_event_id - cursor)
         >= threshold, run an audit.
      3. Sleep until next wakeup.
    """

    def __init__(
        self,
        llm: _auditor.LLMCallable,
        source: str,
        *,
        queue: TriggerQueue | None = None,
        size_threshold: int = DEFAULT_SIZE_THRESHOLD,
        poll_interval_secs: float = DEFAULT_POLL_INTERVAL_SECS,
        max_events_per_run: int = DEFAULT_MAX_EVENTS_PER_RUN,
        on_result: Callable[[_auditor.AuditRunResult], None] | None = None,
    ) -> None:
        if not source.strip():
            raise ValueError("source must be non-empty")
        if size_threshold < 1:
            raise ValueError("size_threshold must be >= 1")
        if poll_interval_secs <= 0:
            raise ValueError("poll_interval_secs must be > 0")
        if max_events_per_run < 1:
            raise ValueError("max_events_per_run must be >= 1")

        self._llm = llm
        self._source = source
        self._queue = queue if queue is not None else TriggerQueue()
        self._size_threshold = int(size_threshold)
        self._poll_interval = float(poll_interval_secs)
        self._max_events = int(max_events_per_run)
        self._on_result = on_result

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def queue_handle(self) -> TriggerQueue:
        """Expose the queue so external code (UI buttons) can enqueue."""
        return self._queue

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run,
            name=f"acatalepsy-auditor-{self._source}",
            daemon=True,
        )
        self._thread.start()

    def stop(self, timeout: float | None = 5.0) -> None:
        self._stop_event.set()
        # Wake the queue.get if it's blocked. bypass_dedup=True ensures
        # the token lands even if a manual trigger was enqueued <1s ago.
        self._queue.enqueue("manual", note="stop-shutdown", bypass_dedup=True)
        thread = self._thread
        if thread is not None:
            thread.join(timeout=timeout)
            if not thread.is_alive():
                self._thread = None

    # ── internals ────────────────────────────────────────────────────

    def _run(self) -> None:
        while not self._stop_event.is_set():
            trigger = self._queue.get(timeout=self._poll_interval)
            if self._stop_event.is_set():
                return
            if trigger is None:
                # Timeout: poll size threshold
                self._maybe_run_size_audit()
                continue
            if trigger.note == "stop-shutdown":
                return
            # Real trigger — run.
            self._run_one(trigger)

    def _maybe_run_size_audit(self) -> None:
        latest = _canonical_log.latest_event_id()
        cursor = _auditor.last_processed_event_id()
        if (latest - cursor) >= self._size_threshold:
            self._run_one(Trigger(kind="size", enqueued_at=time.time()))

    def set_max_events(self, n: int) -> None:
        """Update the per-run slice cap. Takes effect on the next audit
        run — does not affect a run already in flight."""
        if n < 1:
            return
        self._max_events = int(n)

    def _run_one(self, trigger: Trigger) -> None:
        try:
            result = _auditor.run_audit(
                self._llm,
                source=self._source,
                max_events=self._max_events,
            )
        except Exception as exc:
            # run_audit handles its own internal errors, so this is
            # only catching truly unexpected ones. Log and continue —
            # don't die.
            try:
                _canonical_log.append(
                    "auditor_run_failed",
                    payload={
                        "trigger_kind": trigger.kind,
                        "error": f"worker_unexpected:{type(exc).__name__}:{exc}",
                    },
                )
            except Exception:
                pass
            return
        if self._on_result is not None:
            try:
                self._on_result(result)
            except Exception:
                pass
