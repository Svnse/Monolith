"""In-memory ring buffer mirroring UI trace/debug signals.

Subscribes to Qt `sig_trace` / `sig_debug` signals (connected by
bootstrap.py) and stores recent lines for `/log/tail` and `/log/since`
HTTP endpoints served by `engine/agent_server.py`.

Why a ring buffer not a file: file would grow unbounded, require
rotation, add IO latency to signal handlers. Ring buffer is bounded,
in-process, zero-config. Persistence can mirror to disk later if
needed; not v1.

This module is *not* agent-server-specific — it's a general signal
mirror. Any consumer (current server, future server, debug tooling,
audit infrastructure) can read from it.

Public API:
    ring = get_ring()
    ring.append(source, level, text)
    ring.tail(n=200) -> list[dict]
    ring.since(seq) -> list[dict]
    connect_signals(*emitters)  # bootstrap wiring

Flag: MONOLITH_LOG_MIRROR_CAP — max buffered lines. Default 1000.
"""
from __future__ import annotations

import os
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any, Callable

_DEFAULT_CAP = 1000
_CAP_ENV = "MONOLITH_LOG_MIRROR_CAP"


def _resolved_cap() -> int:
    raw = os.environ.get(_CAP_ENV)
    if not raw:
        return _DEFAULT_CAP
    try:
        return max(50, min(int(raw), 10_000))
    except (TypeError, ValueError):
        return _DEFAULT_CAP


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class LogRingBuffer:
    """Bounded sequence-numbered log buffer.

    Sequence numbers are monotonic and survive cap rollover — old lines
    fall off the head, but `since(seq)` callers can detect they missed
    entries by comparing the seq of the first returned line to their
    request `seq + 1`. If the returned head has `seq > request_seq + 1`,
    some lines were dropped between polls.
    """

    def __init__(self, capacity: int | None = None) -> None:
        cap = capacity if capacity is not None else _resolved_cap()
        self._cap = max(1, int(cap))
        self._buf: deque[dict[str, Any]] = deque(maxlen=self._cap)
        self._seq = 0
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._cap

    def append(self, source: str, level: str, text: str, *, ts: str | None = None) -> int:
        """Append a line. Returns the assigned seq number."""
        with self._lock:
            self._seq += 1
            entry = {
                "seq": self._seq,
                "ts": ts or _now_iso(),
                "source": str(source or "unknown"),
                "level": str(level or "trace"),
                "text": str(text or ""),
            }
            self._buf.append(entry)
            return self._seq

    def tail(self, n: int = 200) -> list[dict[str, Any]]:
        """Return up to `n` most-recent lines, oldest-first within the slice."""
        n = max(1, min(int(n or 200), self._cap))
        with self._lock:
            if n >= len(self._buf):
                return list(self._buf)
            return list(self._buf)[-n:]

    def since(self, seq: int) -> list[dict[str, Any]]:
        """Return lines with `seq > given_seq`, oldest-first.

        If the requested seq is older than the buffer's head, callers
        can detect drops by comparing the returned head seq to
        `given_seq + 1`.
        """
        try:
            after = int(seq or 0)
        except (TypeError, ValueError):
            after = 0
        with self._lock:
            return [entry for entry in self._buf if entry["seq"] > after]

    def head_seq(self) -> int:
        """Lowest seq currently in the buffer (0 if empty)."""
        with self._lock:
            return self._buf[0]["seq"] if self._buf else 0

    def latest_seq(self) -> int:
        """Highest seq assigned so far (cumulative, not buffer-relative)."""
        with self._lock:
            return self._seq

    def clear(self) -> None:
        with self._lock:
            self._buf.clear()
            # seq is NOT reset — callers polling with `since(seq)` would
            # otherwise see stale data treated as new after a clear.


# ── module-level singleton ──────────────────────────────────────────


_ring: LogRingBuffer | None = None
_ring_lock = threading.Lock()


def get_ring() -> LogRingBuffer:
    global _ring
    if _ring is None:
        with _ring_lock:
            if _ring is None:
                _ring = LogRingBuffer()
    return _ring


def reset_ring(capacity: int | None = None) -> None:
    """Replace the singleton ring. For tests."""
    global _ring
    with _ring_lock:
        _ring = LogRingBuffer(capacity=capacity)


# ── signal wiring (called from bootstrap) ──────────────────────────


def connect_signals(*emitters_with_attrs: tuple[Any, str, str, str]) -> None:
    """Wire each `(emitter, signal_attr, source_label, level)` tuple.

    Example:
        connect_signals(
            (chat_page, "sig_debug", "chat", "debug"),
            (chat_page, "sig_trace", "chat", "trace"),
            (engine,    "sig_trace", "engine", "trace"),
        )

    A signal that doesn't exist on the emitter is silently skipped —
    log_mirror is best-effort; consumers should not break because a
    Qt signal was renamed.
    """
    ring = get_ring()
    for emitter, attr, source, level in emitters_with_attrs:
        if emitter is None:
            continue
        signal = getattr(emitter, attr, None)
        if signal is None or not hasattr(signal, "connect"):
            continue
        try:
            signal.connect(_make_handler(ring, source, level))
        except Exception:
            # Best-effort; never raise from wiring.
            pass


def _make_handler(ring: LogRingBuffer, source: str, level: str) -> Callable[[str], None]:
    def _handler(text: str) -> None:
        try:
            ring.append(source, level, str(text or ""))
        except Exception:
            pass

    return _handler
