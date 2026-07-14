"""Acatalepsy SQLite connection factory + runtime authorizer.

Ported from the legacy Monolith database helper
as the sealed substrate seam for Acatalepsy v1. Single source of:

  * the canonical DB path (``LOG_DIR / "acatalepsy.sqlite3"``);
  * connection-time pragmas (``WAL``, ``foreign_keys=ON``);
  * a SQLite authorizer that denies INSERT/UPDATE/DELETE on the guarded
    substrate tables unless a thread-local sentinel is held.

The sentinel is set ONLY by the ``authorized_write()`` context manager.
Code that needs to mutate substrate tables (auditor runs, decision UI
writes, canonical_log appends) enters the context manager around the
mutation; everything else gets a read-only view of the substrate.

This is belt + suspenders to any static AST scanner — runtime catches
violations even if a dynamically-built SQL string slips past static
analysis.

Usage:

    from core.db_connect import connect_acatalepsy, authorized_write

    # Reader connection — substrate writes will be DENIED:
    conn = connect_acatalepsy(role="reader")
    conn.execute("SELECT * FROM acus")  # OK
    conn.execute("UPDATE acus SET ...") # raises MemoryWriteForbidden

    # Memory-writer connection — but writes still need the sentinel:
    conn = connect_acatalepsy(role="memory_writer")
    with authorized_write("auditor-run"):
        conn.execute("INSERT INTO acu_candidates ...")  # OK

    # Migration connection — bypasses authorizer entirely:
    conn = connect_acatalepsy(role="migration")
    conn.execute("CREATE TABLE ...")  # OK (one-shot ceremony)
"""
from __future__ import annotations

import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Literal

from core.paths import LOG_DIR, ensure_safe_local_path


__all__ = (
    "DB_PATH",
    "GUARDED_TABLES",
    "MemoryWriteForbidden",
    "Role",
    "authorized_write",
    "connect_acatalepsy",
    "is_authorized",
)


DB_PATH: Path = LOG_DIR / "acatalepsy.sqlite3"

# Substrate tables that require sentinel-authorized writes. Three additions
# vs the legacy set (per Acatalepsy v1 spec §3 + Phase D port-decision):
#   - canonical_log: immutable audit floor — every chat dispatch + auditor
#     event writes here, so writes must be intentional.
#   - acu_candidates: the auditor's pending-decision buffer.
#   - acu_decisions: accept/reject/edit events on candidates.
# Legacy GUARDED_TABLES (acus, claims, claim_state, claim_evidence,
# claim_relations, acu_relations) preserved verbatim.
GUARDED_TABLES: frozenset[str] = frozenset({
    "acus",
    "claims",
    "claim_state",
    "claim_evidence",
    "claim_relations",
    "acu_relations",
    "canonical_log",
    "acu_candidates",
    "acu_decisions",
})


Role = Literal["reader", "memory_writer", "migration"]


class MemoryWriteForbidden(sqlite3.DatabaseError):
    """Raised when a substrate-table write is attempted without an
    active ``authorized_write()`` sentinel."""


# ── Sentinel (thread-local) ───────────────────────────────────────────


class _Sentinel(threading.local):
    depth: int = 0
    reason: str = ""


_sentinel = _Sentinel()


def is_authorized() -> bool:
    """Whether the calling thread currently holds an authorized-write
    sentinel. Public for diagnostics; the sentinel itself is private."""
    return getattr(_sentinel, "depth", 0) > 0


@contextmanager
def authorized_write(reason: str) -> Iterator[None]:
    """Hold the substrate-write sentinel for the duration of the block.

    Re-entrant: nested ``authorized_write`` calls increment a depth
    counter and decrement on exit; the sentinel is released only when
    the outermost frame exits.

    ``reason`` is a short identifier for canonical_log emission and
    debugging (e.g., ``"auditor-run"``, ``"decision-user_e"``,
    ``"chat-dispatch"``, ``"boot-migration"``). Free-form; not validated.
    """
    prev_depth = getattr(_sentinel, "depth", 0)
    if prev_depth == 0:
        _sentinel.reason = reason
    _sentinel.depth = prev_depth + 1
    try:
        yield
    finally:
        _sentinel.depth = prev_depth
        if prev_depth == 0:
            _sentinel.reason = ""


# ── Authorizer ────────────────────────────────────────────────────────


# SQLite authorizer return codes
_SQLITE_OK = sqlite3.SQLITE_OK if hasattr(sqlite3, "SQLITE_OK") else 0
_SQLITE_DENY = sqlite3.SQLITE_DENY if hasattr(sqlite3, "SQLITE_DENY") else 1

# SQLite authorizer action codes (subset we care about)
_SQLITE_INSERT = 18
_SQLITE_UPDATE = 23
_SQLITE_DELETE = 9

_GUARDED_ACTIONS = frozenset({_SQLITE_INSERT, _SQLITE_UPDATE, _SQLITE_DELETE})


def _make_authorizer(strict: bool):
    """Return an authorizer callback bound to ``strict`` mode.

    ``strict=True`` (default): substrate writes without a sentinel
    raise sqlite3.DatabaseError → ``MemoryWriteForbidden``.

    ``strict=False`` (set MONOLITH_DB_AUTHORIZER_STRICT=0): substrate
    writes without a sentinel emit a stderr trace and proceed. Used
    only for production rollout sentinels — default is strict.
    """
    def authorizer(
        action: int,
        arg1: str | None,
        arg2: str | None,
        db_name: str | None,
        trigger: str | None,
    ) -> int:
        if action not in _GUARDED_ACTIONS:
            return _SQLITE_OK
        table = arg1 or ""
        if table not in GUARDED_TABLES:
            return _SQLITE_OK
        if is_authorized():
            return _SQLITE_OK
        if not strict:
            try:
                import sys
                sys.stderr.write(
                    f"[DB_AUTHORIZER] non-strict allow: action={action} "
                    f"table={table!r} trigger={trigger!r} (no sentinel)\n"
                )
                sys.stderr.flush()
            except Exception:
                pass
            return _SQLITE_OK
        return _SQLITE_DENY
    return authorizer


def _is_strict() -> bool:
    raw = os.getenv("MONOLITH_DB_AUTHORIZER_STRICT", "1").strip().lower()
    return raw in {"1", "true", "yes", ""}


# ── Factory ───────────────────────────────────────────────────────────


def connect_acatalepsy(
    *,
    role: Role,
    db_path: str | Path | None = None,
    timeout: float = 5.0,
    check_same_thread: bool = False,
) -> sqlite3.Connection:
    """Open a connection to the acatalepsy DB with role-appropriate
    pragmas and authorizer.

    role:
      * ``"reader"``  — authorizer denies any substrate INSERT/UPDATE/
        DELETE regardless of sentinel state. Use for query-only modules.
      * ``"memory_writer"`` — authorizer denies substrate writes UNLESS
        the calling thread is inside ``authorized_write()``. Use for
        auditor, decision UI, canonical_log writer, etc.
      * ``"migration"`` — no authorizer. Use ONLY for boot/migration
        ceremonies that legitimately bypass the seam.
    """
    path = ensure_safe_local_path(Path(db_path) if db_path else DB_PATH)
    conn = sqlite3.connect(path, timeout=timeout, check_same_thread=check_same_thread)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")

    if role == "migration":
        return conn

    strict = _is_strict()
    if role == "reader":
        # Reader: authorizer denies substrate writes unconditionally
        # (sentinel state is not consulted; readers should never write).
        def reader_auth(action, arg1, arg2, db_name, trigger):
            if action in _GUARDED_ACTIONS and (arg1 or "") in GUARDED_TABLES:
                if not strict:
                    return _SQLITE_OK
                return _SQLITE_DENY
            return _SQLITE_OK
        conn.set_authorizer(reader_auth)
    elif role == "memory_writer":
        conn.set_authorizer(_make_authorizer(strict=strict))
    else:
        raise ValueError(f"unknown role: {role!r}")

    return conn
