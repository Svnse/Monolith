"""Turn trace — causal visibility for Monolith.

Four layers, all in this module:
  Layer A — StageTraceRecord    : per-stage record of what each interceptor /
                                   prompt-assembly stage did on a turn.
  Layer B — FrameTraceRecord    : per-turn record of what the LLM saw
                                   (final messages snapshot, sizes, backend,
                                   resolved effort tier).
  Layer C — get_turn_trace etc. : joined query API for /trace + inspect_trace.
  Layer D — OutcomeTraceRecord  : post-turn outcome signals (thumbs up/down,
                                   copy, delete, regen, /rating with reason).
                                   Closes the causal loop — converts the
                                   trace from forensics into a learning signal.

All layers persist to a dedicated SQLite store at LOG_DIR/turn_trace.sqlite3.
The store is separate from EventLedger (already bifurcated; not adding writes
there). Join key is turn_id = Task.id UUID — already exists per-turn.

Spec: docs/specs/turn_trace_spec_v1.md
Flag: MONOLITH_TURN_TRACE_V1 (default ON). =0 disables writes; queries return
empty results gracefully.

Q1–Q7 defaults from spec §7 are baked in here:
  Q1: turn_id propagation via config["_turn_id"] (engine injects).
  Q2: tool followups get separate turn_ids; parent_turn_id populated.
  Q3: 500-char content preview.
  Q4: 4 KB hard cap on metadata json.
  Q5: sha256[:12] hash.
  Q6: inspect_trace tool is verb-dispatched (recent / errors / one).
  Q7: trace writes are best-effort; failure logs to stderr but never raises.
"""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import sys
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from core.paths import LOG_DIR


# ── flags + caps ─────────────────────────────────────────────────────


_FLAG_ENV = "MONOLITH_TURN_TRACE_V1"
_TTL_DAYS_ENV = "MONOLITH_TURN_TRACE_TTL_DAYS"
_DEFAULT_TTL_DAYS = 30

_PREVIEW_CHARS = 500
_METADATA_CAP_BYTES = 4096
_HASH_PREFIX_LEN = 12

_DB_PATH = LOG_DIR / "turn_trace.sqlite3"


# Layer-E fault_traces payload-format version. Stored per-row so consumers
# reading old rows can tell which payload schema applied at write time.
# Bump when the JSON shape of payload_json changes in a non-backward-
# compatible way for any event type. Backward-compatible additions (new
# optional fields) don't require a bump. Current schema = 1: every
# PipelineEvent subclass's to_payload() shape as of 2026-05.
FAULT_TRACES_PAYLOAD_SCHEMA_VERSION = 1


def _flag_enabled() -> bool:
    raw = str(os.environ.get(_FLAG_ENV, "1")).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _ttl_days() -> int:
    raw = os.environ.get(_TTL_DAYS_ENV)
    if not raw:
        return _DEFAULT_TTL_DAYS
    try:
        return max(1, int(raw))
    except (TypeError, ValueError):
        return _DEFAULT_TTL_DAYS


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── primitives ──────────────────────────────────────────────────────


class DropReason(str, Enum):
    DUPLICATE = "duplicate"
    FLAG_OFF = "flag_off"
    EMPTY = "empty"
    ERRORED = "errored"


class OutcomeKind(str, Enum):
    THUMBS_UP = "thumbs_up"
    THUMBS_DOWN = "thumbs_down"
    COPY = "copy"
    DELETE = "delete"
    REGEN = "regen"
    RATING = "rating"


def _hash(content: str) -> str:
    h = hashlib.sha256(content.encode("utf-8", errors="replace")).hexdigest()
    return h[:_HASH_PREFIX_LEN]


def _preview(content: str, limit: int = _PREVIEW_CHARS) -> str:
    text = str(content or "")
    text = text.replace("\n", " ").replace("\r", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "…"


def _bound_metadata(meta: dict | None) -> str:
    if not isinstance(meta, dict) or not meta:
        return "{}"
    try:
        encoded = json.dumps(meta, default=str, ensure_ascii=False)
    except Exception:
        return "{}"
    if len(encoded.encode("utf-8")) <= _METADATA_CAP_BYTES:
        return encoded
    # Cap exceeded. Rather than nuke the WHOLE dict (which threw out tiny
    # audit-critical fields like failure_tags / surface_note along with an
    # oversized think_block — making trainer-driven ratings read as tagless
    # in outcome_traces), drop the LARGEST string field(s) one at a time,
    # marking each, until we fit. Non-string fields (e.g. failure_tags lists)
    # and small strings survive. ``_truncated`` stays True for back-compat;
    # ``_truncated_fields`` records exactly what was dropped.
    orig_bytes = len(encoded.encode("utf-8"))
    reduced = dict(meta)
    dropped: list[str] = []
    while True:
        out = dict(reduced)
        out["_truncated"] = True
        out["_orig_bytes"] = orig_bytes
        if dropped:
            out["_truncated_fields"] = dropped
        try:
            enc = json.dumps(out, default=str, ensure_ascii=False)
        except Exception:
            return json.dumps({"_truncated": True, "_orig_bytes": orig_bytes})
        if len(enc.encode("utf-8")) <= _METADATA_CAP_BYTES:
            return enc
        candidates = sorted(
            (
                (len(str(v).encode("utf-8")), k)
                for k, v in reduced.items()
                if isinstance(v, str) and not str(v).startswith("[dropped:")
            ),
            reverse=True,
        )
        if not candidates:
            # Nothing left to shrink (non-string bloat) — last-resort stub.
            return json.dumps({"_truncated": True, "_orig_bytes": orig_bytes})
        big_bytes, big_key = candidates[0]
        reduced[big_key] = f"[dropped:{big_bytes}b]"
        dropped.append(big_key)


@dataclass(frozen=True)
class StageItem:
    kind: str
    content_hash: str
    content_preview: str
    source: str
    drop_reason: str | None = None

    @classmethod
    def added(cls, kind: str, content: str, source: str) -> "StageItem":
        return cls(
            kind=kind,
            content_hash=_hash(content),
            content_preview=_preview(content),
            source=source,
            drop_reason=None,
        )

    @classmethod
    def dropped(cls, kind: str, content: str, source: str, reason: str) -> "StageItem":
        return cls(
            kind=kind,
            content_hash=_hash(content),
            content_preview=_preview(content),
            source=source,
            drop_reason=reason,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "content_hash": self.content_hash,
            "content_preview": self.content_preview,
            "source": self.source,
            "drop_reason": self.drop_reason,
        }


@dataclass(frozen=True)
class StageTraceRecord:
    turn_id: str
    seq: int
    stage_name: str
    stage_kind: str  # "interceptor" | "prompt_stage"
    entered_at: str
    exited_at: str
    outcome: str  # "ran" | "skipped" | "errored"
    outcome_reason: str | None
    messages_in: int
    messages_out: int
    items_added: tuple[StageItem, ...] = ()
    items_dropped: tuple[StageItem, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_turn_id: str | None = None

    def __post_init__(self) -> None:
        if self.outcome not in {"ran", "skipped", "errored"}:
            raise ValueError(f"invalid outcome: {self.outcome!r}")
        if self.outcome == "ran" and self.outcome_reason is not None:
            raise ValueError(f"outcome=ran must not carry outcome_reason ({self.outcome_reason!r})")
        if self.outcome != "ran" and not self.outcome_reason:
            raise ValueError(f"outcome={self.outcome} must carry outcome_reason")
        if self.entered_at > self.exited_at:
            raise ValueError(f"entered_at > exited_at for {self.stage_name}")

    def to_payload(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "parent_turn_id": self.parent_turn_id,
            "seq": self.seq,
            "stage_name": self.stage_name,
            "stage_kind": self.stage_kind,
            "entered_at": self.entered_at,
            "exited_at": self.exited_at,
            "outcome": self.outcome,
            "outcome_reason": self.outcome_reason,
            "messages_in": self.messages_in,
            "messages_out": self.messages_out,
            "items_added": [it.to_dict() for it in self.items_added],
            "items_dropped": [it.to_dict() for it in self.items_dropped],
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FrameMessage:
    role: str
    content_hash: str
    content_preview: str
    content_chars: int
    ephemeral: bool
    source: str | None

    @classmethod
    def from_message(cls, msg: dict) -> "FrameMessage":
        content = str(msg.get("content", "") or "")
        return cls(
            role=str(msg.get("role", "") or "unknown"),
            content_hash=_hash(content),
            content_preview=_preview(content),
            content_chars=len(content),
            ephemeral=bool(msg.get("ephemeral", False)),
            source=msg.get("source") if isinstance(msg.get("source"), str) else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "role": self.role,
            "content_hash": self.content_hash,
            "content_preview": self.content_preview,
            "content_chars": self.content_chars,
            "ephemeral": self.ephemeral,
            "source": self.source,
        }


@dataclass(frozen=True)
class FrameTraceRecord:
    turn_id: str
    captured_at: str
    backend: str
    engine_key: str
    gen_id: int
    final_messages: tuple[FrameMessage, ...]
    system_prompt_chars: int
    user_prompt_chars: int
    total_chars: int
    config_snapshot: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    parent_turn_id: str | None = None
    effort_tier: str | None = None
    reasoning_mode: str | None = None
    prompts_applied: list[str] | None = None
    monothink_active: bool = False
    classification: dict[str, Any] | None = None  # TurnShape.to_dict() output; None until classifier runs

    def to_payload(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "parent_turn_id": self.parent_turn_id,
            "captured_at": self.captured_at,
            "backend": self.backend,
            "engine_key": self.engine_key,
            "gen_id": self.gen_id,
            "effort_tier": self.effort_tier,
            "reasoning_mode": self.reasoning_mode,
            "prompts_applied": list(self.prompts_applied) if self.prompts_applied else None,
            "monothink_active": self.monothink_active,
            "classification": dict(self.classification) if isinstance(self.classification, dict) else None,
            "system_prompt_chars": self.system_prompt_chars,
            "user_prompt_chars": self.user_prompt_chars,
            "total_chars": self.total_chars,
            "final_messages": [m.to_dict() for m in self.final_messages],
            "config_snapshot": dict(self.config_snapshot),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class OutcomeTraceRecord:
    """Layer D — a single outcome signal attached to a turn.

    Kinds:
      thumbs_up / thumbs_down — explicit binary feedback
      copy                    — user copied the assistant message (positive engagement)
      delete                  — user deleted the assistant message (negative)
      regen                   — user regenerated the response (negative)
      rating                  — explicit 0–100 score with optional reason

    `rating_value` is required when kind="rating" and forbidden otherwise.
    `reason` is optional free-text (the "why?" on /rating). Multiple records
    may exist per turn (e.g. user clicks copy then later regens).
    """
    turn_id: str
    recorded_at: str
    kind: str
    rating_value: int | None = None
    reason: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        valid_kinds = {k.value for k in OutcomeKind}
        if self.kind not in valid_kinds:
            raise ValueError(f"invalid outcome kind: {self.kind!r}")
        if self.kind == "rating":
            if self.rating_value is None:
                raise ValueError("kind=rating must carry rating_value")
            try:
                rv = int(self.rating_value)
            except (TypeError, ValueError):
                raise ValueError("rating_value must be an integer")
            if rv < 0 or rv > 100:
                raise ValueError(f"rating_value out of range [0, 100]: {rv}")
        elif self.rating_value is not None:
            raise ValueError(
                f"kind={self.kind} must not carry rating_value (only kind=rating does)"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "recorded_at": self.recorded_at,
            "kind": self.kind,
            "rating_value": self.rating_value,
            "reason": self.reason,
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class FaultTraceRecord:
    """Layer E — a single Turn Pipeline event record.

    Stores every event the pipeline bus publishes — kernel lifecycle events
    (TurnStreamStartedEvent, TurnCompleteEvent), policy emissions, and
    fault detections. Despite the name "fault", most rows are not faults;
    fault_kind and severity are NULL except on FaultDetectedEvent rows.

    seq is monotonic per turn (turn-scoped ordinal), assigned by the
    pipeline kernel at publish() time. payload carries the event-specific
    fields as a dict; the table stores it as JSON.

    Independence: this dataclass lives in turn_trace alongside the other
    layers because they share the same SQLite store and migration path.
    It does NOT couple Turn Pipeline to MonoBase — the consuming code is
    monokernel/turn_pipeline.py, not anything ACU/acatalepsy.
    """
    turn_id: str
    parent_turn_id: str | None
    seq: int
    emitted_at: str
    event_kind: str
    source_kind: str  # "producer" | "policy" | "kernel"
    source_name: str
    authority_tier: str | None = None  # "observation" | "mutation" | "dispatch" | None
    fault_kind: str | None = None
    severity: str | None = None        # "warn" | "hard" | None
    payload: dict[str, Any] = field(default_factory=dict)
    # Schema version of payload at write time. Set by record_fault() to
    # FAULT_TRACES_PAYLOAD_SCHEMA_VERSION. Readers fall back to 1 when
    # NULL (pre-versioning rows from before this column existed).
    payload_schema_version: int = FAULT_TRACES_PAYLOAD_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.source_kind not in {"producer", "policy", "kernel"}:
            raise ValueError(f"invalid source_kind: {self.source_kind!r}")
        if self.authority_tier is not None and self.authority_tier not in {
            "observation", "mutation", "dispatch"
        }:
            raise ValueError(f"invalid authority_tier: {self.authority_tier!r}")
        if self.severity is not None and self.severity not in {"warn", "hard"}:
            raise ValueError(f"invalid severity: {self.severity!r}")
        # fault_kind/severity must agree on null-ness
        if (self.fault_kind is None) != (self.severity is None):
            raise ValueError(
                "fault_kind and severity must both be set or both be None "
                f"(got fault_kind={self.fault_kind!r}, severity={self.severity!r})"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "turn_id": self.turn_id,
            "parent_turn_id": self.parent_turn_id,
            "seq": self.seq,
            "emitted_at": self.emitted_at,
            "event_kind": self.event_kind,
            "source_kind": self.source_kind,
            "source_name": self.source_name,
            "authority_tier": self.authority_tier,
            "fault_kind": self.fault_kind,
            "severity": self.severity,
            "payload": dict(self.payload),
        }


# ── joined-query types ─────────────────────────────────────────────


@dataclass(frozen=True)
class TurnTraceJoined:
    turn_id: str
    parent_turn_id: str | None
    stages: tuple[StageTraceRecord, ...]
    frame: FrameTraceRecord | None
    outcomes: tuple[OutcomeTraceRecord, ...]
    summary: dict[str, Any]


@dataclass(frozen=True)
class TurnTraceSummary:
    turn_id: str
    parent_turn_id: str | None
    captured_at: str
    backend: str
    stage_count: int
    errored_stage_count: int
    total_chars: int
    frame_present: bool


# ── store ──────────────────────────────────────────────────────────


_SCHEMA_VERSION = 4

_DDL = (
    """
    CREATE TABLE IF NOT EXISTS schema_version (
        version INTEGER PRIMARY KEY
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS stage_traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        turn_id TEXT NOT NULL,
        parent_turn_id TEXT,
        seq INTEGER NOT NULL,
        stage_name TEXT NOT NULL,
        stage_kind TEXT NOT NULL,
        entered_at TEXT NOT NULL,
        exited_at TEXT NOT NULL,
        outcome TEXT NOT NULL,
        outcome_reason TEXT,
        messages_in INTEGER NOT NULL,
        messages_out INTEGER NOT NULL,
        items_added_json TEXT NOT NULL DEFAULT '[]',
        items_dropped_json TEXT NOT NULL DEFAULT '[]',
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_stage_turn ON stage_traces(turn_id, seq)",
    "CREATE INDEX IF NOT EXISTS idx_stage_time ON stage_traces(entered_at)",
    """
    CREATE TABLE IF NOT EXISTS frame_traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        turn_id TEXT NOT NULL UNIQUE,
        parent_turn_id TEXT,
        captured_at TEXT NOT NULL,
        backend TEXT NOT NULL,
        engine_key TEXT NOT NULL,
        gen_id INTEGER NOT NULL,
        effort_tier TEXT,
        reasoning_mode TEXT,
        classification_json TEXT DEFAULT '{}',
        system_prompt_chars INTEGER NOT NULL,
        user_prompt_chars INTEGER NOT NULL,
        total_chars INTEGER NOT NULL,
        final_messages_json TEXT NOT NULL DEFAULT '[]',
        config_snapshot_json TEXT NOT NULL DEFAULT '{}',
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_frame_turn ON frame_traces(turn_id)",
    "CREATE INDEX IF NOT EXISTS idx_frame_time ON frame_traces(captured_at)",
    """
    CREATE TABLE IF NOT EXISTS outcome_traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        turn_id TEXT NOT NULL,
        recorded_at TEXT NOT NULL,
        kind TEXT NOT NULL,
        rating_value INTEGER,
        reason TEXT,
        metadata_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_outcome_turn ON outcome_traces(turn_id, recorded_at)",
    "CREATE INDEX IF NOT EXISTS idx_outcome_time ON outcome_traces(recorded_at)",
    "CREATE INDEX IF NOT EXISTS idx_outcome_kind ON outcome_traces(kind)",
    # ── Layer E — Fault & Intervention Trace (Turn Pipeline) ──────
    # Despite the table name, this stores ALL pipeline events, not just
    # faults. fault_kind / severity columns are nullable and only set on
    # FaultDetectedEvent rows. Every event published on the pipeline bus
    # produces exactly one row here, joined on turn_id with Layers A/B/C/D.
    """
    CREATE TABLE IF NOT EXISTS fault_traces (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        turn_id TEXT NOT NULL,
        parent_turn_id TEXT,
        seq INTEGER NOT NULL,
        emitted_at TEXT NOT NULL,
        event_kind TEXT NOT NULL,
        source_kind TEXT NOT NULL,
        source_name TEXT NOT NULL,
        authority_tier TEXT,
        fault_kind TEXT,
        severity TEXT,
        payload_json TEXT NOT NULL DEFAULT '{}',
        payload_schema_version INTEGER NOT NULL DEFAULT 1
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_fault_turn ON fault_traces(turn_id, seq)",
    "CREATE INDEX IF NOT EXISTS idx_fault_kind ON fault_traces(event_kind)",
    "CREATE INDEX IF NOT EXISTS idx_fault_time ON fault_traces(emitted_at)",
    "CREATE INDEX IF NOT EXISTS idx_fault_fault_kind ON fault_traces(fault_kind) WHERE fault_kind IS NOT NULL",
    # ── Stats addon ───────────────────────────────────────────────
    # daily_rollups: one row per UTC date with denormalized JSON columns
    # holding the day's aggregates. Materialized lazily by core/stats_rollup
    # on PageStats open; today's row is never stored (always computed live).
    """
    CREATE TABLE IF NOT EXISTS daily_rollups (
        date TEXT PRIMARY KEY,
        turns INTEGER NOT NULL,
        total_chars INTEGER NOT NULL,
        ratings_count INTEGER NOT NULL,
        ratings_sum INTEGER NOT NULL,
        ratings_histogram_json TEXT NOT NULL DEFAULT '{}',
        effort_distribution_json TEXT NOT NULL DEFAULT '{}',
        conversation_mode_dist_json TEXT NOT NULL DEFAULT '{}',
        reasoning_mode_dist_json TEXT NOT NULL DEFAULT '{}',
        linguency_mode_dist_json TEXT NOT NULL DEFAULT '{}',
        tool_usage_json TEXT NOT NULL DEFAULT '{}',
        fault_summary_json TEXT NOT NULL DEFAULT '{}',
        time_rhythm_json TEXT NOT NULL DEFAULT '{}',
        stage_latency_json TEXT NOT NULL DEFAULT '{}',
        computed_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_daily_rollups_date ON daily_rollups(date)",
    # achievements: bracket-tagged unlock events shown in PageStats'
    # AchievementFeed section. Inserted by core/stats_rollup.check_achievements.
    """
    CREATE TABLE IF NOT EXISTS achievements (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        unlocked_at TEXT NOT NULL,
        tag TEXT NOT NULL,
        description TEXT NOT NULL,
        payload_json TEXT NOT NULL DEFAULT '{}'
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_achievements_unlocked_at ON achievements(unlocked_at)",
    "CREATE INDEX IF NOT EXISTS idx_achievements_tag ON achievements(tag)",
)


def _migrate(conn: sqlite3.Connection) -> None:
    """Idempotent column adds for DBs created under older schema versions."""
    try:
        cur = conn.execute("PRAGMA table_info(frame_traces)")
        cols = {row[1] for row in cur.fetchall()}
    except sqlite3.Error:
        return
    if "effort_tier" not in cols:
        try:
            conn.execute("ALTER TABLE frame_traces ADD COLUMN effort_tier TEXT")
        except sqlite3.Error:
            pass
    if "classification_json" not in cols:
        try:
            conn.execute(
                "ALTER TABLE frame_traces ADD COLUMN classification_json TEXT DEFAULT '{}'"
            )
        except sqlite3.Error:
            pass
    if "reasoning_mode" not in cols:
        try:
            conn.execute("ALTER TABLE frame_traces ADD COLUMN reasoning_mode TEXT")
        except sqlite3.Error:
            pass
    if "prompts_applied" not in cols:
        try:
            conn.execute("ALTER TABLE frame_traces ADD COLUMN prompts_applied TEXT")
        except sqlite3.Error:
            pass
    if "monothink_active" not in cols:
        try:
            conn.execute("ALTER TABLE frame_traces ADD COLUMN monothink_active INTEGER DEFAULT 0")
        except sqlite3.Error:
            pass

    # fault_traces.payload_schema_version: per-row payload-format tag.
    # Old rows get NULL; readers treat NULL as "schema 1" (pre-versioning).
    # Future writes set the column explicitly.
    try:
        cur = conn.execute("PRAGMA table_info(fault_traces)")
        fault_cols = {row[1] for row in cur.fetchall()}
    except sqlite3.Error:
        return
    if "payload_schema_version" not in fault_cols:
        try:
            conn.execute(
                "ALTER TABLE fault_traces ADD COLUMN payload_schema_version INTEGER"
            )
        except sqlite3.Error:
            pass


_db_lock = threading.Lock()
_conn: sqlite3.Connection | None = None
_db_path_override: Path | None = None


def _get_db_path() -> Path:
    return _db_path_override if _db_path_override is not None else _DB_PATH


def set_db_path(path: Path | None) -> None:
    """Override the DB path (for tests). None resets to default."""
    global _conn, _db_path_override
    with _db_lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None
        _db_path_override = path


def _get_conn() -> sqlite3.Connection | None:
    global _conn
    if _conn is not None:
        return _conn
    path = _get_db_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
        _conn.row_factory = sqlite3.Row
        for stmt in _DDL:
            _conn.execute(stmt)
        _migrate(_conn)
        cur = _conn.execute("SELECT version FROM schema_version LIMIT 1")
        row = cur.fetchone()
        if row is None:
            _conn.execute("INSERT INTO schema_version(version) VALUES(?)", (_SCHEMA_VERSION,))
        elif int(row[0]) < _SCHEMA_VERSION:
            _conn.execute("UPDATE schema_version SET version = ?", (_SCHEMA_VERSION,))
        return _conn
    except Exception as exc:
        _trace_failure(f"open store failed: {exc}")
        _conn = None
        return None


def _trace_failure(msg: str) -> None:
    """Log a best-effort write failure to stderr (Q7)."""
    try:
        sys.stderr.write(f"[turn_trace] {msg}\n")
    except Exception:
        pass


def close_store() -> None:
    global _conn
    with _db_lock:
        if _conn is not None:
            try:
                _conn.close()
            except Exception:
                pass
            _conn = None


# ── write paths (best-effort, never raise) ────────────────────────


def record_stage(record: StageTraceRecord) -> None:
    """Persist a Layer A record. Best-effort; logs on failure."""
    if not _flag_enabled():
        return
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return
            conn.execute(
                """
                INSERT INTO stage_traces(
                    turn_id, parent_turn_id, seq, stage_name, stage_kind,
                    entered_at, exited_at, outcome, outcome_reason,
                    messages_in, messages_out,
                    items_added_json, items_dropped_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.turn_id,
                    record.parent_turn_id,
                    int(record.seq),
                    record.stage_name,
                    record.stage_kind,
                    record.entered_at,
                    record.exited_at,
                    record.outcome,
                    record.outcome_reason,
                    int(record.messages_in),
                    int(record.messages_out),
                    json.dumps([it.to_dict() for it in record.items_added]),
                    json.dumps([it.to_dict() for it in record.items_dropped]),
                    _bound_metadata(record.metadata),
                ),
            )
    except Exception as exc:
        _trace_failure(f"record_stage failed: {exc}")


def record_frame(record: FrameTraceRecord) -> None:
    """Persist a Layer B record. Best-effort; logs on failure."""
    if not _flag_enabled():
        return
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return
            classification_json = (
                json.dumps(record.classification, ensure_ascii=False)
                if isinstance(record.classification, dict) else "{}"
            )
            prompts_json = (
                json.dumps(record.prompts_applied, ensure_ascii=False)
                if record.prompts_applied else None
            )
            conn.execute(
                """
                INSERT OR REPLACE INTO frame_traces(
                    turn_id, parent_turn_id, captured_at, backend, engine_key, gen_id,
                    effort_tier, reasoning_mode, prompts_applied, monothink_active,
                    classification_json,
                    system_prompt_chars, user_prompt_chars, total_chars,
                    final_messages_json, config_snapshot_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.turn_id,
                    record.parent_turn_id,
                    record.captured_at,
                    record.backend,
                    record.engine_key,
                    int(record.gen_id),
                    record.effort_tier,
                    record.reasoning_mode,
                    prompts_json,
                    1 if record.monothink_active else 0,
                    classification_json,
                    int(record.system_prompt_chars),
                    int(record.user_prompt_chars),
                    int(record.total_chars),
                    json.dumps([m.to_dict() for m in record.final_messages]),
                    _bound_metadata(record.config_snapshot),
                    _bound_metadata(record.metadata),
                ),
            )
    except Exception as exc:
        _trace_failure(f"record_frame failed: {exc}")


def record_source_tier(
    turn_id: str,
    source_tier: str,
    region_tiers: dict[str, str] | None = None,
) -> None:
    """Additively stamp the Source-Tier Gate result onto an existing frame
    row's metadata_json (targeted UPDATE — the frame is written pre-generation
    by record_frame, so the tier is a follow-up write, not a fold into that
    call). Best-effort; double-gated by the store flag and the feature flag.
    Never raises into the caller.
    """
    if not _flag_enabled():
        return
    from core.source_tier import source_tier_enabled
    if not source_tier_enabled():
        return
    if not turn_id or not source_tier:
        return
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return
            row = conn.execute(
                "SELECT metadata_json FROM frame_traces WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if row is None:
                return  # frame not written (substrate-only turn / flag race) — skip
            try:
                meta = json.loads(row["metadata_json"] or "{}")
                if not isinstance(meta, dict):
                    meta = {}
            except (TypeError, ValueError):
                meta = {}
            meta["source_tier"] = source_tier
            if region_tiers:
                meta["region_tiers"] = dict(region_tiers)
            conn.execute(
                "UPDATE frame_traces SET metadata_json = ? WHERE turn_id = ?",
                (_bound_metadata(meta), turn_id),
            )
    except Exception as exc:  # noqa: BLE001 — best-effort, never break a turn
        _trace_failure(f"record_source_tier failed: {exc}")


def record_grounded_verdict(turn_id: str, verdict: dict | None) -> None:
    """Additively stamp the V1 grounded-verdict onto the turn's frame
    metadata_json (a follow-up UPDATE, mirroring record_source_tier — the frame
    is written pre-generation). NON-PERFORMATIVE telemetry: the verdict is never
    injected back into the model (it already self-cites). Best-effort,
    double-gated by the store flag and the feature flag; never raises.
    """
    if not _flag_enabled():
        return
    from core.grounded_verdict import grounded_verdict_enabled
    if not grounded_verdict_enabled():
        return
    if not turn_id or not isinstance(verdict, dict):
        return
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return
            row = conn.execute(
                "SELECT metadata_json FROM frame_traces WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            if row is None:
                return  # frame not written (substrate-only turn / flag race) — skip
            try:
                meta = json.loads(row["metadata_json"] or "{}")
                if not isinstance(meta, dict):
                    meta = {}
            except (TypeError, ValueError):
                meta = {}
            meta["grounded_verdict"] = dict(verdict)
            conn.execute(
                "UPDATE frame_traces SET metadata_json = ? WHERE turn_id = ?",
                (_bound_metadata(meta), turn_id),
            )
    except Exception as exc:  # noqa: BLE001 — best-effort, never break a turn
        _trace_failure(f"record_grounded_verdict failed: {exc}")


def record_outcome(record: OutcomeTraceRecord) -> None:
    """Persist a Layer D record. Best-effort; logs on failure.

    Multiple outcomes per turn are allowed (e.g. user copies then later regens).

    Side hook: when ``record.kind == "rating"`` and the insert succeeded, fires
    :func:`core.monothink.maybe_evolve_after_rating` outside the DB lock so the
    LLM call (sync) doesn't block other writers. The hook handles its own
    monothink-tier gating, idempotence, and error isolation — failures here
    only emit a trace line, they never propagate.
    """
    if not _flag_enabled():
        return
    inserted = False
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return
            conn.execute(
                """
                INSERT INTO outcome_traces(
                    turn_id, recorded_at, kind, rating_value, reason, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    record.turn_id,
                    record.recorded_at,
                    record.kind,
                    None if record.rating_value is None else int(record.rating_value),
                    record.reason,
                    _bound_metadata(record.metadata),
                ),
            )
            inserted = True
    except Exception as exc:
        _trace_failure(f"record_outcome failed: {exc}")
        return

    # ── MonoThink evolution hook (rating outcomes only) ────────────────
    # SP1: evolution is driven by a closed `failure_tags` enum, not free text.
    # Thumbs ↑/↓ are CUT from the evolution trigger entirely — they are still
    # RECORDED as outcomes above this hook (kind=thumbs_up/down stays a clean,
    # rating_value=NULL signal), but they no longer fire maybe_evolve_after_rating.
    # Only kind=="rating" reaches the decider, and only when the rating carried
    # at least one VALID failure_tag (the triviality gate — a tag-less rating
    # carries no directional signal and must not mutate the scaffold).
    if inserted and record.kind == "rating":
        try:
            # failure_tags ride in the outcome metadata (stashed by
            # chat.py:_record_turn_outcome). Normalize + gate: a rating that
            # normalizes to zero valid tags does not drive evolution.
            from core.failure_tags import normalize_tags
            _tags = []
            if isinstance(record.metadata, dict):
                _raw_tags = record.metadata.get("failure_tags")
                if isinstance(_raw_tags, list):
                    _tags = normalize_tags([str(t) for t in _raw_tags])
            if _tags:
                # think_block (the rated turn's <think>...</think> reasoning
                # trace) is stashed by chat.py:_record_turn_outcome into outcome
                # metadata so the model sees *how* it thought alongside the
                # rating signal when proposing scaffold edits.
                _think = None
                _replay = None
                _note = None
                if isinstance(record.metadata, dict):
                    _rawth = record.metadata.get("think_block")
                    if isinstance(_rawth, str) and _rawth.strip():
                        _think = _rawth
                    _raw_replay = record.metadata.get("replay_input")
                    if isinstance(_raw_replay, str) and _raw_replay.strip():
                        _replay = _raw_replay
                    _raw_note = record.metadata.get("surface_note")
                    if isinstance(_raw_note, str) and _raw_note.strip():
                        _note = _raw_note
                from core.monothink import maybe_evolve_after_rating
                if _replay is not None or _note is not None:
                    maybe_evolve_after_rating(
                        record.turn_id, record.rating_value, _tags,
                        think_block=_think, replay_input=_replay,
                        rater_note=_note,
                    )
                else:
                    maybe_evolve_after_rating(
                        record.turn_id, record.rating_value, _tags,
                        think_block=_think,
                    )
        except Exception as exc:
            _trace_failure(f"monothink hook failed: {exc}")

        # ── Identity-emergence detector (feedback heartbeat) ───────────────
        # Deterministic, no LLM, watermark-throttled. Ships dark (its own flag
        # default OFF). Surfaces a read-only advisory when high-confidentity
        # self-derived claims have accrued; NEVER proposes/applies here
        # (propose-only — the bidden identity_review skill drafts, E applies).
        # Isolated like the monothink hook so it can never break the rating loop.
        try:
            from core.identity_emergence import detect_emergence_best_effort
            detect_emergence_best_effort()
        except Exception as exc:
            _trace_failure(f"identity emergence hook failed: {exc}")

        # ── Curiosity detector (M3 feedback heartbeat) ─────────────────────
        # The fresh disposition of the identity signal. Deterministic, no LLM,
        # retirement-bounded. Ships dark (MONOLITH_CURIOSITY_V1 default OFF).
        # Propose-only: surfaces pulls; never pursues. Isolated.
        try:
            from core.curiosity import detect_pulls
            detect_pulls()
        except Exception as exc:
            _trace_failure(f"curiosity hook failed: {exc}")


def record_fault(record: FaultTraceRecord) -> None:
    """Persist a Layer E record. Best-effort; logs on failure.

    Called by the Turn Pipeline kernel for every event it publishes —
    lifecycle events, policy emissions, and fault detections all flow
    through here. seq is assigned by the kernel before this is called.
    """
    if not _flag_enabled():
        return
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return
            conn.execute(
                """
                INSERT INTO fault_traces(
                    turn_id, parent_turn_id, seq, emitted_at,
                    event_kind, source_kind, source_name,
                    authority_tier, fault_kind, severity, payload_json,
                    payload_schema_version
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.turn_id,
                    record.parent_turn_id,
                    int(record.seq),
                    record.emitted_at,
                    record.event_kind,
                    record.source_kind,
                    record.source_name,
                    record.authority_tier,
                    record.fault_kind,
                    record.severity,
                    _bound_metadata(record.payload),
                    int(record.payload_schema_version),
                ),
            )
    except Exception as exc:
        _trace_failure(f"record_fault failed: {exc}")


# ── query API ──────────────────────────────────────────────────────


def _row_to_stage(row: sqlite3.Row) -> StageTraceRecord:
    items_added_raw = json.loads(row["items_added_json"] or "[]")
    items_dropped_raw = json.loads(row["items_dropped_json"] or "[]")
    metadata_raw = json.loads(row["metadata_json"] or "{}")
    return StageTraceRecord(
        turn_id=row["turn_id"],
        parent_turn_id=row["parent_turn_id"],
        seq=int(row["seq"]),
        stage_name=row["stage_name"],
        stage_kind=row["stage_kind"],
        entered_at=row["entered_at"],
        exited_at=row["exited_at"],
        outcome=row["outcome"],
        outcome_reason=row["outcome_reason"],
        messages_in=int(row["messages_in"]),
        messages_out=int(row["messages_out"]),
        items_added=tuple(StageItem(**it) for it in items_added_raw),
        items_dropped=tuple(StageItem(**it) for it in items_dropped_raw),
        metadata=metadata_raw if isinstance(metadata_raw, dict) else {},
    )


def _row_to_frame(row: sqlite3.Row) -> FrameTraceRecord:
    msgs_raw = json.loads(row["final_messages_json"] or "[]")
    config_raw = json.loads(row["config_snapshot_json"] or "{}")
    metadata_raw = json.loads(row["metadata_json"] or "{}")
    # effort_tier / reasoning_mode / classification may be missing on rows from
    # pre-migration DBs.
    try:
        effort_tier = row["effort_tier"]
    except (IndexError, KeyError):
        effort_tier = None
    try:
        reasoning_mode = row["reasoning_mode"]
    except (IndexError, KeyError):
        reasoning_mode = None
    classification: dict[str, Any] | None
    try:
        classification_raw = row["classification_json"]
    except (IndexError, KeyError):
        classification_raw = None
    if classification_raw:
        try:
            parsed = json.loads(classification_raw)
            classification = parsed if isinstance(parsed, dict) and parsed else None
        except (TypeError, ValueError):
            classification = None
    else:
        classification = None
    return FrameTraceRecord(
        turn_id=row["turn_id"],
        parent_turn_id=row["parent_turn_id"],
        captured_at=row["captured_at"],
        backend=row["backend"],
        engine_key=row["engine_key"],
        gen_id=int(row["gen_id"]),
        effort_tier=effort_tier if isinstance(effort_tier, str) else None,
        reasoning_mode=reasoning_mode if isinstance(reasoning_mode, str) else None,
        classification=classification,
        system_prompt_chars=int(row["system_prompt_chars"]),
        user_prompt_chars=int(row["user_prompt_chars"]),
        total_chars=int(row["total_chars"]),
        final_messages=tuple(FrameMessage(**m) for m in msgs_raw),
        config_snapshot=config_raw if isinstance(config_raw, dict) else {},
        metadata=metadata_raw if isinstance(metadata_raw, dict) else {},
    )


def _row_to_outcome(row: sqlite3.Row) -> OutcomeTraceRecord:
    metadata_raw = json.loads(row["metadata_json"] or "{}")
    rv = row["rating_value"]
    return OutcomeTraceRecord(
        turn_id=row["turn_id"],
        recorded_at=row["recorded_at"],
        kind=row["kind"],
        rating_value=int(rv) if rv is not None else None,
        reason=row["reason"],
        metadata=metadata_raw if isinstance(metadata_raw, dict) else {},
    )


def _row_to_fault(row: sqlite3.Row) -> FaultTraceRecord:
    payload_raw = json.loads(row["payload_json"] or "{}")
    # Old rows (pre-versioning) have NULL payload_schema_version. Treat
    # NULL as 1 since version 1 is the schema those rows were written
    # under. Missing column entirely (PRAGMA-less reader on a very old
    # DB) also falls back to 1.
    try:
        raw_version = row["payload_schema_version"]
    except (IndexError, KeyError):
        raw_version = None
    schema_version = int(raw_version) if raw_version is not None else 1
    return FaultTraceRecord(
        turn_id=row["turn_id"],
        parent_turn_id=row["parent_turn_id"],
        seq=int(row["seq"]),
        emitted_at=row["emitted_at"],
        event_kind=row["event_kind"],
        source_kind=row["source_kind"],
        source_name=row["source_name"],
        authority_tier=row["authority_tier"],
        fault_kind=row["fault_kind"],
        severity=row["severity"],
        payload=payload_raw if isinstance(payload_raw, dict) else {},
        payload_schema_version=schema_version,
    )


def list_pipeline_events(turn_id: str) -> list[FaultTraceRecord]:
    """Return every Layer E event for a turn, ordered by seq.

    Used by the Pipeline Inspector and inspect_pipeline tool. Empty list on
    no rows or store unavailable.
    """
    if not _flag_enabled():
        return []
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return []
            rows = list(conn.execute(
                """
                SELECT turn_id, parent_turn_id, seq, emitted_at,
                       event_kind, source_kind, source_name,
                       authority_tier, fault_kind, severity, payload_json,
                       payload_schema_version
                FROM fault_traces
                WHERE turn_id = ?
                ORDER BY seq ASC
                """,
                (str(turn_id),),
            ))
    except Exception as exc:
        _trace_failure(f"list_pipeline_events failed: {exc}")
        return []
    return [_row_to_fault(r) for r in rows]


def most_recent_pipeline_turn_id() -> str | None:
    """Return the turn_id with the most recent fault_traces row, or None.

    Used by /pipeline-last and inspect_pipeline(verb='last') so they work
    even on substrate-only turns (no frame_traces row exists for those).
    """
    if not _flag_enabled():
        return None
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return None
            row = conn.execute(
                "SELECT turn_id FROM fault_traces ORDER BY id DESC LIMIT 1"
            ).fetchone()
    except Exception as exc:
        _trace_failure(f"most_recent_pipeline_turn_id failed: {exc}")
        return None
    return row["turn_id"] if row is not None else None


def list_faults_since(
    since_iso: str,
    *,
    fault_kind: str | None = None,
    limit: int = 200,
) -> list[FaultTraceRecord]:
    """Return FaultDetectedEvent rows newer than *since_iso*.

    Filters to rows where fault_kind IS NOT NULL (so this is the *fault*
    view, not the full event stream). Optionally restrict to a single
    fault_kind. Newest-first, capped at *limit*.

    Used by /pipeline faults and Overseer's slow-burn drift surface.
    """
    if not _flag_enabled():
        return []
    sql = (
        "SELECT turn_id, parent_turn_id, seq, emitted_at, event_kind, "
        "source_kind, source_name, authority_tier, fault_kind, severity, "
        "payload_json, payload_schema_version FROM fault_traces "
        "WHERE fault_kind IS NOT NULL AND emitted_at >= ?"
    )
    params: list[Any] = [str(since_iso)]
    if fault_kind:
        sql += " AND fault_kind = ?"
        params.append(str(fault_kind))
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(max(1, int(limit)))
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return []
            rows = list(conn.execute(sql, tuple(params)))
    except Exception as exc:
        _trace_failure(f"list_faults_since failed: {exc}")
        return []
    return [_row_to_fault(r) for r in rows]


def get_last_verification_result() -> dict[str, Any] | None:
    """Most-recent verifier verdict, for the Self-Check Loop (read side).

    The verdict is already durable: ``verifier_bridge`` publishes a
    ``VerifierVerdictEvent`` every turn and the kernel persists every event to
    ``fault_traces`` (``monokernel/turn_pipeline.py`` ->
    ``_record_event_to_fault_traces``). This returns the latest such row,
    parsed — NOT a new persistence seam.

    Returns ``{"turn_id", "emitted_at", "verdict", "findings"}`` or ``None``
    when there is no verdict row, the flag is off, or the payload carries no
    verdict. Best-effort; never raises.
    """
    if not _flag_enabled():
        return None
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return None
            row = conn.execute(
                "SELECT turn_id, emitted_at, payload_json FROM fault_traces "
                "WHERE event_kind = 'VerifierVerdictEvent' "
                "ORDER BY id DESC LIMIT 1"
            ).fetchone()
    except Exception as exc:
        _trace_failure(f"get_last_verification_result failed: {exc}")
        return None
    if row is None:
        return None
    turn_id, emitted_at, payload_json = row[0], row[1], row[2]
    try:
        payload = json.loads(payload_json) if payload_json else {}
    except Exception:
        payload = {}
    verdict = payload.get("verdict")
    if not verdict:
        return None
    findings = payload.get("findings")
    if not isinstance(findings, list):
        findings = []
    return {
        "turn_id": turn_id,
        "emitted_at": emitted_at,
        "verdict": verdict,
        "findings": findings,
    }


def recent_ratings_summary(*, window: int = 10) -> dict[str, Any]:
    """Aggregate recent rating outcomes for telemetry injection.

    Reads up to *window* most recent rating outcomes (kind='rating') and
    returns rolling average, the recent values (oldest→newest), and the
    worst/best within the window with their reasons.

    Best-effort. Returns the empty shape on read failure or when no ratings
    exist; never raises.

    Returns:
        {
            "count":       <int — rated turns counted (may be < window)>,
            "rolling_avg": <float — average of recent values>,
            "recent":      [<int>, ...]  # oldest→newest within window,
            "worst":       {"turn_id": str, "value": int, "reason": str | None} | None,
            "best":        {"turn_id": str, "value": int, "reason": str | None} | None,
        }
    """
    empty = {
        "count": 0, "rolling_avg": 0.0, "recent": [],
        "worst": None, "best": None,
    }
    if not _flag_enabled():
        return empty
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return empty
            rows = list(conn.execute(
                """
                SELECT turn_id, rating_value, reason FROM outcome_traces
                WHERE kind = 'rating' AND rating_value IS NOT NULL
                ORDER BY id DESC LIMIT ?
                """,
                (max(1, int(window)),),
            ))
    except Exception as exc:
        _trace_failure(f"recent_ratings_summary failed: {exc}")
        return empty
    if not rows:
        return empty
    # Rows are newest-first; reverse to oldest-first for downstream display.
    samples = [
        (str(r["turn_id"]), int(r["rating_value"]), r["reason"])
        for r in rows
    ]
    samples.reverse()
    values = [s[1] for s in samples]
    rolling_avg = sum(values) / len(values)
    worst = min(samples, key=lambda s: s[1])
    best = max(samples, key=lambda s: s[1])
    return {
        "count": len(samples),
        "rolling_avg": rolling_avg,
        "recent": values,
        "worst": {"turn_id": worst[0], "value": worst[1], "reason": worst[2]},
        "best":  {"turn_id":  best[0], "value":  best[1], "reason":  best[2]},
    }


@dataclass(frozen=True)
class OutcomeReadRow:
    """A read view of one ``outcome_traces`` row — like :class:`OutcomeTraceRecord`
    but carrying the DB ``id`` (the stable per-row key the monosearch adapter needs
    for namespaced ids and get-by-id). ``metadata`` is the parsed dict."""
    id: int
    turn_id: str
    recorded_at: str
    kind: str
    rating_value: int | None
    reason: str | None
    metadata: dict[str, Any]


_OUTCOME_READ_COLS = "id, turn_id, recorded_at, kind, rating_value, reason, metadata_json"


def _row_to_outcome_read(row: sqlite3.Row) -> OutcomeReadRow:
    try:
        meta = json.loads(row["metadata_json"] or "{}")
    except Exception:
        meta = {}
    rv = row["rating_value"]
    return OutcomeReadRow(
        id=int(row["id"]),
        turn_id=row["turn_id"],
        recorded_at=row["recorded_at"],
        kind=row["kind"],
        rating_value=int(rv) if rv is not None else None,
        reason=row["reason"],
        metadata=meta if isinstance(meta, dict) else {},
    )


def read_recent_outcomes(
    limit: int = 20, *, since: str | None = None, keyword: str | None = None
) -> list[OutcomeReadRow]:
    """Recent outcome rows, newest-first (id DESC). Optional ISO ``since`` cutoff
    and case-insensitive ``keyword`` substring — matched against the RAW stored
    ``reason`` and ``metadata_json`` strings, so a failure_tag like
    'premise_unchecked' is found via its serialized form (not a re-parsed dict).

    Best-effort; returns [] on read failure or when the store is disabled.
    """
    if limit <= 0:
        return []
    clauses: list[str] = []
    params: list = []
    if since:
        clauses.append("recorded_at >= ?")
        params.append(since)
    if keyword:
        kw = f"%{keyword.lower()}%"
        clauses.append("(lower(reason) LIKE ? OR lower(metadata_json) LIKE ?)")
        params.extend([kw, kw])
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT {_OUTCOME_READ_COLS} FROM outcome_traces{where} ORDER BY id DESC LIMIT ?"
    params.append(int(limit))
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return []
            rows = list(conn.execute(sql, tuple(params)))
    except Exception as exc:
        _trace_failure(f"read_recent_outcomes failed: {exc}")
        return []
    return [_row_to_outcome_read(r) for r in rows]


def read_outcome(outcome_id: int) -> OutcomeReadRow | None:
    """One outcome row by its DB id, or None if absent / on read failure."""
    try:
        oid = int(outcome_id)
    except (TypeError, ValueError):
        return None
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return None
            row = conn.execute(
                f"SELECT {_OUTCOME_READ_COLS} FROM outcome_traces WHERE id = ?",
                (oid,),
            ).fetchone()
    except Exception as exc:
        _trace_failure(f"read_outcome failed: {exc}")
        return None
    return _row_to_outcome_read(row) if row is not None else None


def list_outcomes_for_turn(turn_id: str) -> list[OutcomeTraceRecord]:
    """Return outcomes for *turn_id* in the order they were recorded."""
    if not turn_id:
        return []
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return []
            rows = list(conn.execute(
                "SELECT * FROM outcome_traces WHERE turn_id = ? ORDER BY id",
                (turn_id,),
            ))
    except Exception as exc:
        _trace_failure(f"list_outcomes_for_turn failed: {exc}")
        return []
    return [_row_to_outcome(r) for r in rows]


def get_turn_effort_tier(turn_id: str) -> str | None:
    """Public read: ``effort_tier`` for *turn_id* from ``frame_traces``.

    Cheap single-column lookup. Returns ``None`` on any error or when the
    turn has no frame row.
    """
    if not turn_id:
        return None
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return None
            row = conn.execute(
                "SELECT effort_tier FROM frame_traces WHERE turn_id = ? LIMIT 1",
                (str(turn_id),),
            ).fetchone()
    except Exception as exc:
        _trace_failure(f"get_turn_effort_tier failed: {exc}")
        return None
    if row is None:
        return None
    val = row[0]
    return str(val) if val else None


def get_turn_reasoning_mode(turn_id: str) -> str | None:
    """Public read: ``reasoning_mode`` for *turn_id* from ``frame_traces``.

    Cheap single-column lookup used by ``core.monothink`` to gate its
    bounded-autonomy evolution trigger. Returns ``None`` on any error,
    when the turn has no frame row, or when reasoning_mode was unset
    for that turn (default opt-in state).
    """
    if not turn_id:
        return None
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return None
            row = conn.execute(
                "SELECT reasoning_mode FROM frame_traces WHERE turn_id = ? LIMIT 1",
                (str(turn_id),),
            ).fetchone()
    except Exception as exc:
        _trace_failure(f"get_turn_reasoning_mode failed: {exc}")
        return None
    if row is None:
        return None
    val = row[0]
    return str(val) if val else None


def get_turn_monothink_active(turn_id: str) -> bool:
    """Public read: whether monothink was active for *turn_id*.

    Used by ``core.monothink`` to gate its evolution trigger (replaces the
    old ``get_turn_reasoning_mode`` gate after the /prompt consolidation).
    """
    if not turn_id:
        return False
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return False
            row = conn.execute(
                "SELECT monothink_active FROM frame_traces WHERE turn_id = ? LIMIT 1",
                (str(turn_id),),
            ).fetchone()
    except Exception:
        return False
    if row is None:
        return False
    return bool(row[0])


def set_turn_monothink_active(turn_id: str, active: bool = True) -> bool:
    """Follow-up UPDATE: stamp ``monothink_active`` on an already-recorded frame
    (the record_source_tier pattern — the frame is written by its producer, the
    stamp is a targeted after-write).

    Exists for thinkpad branch turns: ``run_subagent`` records their frames with
    the default ``monothink_active=False``, but a branch genuinely runs the
    monothink scaffold in-frame — without the stamp, rating a branch fails the
    ``will_evolve`` gate. Returns True iff a row was stamped (honest no-op on an
    unknown turn).
    """
    if not turn_id:
        return False
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return False
            cur = conn.execute(
                "UPDATE frame_traces SET monothink_active = ? WHERE turn_id = ?",
                (1 if active else 0, str(turn_id)),
            )
            conn.commit()
            return cur.rowcount > 0
    except Exception:
        return False


def latest_outer_turn() -> dict | None:
    """Return ``{"turn_id", "monothink_active"}`` for the most recently recorded
    OUTER turn (``parent_turn_id IS NULL``), or ``None`` if there are no frames.

    Used by the agent server's /chat handler to tell a caller which turn it just
    produced — and whether that turn is monothink-trainable — so the caller can
    rate that exact turn_id via /rating. Tool-followup turns (which carry a
    parent_turn_id) are excluded; only the user-facing outer turn is returned.
    Safe under the agent server's single-pending-request serialization: between
    on_message and push_done exactly one outer turn is in flight.
    """
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return None
            row = conn.execute(
                "SELECT turn_id, monothink_active FROM frame_traces "
                "WHERE parent_turn_id IS NULL "
                "ORDER BY captured_at DESC, ROWID DESC LIMIT 1",
            ).fetchone()
    except Exception as exc:
        _trace_failure(f"latest_outer_turn failed: {exc}")
        return None
    if row is None:
        return None
    return {"turn_id": str(row[0]), "monothink_active": bool(row[1])}


def get_turn_trace(turn_id: str) -> TurnTraceJoined | None:
    """Return joined Layers A + B + D for *turn_id*, or None if not found."""
    if not turn_id:
        return None
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return None
            stage_rows = list(conn.execute(
                "SELECT * FROM stage_traces WHERE turn_id = ? ORDER BY seq",
                (turn_id,),
            ))
            frame_row = conn.execute(
                "SELECT * FROM frame_traces WHERE turn_id = ?",
                (turn_id,),
            ).fetchone()
            outcome_rows = list(conn.execute(
                "SELECT * FROM outcome_traces WHERE turn_id = ? ORDER BY id",
                (turn_id,),
            ))
    except Exception as exc:
        _trace_failure(f"get_turn_trace failed: {exc}")
        return None
    if not stage_rows and frame_row is None and not outcome_rows:
        return None
    stages = tuple(_row_to_stage(r) for r in stage_rows)
    frame = _row_to_frame(frame_row) if frame_row is not None else None
    outcomes = tuple(_row_to_outcome(r) for r in outcome_rows)
    parent = (
        frame.parent_turn_id if frame is not None
        else (stages[0].parent_turn_id if stages else None)
    )
    errored = sum(1 for s in stages if s.outcome == "errored")
    last_rating = next((o.rating_value for o in reversed(outcomes) if o.kind == "rating"), None)
    summary = {
        "stage_count": len(stages),
        "errored_stage_count": errored,
        "total_chars": frame.total_chars if frame is not None else 0,
        "frame_present": frame is not None,
        "outcome_count": len(outcomes),
        "last_rating": last_rating,
        "effort_tier": frame.effort_tier if frame is not None else None,
    }
    return TurnTraceJoined(
        turn_id=turn_id,
        parent_turn_id=parent,
        stages=stages,
        frame=frame,
        outcomes=outcomes,
        summary=summary,
    )


def list_recent_turns(limit: int = 20) -> list[TurnTraceSummary]:
    """Return up to *limit* recent turn summaries, newest first."""
    limit = max(1, min(int(limit or 20), 200))
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return []
            rows = list(conn.execute(
                """
                SELECT
                    f.turn_id, f.parent_turn_id, f.captured_at, f.backend,
                    f.total_chars,
                    (SELECT COUNT(*) FROM stage_traces s WHERE s.turn_id = f.turn_id) AS stage_count,
                    (SELECT COUNT(*) FROM stage_traces s WHERE s.turn_id = f.turn_id AND s.outcome = 'errored') AS errored_count
                FROM frame_traces f
                ORDER BY f.id DESC
                LIMIT ?
                """,
                (limit,),
            ))
    except Exception as exc:
        _trace_failure(f"list_recent_turns failed: {exc}")
        return []
    return [
        TurnTraceSummary(
            turn_id=r["turn_id"],
            parent_turn_id=r["parent_turn_id"],
            captured_at=r["captured_at"],
            backend=r["backend"],
            stage_count=int(r["stage_count"] or 0),
            errored_stage_count=int(r["errored_count"] or 0),
            total_chars=int(r["total_chars"] or 0),
            frame_present=True,
        )
        for r in rows
    ]


def search_turns(
    *,
    since: str | None = None,
    until: str | None = None,
    backend: str | None = None,
    has_errored_stage: bool | None = None,
    limit: int = 50,
) -> list[TurnTraceSummary]:
    """Filter recent turns by time / backend / has-errored. AND-combined."""
    limit = max(1, min(int(limit or 50), 200))
    where: list[str] = []
    params: list[Any] = []
    if since:
        where.append("f.captured_at >= ?")
        params.append(since)
    if until:
        where.append("f.captured_at <= ?")
        params.append(until)
    if backend:
        where.append("f.backend = ?")
        params.append(backend)
    if has_errored_stage is True:
        where.append("EXISTS (SELECT 1 FROM stage_traces s WHERE s.turn_id = f.turn_id AND s.outcome = 'errored')")
    elif has_errored_stage is False:
        where.append("NOT EXISTS (SELECT 1 FROM stage_traces s WHERE s.turn_id = f.turn_id AND s.outcome = 'errored')")
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""
    sql = (
        "SELECT f.turn_id, f.parent_turn_id, f.captured_at, f.backend, f.total_chars, "
        "(SELECT COUNT(*) FROM stage_traces s WHERE s.turn_id = f.turn_id) AS stage_count, "
        "(SELECT COUNT(*) FROM stage_traces s WHERE s.turn_id = f.turn_id AND s.outcome = 'errored') AS errored_count "
        f"FROM frame_traces f{where_sql} ORDER BY f.id DESC LIMIT ?"
    )
    params.append(limit)
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return []
            rows = list(conn.execute(sql, params))
    except Exception as exc:
        _trace_failure(f"search_turns failed: {exc}")
        return []
    return [
        TurnTraceSummary(
            turn_id=r["turn_id"],
            parent_turn_id=r["parent_turn_id"],
            captured_at=r["captured_at"],
            backend=r["backend"],
            stage_count=int(r["stage_count"] or 0),
            errored_stage_count=int(r["errored_count"] or 0),
            total_chars=int(r["total_chars"] or 0),
            frame_present=True,
        )
        for r in rows
    ]


# ── retention ──────────────────────────────────────────────────────


# ── governance / subagent tree reads (Phase A; single store, no schema change) ──
# These do NOT gate on _flag_enabled() -- they mirror latest_outer_turn (read
# unconditionally). They reconstruct the L1->L2->L3 subagent tree from the existing
# parent_turn_id edges; a DENIED spawn has no child frame, so its fault is keyed on the
# caller turn-id and surfaces via list_governance_events.

def latest_governance_root() -> str | None:
    """The most recent OUTER (L1) turn that has at least one subagent child frame OR a
    spawn governance fault -- the live Workshop tree root, or None."""
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return None
            row = conn.execute(
                """
                SELECT turn_id FROM frame_traces
                WHERE parent_turn_id IS NULL
                  AND turn_id IN (
                    SELECT DISTINCT parent_turn_id FROM frame_traces
                    WHERE parent_turn_id IS NOT NULL
                    UNION
                    SELECT DISTINCT turn_id FROM fault_traces
                    WHERE event_kind IN ('subagent_spawned','subagent_folded',
                                         'spawn_denied','spawn_budget_exhausted',
                                         'monoline_block')  -- Phase B: Monoline runs surface as Workshop roots
                  )
                ORDER BY captured_at DESC, ROWID DESC LIMIT 1
                """,
            ).fetchone()
    except Exception as exc:
        _trace_failure(f"latest_governance_root failed: {exc}")
        return None
    return str(row[0]) if row is not None else None


def list_child_frames(parent_turn_id: str) -> list[FrameTraceRecord]:
    """Direct child frames of a turn (one hop). The full tree walk is the recursive CTE
    in list_governance_events; this is the single-hop primitive the UI recurses on."""
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return []
            rows = list(conn.execute(
                "SELECT * FROM frame_traces WHERE parent_turn_id = ? "
                "ORDER BY captured_at ASC, ROWID ASC",
                (str(parent_turn_id),),
            ))
    except Exception as exc:
        _trace_failure(f"list_child_frames failed: {exc}")
        return []
    return [_row_to_frame(r) for r in rows]


def list_governance_events(root_turn_id: str) -> list[FaultTraceRecord]:
    """Every spawn governance fault (allow + deny) for the whole tree rooted at
    root_turn_id, via a recursive CTE over frame_traces parent edges. A denied spawn has
    no frame, so its fault (keyed on the caller) surfaces because the caller is in the
    CTE node set."""
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return []
            rows = list(conn.execute(
                """
                WITH RECURSIVE tree(turn_id) AS (
                    SELECT turn_id FROM frame_traces WHERE turn_id = ?
                    UNION
                    SELECT f.turn_id FROM frame_traces f
                    JOIN tree t ON f.parent_turn_id = t.turn_id
                )
                SELECT turn_id, parent_turn_id, seq, emitted_at,
                       event_kind, source_kind, source_name,
                       authority_tier, fault_kind, severity, payload_json,
                       payload_schema_version
                FROM fault_traces
                WHERE turn_id IN (SELECT turn_id FROM tree)
                  AND event_kind IN ('subagent_spawned','subagent_folded',
                                     'spawn_denied','spawn_budget_exhausted',
                                     'monoline_block')  -- Phase B: Monoline block faults in the tree
                ORDER BY emitted_at ASC, ROWID ASC
                """,
                (str(root_turn_id),),
            ))
    except Exception as exc:
        _trace_failure(f"list_governance_events failed: {exc}")
        return []
    return [_row_to_fault(r) for r in rows]


def list_recent_runs(limit: int = 20):
    """Recent Monoline run roots, newest-first (the companion run browser's list).

    A run root is a frame with backend='monoline' AND metadata.kind='workflow' (per-block
    frames are kind='monoline_block' and excluded). Read unconditionally (mirrors the other
    governance readers). Returns list[core.run_model.RunSummary]."""
    from core.run_model import RunSummary
    limit = max(1, min(int(limit or 20), 200))
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return []
            rows = list(conn.execute(
                "SELECT turn_id, captured_at, metadata_json FROM frame_traces "
                "WHERE backend = 'monoline' ORDER BY captured_at DESC, id DESC LIMIT 500"))
    except Exception as exc:
        _trace_failure(f"list_recent_runs failed: {exc}")
        return []
    out = []
    for r in rows:
        try:
            meta = json.loads(r["metadata_json"] or "{}")
        except Exception:
            continue
        if str(meta.get("kind", "")) != "workflow":
            continue
        out.append(RunSummary(run_id=str(r["turn_id"]), flow_id=str(meta.get("flow", "")),
                              name=str(meta.get("name", "")), captured_at=str(r["captured_at"])))
        if len(out) >= limit:
            break
    return out


def rehydrate_run(root_turn_id: str):
    """Rebuild a core.run_model.RunModel for a past run from turn_trace ALONE, via the SAME
    fold path as a live run (RunModelBuilder) so the historical model is identical in shape.

    Reads the run-root frame metadata (flow/name/user_input/graph/wires -- persisted by the
    bridge so this survives a deleted .monoline) + the per-block monoline_block fault rows
    (block_id/outputs/timing/status/verdict). Returns None for an unknown / non-workflow root."""
    from core.run_model import BlockFinished, RunBlockSpec, RunModelBuilder, RunStarted
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return None
            row = conn.execute(
                "SELECT metadata_json FROM frame_traces WHERE turn_id = ?",
                (str(root_turn_id),)).fetchone()
    except Exception as exc:
        _trace_failure(f"rehydrate_run frame read failed: {exc}")
        return None
    if row is None:
        return None
    try:
        meta = json.loads(row["metadata_json"] or "{}")
    except Exception:
        meta = {}
    if str(meta.get("kind", "")) != "workflow":
        return None
    graph = [RunBlockSpec(id=str(g.get("id", "")), label=str(g.get("label", "")),
                          kind=str(g.get("kind", "")))
             for g in (meta.get("graph") or [])]
    builder = RunModelBuilder()
    builder.apply(RunStarted(
        run_id=str(root_turn_id), flow_id=str(meta.get("flow", "")),
        name=str(meta.get("name", "")), user_input=str(meta.get("user_input", "")),
        graph=graph, wires=list(meta.get("wires") or [])))
    for e in list_governance_events(str(root_turn_id)):
        if getattr(e, "event_kind", "") != "monoline_block":
            continue
        p = getattr(e, "payload", {}) or {}
        ok = bool(p.get("ok", True))
        builder.apply(BlockFinished(
            run_id=str(root_turn_id), block_id=str(p.get("block_id", "") or ""),
            label=str(p.get("block_label", "") or ""), kind=str(p.get("step_kind", "") or ""),
            outputs=dict(p.get("outputs") or {}),
            started_at=float(p.get("started_at", 0.0) or 0.0),
            completed_at=float(p.get("completed_at", 0.0) or 0.0),
            status="done" if ok else "error", error=str(p.get("error", "") or ""),
            verdict=p.get("verdict"), detectors=p.get("detectors")))
    m = builder.model
    if m is not None:
        # the run is over; RunFinished isn't persisted, so infer terminal status from blocks.
        m.status = "error" if any(b.status == "error" for b in m.block_list()) else "done"
    return m


def cleanup_old_records(*, ttl_days: int | None = None) -> dict[str, int]:
    """Delete records older than the configured TTL. Returns count of rows deleted."""
    if ttl_days is None:
        ttl_days = _ttl_days()
    cutoff = (datetime.now(timezone.utc).timestamp() - ttl_days * 86400)
    cutoff_iso = datetime.fromtimestamp(cutoff, tz=timezone.utc).isoformat()
    try:
        with _db_lock:
            conn = _get_conn()
            if conn is None:
                return {"stage": 0, "frame": 0, "outcome": 0}
            stage_n = conn.execute(
                "DELETE FROM stage_traces WHERE entered_at < ?", (cutoff_iso,)
            ).rowcount
            frame_n = conn.execute(
                "DELETE FROM frame_traces WHERE captured_at < ?", (cutoff_iso,)
            ).rowcount
            outcome_n = conn.execute(
                "DELETE FROM outcome_traces WHERE recorded_at < ?", (cutoff_iso,)
            ).rowcount
            fault_n = conn.execute(
                "DELETE FROM fault_traces WHERE emitted_at < ?", (cutoff_iso,)
            ).rowcount
        return {
            "stage": int(stage_n or 0),
            "frame": int(frame_n or 0),
            "outcome": int(outcome_n or 0),
            "fault": int(fault_n or 0),
        }
    except Exception as exc:
        _trace_failure(f"cleanup_old_records failed: {exc}")
        return {"stage": 0, "frame": 0, "outcome": 0, "fault": 0}


# ── helpers used by producers ──────────────────────────────────────


def diff_added_messages(before: list[dict], after: list[dict]) -> list[dict]:
    """Return messages present in *after* but not in *before*.

    Identity is by the dict object (Python id) when possible — interceptors
    that build new lists with `list(messages)` and `.insert(...)` produce new
    dict objects only for the inserted entries. Falls back to content-hash
    comparison when identity doesn't match.
    """
    before_ids = {id(m) for m in before}
    if all(id(m) in before_ids or m not in before for m in after):
        # cheap path: identity comparison
        return [m for m in after if id(m) not in before_ids]
    # fallback: content-shape comparison
    before_keys = {(_hash(str(m.get("content", ""))), m.get("role"))
                   for m in before}
    return [m for m in after
            if (_hash(str(m.get("content", ""))), m.get("role")) not in before_keys]
