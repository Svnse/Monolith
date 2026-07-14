"""Plan store (M1 V0) — persisted task/goal plans in turn_trace.sqlite3.

A plan = a goal decomposed into an ordered DAG of steps (verb, target,
depends_on). Lives alongside the turn_trace layers in the same DB file (E's
choice) via dedicated `plans` + `plan_steps` tables, but in its own module +
connection (mirroring the turn_trace store pattern: shared conn, _db_lock,
set_db_path test override, idempotent DDL).

Propose-only: this store holds proposed/active plans and tracks step status as
the human / existing gated tool loop executes them. The planner NEVER drives
execution from here.
"""
from __future__ import annotations

import json
import os
import sqlite3
import sys
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.paths import LOG_DIR
from core import grounded_verdict

_DB_PATH = LOG_DIR / "turn_trace.sqlite3"

VALID_PLAN_STATUS = frozenset({"proposed", "active", "done", "abandoned"})
VALID_STEP_STATUS = frozenset({"pending", "done", "failed", "skipped"})
VALID_OBSERVATION_KIND = frozenset({"visited", "finding"})
VALID_CRITERION_STATUS = frozenset({"open", "met", "failed"})
_OPEN_STATUSES = ("proposed", "active")

_DONE_GATE_FLAG = "MONOLITH_PLAN_DONE_GATE_V1"


def done_gate_enabled() -> bool:
    """Grounded done-gate + bearing render-view flag. Default OFF (dark);
    flag-off path byte-identical."""
    return str(os.environ.get(_DONE_GATE_FLAG, "0")).strip().lower() in {"1", "true", "yes", "on"}


_DDL = (
    """
    CREATE TABLE IF NOT EXISTS plans (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plan_uid TEXT NOT NULL UNIQUE,
        goal TEXT NOT NULL,
        source TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'proposed',
        created_at TEXT NOT NULL,
        turn_id TEXT,
        completed_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_plans_status ON plans(status, id)",
    """
    CREATE TABLE IF NOT EXISTS plan_steps (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plan_id INTEGER NOT NULL,
        seq INTEGER NOT NULL,
        verb TEXT NOT NULL,
        target TEXT NOT NULL,
        depends_on TEXT NOT NULL DEFAULT '[]',
        status TEXT NOT NULL DEFAULT 'pending',
        note TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_plan_steps_plan ON plan_steps(plan_id, seq)",
    # MonoExplore observed-ledger: per-expedition "what I've already seen" set,
    # scoped by plan_id (new expedition = new plan = fresh ledger). The reader
    # the expedition lacked — closes the producer-with-no-reader loop without
    # touching the shared bearing. See docs/superpowers/specs/
    # 2026-06-10-monoexplore-observed-ledger-design.md.
    """
    CREATE TABLE IF NOT EXISTS plan_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        plan_id INTEGER NOT NULL,
        tick TEXT NOT NULL,
        kind TEXT NOT NULL,
        content TEXT NOT NULL,
        created_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_plan_obs_plan ON plan_observations(plan_id, id)",
    """
    CREATE TABLE IF NOT EXISTS plan_criteria (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        plan_id     INTEGER NOT NULL,
        seq         INTEGER NOT NULL,
        criterion   TEXT    NOT NULL,
        status      TEXT    NOT NULL DEFAULT 'open',
        evidence    TEXT,
        cite_handle TEXT,
        ground_kind TEXT,
        attested_at TEXT
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_plan_criteria_plan ON plan_criteria(plan_id, seq)",
)

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


def _trace_failure(msg: str) -> None:
    try:
        sys.stderr.write(f"[plans] {msg}\n")
    except Exception:
        pass


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is not None:
        return _conn
    path = _get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(path), check_same_thread=False, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA busy_timeout=4000")  # shared file with turn_trace — tolerate brief locks
    for stmt in _DDL:
        conn.execute(stmt)
    cols = {r["name"] for r in conn.execute("PRAGMA table_info(plans)").fetchall()}
    if "completed_at" not in cols:
        conn.execute("ALTER TABLE plans ADD COLUMN completed_at TEXT")
    _conn = conn
    return conn


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── write ─────────────────────────────────────────────────────────────

def create_plan(goal: str, source: str, steps: list[dict], turn_id: str | None = None) -> str:
    """Persist a proposed plan. ``steps`` = ordered list of
    {verb, target, depends_on:[earlier seqs]}. Returns the plan_uid."""
    goal_s = str(goal or "").strip()
    if not goal_s:
        raise ValueError("plan requires a non-empty goal")
    if not steps:
        raise ValueError("plan requires at least one step")
    uid = uuid.uuid4().hex
    with _db_lock:
        conn = _get_conn()
        cur = conn.execute(
            "INSERT INTO plans(plan_uid, goal, source, status, created_at, turn_id) "
            "VALUES (?, ?, ?, 'proposed', ?, ?)",
            (uid, goal_s, str(source or "explicit"), _now_iso(), turn_id),
        )
        plan_id = int(cur.lastrowid)
        for i, step in enumerate(steps, start=1):
            deps = [int(d) for d in (step.get("depends_on") or []) if int(d) < i]
            conn.execute(
                "INSERT INTO plan_steps(plan_id, seq, verb, target, depends_on, status, note) "
                "VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                (plan_id, i, str(step.get("verb", "")).strip(), str(step.get("target", "")).strip(),
                 json.dumps(deps), step.get("note")),
            )
    return uid


def mark_step(plan_uid: str, seq: int, status: str) -> None:
    if status not in VALID_STEP_STATUS:
        raise ValueError(f"invalid step status {status!r}")
    with _db_lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE plan_steps SET status=? WHERE seq=? AND plan_id="
            "(SELECT id FROM plans WHERE plan_uid=?)",
            (status, int(seq), plan_uid),
        )


def set_plan_status(plan_uid: str, status: str) -> None:
    if status not in VALID_PLAN_STATUS:
        raise ValueError(f"invalid plan status {status!r}")
    with _db_lock:
        conn = _get_conn()
        conn.execute("UPDATE plans SET status=? WHERE plan_uid=?", (status, plan_uid))


def record_observations(
    plan_uid: str, tick: str, visited: list[str], findings: list[str]
) -> int:
    """Append an expedition tick's observations to the plan-scoped ledger.

    ``visited`` entries ("<tool> <target>") are write-deduped against what this
    plan has already seen — re-running list_files on the same path adds nothing
    (the anti-re-listing signal). ``findings`` (atomic triples) are appended.
    Unknown plan_uid is a no-op. Returns the number of rows written."""
    with _db_lock:
        conn = _get_conn()
        prow = conn.execute("SELECT id FROM plans WHERE plan_uid=?", (plan_uid,)).fetchone()
        if prow is None:
            return 0
        plan_id = int(prow["id"])
        seen = {
            r["content"] for r in conn.execute(
                "SELECT content FROM plan_observations WHERE plan_id=? AND kind='visited'",
                (plan_id,),
            ).fetchall()
        }
        now = _now_iso()
        written = 0
        for v in visited:
            v = str(v or "").strip()
            if not v or v in seen:
                continue
            seen.add(v)
            conn.execute(
                "INSERT INTO plan_observations(plan_id, tick, kind, content, created_at) "
                "VALUES (?, ?, 'visited', ?, ?)",
                (plan_id, str(tick), v, now),
            )
            written += 1
        for f in findings:
            f = str(f or "").strip()
            if not f:
                continue
            conn.execute(
                "INSERT INTO plan_observations(plan_id, tick, kind, content, created_at) "
                "VALUES (?, ?, 'finding', ?, ?)",
                (plan_id, str(tick), f, now),
            )
            written += 1
        return written


def record_finding(plan_uid: str, content: str, tick: str = "explicit") -> int | None:
    """Append ONE finding to the plan ledger and return its observation id — the
    citable handle is `obs:<id>`. Blank content / unknown plan → None."""
    content = str(content or "").strip()
    if not content:
        return None
    with _db_lock:
        conn = _get_conn()
        prow = conn.execute("SELECT id FROM plans WHERE plan_uid=?", (plan_uid,)).fetchone()
        if prow is None:
            return None
        cur = conn.execute(
            "INSERT INTO plan_observations(plan_id, tick, kind, content, created_at) "
            "VALUES (?, ?, 'finding', ?, ?)",
            (int(prow["id"]), str(tick), content, _now_iso()),
        )
        return int(cur.lastrowid)


def get_observation(plan_uid: str, obs_id: int) -> dict | None:
    """Resolve an observation id to its row IFF it belongs to this plan. A cross-
    plan or unknown id → None (the plan-scoped grounding guard)."""
    with _db_lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT o.id, o.kind, o.content FROM plan_observations o "
            "JOIN plans p ON p.id = o.plan_id WHERE p.plan_uid=? AND o.id=? AND o.kind='finding'",
            (plan_uid, int(obs_id)),
        ).fetchone()
        return dict(row) if row else None


def list_findings(plan_uid: str, limit: int = 20) -> list[dict]:
    """The plan's finding observations as citable grounds: [{id, content}] in
    chronological order (most recent `limit`). The model cites one as
    `[cite: obs:<id>]` when attesting a criterion — this is what makes a ground
    discoverable at attest time (advisor pass #1). Unknown plan → []."""
    with _db_lock:
        conn = _get_conn()
        prow = conn.execute("SELECT id FROM plans WHERE plan_uid=?", (plan_uid,)).fetchone()
        if prow is None:
            return []
        rows = conn.execute(
            "SELECT id, content FROM plan_observations WHERE plan_id=? AND kind='finding' "
            "ORDER BY id ASC",
            (int(prow["id"]),),
        ).fetchall()
        out = [{"id": int(r["id"]), "content": r["content"]} for r in rows]
        return out[-int(limit):] if limit and limit > 0 else out


def set_criteria(plan_uid: str, criteria: list[str]) -> int:
    """Replace the plan's success criteria (seq 1..N, all 'open'). Blank entries
    dropped. Unknown plan_uid → 0. Returns the count written."""
    crits = [str(c or "").strip() for c in (criteria or []) if str(c or "").strip()]
    with _db_lock:
        conn = _get_conn()
        prow = conn.execute("SELECT id FROM plans WHERE plan_uid=?", (plan_uid,)).fetchone()
        if prow is None:
            return 0
        plan_id = int(prow["id"])
        conn.execute("DELETE FROM plan_criteria WHERE plan_id=?", (plan_id,))
        for i, c in enumerate(crits, start=1):
            conn.execute(
                "INSERT INTO plan_criteria(plan_id, seq, criterion, status) "
                "VALUES (?, ?, ?, 'open')",
                (plan_id, i, c),
            )
    return len(crits)


def get_criteria(plan_uid: str) -> list[dict]:
    """Criterion rows (seq order) for a plan, or [] if unknown."""
    with _db_lock:
        conn = _get_conn()
        prow = conn.execute("SELECT id FROM plans WHERE plan_uid=?", (plan_uid,)).fetchone()
        if prow is None:
            return []
        rows = conn.execute(
            "SELECT seq, criterion, status, evidence, cite_handle, ground_kind, attested_at "
            "FROM plan_criteria WHERE plan_id=? ORDER BY seq ASC",
            (int(prow["id"]),),
        ).fetchall()
        return [dict(r) for r in rows]


def _log_fabricated(plan_uid: str, seq: int, handles: list[str]) -> None:
    try:
        from core.acatalepsy import canonical_log
        canonical_log.append(
            "plan_criterion_fabricated",
            payload={"plan_uid": plan_uid, "seq": int(seq), "handles": list(handles)},
        )
    except Exception:
        pass


def attest_criterion(plan_uid: str, seq: int, evidence: str, resolve) -> dict:
    """Attest one criterion with cited evidence. `resolve` is a plan-scoped
    Resolver (see core.plan_grounding). `met` iff >=1 cite resolves AND none are
    fabricated; any fabricated cite → `failed` (logged); only [no-ground]/no cite
    → stays `open`. Returns the updated criterion dict, or {} on unknown plan/seq."""
    parsed = grounded_verdict.parse_cites(evidence)
    winning: str | None = None
    fabricated: list[str] = []
    for h in parsed.handles:
        if resolve(h) is None:
            fabricated.append(h)
        elif winning is None:
            winning = h
    if winning is not None and not fabricated:
        status, cite_handle = "met", winning
        ground_kind = winning.split(":", 1)[0] if ":" in winning else "ground"
    elif fabricated:
        status, cite_handle, ground_kind = "failed", None, None
    else:
        status, cite_handle, ground_kind = "open", None, None
    with _db_lock:
        conn = _get_conn()
        prow = conn.execute("SELECT id FROM plans WHERE plan_uid=?", (plan_uid,)).fetchone()
        if prow is None:
            return {}
        cur = conn.execute(
            "UPDATE plan_criteria SET status=?, evidence=?, cite_handle=?, ground_kind=?, "
            "attested_at=? WHERE plan_id=? AND seq=?",
            (status, str(evidence or ""), cite_handle, ground_kind, _now_iso(),
             int(prow["id"]), int(seq)),
        )
        if cur.rowcount == 0:
            return {}
    if fabricated:
        _log_fabricated(plan_uid, seq, fabricated)
    matches = [c for c in get_criteria(plan_uid) if c["seq"] == int(seq)]
    return matches[0] if matches else {}


def _log_plan_done(plan_uid: str) -> None:
    try:
        from core.acatalepsy import canonical_log
        canonical_log.append("plan_status_changed", payload={"plan_uid": plan_uid, "status": "done"})
    except Exception:
        pass


def complete_plan(plan_uid: str) -> dict:
    """The done-gate audit (mirrors /goal complete). Refuses — no status change —
    if any step is not done/skipped, the plan has zero criteria, or any criterion
    is not `met`. Does NOT re-derive evidence: criteria reach `met` only via prior
    attest_criterion. On full pass → status='done' + completed_at stamped.
    Returns the audit dict (`ok`, `reason`, `steps_open`, `criteria_total`,
    `criteria_unmet`)."""
    p = get_plan(plan_uid)
    if p is None:
        return {"ok": False, "reason": "unknown_plan"}
    if p["status"] == "done":
        return {"ok": True, "plan_uid": plan_uid, "already_done": True}
    if p["status"] == "abandoned":
        return {"ok": False, "plan_uid": plan_uid, "reason": "abandoned"}
    steps_open = [s["seq"] for s in p["steps"] if s["status"] not in ("done", "skipped")]
    crits = get_criteria(plan_uid)
    unmet = [c["seq"] for c in crits if c["status"] != "met"]
    audit = {
        "plan_uid": plan_uid,
        "steps_open": steps_open,
        "criteria_total": len(crits),
        "criteria_unmet": unmet,
    }
    if steps_open:
        audit.update(ok=False, reason="steps_incomplete")
        return audit
    if not crits:
        audit.update(ok=False, reason="no_criteria")
        return audit
    if unmet:
        audit.update(ok=False, reason="criteria_unmet")
        return audit
    with _db_lock:
        conn = _get_conn()
        conn.execute(
            "UPDATE plans SET status='done', completed_at=? WHERE plan_uid=?",
            (_now_iso(), plan_uid),
        )
    _log_plan_done(plan_uid)
    audit["ok"] = True
    return audit


# ── read ──────────────────────────────────────────────────────────────

def _row_to_plan(conn: sqlite3.Connection, prow: sqlite3.Row) -> dict:
    steps = []
    for s in conn.execute(
        "SELECT seq, verb, target, depends_on, status, note FROM plan_steps "
        "WHERE plan_id=? ORDER BY seq ASC", (int(prow["id"]),)
    ).fetchall():
        steps.append({
            "seq": int(s["seq"]), "verb": s["verb"], "target": s["target"],
            "depends_on": json.loads(s["depends_on"] or "[]"),
            "status": s["status"], "note": s["note"],
        })
    return {
        "plan_uid": prow["plan_uid"], "goal": prow["goal"], "source": prow["source"],
        "status": prow["status"], "created_at": prow["created_at"], "turn_id": prow["turn_id"],
        "completed_at": prow["completed_at"],
        "steps": steps,
    }


def get_plan(plan_uid: str) -> dict | None:
    with _db_lock:
        conn = _get_conn()
        prow = conn.execute("SELECT * FROM plans WHERE plan_uid=?", (plan_uid,)).fetchone()
        return _row_to_plan(conn, prow) if prow else None


def get_observations(plan_uid: str, max_visited: int = 25, max_findings: int = 12) -> dict:
    """Read the plan's observed-ledger for injection into the next tick.

    Returns {"visited": [...], "findings": [...]} in chronological order, keeping
    the most-recent ``max_*`` of each. Unknown plan_uid → empty slots."""
    empty = {"visited": [], "findings": []}
    with _db_lock:
        conn = _get_conn()
        prow = conn.execute("SELECT id FROM plans WHERE plan_uid=?", (plan_uid,)).fetchone()
        if prow is None:
            return empty
        plan_id = int(prow["id"])

        def _tail(kind: str, cap: int) -> list[str]:
            rows = conn.execute(
                "SELECT content FROM plan_observations WHERE plan_id=? AND kind=? ORDER BY id ASC",
                (plan_id, kind),
            ).fetchall()
            contents = [r["content"] for r in rows]
            return contents[-int(cap):] if cap and cap > 0 else contents

        return {"visited": _tail("visited", max_visited), "findings": _tail("finding", max_findings)}


def get_active_plan() -> dict | None:
    """The most-recently-created open (proposed/active) plan, or None."""
    with _db_lock:
        conn = _get_conn()
        prow = conn.execute(
            f"SELECT * FROM plans WHERE status IN {_OPEN_STATUSES} ORDER BY id DESC LIMIT 1"
        ).fetchone()
        return _row_to_plan(conn, prow) if prow else None


def list_plans(limit: int = 20) -> list[dict]:
    with _db_lock:
        conn = _get_conn()
        rows = conn.execute("SELECT * FROM plans ORDER BY id DESC LIMIT ?", (int(limit),)).fetchall()
        return [_row_to_plan(conn, r) for r in rows]


def next_ready_steps(plan_uid: str) -> list[dict]:
    """Pending steps whose dependencies are all `done` — the executable frontier."""
    p = get_plan(plan_uid)
    if p is None:
        return []
    done = {s["seq"] for s in p["steps"] if s["status"] == "done"}
    ready = []
    for s in p["steps"]:
        if s["status"] != "pending":
            continue
        if all(dep in done for dep in s["depends_on"]):
            ready.append(s)
    return ready
