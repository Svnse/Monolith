"""The MonoSearch salience ledger (spec §7) — the selection-function core.

A thin DERIVED, REGENERABLE store: counts only, never source content, never the
salience VALUE itself (that is computed on read). Lives in its own
monosearch.sqlite3 so it sidesteps the acatalepsy reader-authorizer entirely.

THE LOAD-BEARING SUBTLETY: salience recency uses ``decay.decay_factor`` DIRECTLY
with a MonoSearch-owned half-life. It must NOT route through
``decay.effective_reinforcement``, which applies the self=1 / user=4 / world=4
provenance multiplier — that would re-import the exact self-suppression
MonoSearch exists to escape (a self-pattern fading 4x faster than external).
"""
from __future__ import annotations

import sqlite3
import threading

from core.acatalepsy.decay import decay_factor
from core.monosearch.record import Provenance
from core.paths import LOG_DIR

# MonoSearch-owned recency half-life (days). A standalone constant on purpose —
# see the module docstring on why we do NOT reuse decay's provenance-multiplied
# effective_reinforcement path.
HALF_LIFE_DAYS = 30.0

_DB_PATH = LOG_DIR / "monosearch.sqlite3"
_db_lock = threading.Lock()
_conn: sqlite3.Connection | None = None

_SCHEMA = """
CREATE TABLE IF NOT EXISTS salience (
    recurrence_key TEXT NOT NULL,
    source         TEXT NOT NULL,
    provenance     TEXT NOT NULL,
    count          INTEGER NOT NULL DEFAULT 0,
    first_seen     REAL,
    last_seen      REAL,
    PRIMARY KEY (recurrence_key, source)
);
CREATE INDEX IF NOT EXISTS idx_salience_source ON salience(source);
"""


def set_db_path(path) -> None:
    """Test/override hook (mirrors core.plans.set_db_path)."""
    global _DB_PATH, _conn
    with _db_lock:
        if _conn is not None:
            _conn.close()
            _conn = None
        _DB_PATH = path


def _get_conn() -> sqlite3.Connection:
    global _conn
    if _conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        _conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False, timeout=5.0)
        _conn.row_factory = sqlite3.Row
        _conn.executescript(_SCHEMA)
    return _conn


def ensure_schema() -> None:
    with _db_lock:
        _get_conn()


def close() -> None:
    """Close the connection without changing the path (test teardown — frees the
    file on Windows so a temp dir can be removed). The next call re-opens."""
    global _conn
    with _db_lock:
        if _conn is not None:
            _conn.close()
            _conn = None


def record_observation(
    recurrence_key: str | None,
    provenance: Provenance,
    source: str,
    ts: float | None,
) -> None:
    """Upsert one observation of a recurrence_key: count += 1, track first/last seen.
    A None recurrence_key is not salience-eligible and is ignored."""
    if recurrence_key is None:
        return
    ts = float(ts) if ts is not None else 0.0
    prov = provenance.value if isinstance(provenance, Provenance) else str(provenance)
    with _db_lock:
        conn = _get_conn()
        conn.execute(
            "INSERT INTO salience(recurrence_key, source, provenance, count, first_seen, last_seen) "
            "VALUES (?, ?, ?, 1, ?, ?) "
            "ON CONFLICT(recurrence_key, source) DO UPDATE SET "
            "count = count + 1, "
            "first_seen = MIN(first_seen, excluded.first_seen), "
            "last_seen = MAX(last_seen, excluded.last_seen)",
            (recurrence_key, source, prov, ts, ts),
        )
        conn.commit()


def get_row(recurrence_key: str, source: str) -> dict | None:
    with _db_lock:
        conn = _get_conn()
        row = conn.execute(
            "SELECT recurrence_key, source, provenance, count, first_seen, last_seen "
            "FROM salience WHERE recurrence_key = ? AND source = ?",
            (recurrence_key, source),
        ).fetchone()
    return dict(row) if row is not None else None


def clear() -> None:
    with _db_lock:
        conn = _get_conn()
        conn.execute("DELETE FROM salience")
        conn.commit()


def _delete_sources(sources: list[str]) -> None:
    """Clear only the named sources' rows (per-source rebuild). Lets `failing`
    refresh just fault_traces without wiping the other sources' ledger rows."""
    if not sources:
        return
    with _db_lock:
        conn = _get_conn()
        conn.executemany("DELETE FROM salience WHERE source = ?", [(s,) for s in sources])
        conn.commit()


# ── salience (computed on read, never persisted) ─────────────────────────────


def _salience(count: int, last_seen: float | None, now: float) -> float:
    """recurrence x recency. decay_factor() called DIRECTLY (NOT
    effective_reinforcement — see module docstring)."""
    if not count:
        return 0.0
    if last_seen is None:
        return float(count)
    age_days = max(0.0, (now - last_seen) / 86400.0)
    return float(count) * decay_factor(age_days, HALF_LIFE_DAYS)


def _ranked(rows: list[dict], now: float, limit: int) -> list[dict]:
    scored = []
    for r in rows:
        r = dict(r)
        r["salience"] = _salience(r["count"], r["last_seen"], now)
        scored.append(r)
    scored.sort(key=lambda r: r["salience"], reverse=True)
    return scored[: max(1, int(limit))]


def failing(now: float, limit: int = 10) -> list[dict]:
    """Top SELF-sourced fault recurrence keys by salience — 'what I keep failing'."""
    with _db_lock:
        conn = _get_conn()
        rows = [dict(r) for r in conn.execute(
            "SELECT recurrence_key, source, provenance, count, first_seen, last_seen "
            "FROM salience WHERE source = 'fault_traces' AND provenance = 'self'"
        )]
    return _ranked(rows, now, limit)


def recurring(now: float, limit: int = 10) -> list[dict]:
    """Top recurrence keys across ALL sources by salience."""
    with _db_lock:
        conn = _get_conn()
        rows = [dict(r) for r in conn.execute(
            "SELECT recurrence_key, source, provenance, count, first_seen, last_seen FROM salience"
        )]
    return _ranked(rows, now, limit)


def rebuild(adapters: list, now: float, scan_limit: int = 5000) -> int:
    """Regenerate the ledger from adapters (proves it owns no source-of-truth).
    Returns the number of observations recorded.

    PER-ADAPTER ISOLATION (mirrors router.search): one source raising/hanging on
    read must NOT take down the others. Without this, a canonical_log read problem
    in prod would silently disable `failing` — which doesn't even use that source.

    PER-SOURCE CLEAR: only the rebuilt sources' rows are deleted, so a scoped
    rebuild (e.g. `failing` refreshing only fault_traces) does not wipe the others.
    """
    _delete_sources([a.name for a in adapters])
    n = 0
    for adapter in adapters:
        try:
            recs = adapter.list({}, scan_limit)
        except Exception:
            continue  # one bad adapter drops to zero contributions, doesn't kill the rest
        for rec in recs:
            if rec.recurrence_key is None:
                continue
            try:
                record_observation(rec.recurrence_key, rec.provenance, rec.source, rec.ts)
                n += 1
            except Exception:
                continue
    return n
