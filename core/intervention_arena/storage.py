"""Sqlite storage for the Intervention Arena.

Three tables at LOG_DIR/intervention_arena.sqlite3:
  - interventions          : active + graduated interventions
  - intervention_observations : per-turn observations rows
  - retired_interventions  : archive of every exit
"""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

from core.paths import LOG_DIR


_DEFAULT_DB_PATH = LOG_DIR / "intervention_arena.sqlite3"


def _resolve_db_path() -> Path:
    raw = os.environ.get("MONOLITH_ARENA_DB_PATH")
    if raw:
        return Path(raw)
    return _DEFAULT_DB_PATH


_DB_PATH = _resolve_db_path()


_DDL = (
    """
    CREATE TABLE IF NOT EXISTS interventions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        kind TEXT NOT NULL CHECK (kind IN ('additive', 'subtractive')),
        channel TEXT NOT NULL CHECK (channel IN ('A_move_tag', 'B_claim_source', 'other')),
        bucket_signature TEXT NOT NULL,
        predicate_text TEXT NOT NULL,
        predicate_hash TEXT NOT NULL,
        prompt_patch TEXT,
        inverse_patch TEXT NOT NULL,
        target_file TEXT NOT NULL DEFAULT 'prompts/system.md',
        target_file_hash_at_entry TEXT NOT NULL,
        entered_at TEXT NOT NULL,
        activated_at TEXT,
        validation_deadline_turns INTEGER NOT NULL,
        matched_turns_count INTEGER NOT NULL DEFAULT 0,
        baseline_composite_rate REAL,
        baseline_layer_d_dist TEXT,
        baseline_n_observations INTEGER,
        baseline_at_turn_id TEXT,
        status TEXT NOT NULL CHECK (status IN (
            'proposed', 'entered', 'active', 'stage_1_pass',
            'graduated', 'expired', 'retired'
        )),
        status_changed_at TEXT NOT NULL,
        quarantined_until_n_layer_d INTEGER
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS intervention_observations (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        intervention_id INTEGER NOT NULL REFERENCES interventions(id),
        turn_id TEXT NOT NULL,
        bucket_match BOOLEAN NOT NULL,
        predicate_outcome BOOLEAN,
        composite_signal BOOLEAN,
        layer_d_polarity INTEGER,
        recorded_at TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_obs_intervention ON intervention_observations(intervention_id, recorded_at)",
    "CREATE INDEX IF NOT EXISTS idx_obs_turn ON intervention_observations(turn_id)",
    """
    CREATE TABLE IF NOT EXISTS retired_interventions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_intervention_id INTEGER,
        name TEXT NOT NULL,
        kind TEXT NOT NULL,
        channel TEXT NOT NULL,
        bucket_signature TEXT NOT NULL,
        predicate_hash TEXT NOT NULL,
        exit_reason TEXT NOT NULL CHECK (exit_reason IN (
            'deadline_no_lift',
            'post_graduation_regression',
            'voluntarily_retired',
            'predicate_quarantined_persistent',
            'subsumed_by_replacement',
            'insufficient_baseline'
        )),
        exit_at TEXT NOT NULL,
        lift_estimate_at_exit REAL,
        matched_turns_at_exit INTEGER,
        summary_json TEXT NOT NULL
    )
    """,
    "CREATE INDEX IF NOT EXISTS idx_retired_signature ON retired_interventions(predicate_hash)",
    "CREATE INDEX IF NOT EXISTS idx_retired_bucket ON retired_interventions(bucket_signature)",
)


@contextmanager
def conn() -> Iterator[sqlite3.Connection]:
    """Yield a sqlite3 Connection with row_factory set, foreign keys on,
    committing on success and rolling back on exception."""
    db_path = _DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(db_path)
    c.execute("PRAGMA foreign_keys = ON")
    c.row_factory = sqlite3.Row
    try:
        yield c
        c.commit()
    except Exception:
        c.rollback()
        raise
    finally:
        c.close()


def init_db() -> None:
    """Create the three tables if missing. Idempotent."""
    with conn() as c:
        for stmt in _DDL:
            c.execute(stmt)
