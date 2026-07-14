"""Tests for core.intervention_arena.storage."""
from __future__ import annotations

import sqlite3

from core.intervention_arena import storage


def test_init_creates_three_tables(tmp_arena_db):
    storage.init_db()
    with sqlite3.connect(tmp_arena_db) as conn:
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        tables = {row[0] for row in cursor.fetchall()}
    assert "interventions" in tables
    assert "intervention_observations" in tables
    assert "retired_interventions" in tables


def test_init_is_idempotent(tmp_arena_db):
    storage.init_db()
    storage.init_db()  # Second call must not raise.
    with sqlite3.connect(tmp_arena_db) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM interventions")
        assert cursor.fetchone()[0] == 0


def test_interventions_schema_columns(tmp_arena_db):
    storage.init_db()
    with sqlite3.connect(tmp_arena_db) as conn:
        cursor = conn.execute("PRAGMA table_info(interventions)")
        columns = {row[1] for row in cursor.fetchall()}
    required = {
        "id", "name", "kind", "channel", "bucket_signature",
        "predicate_text", "predicate_hash", "prompt_patch", "inverse_patch",
        "target_file", "target_file_hash_at_entry",
        "entered_at", "activated_at",
        "validation_deadline_turns", "matched_turns_count",
        "baseline_composite_rate", "baseline_layer_d_dist",
        "baseline_n_observations", "baseline_at_turn_id",
        "status", "status_changed_at", "quarantined_until_n_layer_d",
    }
    missing = required - columns
    assert not missing, f"missing columns: {missing}"
