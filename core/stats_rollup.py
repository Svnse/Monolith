"""Daily-rollup materializer for the stats addon.

Reads from frame_traces / outcome_traces / fault_traces / stage_traces and
writes one row per past UTC date into daily_rollups. Today's row is never
written — it's always computed live by StatsStore. Idempotent: INSERT OR
REPLACE keyed on date, so re-running is safe.

Also runs achievement-unlock checks via check_achievements after each new
rollup row is written.
"""
from __future__ import annotations

import json
from datetime import date as _date_cls, datetime, timezone, timedelta
from typing import Any

import sqlite3


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _bucket_rating(value: int) -> str:
    if value >= 100:
        return "90-100"
    lo = (value // 10) * 10
    return f"{lo}-{lo + 9}"


def _date_range(start_iso: str, end_iso: str) -> list[str]:
    start = _date_cls.fromisoformat(start_iso)
    end = _date_cls.fromisoformat(end_iso)
    out = []
    cursor = start
    while cursor <= end:
        out.append(cursor.isoformat())
        cursor += timedelta(days=1)
    return out


def materialize_missing_days(conn: sqlite3.Connection) -> int:
    """Compute rollup rows for every date between the earliest frame_trace
    and yesterday (today is excluded; it's always live). Returns the count
    of days written this call (0 if nothing missing)."""
    row = conn.execute(
        "SELECT MIN(SUBSTR(captured_at, 1, 10)) FROM frame_traces"
    ).fetchone()
    if not row or not row[0]:
        return 0
    earliest = str(row[0])
    today = _today_iso()
    yesterday = (_date_cls.fromisoformat(today) - timedelta(days=1)).isoformat()
    if yesterday < earliest:
        return 0
    existing = {
        r[0] for r in conn.execute("SELECT date FROM daily_rollups").fetchall()
    }
    written = 0
    for date_iso in _date_range(earliest, yesterday):
        if date_iso in existing:
            continue
        _materialize_one_day(conn, date_iso)
        written += 1
    conn.commit()
    return written


def _materialize_one_day(conn: sqlite3.Connection, date_iso: str) -> None:
    """Compute and insert one daily_rollups row for date_iso."""
    frame_row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(total_chars), 0) "
        "FROM frame_traces WHERE SUBSTR(captured_at, 1, 10) = ?",
        (date_iso,),
    ).fetchone()
    turns = int(frame_row[0] or 0)
    total_chars = int(frame_row[1] or 0)
    rating_rows = conn.execute(
        "SELECT rating_value FROM outcome_traces "
        "WHERE kind = 'rating' AND rating_value IS NOT NULL "
        "AND SUBSTR(recorded_at, 1, 10) = ?",
        (date_iso,),
    ).fetchall()
    ratings_count = len(rating_rows)
    ratings_sum = sum(int(r[0]) for r in rating_rows)
    ratings_hist: dict[str, int] = {}
    for (val,) in rating_rows:
        bucket = _bucket_rating(int(val))
        ratings_hist[bucket] = ratings_hist.get(bucket, 0) + 1
    effort_dist = _column_distribution(conn, "effort_tier", date_iso)
    reasoning_dist = _column_distribution(conn, "reasoning_mode", date_iso)
    fault_rows = conn.execute(
        "SELECT fault_kind, COUNT(*) FROM fault_traces "
        "WHERE fault_kind IS NOT NULL "
        "AND SUBSTR(emitted_at, 1, 10) = ? GROUP BY fault_kind",
        (date_iso,),
    ).fetchall()
    fault_summary = {str(k): int(v) for k, v in fault_rows if k}
    tool_rows = conn.execute(
        "SELECT source_name, COUNT(*) FROM fault_traces "
        "WHERE source_kind = 'producer' AND SUBSTR(emitted_at, 1, 10) = ? "
        "GROUP BY source_name",
        (date_iso,),
    ).fetchall()
    tool_usage = {str(t): {"count": int(c)} for t, c in tool_rows if t}
    rhythm: dict[str, int] = {}
    rhythm_rows = conn.execute(
        "SELECT captured_at FROM frame_traces WHERE SUBSTR(captured_at, 1, 10) = ?",
        (date_iso,),
    ).fetchall()
    for (captured_at,) in rhythm_rows:
        try:
            dt = datetime.fromisoformat(str(captured_at).replace("Z", "+00:00"))
            key = f"{dt.weekday()}:{dt.hour:02d}"
            rhythm[key] = rhythm.get(key, 0) + 1
        except (TypeError, ValueError):
            continue
    stage_rows = conn.execute(
        "SELECT stage_name, entered_at, exited_at FROM stage_traces "
        "WHERE exited_at IS NOT NULL AND SUBSTR(entered_at, 1, 10) = ?",
        (date_iso,),
    ).fetchall()
    stage_accum: dict[str, list[float]] = {}
    for stage_name, entered_at, exited_at in stage_rows:
        try:
            a = datetime.fromisoformat(str(entered_at).replace("Z", "+00:00"))
            b = datetime.fromisoformat(str(exited_at).replace("Z", "+00:00"))
            delta = (b - a).total_seconds() * 1000.0
            if delta >= 0:
                stage_accum.setdefault(str(stage_name), []).append(delta)
        except (TypeError, ValueError):
            continue
    stage_latency = {
        name: {"mean_ms": sum(d) / len(d), "count": len(d)}
        for name, d in stage_accum.items() if d
    }
    conversation_dist: dict[str, int] = {}
    linguency_dist: dict[str, int] = {}

    conn.execute(
        "INSERT OR REPLACE INTO daily_rollups ("
        "date, turns, total_chars, ratings_count, ratings_sum, "
        "ratings_histogram_json, effort_distribution_json, "
        "conversation_mode_dist_json, reasoning_mode_dist_json, "
        "linguency_mode_dist_json, tool_usage_json, fault_summary_json, "
        "time_rhythm_json, stage_latency_json, computed_at"
        ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (
            date_iso, turns, total_chars, ratings_count, ratings_sum,
            json.dumps(ratings_hist), json.dumps(effort_dist),
            json.dumps(conversation_dist), json.dumps(reasoning_dist),
            json.dumps(linguency_dist), json.dumps(tool_usage),
            json.dumps(fault_summary), json.dumps(rhythm),
            json.dumps(stage_latency), datetime.now(timezone.utc).isoformat(),
        ),
    )
    check_achievements(conn, date_iso)


def _column_distribution(conn: sqlite3.Connection, column: str, date_iso: str) -> dict[str, int]:
    rows = conn.execute(
        f"SELECT {column}, COUNT(*) FROM frame_traces "
        f"WHERE {column} IS NOT NULL AND SUBSTR(captured_at, 1, 10) = ? "
        f"GROUP BY {column}",
        (date_iso,),
    ).fetchall()
    return {str(k): int(v) for k, v in rows if k}


_MILESTONES_TURNS = (100, 250, 500, 1000, 2500, 5000, 10000)
_MILESTONES_TOKENS = (100_000, 500_000, 1_000_000, 5_000_000)
_STREAK_MILESTONES = (7, 30, 100)


def check_achievements(conn: sqlite3.Connection, date_iso: str) -> list[dict[str, Any]]:
    """Evaluate unlock conditions against the state as of `date_iso`. Inserts
    any new achievements into the achievements table and returns the inserted
    rows. Idempotent — checks `description` uniqueness within `tag` before
    inserting so re-running doesn't double-unlock."""
    inserted: list[dict[str, Any]] = []

    row = conn.execute(
        "SELECT COUNT(*), COALESCE(SUM(total_chars), 0) "
        "FROM frame_traces WHERE SUBSTR(captured_at, 1, 10) <= ?",
        (date_iso,),
    ).fetchone()
    lifetime_turns = int(row[0] or 0)
    lifetime_tokens = int(row[1] or 0) // 4

    for threshold in _MILESTONES_TURNS:
        if lifetime_turns >= threshold:
            desc = f"{threshold} lifetime turns"
            if _insert_if_new(conn, date_iso, "MILESTONE", desc, {"threshold": threshold}):
                inserted.append({"tag": "MILESTONE", "description": desc})
    for threshold in _MILESTONES_TOKENS:
        if lifetime_tokens >= threshold:
            desc = f"{threshold:,} lifetime tokens"
            if _insert_if_new(conn, date_iso, "MILESTONE", desc, {"threshold": threshold, "kind": "tokens"}):
                inserted.append({"tag": "MILESTONE", "description": desc})

    days_with_turns = {
        r[0] for r in conn.execute(
            "SELECT DISTINCT SUBSTR(captured_at, 1, 10) FROM frame_traces "
            "WHERE SUBSTR(captured_at, 1, 10) <= ?",
            (date_iso,),
        ).fetchall()
    }
    streak = 0
    cursor = _date_cls.fromisoformat(date_iso)
    while cursor.isoformat() in days_with_turns:
        streak += 1
        cursor -= timedelta(days=1)
    for threshold in _STREAK_MILESTONES:
        if streak >= threshold:
            desc = f"{threshold}-day streak"
            if _insert_if_new(conn, date_iso, "STREAK", desc, {"threshold": threshold, "streak_at_unlock": streak}):
                inserted.append({"tag": "STREAK", "description": desc})

    return inserted


def _insert_if_new(conn: sqlite3.Connection, date_iso: str, tag: str,
                   description: str, payload: dict) -> bool:
    existing = conn.execute(
        "SELECT 1 FROM achievements WHERE tag = ? AND description = ?",
        (tag, description),
    ).fetchone()
    if existing:
        return False
    conn.execute(
        "INSERT INTO achievements (unlocked_at, tag, description, payload_json) "
        "VALUES (?, ?, ?, ?)",
        (date_iso, tag, description, json.dumps(payload)),
    )
    return True
