"""StatsStore — query API for the stats addon.

Reads from turn_trace.sqlite3's frame_traces, outcome_traces, fault_traces,
stage_traces, and the daily_rollups + achievements tables. Pure data layer:
no UI, no signals, no theme. Reuses turn_trace._get_conn() — does NOT open
a parallel connection (SQLite write contention risk).

Today's data is always computed live from frame_traces; past days are read
from daily_rollups when available, falling back to live computation if the
materializer hasn't filled that day yet.
"""
from __future__ import annotations

from datetime import date as _date_cls, datetime, timezone, timedelta
from typing import Any

from core import turn_trace as tt


def _today_iso() -> str:
    """Today's date in UTC, ISO format (YYYY-MM-DD). Pulled out as a helper
    so tests can monkeypatch it to a fixed value without freezing the system
    clock."""
    return datetime.now(timezone.utc).date().isoformat()


def _range_to_dates(range_key: str) -> tuple[str, str]:
    """Convert a range key ('all' | 'year' | 'month' | 'week') into
    (start_iso, end_iso). 'all' returns ('0001-01-01', today)."""
    today = _date_cls.fromisoformat(_today_iso())
    if range_key == "week":
        start = today - timedelta(days=7)
    elif range_key == "month":
        start = today - timedelta(days=30)
    elif range_key == "year":
        start = today - timedelta(days=365)
    else:
        start = _date_cls(1, 1, 1)
    return (start.isoformat(), today.isoformat())


def _bucket_rating(value: int) -> str:
    """Map 0-100 rating into 10-bucket label. Top bucket is 90-100."""
    if value >= 90:
        return "90-100"
    lo = (value // 10) * 10
    hi = lo + 9
    return f"{lo}-{hi}"


class StatsStore:
    """Query API for the stats addon. Stateless aside from the shared
    turn_trace connection."""

    def get_lifetime_summary(self) -> dict[str, Any]:
        """Total turns, total chars, first/last turn dates, distinct days."""
        conn = tt._get_conn()
        if conn is None:
            return {"turns": 0, "total_chars": 0, "first_turn_date": None,
                    "last_turn_date": None, "day_count": 0}
        with tt._db_lock:
            row = conn.execute(
                "SELECT COUNT(*), COALESCE(SUM(total_chars), 0), "
                "MIN(SUBSTR(captured_at, 1, 10)), "
                "MAX(SUBSTR(captured_at, 1, 10)), "
                "COUNT(DISTINCT SUBSTR(captured_at, 1, 10)) "
                "FROM frame_traces"
            ).fetchone()
        return {
            "turns": int(row[0] or 0),
            "total_chars": int(row[1] or 0),
            "first_turn_date": row[2],
            "last_turn_date": row[3],
            "day_count": int(row[4] or 0),
        }

    def get_streak(self) -> int:
        """Consecutive days through today with >= 1 turn. Returns 0 if today
        or any day in the streak has zero turns."""
        conn = tt._get_conn()
        if conn is None:
            return 0
        with tt._db_lock:
            rows = conn.execute(
                "SELECT DISTINCT SUBSTR(captured_at, 1, 10) AS d "
                "FROM frame_traces "
                "ORDER BY d DESC"
            ).fetchall()
        days_with_turns = {r[0] for r in rows}
        today = _today_iso()
        streak = 0
        cursor = _date_cls.fromisoformat(today)
        while cursor.isoformat() in days_with_turns:
            streak += 1
            cursor -= timedelta(days=1)
        return streak

    def get_rating_histogram(self, range_key: str) -> dict[str, int]:
        """Bucket ratings into 0-9, 10-19, ..., 90-100 within the range."""
        start, end = _range_to_dates(range_key)
        conn = tt._get_conn()
        if conn is None:
            return {}
        with tt._db_lock:
            rows = conn.execute(
                "SELECT rating_value FROM outcome_traces "
                "WHERE kind = 'rating' AND rating_value IS NOT NULL "
                "AND SUBSTR(recorded_at, 1, 10) BETWEEN ? AND ?",
                (start, end),
            ).fetchall()
        hist: dict[str, int] = {}
        for (val,) in rows:
            label = _bucket_rating(int(val))
            hist[label] = hist.get(label, 0) + 1
        return hist

    _PLANE_COLUMN = {
        "effort":       "effort_tier",
        "reasoning":    "reasoning_mode",
    }

    def get_effort_distribution(self, range_key: str) -> dict[str, int]:
        """Counts of turns per effort tier within the range."""
        return self._distribution_for_plane("effort", range_key)

    def get_mode_distribution(self, plane: str, range_key: str) -> dict[str, int]:
        """Counts of turns per mode within the range, for the given plane."""
        return self._distribution_for_plane(plane, range_key)

    def _distribution_for_plane(self, plane: str, range_key: str) -> dict[str, int]:
        col = self._PLANE_COLUMN.get(plane)
        if col is None:
            return {}
        start, end = _range_to_dates(range_key)
        conn = tt._get_conn()
        if conn is None:
            return {}
        with tt._db_lock:
            rows = conn.execute(
                f"SELECT {col}, COUNT(*) FROM frame_traces "
                f"WHERE {col} IS NOT NULL "
                f"AND SUBSTR(captured_at, 1, 10) BETWEEN ? AND ? "
                f"GROUP BY {col}",
                (start, end),
            ).fetchall()
        return {str(name): int(count) for name, count in rows if name}

    def get_fault_summary(self, range_key: str) -> dict[str, int]:
        """Counts of fault_traces rows grouped by fault_kind within range."""
        start, end = _range_to_dates(range_key)
        conn = tt._get_conn()
        if conn is None:
            return {}
        with tt._db_lock:
            rows = conn.execute(
                "SELECT fault_kind, COUNT(*) FROM fault_traces "
                "WHERE fault_kind IS NOT NULL "
                "AND SUBSTR(emitted_at, 1, 10) BETWEEN ? AND ? "
                "GROUP BY fault_kind",
                (start, end),
            ).fetchall()
        return {str(kind): int(count) for kind, count in rows if kind}

    def get_tool_usage(self, range_key: str, top_n: int = 5) -> list[dict[str, Any]]:
        """Top-N tool source names by fault_traces count within range.
        Returns: [{tool, count, success_rate}, ...]
        Success rate is computed as 1 - (hard_failure + tool_no_fire) / total."""
        start, end = _range_to_dates(range_key)
        conn = tt._get_conn()
        if conn is None:
            return []
        with tt._db_lock:
            rows = conn.execute(
                "SELECT source_name, COUNT(*) AS total, "
                "SUM(CASE WHEN fault_kind IN ('hard_failure', 'tool_no_fire') THEN 1 ELSE 0 END) AS failed "
                "FROM fault_traces "
                "WHERE source_kind = 'producer' "
                "AND SUBSTR(emitted_at, 1, 10) BETWEEN ? AND ? "
                "GROUP BY source_name "
                "ORDER BY total DESC LIMIT ?",
                (start, end, top_n),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for source_name, total, failed in rows:
            total_int = int(total or 0)
            failed_int = int(failed or 0)
            success = 1.0 if total_int == 0 else max(0.0, 1.0 - failed_int / total_int)
            out.append({"tool": str(source_name), "count": total_int, "success_rate": success})
        return out

    def get_time_rhythm(self, range_key: str) -> dict[tuple[int, int], int]:
        """Per-(weekday, hour) turn density across the range. Weekday is
        0=Monday through 6=Sunday."""
        start, end = _range_to_dates(range_key)
        conn = tt._get_conn()
        if conn is None:
            return {}
        with tt._db_lock:
            rows = conn.execute(
                "SELECT captured_at FROM frame_traces "
                "WHERE SUBSTR(captured_at, 1, 10) BETWEEN ? AND ?",
                (start, end),
            ).fetchall()
        out: dict[tuple[int, int], int] = {}
        for (captured_at,) in rows:
            try:
                dt = datetime.fromisoformat(str(captured_at).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                continue
            key = (dt.weekday(), dt.hour)
            out[key] = out.get(key, 0) + 1
        return out

    def get_pipeline_cost_breakdown(self, range_key: str) -> dict[str, dict[str, float]]:
        """Mean stage latency per stage_name across the range."""
        start, end = _range_to_dates(range_key)
        conn = tt._get_conn()
        if conn is None:
            return {}
        with tt._db_lock:
            rows = conn.execute(
                "SELECT stage_name, entered_at, exited_at FROM stage_traces "
                "WHERE exited_at IS NOT NULL "
                "AND SUBSTR(entered_at, 1, 10) BETWEEN ? AND ?",
                (start, end),
            ).fetchall()
        accum: dict[str, list[float]] = {}
        for stage_name, entered_at, exited_at in rows:
            try:
                a = datetime.fromisoformat(str(entered_at).replace("Z", "+00:00"))
                b = datetime.fromisoformat(str(exited_at).replace("Z", "+00:00"))
            except (TypeError, ValueError):
                continue
            delta_ms = (b - a).total_seconds() * 1000.0
            if delta_ms < 0:
                continue
            accum.setdefault(str(stage_name), []).append(delta_ms)
        out = {}
        for stage_name, deltas in accum.items():
            if not deltas:
                continue
            out[stage_name] = {"mean_ms": sum(deltas) / len(deltas), "count": len(deltas)}
        return out

    def get_personal_records(self) -> dict[str, dict[str, Any]]:
        """Six 'personal record' cards. All six keys are always present."""
        conn = tt._get_conn()
        # Initialize all six keys with defaults so they're always present
        out: dict[str, dict[str, Any]] = {
            "biggest_prompt":     {"value": 0, "subtitle": "chars", "source_id": None},
            "highest_rated_turn": {"value": 0, "subtitle": "", "source_id": None},
            "best_day":           {"value": 0, "subtitle": "turns", "source_id": None},
            "longest_chat":       {"value": 0, "subtitle": "turns (best_day proxy)", "source_id": None},
            "longest_tool_chain": {"value": 0, "subtitle": "distinct tools", "source_id": None},
            "oldest_active_pin":  {"value": 0, "subtitle": "see SubstrateBlock", "source_id": None},
        }
        if conn is None:
            return out
        with tt._db_lock:
            row = conn.execute(
                "SELECT turn_id, total_chars FROM frame_traces "
                "ORDER BY total_chars DESC LIMIT 1"
            ).fetchone()
            if row is not None:
                out["biggest_prompt"] = {"value": int(row[1]), "subtitle": "chars", "source_id": row[0]}
            row = conn.execute(
                "SELECT turn_id, rating_value, reason FROM outcome_traces "
                "WHERE kind = 'rating' AND rating_value IS NOT NULL "
                "ORDER BY rating_value DESC LIMIT 1"
            ).fetchone()
            if row is not None:
                out["highest_rated_turn"] = {
                    "value": int(row[1]),
                    "subtitle": str(row[2] or "")[:80],
                    "source_id": row[0],
                }
            row = conn.execute(
                "SELECT SUBSTR(captured_at, 1, 10) AS d, COUNT(*) AS n "
                "FROM frame_traces GROUP BY d ORDER BY n DESC LIMIT 1"
            ).fetchone()
            if row is not None:
                out["best_day"] = {"value": int(row[1]), "subtitle": "turns", "source_id": row[0]}
            out["longest_chat"] = {
                "value": out.get("best_day", {}).get("value", 0),
                "subtitle": "turns (best_day proxy)",
                "source_id": out.get("best_day", {}).get("source_id"),
            }
            row = conn.execute(
                "SELECT turn_id, COUNT(DISTINCT source_name) AS n "
                "FROM fault_traces WHERE source_kind = 'producer' "
                "GROUP BY turn_id ORDER BY n DESC LIMIT 1"
            ).fetchone()
            if row is not None:
                out["longest_tool_chain"] = {
                    "value": int(row[1]), "subtitle": "distinct tools", "source_id": row[0],
                }
        return out

    def get_substrate_summary(self) -> dict[str, Any]:
        """Three sub-blocks: backends, monothink, continuity. Best-effort:
        any sub-block failing returns {} for that block, not raising."""
        from pathlib import Path
        import json
        out: dict[str, Any] = {"backends": {}, "monothink": {}, "continuity": {}}

        conn = tt._get_conn()
        if conn is not None:
            try:
                with tt._db_lock:
                    rows = conn.execute(
                        "SELECT backend, COUNT(*) FROM frame_traces GROUP BY backend"
                    ).fetchall()
                total = sum(int(c) for _, c in rows) or 1
                out["backends"] = {
                    str(b): {"count": int(c), "pct": int(c) * 100 // total}
                    for b, c in rows
                }
            except Exception:
                out["backends"] = {}

        try:
            journal_path = Path(__file__).resolve().parent.parent / "prompts" / "reasoning" / "monothink.journal.jsonl"
            if journal_path.exists():
                applied = 0
                rejected = 0
                for line in journal_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        entry = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if entry.get("applied"):
                        applied += 1
                    elif entry.get("reject_reason"):
                        rejected += 1
                scaffold_path = journal_path.parent / "monothink.md"
                scaffold_chars = len(scaffold_path.read_text(encoding="utf-8")) if scaffold_path.exists() else 0
                from core.monothink import _SIZE_CAP as _mt_cap
                out["monothink"] = {
                    "applied": applied, "rejected": rejected,
                    "scaffold_chars": scaffold_chars, "cap_chars": _mt_cap,
                }
        except Exception:
            out["monothink"] = {}

        try:
            from core import paths as _paths
            continuity_path = Path(_paths.CONFIG_DIR) / "continuity.json"
            if continuity_path.exists():
                data = json.loads(continuity_path.read_text(encoding="utf-8"))
                pins = data.get("pins", []) if isinstance(data, dict) else []
                active: dict[str, int] = {}
                retired = 0
                for pin in pins:
                    if not isinstance(pin, dict):
                        continue
                    if pin.get("active", True):
                        cat = str(pin.get("category", "lesson"))
                        active[cat] = active.get(cat, 0) + 1
                    else:
                        retired += 1
                out["continuity"] = {"active": active, "retired": retired}
        except Exception:
            out["continuity"] = {}

        return out

    def get_rating_trend(self, days: int = 30) -> list[tuple[str, float]]:
        """List of (date, mean_rating) for the last `days` days, ascending."""
        today = _date_cls.fromisoformat(_today_iso())
        start = today - timedelta(days=days)
        conn = tt._get_conn()
        if conn is None:
            return []
        with tt._db_lock:
            rows = conn.execute(
                "SELECT SUBSTR(recorded_at, 1, 10) AS d, "
                "AVG(rating_value) AS mean_rating "
                "FROM outcome_traces "
                "WHERE kind = 'rating' AND rating_value IS NOT NULL "
                "AND SUBSTR(recorded_at, 1, 10) BETWEEN ? AND ? "
                "GROUP BY d ORDER BY d ASC",
                (start.isoformat(), today.isoformat()),
            ).fetchall()
        return [(r[0], float(r[1])) for r in rows]

    def refresh(self) -> int:
        """Materialize any missing daily_rollups rows. Returns the count of
        days materialized. Delegates to core.stats_rollup."""
        from core.stats_rollup import materialize_missing_days
        conn = tt._get_conn()
        if conn is None:
            return 0
        return materialize_missing_days(conn)

    def get_achievements(self, limit: int = 8) -> list[dict[str, Any]]:
        """Newest-first list of bracket-tagged unlock events."""
        import json
        conn = tt._get_conn()
        if conn is None:
            return []
        with tt._db_lock:
            rows = conn.execute(
                "SELECT id, unlocked_at, tag, description, payload_json "
                "FROM achievements ORDER BY unlocked_at DESC, id DESC LIMIT ?",
                (int(limit),),
            ).fetchall()
        out = []
        for row in rows:
            try:
                payload = json.loads(row[4]) if row[4] else {}
            except json.JSONDecodeError:
                payload = {}
            out.append({
                "id": int(row[0]),
                "unlocked_at": str(row[1]),
                "tag": str(row[2]),
                "description": str(row[3]),
                "payload": payload,
            })
        return out
