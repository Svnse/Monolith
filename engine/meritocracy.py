from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path
from typing import Any

from core.paths import LOG_DIR


class CompetenceHorizon:
    TIERS = {
        "C": {"budget": 12, "tools": ["read_file", "search"]},
        "B": {"budget": 25, "tools": ["read_file", "search", "write_file"]},
        "A": {"budget": 40, "tools": ["read_file", "write_file", "run_cmd"]},
        "S": {"budget": 60, "tools": ["read_file", "write_file", "run_cmd", "sub_agent"]},
    }

    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else LOG_DIR / "overseer.sqlite3"
        self._create_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _create_schema(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS agent_merit (
                    user_id TEXT PRIMARY KEY,
                    tier TEXT DEFAULT 'C',
                    rolling_avg_score REAL DEFAULT 50.0,
                    consecutive_high_scores INTEGER DEFAULT 0,
                    consecutive_low_scores INTEGER DEFAULT 0,
                    total_runs INTEGER DEFAULT 0,
                    lifetime_tokens_used INTEGER DEFAULT 0
                )
                """
            )

    def resolve_config(self, user_id: str) -> dict[str, Any]:
        with self._connect() as conn:
            row = conn.execute("SELECT tier FROM agent_merit WHERE user_id = ?", (user_id,)).fetchone()
            if row is None:
                conn.execute("INSERT INTO agent_merit(user_id) VALUES(?)", (user_id,))
                tier = "C"
            else:
                tier = row["tier"] if row["tier"] in self.TIERS else "C"

        tier_config = self.TIERS[tier]
        tier_budget = tier_config["budget"]
        tier_tools = list(tier_config["tools"])

        return {
            "max_steps": tier_budget,
            "tools": tier_tools,
            "system_prompt_suffix": f"\n[GOVERNANCE] Tier={tier} Budget={tier_budget}",
        }

    def record_score(self, user_id: str, score: float) -> str:
        score = max(0.0, min(100.0, float(score)))
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM agent_merit WHERE user_id = ?", (user_id,)).fetchone()
            if row is None:
                conn.execute("INSERT INTO agent_merit(user_id) VALUES(?)", (user_id,))
                row = conn.execute("SELECT * FROM agent_merit WHERE user_id = ?", (user_id,)).fetchone()

            tier = row["tier"] if row["tier"] in self.TIERS else "C"
            total_runs = int(row["total_runs"] or 0) + 1
            rolling = float(row["rolling_avg_score"] or 50.0)
            rolling = ((rolling * (total_runs - 1)) + score) / total_runs

            high = int(row["consecutive_high_scores"] or 0)
            low = int(row["consecutive_low_scores"] or 0)

            if score >= self._promotion_threshold(tier):
                high += 1
                low = 0
            elif score < self._demotion_threshold(tier):
                low += 1
                high = 0
            else:
                high = 0
                low = 0

            new_tier = self._apply_tier_transition(tier, high, low)
            if new_tier != tier:
                high = 0
                low = 0

            conn.execute(
                """
                UPDATE agent_merit
                SET tier = ?,
                    rolling_avg_score = ?,
                    consecutive_high_scores = ?,
                    consecutive_low_scores = ?,
                    total_runs = ?
                WHERE user_id = ?
                """,
                (new_tier, rolling, high, low, total_runs, user_id),
            )

            return new_tier

    def update_after_trace(self, user_id: str, trace: dict[str, Any]) -> float:
        score = calculate_score(trace)
        if "CRITICAL_ERROR" in trace.get("error_flags", []):
            with self._connect() as conn:
                row = conn.execute("SELECT * FROM agent_merit WHERE user_id = ?", (user_id,)).fetchone()
                if row is None:
                    conn.execute("INSERT INTO agent_merit(user_id) VALUES(?)", (user_id,))
                    row = conn.execute("SELECT * FROM agent_merit WHERE user_id = ?", (user_id,)).fetchone()
                total_runs = int(row["total_runs"] or 0) + 1
                rolling = float(row["rolling_avg_score"] or 50.0)
                rolling = ((rolling * (total_runs - 1)) + score) / total_runs
                conn.execute(
                    """
                    UPDATE agent_merit
                    SET tier = 'C',
                        rolling_avg_score = ?,
                        consecutive_high_scores = 0,
                        consecutive_low_scores = 0,
                        total_runs = ?
                    WHERE user_id = ?
                    """,
                    (rolling, total_runs, user_id),
                )
            return score

        self.record_score(user_id, score)
        return score

    def _apply_tier_transition(self, tier: str, high: int, low: int) -> str:
        if tier == "C" and high >= 3:
            return "B"
        if tier == "B" and high >= 5:
            return "A"
        if tier == "A" and high >= 10:
            return "S"
        if tier == "S" and low >= 2:
            return "A"
        if tier == "A" and low >= 3:
            return "B"
        if tier == "B" and low >= 3:
            return "C"
        return tier

    def _promotion_threshold(self, tier: str) -> float:
        if tier == "C":
            return 80.0
        if tier == "B":
            return 85.0
        if tier == "A":
            return 95.0
        return 101.0

    def _demotion_threshold(self, tier: str) -> float:
        if tier == "S":
            return 70.0
        if tier == "A":
            return 60.0
        if tier == "B":
            return 50.0
        return -1.0


def calculate_score(trace: dict) -> float:
    steps_used = max(0, int(trace.get("steps_used", 0) or 0))
    budget = max(1, int(trace.get("budget", 1) or 1))
    tool_calls = trace.get("tool_calls", []) or []
    error_flags = set(trace.get("error_flags", []) or [])

    if "CRITICAL_ERROR" in error_flags or "UNRECOVERED_ERROR" in error_flags:
        correctness = 0.0
    else:
        correctness = 1.0

    if correctness == 0:
        velocity = 0.0
    else:
        velocity = max(0.0, (budget - steps_used) / budget)

    total_calls = len(tool_calls)
    if total_calls == 0:
        precision = 1.0
        stability = 1.0
    else:
        seen: set[str] = set()
        duplicate_calls = 0
        for call in tool_calls:
            name = str(call.get("name", ""))
            arguments = call.get("arguments", {})
            encoded = json.dumps(arguments, sort_keys=True, default=str)
            signature = hashlib.sha256(f"{name}|{encoded}".encode("utf-8")).hexdigest()
            if signature in seen:
                duplicate_calls += 1
            else:
                seen.add(signature)

        duplicate_ratio = duplicate_calls / total_calls
        precision = 1.0 - duplicate_ratio
        stability = 1.0 - duplicate_ratio

    score = (
        correctness * 0.40
        + velocity * 0.25
        + precision * 0.20
        + stability * 0.15
    ) * 100

    score = max(0.0, min(100.0, score))
    return round(score, 2)
