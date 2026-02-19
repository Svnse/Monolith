"""
Governance module — CompetenceHorizon tier system + Adaptive Governance Layer.

Phase 4: PerfVector evaluation, PolicyState mutation, regression detection,
checkpoint ring buffer, and rollback. All mutation is post-execution only.
"""

from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.paths import LOG_DIR


# ---------------------------------------------------------------------------
# Phase 4 dataclasses
# ---------------------------------------------------------------------------

@dataclass
class PolicyState:
    """Bounded, clamped policy parameters for a user."""
    user_id: str = ""
    tier: str = "C"
    retry_tolerance: float = 0.2
    step_limit_multiplier: float = 1.0
    autonomy_bias: float = 0.3
    verbosity_bias: float = 0.3
    critique_enabled: bool = False
    total_mutations: int = 0
    frozen_until_run: int = 0  # run count after which mutations resume


@dataclass(frozen=True)
class PolicyDelta:
    """Single mutation record, linked to a run. Dominance is GREEN or YELLOW only."""
    run_id: str
    user_id: str
    timestamp: float
    dominance: str
    mutations: tuple[tuple[str, float, float], ...]  # (param_name, old_value, new_value)


@dataclass(frozen=True)
class PolicyCheckpoint:
    """Snapshot of PolicyState + rolling performance metrics."""
    checkpoint_id: int
    user_id: str
    tier: str
    retry_tolerance: float
    step_limit_multiplier: float
    autonomy_bias: float
    verbosity_bias: float
    critique_enabled: bool
    rolling_composite_score: float
    run_count_at_checkpoint: int
    timestamp: float


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

TIER_CEILINGS: dict[str, dict[str, Any]] = {
    "C": {
        "retry_tolerance": (0.0, 0.3),
        "step_limit_multiplier": (0.8, 1.2),
        "autonomy_bias": (0.0, 0.3),
        "verbosity_bias": (0.0, 0.5),
        "critique_enabled": False,  # locked off at tier C
    },
    "B": {
        "retry_tolerance": (0.0, 0.5),
        "step_limit_multiplier": (0.7, 1.5),
        "autonomy_bias": (0.0, 0.5),
        "verbosity_bias": (0.0, 0.7),
        "critique_enabled": None,  # unlocked
    },
    "A": {
        "retry_tolerance": (0.0, 0.7),
        "step_limit_multiplier": (0.5, 1.8),
        "autonomy_bias": (0.0, 0.8),
        "verbosity_bias": (0.0, 0.9),
        "critique_enabled": None,
    },
    "S": {
        "retry_tolerance": (0.0, 1.0),
        "step_limit_multiplier": (0.5, 2.0),
        "autonomy_bias": (0.0, 1.0),
        "verbosity_bias": (0.0, 1.0),
        "critique_enabled": None,
    },
}

WARMUP_MIN_RUNS = 10
WARMUP_MAX_VARIANCE = 0.25
CHECKPOINT_RING_SIZE = 5
ROLLBACK_FREEZE_RUNS = 5
MUTATION_STEP_SIZE = 0.05
REGRESSION_THRESHOLDS = {"C": 0.20, "B": 0.15, "A": 0.12, "S": 0.10}


# ---------------------------------------------------------------------------
# CompetenceHorizon
# ---------------------------------------------------------------------------

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
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS run_summaries (
                    run_id TEXT PRIMARY KEY,
                    contract_id TEXT,
                    outcome TEXT,
                    termination_reason TEXT,
                    llm_calls INTEGER,
                    tool_calls INTEGER,
                    format_retries INTEGER,
                    steps_used INTEGER,
                    max_inferences INTEGER,
                    budget_remaining INTEGER,
                    elapsed_ms REAL,
                    start_time REAL,
                    end_time REAL,
                    had_protocol_error INTEGER,
                    had_validation_error INTEGER,
                    had_cycle_violation INTEGER,
                    had_budget_exhaustion INTEGER,
                    model_profile_id TEXT,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
                """
            )
            # Phase 4 tables
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS perf_vectors (
                    run_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    protocol_compliant INTEGER NOT NULL,
                    retry_count INTEGER NOT NULL,
                    budget_efficiency REAL NOT NULL,
                    duplicate_call_ratio REAL NOT NULL,
                    anomaly_ignored_count INTEGER NOT NULL,
                    hard_failure INTEGER NOT NULL,
                    dominance TEXT NOT NULL,
                    composite_score REAL NOT NULL,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_state (
                    user_id TEXT PRIMARY KEY,
                    tier TEXT NOT NULL DEFAULT 'C',
                    retry_tolerance REAL NOT NULL DEFAULT 0.2,
                    step_limit_multiplier REAL NOT NULL DEFAULT 1.0,
                    autonomy_bias REAL NOT NULL DEFAULT 0.3,
                    verbosity_bias REAL NOT NULL DEFAULT 0.3,
                    critique_enabled INTEGER NOT NULL DEFAULT 0,
                    total_mutations INTEGER NOT NULL DEFAULT 0,
                    frozen_until_run INTEGER NOT NULL DEFAULT 0,
                    last_updated REAL DEFAULT (strftime('%s', 'now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_deltas (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    dominance TEXT NOT NULL,
                    mutations TEXT NOT NULL,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS policy_checkpoints (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT NOT NULL,
                    tier TEXT NOT NULL,
                    retry_tolerance REAL NOT NULL,
                    step_limit_multiplier REAL NOT NULL,
                    autonomy_bias REAL NOT NULL,
                    verbosity_bias REAL NOT NULL,
                    critique_enabled INTEGER NOT NULL,
                    rolling_composite_score REAL NOT NULL,
                    run_count_at_checkpoint INTEGER NOT NULL,
                    created_at REAL DEFAULT (strftime('%s', 'now'))
                )
                """
            )

    # ------------------------------------------------------------------
    # Tier resolution (read-only + policy overlay)
    # ------------------------------------------------------------------

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

        state = self._load_policy_state(user_id)
        adjusted_budget = max(1, int(tier_budget * state.step_limit_multiplier))

        return {
            "max_steps": adjusted_budget,
            "tools": tier_tools,
            "system_prompt_suffix": f"\n[GOVERNANCE] Tier={tier} Budget={adjusted_budget}",
            "policy": {
                "retry_tolerance": state.retry_tolerance,
                "autonomy_bias": state.autonomy_bias,
                "verbosity_bias": state.verbosity_bias,
                "critique_enabled": state.critique_enabled,
                "step_limit_multiplier": state.step_limit_multiplier,
            },
        }

    # ------------------------------------------------------------------
    # RunSummary ingestion + Phase 4 governance pipeline
    # ------------------------------------------------------------------

    def consume_summary(self, summary: dict[str, Any], user_id: str = "default") -> None:
        """
        Store a RunSummary execution receipt and run the governance pipeline.

        Pipeline:
        1. Store summary
        2. Compute + store PerfVector
        3. Update agent_merit counters
        4. Check warmup → return if not met
        5. Load PolicyState; check freeze → return if frozen
        6. RED dominance → return (no mutation)
        7. Compute rolling score; check regression → rollback if detected
        8. Compute delta → apply if not None
        9. Save checkpoint
        """
        if not isinstance(summary, dict):
            return
        run_id = summary.get("run_id", "")
        if not run_id:
            return

        # Step 1: Store summary
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO run_summaries (
                    run_id, contract_id, outcome, termination_reason,
                    llm_calls, tool_calls, format_retries, steps_used,
                    max_inferences, budget_remaining, elapsed_ms,
                    start_time, end_time,
                    had_protocol_error, had_validation_error,
                    had_cycle_violation, had_budget_exhaustion,
                    model_profile_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    summary.get("contract_id", ""),
                    summary.get("outcome", ""),
                    summary.get("termination_reason", ""),
                    int(summary.get("llm_calls", 0)),
                    int(summary.get("tool_calls", 0)),
                    int(summary.get("format_retries", 0)),
                    int(summary.get("steps_used", 0)),
                    int(summary.get("max_inferences", 0)),
                    int(summary.get("budget_remaining", 0)),
                    float(summary.get("elapsed_ms", 0.0)),
                    float(summary.get("start_time", 0.0)),
                    float(summary.get("end_time", 0.0)),
                    int(summary.get("had_protocol_error", False)),
                    int(summary.get("had_validation_error", False)),
                    int(summary.get("had_cycle_violation", False)),
                    int(summary.get("had_budget_exhaustion", False)),
                    summary.get("model_profile_id", ""),
                ),
            )

        # Step 2: Compute and store PerfVector
        pv = self._compute_perf_vector(summary)
        self._store_perf_vector(pv, user_id)

        # Step 3: Update agent_merit counters
        with self._connect() as conn:
            row = conn.execute("SELECT total_runs FROM agent_merit WHERE user_id = ?", (user_id,)).fetchone()
            if row is None:
                conn.execute("INSERT INTO agent_merit(user_id) VALUES(?)", (user_id,))
                total_runs = 0
            else:
                total_runs = row["total_runs"]

            total_runs += 1
            conn.execute(
                "UPDATE agent_merit SET total_runs = ?, lifetime_tokens_used = lifetime_tokens_used + ? WHERE user_id = ?",
                (total_runs, int(summary.get("tokens_consumed", 0)), user_id),
            )

        # Step 4: Check warmup
        if not self._check_warmup(user_id):
            return

        # Step 5: Load policy state; check freeze
        state = self._load_policy_state(user_id)
        if state.frozen_until_run > 0 and total_runs < state.frozen_until_run:
            return

        # Step 6: RED dominance — no mutation
        if pv.dominance == "RED":
            return

        # Step 7: Compute rolling score; check regression
        rolling_score = self._rolling_composite_score(user_id, window=WARMUP_MIN_RUNS)
        if self._check_regression(user_id, rolling_score):
            self._rollback(user_id, total_runs)
            return

        # Step 8: Compute delta — apply if not None
        delta = self._compute_delta(pv, state)
        if delta is None:
            self._save_checkpoint(state, rolling_score, total_runs)
            return

        # Step 9: Apply delta, store, checkpoint
        state = self._apply_delta(state, delta)
        self._store_delta(delta)
        self._save_policy_state(state)
        self._save_checkpoint(state, rolling_score, total_runs)

    # ------------------------------------------------------------------
    # PerfVector computation (deterministic)
    # ------------------------------------------------------------------

    def _compute_perf_vector(self, summary: dict[str, Any]) -> "PerfVector":
        from engine.contract import PerfVector

        run_id = summary.get("run_id", "")
        outcome = summary.get("outcome", "")

        protocol_compliant = not summary.get("had_protocol_error", False)
        retry_count = int(summary.get("format_retries", 0))

        max_inf = int(summary.get("max_inferences", 1)) or 1
        inferences_used = max_inf - int(summary.get("budget_remaining", 0))
        budget_efficiency = max(0.0, min(1.0, inferences_used / max_inf))

        unique = int(summary.get("unique_tool_signatures", 0))
        total = int(summary.get("total_tool_invocations", 0))
        if total == 0:
            duplicate_call_ratio = 0.0
        else:
            duplicate_call_ratio = max(0.0, 1.0 - (unique / total))

        anomaly_count = 0
        if summary.get("had_validation_error", False):
            anomaly_count += 1
        if summary.get("had_cycle_violation", False):
            anomaly_count += 1

        hard_failure = outcome in (
            "FAILED_PREFLIGHT", "FAILED_PROTOCOL_NO_TOOLS",
            "FAILED_PROTOCOL_MALFORMED", "FAILED_VALIDATION",
            "FAILED_BUDGET_EXHAUSTED", "FAILED_TIMEOUT",
            "FAILED_CONTRACT_VIOLATION",
        )

        # Dominance
        if hard_failure or (not protocol_compliant and retry_count > 1 and budget_efficiency > 0.95):
            dominance = "RED"
        elif retry_count > 0 or duplicate_call_ratio > 0.3 or anomaly_count > 0 or budget_efficiency > 0.8:
            dominance = "YELLOW"
        else:
            dominance = "GREEN"

        # Composite score
        max_retries = int(summary.get("max_format_retries", 1)) or 1
        composite_score = (
            0.30 * (1.0 if protocol_compliant else 0.0)
            + 0.20 * (1.0 - budget_efficiency)
            + 0.15 * (1.0 - duplicate_call_ratio)
            + 0.15 * (1.0 - min(retry_count / max_retries, 1.0))
            + 0.10 * (1.0 if not hard_failure else 0.0)
            + 0.10 * (1.0 - min(anomaly_count / 5.0, 1.0))
        )

        return PerfVector(
            run_id=run_id,
            protocol_compliant=protocol_compliant,
            retry_count=retry_count,
            budget_efficiency=round(budget_efficiency, 4),
            duplicate_call_ratio=round(duplicate_call_ratio, 4),
            anomaly_ignored_count=anomaly_count,
            hard_failure=hard_failure,
            dominance=dominance,
            composite_score=round(composite_score, 4),
        )

    def _store_perf_vector(self, pv: "PerfVector", user_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO perf_vectors (
                    run_id, user_id, protocol_compliant, retry_count,
                    budget_efficiency, duplicate_call_ratio,
                    anomaly_ignored_count, hard_failure, dominance,
                    composite_score
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    pv.run_id, user_id, int(pv.protocol_compliant), pv.retry_count,
                    pv.budget_efficiency, pv.duplicate_call_ratio,
                    pv.anomaly_ignored_count, int(pv.hard_failure), pv.dominance,
                    pv.composite_score,
                ),
            )

    # ------------------------------------------------------------------
    # Warmup guard
    # ------------------------------------------------------------------

    def _check_warmup(self, user_id: str) -> bool:
        """Returns True if mutations are allowed (warmup complete)."""
        with self._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM perf_vectors WHERE user_id = ?", (user_id,)
            ).fetchone()[0]
            if count < WARMUP_MIN_RUNS:
                return False

            rows = conn.execute(
                "SELECT budget_efficiency FROM perf_vectors WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, WARMUP_MIN_RUNS),
            ).fetchall()
            values = [r["budget_efficiency"] for r in rows]
            if len(values) < WARMUP_MIN_RUNS:
                return False
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std_dev = variance ** 0.5
            return std_dev < WARMUP_MAX_VARIANCE

    # ------------------------------------------------------------------
    # PolicyState CRUD
    # ------------------------------------------------------------------

    def _load_policy_state(self, user_id: str) -> PolicyState:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM policy_state WHERE user_id = ?", (user_id,)).fetchone()
            if row is None:
                return PolicyState(user_id=user_id, tier="C")
            return PolicyState(
                user_id=row["user_id"],
                tier=row["tier"],
                retry_tolerance=row["retry_tolerance"],
                step_limit_multiplier=row["step_limit_multiplier"],
                autonomy_bias=row["autonomy_bias"],
                verbosity_bias=row["verbosity_bias"],
                critique_enabled=bool(row["critique_enabled"]),
                total_mutations=row["total_mutations"],
                frozen_until_run=row["frozen_until_run"],
            )

    def _save_policy_state(self, state: PolicyState) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO policy_state (
                    user_id, tier, retry_tolerance, step_limit_multiplier,
                    autonomy_bias, verbosity_bias, critique_enabled,
                    total_mutations, frozen_until_run, last_updated
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.user_id, state.tier, state.retry_tolerance,
                    state.step_limit_multiplier, state.autonomy_bias,
                    state.verbosity_bias, int(state.critique_enabled),
                    state.total_mutations, state.frozen_until_run, time.time(),
                ),
            )

    def _clamp_policy(self, state: PolicyState) -> PolicyState:
        """Enforce tier ceilings on all policy parameters."""
        ceilings = TIER_CEILINGS.get(state.tier, TIER_CEILINGS["C"])
        for param in ("retry_tolerance", "step_limit_multiplier", "autonomy_bias", "verbosity_bias"):
            lo, hi = ceilings[param]
            current = getattr(state, param)
            setattr(state, param, round(max(lo, min(hi, current)), 4))
        if ceilings["critique_enabled"] is False:
            state.critique_enabled = False
        return state

    # ------------------------------------------------------------------
    # Mutation pipeline
    # ------------------------------------------------------------------

    def _compute_delta(self, pv: "PerfVector", state: PolicyState) -> PolicyDelta | None:
        """
        Compute a policy delta from PerfVector signals.

        Only GREEN or YELLOW dominance. Never RED.
        No contradictory mutations in a single delta.
        """
        mutations: list[tuple[str, float, float]] = []
        mutated_params: set[str] = set()

        def _propose(param: str, direction: float) -> None:
            if param in mutated_params:
                return
            old_val = getattr(state, param)
            new_val = round(old_val + direction * MUTATION_STEP_SIZE, 4)
            # Clamp to tier
            ceilings = TIER_CEILINGS.get(state.tier, TIER_CEILINGS["C"])
            lo, hi = ceilings[param]
            new_val = round(max(lo, min(hi, new_val)), 4)
            if new_val != old_val:
                mutations.append((param, old_val, new_val))
                mutated_params.add(param)

        if pv.dominance == "GREEN":
            # Efficient run — tighten budget if wasteful
            if pv.budget_efficiency < 0.5:
                _propose("step_limit_multiplier", -1.0)
            # Clean tools — boost autonomy
            if pv.duplicate_call_ratio == 0.0 and pv.retry_count == 0:
                _propose("autonomy_bias", +1.0)

        elif pv.dominance == "YELLOW":
            # Retries present — increase tolerance
            if pv.retry_count > 0:
                _propose("retry_tolerance", +1.0)
            # High budget usage — increase limit
            if pv.budget_efficiency > 0.8:
                _propose("step_limit_multiplier", +1.0)
            # Duplicates — decrease autonomy
            if pv.duplicate_call_ratio > 0.3:
                _propose("autonomy_bias", -1.0)

        if not mutations:
            return None

        return PolicyDelta(
            run_id=pv.run_id,
            user_id=state.user_id,
            timestamp=time.time(),
            dominance=pv.dominance,
            mutations=tuple(mutations),
        )

    def _apply_delta(self, state: PolicyState, delta: PolicyDelta) -> PolicyState:
        """Apply mutations from delta to state, clamp, return updated state."""
        for param, _old, new_val in delta.mutations:
            setattr(state, param, new_val)
        state.total_mutations += 1
        state = self._clamp_policy(state)
        return state

    def _store_delta(self, delta: PolicyDelta) -> None:
        mutations_json = json.dumps([(p, o, n) for p, o, n in delta.mutations])
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO policy_deltas (run_id, user_id, dominance, mutations)
                VALUES (?, ?, ?, ?)
                """,
                (delta.run_id, delta.user_id, delta.dominance, mutations_json),
            )

    # ------------------------------------------------------------------
    # Checkpoint ring buffer
    # ------------------------------------------------------------------

    def _save_checkpoint(self, state: PolicyState, rolling_score: float, run_count: int) -> None:
        with self._connect() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM policy_checkpoints WHERE user_id = ?",
                (state.user_id,),
            ).fetchone()[0]

            if count >= CHECKPOINT_RING_SIZE:
                conn.execute(
                    "DELETE FROM policy_checkpoints WHERE id = ("
                    "  SELECT id FROM policy_checkpoints WHERE user_id = ? ORDER BY created_at ASC LIMIT 1"
                    ")",
                    (state.user_id,),
                )

            conn.execute(
                """
                INSERT INTO policy_checkpoints (
                    user_id, tier, retry_tolerance, step_limit_multiplier,
                    autonomy_bias, verbosity_bias, critique_enabled,
                    rolling_composite_score, run_count_at_checkpoint
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    state.user_id, state.tier, state.retry_tolerance,
                    state.step_limit_multiplier, state.autonomy_bias,
                    state.verbosity_bias, int(state.critique_enabled),
                    rolling_score, run_count,
                ),
            )

    def _rolling_composite_score(self, user_id: str, window: int = WARMUP_MIN_RUNS) -> float:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT composite_score FROM perf_vectors WHERE user_id = ? ORDER BY created_at DESC LIMIT ?",
                (user_id, window),
            ).fetchall()
            if not rows:
                return 0.0
            return sum(r["composite_score"] for r in rows) / len(rows)

    # ------------------------------------------------------------------
    # Regression detection + rollback
    # ------------------------------------------------------------------

    def _check_regression(self, user_id: str, current_rolling_score: float) -> bool:
        """Returns True if regression detected (rollback needed)."""
        with self._connect() as conn:
            best_row = conn.execute(
                "SELECT MAX(rolling_composite_score) as best FROM policy_checkpoints WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if best_row is None or best_row["best"] is None:
                return False

            best_score = best_row["best"]
            tier_row = conn.execute("SELECT tier FROM agent_merit WHERE user_id = ?", (user_id,)).fetchone()
            tier = tier_row["tier"] if tier_row else "C"
            threshold = REGRESSION_THRESHOLDS.get(tier, 0.20)

            if best_score <= 0:
                return False
            relative_drop = (best_score - current_rolling_score) / best_score
            return relative_drop > threshold

    def _rollback(self, user_id: str, total_runs: int) -> None:
        """Restore PolicyState from the best checkpoint and freeze mutations."""
        with self._connect() as conn:
            best = conn.execute(
                "SELECT * FROM policy_checkpoints WHERE user_id = ? ORDER BY rolling_composite_score DESC LIMIT 1",
                (user_id,),
            ).fetchone()
            if best is None:
                return

            state = PolicyState(
                user_id=user_id,
                tier=best["tier"],
                retry_tolerance=best["retry_tolerance"],
                step_limit_multiplier=best["step_limit_multiplier"],
                autonomy_bias=best["autonomy_bias"],
                verbosity_bias=best["verbosity_bias"],
                critique_enabled=bool(best["critique_enabled"]),
                frozen_until_run=total_runs + ROLLBACK_FREEZE_RUNS,
            )
            # Preserve total_mutations from current state
            current = self._load_policy_state(user_id)
            state.total_mutations = current.total_mutations

            self._save_policy_state(state)

    # ------------------------------------------------------------------
    # Dormant: tier ladder (defined but inactive)
    # ------------------------------------------------------------------

    def _apply_tier_transition(self, tier: str, high: int, low: int) -> str:
        """Tier promotion/demotion rules (dormant)."""
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
        """Score threshold for promotion (dormant)."""
        if tier == "C":
            return 80.0
        if tier == "B":
            return 85.0
        if tier == "A":
            return 95.0
        return 101.0

    def _demotion_threshold(self, tier: str) -> float:
        """Score threshold for demotion (dormant)."""
        if tier == "S":
            return 70.0
        if tier == "A":
            return 60.0
        if tier == "B":
            return 50.0
        return -1.0
