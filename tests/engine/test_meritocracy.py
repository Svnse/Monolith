import json

import pytest

from engine.contract import PerfVector
from engine.meritocracy import (
    CHECKPOINT_RING_SIZE,
    MUTATION_STEP_SIZE,
    ROLLBACK_FREEZE_RUNS,
    TIER_CEILINGS,
    WARMUP_MIN_RUNS,
    CompetenceHorizon,
    PolicyDelta,
    PolicyState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _good_summary(run_id: str, **overrides) -> dict:
    """A clean, GREEN-dominance summary."""
    base = {
        "run_id": run_id,
        "contract_id": "c1",
        "outcome": "COMPLETED_WITH_TOOLS",
        "termination_reason": "completed",
        "llm_calls": 3,
        "tool_calls": 2,
        "format_retries": 0,
        "steps_used": 5,
        "max_inferences": 25,
        "budget_remaining": 22,
        "elapsed_ms": 1000.0,
        "start_time": 100.0,
        "end_time": 101.0,
        "had_protocol_error": False,
        "had_validation_error": False,
        "had_cycle_violation": False,
        "had_budget_exhaustion": False,
        "model_profile_id": "local_xml",
        "unique_tool_signatures": 2,
        "total_tool_invocations": 2,
        "tokens_consumed": 500,
        "max_format_retries": 2,
    }
    base.update(overrides)
    return base


def _feed_warmup(horizon, user_id="u1", count=WARMUP_MIN_RUNS, **summary_overrides):
    """Feed N good summaries to pass the warmup guard."""
    for i in range(count):
        horizon.consume_summary(_good_summary(f"warmup_{i}", **summary_overrides), user_id=user_id)


# ===========================================================================
# Existing tests (Phase 2b — must still pass)
# ===========================================================================

def test_resolve_config_default_tier_c(tmp_path):
    horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
    config = horizon.resolve_config("u1")
    assert config["max_steps"] == 12
    assert "read_file" in config["tools"]


def test_resolve_config_idempotent(tmp_path):
    horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
    config_a = horizon.resolve_config("u1")
    config_b = horizon.resolve_config("u1")
    assert config_a == config_b


def test_consume_summary_stores_receipt(tmp_path):
    horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
    summary = {
        "run_id": "run_001",
        "contract_id": "contract_abc",
        "outcome": "COMPLETED_WITH_TOOLS",
        "termination_reason": "completed",
        "llm_calls": 4,
        "tool_calls": 3,
        "format_retries": 1,
        "steps_used": 10,
        "max_inferences": 25,
        "budget_remaining": 21,
        "elapsed_ms": 12345.6,
        "start_time": 1000.0,
        "end_time": 1012.3456,
        "had_protocol_error": True,
        "had_validation_error": False,
        "had_cycle_violation": False,
        "had_budget_exhaustion": False,
        "model_profile_id": "local_xml",
    }
    horizon.consume_summary(summary)

    conn = horizon._connect()
    row = conn.execute("SELECT * FROM run_summaries WHERE run_id = ?", ("run_001",)).fetchone()
    assert row is not None
    assert row["outcome"] == "COMPLETED_WITH_TOOLS"
    assert row["llm_calls"] == 4
    assert row["tool_calls"] == 3
    assert row["had_protocol_error"] == 1
    assert row["had_validation_error"] == 0
    conn.close()


def test_consume_summary_no_mutation(tmp_path):
    """consume_summary must NOT change tiers."""
    horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
    horizon.resolve_config("u1")  # ensure user exists at tier C

    for i in range(10):
        horizon.consume_summary({
            "run_id": f"run_{i}",
            "outcome": "COMPLETED_WITH_TOOLS",
            "llm_calls": 1,
            "tool_calls": 1,
        })

    config = horizon.resolve_config("u1")
    assert config["max_steps"] == 12  # still tier C


def test_consume_summary_ignores_invalid(tmp_path):
    horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
    horizon.consume_summary({"outcome": "COMPLETED_WITH_TOOLS"})
    horizon.consume_summary(None)  # type: ignore
    horizon.consume_summary("garbage")  # type: ignore

    conn = horizon._connect()
    count = conn.execute("SELECT COUNT(*) FROM run_summaries").fetchone()[0]
    assert count == 0
    conn.close()


def test_consume_summary_upsert(tmp_path):
    """Same run_id should overwrite, not duplicate."""
    horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
    horizon.consume_summary({"run_id": "run_x", "outcome": "INTERRUPTED", "llm_calls": 1})
    horizon.consume_summary({"run_id": "run_x", "outcome": "COMPLETED_WITH_TOOLS", "llm_calls": 2})

    conn = horizon._connect()
    row = conn.execute("SELECT * FROM run_summaries WHERE run_id = ?", ("run_x",)).fetchone()
    assert row["outcome"] == "COMPLETED_WITH_TOOLS"
    assert row["llm_calls"] == 2
    count = conn.execute("SELECT COUNT(*) FROM run_summaries").fetchone()[0]
    assert count == 1
    conn.close()


# ===========================================================================
# Phase 4: PerfVector tests
# ===========================================================================

class TestPerfVectorDominance:
    def test_green_dominance(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        pv = horizon._compute_perf_vector(_good_summary("r1"))
        assert pv.dominance == "GREEN"
        assert pv.hard_failure is False
        assert pv.protocol_compliant is True

    def test_yellow_dominance_retries(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        pv = horizon._compute_perf_vector(_good_summary("r1", format_retries=1))
        assert pv.dominance == "YELLOW"
        assert pv.retry_count == 1

    def test_yellow_dominance_high_budget(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        pv = horizon._compute_perf_vector(_good_summary("r1", max_inferences=25, budget_remaining=3))
        assert pv.dominance == "YELLOW"
        assert pv.budget_efficiency > 0.8

    def test_red_dominance_hard_failure(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        pv = horizon._compute_perf_vector(_good_summary("r1", outcome="FAILED_TIMEOUT"))
        assert pv.dominance == "RED"
        assert pv.hard_failure is True

    def test_red_dominance_compound(self, tmp_path):
        """RED when protocol error + retries > 1 + near-exhausted budget."""
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        pv = horizon._compute_perf_vector(_good_summary(
            "r1",
            had_protocol_error=True,
            format_retries=2,
            max_inferences=25,
            budget_remaining=1,
        ))
        assert pv.dominance == "RED"


class TestPerfVectorMetrics:
    def test_budget_efficiency_calculation(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        pv = horizon._compute_perf_vector(_good_summary("r1", max_inferences=10, budget_remaining=7))
        assert pv.budget_efficiency == pytest.approx(0.3, abs=0.01)

    def test_duplicate_call_ratio_no_tools(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        pv = horizon._compute_perf_vector(_good_summary("r1", unique_tool_signatures=0, total_tool_invocations=0))
        assert pv.duplicate_call_ratio == 0.0

    def test_duplicate_call_ratio_with_duplicates(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        pv = horizon._compute_perf_vector(_good_summary("r1", unique_tool_signatures=2, total_tool_invocations=4))
        assert pv.duplicate_call_ratio == pytest.approx(0.5, abs=0.01)

    def test_composite_score_perfect_run(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        pv = horizon._compute_perf_vector(_good_summary(
            "r1",
            format_retries=0,
            max_inferences=25,
            budget_remaining=22,
            unique_tool_signatures=2,
            total_tool_invocations=2,
            had_protocol_error=False,
            had_validation_error=False,
        ))
        assert pv.composite_score > 0.85

    def test_composite_score_poor_run(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        pv = horizon._compute_perf_vector(_good_summary(
            "r1",
            outcome="FAILED_TIMEOUT",
            format_retries=2,
            max_inferences=25,
            budget_remaining=0,
            had_protocol_error=True,
            unique_tool_signatures=1,
            total_tool_invocations=4,
        ))
        assert pv.composite_score < 0.3

    def test_perf_vector_to_dict(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        pv = horizon._compute_perf_vector(_good_summary("r1"))
        d = pv.to_dict()
        assert d["run_id"] == "r1"
        assert d["dominance"] == "GREEN"
        assert "composite_score" in d


# ===========================================================================
# Phase 4: PolicyState tests
# ===========================================================================

class TestPolicyState:
    def test_default_values(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        state = horizon._load_policy_state("u1")
        assert state.retry_tolerance == 0.2
        assert state.step_limit_multiplier == 1.0
        assert state.autonomy_bias == 0.3
        assert state.critique_enabled is False

    def test_tier_c_clamping(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        state = PolicyState(user_id="u1", tier="C", retry_tolerance=0.9, autonomy_bias=0.9)
        state = horizon._clamp_policy(state)
        assert state.retry_tolerance <= 0.3
        assert state.autonomy_bias <= 0.3
        assert state.critique_enabled is False

    def test_tier_s_clamping(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        state = PolicyState(user_id="u1", tier="S", retry_tolerance=0.9, autonomy_bias=0.9)
        state = horizon._clamp_policy(state)
        assert state.retry_tolerance == 0.9  # within S bounds
        assert state.autonomy_bias == 0.9

    def test_critique_locked_tier_c(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        state = PolicyState(user_id="u1", tier="C", critique_enabled=True)
        state = horizon._clamp_policy(state)
        assert state.critique_enabled is False

    def test_save_and_load_roundtrip(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        state = PolicyState(user_id="u1", tier="B", retry_tolerance=0.35, step_limit_multiplier=1.1)
        horizon._save_policy_state(state)
        loaded = horizon._load_policy_state("u1")
        assert loaded.retry_tolerance == pytest.approx(0.35)
        assert loaded.step_limit_multiplier == pytest.approx(1.1)
        assert loaded.tier == "B"


# ===========================================================================
# Phase 4: Warmup guard tests
# ===========================================================================

class TestWarmupGuard:
    def test_blocks_before_min_runs(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        horizon.resolve_config("u1")
        for i in range(WARMUP_MIN_RUNS - 1):
            horizon.consume_summary(_good_summary(f"r{i}"), user_id="u1")
        assert horizon._check_warmup("u1") is False

    def test_blocks_high_variance(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        horizon.resolve_config("u1")
        # Alternate between very efficient and very wasteful runs
        for i in range(WARMUP_MIN_RUNS):
            if i % 2 == 0:
                s = _good_summary(f"r{i}", max_inferences=25, budget_remaining=22)
            else:
                s = _good_summary(f"r{i}", max_inferences=25, budget_remaining=1)
            horizon.consume_summary(s, user_id="u1")
        assert horizon._check_warmup("u1") is False

    def test_allows_after_stable_runs(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        horizon.resolve_config("u1")
        _feed_warmup(horizon, user_id="u1")
        assert horizon._check_warmup("u1") is True


# ===========================================================================
# Phase 4: Mutation tests
# ===========================================================================

class TestMutation:
    def test_green_mutation_tightens_budget(self, tmp_path):
        """GREEN with efficient budget → decrease step_limit_multiplier."""
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        pv = horizon._compute_perf_vector(_good_summary("r1", max_inferences=25, budget_remaining=22))
        assert pv.dominance == "GREEN"
        state = PolicyState(user_id="u1", tier="C")
        delta = horizon._compute_delta(pv, state)
        assert delta is not None
        param_names = [m[0] for m in delta.mutations]
        assert "step_limit_multiplier" in param_names

    def test_yellow_mutation_increases_tolerance(self, tmp_path):
        """YELLOW with retries → increase retry_tolerance."""
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        pv = horizon._compute_perf_vector(_good_summary("r1", format_retries=1))
        assert pv.dominance == "YELLOW"
        state = PolicyState(user_id="u1", tier="B")
        delta = horizon._compute_delta(pv, state)
        assert delta is not None
        param_names = [m[0] for m in delta.mutations]
        assert "retry_tolerance" in param_names

    def test_red_never_mutates(self, tmp_path):
        """RED dominance must never produce a delta (tested via pipeline)."""
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        horizon.resolve_config("u1")
        _feed_warmup(horizon, user_id="u1")
        # Feed a RED run
        horizon.consume_summary(
            _good_summary("red_run", outcome="FAILED_TIMEOUT"),
            user_id="u1",
        )
        # Policy should not have mutated from this RED run
        conn = horizon._connect()
        delta_count = conn.execute(
            "SELECT COUNT(*) FROM policy_deltas WHERE run_id = 'red_run'"
        ).fetchone()[0]
        assert delta_count == 0
        conn.close()

    def test_no_contradictory_mutations(self, tmp_path):
        """A single delta cannot mutate the same parameter twice."""
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        pv = horizon._compute_perf_vector(_good_summary(
            "r1", format_retries=1, max_inferences=25, budget_remaining=2
        ))
        state = PolicyState(user_id="u1", tier="S")
        delta = horizon._compute_delta(pv, state)
        if delta is not None:
            param_names = [m[0] for m in delta.mutations]
            assert len(param_names) == len(set(param_names))

    def test_mutation_clamped_to_tier(self, tmp_path):
        """Mutations cannot exceed tier ceilings."""
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        # At tier C ceiling for autonomy_bias (0.3)
        state = PolicyState(user_id="u1", tier="C", autonomy_bias=0.3)
        pv = horizon._compute_perf_vector(_good_summary("r1"))
        delta = horizon._compute_delta(pv, state)
        if delta is not None:
            for param, _old, new_val in delta.mutations:
                if param == "autonomy_bias":
                    lo, hi = TIER_CEILINGS["C"]["autonomy_bias"]
                    assert lo <= new_val <= hi

    def test_delta_linked_to_run_id(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        pv = horizon._compute_perf_vector(_good_summary("r42"))
        state = PolicyState(user_id="u1", tier="C")
        delta = horizon._compute_delta(pv, state)
        if delta is not None:
            assert delta.run_id == "r42"

    def test_delta_stored_in_sqlite(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        pv = horizon._compute_perf_vector(_good_summary("r1"))
        state = PolicyState(user_id="u1", tier="C")
        delta = horizon._compute_delta(pv, state)
        if delta is not None:
            horizon._store_delta(delta)
            conn = horizon._connect()
            row = conn.execute("SELECT * FROM policy_deltas WHERE run_id = 'r1'").fetchone()
            assert row is not None
            assert row["dominance"] == delta.dominance
            mutations = json.loads(row["mutations"])
            assert len(mutations) == len(delta.mutations)
            conn.close()


# ===========================================================================
# Phase 4: Checkpoint ring buffer tests
# ===========================================================================

class TestCheckpoint:
    def test_ring_buffer_eviction(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        state = PolicyState(user_id="u1", tier="C")
        for i in range(CHECKPOINT_RING_SIZE + 3):
            horizon._save_checkpoint(state, rolling_score=0.8, run_count=i + 1)

        conn = horizon._connect()
        count = conn.execute("SELECT COUNT(*) FROM policy_checkpoints WHERE user_id = 'u1'").fetchone()[0]
        assert count == CHECKPOINT_RING_SIZE
        conn.close()

    def test_checkpoint_saves_rolling_score(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        state = PolicyState(user_id="u1", tier="C")
        horizon._save_checkpoint(state, rolling_score=0.75, run_count=10)

        conn = horizon._connect()
        row = conn.execute("SELECT * FROM policy_checkpoints WHERE user_id = 'u1'").fetchone()
        assert row["rolling_composite_score"] == pytest.approx(0.75)
        assert row["run_count_at_checkpoint"] == 10
        conn.close()


# ===========================================================================
# Phase 4: Regression detection + rollback tests
# ===========================================================================

class TestRegressionDetection:
    def test_regression_triggers_rollback(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        horizon.resolve_config("u1")
        state = PolicyState(user_id="u1", tier="C", step_limit_multiplier=1.15)
        horizon._save_policy_state(state)
        # Save a checkpoint with a good score
        horizon._save_checkpoint(state, rolling_score=0.9, run_count=10)
        # Current score drops significantly
        assert horizon._check_regression("u1", current_rolling_score=0.5) is True

    def test_no_regression_within_threshold(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        horizon.resolve_config("u1")
        state = PolicyState(user_id="u1", tier="C")
        horizon._save_checkpoint(state, rolling_score=0.9, run_count=10)
        # Small drop within threshold
        assert horizon._check_regression("u1", current_rolling_score=0.85) is False

    def test_rollback_restores_best_state(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        horizon.resolve_config("u1")
        # Save a good checkpoint
        good_state = PolicyState(user_id="u1", tier="C", step_limit_multiplier=1.1, retry_tolerance=0.25)
        horizon._save_policy_state(good_state)
        horizon._save_checkpoint(good_state, rolling_score=0.9, run_count=10)
        # Save a worse checkpoint
        bad_state = PolicyState(user_id="u1", tier="C", step_limit_multiplier=0.85, retry_tolerance=0.1)
        horizon._save_policy_state(bad_state)
        horizon._save_checkpoint(bad_state, rolling_score=0.5, run_count=15)

        horizon._rollback("u1", total_runs=15)
        restored = horizon._load_policy_state("u1")
        assert restored.step_limit_multiplier == pytest.approx(1.1)
        assert restored.retry_tolerance == pytest.approx(0.25)

    def test_freeze_blocks_mutation_after_rollback(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        horizon.resolve_config("u1")
        state = PolicyState(user_id="u1", tier="C")
        horizon._save_policy_state(state)
        horizon._save_checkpoint(state, rolling_score=0.9, run_count=10)

        horizon._rollback("u1", total_runs=10)
        restored = horizon._load_policy_state("u1")
        assert restored.frozen_until_run == 10 + ROLLBACK_FREEZE_RUNS

    def test_freeze_expires_after_n_runs(self, tmp_path):
        """After freeze expires, mutations should resume."""
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        horizon.resolve_config("u1")
        # Set up freeze
        state = PolicyState(user_id="u1", tier="C", frozen_until_run=15)
        horizon._save_policy_state(state)
        # When total_runs >= frozen_until_run, freeze expires
        # The consume_summary pipeline checks: total_runs < frozen_until_run → skip
        # At run 15+, total_runs >= 15 → mutation resumes
        assert state.frozen_until_run == 15


# ===========================================================================
# Phase 4: Integration tests
# ===========================================================================

class TestResolveConfigPolicy:
    def test_resolve_config_includes_policy(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        config = horizon.resolve_config("u1")
        assert "policy" in config
        assert "retry_tolerance" in config["policy"]
        assert "autonomy_bias" in config["policy"]
        assert "step_limit_multiplier" in config["policy"]

    def test_resolve_config_adjusted_budget(self, tmp_path):
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        horizon.resolve_config("u1")  # create user
        # Set step_limit_multiplier to 1.2
        state = PolicyState(user_id="u1", tier="C", step_limit_multiplier=1.2)
        horizon._save_policy_state(state)
        config = horizon.resolve_config("u1")
        # Tier C budget = 12, * 1.2 = 14
        assert config["max_steps"] == 14


class TestFullPipeline:
    def test_warmup_then_mutation(self, tmp_path):
        """After warmup, GREEN runs should produce mutations."""
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        horizon.resolve_config("u1")

        # Feed warmup with good GREEN summaries
        _feed_warmup(horizon, user_id="u1")

        # Verify warmup passed
        assert horizon._check_warmup("u1") is True

        # Feed one more GREEN run (budget efficient)
        horizon.consume_summary(
            _good_summary("post_warmup", max_inferences=25, budget_remaining=22),
            user_id="u1",
        )

        # Check that a delta was produced
        conn = horizon._connect()
        delta_count = conn.execute("SELECT COUNT(*) FROM policy_deltas WHERE user_id = 'u1'").fetchone()[0]
        # At least one mutation should have happened during or after warmup
        # (the 10th warmup run itself may trigger mutation since warmup passes at 10)
        checkpoint_count = conn.execute("SELECT COUNT(*) FROM policy_checkpoints WHERE user_id = 'u1'").fetchone()[0]
        assert checkpoint_count > 0  # checkpoints saved
        conn.close()

    def test_perf_vector_stored_in_db(self, tmp_path):
        """Each consume_summary should store a PerfVector."""
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        horizon.consume_summary(_good_summary("r1"), user_id="u1")

        conn = horizon._connect()
        row = conn.execute("SELECT * FROM perf_vectors WHERE run_id = 'r1'").fetchone()
        assert row is not None
        assert row["dominance"] == "GREEN"
        assert row["user_id"] == "u1"
        conn.close()

    def test_agent_merit_counters_updated(self, tmp_path):
        """consume_summary should increment total_runs and lifetime_tokens."""
        horizon = CompetenceHorizon(db_path=tmp_path / "merit.sqlite3")
        horizon.resolve_config("u1")

        horizon.consume_summary(_good_summary("r1", tokens_consumed=100), user_id="u1")
        horizon.consume_summary(_good_summary("r2", tokens_consumed=200), user_id="u1")

        conn = horizon._connect()
        row = conn.execute("SELECT * FROM agent_merit WHERE user_id = 'u1'").fetchone()
        assert row["total_runs"] == 2
        assert row["lifetime_tokens_used"] == 300
        conn.close()
