"""
Phase 3 tests — Typed Outcomes, FSM semantics, budget accounting, tool
execution semantics, state digest emission.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from engine.contract import (
    AgentOutcome,
    AgentRunResult,
    ContextBudget,
    ContractFactory,
    ExecutionContract,
    FSM_TRANSITIONS,
    RuntimeState,
    RunSummary,
    StateDigest,
    ToolOutputBudget,
    ToolPolicy,
)
from engine.agent_runtime import AgentMessage, AgentRuntime, ToolCall


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_runtime(
    *,
    llm_responses=None,
    contract=None,
    should_stop=None,
    emit_events=None,
):
    """Build a testable AgentRuntime with mock llm_call and bridge."""
    responses = list(llm_responses or [])
    call_count = {"n": 0}

    def mock_llm_call(history, tools):
        idx = call_count["n"]
        call_count["n"] += 1
        if idx < len(responses):
            return responses[idx]
        return AgentMessage(role="assistant", content="Done.")

    events = emit_events if emit_events is not None else []

    bridge = MagicMock()
    bridge.execute = MagicMock(return_value={"ok": True, "tool": "test", "result": {"ok": True, "content": "ok"}})

    cap_mgr = MagicMock()
    cap_mgr.profile = "code"
    cap_mgr.capability_digest = "test"
    cap_mgr.allowed_tools.return_value = ["write_file", "read_file", "list_dir", "run_cmd"]
    cap_mgr.tool_schemas.return_value = []
    cap_mgr.validate_tool_name.return_value = MagicMock(ok=True)
    cap_mgr.authorize.return_value = MagicMock(ok=True)

    rt = AgentRuntime(
        llm_call=mock_llm_call,
        bridge=bridge,
        capability_manager=cap_mgr,
        emit_event=lambda e: events.append(e),
        should_stop=should_stop or (lambda: False),
    )
    rt._contract = contract
    return rt, events, bridge


def _make_contract(**overrides):
    """Build a minimal ExecutionContract with overrides."""
    defaults = {
        "contract_id": "test-contract",
        "contract_hash": "abc123",
        "tool_policy": ToolPolicy.OPTIONAL,
        "max_inferences": 25,
        "max_format_retries": 1,
        "model_profile_id": "local_xml",
        "source_page": "code",
        "context_budget": ContextBudget(context_window=8192),
        "tool_output_budget": ToolOutputBudget(),
    }
    defaults.update(overrides)
    return ExecutionContract(**defaults)


def _user_messages(content="Hello"):
    return [{"role": "user", "content": content}]


# ===========================================================================
# FSM Transition Tests
# ===========================================================================

class TestFSMTransitions:
    def test_valid_transitions_allowed(self):
        """All transitions in FSM_TRANSITIONS are valid."""
        for source, targets in FSM_TRANSITIONS.items():
            for target in targets:
                assert target in FSM_TRANSITIONS, f"{target} not in FSM_TRANSITIONS keys"

    def test_terminate_has_no_outgoing(self):
        assert FSM_TRANSITIONS[RuntimeState.TERMINATE] == frozenset()

    def test_precheck_can_reach_infer_or_terminate(self):
        allowed = FSM_TRANSITIONS[RuntimeState.PRECHECK]
        assert RuntimeState.INFER in allowed
        assert RuntimeState.TERMINATE in allowed

    def test_all_states_in_transition_table(self):
        for state in RuntimeState:
            assert state in FSM_TRANSITIONS

    def test_illegal_transition_raises(self):
        contract = _make_contract()
        rt, events, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Hi")],
            contract=contract,
        )
        # Manually put runtime in a state and try illegal transition
        rt._fsm_state = RuntimeState.PRECHECK
        with pytest.raises(RuntimeError, match="Illegal FSM transition"):
            rt._transition_to(RuntimeState.OBSERVE)

    def test_transition_appends_ledger_entry(self):
        contract = _make_contract()
        rt, events, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Hi")],
            contract=contract,
        )
        rt._run_start_time = time.time()
        rt._transition_to(RuntimeState.PRECHECK)  # INIT -> PRECHECK
        ledger = rt._ledger.snapshot()
        transition_entries = [e for e in ledger if e.event_type == "state_transition"]
        assert len(transition_entries) >= 1
        assert "INIT -> PRECHECK" in transition_entries[0].payload.get("fsm_transition", "")

    def test_state_digest_emitted_on_transition(self):
        contract = _make_contract()
        rt, events, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Hi")],
            contract=contract,
        )
        rt._run_start_time = time.time()
        rt._transition_to(RuntimeState.PRECHECK)
        digest_events = [e for e in events if e.get("event") == "STATE_DIGEST"]
        assert len(digest_events) >= 1
        digest = digest_events[0]["digest"]
        assert digest["fsm_state"] == "PRECHECK"


# ===========================================================================
# Return Type Tests
# ===========================================================================

class TestRunReturnType:
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_run_returns_agent_run_result(self, _mock_preflight):
        contract = _make_contract()
        rt, events, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Hello there.")],
            contract=contract,
        )
        result = rt.run(_user_messages())
        assert isinstance(result, AgentRunResult)

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_success_property_matches_outcome(self, _mock_preflight):
        contract = _make_contract()
        rt, events, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Done.")],
            contract=contract,
        )
        result = rt.run(_user_messages())
        assert result.success == result.outcome.is_success

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_result_has_counters(self, _mock_preflight):
        contract = _make_contract()
        rt, events, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Done.")],
            contract=contract,
        )
        result = rt.run(_user_messages())
        assert result.inferences_used >= 1
        assert result.tokens_consumed >= 0
        assert result.tools_executed >= 0

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_chat_only_completion(self, _mock_preflight):
        contract = _make_contract()
        rt, events, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Just text.")],
            contract=contract,
        )
        result = rt.run(_user_messages())
        assert result.outcome == AgentOutcome.COMPLETED_CHAT_ONLY
        assert result.success is True

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_completed_with_tools(self, _mock_preflight):
        contract = _make_contract()
        rt, events, bridge = _make_runtime(
            llm_responses=[
                AgentMessage(
                    role="assistant",
                    content="I'll create the file.",
                    tool_calls=[ToolCall(id="tc1", name="write_file", arguments={"path": "a.py", "content": "x"})],
                ),
                AgentMessage(role="assistant", content="Done."),
            ],
            contract=contract,
        )
        result = rt.run(_user_messages("Create a.py"))
        assert result.outcome == AgentOutcome.COMPLETED_WITH_TOOLS
        assert result.tools_executed >= 1


# ===========================================================================
# Budget Tests
# ===========================================================================

class TestBudgetAccounting:
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_inference_budget_exhausted(self, _mock_preflight):
        contract = _make_contract(max_inferences=2)
        # Both responses have tool calls forcing re-entry, third doesn't exist -> budget hit
        rt, events, _ = _make_runtime(
            llm_responses=[
                AgentMessage(
                    role="assistant", content="Step 1",
                    tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "a"})],
                ),
                AgentMessage(
                    role="assistant", content="Step 2",
                    tool_calls=[ToolCall(id="t2", name="read_file", arguments={"path": "b"})],
                ),
                AgentMessage(role="assistant", content="Final"),
            ],
            contract=contract,
        )
        result = rt.run(_user_messages())
        assert result.outcome == AgentOutcome.FAILED_BUDGET_EXHAUSTED

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_total_timeout_produces_failed_timeout(self, _mock_preflight):
        """Test total_timeout_ms via _evaluate_commit directly (avoids timing flakiness)."""
        contract = _make_contract(total_timeout_ms=100)
        rt, events, _ = _make_runtime(
            llm_responses=[],
            contract=contract,
        )
        # Simulate a run that started 200ms ago
        rt._run_start_time = time.time() - 0.2
        rt._initial_contract_hash = contract.contract_hash
        commit = rt._evaluate_commit(
            inferences=1,
            tools_executed=1,
            format_retries=0,
            tokens_consumed=0,
            run_start_time=rt._run_start_time,
            history=[],
            assistant=AgentMessage(role="assistant", content="Done",
                                   tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "a"})]),
        )
        assert commit["action"] == "terminate"
        assert commit["outcome"] == AgentOutcome.FAILED_TIMEOUT

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_unlimited_token_budget_not_enforced(self, _mock_preflight):
        contract = _make_contract(max_tokens_consumed=0)  # unlimited
        rt, events, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Hi")],
            contract=contract,
        )
        rt._tokens_consumed = 999999
        result = rt.run(_user_messages())
        # Should NOT fail with budget exhaustion (token budget is 0=unlimited)
        assert result.outcome != AgentOutcome.FAILED_BUDGET_EXHAUSTED


# ===========================================================================
# Tool Output Budget Tests
# ===========================================================================

class TestToolOutputBudget:
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_tool_output_truncation(self, _mock_preflight):
        budget = ToolOutputBudget(max_bytes_per_call=50, truncation_marker="[CUT]")
        contract = _make_contract(tool_output_budget=budget)

        large_result = {"ok": True, "tool": "read_file", "result": {"ok": True, "content": "A" * 500}}

        rt, events, bridge = _make_runtime(
            llm_responses=[
                AgentMessage(
                    role="assistant", content="reading",
                    tool_calls=[ToolCall(id="tc1", name="read_file", arguments={"path": "big.txt"})],
                ),
                AgentMessage(role="assistant", content="Done."),
            ],
            contract=contract,
        )
        bridge.execute.return_value = large_result
        result = rt.run(_user_messages())
        # Check that truncation was logged in ledger
        ledger = rt._ledger.snapshot()
        truncation_entries = [e for e in ledger if e.payload.get("kind") == "tool_output_truncated"]
        assert len(truncation_entries) >= 1


# ===========================================================================
# Step Timeout Tests
# ===========================================================================

class TestStepTimeout:
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_step_timeout_produces_failed_timeout(self, _mock_preflight):
        contract = _make_contract(step_timeout_ms=100)

        rt, events, bridge = _make_runtime(
            llm_responses=[
                AgentMessage(
                    role="assistant", content="running",
                    tool_calls=[ToolCall(id="tc1", name="run_cmd", arguments={"command": "sleep 10"})],
                ),
            ],
            contract=contract,
        )
        # Simulate timeout result from bridge
        bridge.execute.return_value = {
            "ok": False,
            "tool": "run_cmd",
            "error": "step_timeout_ms exceeded (100ms)",
            "timeout": True,
            "result": {"ok": False, "content": "", "error": "timeout after 100ms"},
        }
        result = rt.run(_user_messages())
        assert result.outcome == AgentOutcome.FAILED_TIMEOUT
        assert "step_timeout" in result.termination_reason


# ===========================================================================
# Commit-Time Checks
# ===========================================================================

class TestCommitTimeChecks:
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_forced_synthesis_at_ratio(self, _mock_preflight):
        # Small context window so history easily exceeds ratio
        small_ctx = ContextBudget(context_window=200, reserved_system=10, force_synthesis_at_ratio=0.1)
        contract = _make_contract(
            context_budget=small_ctx,
            max_inferences=10,
        )
        rt, events, bridge = _make_runtime(
            llm_responses=[
                AgentMessage(
                    role="assistant", content="A" * 500,
                    tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "a"})],
                ),
                # After forced synthesis, model provides synthesis
                AgentMessage(role="assistant", content="Summary of work."),
            ],
            contract=contract,
        )
        result = rt.run(_user_messages("Do something " + "x" * 200))
        # Should complete (forced synthesis path)
        synthesis_events = [e for e in events if e.get("event") == "FORCED_SYNTHESIS"]
        assert len(synthesis_events) >= 1

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_contract_immutability_violation(self, _mock_preflight):
        contract = _make_contract()
        rt, events, bridge = _make_runtime(
            llm_responses=[
                AgentMessage(
                    role="assistant", content="Working",
                    tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "a"})],
                ),
                AgentMessage(role="assistant", content="Done"),
            ],
            contract=contract,
        )
        # After first inference, mutate the contract hash (simulating bug)
        original_run = rt.run

        def patched_run(messages):
            # Start the run, which sets _initial_contract_hash
            # We need to intercept after preflight to change the hash
            # Instead, just set a different initial hash
            rt._initial_contract_hash = "different_hash"
            return original_run(messages)

        # Directly test _evaluate_commit
        rt._contract = contract
        rt._initial_contract_hash = "wrong_hash"
        rt._run_start_time = time.time()
        commit_result = rt._evaluate_commit(
            inferences=1,
            tools_executed=0,
            format_retries=0,
            tokens_consumed=0,
            run_start_time=time.time(),
            history=[],
            assistant=AgentMessage(role="assistant", content="Hi"),
        )
        assert commit_result["action"] == "terminate"
        assert commit_result["outcome"] == AgentOutcome.FAILED_CONTRACT_VIOLATION


# ===========================================================================
# Tool Execution Semantics
# ===========================================================================

class TestToolSemantics:
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_malformed_envelope_produces_failed_validation(self, _mock_preflight):
        """Tool returning envelope without required keys -> FAILED_VALIDATION."""
        contract = _make_contract()
        rt, events, bridge = _make_runtime(
            llm_responses=[
                AgentMessage(
                    role="assistant", content="Working",
                    tool_calls=[ToolCall(id="tc1", name="read_file", arguments={"path": "x"})],
                ),
            ],
            contract=contract,
        )
        # Return a result that will produce a malformed envelope
        # The bridge returns ok, but the _execute_tool builds the envelope.
        # We need to mock _execute_tool to return a message with bad envelope.
        def bad_execute_tool(call):
            # Return a tool message with missing required envelope keys
            return AgentMessage(
                role="tool",
                name=call.name,
                tool_call_id=call.id,
                content=json.dumps({"bad_key": "no status or tool fields"}),
            )
        rt._execute_tool = bad_execute_tool
        result = rt.run(_user_messages())
        assert result.outcome == AgentOutcome.FAILED_VALIDATION

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_partial_execution_recorded(self, _mock_preflight):
        """When tool 2 of 3 fails, partial execution is logged."""
        contract = _make_contract(cycle_forbid=(("read_file", "read_file"),))
        rt, events, bridge = _make_runtime(
            llm_responses=[
                AgentMessage(
                    role="assistant", content="Reading files",
                    tool_calls=[
                        ToolCall(id="t1", name="write_file", arguments={"path": "a", "content": "x"}),
                        ToolCall(id="t2", name="read_file", arguments={"path": "b"}),
                        ToolCall(id="t3", name="read_file", arguments={"path": "c"}),
                    ],
                ),
            ],
            contract=contract,
        )
        result = rt.run(_user_messages())
        # Should detect cycle: read_file -> read_file
        assert result.outcome == AgentOutcome.FAILED_CONTRACT_VIOLATION
        # Check partial execution was recorded
        ledger = rt._ledger.snapshot()
        partial_entries = [e for e in ledger if e.payload.get("kind") == "partial_batch_execution"]
        assert len(partial_entries) >= 1
        assert partial_entries[0].payload["completed_tools"] >= 1


# ===========================================================================
# State Digest Tests
# ===========================================================================

class TestStateDigest:
    def test_state_digest_dataclass(self):
        digest = StateDigest(
            fsm_state="INFER",
            inferences_used=3,
            inferences_remaining=22,
            tokens_consumed=1500,
        )
        d = digest.to_dict()
        assert d["fsm_state"] == "INFER"
        assert d["inferences_used"] == 3
        assert d["tokens_consumed"] == 1500

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_digests_emitted_during_run(self, _mock_preflight):
        contract = _make_contract()
        rt, events, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Hi")],
            contract=contract,
        )
        rt.run(_user_messages())
        digest_events = [e for e in events if e.get("event") == "STATE_DIGEST"]
        assert len(digest_events) >= 3  # At least PRECHECK, INFER, COMMIT, TERMINATE

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_digests_inferences_monotonic(self, _mock_preflight):
        contract = _make_contract(max_inferences=5)
        rt, events, bridge = _make_runtime(
            llm_responses=[
                AgentMessage(
                    role="assistant", content="Step 1",
                    tool_calls=[ToolCall(id="t1", name="read_file", arguments={"path": "a"})],
                ),
                AgentMessage(role="assistant", content="Done."),
            ],
            contract=contract,
        )
        rt.run(_user_messages())
        digest_events = [e for e in events if e.get("event") == "STATE_DIGEST"]
        inferences = [d["digest"]["inferences_used"] for d in digest_events]
        # Inferences used should be non-decreasing
        for i in range(1, len(inferences)):
            assert inferences[i] >= inferences[i - 1]


# ===========================================================================
# RunSummary Tests
# ===========================================================================

class TestRunSummary:
    def test_run_summary_has_tokens_consumed(self):
        summary = RunSummary(tokens_consumed=500)
        d = summary.to_dict()
        assert d["tokens_consumed"] == 500

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_run_summary_emitted(self, _mock_preflight):
        contract = _make_contract()
        rt, events, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Hi")],
            contract=contract,
        )
        rt.run(_user_messages())
        summary_events = [e for e in events if e.get("event") == "RUN_SUMMARY"]
        assert len(summary_events) == 1
        summary = summary_events[0]["summary"]
        assert summary["outcome"] in [o.value for o in AgentOutcome]


# ===========================================================================
# Interrupt Tests
# ===========================================================================

class TestInterrupt:
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_interrupt_produces_interrupted(self, _mock_preflight):
        contract = _make_contract()
        rt, events, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Hi")],
            contract=contract,
            should_stop=lambda: True,
        )
        result = rt.run(_user_messages())
        assert result.outcome == AgentOutcome.INTERRUPTED


# ===========================================================================
# Preflight Tests
# ===========================================================================

class TestPreflight:
    def test_preflight_bad_max_inferences(self):
        contract = _make_contract(max_inferences=0)
        rt, events, _ = _make_runtime(
            llm_responses=[],
            contract=contract,
        )
        result = rt.run(_user_messages())
        assert result.outcome == AgentOutcome.FAILED_PREFLIGHT

    def test_preflight_bad_format_retries(self):
        contract = _make_contract(max_format_retries=-1)
        rt, events, _ = _make_runtime(
            llm_responses=[],
            contract=contract,
        )
        result = rt.run(_user_messages())
        assert result.outcome == AgentOutcome.FAILED_PREFLIGHT


# ===========================================================================
# Policy Enforcement Tests
# ===========================================================================

class TestPolicyEnforcement:
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_required_tool_no_tools_fails(self, _mock_preflight):
        contract = _make_contract(
            tool_policy=ToolPolicy.REQUIRED,
            max_format_retries=0,
        )
        rt, events, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Just text.")],
            contract=contract,
        )
        result = rt.run(_user_messages())
        assert result.outcome == AgentOutcome.FAILED_PROTOCOL_NO_TOOLS

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_forbidden_tool_completes_chat_only(self, _mock_preflight):
        contract = _make_contract(tool_policy=ToolPolicy.FORBIDDEN)
        rt, events, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Chat response.")],
            contract=contract,
        )
        result = rt.run(_user_messages())
        assert result.outcome == AgentOutcome.COMPLETED_CHAT_ONLY


# ===========================================================================
# Conformance Corpus (parametrized per contract Section 9)
# ===========================================================================

class TestConformanceCorpus:
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_required_valid_tool_call(self, _mock_preflight):
        """Required + valid tool call -> COMPLETED_WITH_TOOLS."""
        contract = _make_contract(tool_policy=ToolPolicy.REQUIRED)
        rt, events, bridge = _make_runtime(
            llm_responses=[
                AgentMessage(
                    role="assistant", content="Creating file.",
                    tool_calls=[ToolCall(id="tc1", name="write_file", arguments={"path": "a.py", "content": "x"})],
                ),
                AgentMessage(role="assistant", content="Done."),
            ],
            contract=contract,
        )
        result = rt.run(_user_messages("Create a file"))
        assert result.outcome == AgentOutcome.COMPLETED_WITH_TOOLS

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_required_prose_only(self, _mock_preflight):
        """Required + prose only -> FAILED_PROTOCOL_NO_TOOLS."""
        contract = _make_contract(tool_policy=ToolPolicy.REQUIRED, max_format_retries=0)
        rt, events, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Here's the code...")],
            contract=contract,
        )
        result = rt.run(_user_messages("Create a file"))
        assert result.outcome == AgentOutcome.FAILED_PROTOCOL_NO_TOOLS
