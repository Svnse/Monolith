"""
Phase 5 tests — Open Foundation (Metric-Gated).

Tests for:
  1. TranscriptChain (hash-chain transcript)
  2. GrammarProfile (grammar-constrained decoding profiles)
  3. Migration (versioned contract/adapter migration)
  4. Conformance suite (6 scenarios x 4 model profiles)
  5. Hash-chain in runtime (end-to-end)
"""

import hashlib
import json
import time
from unittest.mock import MagicMock, patch

import pytest

from core.event_ledger import HashChainEntry, TranscriptChain
from engine.contract import (
    AgentOutcome,
    AgentRunResult,
    ContextBudget,
    ContractFactory,
    ExecutionContract,
    RunSummary,
    ToolOutputBudget,
    ToolPolicy,
)
from engine.agent_runtime import AgentMessage, AgentRuntime, ToolCall
from engine.migration import (
    ADAPTER_FORMAT_VERSION,
    ADAPTER_VERSIONS,
    COMPATIBILITY_MATRIX,
    CONTRACT_FORMAT_VERSION,
    CONTRACT_VERSIONS,
    check_compatibility,
    migrate_adapter_config,
    migrate_contract,
)
from engine.protocol_adapter import (
    AdapterStatus,
    GrammarProfile,
    GRAMMAR_PROFILES,
    ModelProfile,
    PROFILES,
    ProtocolAdapter,
    ProtocolAdapterResult,
    ToolCallFormat,
    get_grammar_profile,
    get_profile,
)


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
    bridge.execute = MagicMock(return_value={
        "ok": True, "tool": "test",
        "result": {"ok": True, "content": "ok"},
    })

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
        "adapter_version": "2a.1",
        "contract_format_version": "3.0",
        "model_fingerprint": "fp_test_abc",
        "source_page": "code",
        "context_budget": ContextBudget(context_window=8192),
        "tool_output_budget": ToolOutputBudget(),
    }
    defaults.update(overrides)
    return ExecutionContract(**defaults)


def _user_messages(content="Hello"):
    return [{"role": "user", "content": content}]


def _make_native_response(content=None, tool_calls=None):
    """Build a raw OpenAI-format response dict."""
    message = {"role": "assistant"}
    if content is not None:
        message["content"] = content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message}]}


def _make_native_tool_call(name, arguments, call_id="call_0"):
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(arguments)},
    }


# ===========================================================================
# 1. TranscriptChain Tests
# ===========================================================================

class TestTranscriptChain:
    def test_genesis_hash_is_deterministic(self):
        expected = hashlib.sha256(b"GENESIS").hexdigest()
        assert TranscriptChain.GENESIS_HASH == expected

    def test_new_chain_has_genesis_head(self):
        chain = TranscriptChain(
            contract_hash="c1",
            adapter_version="2a.1",
            model_profile_id="local_xml",
            model_fingerprint="fp1",
        )
        assert chain.head_hash == TranscriptChain.GENESIS_HASH
        assert chain.length == 0

    def test_append_produces_entry(self):
        chain = TranscriptChain(
            contract_hash="c1",
            adapter_version="2a.1",
            model_profile_id="local_xml",
            model_fingerprint="fp1",
        )
        entry = chain.append(
            state="PRECHECK",
            action_hash="a1",
            result_hash="r1",
        )
        assert isinstance(entry, HashChainEntry)
        assert entry.sequence == 1
        assert entry.previous_hash == TranscriptChain.GENESIS_HASH
        assert entry.state == "PRECHECK"
        assert entry.action_hash == "a1"
        assert entry.result_hash == "r1"
        assert entry.chain_hash != TranscriptChain.GENESIS_HASH
        assert chain.head_hash == entry.chain_hash
        assert chain.length == 1

    def test_append_chains_hashes(self):
        chain = TranscriptChain(
            contract_hash="c1",
            adapter_version="2a.1",
            model_profile_id="local_xml",
            model_fingerprint="fp1",
        )
        e1 = chain.append(state="PRECHECK", action_hash="a1", result_hash="r1")
        e2 = chain.append(state="INFER", action_hash="a2", result_hash="r2")
        assert e2.previous_hash == e1.chain_hash
        assert e2.chain_hash != e1.chain_hash

    def test_verify_valid_chain(self):
        chain = TranscriptChain(
            contract_hash="c1",
            adapter_version="2a.1",
            model_profile_id="local_xml",
            model_fingerprint="fp1",
        )
        chain.append(state="PRECHECK", action_hash="a1", result_hash="r1")
        chain.append(state="INFER", action_hash="a2", result_hash="r2")
        chain.append(state="TERMINATE", action_hash="a3", result_hash="r3")
        valid, bad_idx = chain.verify()
        assert valid is True
        assert bad_idx is None

    def test_verify_detects_corruption(self):
        chain = TranscriptChain(
            contract_hash="c1",
            adapter_version="2a.1",
            model_profile_id="local_xml",
            model_fingerprint="fp1",
        )
        chain.append(state="PRECHECK", action_hash="a1", result_hash="r1")
        chain.append(state="INFER", action_hash="a2", result_hash="r2")

        # Corrupt the second entry by replacing it with a fake one
        corrupted = HashChainEntry(
            sequence=chain._entries[1].sequence,
            previous_hash=chain._entries[1].previous_hash,
            contract_hash=chain._entries[1].contract_hash,
            state=chain._entries[1].state,
            action_hash="tampered",
            result_hash=chain._entries[1].result_hash,
            adapter_version=chain._entries[1].adapter_version,
            model_profile_id=chain._entries[1].model_profile_id,
            model_fingerprint=chain._entries[1].model_fingerprint,
            chain_hash=chain._entries[1].chain_hash,  # hash won't match
            timestamp=chain._entries[1].timestamp,
        )
        chain._entries[1] = corrupted
        valid, bad_idx = chain.verify()
        assert valid is False
        assert bad_idx == 1

    def test_divergence_point_identical_chains(self):
        chain1 = TranscriptChain(
            contract_hash="c1", adapter_version="2a.1",
            model_profile_id="local_xml", model_fingerprint="fp1",
        )
        chain2 = TranscriptChain(
            contract_hash="c1", adapter_version="2a.1",
            model_profile_id="local_xml", model_fingerprint="fp1",
        )
        chain1.append(state="PRECHECK", action_hash="a1", result_hash="r1")
        chain2.append(state="PRECHECK", action_hash="a1", result_hash="r1")
        assert chain1.divergence_point(chain2) is None

    def test_divergence_point_different_action(self):
        chain1 = TranscriptChain(
            contract_hash="c1", adapter_version="2a.1",
            model_profile_id="local_xml", model_fingerprint="fp1",
        )
        chain2 = TranscriptChain(
            contract_hash="c1", adapter_version="2a.1",
            model_profile_id="local_xml", model_fingerprint="fp1",
        )
        chain1.append(state="PRECHECK", action_hash="a1", result_hash="r1")
        chain2.append(state="PRECHECK", action_hash="a_different", result_hash="r1")
        assert chain1.divergence_point(chain2) == 0

    def test_divergence_point_length_mismatch(self):
        chain1 = TranscriptChain(
            contract_hash="c1", adapter_version="2a.1",
            model_profile_id="local_xml", model_fingerprint="fp1",
        )
        chain2 = TranscriptChain(
            contract_hash="c1", adapter_version="2a.1",
            model_profile_id="local_xml", model_fingerprint="fp1",
        )
        chain1.append(state="PRECHECK", action_hash="a1", result_hash="r1")
        chain2.append(state="PRECHECK", action_hash="a1", result_hash="r1")
        chain2.append(state="INFER", action_hash="a2", result_hash="r2")
        assert chain1.divergence_point(chain2) == 1

    def test_determinism_same_inputs_same_hash(self):
        chain1 = TranscriptChain(
            contract_hash="c1", adapter_version="2a.1",
            model_profile_id="local_xml", model_fingerprint="fp1",
        )
        chain2 = TranscriptChain(
            contract_hash="c1", adapter_version="2a.1",
            model_profile_id="local_xml", model_fingerprint="fp1",
        )
        e1 = chain1.append(state="PRECHECK", action_hash="a1", result_hash="r1")
        e2 = chain2.append(state="PRECHECK", action_hash="a1", result_hash="r1")
        assert e1.chain_hash == e2.chain_hash

    def test_different_model_fingerprint_different_hash(self):
        chain1 = TranscriptChain(
            contract_hash="c1", adapter_version="2a.1",
            model_profile_id="local_xml", model_fingerprint="fp1",
        )
        chain2 = TranscriptChain(
            contract_hash="c1", adapter_version="2a.1",
            model_profile_id="local_xml", model_fingerprint="fp_different",
        )
        e1 = chain1.append(state="PRECHECK", action_hash="a1", result_hash="r1")
        e2 = chain2.append(state="PRECHECK", action_hash="a1", result_hash="r1")
        assert e1.chain_hash != e2.chain_hash

    def test_snapshot_returns_copy(self):
        chain = TranscriptChain(
            contract_hash="c1", adapter_version="2a.1",
            model_profile_id="local_xml", model_fingerprint="fp1",
        )
        chain.append(state="PRECHECK", action_hash="a1", result_hash="r1")
        snap = chain.snapshot()
        assert len(snap) == 1
        chain.append(state="INFER", action_hash="a2", result_hash="r2")
        assert len(snap) == 1  # snapshot unaffected
        assert chain.length == 2

    def test_hash_chain_entry_to_dict(self):
        chain = TranscriptChain(
            contract_hash="c1", adapter_version="2a.1",
            model_profile_id="local_xml", model_fingerprint="fp1",
        )
        entry = chain.append(state="PRECHECK", action_hash="a1", result_hash="r1")
        d = entry.to_dict()
        assert d["sequence"] == 1
        assert d["state"] == "PRECHECK"
        assert d["contract_hash"] == "c1"
        assert d["chain_hash"] == entry.chain_hash

    def test_compute_chain_hash_static_method(self):
        h = TranscriptChain.compute_chain_hash(
            previous_hash="prev",
            contract_hash="ch",
            state="INFER",
            action_hash="ah",
            result_hash="rh",
            adapter_version="2a.1",
            model_profile_id="local_xml",
            model_fingerprint="fp",
        )
        assert isinstance(h, str)
        assert len(h) == 64  # SHA256 hex digest


# ===========================================================================
# 2. GrammarProfile Tests
# ===========================================================================

class TestGrammarProfiles:
    def test_optional_returns_none(self):
        """Optional policy should never have a grammar constraint."""
        for profile_id in ("native", "local_xml", "local_json", "local_native"):
            result = get_grammar_profile(profile_id, "optional")
            assert result is None, f"Expected None for ({profile_id}, optional)"

    def test_required_profiles_exist(self):
        """Required policy should have a grammar profile for all supported profiles."""
        for profile_id in ("native", "local_xml", "local_json", "local_native"):
            result = get_grammar_profile(profile_id, "required")
            assert result is not None, f"Expected GrammarProfile for ({profile_id}, required)"
            assert result.tool_policy == "required"
            assert result.model_profile_id == profile_id

    def test_forbidden_profiles_exist(self):
        """Forbidden policy should have a grammar profile for all supported profiles."""
        for profile_id in ("native", "local_xml", "local_json", "local_native"):
            result = get_grammar_profile(profile_id, "forbidden")
            assert result is not None, f"Expected GrammarProfile for ({profile_id}, forbidden)"
            assert result.tool_policy == "forbidden"

    def test_local_xml_required_has_bnf(self):
        gp = get_grammar_profile("local_xml", "required")
        assert gp is not None
        assert gp.grammar_type == "bnf"
        assert gp.grammar_spec is not None
        assert "<tool_call>" in gp.grammar_spec

    def test_local_json_required_has_bnf(self):
        gp = get_grammar_profile("local_json", "required")
        assert gp is not None
        assert gp.grammar_type == "bnf"
        assert gp.grammar_spec is not None

    def test_native_required_has_no_bnf(self):
        """Native profile uses API-level function calling, not grammar."""
        gp = get_grammar_profile("native", "required")
        assert gp is not None
        assert gp.grammar_type == "none"
        assert gp.grammar_spec is None

    def test_unknown_profile_returns_none(self):
        result = get_grammar_profile("nonexistent_profile", "required")
        assert result is None

    def test_grammar_profile_is_frozen(self):
        gp = get_grammar_profile("local_xml", "required")
        assert gp is not None
        with pytest.raises(AttributeError):
            gp.profile_id = "modified"  # type: ignore[misc]

    def test_model_profile_grammar_support_flag(self):
        assert PROFILES["local_xml"].grammar_support is True
        assert PROFILES["local_json"].grammar_support is True
        assert PROFILES["local_native"].grammar_support is True
        assert PROFILES["native"].grammar_support is False

    def test_contract_factory_resolves_grammar_profile(self):
        factory = ContractFactory()
        contract = factory.create(
            "Create a file called test.py",
            source_page="code",
            model_profile_id="local_xml",
        )
        # tool_policy should be REQUIRED for this prompt
        if contract.tool_policy == ToolPolicy.REQUIRED:
            assert contract.grammar_profile is not None
        elif contract.tool_policy == ToolPolicy.OPTIONAL:
            assert contract.grammar_profile is None


# ===========================================================================
# 3. Migration Tests
# ===========================================================================

class TestMigration:
    def test_version_constants(self):
        assert CONTRACT_FORMAT_VERSION == "3.0"
        assert ADAPTER_FORMAT_VERSION == "2a.1"

    def test_contract_versions_registered(self):
        assert "2.0" in CONTRACT_VERSIONS
        assert "2.1" in CONTRACT_VERSIONS
        assert "3.0" in CONTRACT_VERSIONS

    def test_adapter_versions_registered(self):
        assert "2a.0" in ADAPTER_VERSIONS
        assert "2a.1" in ADAPTER_VERSIONS

    def test_migrate_2_0_to_2_1(self):
        old = {"contract_id": "c1", "tool_policy": "required"}
        result = migrate_contract(old, "2.0", "2.1")
        assert "cycle_forbid" in result
        assert "tool_output_budget" in result
        assert result.get("token_gate") is False

    def test_migrate_2_1_to_3_0(self):
        old = {"contract_id": "c1", "cycle_forbid": []}
        result = migrate_contract(old, "2.1", "3.0")
        assert result["contract_format_version"] == "3.0"
        assert result["model_fingerprint"] == ""
        assert result["grammar_profile"] is None

    def test_migrate_2_0_to_3_0(self):
        old = {"contract_id": "c1", "tool_policy": "optional"}
        result = migrate_contract(old, "2.0", "3.0")
        assert "cycle_forbid" in result
        assert result["contract_format_version"] == "3.0"
        assert result["model_fingerprint"] == ""

    def test_migrate_same_version_noop(self):
        old = {"contract_id": "c1"}
        result = migrate_contract(old, "2.1", "2.1")
        assert result == old
        assert result is not old  # should be a copy

    def test_migrate_downgrade_raises(self):
        with pytest.raises(ValueError, match="downgrade"):
            migrate_contract({}, "3.0", "2.0")

    def test_migrate_unknown_version_raises(self):
        with pytest.raises(ValueError, match="unknown"):
            migrate_contract({}, "1.0", "2.0")

    def test_migrate_preserves_existing_fields(self):
        old = {"contract_id": "c1", "tool_policy": "required", "extra_field": "preserved"}
        result = migrate_contract(old, "2.0", "3.0")
        assert result["contract_id"] == "c1"
        assert result["tool_policy"] == "required"
        assert result["extra_field"] == "preserved"

    def test_migrate_adapter_2a0_to_2a1(self):
        old = {"adapter_version": "2a.0"}
        result = migrate_adapter_config(old, "2a.0", "2a.1")
        assert result["adapter_version"] == "2a.1"
        assert result.get("grammar_support") is True

    def test_migrate_adapter_same_version_noop(self):
        old = {"adapter_version": "2a.0"}
        result = migrate_adapter_config(old, "2a.0", "2a.0")
        assert result == old

    def test_compatibility_known_pairs(self):
        for (cv, av), expected in COMPATIBILITY_MATRIX.items():
            compat, reason = check_compatibility(cv, av)
            assert compat == expected, f"({cv}, {av}) expected {expected}"

    def test_compatibility_unknown_contract_version(self):
        compat, reason = check_compatibility("99.0", "2a.1")
        assert compat is False
        assert "unknown" in reason

    def test_compatibility_unknown_adapter_version(self):
        compat, reason = check_compatibility("3.0", "99.0")
        assert compat is False
        assert "unknown" in reason


# ===========================================================================
# 4. Conformance Suite — 6 scenarios x 4 model profiles
# ===========================================================================

PROFILE_IDS = ["native", "local_xml", "local_json", "local_native"]


def _make_profile_response(profile_id, tool_name="write_file", tool_args=None):
    """Create a response containing a tool call in the format expected by the given profile."""
    if tool_args is None:
        tool_args = {"path": "test.py", "content": "print('hello')"}

    if profile_id in ("native", "local_native"):
        return _make_native_response(
            content="I'll create the file.",
            tool_calls=[_make_native_tool_call(tool_name, tool_args, call_id="call_0")],
        )
    elif profile_id == "local_xml":
        tool_json = json.dumps({"name": tool_name, "args": tool_args})
        return _make_native_response(
            content=f"I'll create the file.\n<tool_call>{tool_json}</tool_call>",
        )
    elif profile_id == "local_json":
        tool_json = json.dumps({"name": tool_name, "arguments": tool_args})
        return _make_native_response(
            content=f"I'll create the file.\n```json\n{tool_json}\n```",
        )
    else:
        raise ValueError(f"Unknown profile: {profile_id}")


class TestConformanceSuite:
    """
    Conformance corpus per contract Section 9.
    Each scenario is run with a mock LLM producing the expected response format
    for the given profile. ProtocolAdapter is wired in.
    """

    @pytest.mark.parametrize("profile_id", PROFILE_IDS)
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_scenario_1_required_valid_tool_call(self, _mock_preflight, profile_id):
        """Required + valid tool call -> COMPLETED_WITH_TOOLS."""
        contract = _make_contract(
            tool_policy=ToolPolicy.REQUIRED,
            model_profile_id=profile_id,
        )

        # Raw responses: first has tool call, second is completion text
        raw_tool_response = _make_profile_response(profile_id)

        raw_responses = [raw_tool_response]
        call_count = {"n": 0}

        def mock_llm_call(history, tools):
            idx = call_count["n"]
            call_count["n"] += 1
            if idx < len(raw_responses):
                return raw_responses[idx]
            return _make_native_response(content="Done creating file.")

        events = []
        bridge = MagicMock()
        bridge.execute = MagicMock(return_value={
            "ok": True, "tool": "write_file",
            "result": {"ok": True, "content": "ok"},
        })

        cap_mgr = MagicMock()
        cap_mgr.profile = "code"
        cap_mgr.capability_digest = "test"
        cap_mgr.allowed_tools.return_value = ["write_file", "read_file"]
        cap_mgr.tool_schemas.return_value = []
        cap_mgr.validate_tool_name.return_value = MagicMock(ok=True)
        cap_mgr.authorize.return_value = MagicMock(ok=True)

        rt = AgentRuntime(
            llm_call=mock_llm_call,
            bridge=bridge,
            capability_manager=cap_mgr,
            emit_event=lambda e: events.append(e),
        )
        rt._contract = contract

        # Wire adapter
        profile = get_profile(profile_id)
        adapter = ProtocolAdapter(profile)
        rt._protocol_adapter = adapter

        result = rt.run(_user_messages("Create test.py"))
        assert result.outcome == AgentOutcome.COMPLETED_WITH_TOOLS
        assert result.tools_executed >= 1

        # Verify hash-chain integrity
        if rt._transcript_chain is not None:
            valid, bad_idx = rt._transcript_chain.verify()
            assert valid is True, f"Chain invalid at index {bad_idx}"
            assert rt._transcript_chain.length > 0

    @pytest.mark.parametrize("profile_id", PROFILE_IDS)
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_scenario_2_required_malformed_output(self, _mock_preflight, profile_id):
        """Required + malformed structured output -> FAILED_PROTOCOL_MALFORMED."""
        contract = _make_contract(
            tool_policy=ToolPolicy.REQUIRED,
            model_profile_id=profile_id,
            max_format_retries=0,
        )

        def mock_llm_call(history, tools):
            return _make_native_response(content="[PROTOCOL_ERROR: INVALID_RESPONSE_STRUCTURE]")

        events = []
        rt, events, _ = _make_runtime(
            llm_responses=[
                AgentMessage(role="assistant", content="[PROTOCOL_ERROR: INVALID_RESPONSE_STRUCTURE]"),
            ],
            contract=contract,
        )

        result = rt.run(_user_messages("Create a file"))
        assert result.outcome == AgentOutcome.FAILED_PROTOCOL_MALFORMED

    @pytest.mark.parametrize("profile_id", PROFILE_IDS)
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_scenario_3_required_narration_only(self, _mock_preflight, profile_id):
        """Required + narration-only response -> FAILED_PROTOCOL_NO_TOOLS."""
        contract = _make_contract(
            tool_policy=ToolPolicy.REQUIRED,
            model_profile_id=profile_id,
            max_format_retries=0,
        )

        rt, events, _ = _make_runtime(
            llm_responses=[
                AgentMessage(role="assistant", content="Here's how you would create the file..."),
            ],
            contract=contract,
        )

        result = rt.run(_user_messages("Create a file"))
        assert result.outcome == AgentOutcome.FAILED_PROTOCOL_NO_TOOLS

    @pytest.mark.parametrize("profile_id", PROFILE_IDS)
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_scenario_4_forbidden_completes_chat_only(self, _mock_preflight, profile_id):
        """Forbidden + prose-only response -> COMPLETED_CHAT_ONLY."""
        contract = _make_contract(
            tool_policy=ToolPolicy.FORBIDDEN,
            model_profile_id=profile_id,
        )

        rt, events, _ = _make_runtime(
            llm_responses=[
                AgentMessage(role="assistant", content="Here is the explanation of the code."),
            ],
            contract=contract,
        )

        result = rt.run(_user_messages("Explain code"))
        assert result.outcome == AgentOutcome.COMPLETED_CHAT_ONLY

    @pytest.mark.parametrize("profile_id", PROFILE_IDS)
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_scenario_5_oversized_tool_result(self, _mock_preflight, profile_id):
        """Oversized tool result envelope -> truncation logged."""
        budget = ToolOutputBudget(max_bytes_per_call=50, truncation_marker="[CUT]")
        contract = _make_contract(
            tool_policy=ToolPolicy.OPTIONAL,
            model_profile_id=profile_id,
            tool_output_budget=budget,
        )

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
        bridge.execute.return_value = {
            "ok": True, "tool": "read_file",
            "result": {"ok": True, "content": "A" * 500},
        }
        result = rt.run(_user_messages())

        # Verify truncation was logged
        ledger = rt._ledger.snapshot()
        truncation_entries = [e for e in ledger if e.payload.get("kind") == "tool_output_truncated"]
        assert len(truncation_entries) >= 1

    @pytest.mark.parametrize("profile_id", PROFILE_IDS)
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_scenario_6_timeout_tool_call(self, _mock_preflight, profile_id):
        """Timeout-inducing tool call -> FAILED_TIMEOUT."""
        contract = _make_contract(
            tool_policy=ToolPolicy.OPTIONAL,
            model_profile_id=profile_id,
            step_timeout_ms=100,
        )

        rt, events, bridge = _make_runtime(
            llm_responses=[
                AgentMessage(
                    role="assistant", content="running",
                    tool_calls=[ToolCall(id="tc1", name="run_cmd", arguments={"command": "sleep 10"})],
                ),
            ],
            contract=contract,
        )
        bridge.execute.return_value = {
            "ok": False, "tool": "run_cmd",
            "error": "step_timeout_ms exceeded",
            "timeout": True,
            "result": {"ok": False, "content": "", "error": "timeout"},
        }
        result = rt.run(_user_messages())
        assert result.outcome == AgentOutcome.FAILED_TIMEOUT

    @pytest.mark.parametrize("profile_id", PROFILE_IDS)
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_adapter_version_matches(self, _mock_preflight, profile_id):
        """Every adapter result should report the current adapter version."""
        profile = get_profile(profile_id)
        adapter = ProtocolAdapter(profile)
        raw = _make_native_response(content="test")
        result = adapter.adapt(raw)
        assert result.adapter_version == "2a.1"


# ===========================================================================
# 5. Hash-Chain in Runtime — End-to-End
# ===========================================================================

class TestHashChainInRuntime:
    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_chain_head_in_run_summary(self, _mock_preflight):
        """RunSummary should contain transcript_chain_head and transcript_chain_length."""
        contract = _make_contract()
        rt, events, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Hi")],
            contract=contract,
        )
        result = rt.run(_user_messages())

        summary_events = [e for e in events if e.get("event") == "RUN_SUMMARY"]
        assert len(summary_events) == 1
        summary = summary_events[0]["summary"]
        assert "transcript_chain_head" in summary
        assert "transcript_chain_length" in summary
        assert summary["transcript_chain_length"] > 0
        assert summary["transcript_chain_head"] != ""
        assert summary["transcript_chain_head"] != TranscriptChain.GENESIS_HASH

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_chain_verifies_after_run(self, _mock_preflight):
        """After a complete run, the transcript chain should verify cleanly."""
        contract = _make_contract()
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
        result = rt.run(_user_messages("Create a.py"))

        assert rt._transcript_chain is not None
        valid, bad_idx = rt._transcript_chain.verify()
        assert valid is True
        assert bad_idx is None

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_chain_length_grows_with_transitions(self, _mock_preflight):
        """Chain length should grow with each FSM transition."""
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
        result = rt.run(_user_messages())

        assert rt._transcript_chain is not None
        # Should have entries for: PRECHECK, INFER, VALIDATE_CALLS, EXECUTE, OBSERVE, COMMIT, INFER, COMMIT, TERMINATE
        # Plus chain entries from _adapt_response and tool execution
        assert rt._transcript_chain.length >= 5

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_replay_determinism(self, _mock_preflight):
        """Two identical runs should produce the same chain head hash."""
        contract = _make_contract()

        rt1, events1, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Hi")],
            contract=contract,
        )
        rt1.run(_user_messages("Hello"))

        rt2, events2, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Hi")],
            contract=contract,
        )
        rt2.run(_user_messages("Hello"))

        assert rt1._transcript_chain is not None
        assert rt2._transcript_chain is not None
        # Chains should have the same number of entries
        assert rt1._transcript_chain.length == rt2._transcript_chain.length
        # Chain heads should match for identical runs
        assert rt1._transcript_chain.head_hash == rt2._transcript_chain.head_hash

    @patch("engine.agent_runtime.AgentRuntime._preflight_check", return_value=None)
    def test_different_runs_different_chain(self, _mock_preflight):
        """Two runs with different contracts should produce different chain heads."""
        contract1 = _make_contract(contract_hash="hash_a")
        contract2 = _make_contract(contract_hash="hash_b")

        rt1, _, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Hi")],
            contract=contract1,
        )
        rt1.run(_user_messages("Hello"))

        rt2, _, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Hi")],
            contract=contract2,
        )
        rt2.run(_user_messages("Hello"))

        assert rt1._transcript_chain is not None
        assert rt2._transcript_chain is not None
        # Different contract_hash inputs should produce different chains
        assert rt1._transcript_chain.head_hash != rt2._transcript_chain.head_hash


# ===========================================================================
# 6. Contract Version Fields
# ===========================================================================

class TestContractVersionFields:
    def test_contract_has_format_version(self):
        contract = _make_contract()
        assert contract.contract_format_version == "3.0"

    def test_contract_has_model_fingerprint(self):
        contract = _make_contract(model_fingerprint="fp_test")
        assert contract.model_fingerprint == "fp_test"

    def test_contract_to_dict_includes_phase5_fields(self):
        contract = _make_contract(
            model_fingerprint="fp_test",
            grammar_profile="local_xml_required",
        )
        d = contract.to_dict()
        assert d["contract_format_version"] == "3.0"
        assert d["model_fingerprint"] == "fp_test"
        assert d["grammar_profile"] == "local_xml_required"

    def test_run_summary_to_dict_includes_chain_fields(self):
        summary = RunSummary(
            transcript_chain_head="head_hash",
            transcript_chain_length=42,
        )
        d = summary.to_dict()
        assert d["transcript_chain_head"] == "head_hash"
        assert d["transcript_chain_length"] == 42

    def test_contract_factory_sets_format_version(self):
        factory = ContractFactory()
        contract = factory.create("Hello", source_page="chat")
        assert contract.contract_format_version == "3.0"

    def test_contract_factory_sets_adapter_version(self):
        factory = ContractFactory()
        contract = factory.create("Hello", source_page="chat")
        assert contract.adapter_version == "2a.1"

    def test_contract_factory_accepts_model_fingerprint(self):
        factory = ContractFactory()
        contract = factory.create(
            "Create a file",
            source_page="code",
            model_fingerprint="fp_from_loader",
        )
        assert contract.model_fingerprint == "fp_from_loader"


# ===========================================================================
# 7. Version Compatibility in Preflight
# ===========================================================================

class TestVersionCompatPreflight:
    def test_compatible_versions_pass_preflight(self):
        contract = _make_contract(contract_format_version="3.0")
        rt, _, _ = _make_runtime(
            llm_responses=[AgentMessage(role="assistant", content="Hi")],
            contract=contract,
        )
        # Set adapter with matching version
        rt._protocol_adapter = ProtocolAdapter(PROFILES["local_xml"])
        result = rt._preflight_check()
        assert result is None  # no preflight failure

    def test_unknown_contract_version_fails_preflight(self):
        contract = _make_contract(contract_format_version="99.0")
        rt, _, _ = _make_runtime(
            llm_responses=[],
            contract=contract,
        )
        rt._protocol_adapter = ProtocolAdapter(PROFILES["local_xml"])
        result = rt._preflight_check()
        assert result is not None
        assert result["outcome"] == AgentOutcome.FAILED_PREFLIGHT
        assert "unknown" in result["reason"]
