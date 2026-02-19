"""
Adapter conformance tests — Phase 2a exit criteria.

Tests per model profile that:
  - Native tool calls are parsed correctly
  - Recovery extracts tool calls from content when expected
  - strict_mode blocks recovery
  - Rejected responses have explicit failure codes
  - Every result has raw_hash and adapter_version
  - Sanitizer strips scaffold tags
"""

import json

from engine.protocol_adapter import (
    AdapterStatus,
    ProtocolAdapter,
    ProtocolAdapterResult,
    ModelProfile,
    ToolCallFormat,
    PROFILES,
    _sanitize_content,
)
from engine.agent_runtime import AgentMessage


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Profile: native (strict, no recovery)
# ---------------------------------------------------------------------------

class TestNativeProfile:
    def setup_method(self):
        self.adapter = ProtocolAdapter(PROFILES["native"])

    def test_native_tool_call_parsed(self):
        raw = _make_native_response(
            content="I'll write the file.",
            tool_calls=[_make_native_tool_call("write_file", {"path": "a.py", "content": "x"})],
        )
        result = self.adapter.adapt(raw)
        assert result.status == AdapterStatus.NATIVE
        assert result.canonical_message is not None
        assert len(result.canonical_message.tool_calls) == 1
        assert result.canonical_message.tool_calls[0].name == "write_file"

    def test_no_tool_calls_accepted_as_chat(self):
        raw = _make_native_response(content="Here is the answer.")
        result = self.adapter.adapt(raw)
        assert result.status == AdapterStatus.NATIVE
        assert result.canonical_message.tool_calls is None
        assert result.canonical_message.content == "Here is the answer."

    def test_strict_mode_no_recovery(self):
        """Even if content has XML tool calls, strict mode doesn't recover."""
        raw = _make_native_response(
            content='<tool_call>{"name":"write_file","args":{"path":"a.py","content":"x"}}</tool_call>'
        )
        result = self.adapter.adapt(raw)
        assert result.status == AdapterStatus.NATIVE
        assert result.canonical_message.tool_calls is None  # not recovered

    def test_invalid_structure_rejected(self):
        result = self.adapter.adapt({})
        assert result.status == AdapterStatus.REJECTED
        assert result.failure_code == "INVALID_RESPONSE_STRUCTURE"

    def test_result_has_hash_and_version(self):
        raw = _make_native_response(content="test")
        result = self.adapter.adapt(raw)
        assert result.raw_hash != ""
        assert result.adapter_version == "2a.1"


# ---------------------------------------------------------------------------
# Profile: local_xml (recovery enabled)
# ---------------------------------------------------------------------------

class TestLocalXmlProfile:
    def setup_method(self):
        self.adapter = ProtocolAdapter(PROFILES["local_xml"])

    def test_native_tool_calls_still_work(self):
        raw = _make_native_response(
            tool_calls=[_make_native_tool_call("read_file", {"path": "x.py"})],
        )
        result = self.adapter.adapt(raw)
        assert result.status == AdapterStatus.NATIVE
        assert result.canonical_message.tool_calls[0].name == "read_file"

    def test_xml_recovery(self):
        raw = _make_native_response(
            content='I will create the file.\n<tool_call>{"name":"write_file","args":{"path":"a.py","content":"hello"}}</tool_call>'
        )
        result = self.adapter.adapt(raw)
        assert result.status == AdapterStatus.RECOVERED
        assert result.recovery_detail == "recovered_from_xml_block"
        assert len(result.canonical_message.tool_calls) == 1
        assert result.canonical_message.tool_calls[0].name == "write_file"
        assert result.canonical_message.tool_calls[0].arguments == {"path": "a.py", "content": "hello"}

    def test_xml_recovery_strips_markup_from_content(self):
        raw = _make_native_response(
            content='Thinking...\n<tool_call>{"name":"write_file","args":{"path":"a.py","content":"x"}}</tool_call>\nDone.'
        )
        result = self.adapter.adapt(raw)
        assert result.status == AdapterStatus.RECOVERED
        # Content should have tool call stripped
        assert "<tool_call>" not in (result.canonical_message.content or "")

    def test_json_block_fallback_recovery(self):
        raw = _make_native_response(
            content='```json\n{"name":"read_file","arguments":{"path":"b.py"}}\n```'
        )
        result = self.adapter.adapt(raw)
        assert result.status == AdapterStatus.RECOVERED
        assert "json_block" in (result.recovery_detail or "")

    def test_raw_json_fallback_recovery(self):
        raw = _make_native_response(
            content='{"name":"list_dir","args":{"path":"."}}'
        )
        result = self.adapter.adapt(raw)
        assert result.status == AdapterStatus.RECOVERED
        assert "raw_json" in (result.recovery_detail or "")

    def test_no_tool_calls_anywhere_is_native(self):
        raw = _make_native_response(content="The file contains three functions.")
        result = self.adapter.adapt(raw)
        assert result.status == AdapterStatus.NATIVE
        assert result.canonical_message.tool_calls is None

    def test_malformed_json_in_xml_not_recovered(self):
        raw = _make_native_response(
            content='<tool_call>not valid json</tool_call>'
        )
        result = self.adapter.adapt(raw)
        assert result.status == AdapterStatus.NATIVE  # no tool calls found
        assert result.canonical_message.tool_calls is None

    def test_multiple_tool_calls_recovered(self):
        raw = _make_native_response(
            content=(
                '<tool_call>{"name":"write_file","args":{"path":"a.py","content":"x"}}</tool_call>\n'
                '<tool_call>{"name":"run_cmd","args":{"command":"python a.py"}}</tool_call>'
            )
        )
        result = self.adapter.adapt(raw)
        assert result.status == AdapterStatus.RECOVERED
        assert len(result.canonical_message.tool_calls) == 2

    def test_invalid_structure_rejected(self):
        result = self.adapter.adapt({"choices": []})
        assert result.status == AdapterStatus.REJECTED


# ---------------------------------------------------------------------------
# Profile: local_json
# ---------------------------------------------------------------------------

class TestLocalJsonProfile:
    def setup_method(self):
        self.adapter = ProtocolAdapter(PROFILES["local_json"])

    def test_json_block_recovery(self):
        raw = _make_native_response(
            content='```json\n{"name":"write_file","arguments":{"path":"a.py","content":"hello"}}\n```'
        )
        result = self.adapter.adapt(raw)
        assert result.status == AdapterStatus.RECOVERED
        assert result.canonical_message.tool_calls[0].name == "write_file"


# ---------------------------------------------------------------------------
# Profile: local_native (native with recovery fallback)
# ---------------------------------------------------------------------------

class TestLocalNativeProfile:
    def setup_method(self):
        self.adapter = ProtocolAdapter(PROFILES["local_native"])

    def test_native_tool_calls_used(self):
        raw = _make_native_response(
            tool_calls=[_make_native_tool_call("write_file", {"path": "x.py", "content": "y"})],
        )
        result = self.adapter.adapt(raw)
        assert result.status == AdapterStatus.NATIVE
        assert result.canonical_message.tool_calls[0].name == "write_file"

    def test_recovery_available_for_local_native(self):
        """local_native is not strict — recovery should work."""
        raw = _make_native_response(
            content='<tool_call>{"name":"read_file","args":{"path":"x.py"}}</tool_call>'
        )
        result = self.adapter.adapt(raw)
        assert result.status == AdapterStatus.RECOVERED


# ---------------------------------------------------------------------------
# Content sanitizer
# ---------------------------------------------------------------------------

class TestSanitizer:
    def test_strips_response_tag(self):
        assert _sanitize_content("Done.</response>") == "Done."

    def test_strips_think_tags(self):
        assert _sanitize_content("<think>reasoning</think> Answer.") == "reasoning Answer."

    def test_none_passthrough(self):
        assert _sanitize_content(None) is None

    def test_empty_string(self):
        assert _sanitize_content("") == ""

    def test_clean_content_unchanged(self):
        assert _sanitize_content("File created.") == "File created."

    def test_scaffold_tag_in_middle(self):
        result = _sanitize_content("Part A</output>Part B")
        assert "</output>" not in result
        assert "Part A" in result
        assert "Part B" in result


# ---------------------------------------------------------------------------
# Determinism: same input → same hash
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_input_same_hash(self):
        adapter = ProtocolAdapter(PROFILES["local_xml"])
        raw = _make_native_response(content="hello")
        r1 = adapter.adapt(raw)
        r2 = adapter.adapt(raw)
        assert r1.raw_hash == r2.raw_hash
        assert r1.raw_hash != ""

    def test_different_input_different_hash(self):
        adapter = ProtocolAdapter(PROFILES["local_xml"])
        r1 = adapter.adapt(_make_native_response(content="hello"))
        r2 = adapter.adapt(_make_native_response(content="world"))
        assert r1.raw_hash != r2.raw_hash
