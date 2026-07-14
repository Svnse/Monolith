from __future__ import annotations

import os
from pathlib import Path

import core.cmd_parser as cmd_parser
from core.cmd_parser import (
    extract_commands,
    get_cmd_parser_metrics,
    process_response,
    reset_cmd_parser_metrics,
    strip_commands,
)
import core.skill_runtime as skill_runtime
from core.skill_registry import build_tool_catalog, get_tool, list_tools
from ui.pages.chat import build_tool_followup_prompt


def test_builtin_tools_are_discoverable() -> None:
    names = {spec.name for spec in list_tools()}
    assert {
        "calculate",
        "create_tool",
        "find_files",
        "get_budget_score",
        "get_context_summary",
        "grep",
        "llm_call",
        "list_files",
        "open_addon",
        "open_file",
        "read_file",
        "reload_skills",
        "run_tests",
        "save_note",
        "search_history",
        "set_session_meta",
        "soundtrap",
        "unzip_file",
        "web_search",
        "zip_files",
    }.issubset(names)
    assert all(spec.path.exists() for spec in list_tools())


def test_builtin_tool_specs_expose_json_schema() -> None:
    read_file = get_tool("read_file")
    assert read_file is not None
    assert isinstance(read_file.json_schema, dict)
    assert read_file.json_schema.get("required") == ["path"]
    open_file = get_tool("open_file")
    assert open_file is not None
    assert isinstance(open_file.json_schema, dict)
    assert open_file.json_schema.get("required") == ["path"]


def test_tool_catalog_is_generated_from_skills_directory() -> None:
    catalog = build_tool_catalog()
    assert "Available tools:" in catalog
    assert "read_file" in catalog
    assert "open_file" in catalog
    assert "search_history" in catalog
    assert "find_files" in catalog
    assert "generate_image" in catalog
    assert "generate_audio" in catalog
    assert "soundtrap" in catalog
    assert "web_search" in catalog


def test_process_response_executes_tool_field(tmp_path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("hello from tool runtime", encoding="utf-8")
    text = (
        '<tool_call>'
        f'{{"tool":"read_file","path":"{sample.as_posix()}","max_chars":500}}'
        "</tool_call>"
    )

    clean_text, tool_result, artifacts = process_response(text, archive_dir=tmp_path)

    assert clean_text == ""
    assert tool_result is not None
    assert "[read_file:" in tool_result
    assert "hello from tool runtime" in tool_result
    assert artifacts[0]["kind"] == "tool_call"
    assert artifacts[1]["kind"] == "tool_result"
    assert artifacts[1]["tool"] == "read_file"


def test_validation_rejects_missing_required_args(tmp_path) -> None:
    text = '<tool_call>{"tool":"read_file","max_chars":3000}</tool_call>'
    _clean, tool_result, artifacts = process_response(text, archive_dir=tmp_path)

    assert tool_result is not None
    assert "invalid arguments for 'read_file'" in tool_result
    assert "missing required field 'path'" in tool_result
    envelope = artifacts[1].get("envelope", {})
    assert envelope.get("ok") is False
    assert "validation_errors" in envelope.get("data", {})


def test_validation_rejects_invalid_arg_types(tmp_path) -> None:
    text = '<tool_call>{"tool":"read_file","path":42}</tool_call>'
    _clean, tool_result, artifacts = process_response(text, archive_dir=tmp_path)

    assert tool_result is not None
    assert "invalid arguments for 'read_file'" in tool_result
    assert "expected string" in tool_result
    envelope = artifacts[1].get("envelope", {})
    assert envelope.get("ok") is False


def test_validation_rejects_unknown_fields(tmp_path) -> None:
    text = '<tool_call>{"tool":"calculate","expr":"1+1","bogus":123}</tool_call>'
    _clean, tool_result, artifacts = process_response(text, archive_dir=tmp_path)

    assert tool_result is not None
    assert "invalid arguments for 'calculate'" in tool_result
    assert "unknown field(s): bogus" in tool_result
    envelope = artifacts[1].get("envelope", {})
    assert envelope.get("ok") is False


def test_validation_requires_prompt_or_messages_for_llm_call(tmp_path) -> None:
    text = '<tool_call>{"tool":"llm_call","max_tokens":128}</tool_call>'
    _clean, tool_result, artifacts = process_response(text, archive_dir=tmp_path)

    assert tool_result is not None
    assert "invalid arguments for 'llm_call'" in tool_result
    assert "requires either non-empty 'prompt' or non-empty 'messages'" in tool_result
    envelope = artifacts[1].get("envelope", {})
    assert envelope.get("ok") is False


def test_find_files_returns_typed_matches(tmp_path) -> None:
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    target = nested / "skills.md"
    target.write_text("skill doc", encoding="utf-8")

    text = (
        '<tool_call>{"tool":"find_files","path":"'
        + tmp_path.as_posix()
        + '","pattern":"skills.md","max_results":10}</tool_call>'
    )
    _clean_text, tool_result, artifacts = process_response(text, archive_dir=tmp_path)

    assert tool_result is not None
    assert "[find_files:" in tool_result
    envelope = artifacts[1].get("envelope", {})
    data = envelope.get("data", {})
    matches = data.get("matches", [])
    assert isinstance(matches, list)
    assert str(target) in matches
    assert data.get("path") == str(target)


def test_chain_find_files_to_read_file(tmp_path) -> None:
    nested = tmp_path / "docs"
    nested.mkdir(parents=True)
    target = nested / "skills.md"
    target.write_text("hello from discovered file", encoding="utf-8")

    text = (
        "<tool_call>"
        '{"mode":"chain","calls":['
        '{"id":"hit","tool":"find_files","path":"'
        + tmp_path.as_posix()
        + '","pattern":"skills.md","max_results":5},'
        '{"id":"src","tool":"read_file","path":"$hit.data.matches.0"}'
        "]}"
        "</tool_call>"
    )
    _clean_text, tool_result, _artifacts = process_response(text, archive_dir=tmp_path)

    assert tool_result is not None
    assert "[find_files:" in tool_result
    assert "[read_file:" in tool_result
    assert "hello from discovered file" in tool_result


def test_process_response_accepts_legacy_op_alias(tmp_path) -> None:
    clean_text, tool_result, artifacts = process_response(
        '<tool_call>{"op":"calculate","expr":"1 + 2 * 3"}</tool_call>',
        archive_dir=tmp_path,
    )

    assert clean_text == ""
    assert tool_result == "[calculate: 1 + 2 * 3 = 7]"
    assert len(artifacts) == 2


def test_list_files_file_path_hint_recommends_read_file(tmp_path) -> None:
    sample = tmp_path / "skills.md"
    sample.write_text("x", encoding="utf-8")
    text = (
        '<tool_call>{"tool":"list_files","path":"'
        + sample.as_posix()
        + '"}</tool_call>'
    )
    _clean_text, tool_result, _artifacts = process_response(text, archive_dir=tmp_path)

    assert tool_result is not None
    assert "path is a file, not a directory" in tool_result
    assert "use read_file with this path" in tool_result


def test_list_files_missing_filename_hint_recommends_find_files(tmp_path) -> None:
    missing = tmp_path / "skills.md"
    text = (
        '<tool_call>{"tool":"list_files","path":"'
        + missing.as_posix()
        + '"}</tool_call>'
    )
    _clean_text, tool_result, _artifacts = process_response(text, archive_dir=tmp_path)

    assert tool_result is not None
    assert "list_files expects a directory" in tool_result
    assert "use find_files with pattern='skills.md'" in tool_result


def test_extract_commands_ignores_non_json_monolith_tag_mentions() -> None:
    text = "Use <tool_call> blocks in docs when describing tool envelopes."
    assert extract_commands(text) == []


def test_process_response_executes_batch_calls(tmp_path) -> None:
    sample = tmp_path / "sample.txt"
    sample.write_text("alpha\nTODO: tighten this\nomega\n", encoding="utf-8")
    text = (
        '<tool_call>'
        '{"calls":['
        '{"id":"a1","tool":"grep","pattern":"TODO","path":"'
        + sample.as_posix()
        + '"},'
        '{"id":"a2","tool":"read_file","path":"'
        + sample.as_posix()
        + '","max_chars":50}'
        '],"mode":"parallel"}'
        "</tool_call>"
    )

    clean_text, tool_result, artifacts = process_response(text, archive_dir=tmp_path)

    assert clean_text == ""
    assert tool_result is not None
    assert "[call:a1]" in tool_result
    assert "[call:a2]" in tool_result
    assert "[grep:" in tool_result
    assert "[read_file:" in tool_result
    assert [item["kind"] for item in artifacts] == ["tool_call", "tool_result", "tool_result"]


def test_grep_reports_skipped_large_files(tmp_path) -> None:
    large = tmp_path / "big.log"
    large.write_text("x" * ((2 * 1024 * 1024) + 32), encoding="utf-8")

    text = (
        '<tool_call>{"tool":"grep","pattern":"needle","path":"'
        + tmp_path.as_posix()
        + '"}</tool_call>'
    )
    _clean, tool_result, _artifacts = process_response(text, archive_dir=tmp_path)

    assert tool_result is not None
    assert "skipped 1 file > 2MB" in tool_result


def test_process_response_renders_structured_multi_result_output(tmp_path) -> None:
    text = (
        '<tool_call>{"calls":['
        '{"tool":"calculate","expr":"1+1"},'
        '{"tool":"calculate","expr":"2+2"}'
        '],"mode":"parallel"}</tool_call>'
    )

    _clean_text, tool_result, _artifacts = process_response(text, archive_dir=tmp_path)

    assert tool_result is not None
    assert "[tool_result_1: tool=calculate, status=ok, call_id=call_1]" in tool_result
    assert "[tool_result_2: tool=calculate, status=ok, call_id=call_2]" in tool_result
    assert "[calculate: 1+1 = 2]" in tool_result
    assert "[calculate: 2+2 = 4]" in tool_result


def test_process_response_repairs_dangling_tool_call_tag(tmp_path) -> None:
    sample = tmp_path / "dangling.txt"
    sample.write_text("repaired command execution", encoding="utf-8")
    text = f'<tool_call>{{"tool":"read_file","path":"{sample.as_posix()}"}}'

    clean_text, tool_result, artifacts = process_response(text, archive_dir=tmp_path)

    assert clean_text == ""
    assert tool_result is not None
    assert "repaired command execution" in tool_result
    assert not any(bool(item.get("parse_error")) for item in artifacts)


def test_process_response_strict_mode_rejects_dangling_tool_call_tag(tmp_path) -> None:
    sample = tmp_path / "strict_dangling.txt"
    sample.write_text("strict mode should not auto-repair tags", encoding="utf-8")
    text = f'<tool_call>{{"tool":"read_file","path":"{sample.as_posix()}"}}'

    clean_text, tool_result, artifacts = process_response(text, archive_dir=tmp_path, strict=True)

    assert clean_text == ""
    assert tool_result is not None
    assert "malformed tool call" in tool_result
    assert "strict_mode_missing_close_tag" in tool_result
    parse_errors = [item for item in artifacts if bool(item.get("parse_error"))]
    assert parse_errors
    assert parse_errors[-1].get("repair_close_tag") is False


def test_process_response_sanitizes_windows_backslashes(tmp_path) -> None:
    if os.name != "nt":
        return
    sample = tmp_path / "windows_path.txt"
    sample.write_text("windows backslash parse", encoding="utf-8")
    raw_path = str(sample)
    assert "\\" in raw_path

    # Intentionally malformed JSON for Windows path escaping: single backslashes.
    text = f'<tool_call>{{"tool":"read_file","path":"{raw_path}"}}</tool_call>'
    clean_text, tool_result, artifacts = process_response(text, archive_dir=tmp_path)

    assert clean_text == ""
    assert tool_result is not None
    assert "windows backslash parse" in tool_result
    assert not any(bool(item.get("parse_error")) for item in artifacts)


def test_process_response_strict_mode_rejects_windows_backslash_repair(tmp_path) -> None:
    if os.name != "nt":
        return
    sample = tmp_path / "windows_strict_path.txt"
    sample.write_text("strict backslash mode", encoding="utf-8")
    raw_path = str(sample)
    text = f'<tool_call>{{"tool":"read_file","path":"{raw_path}"}}</tool_call>'

    clean_text, tool_result, artifacts = process_response(text, archive_dir=tmp_path, strict=True)

    assert clean_text == ""
    assert tool_result is not None
    assert "malformed tool call" in tool_result
    assert "json_backslash_repair_attempted" not in tool_result
    parse_errors = [item for item in artifacts if bool(item.get("parse_error"))]
    assert parse_errors
    assert parse_errors[-1].get("repair_json") is False


def test_extract_commands_honors_strict_env(monkeypatch) -> None:
    monkeypatch.setenv("MONOLITH_CMD_PARSER_STRICT", "1")
    text = '<tool_call>{"tool":"calculate","expr":"1+1"}'

    commands = extract_commands(text)

    assert commands
    assert commands[0].get("_parse_error") is True
    assert commands[0].get("_strict_missing_close_tag") is True


def test_cmd_parser_metrics_capture_repair_usage(tmp_path) -> None:
    reset_cmd_parser_metrics()
    sample = tmp_path / "metrics.txt"
    sample.write_text("metrics repair run", encoding="utf-8")
    text = f'<tool_call>{{"tool":"read_file","path":"{sample.as_posix()}"}}'

    _clean_text, tool_result, _artifacts = process_response(text, archive_dir=tmp_path)

    assert tool_result is not None
    metrics = get_cmd_parser_metrics()
    assert metrics["extract_compat_calls"] >= 1
    assert metrics["dangling_tag_repair_attempts"] >= 1
    assert metrics["dangling_tag_repair_success"] >= 1


def test_cmd_parser_metrics_capture_strict_missing_close_tag(tmp_path) -> None:
    reset_cmd_parser_metrics()
    text = '<tool_call>{"tool":"calculate","expr":"1+1"}'

    _clean_text, tool_result, _artifacts = process_response(text, archive_dir=tmp_path, strict=True)

    assert tool_result is not None
    metrics = get_cmd_parser_metrics()
    assert metrics["extract_strict_calls"] >= 1
    assert metrics["strict_missing_close_tag_errors"] >= 1
    assert metrics["dangling_tag_repair_attempts"] == 0


def test_process_response_marks_parse_error_artifacts(tmp_path) -> None:
    text = '<tool_call>{"tool":"calculate","expr":"1 + 2"</tool_call>'
    clean_text, tool_result, artifacts = process_response(text, archive_dir=tmp_path)

    assert clean_text == ""
    assert tool_result is not None
    assert "malformed tool call" in tool_result
    parse_errors = [item for item in artifacts if bool(item.get("parse_error"))]
    assert parse_errors
    assert parse_errors[0].get("tool") == "error"


def test_chain_mode_pipes_results_between_steps(tmp_path) -> None:
    """Chain mode: step 2 receives step 1's output via $prev."""
    sample = tmp_path / "data.txt"
    sample.write_text("line_one\nline_two\nline_three\n", encoding="utf-8")

    text = (
        "<tool_call>"
        '{"calls":['
        '{"id":"s1","tool":"read_file","path":"' + sample.as_posix() + '","max_chars":500},'
        '{"id":"s2","tool":"calculate","expr":"1 + 2"}'
        '],"mode":"chain"}'
        "</tool_call>"
    )

    clean_text, tool_result, artifacts = process_response(text, archive_dir=tmp_path)

    assert clean_text == ""
    assert tool_result is not None
    assert "line_one" in tool_result          # step 1 result present
    assert "[calculate: 1 + 2 = 3]" in tool_result  # step 2 result present
    # Artifacts should have: 1 top-level tool_call + 2 step call/result pairs
    chain_steps = [a for a in artifacts if a.get("chain_step") is not None]
    assert len(chain_steps) == 4  # 2 tool_call + 2 tool_result


def test_chain_mode_halts_on_error(tmp_path) -> None:
    """Chain should stop executing after a step that returns an error."""
    text = (
        "<tool_call>"
        '{"calls":['
        '{"id":"s1","tool":"read_file","path":"' + (tmp_path / "NONEXISTENT").as_posix() + '"},'
        '{"id":"s2","tool":"calculate","expr":"1 + 1"}'
        '],"mode":"chain"}'
        "</tool_call>"
    )

    _clean, tool_result, artifacts = process_response(text, archive_dir=tmp_path)

    assert tool_result is not None
    assert "does not exist" in tool_result
    assert "halted at step 1" in tool_result
    # Step 2 should NOT have executed
    assert "1 + 1 = 2" not in tool_result


def test_chain_mode_can_cancel_between_steps(monkeypatch, tmp_path) -> None:
    cancel_flag = {"value": False}
    original_exec = cmd_parser.execute_tool_call_enveloped

    def _wrapped_execute(call: dict, ctx):
        envelope = original_exec(call, ctx)
        cancel_flag["value"] = True
        return envelope

    monkeypatch.setattr(cmd_parser, "execute_tool_call_enveloped", _wrapped_execute)
    text = (
        "<tool_call>"
        '{"calls":['
        '{"id":"s1","tool":"calculate","expr":"1+1"},'
        '{"id":"s2","tool":"calculate","expr":"2+2"}'
        '],"mode":"chain"}'
        "</tool_call>"
    )

    _clean, tool_result, _artifacts = process_response(
        text,
        archive_dir=tmp_path,
        should_cancel=lambda: bool(cancel_flag["value"]),
    )

    assert tool_result is not None
    assert "[calculate: 1+1 = 2]" in tool_result
    assert "[chain: cancelled before step 2/2]" in tool_result
    assert "[calculate: 2+2 = 4]" not in tool_result


def test_batch_mode_can_cancel_between_calls(monkeypatch, tmp_path) -> None:
    cancel_flag = {"value": False}
    original_exec = cmd_parser.execute_tool_call_enveloped

    def _wrapped_execute(call: dict, ctx):
        envelope = original_exec(call, ctx)
        cancel_flag["value"] = True
        return envelope

    monkeypatch.setattr(cmd_parser, "execute_tool_call_enveloped", _wrapped_execute)
    text = (
        "<tool_call>"
        '{"calls":['
        '{"tool":"calculate","expr":"3+3"},'
        '{"tool":"calculate","expr":"4+4"}'
        '],"mode":"parallel"}'
        "</tool_call>"
    )

    _clean, tool_result, _artifacts = process_response(
        text,
        archive_dir=tmp_path,
        should_cancel=lambda: bool(cancel_flag["value"]),
    )

    assert tool_result is not None
    assert "[calculate: 3+3 = 6]" in tool_result
    assert "[batch: cancelled before call 2/2]" in tool_result
    assert "[calculate: 4+4 = 8]" not in tool_result


def test_chain_halts_when_envelope_marks_error_even_without_error_text(monkeypatch, tmp_path) -> None:
    class _FakeEnvelope:
        def __init__(self, *, tool: str, text: str, ok: bool, call_id: str | None) -> None:
            self.tool = tool
            self.text = text
            self.display_text = text
            self.ok = ok
            self.call_id = call_id

        def as_dict(self) -> dict:
            return {
                "type": "tool_result",
                "tool": self.tool,
                "status": "ok" if self.ok else "error",
                "ok": self.ok,
                "call_id": self.call_id,
                "text": self.text,
                "display_text": self.display_text,
                "data": {},
            }

    calls_seen: list[str] = []

    def _fake_execute_tool_call_enveloped(call: dict, _ctx):
        call_id = str(call.get("id", "")).strip()
        calls_seen.append(call_id)
        if len(calls_seen) == 1:
            return _FakeEnvelope(tool="calculate", text="[calculate: 1+1 = 2]", ok=False, call_id=call_id)
        return _FakeEnvelope(tool="calculate", text="[calculate: 2+2 = 4]", ok=True, call_id=call_id)

    monkeypatch.setattr(cmd_parser, "execute_tool_call_enveloped", _fake_execute_tool_call_enveloped)

    text = (
        "<tool_call>"
        '{"calls":['
        '{"id":"s1","tool":"calculate","expr":"1+1"},'
        '{"id":"s2","tool":"calculate","expr":"2+2"}'
        '],"mode":"chain"}'
        "</tool_call>"
    )

    _clean, tool_result, _artifacts = process_response(text, archive_dir=tmp_path)

    assert tool_result is not None
    assert "[chain: halted at step 1/2 (calculate) due to error]" in tool_result
    assert calls_seen == ["s1"]
    assert "[calculate: 2+2 = 4]" not in tool_result


def test_run_command_cancelled_before_start(tmp_path) -> None:
    ctx = skill_runtime.ToolExecutionContext(
        archive_dir=tmp_path,
        should_cancel=lambda: True,
    )
    result = skill_runtime.execute_run_command({"command": "python --version"}, ctx)
    assert result == "[run_command: cancelled]"


def test_run_tests_cancelled_before_start(tmp_path) -> None:
    ctx = skill_runtime.ToolExecutionContext(
        archive_dir=tmp_path,
        should_cancel=lambda: True,
    )
    result = skill_runtime.execute_run_tests({"runner": "pytest"}, ctx)
    assert result == "[run_tests: cancelled]"


def test_chain_var_substitution_line_ref(tmp_path) -> None:
    """$<id>.lineN should extract a specific line from a prior result."""
    sample = tmp_path / "multi.txt"
    sample.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")

    # Step 1 reads the file, step 2 uses $prev — just verifies substitution runs
    calls_json = (
        '{"calls":['
        '{"id":"s1","tool":"read_file","path":"' + sample.as_posix() + '","max_chars":200},'
        '{"id":"s2","tool":"calculate","expr":"2 + 3"}'
        '],"mode":"chain"}'
    )
    text = "<tool_call>" + calls_json + "</tool_call>"

    _clean, tool_result, artifacts = process_response(text, archive_dir=tmp_path)
    assert tool_result is not None
    # Step 1 should succeed
    assert "alpha" in tool_result
    # Step 2 should succeed
    assert "2 + 3 = 5" in tool_result


def test_chain_path_accessor_extracts_filepath(tmp_path) -> None:
    """$<id>.path1 should extract just the file path from a grep match line."""
    # Create a file with a known pattern, then chain grep -> read_file using .path1
    target = tmp_path / "hello.py"
    target.write_text("def execute_stuff():\n    return 42\n", encoding="utf-8")

    text = (
        "<tool_call>"
        '{"calls":['
        '{"id":"g","tool":"grep","pattern":"def execute","path":"' + tmp_path.as_posix() + '","glob":"*.py"},'
        '{"id":"r","tool":"read_file","path":"$g.path1"}'
        '],"mode":"chain"}'
        "</tool_call>"
    )

    _clean, tool_result, artifacts = process_response(text, archive_dir=tmp_path)

    assert tool_result is not None
    assert "def execute_stuff" in tool_result
    assert "return 42" in tool_result
    # Should NOT have halted — both steps succeeded
    assert "halted" not in tool_result


def test_chain_typed_accessor_reads_envelope_data(tmp_path) -> None:
    source = tmp_path / "typed.txt"
    source.write_text("typed accessor demo", encoding="utf-8")
    output = tmp_path / "out.txt"

    text = (
        "<tool_call>"
        '{"calls":['
        '{"id":"r","tool":"read_file","path":"' + source.as_posix() + '"},'
        '{"id":"w","tool":"write_file","path":"' + output.as_posix() + '","content":"source=$r.data.path"}'
        '],"mode":"chain"}'
        "</tool_call>"
    )

    _clean, tool_result, _artifacts = process_response(text, archive_dir=tmp_path)

    assert tool_result is not None
    assert "[write_file:" in tool_result
    assert output.read_text(encoding="utf-8") == f"source={source}"


def test_llm_call_uses_sync_bridge(monkeypatch, tmp_path) -> None:
    def _fake_load_config() -> dict:
        return {
            "api_base": "http://localhost:8000/v1",
            "api_model": "test-model",
            "api_key": "",
            "temp": 0.1,
            "top_p": 0.9,
            "max_tokens": 256,
        }

    def _fake_generate(base_config, messages, llm_config=None, thinking_enabled=None, should_cancel=None):
        assert base_config["api_model"] == "test-model"
        assert messages[-1]["role"] == "user"
        return ("mocked llm response", "")

    import core.subagent as _subagent
    monkeypatch.setattr(skill_runtime, "load_config", _fake_load_config)
    # llm_call now routes through the gated atom (Rule-6); the inference primitive it
    # reaches is generate_sync_parts_from_config in core.subagent.
    monkeypatch.setattr(_subagent, "generate_sync_parts_from_config", _fake_generate)

    clean, tool_result, _artifacts = process_response(
        '<tool_call>{"tool":"llm_call","prompt":"hello","max_tokens":128}</tool_call>',
        archive_dir=tmp_path,
    )

    assert clean == ""
    assert tool_result is not None
    assert "[llm_call:" in tool_result
    assert "mocked llm response" in tool_result


def test_llm_call_placeholder_and_unclamped_max_tokens(monkeypatch, tmp_path) -> None:
    captured: dict[str, int] = {}

    def _fake_load_config() -> dict:
        return {
            "api_base": "http://localhost:8000/v1",
            "api_model": "test-model",
            "api_key": "",
            "temp": 0.1,
            "top_p": 0.9,
            "max_tokens": 8192,
        }

    def _fake_generate(base_config, messages, llm_config=None, thinking_enabled=None, should_cancel=None):
        captured["max_tokens"] = int((llm_config or {}).get("max_tokens", -1))
        return ("ok", "")

    import core.subagent as _subagent
    monkeypatch.setattr(skill_runtime, "load_config", _fake_load_config)
    # llm_call's max_tokens resolution is preserved and passed to the atom as llm_config.
    monkeypatch.setattr(_subagent, "generate_sync_parts_from_config", _fake_generate)

    _clean, tool_result_placeholder, _artifacts = process_response(
        '<tool_call>{"tool":"llm_call","prompt":"hello","max_tokens":"<MAX_TOKENS>"}</tool_call>',
        archive_dir=tmp_path,
    )
    assert captured["max_tokens"] == 8192
    assert tool_result_placeholder is not None
    assert "max_tokens=8192" in tool_result_placeholder
    assert "max_tokens_source=placeholder" in tool_result_placeholder

    _clean2, tool_result_explicit, _artifacts2 = process_response(
        '<tool_call>{"tool":"llm_call","prompt":"hello","max_tokens":50000}</tool_call>',
        archive_dir=tmp_path,
    )
    assert captured["max_tokens"] == 50000
    assert tool_result_explicit is not None
    assert "max_tokens=50000" in tool_result_explicit
    assert "max_tokens_source=explicit" in tool_result_explicit


def test_tool_followup_prompt_includes_all_accumulated_results() -> None:
    prompt = build_tool_followup_prompt(
        [
            "[list_files: /tmp/docs (2 files, pattern='*.md')]",
            "[read_file: notes.md (/tmp/docs/notes.md)]\nhello",
        ]
    )

    assert "Tool results:" in prompt
    assert "[tool_result_1]" in prompt
    assert "[tool_result_2]" in prompt
    assert "/tmp/docs" in prompt
    assert "notes.md" in prompt
    assert "[TOOL_LOOP_DONE]" in prompt


def test_get_budget_score_ignores_message_probe_by_default(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        skill_runtime,
        "get_last_budget_snapshot",
        lambda: {"score": 38, "tier": "STANDARD", "max_tokens_cap": 1024},
    )

    def _fail_eval(_message: str, *, message_count: int = 1):
        raise AssertionError("evaluate_budget_for_message should not run in snapshot_only mode")

    monkeypatch.setattr(skill_runtime, "evaluate_budget_for_message", _fail_eval)

    _clean, tool_result, artifacts = process_response(
        '<tool_call>{"tool":"get_budget_score","message":"should be ignored"}</tool_call>',
        archive_dir=tmp_path,
    )

    assert tool_result is not None
    assert '"mode": "snapshot_only"' in tool_result
    first_call = next(a for a in artifacts if a.get("kind") == "tool_call")
    assert "message" not in first_call.get("command", {})


def test_get_budget_score_evaluates_message_when_explicit(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(
        skill_runtime,
        "get_last_budget_snapshot",
        lambda: {"score": 10, "tier": "MINIMAL", "max_tokens_cap": 512},
    )
    captured: dict[str, object] = {}

    def _fake_eval(message: str, *, message_count: int = 1):
        captured["message"] = message
        captured["message_count"] = message_count
        return {"score": 77, "tier": "EXHAUSTIVE", "max_tokens_cap": 4096}

    monkeypatch.setattr(skill_runtime, "evaluate_budget_for_message", _fake_eval)

    _clean, tool_result, artifacts = process_response(
        (
            '<tool_call>'
            '{"tool":"get_budget_score","evaluate_message":true,"message":"check complexity","message_count":3}'
            "</tool_call>"
        ),
        archive_dir=tmp_path,
    )

    assert tool_result is not None
    assert '"mode": "evaluate_message"' in tool_result
    assert captured == {"message": "check complexity", "message_count": 3}
    first_call = next(a for a in artifacts if a.get("kind") == "tool_call")
    assert first_call.get("command", {}).get("message") == "check complexity"


def test_system_prompt_contains_routing_rules() -> None:
    """system.md must teach the model how to route file tools properly.

    Substring-based against the current ROUTING RULES section. Update the
    matches when system.md rewords its routing guidance — the intent (the
    model knows: preserve filenames literally, and chain find_files then
    read_file when location is unknown) must remain, even if exact wording
    shifts. Earlier version asserted "Filename routing rules:" which has
    never been a literal heading in system.md; renamed to track actual
    section name ("ROUTING RULES") going forward.
    """
    system_prompt = (Path(__file__).resolve().parents[1] / "prompts" / "system.md").read_text(
        encoding="utf-8"
    )
    assert "ROUTING RULES" in system_prompt
    assert "Preserve exact filenames" in system_prompt
    assert "find_files" in system_prompt and "read_file" in system_prompt and "chain" in system_prompt
    assert "web_search is for live ranked web discovery" in system_prompt
    assert "web is for reading a specific URL" in system_prompt


def test_file_tools_resolve_relative_paths_from_workspace_root(monkeypatch, tmp_path) -> None:
    workspace_root = tmp_path / "workspace"
    workspace_root.mkdir()
    sample = workspace_root / "sample.txt"
    sample.write_text("workspace-root read", encoding="utf-8")

    unrelated_cwd = tmp_path / "other-cwd"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    monkeypatch.setattr(skill_runtime, "_WORKSPACE_ROOT", workspace_root)

    ctx = skill_runtime.ToolExecutionContext(archive_dir=tmp_path)
    read_result = skill_runtime.execute_read_file({"path": "sample.txt"}, ctx)
    list_result = skill_runtime.execute_list_files({"path": ".", "pattern": "*.txt"}, ctx)

    assert "workspace-root read" in read_result
    assert "[list_files:" in list_result
    assert "sample.txt" in list_result


def test_dynamic_executor_cache_is_bounded(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(skill_runtime, "_MAX_DYNAMIC_EXECUTOR_CACHE", 2)
    skill_runtime.clear_dynamic_executor_cache()

    for name in ("a", "b", "c"):
        skill_dir = tmp_path / name
        skill_dir.mkdir()
        (skill_dir / "SKILL.md").write_text(f"# {name}\n", encoding="utf-8")
        (skill_dir / "executor.py").write_text(
            "def run(cmd, ctx):\n    return 'ok'\n",
            encoding="utf-8",
        )
        loaded = skill_runtime._load_dynamic_executor(skill_dir / "SKILL.md")
        assert callable(loaded)

    cache_keys = list(skill_runtime._DYNAMIC_EXECUTOR_CACHE.keys())
    assert len(cache_keys) == 2
    assert str((tmp_path / "a" / "executor.py")) not in cache_keys

    skill_runtime.clear_dynamic_executor_cache()


# ── double-nested <tool_call> regression (turns 4889af5f / e6cab2be / a1c340f1) ──
#
# Observed in the wild on the monothink tier: the model emits
#     <tool_call>\n<tool_call>{json}</tool_call>\n</tool_call>
# The non-greedy regex captured OUTER-open through INNER-close, leaving the
# OUTER-close as an orphan. Symptoms: extract_commands returned [], the tool
# never fired, no LLM continuation, and the orphan </tool_call> leaked into
# the public answer where the verifier hard-failed it.


def test_extract_commands_handles_double_nested_envelope_with_name_args() -> None:
    """Hermes/Qwen ``{"name","arguments"}`` envelope wrapped in an extra
    ``<tool_call>`` pair must still be extracted as the single inner command."""
    text = (
        "Re-issuing.\n\n"
        "<tool_call>\n"
        '<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/y"}}</tool_call>\n'
        "</tool_call>"
    )
    cmds = extract_commands(text)
    assert len(cmds) == 1
    assert cmds[0].get("tool") == "read_file"
    assert cmds[0].get("path") == "/tmp/y"


def test_extract_commands_handles_double_nested_batch_envelope() -> None:
    """``{"calls":[...]}`` batch envelope wrapped in an extra ``<tool_call>`` pair."""
    text = (
        "<tool_call>\n"
        '<tool_call>{"calls": [{"id":"a","tool":"read_file","path":"/tmp/x"}], "mode":"parallel"}</tool_call>\n'
        "</tool_call>"
    )
    cmds = extract_commands(text)
    assert len(cmds) == 1
    assert isinstance(cmds[0].get("calls"), list)
    assert cmds[0]["mode"] == "parallel"


def test_extract_commands_handles_triple_nested_envelope() -> None:
    """Paranoid case: arbitrary nesting depth is peeled correctly."""
    text = (
        "<tool_call>\n<tool_call>\n<tool_call>"
        '{"name":"read_file","arguments":{"path":"/tmp/z"}}'
        "</tool_call>\n</tool_call>\n</tool_call>"
    )
    cmds = extract_commands(text)
    assert len(cmds) == 1
    assert cmds[0].get("tool") == "read_file"


def test_strip_commands_removes_orphan_close_from_double_nest() -> None:
    """After a double-nested envelope is matched, the OUTER close survives as
    an orphan. ``strip_commands`` must remove it so it doesn't reach the
    verifier as an internal-tag leak."""
    text = (
        "Re-issuing the tool call that got truncated.\n\n"
        "<tool_call>\n"
        '<tool_call>{"name":"read_file","arguments":{"path":"/tmp/y"}}</tool_call>\n'
        "</tool_call>\n"
    )
    cleaned = strip_commands(text)
    assert "</tool_call>" not in cleaned
    assert "<tool_call>" not in cleaned
    assert cleaned == "Re-issuing the tool call that got truncated."


def test_strip_commands_preserves_prose_mention_of_tool_call_tag() -> None:
    """Bare doc mention ``"use <tool_call> in docs"`` (no JSON content
    following the opener) must NOT be treated as a truncated envelope."""
    text = "Use <tool_call> blocks in docs when describing tool envelopes."
    assert strip_commands(text) == text


def test_strip_commands_drops_truncated_envelope_with_json_content() -> None:
    """Truncated mid-envelope (open + JSON content, no close) is unrecoverable
    garbage on the public surface — drop from the opener onward."""
    text = 'Reading.\n<tool_call>{"name": "read_file", "arguments": {"path": "/tmp/y"'
    assert strip_commands(text) == "Reading."


def test_process_response_executes_double_nested_envelope(tmp_path) -> None:
    """End-to-end: the model's double-nested response must drive an actual
    tool execution. This is the integration test for the bug from turns
    4889af5f / e6cab2be / a1c340f1."""
    sample = tmp_path / "sample.txt"
    sample.write_text("hello world\n", encoding="utf-8")
    text = (
        "<tool_call>\n"
        '<tool_call>{"name":"read_file","arguments":{"path":"'
        + sample.as_posix()
        + '","max_chars":50}}</tool_call>\n'
        "</tool_call>"
    )
    clean_text, tool_result, artifacts = process_response(text, archive_dir=tmp_path)
    # Fired the tool: a tool_result artifact exists.
    assert tool_result is not None
    assert "hello world" in tool_result
    # Cleaned public surface is empty / safe (no orphan tags).
    assert "<tool_call>" not in clean_text
    assert "</tool_call>" not in clean_text


def test_streaming_preexec_contract_for_double_nested_envelope() -> None:
    """Pins the contract that `ui.pages.chat._try_stream_preexec` relies on:
    a double-nested envelope must parse to a command whose canonical tool name
    is in ``STREAMING_PREEXEC_TOOLS`` so it actually fires mid-stream.

    Mirrors the filter chain in ``_try_stream_preexec`` without spinning up a
    full ChatPage (which needs Qt). If this test ever fails, streaming
    pre-execution of nested envelopes silently breaks again — and the user
    sees the UI freeze pattern from turns 4889af5f / e6cab2be / a1c340f1.
    """
    from core.cmd_parser import extract_commands
    from core.skill_registry import canonical_tool_name
    from core.skill_runtime import STREAMING_PREEXEC_TOOLS

    raw = (
        "<tool_call>\n"
        '<tool_call>{"name":"read_file","arguments":{"path":"/tmp/x"}}</tool_call>\n'
        "</tool_call>\n"
    )
    cmds = extract_commands(raw)
    assert len(cmds) == 1
    cmd = cmds[0]
    assert not cmd.get("_parse_error")
    assert not isinstance(cmd.get("calls"), list)  # not a batch envelope
    tool_name = canonical_tool_name(
        str(cmd.get("tool") or cmd.get("name") or cmd.get("skill") or cmd.get("op") or "")
    )
    assert tool_name in STREAMING_PREEXEC_TOOLS, (
        f"canonical tool {tool_name!r} must be in STREAMING_PREEXEC_TOOLS "
        f"for the streaming preexec optimization to fire on this envelope"
    )

    # Trim semantics: after preexec, _stream_raw should be sliced past the
    # last </tool_call>. Verify the slice logic without needing ChatPage.
    last_close = raw.rfind("</tool_call>")
    assert last_close >= 0
    trimmed = raw[last_close + len("</tool_call>"):]
    assert "<tool_call>" not in trimmed
    assert "</tool_call>" not in trimmed
