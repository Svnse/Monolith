from __future__ import annotations

from core.monosearch import registry, service
from core.monosearch.adapters.skills import SkillsAdapter
from core.monosearch.adapters.tools import ToolsAdapter
from core.skill_registry import clear_skill_cache


def test_tools_adapter_searches_tools_and_gets_exact_schema() -> None:
    clear_skill_cache()
    registry.clear()
    registry.register(ToolsAdapter())

    recs = service.search("edit file", {"source": "tools"}, limit=5)

    assert any(rec.namespaced_id == "tool:edit_file" for rec in recs)
    detail = service.get("tool:edit_file")
    assert detail is not None
    assert "[TOOL]" in detail.text
    assert "name: edit_file" in detail.text
    assert "path" in detail.text
    assert "find" in detail.text
    assert "replace" in detail.text
    assert "example_call" in detail.text


def test_all_tool_call_hints_are_valid_copy_pasteable_json() -> None:
    """The call_hint must parse as JSON. Regression (2026-06-16): _detail_text
    hardcoded arguments:{...} (literal ellipsis) for EVERY tool; the model fetched
    the schema, copied the hint verbatim, and emitted unparseable JSON ->
    '[error: malformed tool call - could not parse JSON]'. A no-args tool must
    get {}, not {...}."""
    import json
    import re

    from core.skill_registry import list_tools

    clear_skill_cache()
    registry.clear()
    registry.register(ToolsAdapter())

    for spec in list_tools():
        detail = service.get(f"tool:{spec.name}")
        assert detail is not None, spec.name
        assert "{...}" not in detail.text, f"{spec.name}: call_hint still has literal ellipsis"
        m = re.search(r"call_hint: use <tool_call>(.*)</tool_call>", detail.text)
        assert m, f"{spec.name}: no call_hint line"
        payload = json.loads(m.group(1))  # must parse — was {...} before the fix
        assert payload["name"] == spec.name
        assert isinstance(payload["arguments"], dict)


def test_spawn_subagent_schema_declares_real_params_not_no_args() -> None:
    """Regression (2026-06-16): spawn_subagent was missing from the curated
    _TOOL_RUNTIME_META, so its schema rendered 'params: no args' with no example.
    The model then emitted empty {} calls (PENDING / generator-busy) because
    nothing told it to pass prompt/level/frame. It actually takes those."""
    import json
    import re

    clear_skill_cache()
    registry.clear()
    registry.register(ToolsAdapter())

    detail = service.get("tool:spawn_subagent")
    assert detail is not None
    assert "no args" not in detail.text
    assert "prompt" in detail.text
    assert "level" in detail.text
    assert "example_call:" in detail.text
    m = re.search(r"call_hint: use <tool_call>(.*)</tool_call>", detail.text)
    assert m
    payload = json.loads(m.group(1))  # still valid JSON
    assert "prompt" in payload["arguments"]  # now an informative hint, not {}
    # Reprobe finding (2026-06-16): the model copies the call_hint VERBATIM, so a
    # "<prompt>" placeholder becomes a literal garbage spawn. The hint must carry
    # the real example_call value, never a "<...>" placeholder.
    assert "<prompt>" not in detail.text
    assert payload["arguments"]["prompt"] != "<prompt>"


def test_no_args_tool_hint_uses_empty_object() -> None:
    import json
    import re

    clear_skill_cache()
    registry.clear()
    registry.register(ToolsAdapter())

    detail = service.get("tool:reload_skills")  # a genuinely no-args tool
    assert detail is not None
    m = re.search(r"call_hint: use <tool_call>(.*)</tool_call>", detail.text)
    assert m
    payload = json.loads(m.group(1))
    assert payload["arguments"] == {}  # no-args -> {}, copy-pasteable & valid


def test_skills_adapter_exposes_skill_card_with_tool_schema_pointer() -> None:
    clear_skill_cache()
    registry.clear()
    registry.register(SkillsAdapter())

    recs = service.search("workflow card", {"source": "skills"}, limit=10)

    assert any(rec.namespaced_id == "skill:author_workshop_card" for rec in recs)
    detail = service.get("skill:author_workshop_card")
    assert detail is not None
    assert "[SKILL]" in detail.text
    assert "tool_schema_id: tool:author_workshop_card" in detail.text


def test_web_search_is_high_signal_in_tool_monosearch() -> None:
    import json
    import re

    clear_skill_cache()
    registry.clear()
    registry.register(ToolsAdapter())

    recs = service.search("latest online", {"source": "tools"}, limit=3)

    assert recs
    assert recs[0].namespaced_id == "tool:web_search"
    assert "current/latest/live online internet search" in recs[0].text
    assert "chain web_search -> web" in recs[0].text
    assert "needed. next: monosearch get id=\"tool:web_search\"" in recs[0].text

    detail = service.get("tool:web_search")
    assert detail is not None
    assert "use_when: current/latest/live online internet search" in detail.text
    assert "aliases: search_web" in detail.text
    assert "- query (required): Live web/internet query" in detail.text
    assert "- start_date (optional):" in detail.text
    assert "- include_usage (optional):" in detail.text
    assert "- timeout (optional):" in detail.text
    assert '"required": ["query"]' in detail.text
    m = re.search(r"call_hint: use <tool_call>(.*)</tool_call>", detail.text)
    assert m
    payload = json.loads(m.group(1))
    assert payload == {
        "name": "web_search",
        "arguments": {"query": "latest AI audio tools", "max_results": 5},
    }


def test_open_file_is_high_signal_for_document_reading_in_tool_monosearch() -> None:
    import json
    import re

    clear_skill_cache()
    registry.clear()
    registry.register(ToolsAdapter())

    recs = service.search("read pdf zip docx xlsx image OCR local file", {"source": "tools"}, limit=5)

    assert recs
    assert recs[0].namespaced_id == "tool:open_file"
    assert "pdf, docx, xlsx" in recs[0].text
    assert "zip/archive" in recs[0].text
    assert "ocr fallback status" in recs[0].text

    detail = service.get("tool:open_file")
    assert detail is not None
    assert "use_when: open/read/inspect local files" in detail.text
    assert "- member (optional): Archive member path" in detail.text
    assert "- sheet (optional): Worksheet name" in detail.text
    assert "aliases: open-file, read_any_file, read-document, read_document" in detail.text
    assert '"required": ["path"]' in detail.text
    m = re.search(r"call_hint: use <tool_call>(.*)</tool_call>", detail.text)
    assert m
    payload = json.loads(m.group(1))
    assert payload == {
        "name": "open_file",
        "arguments": {"path": "C:/Users/name/document.pdf", "max_chars": 8000},
    }


def test_web_search_is_high_signal_in_skill_monosearch() -> None:
    clear_skill_cache()
    registry.clear()
    registry.register(SkillsAdapter())

    recs = service.search("latest online", {"source": "skills"}, limit=3)

    assert recs
    assert recs[0].namespaced_id == "skill:web_search"
    assert "current/latest/live online internet search" in recs[0].text
    assert "needed. tool_schema: tool:web_search" in recs[0].text
    assert "tool_schema: tool:web_search" in recs[0].text
