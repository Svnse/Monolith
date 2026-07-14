"""The spawn_subagent command + result render as a compact one-line Agent
indicator (E 2026-06-16: replace the verbose subagent bubble). The model's
[SUBAGENT_RESULT] fold-back is untouched; this is human display only."""
from __future__ import annotations

from ui.components.tool_bubbles import _call_summary, _result_summary


def test_spawn_call_renders_compact_agent_line() -> None:
    s = _call_summary({"tool": "spawn_subagent", "frame": "read-only view",
                       "prompt": "Read the file", "_validated_child_level": 2})
    assert "Agent" in s
    assert "read-only view" in s
    assert "running" in s.lower()
    assert "\n" not in s  # one line


def test_spawn_call_falls_back_to_level_label_without_frame() -> None:
    s = _call_summary({"tool": "spawn_subagent", "_validated_child_level": 3})
    assert "Agent" in s
    assert "L3" in s


def test_pending_result_is_running_oneliner() -> None:
    s = _result_summary({"tool": "spawn_subagent", "result": "[spawn_subagent: PENDING]"})
    assert "Agent" in s and "running" in s.lower()


def test_subagent_result_is_done_oneliner_not_verbose_dump() -> None:
    verbose = "[SUBAGENT_RESULT ok=true]\n" + ("x" * 4000) + "\n[/SUBAGENT_RESULT]"
    s = _result_summary({"tool": "subagent", "result": verbose})
    assert s == "● Agent (done)"
    assert "xxxx" not in s  # the verbose body is NOT shown to the human
