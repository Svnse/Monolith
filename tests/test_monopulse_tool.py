from __future__ import annotations

from core.monopulse import PulseReport


def test_monopulse_is_discoverable_dispatchable_and_protected_from_trim():
    from core.context_profiles import _PROFILES, trim_tool_specs
    from core.skill_registry import _TOOL_RUNTIME_META, clear_skill_cache, list_tools
    from core.skill_runtime import _TOOL_EXECUTORS

    clear_skill_cache()
    specs = list_tools()
    names = {spec.name for spec in specs}
    kept = {getattr(spec, "name", "") for spec in trim_tool_specs(specs, _PROFILES["tiny_local"])}

    assert "monopulse" in names
    assert "monopulse" in _TOOL_RUNTIME_META
    assert "monopulse" in _TOOL_EXECUTORS
    assert "monopulse" in kept


def test_execute_monopulse_dispatches_to_core(monkeypatch):
    from core import monopulse
    from core.skill_runtime import execute_monopulse

    captured = {}

    def fake_run(verb: str, *, limit: int) -> PulseReport:
        captured["verb"] = verb
        captured["limit"] = limit
        return PulseReport(mode=verb, generated_at="now", items=(), summary={"status": "quiet"})

    monkeypatch.setattr(monopulse, "run", fake_run)
    monkeypatch.setattr(monopulse, "format_report", lambda report: f"formatted {report.mode}")

    out = execute_monopulse({"verb": "drift", "limit": 2}, None)

    assert out == "formatted drift"
    assert captured == {"verb": "drift", "limit": 2}


def test_monopulse_envelope_validation_rejects_unknown_fields(tmp_path):
    from core.skill_runtime import ToolExecutionContext, execute_tool_call_enveloped

    ctx = ToolExecutionContext(archive_dir=tmp_path)
    env = execute_tool_call_enveloped(
        {"tool": "monopulse", "verb": "pulse", "bogus": "nope"},
        ctx,
    )

    assert env.tool == "monopulse"
    assert env.ok is False
    assert "unknown field(s): bogus" in env.text
