"""Tests for the subsystem map walker and the introspect scratchpad op.

Closes the spatial-addressing gap surfaced in the 2026-05-20 audit:
junior can now enumerate its own subsystems via op=introspect instead
of requiring an external agent to grep the codebase.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

from core import subsystem_map as smap


def test_build_subsystem_map_has_expected_top_level_keys() -> None:
    data = smap.build_subsystem_map()
    assert isinstance(data, dict)
    assert data.get("schema_version") == 1
    assert "generated_at" in data
    assert "git_sha" in data
    for section in ("pipeline_policies", "planes", "skills", "interceptors"):
        assert section in data, f"missing section: {section}"
        assert isinstance(data[section], list)


def test_pipeline_policies_have_expected_shape() -> None:
    data = smap.build_subsystem_map()
    policies = data["pipeline_policies"]
    # The registry currently declares 6 policies (output_sanitizer,
    # tool_failure_classifier, tool_loop_continuation, parse_retry,
    # verifier_bridge, subordinate_clause_detector). At least 5 must surface.
    assert len(policies) >= 5
    for p in policies:
        assert isinstance(p.get("name"), str) and p["name"]
        assert isinstance(p.get("subscribes_to"), list)
        assert isinstance(p.get("depends_on"), list)
        assert p.get("authority_tier") in {"observation", "mutation", "dispatch"}


def test_planes_have_expected_shape() -> None:
    data = smap.build_subsystem_map()
    planes = {p["name"]: p for p in data.get("planes", [])}
    # After /prompt consolidation, legacy planes may be absent.
    # If planes are present, verify their shape.
    for p in planes.values():
        assert isinstance(p.get("valid_modes", []), list)
        assert isinstance(p.get("silent_modes", []), list)


def test_skills_include_known_names() -> None:
    data = smap.build_subsystem_map()
    names = {s["name"] for s in data["skills"]}
    for known in ("scratchpad", "read_file", "grep"):
        assert known in names, f"expected skill missing from map: {known}"


def test_dump_and_read_roundtrip(tmp_path) -> None:
    target = tmp_path / "subsystem_map.json"
    written = smap.dump_subsystem_map(path=target)
    assert written == target
    assert target.exists()
    loaded = smap.read_subsystem_map(path=target)
    assert loaded is not None
    assert loaded.get("schema_version") == 1
    raw = json.loads(target.read_text(encoding="utf-8"))
    assert raw == loaded


def test_read_returns_none_on_missing_file(tmp_path) -> None:
    target = tmp_path / "nope.json"
    assert smap.read_subsystem_map(path=target) is None


def test_read_returns_none_on_corrupt_file(tmp_path) -> None:
    target = tmp_path / "bad.json"
    target.write_text("this is not json", encoding="utf-8")
    assert smap.read_subsystem_map(path=target) is None


def test_format_subsystem_map_kind_filter() -> None:
    data = smap.build_subsystem_map()
    only_planes = smap.format_subsystem_map(data, kind="planes")
    assert "== planes" in only_planes
    assert "== policies" not in only_planes
    assert "== skills" not in only_planes


def test_format_subsystem_map_name_filter() -> None:
    data = smap.build_subsystem_map()
    out = smap.format_subsystem_map(data, kind="skills", name_filter="scratchpad")
    assert "scratchpad" in out
    # Other skills should not appear when name_filter narrows
    assert "read_file" not in out


def test_format_subsystem_map_handles_missing_data() -> None:
    out = smap.format_subsystem_map(None)
    assert "not present" in out


@pytest.fixture()
def scratchpad_exec(tmp_path, monkeypatch):
    # Point the subsystem map at tmp_path so we can control it.
    monkeypatch.setattr(smap, "_MAP_PATH", tmp_path / "subsystem_map.json")
    spec_path = Path(__file__).parent.parent / "skills" / "scratchpad" / "executor.py"
    spec = importlib.util.spec_from_file_location("scratchpad_introspect_test", spec_path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod, tmp_path


def test_introspect_op_returns_not_present_before_dump(scratchpad_exec) -> None:
    mod, _ = scratchpad_exec
    out = mod.run({"op": "introspect"}, None)
    assert "not present" in out


def test_introspect_op_returns_formatted_map_after_dump(scratchpad_exec) -> None:
    mod, tmp_path = scratchpad_exec
    smap.dump_subsystem_map(path=tmp_path / "subsystem_map.json")
    out = mod.run({"op": "introspect"}, None)
    assert "[introspect:" in out
    assert "== policies" in out
    assert "== planes" in out
    assert "== skills" in out


def test_introspect_op_filters_by_kind(scratchpad_exec) -> None:
    mod, tmp_path = scratchpad_exec
    smap.dump_subsystem_map(path=tmp_path / "subsystem_map.json")
    out = mod.run({"op": "introspect", "kind": "planes"}, None)
    assert "== planes" in out
    assert "== skills" not in out


def test_introspect_op_filters_by_name(scratchpad_exec) -> None:
    mod, tmp_path = scratchpad_exec
    smap.dump_subsystem_map(path=tmp_path / "subsystem_map.json")
    out = mod.run({"op": "introspect", "kind": "skills", "name": "scratchpad"}, None)
    assert "scratchpad" in out
    assert "read_file" not in out


def test_tool_validation_allows_introspect_fields() -> None:
    from core.tool_validation import validate_tool_arguments

    errors = validate_tool_arguments(
        "scratchpad",
        {"op": "introspect", "kind": "planes", "name": "effort"},
    )
    assert errors == []
