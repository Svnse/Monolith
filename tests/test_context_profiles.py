from __future__ import annotations

from dataclasses import dataclass

from core.context_profiles import build_profiled_tool_catalog, resolve_context_profile, trim_tool_specs


@dataclass(frozen=True)
class _Spec:
    name: str


def test_resolve_context_profile_defaults_to_standard() -> None:
    assert resolve_context_profile("missing").name == "standard_local"
    assert resolve_context_profile("standard_local").monosearch_result_count == 10


def test_trim_tool_specs_preserves_required_tools_first() -> None:
    profile = resolve_context_profile("tiny_local")
    specs = tuple(_Spec(name) for name in ["custom"] + ["read_file", "grep"] + [f"x{i}" for i in range(20)])

    trimmed = trim_tool_specs(specs, profile)

    assert len(trimmed) == profile.max_tool_catalog_entries
    names = [spec.name for spec in trimmed]
    assert names[0:2] == ["read_file", "grep"]


def test_profiled_tool_catalog_is_discovery_kernel_not_full_static_list() -> None:
    catalog = build_profiled_tool_catalog()

    assert "[TOOL DISCOVERY KERNEL]" in catalog
    assert "monosearch" in catalog
    assert 'meta":"tools"' in catalog
    assert "latest online web search" in catalog
    assert "source=\"history\"" in catalog
    assert "meta=debug" in catalog
    assert "does not search canonical_log" in catalog
    assert "Available tools:" not in catalog
