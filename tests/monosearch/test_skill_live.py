"""Live end-to-end + discoverability for the `monosearch` model tool. Proves the
selector is LIVE (the model gets real faults), not a dark store."""
from core import fault_response as fr
from core.monosearch import registry
from core.monosearch.adapter import SourceAdapter
from core.monosearch.adapters.faults import FaultAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record


class _WarrantsAdapter(SourceAdapter):
    name = "acatalepsy-warrants"
    evidence_tier = EvidenceTier.DERIVED

    def search(self, query, filters, limit):
        text = "[WARRANT GRAPH]\nclaim: friction | blocks | workflow\nwarrant: seeded test"
        if query and str(query).lower() not in text.lower():
            return []
        return [
            Record(
                "warrant:acu:1",
                self.name,
                Provenance.SELF,
                None,
                text,
                {"node_kind": "acu"},
                1.0,
                self.evidence_tier,
            )
        ][: int(limit)]

    def get(self, namespaced_id):
        if namespaced_id != "warrant:acu:1":
            return None
        return self.search("", {}, 1)[0]

    def list(self, filters, limit):
        return self.search("", filters, limit)


def test_monosearch_is_discoverable_and_dispatchable():
    from core.skill_registry import list_tools, _TOOL_RUNTIME_META
    from core.skill_runtime import _TOOL_EXECUTORS
    names = {t.name for t in list_tools()}
    assert "monosearch" in names, "monosearch not discoverable (missing skills/monosearch/SKILL.md)"
    assert "monosearch" in _TOOL_EXECUTORS, "monosearch has no executor (not dispatchable)"
    assert "monosearch" in _TOOL_RUNTIME_META


def test_monosearch_failing_returns_live_faults():
    from core.skill_runtime import execute_monosearch
    registry.clear()
    registry.register(FaultAdapter())
    for i in range(3):
        assert fr.emit_fault(turn_id=f"t{i}", fault_kind="think_leak",
                             detector_name="detect_think_leak", evidence=f"<think> leak snippet {i}") > 0
    out = execute_monosearch({"verb": "failing", "limit": 5}, None)
    assert "think_leak" in out, f"failing did not surface think_leak (dark store?): {out!r}"


def test_monosearch_failing_empty_is_graceful():
    from core.skill_runtime import execute_monosearch
    registry.clear()
    registry.register(FaultAdapter())
    out = execute_monosearch({"verb": "failing", "limit": 5}, None)
    assert "monosearch:failing" in out  # a clean 'none' message, not a crash


def test_monosearch_default_limit_comes_from_context_profile(monkeypatch):
    from types import SimpleNamespace

    from core.skill_runtime import execute_monosearch

    monkeypatch.setattr(
        "core.context_profiles.active_context_profile",
        lambda: SimpleNamespace(monosearch_result_count=1),
    )
    registry.clear()
    registry.register(FaultAdapter())
    assert fr.emit_fault("t1", "think_leak", "detect_think_leak", "<think>") > 0
    assert fr.emit_fault("t2", "markdown_corruption", "detect_markdown_corruption", "```") > 0

    out = execute_monosearch({"verb": "failing"}, None)

    assert "[monosearch:failing count=1]" in out


def test_monosearch_requires_verb():
    from core.skill_runtime import execute_monosearch
    out = execute_monosearch({}, None)
    assert "verb" in out.lower()


def test_monosearch_find_meta_tools_returns_catalog_hits():
    from core.monosearch.adapters.tools import ToolsAdapter
    from core.skill_registry import clear_skill_cache
    from core.skill_runtime import execute_monosearch

    clear_skill_cache()
    registry.clear()
    registry.register(ToolsAdapter())

    out = execute_monosearch(
        {"verb": "find", "meta": "tools", "query": "edit file", "limit": 5},
        None,
    )

    assert "[monosearch:find meta=tools" in out
    assert "tool:edit_file" in out
    assert "monosearch get" in out


def test_monosearch_defaults_query_source_call_to_search():
    from core.skill_runtime import execute_monosearch

    registry.clear()
    registry.register(_WarrantsAdapter())

    out = execute_monosearch({"source": "warrants", "query": "friction", "limit": 5}, None)

    assert "[monosearch:search source=warrants count=1]" in out
    assert "warrant:acu:1" in out


def test_monosearch_accepts_literal_search_find_verb_and_warrant_source():
    from core.skill_runtime import execute_monosearch

    registry.clear()
    registry.register(_WarrantsAdapter())

    out = execute_monosearch(
        {"verb": "search/find", "source": "claim_graph", "query": "friction", "limit": 5},
        None,
    )

    assert "[monosearch:search source=claim_graph count=1]" in out
    assert "warrant:acu:1" in out


def test_monosearch_accepts_slash_source_combo_with_guidance():
    from core.skill_runtime import execute_monosearch

    registry.clear()
    registry.register(_WarrantsAdapter())

    out = execute_monosearch(
        {"source": "warrants/claim_graph", "query": "friction", "limit": 5},
        None,
    )

    assert "guidance" in out
    assert "please use source='warrants' or source='claim_graph'" in out
    assert "[monosearch:search source=warrants/claim_graph count=1]" in out
    assert "warrant:acu:1" in out


def test_monosearch_parallel_correlation_id_does_not_force_get():
    """A parallel-batch correlation id ('a', 'call_1') must not hijack a search.

    Regression (2026-06-16): the parallel envelope auto-stamps `id` onto every
    sub-call (cmd_parser.expand_calls), and verb inference checked `id` BEFORE
    source/query/meta — so every verb-less monosearch call inside a batch routed
    to `get` with the label as handle => `[monosearch:get 'a' not found]`. The
    model had the right call and the envelope ate it.
    """
    from core.skill_runtime import execute_monosearch

    registry.clear()
    registry.register(_WarrantsAdapter())

    out = execute_monosearch(
        {"id": "a", "source": "claim_graph", "query": "friction", "limit": 5}, None
    )

    assert "[monosearch:search source=claim_graph count=1]" in out
    assert "warrant:acu:1" in out
    assert "not found" not in out


def test_monosearch_bare_id_still_routes_to_get():
    """Guard: an id-only call with no search signals is still a get-by-handle.
    The precedence fix must not break verb-less `get`."""
    from core.skill_runtime import execute_monosearch

    registry.clear()
    registry.register(_WarrantsAdapter())

    out = execute_monosearch({"id": "warrant:acu:1"}, None)

    assert "[monosearch:get warrant:acu:1]" in out


def test_monosearch_unknown_meta_that_is_a_valid_source_redirects():
    """meta='warrants'/'knowledge' are not meta buckets but ARE sources.

    Round-1 stumble on 2026-06-15 and -16: the model reached for `meta=` (the
    only form the always-on palette teaches) with a store name. Instead of a dead
    'unknown meta' wall, redirect to source= with guidance AND return results
    (house style — see slash-combo and skill-name-source tests)."""
    from core.skill_runtime import execute_monosearch

    registry.clear()
    registry.register(_WarrantsAdapter())

    out = execute_monosearch(
        {"verb": "find", "meta": "warrants", "query": "friction", "limit": 5}, None
    )

    assert "guidance" in out
    assert "source='warrants'" in out
    assert "warrant:acu:1" in out


def test_monosearch_source_hint_lists_stores_as_separate_tokens():
    """The unknown-source hint must present each store as its own comma token, not
    join two into one item ('warrants or claim_graph'). That ambiguity led the
    model to re-submit the invalid combined token 'warrants/claim_graph' 4x on
    2026-06-15 — the suggestion list echoed its own rejected shape."""
    from core.skill_runtime import execute_monosearch

    registry.clear()
    registry.register(_WarrantsAdapter())

    out = execute_monosearch({"source": "totally_unknown_xyz", "query": "x"}, None)

    assert "unknown source" in out
    assert "warrants or claim_graph" not in out
    assert "warrants, claim_graph" in out


def test_monosearch_routes_skill_name_source_to_skill_catalog():
    from core.monosearch.adapters.skills import SkillsAdapter
    from core.skill_registry import clear_skill_cache
    from core.skill_runtime import execute_monosearch

    clear_skill_cache()
    registry.clear()
    registry.register(SkillsAdapter())

    out = execute_monosearch(
        {"source": "author workshop card", "query": "workflow", "limit": 5},
        None,
    )

    assert "guidance" in out
    assert "source='skills'" in out
    assert "skill:author_workshop_card" in out


def test_monosearch_rejects_multi_store_source_with_specific_guidance():
    from core.skill_runtime import execute_monosearch

    registry.clear()
    registry.register(_WarrantsAdapter())

    out = execute_monosearch(
        {"source": "faults/history", "query": "tool", "limit": 5},
        None,
    )

    assert "source guidance" in out
    assert "one source per call" in out


def test_monosearch_bootstraps_registry_when_empty(monkeypatch):
    from core.skill_runtime import execute_monosearch

    registry.clear()

    def fake_init():
        registry.register(_WarrantsAdapter())

    monkeypatch.setattr("core.monosearch.bootstrap.init_monosearch", fake_init)

    out = execute_monosearch({"source": "warrants", "query": "friction", "limit": 5}, None)

    assert "[monosearch:search source=warrants count=1]" in out
    assert "warrant:acu:1" in out


def test_monosearch_survives_tool_catalog_trim_on_smallest_profile():
    """On a tight context profile the catalog is trimmed; the self-knowledge tool
    must be protected like inspect_pipeline, or the model can't call it."""
    from core.context_profiles import trim_tool_specs, _PROFILES
    from core.skill_registry import list_tools
    kept = {getattr(s, "name", "") for s in trim_tool_specs(list_tools(), _PROFILES["tiny_local"])}
    assert "monosearch" in kept, "monosearch trimmed out of the catalog on tiny_local"


def test_full_envelope_path_validates_dispatches_and_is_fresh(tmp_path):
    """The real call path: schema validation (additionalProperties:False) + dispatch
    + freshness (monosearch is NOT in _CACHEABLE_TOOLS, so not served stale)."""
    from core.skill_runtime import execute_tool_call_enveloped, ToolExecutionContext
    registry.clear()
    registry.register(FaultAdapter())
    assert fr.emit_fault(turn_id="t", fault_kind="markdown_corruption",
                         detector_name="detect_markdown_corruption", evidence="fence imbalance") > 0
    ctx = ToolExecutionContext(archive_dir=tmp_path)  # result_cache=None -> no caching
    env = execute_tool_call_enveloped({"tool": "monosearch", "verb": "failing", "limit": 5}, ctx)
    assert env.tool == "monosearch"
    assert env.ok, f"envelope not ok (validation/dispatch failed?): {env.text!r}"
    assert "markdown_corruption" in env.text  # validated + dispatched + live
    assert not (env.data or {}).get("cached", False)  # fresh, not cached
