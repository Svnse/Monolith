"""Source-targeting + diversity in the router — the 'get any info it wants' fixes
for canonical_log domination."""
from core.monosearch import registry, router
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record


class _Stub(SourceAdapter):
    name = "stub"
    evidence_tier = EvidenceTier.LITERAL

    def __init__(self, name, n, tier=EvidenceTier.LITERAL):
        self.name = name
        self.evidence_tier = tier
        self._n = n

    def search(self, q, f, l):
        return [
            Record(f"{self.name}:{i}", self.name, Provenance.SELF, None,
                   f"{self.name} {i}", {}, 1.0, self.evidence_tier)
            for i in range(self._n)
        ]

    def get(self, i):
        return None

    def list(self, f, l):
        return []


def test_search_scoped_to_one_source():
    registry.clear()
    registry.register(_Stub("canonical_log", 20))
    registry.register(_Stub("fault_traces", 5))
    recs = router.search("x", {"source": "faults"}, 10)  # alias -> fault_traces
    assert recs and all(r.source == "fault_traces" for r in recs)


def test_default_search_caps_per_source_for_diversity():
    registry.clear()
    registry.register(_Stub("canonical_log", 50))  # high volume, would flood
    registry.register(_Stub("fault_traces", 5))
    registry.register(_Stub("bearing", 5))
    recs = router.search("x", {}, 12)
    by_src = {}
    for r in recs:
        by_src[r.source] = by_src.get(r.source, 0) + 1
    assert by_src.get("canonical_log", 0) < 12, "canonical_log filled every slot"
    assert len(by_src) >= 2, f"no diversity across sources: {by_src}"


def test_default_search_backfills_when_only_one_source_hits():
    registry.clear()
    registry.register(_Stub("canonical_log", 20))
    recs = router.search("x", {}, 10)
    # only canonical_log matched -> no other source to protect -> backfill to limit
    assert len(recs) == 10 and all(r.source == "canonical_log" for r in recs)


def test_resolve_source_aliases():
    assert router.resolve_source("knowledge") == "acatalepsy-acus"
    assert router.resolve_source("warrants") == "acatalepsy-warrants"
    assert router.resolve_source("claim_graph") == "acatalepsy-warrants"
    assert router.resolve_source("warrants/claim_graph") == "acatalepsy-warrants"
    assert router.resolve_source("claim evidence graph") == "acatalepsy-warrants"
    assert router.resolve_source("conversation/history/canonical_log") == "canonical_log"
    assert router.resolve_source("runtime health") == "runtime_health"
    assert router.resolve_source("run-tests") == "skills"
    assert router.resolve_source("author workshop card") == "skills"
    assert router.resolve_source("faults") == "fault_traces"
    assert router.resolve_source("conversation") == "canonical_log"
    assert router.resolve_source("bearing") == "bearing"
    assert router.resolve_source("stages") == "stage_traces"
    assert router.resolve_source("reminders") == "plan_reminders"
    assert router.resolve_source("investigations") == "investigations"
    assert router.resolve_source("lag") == "lag_watch"
    assert router.resolve_source("health") == "runtime_health"
    assert router.resolve_source("nonsense") is None
    assert router.resolve_source(None) is None


def test_source_guidance_for_near_miss_sources():
    assert "please use source='warrants'" in router.source_usage_hint("warrants/claim_graph")
    assert "source='history'" in router.source_usage_hint("conversation/history/canonical_log")
    assert "meta='capabilities'" in router.source_usage_hint("tools/skills")
    assert "source='skills'" in router.source_usage_hint("author workshop card")
    assert "run_tests" in (router.source_query_hint("run-tests") or "")
    assert "one source per call" in router.source_usage_hint("faults/history")
    assert router.resolve_source("faults/history") is None
