from core.monosearch import registry, router
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier, Provenance, Record


def _rec(nsid, tier, ts=None, text="x"):
    src = nsid.split(":")[0]
    return Record(nsid, src, Provenance.SELF, None, text, {}, ts, tier)


class _Lit(SourceAdapter):
    name = "lit"
    evidence_tier = EvidenceTier.LITERAL
    def search(self, q, f, l): return [_rec("lit:1", EvidenceTier.LITERAL, ts=100.0)]
    def get(self, i): return _rec("lit:1", EvidenceTier.LITERAL) if i == "lit:1" else None
    def list(self, f, l): return []


class _Spec(SourceAdapter):
    name = "spec"
    evidence_tier = EvidenceTier.SPECULATIVE
    def search(self, q, f, l): return [_rec("spec:1", EvidenceTier.SPECULATIVE, ts=200.0)]
    def get(self, i): return None
    def list(self, f, l): return []


def test_evidence_tier_is_a_hard_ordering():
    registry.clear(); registry.register(_Spec()); registry.register(_Lit())
    out = router.search("anything", {}, limit=10)
    # literal must come first even though the speculative record is newer
    assert out[0].namespaced_id == "lit:1"
    assert out[1].namespaced_id == "spec:1"


def test_get_routes_by_namespace_prefix():
    registry.clear(); registry.register(_Lit())
    assert router.get("lit:1").namespaced_id == "lit:1"
    assert router.get("nope:1") is None


def test_since_filter_drops_older_records():
    registry.clear(); registry.register(_Lit())  # lit:1 has ts=100.0
    assert router.search("x", {"since": "1970-01-01T00:00:00+00:00"}, 10)  # ts 100 >= 0 -> kept
    assert router.search("x", {"since": "2000-01-01T00:00:00+00:00"}, 10) == []  # ts 100 < cutoff -> dropped


def test_ratings_source_alias_resolves_to_outcome_traces():
    # 'ratings' (and friends) target the outcome_traces ledger — the source that
    # holds rater failure_tags like premise_unchecked.
    assert router.resolve_source("ratings") == "outcome_traces"
    assert router.resolve_source("grades") == "outcome_traces"
    assert router.resolve_source("outcome_traces") == "outcome_traces"
