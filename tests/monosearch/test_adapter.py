import pytest
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import Record, EvidenceTier, Provenance


def test_cannot_instantiate_abstract_adapter():
    with pytest.raises(TypeError):
        SourceAdapter()  # type: ignore[abstract]


class _Stub(SourceAdapter):
    name = "stub"
    evidence_tier = EvidenceTier.DERIVED

    def search(self, query, filters, limit):
        return [Record("stub:1", "stub", Provenance.SELF, None, query, {}, None, EvidenceTier.DERIVED)]

    def get(self, namespaced_id):
        return None

    def list(self, filters, limit):
        return []


def test_concrete_subclass_satisfies_contract():
    a = _Stub()
    assert a.name == "stub"
    assert a.search("hi", {}, 5)[0].text == "hi"
    assert a.get("stub:9") is None
    assert a.list({}, 5) == []
