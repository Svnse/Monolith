from core.monosearch import registry
from core.monosearch.adapter import SourceAdapter
from core.monosearch.record import EvidenceTier


class _A(SourceAdapter):
    name = "a"
    evidence_tier = EvidenceTier.LITERAL
    def search(self, q, f, l): return []
    def get(self, i): return None
    def list(self, f, l): return []


def test_register_and_retrieve():
    registry.clear()
    a = _A()
    registry.register(a)
    assert registry.get_adapter("a") is a
    assert a in registry.all_adapters()


def test_register_is_idempotent_by_name():
    registry.clear()
    registry.register(_A())
    registry.register(_A())
    assert len(registry.all_adapters()) == 1
