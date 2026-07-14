from core.monosearch.record import Record, EvidenceTier, Provenance


def test_record_is_frozen_and_holds_fields():
    r = Record(
        namespaced_id="fault:7",
        source="fault_traces",
        provenance=Provenance.SELF,
        recurrence_key="think_leak|abc",
        text="leaked a think tag",
        metadata={"fault_kind": "think_leak"},
        ts=1717000000.0,
        evidence_tier=EvidenceTier.LITERAL,
    )
    assert r.namespaced_id == "fault:7"
    assert r.provenance is Provenance.SELF
    assert r.evidence_tier == EvidenceTier.LITERAL


def test_evidence_tier_is_a_hard_ordering():
    # literal must sort BEFORE derived/telemetry/speculative
    assert EvidenceTier.LITERAL < EvidenceTier.DERIVED < EvidenceTier.TELEMETRY < EvidenceTier.SPECULATIVE


def test_record_is_immutable():
    import dataclasses
    r = Record("a", "s", Provenance.SELF, None, "t", {}, None, EvidenceTier.DERIVED)
    try:
        r.text = "x"  # type: ignore[misc]
        assert False, "Record should be frozen"
    except dataclasses.FrozenInstanceError:
        pass
