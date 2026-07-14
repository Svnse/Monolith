"""Non-performative regression: the model-facing monosearch render must show a
CONSTRAINT marker ([unverified]) for un-established hits, never the raw tier name.
This is the property the original Stage-1 build asserted but never tested."""
import core.skill_runtime as sr
import core.monosearch.service as ms_service
from core.skill_runtime import _source_tier_marker
from core.monosearch.record import Record, EvidenceTier, Provenance


def _rec(source_tier=None):
    meta = {}
    if source_tier is not None:
        meta["source_tier"] = source_tier
    return Record(
        namespaced_id="turn:abc", source="turn_trace", provenance=Provenance.SELF,
        recurrence_key=None, text="some prior answer", metadata=meta, ts=None,
        evidence_tier=EvidenceTier.LITERAL,
    )


def test_marker_for_generation_when_flag_on(monkeypatch):
    monkeypatch.setenv("MONOLITH_SOURCE_TIER_V1", "1")
    assert _source_tier_marker(_rec("generation")) == " [unverified]"


def test_marker_empty_for_tool_and_none(monkeypatch):
    monkeypatch.setenv("MONOLITH_SOURCE_TIER_V1", "1")
    assert _source_tier_marker(_rec("tool")) == ""
    assert _source_tier_marker(_rec(None)) == ""


def test_marker_empty_when_flag_off(monkeypatch):
    monkeypatch.delenv("MONOLITH_SOURCE_TIER_V1", raising=False)
    assert _source_tier_marker(_rec("generation")) == ""


def test_marker_never_contains_raw_tier_name(monkeypatch):
    monkeypatch.setenv("MONOLITH_SOURCE_TIER_V1", "1")
    m = _source_tier_marker(_rec("generation"))
    assert "generation" not in m and "source_tier" not in m


def test_get_render_shows_unverified_not_raw_tier(monkeypatch):
    monkeypatch.setenv("MONOLITH_SOURCE_TIER_V1", "1")
    monkeypatch.setattr(ms_service, "get", lambda nsid: _rec("generation"))
    out = sr.execute_monosearch({"verb": "get", "id": "turn:abc", "limit": 10}, None)
    assert "[unverified]" in out          # the constraint reaches the model
    assert "source_tier" not in out       # raw tier name never does
    assert "generation" not in out


def test_get_render_no_marker_for_tool_tier(monkeypatch):
    monkeypatch.setenv("MONOLITH_SOURCE_TIER_V1", "1")
    monkeypatch.setattr(ms_service, "get", lambda nsid: _rec("tool"))
    out = sr.execute_monosearch({"verb": "get", "id": "turn:abc", "limit": 10}, None)
    assert "[unverified]" not in out


def test_get_render_no_marker_when_flag_off(monkeypatch):
    monkeypatch.delenv("MONOLITH_SOURCE_TIER_V1", raising=False)
    monkeypatch.setattr(ms_service, "get", lambda nsid: _rec("generation"))
    out = sr.execute_monosearch({"verb": "get", "id": "turn:abc", "limit": 10}, None)
    assert "[unverified]" not in out
