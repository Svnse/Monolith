from unittest.mock import patch
from core.acatalepsy import canonical_log as cl
from core.monosearch.adapters.canonical_log import CanonicalLogAdapter
from core.monosearch.record import EvidenceTier, Provenance

_USER = cl.Event(event_id=10, ts=1717000000.0, kind="user_message", session_id="ui:x", acu_id=None, payload={"text": "hello"})
_OP = cl.Event(event_id=11, ts=1717000001.0, kind="observer_fired", session_id="ui:x", acu_id=None, payload={"n": 3})


def test_user_message_is_user_provenance_and_not_recurrence_eligible():
    a = CanonicalLogAdapter()
    r = a._to_record(_USER)
    assert r.namespaced_id == "clog:10"
    assert r.provenance is Provenance.USER
    # canonical_log is a search/lookup source — messages do NOT feed `recurring`
    # (a repeated message is not a problem-that-recurs).
    assert r.recurrence_key is None
    assert r.evidence_tier == EvidenceTier.LITERAL
    assert r.metadata["session_id"] == "ui:x"


def test_operational_kind_is_self_and_not_salience_eligible():
    a = CanonicalLogAdapter()
    r = a._to_record(_OP)
    assert r.provenance is Provenance.SELF
    assert r.recurrence_key is None  # operational kinds don't recur into salience


def test_get_routes_by_event_id():
    a = CanonicalLogAdapter()
    with patch.object(cl, "read_one", return_value=_USER) as m:
        r = a.get("clog:10")
    m.assert_called_once_with(10)
    assert r.namespaced_id == "clog:10"
    assert a.get("fault:10") is None
