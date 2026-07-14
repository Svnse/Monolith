from core import fault_response as fr


def test_fault_record_has_severity_field():
    rec = fr.FaultRecord(id=1, turn_id="t", fault_kind="think_leak", detected_at="x", detector_name="d")
    assert rec.severity is None  # default, back-compatible


def test_read_recent_accepts_since_and_keyword_kwargs():
    # signature smoke test — must not raise on the new kwargs (empty DB returns [])
    assert fr.read_recent(limit=5, since=None, keyword=None) == []
    assert fr.read_by_kind("think_leak", limit=5, since=None, keyword=None) == []


def test_read_one_and_read_since_id_exist():
    assert fr.read_one(999999) is None
    assert fr.read_since_id(0, limit=5) == []
