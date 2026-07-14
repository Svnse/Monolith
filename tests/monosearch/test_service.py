"""The consumer-facing service. `failing`/`recurring` POPULATE the ledger (rebuild
from the registered adapters) BEFORE reading — the trigger that makes the selector
live instead of a dark store. Autouse conftest isolates turn_trace + salience."""
from core import fault_response as fr
from core.monosearch import registry, service
from core.monosearch.adapters.faults import FaultAdapter


def test_failing_populates_from_registry_then_reads():
    registry.clear()
    registry.register(FaultAdapter())
    # Seed 4 tool_no_fire faults with DIFFERENT evidence (production reality);
    # kind-level recurrence must aggregate them to one key, count 4.
    for i in range(4):
        assert fr.emit_fault(turn_id=f"t{i}", fault_kind="tool_no_fire",
                             detector_name="detect_tool_no_fire",
                             evidence=f"stated intent {i}, no tool_call emitted") > 0
    rows = service.failing(limit=5)
    assert rows, "service.failing returned [] — population is not wired (dark store)"
    assert rows[0]["recurrence_key"] == "tool_no_fire"
    assert rows[0]["count"] == 4


def test_failing_is_empty_when_no_faults():
    registry.clear()
    registry.register(FaultAdapter())
    assert service.failing(limit=5) == []


def test_recurring_populates_too():
    registry.clear()
    registry.register(FaultAdapter())
    for _ in range(2):
        fr.emit_fault(turn_id="t", fault_kind="think_leak",
                      detector_name="detect_think_leak", evidence="<think> leaked")
    keys = [r["recurrence_key"] for r in service.recurring(limit=10)]
    assert "think_leak" in keys


def test_failing_survives_a_raising_sibling_adapter():
    """Production registers BOTH adapters. If a sibling (e.g. canonical_log) raises
    on read, rebuild must isolate it (like router.search) and still return the fault
    data — failing doesn't even use canonical_log."""
    from core.monosearch.adapter import SourceAdapter
    from core.monosearch.record import EvidenceTier

    class _Boom(SourceAdapter):
        name = "boom"
        evidence_tier = EvidenceTier.DERIVED
        def search(self, q, f, l):
            raise RuntimeError("boom")
        def get(self, i):
            return None
        def list(self, f, l):
            raise RuntimeError("boom on list")

    registry.clear()
    registry.register(FaultAdapter())
    registry.register(_Boom())
    for i in range(3):
        assert fr.emit_fault(turn_id=f"t{i}", fault_kind="regen_mismatch",
                             detector_name="detect_regen_mismatch", evidence=f"mismatch {i}") > 0
    rows = service.failing(limit=5)
    assert rows, "a raising sibling adapter killed failing — rebuild lacks per-adapter isolation"
    assert rows[0]["recurrence_key"] == "regen_mismatch"
    assert rows[0]["count"] == 3


def test_production_shape_both_adapters_then_failing(tmp_path, monkeypatch):
    """The real boot shape: init_monosearch registers Fault + CanonicalLog. With an
    empty (isolated) canonical_log present, failing must still return fault data."""
    import threading
    from core import db_connect as _dbc
    db_path = tmp_path / "test_acatalepsy.sqlite3"
    monkeypatch.setattr(_dbc, "DB_PATH", db_path, raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    conn = _dbc.connect_acatalepsy(role="migration")
    conn.executescript(
        "CREATE TABLE canonical_log (event_id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL NOT NULL,"
        " kind TEXT NOT NULL, session_id TEXT, acu_id INTEGER, payload TEXT);"
    )
    conn.commit()
    conn.close()
    from core.acatalepsy import canonical_log as cl
    monkeypatch.setattr(cl, "_tl", threading.local(), raising=True)

    from core.monosearch.bootstrap import init_monosearch
    registry.clear()
    init_monosearch()  # registers FaultAdapter + CanonicalLogAdapter (production shape)
    for i in range(2):
        assert fr.emit_fault(turn_id=f"t{i}", fault_kind="tool_no_fire",
                             detector_name="detect_tool_no_fire", evidence=f"no call {i}") > 0
    rows = service.failing(limit=5)
    assert any(r["recurrence_key"] == "tool_no_fire" for r in rows)


def test_recurring_excludes_bearing_successes(tmp_path, monkeypatch):
    """`recurring` means 'problems that recur'. Bearing SUCCESSES (applied/cleared)
    must NOT appear; problem kinds (rejected) must — proven against real audit rows."""
    from addons.system.bearing import audit
    from core.monosearch.adapters.bearing import BearingAdapter
    monkeypatch.setattr(audit, "_AUDIT_PATH", tmp_path / "bearing.audit.jsonl")
    for i in range(2):
        audit.append("rejected", turn_id=f"r{i}", failed_rules=["D1"])
    for i in range(3):
        audit.append("applied", turn_id=f"a{i}", slots_changed=["x"])
    registry.clear()
    registry.register(BearingAdapter())
    keys = [r["recurrence_key"] for r in service.recurring(limit=20)]
    assert "rejected" in keys
    assert "applied" not in keys  # success kind must not dilute 'what keeps coming up'


def test_failing_does_not_rescan_unrelated_stores(monkeypatch):
    """Scoped refresh: failing must refresh ONLY fault_traces, never call .list on
    a sibling adapter (so its latency/health isn't coupled to 7 unrelated stores)."""
    from core.monosearch.adapter import SourceAdapter
    from core.monosearch.record import EvidenceTier
    calls = {"n": 0}

    class _Spy(SourceAdapter):
        name = "spy"
        evidence_tier = EvidenceTier.DERIVED
        def search(self, q, f, l): return []
        def get(self, i): return None
        def list(self, f, l):
            calls["n"] += 1
            return []

    registry.clear()
    registry.register(FaultAdapter())
    registry.register(_Spy())
    fr.emit_fault(turn_id="t", fault_kind="think_leak", detector_name="d", evidence="x")
    service.failing(limit=5)
    assert calls["n"] == 0, "failing rescanned an unrelated adapter — refresh is not scoped"
