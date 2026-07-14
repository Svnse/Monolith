"""Day-1 liveness — the whole point of slice 1a.

Over real fault_traces, `failing` must surface the most-recurring fault_kind with
the correct count. This is the model reading its own faults for the first time.

The autouse conftest fixtures isolate BOTH the turn_trace DB and the salience
ledger per test (and set MONOLITH_TURN_TRACE_V1=1), so this test seeds its own
faults into an empty, isolated store — the same code path runs against the live
DB in production.
"""
from core import fault_response as fr
from core.monosearch import salience
from core.monosearch.adapters.faults import FaultAdapter


def test_failing_surfaces_top_recurring_fault_kind():
    # Seed faults: think_leak x3 with DIFFERENT evidence each turn (production
    # reality — the leaked snippet varies). Kind-level recurrence must still
    # aggregate these to one key with count 3. tool_no_fire x1 (a lone kind).
    for i in range(3):
        assert fr.emit_fault(
            turn_id=f"t{i}", fault_kind="think_leak",
            detector_name="detect_think_leak", evidence=f"unbalanced <think> snippet {i}",
        ) > 0
    assert fr.emit_fault(
        turn_id="tz", fault_kind="tool_no_fire",
        detector_name="detect_tool_no_fire", evidence="said it would, did not",
    ) > 0

    # Rebuild the ledger from the live fault adapter, then ask `failing`.
    n = salience.rebuild([FaultAdapter()], now=2_000_000.0)
    assert n == 4  # four faults observed

    top = salience.failing(now=2_000_000.0, limit=5)
    assert top, "failing returned nothing — the model still can't see its faults"
    assert top[0]["recurrence_key"] == "think_leak"  # kind-level
    assert top[0]["count"] == 3  # the recurring kind wins, despite 3 different evidences
    assert top[0]["source"] == "fault_traces"
    assert top[0]["provenance"] == "self"


def test_recurring_includes_the_fault_keys():
    for _ in range(2):
        fr.emit_fault(turn_id="t", fault_kind="regen_mismatch",
                      detector_name="detect_regen_mismatch", evidence="mismatch")
    salience.rebuild([FaultAdapter()], now=2_000_000.0)
    keys = [r["recurrence_key"] for r in salience.recurring(now=2_000_000.0, limit=10)]
    assert "regen_mismatch" in keys
