"""End-to-end wiring smoke for the drift observe hook in bearing_interceptor.

Confirms: (1) flag OFF -> block still injected, NO ledger write (byte-identical);
(2) flag ON -> block still injected AND one drift row written for the outer turn.
Stubs the render/staleness so it doesn't depend on live bearing.json shape.
"""
from __future__ import annotations

import json
import os
import pathlib
import sys
import tempfile
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

os.environ["MONOLITH_BEARING_V1"] = "1"
from addons.system.bearing import compiler, store, drift_observe  # noqa: E402

FRAME = "database migration runbook production incident remediation"
store.get_bearing = lambda: SimpleNamespace(current_frame=FRAME)
store.get_pending_rejection = lambda: None
compiler.format_bearing_block = lambda *a, **k: "[BEARING] stub"
compiler._apply_staleness = lambda block, *a, **k: block
compiler._resolve_current_turn_n = lambda c: None
compiler._resolve_plan_view = lambda c: None

led = pathlib.Path(tempfile.gettempdir()) / "drift_wire_smoke.jsonl"
if led.exists():
    led.unlink()
drift_observe._LEDGER = led

msgs = [{"role": "user", "content": "an unrelated question about feline behavior in cats"}]

# (1) flag OFF
os.environ.pop("MONOLITH_FRAME_DRIFT_V1", None)
r1 = compiler.bearing_interceptor(list(msgs), {"_turn_id": "t1"})
assert r1 is not None and len(r1) == len(msgs) + 1, "flag-off: block must still inject"
assert not led.exists(), "flag-off: NO ledger write"

# (2) flag ON
os.environ["MONOLITH_FRAME_DRIFT_V1"] = "1"
r2 = compiler.bearing_interceptor(list(msgs), {"_turn_id": "t2"})
assert r2 is not None and len(r2) == len(msgs) + 1, "flag-on: block must still inject"
assert led.exists(), "flag-on: ledger must be written"
rows = [json.loads(ln) for ln in led.read_text(encoding="utf-8").splitlines() if ln.strip()]
assert len(rows) == 1 and rows[0]["turn_id"] == "t2", rows
assert rows[0]["overlap"] < 0.18, rows  # frame vs unrelated ask -> low overlap

os.environ.pop("MONOLITH_FRAME_DRIFT_V1", None)
print("WIRING SMOKE OK ->", rows[0])
