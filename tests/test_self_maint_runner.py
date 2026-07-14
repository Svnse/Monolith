"""Tests for the self-maintenance daemon LOOP (engine/self_maint_runner.py).

The dangerous integration: an autonomous loop that wakes the model unattended. The
Qt/model seams (_load_cfg, _generate, _parse_calls, _dispatch, _ensure_ctx) are
overridable so the whole loop is driven deterministically here — no PySide6, no live
model. Asserts every audit-mandated safety property: flag-off no-op, busy/lock/leash
gating, hard tool confinement (the execution allow-list backstop), context built ONLY
via build_wake_context, guaranteed frame + ledger, fault-halt, asymmetric STOP.
Runs without pytest.
"""
from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import threading
import time

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO)

from core import self_maint_daemon as daemon  # noqa: E402
from core import self_maint_leash as leash  # noqa: E402
from core import review_loop  # noqa: E402
from core import turn_trace  # noqa: E402
from engine.self_maint_runner import SelfMaintRunner  # noqa: E402

# Isolate the daemon's wake ledger from the real CONFIG_DIR for the whole test run.
# Several tests exercise the REAL log_wake (skip rows via _tick, fault/halt rows via
# _run_loop) and would otherwise append test noise to the live
# %APPDATA%/Monolith/config/self_maint_trigger.ledger.jsonl. Re-applied after each
# importlib.reload(daemon) below (reload re-runs the module and resets _WAKE_LEDGER).
_TEST_LEDGER = pathlib.Path(tempfile.gettempdir()) / "test_self_maint_trigger.ledger.jsonl"
daemon._WAKE_LEDGER = _TEST_LEDGER


# ── a fake envelope + a fully-stubbed runner (no Qt, no model) ──────────────
class _Env:
    def __init__(self, call_id="", ok=True, text=""):
        self.call_id, self.ok, self.text = call_id, ok, text


class FakeRunner(SelfMaintRunner):
    """Overrides the Qt/model seams so the loop runs in-process."""
    def __init__(self, *, gen_output=("", ""), parsed_calls=None, **kw):
        super().__init__(**kw)
        self._scripted_gen = gen_output
        self._scripted_calls = parsed_calls if parsed_calls is not None else []
        self.dispatched = []        # calls that actually reached _dispatch
        self.gen_calls = 0
        self._ctx = object()        # sentinel; _ensure_ctx is a no-op below

    def _ensure_ctx(self):
        return  # never build the real (Qt) context in tests

    def _load_cfg(self):
        return {}

    def _generate(self, cfg, messages):
        self.gen_calls += 1
        return self._scripted_gen

    def _parse_calls(self, raw):
        return list(self._scripted_calls)

    def _dispatch(self, call):
        self.dispatched.append(dict(call))
        return _Env(call_id=call.get("id", ""), ok=True, text="ok")

    def _finalize(self, raw, turn_id, tool_names):
        return  # ExpeditionRunner-proven; not the subject of these tests

    def _build_system_prompt(self):
        return "SYS"  # keep wake tests hermetic; the real prompt is tested separately


def _reset_flags():
    for k in ("MONOLITH_SELF_MAINT_TRIGGER_V1", "MONOLITH_SELF_MAINT_WAKE_INTERVAL_S"):
        os.environ.pop(k, None)


_SAMPLE_ITEMS = [
    {"id": "acu:25", "kind": "acu", "subkind": "bug", "effective_severity": 5,
     "summary": "ranker drops ties", "reason": "recurring", "age_days": 4.0},
    {"id": "pin:6", "kind": "continuity", "subkind": "pending", "effective_severity": 4,
     "summary": "turn-trace spec impl", "reason": "stale", "age_days": 9.0},
]


# ── lifecycle / flag gate ───────────────────────────────────────────────────
def test_flag_off_never_starts():
    _reset_flags()
    r = FakeRunner()
    started = r.start()
    assert started is False
    assert r._thread is None  # byte-identical to no daemon
    assert r.status in ("disabled", "idle")


def test_flag_on_starts_then_stops():
    _reset_flags()
    os.environ["MONOLITH_SELF_MAINT_TRIGGER_V1"] = "1"
    os.environ["MONOLITH_SELF_MAINT_WAKE_INTERVAL_S"] = "3600"  # long; we stop before it fires
    try:
        r = FakeRunner()
        assert r.start() is True
        r.stop(timeout=2.0)
        assert r._thread is None or not r._thread.is_alive()
        assert r.status == "stopped"
        assert r.gen_calls == 0  # waited on the interval; stopped before any wake
    finally:
        _reset_flags()


# ── tick gating ─────────────────────────────────────────────────────────────
def test_tick_skips_when_external_busy():
    _reset_flags()
    consumed = []
    daemon.try_wake = lambda now=None: (consumed.append(1) or {"ok": True, "count": 1})
    try:
        r = FakeRunner(is_busy=lambda: True)
        r._run_one_wake = lambda turn_id: (_ for _ in ()).throw(AssertionError("woke while busy"))
        r._tick()
        assert consumed == []  # no wake-credit spent when externally busy
    finally:
        import importlib
        importlib.reload(daemon)
        daemon._WAKE_LEDGER = _TEST_LEDGER  # re-isolate after reload reset it


def test_tick_skips_when_lock_held():
    _reset_flags()
    from core.generation import generation_lock
    woke = []
    r = FakeRunner()
    r._run_one_wake = lambda turn_id: woke.append(turn_id)
    assert generation_lock.acquire(blocking=False) is True
    try:
        r._tick()  # lock is held by us → daemon must skip, never block
        assert woke == []
    finally:
        generation_lock.release()


def test_tick_gated_by_daily_leash():
    _reset_flags()
    p = pathlib.Path(tempfile.gettempdir()) / "smr_leash.json"
    if p.exists():
        p.unlink()
    leash._STATE_PATH = p
    os.environ["MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY"] = "0"  # cap 0 → always denied
    try:
        woke = []
        r = FakeRunner()
        r._run_one_wake = lambda turn_id: woke.append(turn_id)
        r._tick()
        assert woke == []  # leash denied → no generation
    finally:
        os.environ.pop("MONOLITH_SELF_MAINT_MAX_WAKES_PER_DAY", None)


# ── the wake directive ──────────────────────────────────────────────────────
def test_directive_contains_review_queue_and_only_safe_actions():
    r = FakeRunner()
    d = r._wake_directive(_SAMPLE_ITEMS, {"unresolved_count": 2})
    assert "[REVIEW QUEUE]" in d
    assert "acu:25" in d  # the model can reference a real item id
    assert "snooze" in d and "escalate" in d
    low = d.lower()
    assert "cannot resolve or dismiss" in low or "you cannot resolve" in low
    # must NOT instruct the model that resolve/dismiss are available actions
    assert "review_act" in d


def test_directive_empty_queue_says_no_action():
    r = FakeRunner()
    d = r._wake_directive([], {"unresolved_count": 0})
    low = d.lower()
    assert "no action" in low or "do nothing" in low or "empty" in low


def test_real_system_prompt_describes_review_act_and_constraint():
    # The real prompt assembly (identity/bearing are best-effort; catalog is static).
    r = SelfMaintRunner()
    sp = r._build_system_prompt()
    assert "review_act" in sp
    low = sp.lower()
    assert "snooze" in low and "escalate" in low
    assert "resolve" in low and "dismiss" in low  # names the forbidden actions as forbidden


# ── HARD tool confinement (execution allow-list backstop) ───────────────────
def test_only_wake_tools_dispatched():
    _reset_flags()
    r = FakeRunner(
        gen_output=("calling tools", ""),
        parsed_calls=[
            {"tool": "write_file", "arguments": {"path": "identity.md", "content": "x"}},
            {"tool": "spawn_subagent", "arguments": {"task": "x"}},
            {"tool": "review_act", "arguments": {"item_id": "acu:25", "action": "snooze"}},
        ],
    )
    review_loop.list_review_items = lambda **k: list(_SAMPLE_ITEMS)
    review_loop.review_summary = lambda **k: {"unresolved_count": 2}
    daemon.try_wake = lambda now=None: {"ok": True, "count": 1}
    daemon.log_wake = lambda row: None
    turn_trace.record_frame = lambda rec: None
    try:
        r._run_one_wake("maint_000001")
        names = [c.get("tool") for c in r.dispatched]
        assert names == ["review_act"]  # the two mutators were filtered, never dispatched
    finally:
        import importlib
        importlib.reload(review_loop)
        importlib.reload(daemon)
        daemon._WAKE_LEDGER = _TEST_LEDGER  # re-isolate after reload reset it
        importlib.reload(turn_trace)


# ── guaranteed observability ────────────────────────────────────────────────
def test_wake_persists_raw_artifact():
    # E's dark-system contract requires a durable raw-I/O artifact. The model's raw output
    # (incl. any <bearing_update> ATTEMPT, which is now applied nowhere) must land in the
    # wake ledger — otherwise an observe-first wake leaves no record of what it said. (re-audit)
    _reset_flags()
    rows = []
    review_loop.list_review_items = lambda **k: list(_SAMPLE_ITEMS)
    review_loop.review_summary = lambda **k: {"unresolved_count": 2}
    daemon.log_wake = lambda row: rows.append(row)
    turn_trace.record_frame = lambda rec: None
    try:
        r = FakeRunner(gen_output=("I would snooze acu:25 <bearing_update>x</bearing_update>", "reasoning"))
        r._run_one_wake("maint_000009")
        assert rows, "a wake must write a ledger row"
        assert "raw" in rows[0] and "snooze acu:25" in rows[0]["raw"]  # the model's words are durable
    finally:
        import importlib
        importlib.reload(review_loop)
        importlib.reload(daemon)
        daemon._WAKE_LEDGER = _TEST_LEDGER  # re-isolate after reload reset it
        importlib.reload(turn_trace)


def test_wake_emits_frame_and_ledger():
    _reset_flags()
    frames, ledger_rows = [], []
    review_loop.list_review_items = lambda **k: list(_SAMPLE_ITEMS)
    review_loop.review_summary = lambda **k: {"unresolved_count": 2}
    daemon.log_wake = lambda row: ledger_rows.append(row)
    turn_trace.record_frame = lambda rec: frames.append(rec)
    try:
        r = FakeRunner(gen_output=("", ""))  # model emits nothing → still must observe
        r._run_one_wake("maint_000007")
        assert len(frames) == 1            # record_frame is guaranteed, not best-effort
        assert len(ledger_rows) == 1
        assert ledger_rows[0]["turn_id"] == "maint_000007"
    finally:
        import importlib
        importlib.reload(review_loop)
        importlib.reload(daemon)
        daemon._WAKE_LEDGER = _TEST_LEDGER  # re-isolate after reload reset it
        importlib.reload(turn_trace)


# ── fault-halt ──────────────────────────────────────────────────────────────
def test_fault_streak_halts_at_3():
    _reset_flags()
    r = FakeRunner()
    r._wake_interval_s = 0  # test-only seam: no sleep (env<=0 now fails closed to default)
    r._tick = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
    r._run_loop()  # synchronous; must return after 3 faults
    assert r.status == "halted"
    assert r._fault_streak >= 3


def test_faults_and_halt_are_logged():
    # Every fault AND the halt must write a durable row to the ungated wake ledger —
    # not just the in-memory streak / a turn_trace-gated emit_fault. (audit #4)
    _reset_flags()
    rows = []
    daemon.log_wake = lambda row: rows.append(row)
    try:
        r = FakeRunner()
        r._wake_interval_s = 0
        r._tick = lambda: (_ for _ in ()).throw(RuntimeError("boom"))
        r._run_loop()
        assert sum(1 for row in rows if "fault" in row) >= 3   # each sub-threshold fault recorded
        assert any("halted" in row for row in rows)            # the halt recorded
    finally:
        import importlib
        importlib.reload(daemon)
        daemon._WAKE_LEDGER = _TEST_LEDGER  # re-isolate after reload reset it


def test_finalize_writes_no_substrate():
    # A maintenance wake is ephemeral triage: _finalize must NOT run the turn finalizer
    # (which would commit <bearing_update>/<frame> to the bearing store under apply-off,
    # breaching observe-first). (audit #1)
    import core.chat_finalize as cf
    called = []
    orig = cf.finalize_assistant_turn
    cf.finalize_assistant_turn = lambda *a, **k: called.append(1)
    try:
        r = SelfMaintRunner()
        r._finalize("<bearing_update>steered</bearing_update>", "maint_000001", [])
        assert called == []  # the wake never reaches the substrate-writing finalizer
    finally:
        cf.finalize_assistant_turn = orig


# ── asymmetric STOP ─────────────────────────────────────────────────────────
def test_stop_prevents_next_wake():
    _reset_flags()
    r = FakeRunner()
    r._stop = True  # already requested
    woke = []
    r._run_one_wake = lambda turn_id: woke.append(turn_id)
    r._tick()
    assert woke == []  # a stop in flight prevents the next wake


# ── the must-preserve invariant: context built ONLY via build_wake_context ──
def test_context_built_via_build_wake_context():
    _reset_flags()
    calls = []
    real = daemon.build_wake_context
    daemon.build_wake_context = lambda should_cancel: (calls.append(should_cancel) or "FAKE_CTX")
    try:
        r = SelfMaintRunner()  # the REAL runner, not the fake
        r._ensure_ctx()
        assert calls, "runner must obtain its context via build_wake_context"
        assert r._ctx == "FAKE_CTX"
    finally:
        daemon.build_wake_context = real


# ── wiring (the activation step) ────────────────────────────────────────────
def test_snapshot_includes_status_apply_and_recent():
    # The companion panel renders snapshot() — it must expose the daemon status, the
    # trigger/apply flag states (observe-first vs applying), and the recent wake rows.
    _reset_flags()
    os.environ.pop("MONOLITH_SELF_MAINT_V1", None)
    r = FakeRunner()
    snap = r.snapshot()
    for key in ("status", "wake", "interval_s", "trigger_on", "apply_on", "recent", "activity"):
        assert key in snap, key
    assert snap["apply_on"] is False        # apply flag default off => observe-first
    assert isinstance(snap["recent"], list)


def test_dispatch_drops_key_split_evasion():
    # A wake that splits naming keys so the filter sees review_act but the dispatcher would
    # resolve self_maint/set_interval must be dropped — the backstop is genuine. (audit)
    r = FakeRunner()
    calls = [{"name": "review_act", "skill": "self_maint", "op": "set_interval", "seconds": 1}]
    envelopes, ran = r._dispatch_calls(calls, "maint_x")
    assert ran == [] and envelopes == []
    assert r.dispatched == []  # never reached _dispatch


def test_start_force_bypasses_trigger_flag():
    # The control skill starts the daemon at runtime regardless of the launcher flag;
    # plain start() still refuses when the flag is off (boot path).
    _reset_flags()
    os.environ["MONOLITH_SELF_MAINT_WAKE_INTERVAL_S"] = "3600"
    try:
        r = FakeRunner()
        assert r.start() is False                 # flag off => boot path refuses
        assert r.status == "disabled"
        assert r.start(force=True) is True         # skill path starts it
        r.stop(timeout=2.0)
        assert r.status == "stopped"
    finally:
        _reset_flags()


def test_set_interval_clamps_to_floor_and_updates():
    _reset_flags()
    r = FakeRunner()
    r.set_interval(60)
    assert r._wake_interval_s == 60               # honored (every minute)
    r.set_interval(2)
    assert r._wake_interval_s == SelfMaintRunner._MIN_INTERVAL_S  # clamped off the spin floor
    r.set_interval("bad")
    assert r._wake_interval_s == SelfMaintRunner._MIN_INTERVAL_S  # garbage => floor, never 0


def test_engine_is_busy_polls_world_state():
    import engine.self_maint_runner as smr

    class WS:
        def __init__(self, status):
            self._s = status

        def snapshot(self):
            return {"engines": {"main": {"status": self._s}}}

    class Bad:
        def snapshot(self):
            raise RuntimeError("boom")

    assert smr.engine_is_busy(None) is False           # headless/test convention
    assert smr.engine_is_busy(WS("running")) is True   # a live turn
    assert smr.engine_is_busy(WS("generating")) is True
    assert smr.engine_is_busy(WS("streaming")) is True
    assert smr.engine_is_busy(WS("READY")) is False
    assert smr.engine_is_busy(Bad()) is False          # never raises => fail-safe (free)


def test_maybe_start_flag_off_returns_false():
    import engine.self_maint_runner as smr
    _reset_flags()
    fake = FakeRunner()
    smr._runner_singleton = fake
    try:
        assert smr.maybe_start_self_maint(None) is False
        assert fake.status == "idle"  # the daemon was never started
    finally:
        smr._runner_singleton = None


def test_maybe_start_flag_on_starts_with_is_busy_bound():
    import engine.self_maint_runner as smr
    _reset_flags()
    os.environ["MONOLITH_SELF_MAINT_WAKE_INTERVAL_S"] = "3600"  # long; we stop before a wake
    os.environ["MONOLITH_SELF_MAINT_TRIGGER_V1"] = "1"
    fake = FakeRunner()
    smr._runner_singleton = fake
    busy = {"v": False}

    class WS:
        def snapshot(self):
            return {"engines": {"main": {"status": "running" if busy["v"] else "ready"}}}

    try:
        assert smr.maybe_start_self_maint(WS()) is True
        assert fake._is_busy is not None          # the live-turn guard is bound (audit #6)
        assert fake._is_busy() is False           # idle engine => wake allowed
        busy["v"] = True
        assert fake._is_busy() is True            # live turn streaming => wake would skip
        fake.stop(timeout=2.0)
    finally:
        smr._runner_singleton = None
        _reset_flags()


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  PASS {fn.__name__}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  FAIL {fn.__name__}: {type(e).__name__}: {e}")
    _reset_flags()
    print(f"\n{len(fns) - failed}/{len(fns)} passed")
    sys.exit(1 if failed else 0)
