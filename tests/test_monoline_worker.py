"""MonolineEngine worker — the contract-conformant event protocol (status/result/error) for
running a Monoline flow INSIDE the spawned engine worker process (Kernel Contract v2 §9 / D10).
Pure: no subprocess, no real Monoline (the child-side run is lazy + patched here).
See docs/reports/GENESIS_CARD_BUILD_LOG.md."""
from __future__ import annotations

import engine._workers.monoline_worker as mw


class _FakeQ:
    def __init__(self, items=None):
        self.items = list(items or [])
        self.put_items: list = []

    def get(self):
        if not self.items:
            raise EOFError  # parent gone / queue closed
        return self.items.pop(0)

    def put(self, x):
        self.put_items.append(x)


def test_run_flow_emits_running_then_result_then_ready(monkeypatch):
    monkeypatch.setattr(mw, "_run_monoline", lambda config: {"output": "HELLO", "error": ""})
    events: list = []
    mw.run_flow({"world": {"id": "x"}}, events.append)
    assert events[0] == {"event": "status", "status": "running"}
    assert any(e.get("event") == "result" and e.get("output") == "HELLO" for e in events)
    assert events[-1] == {"event": "status", "status": "ready"}


def test_run_flow_emits_error_on_exception(monkeypatch):
    def boom(config):
        raise RuntimeError("flow blew up")
    monkeypatch.setattr(mw, "_run_monoline", boom)
    events: list = []
    mw.run_flow({}, events.append)
    assert events[0] == {"event": "status", "status": "running"}
    assert any(e.get("event") == "error" and "flow blew up" in e.get("message", "") for e in events)
    assert events[-1] == {"event": "status", "status": "error"}


def test_main_runs_generate_then_exits_on_shutdown(monkeypatch):
    ran: list = []
    monkeypatch.setattr(mw, "run_flow", lambda config, emit: ran.append(config))
    to_q = _FakeQ([{"op": "generate", "config": {"n": 1}}, {"op": "shutdown"}])
    mw.main(to_q, _FakeQ())
    assert ran == [{"n": 1}]  # generate ran once, then shutdown broke the loop


def test_main_exits_cleanly_when_queue_dies(monkeypatch):
    # parent gone -> get() raises -> loop must end, never hang or propagate
    monkeypatch.setattr(mw, "run_flow", lambda config, emit: None)
    mw.main(_FakeQ([]), _FakeQ())  # no exception == pass
