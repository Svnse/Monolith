"""Stop/cancel must suppress AUTONOMOUS loop continuation (parse-retry, nudges,
followup-retry) — the four sites that re-dispatch a generation through
_dispatch_generation without the cancel guard the followup path already has.

Root cause: a Stop-interrupted generation often leaves a malformed <tool_call>;
process_response reports a parse error; the tool_parse_retry branch dispatches a
NEW generation, never reaching the downstream _tool_cancel_requested check.
"""
from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")


@pytest.fixture(scope="module")
def page():
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from core.state import AppState
    from core.world_state import WorldStateStore
    from bootstrap import init_kernel
    from ui.bridge import UIBridge
    from ui.pages.chat import PageChat

    state = AppState()
    state.world_state = WorldStateStore()
    guard, _dock, bridge, _vision = init_kernel(state)
    ui_bridge = UIBridge()
    p = PageChat(state, ui_bridge, bridge=bridge, guard=guard)
    p._test_state = state
    return p


def test_is_autonomous_continuation_classification():
    from ui.pages.chat import _is_autonomous_continuation
    # Engine-issued loop continuations → suppressible by a Stop.
    for s in ("tool_parse_retry", "incomplete_action_nudge",
              "non_convergence_nudge", "tool_followup_retry", "tool_followup"):
        assert _is_autonomous_continuation(s) is True, s
    # User-initiated dispatches → never suppressed.
    for s in ("send:hi", "edit:x", "regen", "agent:foo", "at_mention:monolith",
              "update", "chat", ""):
        assert _is_autonomous_continuation(s) is False, s


def _arm(page, monkeypatch):
    """Disarm the real generation side effects; record whether one would start."""
    state = page._test_state
    state.world_state.set_active_workflow(None)
    started = []
    monkeypatch.setattr(page, "_start_assistant_stream", lambda: started.append(1))
    monkeypatch.setattr(page, "_set_send_button_state", lambda **k: None)
    monkeypatch.setattr(page, "_dispatch_monoline_run", lambda *a, **k: started.append("mono"))
    try:
        page.sig_generate.disconnect()
    except Exception:
        pass
    return started


def test_stop_suppresses_autonomous_parse_retry(page, monkeypatch):
    started = _arm(page, monkeypatch)
    page._tool_cancel_requested = True
    page._dispatch_generation({"prompt": "retry"}, source="tool_parse_retry")
    assert started == []                            # no new generation fired
    assert page._tool_cancel_requested is False     # flag consumed by the guard


def test_stop_suppresses_autonomous_nudge(page, monkeypatch):
    started = _arm(page, monkeypatch)
    page._tool_cancel_requested = True
    page._dispatch_generation({"prompt": "nudge"}, source="non_convergence_nudge")
    assert started == []


def test_user_send_clears_stale_stop_and_proceeds(page, monkeypatch):
    started = _arm(page, monkeypatch)
    page._tool_cancel_requested = True
    page._dispatch_generation({"prompt": "hi"}, source="send:hi")
    assert started == [1]                           # user send is NOT suppressed
    assert page._tool_cancel_requested is False     # stale stop cleared for the new turn
