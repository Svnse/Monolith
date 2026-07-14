"""Integration test: a REAL PageChat, constructed headlessly (no SimpleNamespace stub), driven
through the actual _dispatch_generation guard. Every other chat test stubs around PageChat -- this
one validates the surface they stub. This is what would have caught the unwired Workshop seam.
"""
from __future__ import annotations

import json
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
    p._test_state = state  # stash for tests
    return p


def _seed(d, wid):
    (d / f"{wid}.monoline").write_text(json.dumps({
        "id": wid, "name": wid, "description": "",
        "blocks": [], "connections": [], "composites": []}), encoding="utf-8")


def test_pagechat_constructs_with_workflow_wiring(page):
    # The foundation: a real PageChat builds + has the SP1/SP2 wiring, and its registry is
    # world_state-bound (active_id reads the live store, defaulting to "" = Genesis).
    assert hasattr(page, "_workflow_registry")
    assert hasattr(page, "_dispatch_generation")
    assert hasattr(page, "_dispatch_monoline_run")
    assert hasattr(page, "_handle_workshop_command")
    assert page._workflow_registry.active_id() == ""


def test_real_guard_diverts_when_flow_active(page, tmp_path, monkeypatch):
    # On the REAL object: an active monoline flow + a send-sourced dispatch diverts to the
    # monoline lane (NOT Genesis). This drives the actual guard, not a stub.
    from core.workflow_registry import WorkflowRegistry
    state = page._test_state
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    reg.bind_world_state(state.world_state)
    monkeypatch.setattr(page, "_workflow_registry", reg)
    _seed(tmp_path, "alpha")
    state.world_state.set_active_workflow("alpha")
    diverted = []
    monkeypatch.setattr(page, "_dispatch_monoline_run",
                        lambda wf, payload, **k: diverted.append(wf.id))
    try:
        page._dispatch_generation({"prompt": "hi"}, source="send:hi")
        assert diverted == ["alpha"]
    finally:
        state.world_state.set_active_workflow(None)


def test_real_guard_genesis_when_no_flow(page, monkeypatch):
    # No active flow -> the guard falls through to the Genesis setup (does NOT divert). sig_generate
    # is disconnected so no real generation fires; we assert the Genesis stream setup ran instead.
    state = page._test_state
    state.world_state.set_active_workflow(None)
    diverted, started = [], []
    monkeypatch.setattr(page, "_dispatch_monoline_run", lambda *a, **k: diverted.append(1))
    monkeypatch.setattr(page, "_start_assistant_stream", lambda: started.append(1))
    monkeypatch.setattr(page, "_set_send_button_state", lambda **k: None)
    try:
        page.sig_generate.disconnect()
    except Exception:
        pass
    page._dispatch_generation({"prompt": "hi"}, source="send:hi")
    assert diverted == []      # not diverted -> Genesis path
    assert started == [1]      # Genesis stream setup ran
