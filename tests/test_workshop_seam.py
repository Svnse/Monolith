"""The Workshop->chat seam: the integration that ties Set-Active to the chat flow guard.

These tests exercise the path that the per-task unit suites stubbed around (they injected
`write=` or faked world_state/registry): the pane's Set-Active button -> the world_state-bound
registry -> world_state -> the real _dispatch_generation guard. This is the wiring that makes
"pick a Monoline flow, it runs in chat" actually do anything in the live app.
"""
from __future__ import annotations

import json
import os
import types
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from core.workflow_registry import WorkflowRegistry
from core.world_state import WorldStateStore


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _seed_monoline(d: Path, wid: str = "alpha") -> None:
    (d / f"{wid}.monoline").write_text(json.dumps({
        "id": wid, "name": wid.title(), "description": "", "modified_at": 1.0,
        "blocks": [], "connections": [], "composites": []}), encoding="utf-8")


def _ws(tmp_path) -> WorldStateStore:
    s = WorldStateStore()
    s._path = Path(tmp_path) / "world_state.json"   # isolate persistence
    return s


def test_set_active_persists_through_real_registry_to_world_state(app, tmp_path):
    # The seam the green unit suites missed: Set-Active with NO injected write= must persist the
    # active-flow id to world_state via the bound registry (else it is a silent no-op).
    from ui.panels.workshop_library import WorkshopLibraryPane
    _seed_monoline(tmp_path)
    ws = _ws(tmp_path)
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    reg.bind_world_state(ws)
    pane = None
    try:
        pane = WorkshopLibraryPane(registry=reg)
        pane.refresh()
        pane._on_set_active("alpha")                     # real registry.set_active path (no write=)
        assert ws.get_active_workflow() == "alpha"       # persisted to the store the guard reads
        assert pane.active_id() == "alpha"
    finally:
        if pane is not None:
            pane.deleteLater()


def test_set_active_then_guard_diverts_end_to_end(app, tmp_path):
    # Full seam: pane Set-Active -> world_state -> the REAL _dispatch_generation guard diverts to
    # the Monoline lane (NOT Genesis). This is the test that would have caught the unwired seam.
    from ui.pages.chat import PageChat
    from ui.panels.workshop_library import WorkshopLibraryPane
    _seed_monoline(tmp_path)
    ws = _ws(tmp_path)
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    reg.bind_world_state(ws)
    pane = WorkshopLibraryPane(registry=reg)
    try:
        pane.refresh()
        pane._on_set_active("alpha")

        stub = types.SimpleNamespace()
        stub.state = types.SimpleNamespace(world_state=ws)   # SAME store the pane wrote to
        stub._workflow_registry = WorkflowRegistry(workflows_dir=tmp_path)
        stub._MONOLINE_ENTRY_SOURCES = PageChat._MONOLINE_ENTRY_SOURCES
        diverted, emitted = [], []
        stub._dispatch_monoline_run = lambda wf, payload, **k: diverted.append(wf.id)
        stub._set_send_button_state = lambda **k: None
        stub._assistant_box = types.SimpleNamespace(start_rewrite_stream=lambda i: None)
        stub._start_assistant_stream = lambda: None
        stub.message_list = types.SimpleNamespace(scrollToBottom=lambda: None)
        stub.sig_debug = types.SimpleNamespace(emit=lambda *a: None)
        stub.sig_generate = types.SimpleNamespace(emit=lambda p: emitted.append(p))
        stub._dispatch_generation = types.MethodType(PageChat._dispatch_generation, stub)

        stub._dispatch_generation({"prompt": "hi"}, source="send:hi")
        assert diverted == ["alpha"]                     # the active Monoline flow ran...
        assert emitted == []                             # ...and Genesis (sig_generate) did NOT
    finally:
        pane.deleteLater()


def test_bind_controller_wires_workshop_library(app, tmp_path):
    # Fix B: bind_controller (called by companion.set_conversation for each panel) routes to the
    # chat host's wire_workshop_library, so Test/Edit/Create/focus signals are live.
    from ui.panels.workshop_library import WorkshopLibraryPane
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    wired = []
    controller = types.SimpleNamespace(wire_workshop_library=lambda p: wired.append(p))
    pane = WorkshopLibraryPane(registry=reg)
    try:
        pane.bind_controller(controller)
        assert wired == [pane]
    finally:
        pane.deleteLater()
