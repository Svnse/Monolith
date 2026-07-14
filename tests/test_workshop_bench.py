from __future__ import annotations

import json
import os
import types
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QMouseEvent
from PySide6.QtCore import QEvent, QPointF, Qt

from core.workflow_registry import WorkflowRegistry


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _seed(d: Path, wid: str):
    (d / f"{wid}.monoline").write_text(json.dumps({
        "id": wid, "name": wid.title(), "description": "a flow",
        "blocks": [], "connections": [], "composites": []}), encoding="utf-8")


def _release(card, x=3, y=3):
    ev = QMouseEvent(QEvent.Type.MouseButtonRelease, QPointF(x, y),
                     Qt.MouseButton.LeftButton, Qt.MouseButton.LeftButton,
                     Qt.KeyboardModifier.NoModifier)
    card.mouseReleaseEvent(ev)


def test_click_row_equips(app, tmp_path, monkeypatch):
    # The headline UX: a release on the row BODY (childAt -> not a button) equips that workflow.
    # childAt is monkeypatched to None so the test is deterministic regardless of unshown geometry.
    from ui.panels.workshop_library import WorkshopLibraryPane
    _seed(Path(tmp_path), "alpha")
    pane = WorkshopLibraryPane(registry=WorkflowRegistry(workflows_dir=tmp_path))
    try:
        pane.refresh()
        assert pane.active_id() == ""        # default == Genesis
        card = pane._cards["alpha"]
        monkeypatch.setattr(card, "childAt", lambda *a: None)  # release on the row body
        _release(card)
        assert pane.active_id() == "alpha"   # equipped by the row click
    finally:
        pane.deleteLater()


def test_section_headers_not_counted_as_cards(app, tmp_path):
    from ui.panels.workshop_library import WorkshopLibraryPane, _WorkflowCard, _CreateCard
    _seed(Path(tmp_path), "alpha")
    pane = WorkshopLibraryPane(registry=WorkflowRegistry(workflows_dir=tmp_path))
    try:
        pane.refresh()
        # genesis row + alpha row + forge = 3; the NATIVE / MY WORKFLOWS headers are NOT counted.
        assert pane.card_count() == 3
        widgets = [pane._cards_layout.itemAt(i).widget()
                   for i in range(pane._cards_layout.count())]
        cards = [w for w in widgets if isinstance(w, (_WorkflowCard, _CreateCard))]
        headers = [w for w in widgets if w is not None and w.objectName() == "workshop_section"]
        assert len(cards) == 3 and len(headers) == 2
    finally:
        pane.deleteLater()


def test_active_row_spins_when_running(app, tmp_path):
    from ui.panels.workshop_library import WorkshopLibraryPane
    _seed(Path(tmp_path), "alpha")
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    reg.bind_world_state(types.SimpleNamespace(
        snapshot=lambda: {"engines": {"workshop": {"status": "RUNNING"}}},
        get_active_workflow=lambda: ""))
    pane = WorkshopLibraryPane(registry=reg)
    try:
        pane.refresh()
        pane._on_set_active("alpha", write=lambda w: None)
        pane._poll_running()
        assert pane._cards["alpha"]._running is True      # the equipped row spins
        assert pane._cards["genesis"]._running is False   # others do not
        reg._ws.snapshot = lambda: {"engines": {"workshop": {"status": "idle"}}}
        pane._poll_running()
        assert pane._cards["alpha"]._running is False
    finally:
        pane.deleteLater()


def test_workflow_rows_do_not_render_per_row_action_buttons(app, tmp_path):
    # Edit/Test are contextual controls for the selected workflow, not repeated on every row.
    from ui.panels.workshop_library import WorkshopLibraryPane
    from PySide6.QtWidgets import QAbstractButton
    _seed(Path(tmp_path), "alpha")
    pane = WorkshopLibraryPane(registry=WorkflowRegistry(workflows_dir=tmp_path))
    try:
        pane.refresh()
        card = pane._cards["alpha"]
        assert card.findChildren(QAbstractButton) == []
    finally:
        pane.deleteLater()
