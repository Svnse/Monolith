from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from core.workflow_registry import WorkflowRegistry, GENESIS_ID


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _seed(d: Path):
    (d / "alpha.monoline").write_text(json.dumps({
        "schema_version": 1, "id": "alpha", "name": "Alpha",
        "description": "first flow", "modified_at": 2.0,
        "blocks": [], "connections": [], "composites": []}), encoding="utf-8")


def test_library_builds_card_per_workflow_plus_create(app, tmp_path):
    from ui.panels.workshop_library import WorkshopLibraryPane
    _seed(Path(tmp_path))
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    pane = None
    try:
        pane = WorkshopLibraryPane(registry=reg)
        pane.show()
        pane.refresh()
        # genesis card + alpha card + create card
        assert pane.card_count() == 3
        assert pane.has_create_card() is True
        assert pane.workflow_graph().node_count() == 1  # Genesis native preview
        ids = pane.card_ids()
        assert ids[0] == GENESIS_ID and "alpha" in ids
    finally:
        if pane is not None:
            pane.deleteLater()


def test_set_active_writes_flag_only(app, tmp_path, monkeypatch):
    from ui.panels.workshop_library import WorkshopLibraryPane
    _seed(Path(tmp_path))
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    written = {}
    pane = None
    try:
        pane = WorkshopLibraryPane(registry=reg)
        pane._on_set_active("alpha", write=lambda wid: written.__setitem__("id", wid))
        assert written["id"] == "alpha"
        # active styling repaints in place (no full rebuild)
        assert pane.active_id() == "alpha"
    finally:
        if pane is not None:
            pane.deleteLater()


def test_running_tab_embeds_workshop_pane(app, tmp_path):
    from ui.panels.workshop_library import WorkshopLibraryPane
    from ui.panels.workshop import WorkshopPane
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    pane = None
    try:
        pane = WorkshopLibraryPane(registry=reg)
        pane.show()
        assert isinstance(pane.running_pane(), WorkshopPane)
    finally:
        if pane is not None:
            pane.deleteLater()


def test_refresh_is_idempotent_no_scroll_reset(app, tmp_path):
    from ui.panels.workshop_library import WorkshopLibraryPane
    _seed(Path(tmp_path))
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    pane = None
    try:
        pane = WorkshopLibraryPane(registry=reg)
        pane.show()
        pane.refresh()
        n1 = pane.card_count()
        prev = pane._scroll.verticalScrollBar().value()
        pane.refresh()  # no change -> guard skips rebuild
        assert pane.card_count() == n1
        assert pane._scroll.verticalScrollBar().value() == prev
    finally:
        if pane is not None:
            pane.deleteLater()
