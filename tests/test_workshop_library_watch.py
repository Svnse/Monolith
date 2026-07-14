from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from core.workflow_registry import WorkflowRegistry


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _seed(d: Path, wid: str):
    (d / f"{wid}.monoline").write_text(json.dumps({
        "id": wid, "name": wid, "description": "", "blocks": [], "connections": [], "composites": []}),
        encoding="utf-8")


def test_pane_watches_worlds_dir(app, tmp_path):
    from ui.panels.workshop_library import WorkshopLibraryPane
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    pane = None
    try:
        pane = WorkshopLibraryPane(registry=reg)
        assert str(tmp_path) in pane._watcher.directories()
    finally:
        if pane is not None:
            pane.deleteLater()


def test_on_dir_changed_refreshes(app, tmp_path):
    from ui.panels.workshop_library import WorkshopLibraryPane
    reg = WorkflowRegistry(workflows_dir=tmp_path)
    pane = None
    try:
        pane = WorkshopLibraryPane(registry=reg)
        pane.refresh()
        before = pane.card_count()
        _seed(Path(tmp_path), "newflow")
        pane._on_worlds_changed(str(tmp_path))   # simulate the watcher firing
        assert pane.card_count() == before + 1
        assert "newflow" in pane.card_ids()
    finally:
        if pane is not None:
            pane.deleteLater()
