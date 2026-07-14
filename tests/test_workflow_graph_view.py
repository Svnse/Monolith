from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from core.run_model import BlockFinished, RunBlockSpec, RunModelBuilder, RunStarted
from core.workflow_registry import Workflow


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _workflow(tmp_path: Path) -> Workflow:
    path = tmp_path / "alpha.monoline"
    path.write_text(json.dumps({
        "id": "alpha",
        "name": "Alpha",
        "blocks": [
            {"id": "input", "kind": "port", "label": "Input", "position": [0, 0]},
            {"id": "draft", "kind": "llm", "label": "Draft", "position": [250, 0]},
            {"id": "output", "kind": "port", "label": "Output", "position": [500, 0]},
        ],
        "connections": [
            {"from_block": "input", "to_block": "draft"},
            {"from_block": "draft", "to_block": "output"},
        ],
    }), encoding="utf-8")
    return Workflow(id="alpha", name="Alpha", description="", kind="monoline", source_path=path)


def _dense_workflow(tmp_path: Path) -> Workflow:
    path = tmp_path / "dense.monoline"
    blocks = [
        {"id": f"b{i}", "kind": "llm", "label": f"Block {i}", "position": [i * 260, 300]}
        for i in range(7)
    ]
    path.write_text(json.dumps({
        "id": "dense",
        "name": "Dense",
        "blocks": blocks,
        "connections": [
            {"from_block": f"b{i}", "to_block": f"b{i + 1}"}
            for i in range(len(blocks) - 1)
        ],
    }), encoding="utf-8")
    return Workflow(id="dense", name="Dense", description="", kind="monoline", source_path=path)


def test_workflow_graph_reads_monoline_blocks_and_wires(app, tmp_path):
    from ui.components.workflow_graph import WorkflowGraphView

    view = WorkflowGraphView()
    view.bind_workflow(_workflow(Path(tmp_path)))

    assert view.node_count() == 3
    assert view.wire_count() == 2
    assert view.node_labels() == ["Input", "Draft", "Output"]


def test_workflow_graph_can_reflect_run_status(app, tmp_path):
    from ui.components.workflow_graph import WorkflowGraphView

    builder = RunModelBuilder()
    builder.apply(RunStarted(
        run_id="r1",
        flow_id="alpha",
        name="Alpha",
        user_input="hi",
        graph=[
            RunBlockSpec(id="input", label="Input", kind="port"),
            RunBlockSpec(id="draft", label="Draft", kind="llm"),
            RunBlockSpec(id="output", label="Output", kind="port"),
        ],
        wires=["input.value -> draft.prompt", "draft.response -> output.value"],
    ))
    builder.apply(BlockFinished(
        run_id="r1",
        block_id="draft",
        label="Draft",
        kind="llm",
        outputs={"response": "drafted"},
        started_at=1.0,
        completed_at=2.0,
        status="done",
    ))

    view = WorkflowGraphView()
    view.bind_workflow(_workflow(Path(tmp_path)))
    view.bind_run_model(builder.model)

    assert view.status_for("draft") == "done"


def test_workflow_graph_wraps_dense_workflows_when_narrow(app, tmp_path):
    from ui.components.workflow_graph import WorkflowGraphView

    view = WorkflowGraphView()
    view.bind_workflow(_dense_workflow(Path(tmp_path)))

    assert view.layout_row_count_for_width(260) > 1
    assert view.heightForWidth(260) > view.heightForWidth(720)

    view.resize(260, view.heightForWidth(260))
    rects = view._layout_rects()
    values = list(rects.values())
    assert len(values) == 7
    for idx, rect in enumerate(values):
        for other in values[idx + 1:]:
            assert not rect.intersects(other)
