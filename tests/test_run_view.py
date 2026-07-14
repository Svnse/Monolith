from __future__ import annotations

import os

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PySide6")
from PySide6.QtWidgets import QApplication

from core.run_model import (
    BlockFinished,
    RunBlockSpec,
    RunFinished,
    RunModelBuilder,
    RunStarted,
)


@pytest.fixture
def app():
    return QApplication.instance() or QApplication([])


def _model() -> RunModelBuilder:
    b = RunModelBuilder()
    b.apply(RunStarted(
        run_id="r1", flow_id="two-step", name="Two-Step", user_input="hi",
        graph=[
            RunBlockSpec(id="input", label="request", kind="port"),
            RunBlockSpec(id="draft", label="draft", kind="llm"),
            RunBlockSpec(id="output", label="response", kind="port"),
        ],
        wires=["input.value -> draft.prompt", "draft.response -> output.value"]))
    return b


def _draft_done(b, out="DRAFTED"):
    b.apply(BlockFinished(run_id="r1", block_id="draft", label="draft", kind="llm",
                          outputs={"response": out}, started_at=1.0, completed_at=2.0,
                          status="done"))


def test_runview_renders_a_row_per_block(app):
    from ui.components.run_view import RunView
    v = RunView(_model().model)
    v.show()
    assert v.block_row_count() == 3
    assert "Two-Step" in v.header_text()
    assert "running" in v.header_text().lower()


def test_runview_updates_and_emits_height_on_block_finish(app):
    from ui.components.run_view import RunView
    b = _model()
    v = RunView(b.model)
    v.show()
    bumps: list = []
    v.sig_height_changed.connect(lambda: bumps.append(1))
    _draft_done(b)
    assert v.row_status("draft") == "done"
    assert bumps  # a height change was emitted on the model update


def test_runview_row_expands_to_show_io(app):
    from ui.components.run_view import RunView
    b = _model()
    _draft_done(b)
    v = RunView(b.model)
    v.show()
    v.expand_row("draft")
    detail = v.row_detail_text("draft")
    assert "DRAFTED" in detail   # output shown
    assert "hi" in detail        # derived input (user_input via the input port)


def test_runview_rows_have_spacing_and_preview(app):
    from ui.components.run_view import RunView
    b = _model()
    _draft_done(b)
    v = RunView(b.model)
    v.show()
    assert v.row_spacing() >= 4
    assert "DRAFTED" in v.row_preview_text("draft")


def test_runview_collapsible_starts_small_and_opens(app):
    # Inline chat mount = ONE collapsible "tool block": small by default (inner blocks hidden),
    # toggle to reveal the workflow's blocks. The summary header is always shown.
    from ui.components.run_view import RunView
    b = _model()
    _draft_done(b)
    v = RunView(b.model, collapsible=True)
    v.show()
    assert v.is_collapsed() is True
    assert v.rows_visible() is False
    assert "blocks" in v.header_text()
    bumps = []
    v.sig_height_changed.connect(lambda: bumps.append(1))
    v.toggle_collapsed()
    assert v.is_collapsed() is False
    assert v.rows_visible() is True
    assert v.block_row_count() == 3      # inner blocks now revealed
    assert bumps                          # toggling re-lays-out the chat row


def test_runview_non_collapsible_always_shows_rows(app):
    # Browser/default mount is the inspector: rows always shown, never collapsed.
    from ui.components.run_view import RunView
    v = RunView(_model().model)
    v.show()
    assert v.is_collapsed() is False
    assert v.rows_visible() is True


def test_runview_shows_final_output_on_run_finished(app):
    from ui.components.run_view import RunView
    b = _model()
    b.apply(RunFinished(run_id="r1", output="FINAL ANSWER", error=""))
    v = RunView(b.model)
    v.show()
    assert "done" in v.header_text().lower()
    assert "FINAL ANSWER" in v.final_output_text()


def test_runview_hides_final_line_when_show_final_false(app):
    # Inline-in-chat mount: the assistant bubble carries the answer, so the card must NOT
    # also print the final "→ <answer>" line (the duplicate-answer declutter).
    from ui.components.run_view import RunView
    b = _model()
    b.apply(RunFinished(run_id="r1", output="FINAL ANSWER", error=""))
    v = RunView(b.model, show_final=False)
    v.show()
    assert v.final_output_text() == ""        # no duplicated final line inline
    assert v.block_row_count() == 3           # blocks still render


def test_runview_ignores_updates_from_stale_model(app):
    from ui.components.run_view import RunView
    b1, b2 = _model(), _model()
    v = RunView(b1.model)
    v.show()
    v.bind(b2.model)               # rebind to a different run
    _draft_done(b1, out="STALE")   # updating the OLD model must not touch the view
    assert v.row_status("draft") == "pending"   # b2's draft is still pending


def test_runview_renders_stopped_state_without_error(app):
    # On STOP: the interrupted block + run render a neutral "stopped" state — its own glyph,
    # NOT the red error styling, and no "error:" final line.
    from ui.components.run_view import RunView, _DOT
    assert _DOT.get("stopped")  # a dedicated glyph exists for the stopped state
    b = _model()
    b.apply(BlockFinished(run_id="r1", block_id="draft", label="draft", kind="llm",
                          outputs={}, started_at=1.0, completed_at=2.0, status="stopped"))
    b.apply(RunFinished(run_id="r1", output="", error="Activation stopped.", stopped=True))
    v = RunView(b.model)
    v.show()
    assert v.row_status("draft") == "stopped"            # the block shows the stopped state
    head = v._rows["draft"]._head.text()
    assert _DOT["stopped"] in head and "stopped" in head  # glyph + word rendered on the row
    assert "stopped" in v.header_text().lower()           # run header reads "... stopped ..."
    assert "error" not in v.final_output_text().lower()   # no error line for an intentional stop
