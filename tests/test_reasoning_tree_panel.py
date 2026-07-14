# tests/test_reasoning_tree_panel.py
from core import branch_tree
from ui.panels.reasoning_tree import build_rows, extract_think


def _fixture_tree():
    t = branch_tree.new_tree()
    u1 = branch_tree.append_child_payload(t, branch_tree.ROOT_ID, {"role": "user", "text": "explain the verifier please"})
    a1 = branch_tree.append_child_payload(t, u1, {"role": "assistant", "text": "<think>hmm</think>five checks"})
    u2 = branch_tree.append_child_payload(t, a1, {"role": "user", "text": "regen that"})
    a2a = branch_tree.append_child_payload(t, u2, {"role": "assistant", "text": "take one"}, origin="regen")
    a2b = branch_tree.append_child_payload(t, u2, {"role": "assistant", "text": ""}, origin="thinkpad:c1")
    d = branch_tree.append_child_payload(t, u2, {"role": "user", "text": "different"}, origin="edit",
                                         branch_kind="divergent", divergence=0.8)
    branch_tree.set_active_leaf(t, a2b)
    return t, {"u1": u1, "a1": a1, "u2": u2, "a2a": a2a, "a2b": a2b, "d": d}


def test_build_rows_marks_path_and_siblings():
    t, ids = _fixture_tree()
    rows = {r.node_id: r for r in build_rows(t)}
    assert len(rows) == 6
    assert rows[ids["a1"]].on_path is True
    assert rows[ids["a2a"]].on_path is False
    assert rows[ids["a2b"]].on_path is True
    assert rows[ids["a2b"]].take == (2, 3)                # 2nd of 3 siblings under u2
    assert rows[ids["a2a"]].take == (1, 3)
    assert rows[ids["a1"]].take is None                   # only child → no counter


def test_build_rows_chips_and_preview():
    t, ids = _fixture_tree()
    rows = {r.node_id: r for r in build_rows(t)}
    assert "explain the verifier" in rows[ids["u1"]].label
    assert "[regen]" in rows[ids["a2a"]].label
    assert "[thinkpad:c1]" in rows[ids["a2b"]].label
    assert "✎" in rows[ids["d"]].label                    # divergent edit chip
    assert "⟳" in rows[ids["a2b"]].label                  # empty text → streaming chip


def test_build_rows_sibling_glyphs():
    t, ids = _fixture_tree()
    rows = {r.node_id: r for r in build_rows(t)}
    assert rows[ids["a2a"]].glyph.startswith("├")
    assert rows[ids["d"]].glyph.startswith("└")
    assert rows[ids["a1"]].glyph.startswith("●")          # single child on path


def test_extract_think():
    assert extract_think("<think>alpha</think>answer") == "alpha"
    assert extract_think("<analysis>b</analysis>x<think>c</think>") == "b\nc"
    assert extract_think("no envelope here") == ""
    assert extract_think("") == ""


def test_build_rows_handles_long_linear_sessions():
    t = branch_tree.new_tree()
    parent = branch_tree.ROOT_ID
    for i in range(1500):
        parent = branch_tree.append_child_payload(
            t, parent, {"role": "user" if i % 2 == 0 else "assistant", "text": f"m{i}"})
    rows = build_rows(t)
    assert len(rows) == 1500


# ── Task 8: panel widget + hover think popup ──────────────────────────────
import pytest
from PySide6.QtWidgets import QApplication


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance() or QApplication([])
    yield app


class _FakeController:
    def __init__(self, tree):
        self._session = {"tree": tree, "messages": []}
        self.switched = []

    def current_session_data(self):
        return self._session

    def switch_reasoning_path(self, node_id):
        self.switched.append(node_id)
        return True


def test_panel_renders_fixture_rows(qapp, monkeypatch):
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    from ui.panels.reasoning_tree import ReasoningTreePanel
    t, ids = _fixture_tree()
    panel = ReasoningTreePanel()
    panel.bind_controller(_FakeController(t))
    panel.show(); panel.refresh()
    assert panel._list.count() == 6
    assert "3 forks" in panel._chip.text() or "fork" in panel._chip.text()


def test_panel_click_switches(qapp, monkeypatch):
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    from ui.panels.reasoning_tree import ReasoningTreePanel
    t, ids = _fixture_tree()
    ctrl = _FakeController(t)
    panel = ReasoningTreePanel()
    panel.bind_controller(ctrl)
    panel.show(); panel.refresh()
    for r in range(panel._list.count()):
        item = panel._list.item(r)
        if item.data(panel.NODE_ROLE) == ids["a2a"]:
            panel._on_item_clicked(item)
            break
    assert ctrl.switched == [ids["a2a"]]


def test_panel_flag_off_disabled_hint(qapp, monkeypatch):
    monkeypatch.delenv("MONOLITH_BRANCH_TREE_V1", raising=False)
    from ui.panels.reasoning_tree import ReasoningTreePanel
    panel = ReasoningTreePanel()
    panel.bind_controller(_FakeController(branch_tree.new_tree()))
    panel.show(); panel.refresh()
    assert "disabled" in panel._hint.text().lower()


def test_think_popup_content(qapp, monkeypatch):
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    from ui.panels.reasoning_tree import ReasoningTreePanel
    t, ids = _fixture_tree()
    panel = ReasoningTreePanel()
    panel.bind_controller(_FakeController(t))
    panel.show(); panel.refresh()
    assert panel._popup_text_for(ids["a1"]) == "hmm"
    assert panel._popup_text_for(ids["a2a"]) == "(no think recorded)"
    assert panel._popup_text_for(ids["u1"]) is None      # user node → suppressed


def test_popup_survives_cursor_inside_geometry(qapp, monkeypatch):
    """_maybe_hide_popup leaves the popup open when cursor_pos is inside its frame."""
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    from ui.panels.reasoning_tree import ReasoningTreePanel
    from PySide6.QtCore import QPoint
    t, ids = _fixture_tree()
    panel = ReasoningTreePanel()
    panel.bind_controller(_FakeController(t))
    panel.show(); panel.refresh()
    panel._popup.present("h", "body", panel.mapToGlobal(panel.rect().center()))
    # Simulate cursor inside: pass center of popup frame explicitly (QCursor.setPos
    # is unreliable in headless CI, so we use the cursor_pos parameter instead).
    inside = panel._popup.frameGeometry().center()
    panel._maybe_hide_popup(cursor_pos=inside)
    assert panel._popup.isVisible()           # cursor inside → popup stays

    # Simulate cursor outside: far away from the popup.
    outside = panel._popup.frameGeometry().bottomRight() + QPoint(500, 500)
    panel._maybe_hide_popup(cursor_pos=outside)
    assert not panel._popup.isVisible()       # cursor outside → popup hides


def test_close_event_deletes_popup(qapp, monkeypatch):
    """closeEvent hides the popup, calls deleteLater, and sets _popup to None."""
    monkeypatch.setenv("MONOLITH_BRANCH_TREE_V1", "1")
    from ui.panels.reasoning_tree import ReasoningTreePanel
    t, ids = _fixture_tree()
    panel = ReasoningTreePanel()
    panel.bind_controller(_FakeController(t))
    panel.show()
    panel.close()
    assert panel._popup is None


# ── Task 9: registration ──────────────────────────────────────────────────
def test_companion_state_has_reasoning_tree():
    from ui.companion_pane import CompanionState
    assert hasattr(CompanionState, "REASONING_TREE")


# ── Task 10: in-chat ‹k/n› take switcher ──────────────────────────────────
def test_take_info_for_index():
    from ui.pages.session_tree import take_info_for_index
    from core.branch_tree import project_to_messages
    t, ids = _fixture_tree()
    session = {"tree": t, "messages": project_to_messages(t)}
    # the projection's last row is a2b (active leaf), 2nd of 3 takes
    idx = len(session["messages"]) - 1
    info = take_info_for_index(session, idx)
    assert info == (2, 3)
    assert take_info_for_index(session, 0) is None     # u1 has no siblings


def test_switch_take_steps_to_neighbor():
    from ui.pages.session_tree import switch_take
    from core.branch_tree import project_to_messages
    t, ids = _fixture_tree()
    session = {"tree": t, "messages": project_to_messages(t)}
    idx = len(session["messages"]) - 1                  # a2b, take 2/3
    assert switch_take(session, idx, -1) is True        # → a2a
    assert session["messages"][-1]["text"] == "take one"
