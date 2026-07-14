from __future__ import annotations

from PySide6.QtCore import QFileSystemWatcher, Qt, QTimer, Signal
from PySide6.QtWidgets import (QAbstractButton, QButtonGroup, QFrame, QHBoxLayout,
                               QLabel, QScrollArea, QSizePolicy, QStackedWidget,
                               QVBoxLayout, QWidget)

import core.style as _s
from ui.components.atoms import MonoButton
from ui.components.workflow_graph import WorkflowGraphView
from ui.panels.workshop import WorkshopPane
from core.workflow_registry import GENESIS_ID, WorkflowRegistry


def _tok(name: str, fallback: str) -> str:
    return getattr(_s, name, fallback)


class _WorkflowCard(QFrame):
    """One bench list row. The whole row equips; actions live in the selected-workflow strip."""

    _SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"

    def __init__(self, workflow, *, is_active: bool, on_equip, on_edit=None, on_test=None) -> None:
        super().__init__()
        self.workflow = workflow
        self._on_equip = on_equip
        self._is_active = is_active
        self._running = False
        self._spin_i = 0
        self.setObjectName("workflow_card")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        if workflow.description:
            self.setToolTip(workflow.description)

        row = QHBoxLayout(self)
        row.setContentsMargins(8, 5, 8, 5)
        row.setSpacing(6)
        self._dot = QLabel()
        self._dot.setFixedWidth(14)
        row.addWidget(self._dot)
        self._name = QLabel(workflow.name)
        self._name.setStyleSheet(f"color:{_s.FG_TEXT}; font-weight:600;")
        # Ignored horizontal policy: the name never dictates a minimum width, so a long name
        # clips to the panel instead of forcing a horizontal scrollbar.
        self._name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        row.addWidget(self._name, 1)
        badge = QLabel(workflow.kind.upper())
        badge.setStyleSheet(f"color:{_s.FG_DIM}; font-size:8px; letter-spacing:1px;")
        row.addWidget(badge)

        self._spin = QTimer(self)
        self._spin.setInterval(110)
        self._spin.timeout.connect(self._tick)
        self._apply_active(is_active)

    def _apply_active(self, is_active: bool) -> None:
        self._is_active = is_active
        if not self._running:
            self._dot.setText("◆" if is_active else "◇")
            self._dot.setStyleSheet(
                f"color:{_s.ACCENT_PRIMARY if is_active else _s.FG_DIM}; font-size:12px;")
        if is_active:
            self.setStyleSheet(
                f"#workflow_card {{ border-left:3px solid {_s.ACCENT_PRIMARY}; "
                f"background:{_tok('BG_SURFACE_2', _s.BG_PANEL)}; }}")
        else:
            self.setStyleSheet(
                "#workflow_card { border-left:3px solid transparent; background:transparent; }")

    def set_active(self, is_active: bool) -> None:
        self._apply_active(is_active)

    def set_running(self, running: bool) -> None:
        running = bool(running)
        if running == self._running:
            return
        self._running = running
        if running:
            self._spin.start()
        else:
            self._spin.stop()
            self._apply_active(self._is_active)  # restore the dot glyph

    def _tick(self) -> None:
        self._spin_i = (self._spin_i + 1) % len(self._SPIN)
        self._dot.setText(self._SPIN[self._spin_i])
        self._dot.setStyleSheet(f"color:{_s.ACCENT_PRIMARY}; font-size:12px;")

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802
        try:
            pos = event.position().toPoint()
        except Exception:
            pos = event.pos()
        child = self.childAt(pos)
        if not isinstance(child, QAbstractButton):
            self._on_equip(self.workflow.id)
        super().mouseReleaseEvent(event)


class _CreateCard(QFrame):
    def __init__(self, on_create) -> None:
        super().__init__()
        self.setObjectName("workflow_create_card")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(8, 4, 8, 4)
        btn = MonoButton("+ new workflow", accent=True)
        btn.clicked.connect(lambda: on_create())
        lay.addWidget(btn)
        lay.addStretch(1)


class WorkshopLibraryPane(QWidget):
    """The WORKSHOP companion view: a sectioned, click-to-equip list (NATIVE / MY WORKFLOWS) +
    a forge row, plus the existing run-tree as a RUNNING sub-view. Clicking a row equips it
    (WRITES the active-flow flag the chat guard reads); the equipped row is highlighted in place.
    No separate 'equipped' slot. The panel scrolls vertically only -- rows always fit the width."""

    sig_open_monoline = Signal(object)  # str id (edit) or None (forge/create)
    sig_focus_chat = Signal()
    sig_test_run = Signal(str)          # workflow id -> in-panel dry-run

    def __init__(self, *, registry: WorkflowRegistry | None = None) -> None:
        super().__init__()
        self._registry = registry or WorkflowRegistry()
        self._active_id = ""
        self._rendered_sig: tuple = ()
        self._cards: dict[str, _WorkflowCard] = {}
        self._workflows: list = []  # set by refresh(); init here so _on_set_active works pre-show()

        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(6)
        # No title label — identity lives in the pane header (UI_CONTRACT §2).
        tabs = QHBoxLayout()
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        self._btn_library = MonoButton("FLOWS")
        self._btn_running = MonoButton("RUNS")
        for i, b in enumerate((self._btn_library, self._btn_running)):
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, idx=i: self._show_view(idx))
            self._group.addButton(b, i)
            tabs.addWidget(b)
        tabs.addStretch(1)
        root.addLayout(tabs)

        self._stack = QStackedWidget()
        # page 0: BENCH (selected graph + compact click-to-equip list)
        self._bench = QWidget()
        bench_layout = QVBoxLayout(self._bench)
        bench_layout.setContentsMargins(0, 0, 0, 0)
        bench_layout.setSpacing(6)
        self._graph = WorkflowGraphView()
        bench_layout.addWidget(self._graph)

        self._selected_bar = QFrame()
        self._selected_bar.setObjectName("workshop_selected_bar")
        self._selected_bar.setStyleSheet(
            f"#workshop_selected_bar {{ background:{_tok('BG_SURFACE_2', _s.BG_PANEL)}; "
            f"border:1px solid {_s.BORDER_SUBTLE}; border-radius:6px; }}"
        )
        selected_layout = QHBoxLayout(self._selected_bar)
        selected_layout.setContentsMargins(8, 5, 8, 5)
        selected_layout.setSpacing(6)
        self._selected_name = QLabel("")
        self._selected_name.setStyleSheet(f"color:{_s.FG_TEXT}; font-weight:600;")
        self._selected_name.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        selected_layout.addWidget(self._selected_name, 1)
        self._btn_edit_selected = MonoButton("Edit")
        self._btn_edit_selected.clicked.connect(self._edit_selected)
        selected_layout.addWidget(self._btn_edit_selected)
        self._btn_test_selected = MonoButton("Test")
        self._btn_test_selected.clicked.connect(self._test_selected)
        selected_layout.addWidget(self._btn_test_selected)
        bench_layout.addWidget(self._selected_bar)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QFrame.Shape.NoFrame)
        # vertical scroll only -- a row can never trigger a horizontal scrollbar / sideways spill.
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._cards_host = QWidget()
        self._cards_layout = QVBoxLayout(self._cards_host)
        self._cards_layout.setContentsMargins(2, 2, 2, 2)
        self._cards_layout.setSpacing(3)
        self._scroll.setWidget(self._cards_host)
        bench_layout.addWidget(self._scroll, 1)
        self._stack.addWidget(self._bench)
        # page 1: RUNNING (the existing run-tree, embedded verbatim)
        self._run_tree = WorkshopPane()
        self._stack.addWidget(self._run_tree)
        root.addWidget(self._stack, 1)
        self._btn_library.setChecked(True)

        # poll the workshop activity flag so the equipped row spins while a run is in flight.
        self._run_poll = QTimer(self)
        self._run_poll.setInterval(1000)
        self._run_poll.timeout.connect(self._poll_running)

        # Auto-refresh when a flow is saved in the (separate) Monoline canvas.
        self._watcher = QFileSystemWatcher(self)
        try:
            wdir = self._registry.workflows_dir
            wdir.mkdir(parents=True, exist_ok=True)   # addPath requires an existing dir
            self._watcher.addPath(str(wdir))
        except Exception:
            pass
        self._watcher.directoryChanged.connect(self._on_worlds_changed)

    def _on_worlds_changed(self, _path: str) -> None:
        self.refresh()
        try:  # some platforms drop the watch after a dir-replace; re-add defensively.
            wdir = str(self._registry.workflows_dir)
            if wdir not in self._watcher.directories():
                self._watcher.addPath(wdir)
        except Exception:
            pass

    # -- companion contract --
    def bind_controller(self, controller) -> None:
        wire = getattr(controller, "wire_workshop_library", None)
        if callable(wire):
            try:
                wire(self)
            except Exception:
                pass

    def running_pane(self) -> WorkshopPane:
        return self._run_tree

    def workflow_graph(self) -> WorkflowGraphView:
        return self._graph

    def active_id(self) -> str:
        return self._active_id

    # -- introspection (smoke test) --
    def card_count(self) -> int:
        # count only the workflow rows + the forge card; the section-header QLabels and the trailing
        # stretch are NOT counted.
        return sum(1 for i in range(self._cards_layout.count())
                   if isinstance(self._cards_layout.itemAt(i).widget(), (_WorkflowCard, _CreateCard)))

    def card_ids(self) -> list[str]:
        return [w.id for w in self._workflows]

    def has_create_card(self) -> bool:
        return any(isinstance(self._cards_layout.itemAt(i).widget(), _CreateCard)
                   for i in range(self._cards_layout.count()))

    # -- view switching --
    def _show_view(self, idx: int) -> None:
        self._stack.setCurrentIndex(idx)
        self._btn_library.setChecked(idx == 0)
        self._btn_running.setChecked(idx == 1)

    def showEvent(self, event) -> None:  # noqa: N802
        self._pick_default_tab()
        self.refresh()
        self._run_poll.start()
        self._poll_running()
        super().showEvent(event)

    def hideEvent(self, event) -> None:  # noqa: N802
        self._run_poll.stop()
        super().hideEvent(event)

    def _pick_default_tab(self) -> None:
        self._show_view(1 if self._is_workshop_running() else 0)

    def _is_workshop_running(self) -> bool:
        return self._poll_running()

    def _poll_running(self) -> bool:
        running = False
        ws = getattr(self._registry, "_ws", None)
        if ws is not None:
            try:
                engines = (ws.snapshot() or {}).get("engines", {}) or {}
                st = str((engines.get("workshop") or {}).get("status", "")).strip().lower()
                running = st in ("running", "generating", "streaming")
            except Exception:
                running = False
        # spin the equipped row (the one that runs) while a run is in flight.
        for cid, card in self._cards.items():
            try:
                card.set_running(running and self._is_active_id(cid))
            except Exception:
                pass
        return running

    # -- active-flow identity ("" == Genesis default; the Genesis row owns both "" and GENESIS_ID) --
    def _is_active_id(self, wid: str) -> bool:
        if wid == self._active_id:
            return True
        return self._active_id in ("", GENESIS_ID) and wid == GENESIS_ID

    # -- live render --
    def refresh(self) -> None:
        self._active_id = self._registry_active_id()
        self._workflows = self._registry.list_workflows()
        sig = (tuple((w.id, self._workflow_mtime(w)) for w in self._workflows), self._active_id)
        if sig == self._rendered_sig:
            self._sync_selected_workflow()
            return  # churn guard: no rebuild
        prev = self._scroll.verticalScrollBar().value()
        self._rebuild_cards()
        self._rendered_sig = sig
        self._sync_selected_workflow()
        bar = self._scroll.verticalScrollBar()
        bar.setValue(max(0, min(prev, bar.maximum())))

    def _workflow_mtime(self, workflow) -> float:
        path = getattr(workflow, "source_path", None)
        try:
            return path.stat().st_mtime if path is not None else 0.0
        except OSError:
            return 0.0

    def _registry_active_id(self) -> str:
        getter = getattr(self._registry, "active_id", None)
        return getter() if callable(getter) else self._active_id

    def _rebuild_cards(self) -> None:
        while self._cards_layout.count():
            item = self._cards_layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
        self._cards = {}
        native = [w for w in self._workflows if w.kind != "monoline"]
        monoline = [w for w in self._workflows if w.kind == "monoline"]
        if native:
            self._cards_layout.addWidget(self._section("NATIVE"))
            for wf in native:
                self._add_row(wf)
        self._cards_layout.addWidget(self._section("MY WORKFLOWS"))
        for wf in monoline:
            self._add_row(wf)
        self._cards_layout.addWidget(_CreateCard(
            on_create=lambda: self.sig_open_monoline.emit(None)))
        self._cards_layout.addStretch(1)

    def _add_row(self, wf) -> None:
        card = _WorkflowCard(
            wf, is_active=self._is_active_id(wf.id),
            on_equip=self._on_set_active,
            on_edit=lambda wid: self.sig_open_monoline.emit(wid),
            on_test=lambda wid: self.sig_test_run.emit(wid))
        self._cards[wf.id] = card
        self._cards_layout.addWidget(card)

    def _section(self, text: str) -> QLabel:
        lbl = QLabel(f"── {text} " + "─" * max(2, 24 - len(text)))
        lbl.setObjectName("workshop_section")
        lbl.setStyleSheet(f"color:{_s.FG_DIM}; font-size:9px; letter-spacing:1px;")
        return lbl

    def _active_workflow(self):
        if not self._workflows:
            return None
        for wf in self._workflows:
            if self._is_active_id(wf.id):
                return wf
        return self._workflows[0]

    def _sync_selected_workflow(self) -> None:
        wf = self._active_workflow()
        if wf is None:
            self._selected_name.setText("No workflow")
            self._btn_edit_selected.setVisible(False)
            self._btn_test_selected.setVisible(False)
            self._graph.bind_workflow(None)
            return
        self._selected_name.setText(wf.name)
        self._graph.bind_workflow(wf)
        is_monoline = wf.kind == "monoline"
        self._btn_edit_selected.setVisible(is_monoline)
        self._btn_test_selected.setVisible(is_monoline)
        self._btn_edit_selected.setEnabled(is_monoline)
        self._btn_test_selected.setEnabled(is_monoline)

    def _edit_selected(self) -> None:
        wf = self._active_workflow()
        if wf is not None and wf.kind == "monoline":
            self.sig_open_monoline.emit(wf.id)

    def _test_selected(self) -> None:
        wf = self._active_workflow()
        if wf is not None and wf.kind == "monoline":
            self.sig_test_run.emit(wf.id)

    def _on_set_active(self, workflow_id: str, *, write=None) -> None:
        self._active_id = workflow_id or ""
        if write is not None:
            write(self._active_id)  # injectable for tests
        else:
            setter = getattr(self._registry, "set_active", None)
            if callable(setter):
                setter(self._active_id)
        # repaint every row's active state in place (cheap; handles the ""/GENESIS_ID duality
        # without tracking the previous id) -> no rebuild, no scroll reset.
        for cid, card in self._cards.items():
            card.set_active(self._is_active_id(cid))
        self._sync_selected_workflow()
