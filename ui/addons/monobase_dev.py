"""MonoBase dev panel — brutally thin candidate triage for A1.

Single-file Qt widget. ~1-2 hours of UI work, intentional. The point
is *behavior validation inside Monolith's actual operating surface*,
not polish.

A1 surface (this file):
  - Pending candidate list (left)
  - Evidence text + reason + source + run_id (right)
  - Accept / Reject buttons (Reject prompts for reason)
  - "Audit now" button → enqueues a manual trigger on the worker
  - "Refresh" button → re-reads pending list
  - Status bar: cursor value, pending count

A1 explicitly does NOT have:
  - Keyboard shortcuts (J/K/A/R/E/D)
  - Bulk operations
  - Edit affordance
  - Provenance chain view
  - Leaf-ACU view
  - Auto-refresh / contradiction side-by-side
  - Defer button (defer is SQL-only in A1)

All deferred to A2.

The panel takes a TriggerQueue at construction so it can enqueue
manual triggers. The worker (and LLM) is managed externally — the
panel doesn't care who's processing the queue, only that someone is.
"""
from __future__ import annotations

from typing import Callable

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QButtonGroup,
    QCheckBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import core.style as _s
from ui.components.atoms import MonoButton, CollapsibleSection
from ui.addons.monobase_widgets import (
    AcatalepsyMapDot,
    AcatalepsyMapWidget,
    MonoBaseStatusStrip,
)
from core.acatalepsy import candidates as _candidates
from core.acatalepsy import decisions as _decisions
from core.acatalepsy import canonical_log as _canonical_log
from core.acatalepsy import auditor as _auditor
from core.acatalepsy import auto_review as _auto_review
from core.acatalepsy import monobase_status as _monobase_status
from core.acatalepsy import runtime as _runtime
from core.acatalepsy.triggers import TriggerQueue


__all__ = ("MonoBaseDevPanel",)


_LIST_ROW_FMT = "[{id}] {canonical}"


class MonoBaseDevPanel(QWidget):
    """Thin validator UI for Acatalepsy v1 A1.

    Args:
        trigger_queue: the worker's queue — used to enqueue manual audit triggers.
            If None, the "Audit now" button is disabled (read-only mode).
        decider_id: who's making decisions through this panel.
            Default "user_e". Pass "agent_X" for cross-LLM curating.
        on_decision: optional callback fired after each accept/reject.
            Useful for tests or for triggering follow-up actions.
    """

    def __init__(
        self,
        trigger_queue: TriggerQueue | None = None,
        *,
        decider_id: str = "user_e",
        on_decision: Callable[[int, str], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # If no queue passed explicitly, try to pick up the active
        # worker from the runtime singleton (set by bootstrap). Falls
        # back to None → "Audit now" button disabled (read-only mode).
        if trigger_queue is None:
            active = _runtime.get_active_worker()
            if active is not None:
                trigger_queue = active.queue_handle
        self._trigger_queue = trigger_queue
        self._decider_id = decider_id
        self._on_decision = on_decision
        # Ids currently rendered in each detail pane (and the last LLM-output
        # text). The 2s auto-refresh re-asserts the same selection / re-sets the
        # same text every tick; without these guards each tick re-ran
        # setPlainText(), snapping the pane's scrollbar to the top — so reading
        # down a detail pane "teleported" back to the top every ~2 seconds.
        self._rendered_cid: int | None = None
        self._rendered_acu_id: int | None = None
        self._rendered_run_id: int | None = None
        self._rendered_log_id: int | None = None
        self._last_llm_output: str | None = None

        self.setWindowTitle("MonoBase (dev panel) — Acatalepsy v1 A1")
        # NOTE: do NOT call self.resize() — this widget is hosted inside the
        # companion pane (typical width 360-600px). Calling resize() here only
        # applies when the panel is launched as a top-level window, but it
        # used to fight the companion's layout. Removed to fit the panel
        # surface natively.

        # Auto-refresh so the user sees activity without clicking Refresh.
        # Cadence flips between idle (2s) and live (500ms) based on whether
        # an audit is in flight — set in _refresh_status. Created BEFORE
        # _refresh_all() so the initial refresh can read _refresh_timer
        # safely. Track in-flight state across refreshes for clean
        # status-bar transition messages.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.timeout.connect(self._refresh_all)
        self._was_in_flight: bool = False

        # Crash recovery: if the previous Monolith process died mid-audit,
        # the canonical log has an auditor_run_started without a matching
        # terminator. The in-flight check would report "RUNNING" forever
        # — which is why a freshly-restarted Monolith shows audits stuck.
        # Close stale runs (older than 10 minutes) on panel init so the
        # state matches reality.
        try:
            closed = _auditor.close_orphaned_runs(stale_after_secs=600.0)
            if closed > 0:
                # Status bar exists post-_build_ui, so defer the message.
                self._pending_init_status = (
                    f"closed {closed} orphaned run(s) from prior session"
                )
            else:
                self._pending_init_status = None
        except Exception:
            self._pending_init_status = None

        self._build_ui()
        if self._pending_init_status:
            self._status.showMessage(self._pending_init_status, 5000)
        self._refresh_all()
        self._refresh_timer.start(2000)

    # ── UI construction ───────────────────────────────────────────────

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        root.setSpacing(6)

        # ── SCAN PROGRESS + RESET ────────────────────────────────────────
        # Prominent line at the top: cursor → latest_event_id, plus delta.
        # The "↺" button next to it resets the cursor to 0 so the next
        # audit re-processes the entire canonical log from the beginning
        # (useful when you want to validate prompt changes or after a
        # config tweak that made earlier proposals look different).
        self._acu_map = AcatalepsyMapWidget()
        self._acu_map.dotActivated.connect(self._on_map_dot_activated)
        root.addWidget(self._acu_map)

        self._status_strip = MonoBaseStatusStrip()
        root.addWidget(self._status_strip)

        scan_row = QHBoxLayout()
        scan_row.setSpacing(4)
        self._scan_progress = QLabel("scan: cursor — / latest —")
        self._scan_progress.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-size: 11px; font-family: Consolas;"
            f" font-weight: bold; padding: 4px;"
            f" background: {_s.BG_INPUT};"
            f" border: 1px solid {_s.BORDER_SUBTLE}; border-radius: 3px;"
        )
        self._scan_progress.setToolTip(
            "cursor = last event id the auditor processed\n"
            "latest = newest event id in the canonical log\n"
            "Δ = events behind (waiting to be audited)"
        )
        scan_row.addWidget(self._scan_progress, 1)

        self._reset_btn = MonoButton("↺")
        self._reset_btn.setFixedSize(28, 28)
        self._reset_btn.setToolTip(
            "Reset cursor to 0 — the next audit will re-process the\n"
            "entire canonical log from the beginning. Confirms first."
        )
        self._reset_btn.clicked.connect(self._on_reset_cursor)
        scan_row.addWidget(self._reset_btn)
        root.addLayout(scan_row)

        # ── MODEL INDICATOR ──────────────────────────────────────────────
        # Shows which LLM the auditor WILL use when started. Reads the live
        # LLMConfig (api_base + api_model). Updates whenever the user changes
        # their loaded model. If config isn't openai-compat enough for the
        # sidecar, this shows the reason instead of a model name.
        self._model_label = QLabel("model: —")
        self._model_label.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas; padding: 2px 4px;"
        )
        self._model_label.setWordWrap(True)
        self._model_label.setToolTip(
            "The LLM the auditor will use to produce ACU candidates.\n"
            "Matches whatever model is currently configured in LLMConfig.\n"
            "Requires an openai-compat backend (api_base + api_model)."
        )
        root.addWidget(self._model_label)

        # ── CONTROLS: auditor on/off toggle + manual audit + refresh ─────
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)

        # Audit flip switch — flipping ON lazy-creates an AuditorWorker
        # using the currently-loaded LLM (via core.acatalepsy.llm_sidecar.
        # make_auditor_llm), registers it with the runtime, and starts the
        # background thread. Flipping OFF stops the thread but keeps the
        # worker registered so re-enabling is fast. The manual "Audit"
        # button works independently — it triggers a one-shot run on the
        # registered worker if any.
        self._auditor_toggle = QCheckBox("Auditor")
        self._auditor_toggle.setStyleSheet(
            f"QCheckBox {{ color: {_s.FG_TEXT}; font-size: 10px; font-family: Consolas; }}"
            f"QCheckBox::indicator {{ width: 12px; height: 12px; }}"
            f"QCheckBox::indicator:checked {{ background: {_s.ACCENT_PRIMARY}; "
            f"border: 1px solid {_s.ACCENT_PRIMARY}; }}"
            f"QCheckBox::indicator:unchecked {{ background: {_s.BG_INPUT}; "
            f"border: 1px solid {_s.BORDER_LIGHT}; }}"
        )
        self._auditor_toggle.toggled.connect(self._on_auditor_toggled)
        toolbar.addWidget(self._auditor_toggle)

        # Audit / Stop morph button — same widget, different behavior
        # based on whether a run is in flight. Click while idle =
        # enqueue trigger; click while running = cancel run + signal
        # worker to stop pulling triggers.
        self._audit_btn = MonoButton("Audit", accent=True)
        self._audit_btn.setFixedHeight(24)
        self._audit_btn.setToolTip("Enqueue a one-shot audit trigger on the worker.")
        self._audit_btn.clicked.connect(self._on_audit_or_stop_clicked)
        toolbar.addWidget(self._audit_btn)

        self._refresh_btn = MonoButton("Refresh")
        self._refresh_btn.setFixedHeight(24)
        self._refresh_btn.clicked.connect(self._refresh_all)
        toolbar.addWidget(self._refresh_btn)

        # Max-events-per-run spinbox. Defaults to 50 (sensible for cloud
        # LLMs — 483-event slices in one prompt invite hallucination).
        # User can crank up to 500 if they want longer slices, but the
        # auditor advances the cursor by this much per run, so multiple
        # runs cover a long log regardless.
        max_events_lbl = QLabel("max")
        max_events_lbl.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas; padding-left: 4px;"
        )
        toolbar.addWidget(max_events_lbl)
        self._max_events_spin = QSpinBox()
        self._max_events_spin.setRange(10, 500)
        self._max_events_spin.setValue(50)
        self._max_events_spin.setFixedWidth(64)
        self._max_events_spin.setFixedHeight(24)
        self._max_events_spin.setStyleSheet(
            f"QSpinBox {{ background: {_s.BG_INPUT}; color: {_s.FG_TEXT};"
            f" border: 1px solid {_s.BORDER_DARK}; border-radius: 3px;"
            f" padding: 2px 4px; font-size: 10px; font-family: Consolas; }}"
        )
        self._max_events_spin.setToolTip(
            "Max canonical-log events the auditor reads per run.\n"
            "Smaller = less context per LLM call (less hallucination risk,\n"
            "faster), but more runs to catch up on a long log. Default 50."
        )
        self._max_events_spin.valueChanged.connect(self._on_max_events_changed)
        toolbar.addWidget(self._max_events_spin)
        toolbar.addStretch(1)

        self._decider_label = QLabel(self._decider_id)
        self._decider_label.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas;"
        )
        self._decider_label.setToolTip(f"Decisions recorded as: {self._decider_id}")
        toolbar.addWidget(self._decider_label)
        root.addLayout(toolbar)

        # Initial sync — populates model label, sets toggle state + enabled.
        self._sync_auditor_controls()

        self._view_buttons: list[MonoButton] = []
        self._view_group = QButtonGroup(self)
        self._view_group.setExclusive(True)
        tabs = QHBoxLayout()
        tabs.setSpacing(4)
        for idx, label in enumerate(("ALL ACUs", "AUDIT LOG", "REVIEW", "LLM OUTPUT")):
            btn = MonoButton(label, accent=(idx == 2))
            btn.setCheckable(True)
            btn.setFixedHeight(24)
            btn.clicked.connect(lambda checked=False, i=idx: self._show_view(i))
            self._view_group.addButton(btn, idx)
            self._view_buttons.append(btn)
            tabs.addWidget(btn)
        root.addLayout(tabs)

        self._view_stack = QStackedWidget()
        root.addWidget(self._view_stack, stretch=1)

        acus_page = QWidget()
        acus_layout = QVBoxLayout(acus_page)
        acus_layout.setContentsMargins(0, 0, 0, 0)
        acus_layout.setSpacing(0)
        self._acus_list = QListWidget()
        self._acus_list.setFrameShape(QListWidget.NoFrame)
        self._acus_list.setStyleSheet(self._list_style(font_size=10, item_padding=4))
        self._acus_list.itemSelectionChanged.connect(self._on_acu_selection_changed)
        acus_layout.addWidget(self._acus_list, 1)
        self._acu_detail = QTextEdit()
        self._acu_detail.setReadOnly(True)
        self._acu_detail.setFrameShape(QTextEdit.NoFrame)
        self._acu_detail.setStyleSheet(
            f"QTextEdit {{ background: {_s.BG_INPUT}; color: {_s.FG_TEXT};"
            f" font-family: Consolas, monospace; font-size: 10px;"
            f" border: 1px solid {_s.BORDER_SUBTLE}; border-radius: 3px;"
            f" padding: 6px; }}"
        )
        self._acu_detail.setPlainText("Select an ACU to inspect its provenance.")
        acus_layout.addWidget(self._acu_detail, 1)
        self._view_stack.addWidget(acus_page)

        audit_page = QWidget()
        audit_layout = QVBoxLayout(audit_page)
        audit_layout.setContentsMargins(0, 0, 0, 0)
        audit_layout.setSpacing(4)

        # ── RECENT RUNS (collapsed by default) ───────────────────────────
        # Shows the last N audit runs with their stats. Useful for
        # answering "did the auditor actually run and why no candidates?"
        # — the proposals_returned vs. candidates_inserted split tells you
        # whether the LLM proposed nothing or whether the atomicity gate
        # filtered them out.
        self._runs_section = CollapsibleSection("RECENT RUNS")
        runs_inner = QVBoxLayout()
        runs_inner.setContentsMargins(0, 4, 0, 4)
        runs_inner.setSpacing(3)
        self._runs_list = QListWidget()
        self._runs_list.setFrameShape(QListWidget.NoFrame)
        self._runs_list.setMaximumHeight(140)
        self._runs_list.itemSelectionChanged.connect(self._on_run_selection_changed)
        self._runs_list.setStyleSheet(
            f"QListWidget {{ background: {_s.BG_INPUT}; border: 1px solid {_s.BORDER_SUBTLE};"
            f" border-radius: 3px; color: {_s.FG_TEXT}; font-family: Consolas; font-size: 10px; }}"
            f"QListWidget::item {{ padding: 4px 6px; }}"
            f"QListWidget::item:selected {{ background: {_s.ACCENT_PRIMARY}; color: {_s.BG_MAIN}; }}"
        )
        runs_inner.addWidget(self._runs_list)
        self._run_detail = QTextEdit()
        self._run_detail.setReadOnly(True)
        self._run_detail.setFrameShape(QTextEdit.NoFrame)
        self._run_detail.setMaximumHeight(132)
        self._run_detail.setStyleSheet(self._detail_style(font_size=10, padding=6))
        self._run_detail.setPlainText("Select a run to inspect what happened.")
        runs_inner.addWidget(self._run_detail)
        self._runs_empty_label = QLabel("(no runs yet — toggle Auditor on and click Audit)")
        self._runs_empty_label.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas; padding: 2px 4px;"
        )
        runs_inner.addWidget(self._runs_empty_label)
        self._runs_section.set_content_layout(runs_inner)
        self._runs_section.btn_toggle.hide()
        self._runs_section.content_area.setMaximumHeight(230)
        audit_layout.addWidget(self._runs_section)

        # ── AUDIT LOG (live tail, collapsed by default) ──────────────────
        # Newest auditor_* canonical-log events. Refreshes on the same
        # cadence as the rest of the panel. Lets the user SEE what the
        # auditor is doing right now (run_started → cursor_advance →
        # candidate_emitted → atomicity_reject → run_complete) instead of
        # staring at a spinner-less panel.
        self._log_section = CollapsibleSection("AUDIT LOG")
        log_inner = QVBoxLayout()
        log_inner.setContentsMargins(0, 4, 0, 4)
        log_inner.setSpacing(3)
        self._log_list = QListWidget()
        self._log_list.setFrameShape(QListWidget.NoFrame)
        self._log_list.setMinimumHeight(170)
        self._log_list.itemSelectionChanged.connect(self._on_log_selection_changed)
        self._log_list.setStyleSheet(
            f"QListWidget {{ background: {_s.BG_INPUT}; border: 1px solid {_s.BORDER_SUBTLE};"
            f" border-radius: 3px; color: {_s.FG_TEXT}; font-family: Consolas; font-size: 9px; }}"
            f"QListWidget::item {{ padding: 2px 6px; }}"
            f"QListWidget::item:selected {{ background: {_s.ACCENT_PRIMARY}; color: {_s.BG_MAIN}; }}"
        )
        log_inner.addWidget(self._log_list, 1)
        self._log_detail = QTextEdit()
        self._log_detail.setReadOnly(True)
        self._log_detail.setFrameShape(QTextEdit.NoFrame)
        self._log_detail.setMinimumHeight(130)
        self._log_detail.setStyleSheet(self._detail_style(font_size=10, padding=6))
        self._log_detail.setPlainText("Select an audit event to inspect its payload.")
        log_inner.addWidget(self._log_detail, 1)
        self._log_section.set_content_layout(log_inner)
        self._log_section.btn_toggle.hide()
        self._log_section.content_area.setMaximumHeight(16777215)
        self._log_section.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._log_section.content_area.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        audit_layout.addWidget(self._log_section, stretch=1)
        self._view_stack.addWidget(audit_page)

        # Splitter — vertical (list on top, evidence below) so the panel fits
        # the narrow companion-pane width. Horizontal would clip both panes.
        review_page = QWidget()
        review_layout = QVBoxLayout(review_page)
        review_layout.setContentsMargins(0, 0, 0, 0)
        review_layout.setSpacing(4)
        self._acu_writes_list = QListWidget()
        self._acu_writes_list.setFrameShape(QListWidget.NoFrame)
        self._acu_writes_list.setMaximumHeight(66)
        self._acu_writes_list.setStyleSheet(self._list_style(font_size=9, item_padding=3))
        review_layout.addWidget(self._acu_writes_list)
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)
        review_layout.addWidget(splitter, stretch=1)

        # Top: candidate list
        self._list = QListWidget()
        self._list.setFrameShape(QListWidget.NoFrame)
        self._list.setStyleSheet(self._list_style(font_size=10, item_padding=4))
        self._list.itemSelectionChanged.connect(self._on_selection_changed)
        splitter.addWidget(self._list)

        # Bottom: evidence + decision buttons
        bottom = QWidget()
        bottom_layout = QVBoxLayout(bottom)
        bottom_layout.setContentsMargins(0, 6, 0, 0)
        bottom_layout.setSpacing(4)

        self._evidence = QTextEdit()
        self._evidence.setReadOnly(True)
        self._evidence.setFrameShape(QTextEdit.NoFrame)
        self._evidence.setStyleSheet(
            f"QTextEdit {{ background: {_s.BG_INPUT}; color: {_s.FG_TEXT};"
            f" font-family: Consolas, monospace; font-size: 10px;"
            f" border: 1px solid {_s.BORDER_SUBTLE}; border-radius: 3px;"
            f" padding: 4px; }}"
        )
        bottom_layout.addWidget(self._evidence, stretch=1)

        decision_row = QHBoxLayout()
        decision_row.setSpacing(6)
        self._accept_btn = MonoButton("Accept", accent=True)
        self._accept_btn.setFixedHeight(24)
        self._accept_btn.clicked.connect(self._on_accept)
        self._accept_btn.setEnabled(False)
        decision_row.addWidget(self._accept_btn)

        self._reject_btn = MonoButton("Reject")
        self._reject_btn.setFixedHeight(24)
        self._reject_btn.clicked.connect(self._on_reject)
        self._reject_btn.setEnabled(False)
        decision_row.addWidget(self._reject_btn)

        self._auto_review_btn = MonoButton("Auto Review")
        self._auto_review_btn.setFixedHeight(24)
        self._auto_review_btn.setToolTip(
            "Auto accept safe auditor candidates and auto reject stale invalid ones.\n"
            "Uses agent_<auditor> deciders, not user_e."
        )
        self._auto_review_btn.clicked.connect(self._on_auto_review)
        self._auto_review_btn.setEnabled(False)
        decision_row.addWidget(self._auto_review_btn)
        decision_row.addStretch(1)
        bottom_layout.addLayout(decision_row)

        splitter.addWidget(bottom)
        # Roughly equal split; vertical so width isn't a constraint.
        splitter.setSizes([200, 300])

        self._view_stack.addWidget(review_page)

        llm_page = QWidget()
        llm_layout = QVBoxLayout(llm_page)
        llm_layout.setContentsMargins(0, 0, 0, 0)
        llm_layout.setSpacing(0)
        self._llm_output = QTextEdit()
        self._llm_output.setReadOnly(True)
        self._llm_output.setFrameShape(QTextEdit.NoFrame)
        self._llm_output.setStyleSheet(
            f"QTextEdit {{ background: {_s.BG_INPUT}; color: {_s.FG_TEXT};"
            f" font-family: Consolas, monospace; font-size: 10px;"
            f" border: 1px solid {_s.BORDER_SUBTLE}; border-radius: 3px;"
            f" padding: 6px; }}"
        )
        llm_layout.addWidget(self._llm_output)
        self._view_stack.addWidget(llm_page)
        self._show_view(2)

        # Status bar
        self._status = QStatusBar()
        self._status.setStyleSheet(
            f"QStatusBar {{ color: {_s.FG_DIM}; font-size: 9px; font-family: Consolas; }}"
        )
        root.addWidget(self._status)

    def _list_style(self, *, font_size: int, item_padding: int) -> str:
        return (
            f"QListWidget {{ background: {_s.BG_INPUT}; border: 1px solid {_s.BORDER_SUBTLE};"
            f" border-radius: 3px; color: {_s.FG_TEXT}; font-family: Consolas;"
            f" font-size: {font_size}px; }}"
            f"QListWidget::item {{ padding: {item_padding}px 6px; }}"
            f"QListWidget::item:selected {{ background: {_s.ACCENT_PRIMARY}; color: {_s.BG_MAIN}; }}"
        )

    def _detail_style(self, *, font_size: int, padding: int) -> str:
        return (
            f"QTextEdit {{ background: {_s.BG_INPUT}; color: {_s.FG_TEXT};"
            f" font-family: Consolas, monospace; font-size: {font_size}px;"
            f" border: 1px solid {_s.BORDER_SUBTLE}; border-radius: 3px;"
            f" padding: {padding}px; }}"
        )

    def _show_view(self, idx: int) -> None:
        if not hasattr(self, "_view_stack"):
            return
        self._view_stack.setCurrentIndex(idx)
        for i, btn in enumerate(self._view_buttons):
            btn.setChecked(i == idx)
            if i == idx:
                btn.setStyleSheet(
                    f"background: {_s.ACCENT_PRIMARY}; color: {_s.BG_MAIN};"
                    f" border: 1px solid {_s.ACCENT_PRIMARY}; border-radius: 3px;"
                    " font-size: 9px; font-family: Consolas; font-weight: bold;"
                )
            else:
                btn.setStyleSheet(
                    f"background: {_s.BG_INPUT}; color: {_s.FG_DIM};"
                    f" border: 1px solid {_s.BORDER_SUBTLE}; border-radius: 3px;"
                    " font-size: 9px; font-family: Consolas; font-weight: bold;"
                )

    # ── data refresh ─────────────────────────────────────────────────

    def _refresh_all(self) -> None:
        self._refresh_pending_list()
        self._refresh_status()
        self._refresh_acus_list()
        self._refresh_runs_list()
        self._refresh_log_tail()
        self._refresh_llm_output()

    def _refresh_log_tail(self) -> None:
        """Repopulate the AUDIT LOG list from the canonical-log tail.
        Filters to auditor_* event kinds. Newest at the top — easier to
        watch the live stream than scrolling to find the bottom."""
        prev_scroll = self._log_list.verticalScrollBar().value()
        sel_items = self._log_list.selectedItems()
        prev_sel = sel_items[0].data(Qt.ItemDataRole.UserRole) if sel_items else None
        try:
            events = _auditor.read_audit_log_tail(limit=50)
        except Exception as exc:
            self._log_list.clear()
            self._log_list.addItem(QListWidgetItem(f"log read failed: {exc}"))
            return

        self._log_list.blockSignals(True)
        self._log_list.clear()
        if not events:
            self._log_list.blockSignals(False)
            self._log_list.addItem(QListWidgetItem("(no auditor activity in log yet)"))
            self._on_log_selection_changed()
            return

        import time as _time
        now = _time.time()
        restore_row = -1
        for ev in events:
            kind = ev.get("kind") or "?"
            payload = ev.get("payload") or {}
            ts = float(ev.get("ts") or 0)
            ago = max(0, int(now - ts))
            # Compact one-liner per kind so the user can read the live stream
            # without expanding each row. Tooltip carries the full payload.
            if kind == "auditor_run_started":
                slice_start = payload.get('slice_start_event_id', '?')
                slice_end = payload.get('slice_end_event_id', '?')
                max_evs = payload.get('max_events_per_run')
                cap_suffix = f"  max={max_evs}" if max_evs else ""
                detail = (
                    f"slice {slice_start}→{slice_end}{cap_suffix}"
                    f"  src={payload.get('source', '?')}"
                )
            elif kind == "auditor_llm_call_started":
                detail = (
                    f"prompt={payload.get('prompt_chars', 0)}c"
                    f"  user={payload.get('user_chars', 0)}c"
                    f"  events={payload.get('events_in_slice', 0)}"
                    f"  (LLM call started — usually 5-180s)"
                )
            elif kind == "auditor_llm_call_returned":
                elapsed_s = payload.get('elapsed_secs', 0)
                status = payload.get('status', '?')
                if status == "ok":
                    detail = (
                        f"{elapsed_s}s  response={payload.get('response_chars', 0)}c"
                    )
                else:
                    err = str(payload.get('error') or '?')[:60]
                    detail = f"{elapsed_s}s  ERROR  {err}"
            elif kind == "auditor_run_complete":
                detail = (
                    f"status={payload.get('status', '?')}"
                    f"  prop={payload.get('proposals_returned', 0)}"
                    f"  ok={payload.get('candidates_inserted', 0)}"
                    f"  rej={payload.get('candidates_rejected', 0)}"
                )
            elif kind == "auditor_run_failed":
                err = str(payload.get('error') or '?')[:60]
                detail = f"error={err}"
            elif kind == "auditor_cursor_advance":
                reset_marker = " RESET" if payload.get("reset") else ""
                detail = f"cursor→{payload.get('cursor_value', 0)}{reset_marker}"
            elif kind == "auditor_atomicity_reject":
                form = str(payload.get('canonical_form') or '')[:40]
                reason = str(payload.get('reason') or '')[:30]
                detail = f'"{form}" ({reason})'
            elif kind == "candidate_emitted":
                form = str(payload.get('canonical_form') or '')[:50]
                detail = f'"{form}"'
            else:
                detail = ""

            line = f"{ago:>3}s  {kind:<28}  {detail}"
            item = QListWidgetItem(line)
            item.setData(Qt.ItemDataRole.UserRole, int(ev.get("event_id") or 0))
            import json as _json
            try:
                item.setToolTip(_json.dumps(payload, indent=2, ensure_ascii=False))
            except Exception:
                item.setToolTip(str(payload))
            self._log_list.addItem(item)
            if prev_sel is not None and int(ev.get("event_id") or 0) == prev_sel:
                restore_row = self._log_list.count() - 1
        if restore_row >= 0:
            self._log_list.setCurrentRow(restore_row)
        self._log_list.blockSignals(False)
        self._on_log_selection_changed()
        self._restore_scroll(self._log_list, prev_scroll)

    def _refresh_acus_list(self) -> None:
        selected_id = self._selected_acu_id()
        prev_scroll = self._acus_list.verticalScrollBar().value()
        self._acus_list.blockSignals(True)
        try:
            rows = _monobase_status.read_recent_acus(limit=500)
        except Exception as exc:
            self._acus_list.clear()
            self._acus_list.addItem(QListWidgetItem(f"ACU read failed: {exc}"))
            self._acus_list.blockSignals(False)
            return

        self._acus_list.clear()
        if not rows:
            self._acus_list.addItem(QListWidgetItem("(no ACUs written yet)"))
            self._acu_detail.setPlainText("No ACUs have been written yet.")
            self._rendered_acu_id = None
            self._acus_list.blockSignals(False)
            return

        restore_row = -1
        for acu in rows:
            line = (
                f"#{acu.id:<5} seen={acu.reinforcement:<2} v={acu.veracity:.1f} "
                f"{acu.source}  {acu.canonical}"
            )
            item = QListWidgetItem(line)
            item.setData(Qt.ItemDataRole.UserRole, acu.id)
            item.setToolTip(
                "\n".join(
                    [
                        f"acu_id: {acu.id}",
                        f"source: {acu.source}",
                        f"created_at: {acu.created_at}",
                        f"last_seen: {acu.last_seen}",
                        f"candidate_id: {acu.candidate_id}",
                        f"decision_id: {acu.decision_id}",
                        "",
                        acu.canonical,
                    ]
                )
            )
            self._acus_list.addItem(item)
            if selected_id == acu.id:
                restore_row = self._acus_list.count() - 1
        self._acus_list.blockSignals(False)
        if restore_row >= 0:
            self._acus_list.setCurrentRow(restore_row)
            self._on_acu_selection_changed()
        elif selected_id is not None:
            self._acu_detail.setPlainText("Select an ACU to inspect its provenance.")
            self._rendered_acu_id = None
        self._restore_scroll(self._acus_list, prev_scroll)

    def _refresh_llm_output(self) -> None:
        lines: list[str] = []
        try:
            in_flight = _auditor.current_in_flight_run()
        except Exception:
            in_flight = None

        if in_flight is not None:
            run_id = in_flight.get("event_id") or "?"
            slice_start = in_flight.get("slice_start_event_id", 0)
            slice_end = in_flight.get("slice_end_event_id", 0)
            lines.extend(
                [
                    f"Current run: #{run_id}  slice {slice_start}-{slice_end}",
                    "Status: waiting for the auditor LLM response",
                    "",
                ]
            )

        try:
            output = _monobase_status.read_latest_llm_output()
        except Exception as exc:
            self._set_llm_text(f"LLM output read failed: {exc}")
            return

        if output is None:
            lines.append("No auditor LLM output has been recorded yet.")
            self._set_llm_text("\n".join(lines))
            return

        elapsed = (
            _monobase_status.format_elapsed(output.elapsed_secs)
            if output.elapsed_secs is not None
            else "--"
        )
        chars = output.response_chars if output.response_chars is not None else 0
        lines.extend(
            [
                f"Latest returned output: event #{output.event_id}  run #{output.run_id or '?'}",
                f"status={output.status or '?'}  elapsed={elapsed}  chars={chars}",
            ]
        )
        if output.error:
            lines.append(f"error={output.error}")
        if output.response_truncated:
            lines.append("(preview truncated)")
        lines.append("")
        lines.append(output.response_preview or "(no response preview recorded for this run)")
        self._set_llm_text("\n".join(lines))

    def _refresh_runs_list(self) -> None:
        """Repopulate the RECENT RUNS list from the auditor's run-summary
        events in the canonical log. Newest first. Each row encodes the
        result inline so the user can answer 'did it run?' at a glance."""
        prev_scroll = self._runs_list.verticalScrollBar().value()
        sel_items = self._runs_list.selectedItems()
        prev_sel = sel_items[0].data(Qt.ItemDataRole.UserRole) if sel_items else None
        try:
            runs = _auditor.read_recent_runs(limit=15)
        except Exception as exc:
            self._runs_list.clear()
            self._runs_empty_label.setText(f"runs read failed: {exc}")
            self._runs_empty_label.setVisible(True)
            return

        self._runs_list.blockSignals(True)
        self._runs_list.clear()
        if not runs:
            self._runs_list.blockSignals(False)
            self._runs_empty_label.setText("(no runs yet — toggle Auditor on and click Audit)")
            self._runs_empty_label.setVisible(True)
            self._on_run_selection_changed()
            return

        self._runs_empty_label.setVisible(False)
        restore_row = -1
        for run in runs:
            run_id = run.get("run_id") or 0
            status = run.get("status") or "?"
            events = run.get("events_processed") or 0
            proposals = run.get("proposals_returned") or 0
            inserted = run.get("candidates_inserted") or 0
            rejected = run.get("candidates_rejected") or 0
            error = run.get("error")

            if error:
                summary = f"#{run_id}  FAILED  {error[:60]}"
            elif status == "empty_slice":
                summary = f"#{run_id}  empty_slice  (nothing new since last cursor)"
            else:
                summary = (
                    f"#{run_id}  {status}  ev={events}"
                    f"  prop={proposals}  ok={inserted}  rej={rejected}"
                )
            item = QListWidgetItem(summary)
            item.setData(Qt.ItemDataRole.UserRole, int(run_id))
            # Tooltip carries the fuller picture so the user can hover for
            # the breakdown without needing a click-through detail panel.
            tooltip_lines = [
                f"run_id: {run_id}",
                f"status: {status}",
                f"events_processed: {events}",
                f"proposals_returned: {proposals}",
                f"candidates_inserted: {inserted}",
                f"candidates_rejected: {rejected}",
            ]
            if error:
                tooltip_lines.append(f"error: {error}")
            item.setToolTip("\n".join(tooltip_lines))
            self._runs_list.addItem(item)
            if prev_sel is not None and int(run_id) == prev_sel:
                restore_row = self._runs_list.count() - 1
        if restore_row >= 0:
            self._runs_list.setCurrentRow(restore_row)
        self._runs_list.blockSignals(False)
        self._on_run_selection_changed()
        self._restore_scroll(self._runs_list, prev_scroll)

    def _on_reset_cursor(self) -> None:
        """Reset the auditor cursor to 0 so the next run audits the entire
        canonical log from event 1. Confirms first — re-auditing a long
        log can be slow and expensive on a cloud LLM."""
        latest = 0
        try:
            latest = _canonical_log.latest_event_id()
        except Exception:
            pass
        confirm = QMessageBox.question(
            self,
            "Reset auditor cursor",
            (
                f"Reset cursor to 0?\n\n"
                f"The next audit run will re-process the full canonical log "
                f"({latest} events). This can be slow on a cloud LLM and "
                f"may produce duplicate candidates for events already audited."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            _auditor.reset_cursor()
            self._status.showMessage("cursor reset to 0 — next audit re-runs from start", 5000)
        except Exception as exc:
            self._status.showMessage(f"reset failed: {type(exc).__name__}: {exc}", 6000)
        self._refresh_all()

    def _refresh_pending_list(self) -> None:
        selected_id = self._selected_candidate_id()
        self._list.blockSignals(True)
        self._list.clear()
        try:
            pending = _candidates.read_pending(limit=200)
        except Exception as exc:
            self._list.blockSignals(False)
            self._auto_review_btn.setEnabled(False)
            QMessageBox.warning(self, "Read failed", f"read_pending error:\n{exc}")
            return
        self._auto_review_btn.setEnabled(bool(pending))
        restore_item: QListWidgetItem | None = None
        for cand in pending:
            item = QListWidgetItem(
                _LIST_ROW_FMT.format(id=cand.id, canonical=cand.canonical_form)
            )
            item.setData(Qt.ItemDataRole.UserRole, cand.id)
            self._list.addItem(item)
            if selected_id == cand.id:
                restore_item = item
        self._list.blockSignals(False)
        if restore_item is not None:
            self._list.setCurrentItem(restore_item)
            self._on_selection_changed()
            return
        if selected_id is None and self._list.count() > 0:
            self._list.setCurrentRow(0)
            self._on_selection_changed()
            return
        # Clear evidence pane if nothing selected after refresh
        if self._list.count() == 0 or selected_id is not None:
            self._evidence.clear()
            self._rendered_cid = None
            self._accept_btn.setEnabled(False)
            self._reject_btn.setEnabled(False)

    # ── auditor lifecycle (lazy-create from current LLMConfig) ──────────

    def _probe_sidecar(self) -> tuple[bool, str, str]:
        """Read the current LLMConfig and try to validate it for the
        auditor sidecar. Returns (ok, model_name, reason_or_status).

        On success: ok=True, model_name="deepseek-v4-pro" (or similar),
            reason_or_status="" (no error).
        On failure: ok=False, model_name="" (no usable model),
            reason_or_status="cloud not configured / set api_base / etc."

        Does NOT actually construct an OpenAI client — just reads config
        and checks the same fields make_auditor_llm() requires. Cheap to
        call from a 250ms refresh loop.
        """
        try:
            from core.config import get_config
            cfg = get_config().llm
        except Exception as exc:
            return False, "", f"config read failed: {exc}"

        backend = str(getattr(cfg, "backend", "") or "").strip().lower()
        api_base = str(getattr(cfg, "api_base", "") or "").strip()
        api_model = str(getattr(cfg, "api_model", "") or "").strip()

        # Same logic as make_auditor_llm(): need api_base + api_model. The
        # raw gguf backend without an api_base can't drive the sidecar.
        if backend == "gguf" and not api_base:
            return False, "", "gguf-only backend (no api_base) — switch to gguf_api or set api_base"
        if not api_base:
            return False, "", "no api_base set in LLMConfig"
        if not api_model:
            return False, "", "no api_model set in LLMConfig"
        return True, api_model, ""

    def _sync_auditor_controls(self) -> None:
        """Refresh the model label + toggle enabled state + checked state
        from the current LLMConfig and registered worker. Called on init,
        on each 250ms status refresh, and after any toggle action."""
        ok, model, reason = self._probe_sidecar()
        self._sidecar_ready = ok
        self._sidecar_block_reason = reason
        worker = _runtime.get_active_worker()
        running = bool(worker and getattr(worker, "_thread", None) and worker._thread.is_alive())

        if ok:
            self._model_label.setText(f"model: {model}")
            self._model_label.setStyleSheet(
                f"color: {_s.FG_TEXT}; font-size: 10px; font-family: Consolas; padding: 2px 4px;"
            )
        else:
            self._model_label.setText(f"model: ({reason})")
            self._model_label.setStyleSheet(
                f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas; padding: 2px 4px;"
            )

        # Toggle is enabled when we either have a running worker OR can
        # build one. Disabling only the truly impossible cases (no config).
        can_toggle = ok or worker is not None
        self._auditor_toggle.setEnabled(can_toggle)
        if can_toggle:
            self._auditor_toggle.setToolTip(
                "Toggle the background auditor on/off.\n"
                "When ON: produces ACU candidates automatically as events accumulate.\n"
                "When OFF: only the 'Audit' button triggers runs."
            )
        else:
            self._auditor_toggle.setToolTip(
                f"Cannot start auditor: {reason}\n"
                "Set api_base + api_model in LLMConfig and re-open this panel."
            )

        # Reflect actual running state in the switch
        self._auditor_toggle.blockSignals(True)
        self._auditor_toggle.setChecked(running)
        self._auditor_toggle.blockSignals(False)

        # The Audit button needs a registered worker (its queue is what we
        # enqueue triggers on). Sync that, too.
        if worker is not None:
            self._trigger_queue = worker.queue_handle
            self._audit_btn.setEnabled(True)
            self._audit_btn.setToolTip("Enqueue a one-shot audit trigger on the worker.")
        elif ok:
            self._trigger_queue = None
            self._audit_btn.setEnabled(True)
            self._audit_btn.setToolTip("Start the auditor worker and enqueue one audit run.")
        else:
            self._trigger_queue = None
            self._audit_btn.setEnabled(False)
            self._audit_btn.setToolTip("Auditor sidecar unavailable.")

    def _on_auditor_toggled(self, checked: bool) -> None:
        """User flipped the Auditor switch.

        ON: if no worker is registered, build one from the current
            LLMConfig via make_auditor_llm() + AuditorWorker, register it,
            then start. If a worker IS registered, just call .start().
        OFF: call .stop() on the registered worker (keep registered so a
            subsequent ON is fast).
        """
        if checked:
            worker = _runtime.get_active_worker()
            if worker is None:
                # Lazy-create from the loaded model's config
                try:
                    from core.acatalepsy.llm_sidecar import make_auditor_llm
                    from core.acatalepsy.triggers import AuditorWorker
                    llm = make_auditor_llm()
                    # Pass the current spinbox value as the per-run slice
                    # cap so the very first run respects it. Subsequent
                    # spinbox changes propagate via _on_max_events_changed.
                    worker = AuditorWorker(
                        llm=llm,
                        source="auditor_monolith",
                        max_events_per_run=int(self._max_events_spin.value()),
                    )
                    _runtime.register_worker(worker)
                    self._status.showMessage("auditor created — using current model", 3000)
                except Exception as exc:
                    self._status.showMessage(
                        f"auditor create failed: {type(exc).__name__}: {exc}", 6000
                    )
                    self._sync_auditor_controls()
                    return
            try:
                worker.start()
                self._status.showMessage("auditor started", 3000)
            except Exception as exc:
                self._status.showMessage(
                    f"auditor start failed: {type(exc).__name__}: {exc}", 6000
                )
        else:
            worker = _runtime.get_active_worker()
            if worker is not None:
                try:
                    worker.stop()
                    self._status.showMessage("auditor stopped", 3000)
                except Exception as exc:
                    self._status.showMessage(
                        f"auditor stop failed: {type(exc).__name__}: {exc}", 6000
                    )
        self._sync_auditor_controls()

    def _refresh_status(self) -> None:
        try:
            snapshot = _monobase_status.build_monobase_snapshot()
        except Exception as exc:
            self._status_strip.show_error(f"status read failed: {exc}")
            self._status.showMessage(f"status read failed: {exc}", 5000)
            return

        self._apply_status_snapshot(snapshot)
        return

    def _apply_status_snapshot(
        self,
        snapshot: _monobase_status.MonobaseSnapshot,
    ) -> None:
        self._status_strip.apply_snapshot(snapshot)
        self._refresh_acatalepsy_map(snapshot)
        self._refresh_acu_writes_list(snapshot)

        active_phase = snapshot.phase in {
            "auditing_log",
            "calling_llm",
            "updating_candidates",
            "queued",
            "stopping",
        }
        target_interval = 500 if active_phase else 2000
        if self._refresh_timer.interval() != target_interval:
            self._refresh_timer.setInterval(target_interval)

        is_in_flight = snapshot.in_flight_run is not None
        if self._was_in_flight and not is_in_flight:
            self._status.showMessage("audit finished - check Recent Runs for outcome", 4000)
        elif is_in_flight and not self._was_in_flight:
            self._status.showMessage("audit started - live status is pinned above", 3000)
        self._was_in_flight = is_in_flight

        if is_in_flight:
            run = snapshot.in_flight_run or {}
            slice_start = run.get("slice_start_event_id", 0)
            slice_end = run.get("slice_end_event_id", 0)
            self._scan_progress.setText(
                f"scan: slice {slice_start}-{slice_end} | "
                f"cursor {self._event_id_text(snapshot.cursor)} / "
                f"latest {self._event_id_text(snapshot.latest_event_id)}"
            )
            self._scan_progress.setStyleSheet(
                f"color: {_s.BG_MAIN}; font-size: 11px; font-family: Consolas;"
                f" font-weight: bold; padding: 4px;"
                f" background: {_s.ACCENT_PRIMARY};"
                f" border: 1px solid {_s.ACCENT_PRIMARY}; border-radius: 3px;"
            )
        else:
            self._scan_progress.setText(
                f"scan: cursor {self._event_id_text(snapshot.cursor)} / latest "
                f"{self._event_id_text(snapshot.latest_event_id)}  "
                f"behind={snapshot.pending_log_events}"
            )
            self._scan_progress.setStyleSheet(
                f"color: {_s.FG_TEXT}; font-size: 11px; font-family: Consolas;"
                f" font-weight: bold; padding: 4px;"
                f" background: {_s.BG_INPUT};"
                f" border: 1px solid {_s.BORDER_SUBTLE}; border-radius: 3px;"
            )

        self._sync_auditor_controls()
        if is_in_flight or snapshot.phase == "stopping":
            self._audit_btn.setText("Stop")
            self._audit_btn.setToolTip(
                "Cancel the current audit run. The Python thread cannot kill "
                "an LLM call mid-request, so the worker may still finish returning."
            )
        else:
            worker = _runtime.get_active_worker()
            if worker is None and getattr(self, "_sidecar_ready", False):
                self._audit_btn.setText("Start + Audit")
            else:
                self._audit_btn.setText("Audit")
            if self._trigger_queue is not None:
                self._audit_btn.setToolTip("Enqueue a one-shot audit trigger on the worker.")
            elif getattr(self, "_sidecar_ready", False):
                self._audit_btn.setToolTip("Start the auditor worker and enqueue one audit run.")
            else:
                reason = getattr(self, "_sidecar_block_reason", "")
                suffix = f" ({reason})" if reason else ""
                self._audit_btn.setToolTip(f"Auditor sidecar unavailable{suffix}.")

    def _event_id_text(self, value: int | None) -> str:
        return "--" if value is None else str(int(value))

    def _refresh_acatalepsy_map(
        self,
        snapshot: _monobase_status.MonobaseSnapshot,
    ) -> None:
        groups: dict[str, list[AcatalepsyMapDot]] = {
            "log": [],
            "candidate": [],
            "run": [],
            "acu": [],
        }
        totals: dict[str, int] = {
            "log": snapshot.pending_log_events,
            "candidate": snapshot.pending_candidate_count,
            "run": len(snapshot.recent_runs),
            "acu": len(snapshot.recent_acu_writes),
        }

        if snapshot.pending_log_events > 0:
            groups["log"].append(
                AcatalepsyMapDot(
                    kind="log",
                    item_id=snapshot.latest_event_id,
                    label=f"{snapshot.pending_log_events} unaudited event(s)",
                    tone="warn",
                    tooltip=(
                        f"Cursor {snapshot.cursor}; latest {snapshot.latest_event_id}; "
                        f"{snapshot.pending_log_events} event(s) waiting for audit."
                    ),
                )
            )
        elif snapshot.in_flight_run is not None:
            run_id = int(snapshot.in_flight_run.get("event_id") or 0)
            groups["log"].append(
                AcatalepsyMapDot(
                    kind="run",
                    item_id=run_id,
                    label=f"run #{run_id}",
                    tone="active",
                    tooltip="Current audit run is reading the canonical log.",
                )
            )

        try:
            pending = _candidates.read_pending(limit=24)
        except Exception:
            pending = []
        for cand in pending:
            groups["candidate"].append(
                AcatalepsyMapDot(
                    kind="candidate",
                    item_id=cand.id,
                    label=f"candidate #{cand.id}",
                    tone="warn",
                    tooltip="\n".join(
                        [
                            f"candidate #{cand.id}",
                            cand.canonical_form,
                            "",
                            cand.reason,
                        ]
                    ),
                )
            )

        for run in snapshot.recent_runs[:12]:
            run_id = int(run.get("run_id") or 0)
            status = str(run.get("status") or "?")
            error = run.get("error")
            tone = "error" if error else ("ok" if status == "success" else "idle")
            groups["run"].append(
                AcatalepsyMapDot(
                    kind="run",
                    item_id=run_id,
                    label=f"run #{run_id}",
                    tone=tone,
                    tooltip=(
                        f"run #{run_id}\n"
                        f"status: {status}\n"
                        f"events: {run.get('events_processed') or 0}\n"
                        f"proposals: {run.get('proposals_returned') or 0}\n"
                        f"inserted: {run.get('candidates_inserted') or 0}\n"
                        f"rejected: {run.get('candidates_rejected') or 0}"
                    ),
                )
            )

        try:
            acus = _monobase_status.read_recent_acus(limit=24)
        except Exception:
            acus = []
        totals["acu"] = len(acus)
        for acu in acus:
            groups["acu"].append(
                AcatalepsyMapDot(
                    kind="acu",
                    item_id=acu.id,
                    label=f"ACU #{acu.id}",
                    tone="ok",
                    tooltip="\n".join(
                        [
                            f"ACU #{acu.id}",
                            f"source: {acu.source}",
                            f"seen: {acu.reinforcement}",
                            "",
                            acu.canonical,
                        ]
                    ),
                )
            )

        summary = (
            f"{snapshot.pending_log_events} behind | "
            f"{snapshot.pending_candidate_count} pending | "
            f"{totals['acu']} ACUs"
        )
        self._acu_map.apply_map(groups, totals=totals, summary=summary)

    def _on_map_dot_activated(self, kind: str, item_id: int) -> None:
        if kind == "candidate":
            self._show_view(2)
            if self._select_item_by_id(self._list, item_id):
                self._on_selection_changed()
            return
        if kind == "acu":
            self._show_view(0)
            self._select_item_by_id(self._acus_list, item_id)
            return
        if kind == "run":
            self._show_view(1)
            self._select_item_by_id(self._runs_list, item_id)
            return
        if kind == "log":
            self._show_view(1)
            if not self._select_item_by_id(self._log_list, item_id):
                self._log_detail.setPlainText(
                    "\n".join(
                        [
                            f"Canonical log event #{item_id}",
                            "",
                            "This event is waiting for an auditor run.",
                            "Use Start + Audit to have MonoBase inspect the pending slice.",
                        ]
                    )
                )
            return
        self._show_view(1)

    def _select_item_by_id(self, list_widget: QListWidget, item_id: int) -> bool:
        for row in range(list_widget.count()):
            item = list_widget.item(row)
            if int(item.data(Qt.ItemDataRole.UserRole) or -1) == int(item_id):
                list_widget.setCurrentRow(row)
                return True
        return False

    def _refresh_acu_writes_list(
        self,
        snapshot: _monobase_status.MonobaseSnapshot,
    ) -> None:
        prev_scroll = self._acu_writes_list.verticalScrollBar().value()
        self._acu_writes_list.clear()
        if not snapshot.recent_acu_writes:
            self._acu_writes_list.addItem(
                QListWidgetItem("ACU writes: none yet - Accept/Edit creates an ACU")
            )
            return

        for write in snapshot.recent_acu_writes:
            item = QListWidgetItem(
                _monobase_status.format_recent_acu_write(write, now=snapshot.now)
            )
            tooltip_lines = [
                f"event_id: {write.event_id}",
                f"kind: {write.kind}",
                f"candidate_id: {write.candidate_id}",
                f"decision_id: {write.decision_id}",
                f"acu_id: {write.acu_id}",
                f"decided_by: {write.decided_by or '--'}",
            ]
            if write.canonical_form:
                tooltip_lines.extend(["", write.canonical_form])
            item.setToolTip("\n".join(tooltip_lines))
            self._acu_writes_list.addItem(item)
        self._restore_scroll(self._acu_writes_list, prev_scroll)

    # ── selection / details ──────────────────────────────────────────

    @staticmethod
    def _restore_scroll(list_widget: QListWidget, value: int) -> None:
        """Re-apply a captured vertical scroll position after a list rebuild.
        QListWidget.clear() resets the scrollbar to the top, so list scroll is
        saved before the rebuild and restored after (clamped to the new max)."""
        bar = list_widget.verticalScrollBar()
        bar.setValue(max(0, min(value, bar.maximum())))

    def _set_llm_text(self, text: str) -> None:
        """Set the LLM-output pane only when the text actually changed, so the
        2s refresh doesn't reset the scrollbar while the output is unchanged."""
        if self._last_llm_output == text:
            return
        self._last_llm_output = text
        self._llm_output.setPlainText(text)

    def _on_run_selection_changed(self, *, force: bool = False) -> None:
        items = self._runs_list.selectedItems()
        if not items:
            self._run_detail.setPlainText("Select a run to inspect what happened.")
            self._rendered_run_id = None
            return
        item = items[0]
        run_id = item.data(Qt.ItemDataRole.UserRole)
        if run_id == self._rendered_run_id and not force:
            return
        details = item.toolTip().strip() or item.text()
        self._run_detail.setPlainText("Run summary\n\n" + details)
        self._rendered_run_id = run_id

    def _on_log_selection_changed(self, *, force: bool = False) -> None:
        items = self._log_list.selectedItems()
        if not items:
            self._log_detail.setPlainText("Select an audit event to inspect its payload.")
            self._rendered_log_id = None
            return
        item = items[0]
        event_id = item.data(Qt.ItemDataRole.UserRole)
        if event_id == self._rendered_log_id and not force:
            return
        payload = item.toolTip().strip() or "{}"
        lines = [
            f"Audit event #{event_id or '?'}",
            "",
            item.text(),
            "",
            "payload:",
            payload,
        ]
        self._log_detail.setPlainText("\n".join(lines))
        self._rendered_log_id = event_id

    def _selected_acu_id(self) -> int | None:
        items = self._acus_list.selectedItems()
        if not items:
            return None
        value = items[0].data(Qt.ItemDataRole.UserRole)
        return int(value) if value is not None else None

    def _on_acu_selection_changed(self, *, force: bool = False) -> None:
        acu_id = self._selected_acu_id()
        if acu_id is None:
            self._acu_detail.setPlainText("Select an ACU to inspect its provenance.")
            self._rendered_acu_id = None
            return
        # Same guard as the evidence pane: skip the re-render (which would reset
        # the scrollbar) when the same ACU is already shown.
        if acu_id == self._rendered_acu_id and not force:
            return
        try:
            acu = _monobase_status.read_acu(acu_id)
        except Exception as exc:
            self._acu_detail.setPlainText(f"ACU read failed: {type(exc).__name__}: {exc}")
            self._rendered_acu_id = None
            return
        if acu is None:
            self._acu_detail.setPlainText(f"ACU #{acu_id} not found.")
            self._rendered_acu_id = None
            return
        lines = [
            f"ACU #{acu.id}",
            "",
            "canonical:",
            acu.canonical,
            "",
            f"source:        {acu.source}",
            f"created_at:    {acu.created_at}",
            f"last_seen:     {acu.last_seen}",
            f"reinforcement: {acu.reinforcement}",
            f"candidate_id:  {acu.candidate_id}",
            f"decision_id:   {acu.decision_id}",
        ]
        self._acu_detail.setPlainText("\n".join(lines))
        self._rendered_acu_id = acu_id

    def _selected_candidate_id(self) -> int | None:
        items = self._list.selectedItems()
        if not items:
            return None
        return int(items[0].data(Qt.ItemDataRole.UserRole))

    def _on_selection_changed(self, *, force: bool = False) -> None:
        cid = self._selected_candidate_id()
        if cid is None:
            self._evidence.clear()
            self._rendered_cid = None
            self._accept_btn.setEnabled(False)
            self._reject_btn.setEnabled(False)
            return
        # Skip the re-render when the same candidate is already shown (the 2s
        # auto-refresh re-asserts the selection on every tick). Re-running
        # setPlainText() below would reset the evidence scrollbar to the top.
        # User-driven selection of a *different* candidate changes cid and
        # renders normally; pass force=True to bypass when content must update.
        if cid == self._rendered_cid and not force:
            return
        cand = _candidates.read_one(cid)
        if cand is None:
            self._evidence.setPlainText(f"candidate id={cid} not found")
            self._rendered_cid = None
            self._accept_btn.setEnabled(False)
            self._reject_btn.setEnabled(False)
            return
        # Build the detail view
        lines: list[str] = [
            f"=== candidate id={cand.id} ===",
            f"canonical_form: {cand.canonical_form}",
            f"source:         {cand.source}",
            f"state:          {cand.state}",
            f"created_at:     {cand.created_at}",
            f"auditor_run_id: {cand.auditor_run_id}",
            f"reinforcement_count: {cand.reinforcement_count}",
            f"contradicts_acu_id:  {cand.contradicts_acu_id}",
            "",
            "=== reason (auditor's why) ===",
            cand.reason,
            "",
            "=== evidence ===",
            f"log_id:    {cand.evidence_log_id}",
            f"char_span: [{cand.evidence_char_start}, {cand.evidence_char_end})",
            "",
            cand.evidence_span,
        ]
        self._evidence.setPlainText("\n".join(lines))
        self._rendered_cid = cid
        # Only allow decisions on pending candidates
        is_pending = cand.state == "pending"
        self._accept_btn.setEnabled(is_pending)
        self._reject_btn.setEnabled(is_pending)

    # ── decision actions ─────────────────────────────────────────────

    def _on_accept(self) -> None:
        cid = self._selected_candidate_id()
        if cid is None:
            return
        try:
            decision_id = _decisions.insert_decision(
                candidate_id=cid,
                decision="accept",
                decided_by=self._decider_id,
            )
        except _decisions.DecisionAuthorizationError as exc:
            QMessageBox.warning(self, "Not authorized", str(exc))
            return
        except Exception as exc:
            QMessageBox.warning(self, "Accept failed", f"{type(exc).__name__}: {exc}")
            return
        if self._on_decision is not None:
            try:
                self._on_decision(cid, "accept")
            except Exception:
                pass
        decision = _decisions.read_one(decision_id)
        acu_id = decision.resulting_acu_id if decision is not None else None
        self._refresh_all()
        if acu_id is not None:
            self._status.showMessage(
                f"accepted candidate {cid} -> decision {decision_id} -> ACU {acu_id}",
                5000,
            )
            return
        self._status.showMessage(
            f"accepted candidate {cid} → decision {decision_id}", 4000
        )

    def _on_reject(self) -> None:
        cid = self._selected_candidate_id()
        if cid is None:
            return
        reason, ok = QInputDialog.getText(
            self, "Reject reason", "Why are you rejecting this candidate?"
        )
        if not ok or not reason.strip():
            return
        try:
            decision_id = _decisions.insert_decision(
                candidate_id=cid,
                decision="reject",
                decided_by=self._decider_id,
                reject_reason=reason.strip(),
            )
        except _decisions.DecisionAuthorizationError as exc:
            QMessageBox.warning(self, "Not authorized", str(exc))
            return
        except Exception as exc:
            QMessageBox.warning(self, "Reject failed", f"{type(exc).__name__}: {exc}")
            return
        if self._on_decision is not None:
            try:
                self._on_decision(cid, "reject")
            except Exception:
                pass
        self._refresh_all()
        self._status.showMessage(
            f"rejected candidate {cid} → decision {decision_id}", 4000
        )

    # ── trigger ──────────────────────────────────────────────────────

    def _on_auto_review(self) -> None:
        pending_count = self._list.count()
        if pending_count <= 0:
            return
        confirm = QMessageBox.question(
            self,
            "Auto review pending candidates",
            (
                f"Auto-review up to {pending_count} pending candidate(s)?\n\n"
                "Safe auditor candidates will be accepted by their matching "
                "agent decider. Invalid candidates will be rejected. "
                "Contradictions and unauthorized sources will be skipped."
            ),
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if confirm != QMessageBox.Yes:
            return
        try:
            summary = _auto_review.review_pending(limit=pending_count)
        except Exception as exc:
            QMessageBox.warning(self, "Auto review failed", f"{type(exc).__name__}: {exc}")
            return
        if self._on_decision is not None:
            for item in summary.items:
                if item.action in ("accept", "reject"):
                    try:
                        self._on_decision(item.candidate_id, item.action)
                    except Exception:
                        pass
        self._refresh_all()
        self._status.showMessage(
            "auto review: "
            f"{summary.accepted} accepted, {summary.rejected} rejected, "
            f"{summary.skipped} skipped",
            7000,
        )

    def _on_audit_now(self) -> None:
        if self._trigger_queue is None:
            return
        enqueued = self._trigger_queue.enqueue("manual", enqueued_by=self._decider_id)
        if enqueued:
            self._status.showMessage("manual audit enqueued — refresh to see results", 3000)
        else:
            self._status.showMessage("dedup: manual trigger already enqueued recently", 3000)

    def _on_audit_or_stop_clicked(self) -> None:
        """Single button handler — decides Audit vs Stop based on whether
        a run is currently in flight. Click while idle = enqueue trigger;
        click while running = mark the run cancelled in the canonical log
        and signal the worker to stop pulling new triggers.

        Note: the in-flight LLM call can NOT be interrupted from Python.
        The status bar warns the user the call has to finish (or hit its
        timeout, default 180s) before the worker thread truly exits."""
        try:
            in_flight = _auditor.current_in_flight_run()
        except Exception:
            in_flight = None
        if in_flight is None:
            worker = _runtime.get_active_worker()
            if worker is None:
                if not getattr(self, "_sidecar_ready", False):
                    self._status.showMessage("auditor sidecar unavailable", 5000)
                    self._sync_auditor_controls()
                    return
                self._on_auditor_toggled(True)
                worker = _runtime.get_active_worker()
                if worker is None:
                    self._status.showMessage("auditor did not start", 5000)
                    self._sync_auditor_controls()
                    return
            thread = getattr(worker, "_thread", None) if worker is not None else None
            stop_event = getattr(worker, "_stop_event", None) if worker is not None else None
            if (
                thread is not None
                and thread.is_alive()
                and stop_event is not None
                and stop_event.is_set()
            ):
                self._status.showMessage(
                    "auditor is stopping - waiting for current LLM call to return",
                    5000,
                )
                return
            self._on_audit_now()
            return
        # In flight → cancel.
        run_id = _auditor.cancel_current_run(reason="user_cancelled")
        worker = _runtime.get_active_worker()
        if worker is not None:
            try:
                worker._stop_event.set()  # signal worker loop to exit on next pull
            except Exception:
                pass
        if run_id is not None:
            self._status.showMessage(
                f"cancel sent for run #{run_id} — LLM call must finish (~180s max)", 6000
            )
        else:
            self._status.showMessage("no run to cancel", 3000)
        self._refresh_all()

    def _on_max_events_changed(self, value: int) -> None:
        """User adjusted the per-run slice cap. Pushes the new value to
        the active worker if any; new workers (created by toggling on
        from cold) pick it up via the same spinbox in _on_auditor_toggled."""
        worker = _runtime.get_active_worker()
        if worker is None:
            return
        try:
            worker.set_max_events(int(value))
        except Exception:
            pass
