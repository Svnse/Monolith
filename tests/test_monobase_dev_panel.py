from __future__ import annotations

import os
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QPushButton


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def _app() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _setup_isolated_db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    db_path = tmp_path / "test_acatalepsy.sqlite3"
    from core import db_connect as _dbc

    monkeypatch.setattr(_dbc, "DB_PATH", db_path, raising=True)
    monkeypatch.setenv("MONOLITH_DB_AUTHORIZER_STRICT", "1")
    conn = _dbc.connect_acatalepsy(role="migration")
    conn.executescript(
        """
        CREATE TABLE canonical_log (
            event_id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts REAL NOT NULL,
            kind TEXT NOT NULL,
            session_id TEXT,
            acu_id INTEGER,
            payload TEXT
        );
        CREATE TABLE acus (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical TEXT NOT NULL,
            veracity REAL NOT NULL DEFAULT 5.0,
            reinforcement INTEGER NOT NULL DEFAULT 1,
            source TEXT NOT NULL DEFAULT 'model',
            created_at TEXT NOT NULL,
            last_seen TEXT NOT NULL,
            candidate_id INTEGER,
            decision_id INTEGER
        );
        CREATE TABLE acu_candidates (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            canonical_form TEXT NOT NULL,
            evidence_log_id INTEGER NOT NULL,
            evidence_char_start INTEGER NOT NULL,
            evidence_char_end INTEGER NOT NULL,
            evidence_span TEXT NOT NULL,
            source TEXT NOT NULL,
            reason TEXT NOT NULL,
            reinforcement_count INTEGER NOT NULL DEFAULT 1,
            contradicts_acu_id INTEGER,
            state TEXT NOT NULL DEFAULT 'pending',
            created_at TEXT NOT NULL,
            auditor_run_id INTEGER
        );
        CREATE TABLE acu_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            candidate_id INTEGER NOT NULL,
            decision TEXT NOT NULL,
            decided_by TEXT NOT NULL,
            decided_at TEXT NOT NULL,
            reject_reason TEXT,
            edited_form TEXT,
            note TEXT,
            resulting_acu_id INTEGER
        );
        """
    )
    conn.commit()
    conn.close()

    from core.acatalepsy import schema
    schema.migrate()

    from core.acatalepsy import canonical_log, candidates, decisions, intake, runtime

    runtime.deregister_worker()
    for mod in (canonical_log, candidates, decisions, intake):
        if hasattr(mod, "_tl"):
            for attr in ("writer_conn", "reader_conn", "writer", "reader"):
                if hasattr(mod._tl, attr):
                    delattr(mod._tl, attr)
    return db_path


@pytest.fixture
def db(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    return _setup_isolated_db(tmp_path, monkeypatch)


def test_pending_selection_survives_refresh(db: Path) -> None:
    _app()
    from core.acatalepsy import canonical_log, candidates
    from ui.addons.monobase_dev import MonoBaseDevPanel

    log_id = canonical_log.append("user_message", payload={"text": "test"})
    first = candidates.insert_candidate(
        canonical_form="Monolith | uses | seven effort tiers",
        evidence_log_id=log_id,
        evidence_char_start=0,
        evidence_char_end=4,
        evidence_span="test",
        source="auditor_test",
        reason="load-bearing scaffold count",
    )
    second = candidates.insert_candidate(
        canonical_form="core/acu_store.py | defines | ACU table",
        evidence_log_id=log_id,
        evidence_char_start=0,
        evidence_char_end=4,
        evidence_span="test",
        source="auditor_test",
        reason="schema reference",
    )

    panel = MonoBaseDevPanel()
    try:
        panel._refresh_timer.stop()
        assert panel._list.count() == 2
        panel._list.setCurrentRow(1)
        assert panel._selected_candidate_id() == second

        panel._refresh_pending_list()

        assert panel._selected_candidate_id() == second
        assert panel._accept_btn.isEnabled()
        assert panel._reject_btn.isEnabled()
        assert panel._list.item(0).data(Qt.ItemDataRole.UserRole) == first
    finally:
        panel._refresh_timer.stop()
        panel.close()


def test_first_pending_candidate_is_selected_on_open(db: Path) -> None:
    _app()
    from core.acatalepsy import canonical_log, candidates
    from ui.addons.monobase_dev import MonoBaseDevPanel

    log_id = canonical_log.append("user_message", payload={"text": "test"})
    candidate_id = candidates.insert_candidate(
        canonical_form="Monolith | uses | seven effort tiers",
        evidence_log_id=log_id,
        evidence_char_start=0,
        evidence_char_end=4,
        evidence_span="test",
        source="auditor_test",
        reason="load-bearing scaffold count",
    )

    panel = MonoBaseDevPanel()
    try:
        panel._refresh_timer.stop()

        assert panel._selected_candidate_id() == candidate_id
        assert "canonical_form: Monolith | uses | seven effort tiers" in panel._evidence.toPlainText()
        assert panel._accept_btn.isEnabled()
        assert panel._reject_btn.isEnabled()
    finally:
        panel._refresh_timer.stop()
        panel.close()


def test_acatalepsy_map_dot_stylesheet_is_valid_shape(db: Path) -> None:
    _app()
    from ui.addons.monobase_widgets import AcatalepsyMapWidget

    widget = AcatalepsyMapWidget()
    try:
        style = widget._dot_style("active")

        assert "}}" not in style
        assert style.count("{") == style.count("}")
        assert "QPushButton#map_dot:hover" in style
    finally:
        widget.close()


def test_map_dot_opens_pending_candidate(db: Path) -> None:
    _app()
    from core.acatalepsy import canonical_log, candidates
    from ui.addons.monobase_dev import MonoBaseDevPanel

    log_id = canonical_log.append("user_message", payload={"text": "test"})
    candidate_id = candidates.insert_candidate(
        canonical_form="Monolith | uses | seven effort tiers",
        evidence_log_id=log_id,
        evidence_char_start=0,
        evidence_char_end=4,
        evidence_span="test",
        source="auditor_test",
        reason="load-bearing scaffold count",
    )

    panel = MonoBaseDevPanel()
    try:
        panel._refresh_timer.stop()
        buttons = panel._acu_map.findChildren(QPushButton, "map_dot")
        candidate_buttons = [
            button for button in buttons
            if "candidate #" in button.toolTip()
            and "Monolith | uses | seven effort tiers" in button.toolTip()
        ]

        assert candidate_buttons
        candidate_buttons[0].click()

        assert panel._view_stack.currentIndex() == 2
        assert panel._selected_candidate_id() == candidate_id
    finally:
        panel._refresh_timer.stop()
        panel.close()


def test_map_log_dot_opens_pending_log_explanation(db: Path) -> None:
    _app()
    from core.acatalepsy import canonical_log
    from ui.addons.monobase_dev import MonoBaseDevPanel

    latest = canonical_log.append(
        "user_message",
        payload={"text": "MonoBase writes ACUs only on Accept."},
    )

    panel = MonoBaseDevPanel()
    try:
        panel._refresh_timer.stop()
        buttons = panel._acu_map.findChildren(QPushButton, "map_dot")
        log_buttons = [
            button for button in buttons
            if "waiting for audit" in button.toolTip()
        ]

        assert log_buttons
        log_buttons[0].click()

        assert panel._view_stack.currentIndex() == 1
        detail = panel._log_detail.toPlainText()
        assert f"Canonical log event #{latest}" in detail
        assert "waiting for an auditor run" in detail
    finally:
        panel._refresh_timer.stop()
        panel.close()


def test_map_dot_opens_acu_detail(db: Path) -> None:
    _app()
    from core.acatalepsy import canonical_log, candidates, decisions
    from core.acatalepsy.normalize import normalize_canonical
    from ui.addons.monobase_dev import MonoBaseDevPanel

    log_id = canonical_log.append("user_message", payload={"text": "test"})
    candidate_id = candidates.insert_candidate(
        canonical_form="Monolith | uses | seven effort tiers",
        evidence_log_id=log_id,
        evidence_char_start=0,
        evidence_char_end=4,
        evidence_span="test",
        source="auditor_test",
        reason="load-bearing scaffold count",
    )
    decision_id = decisions.insert_decision(
        candidate_id=candidate_id,
        decision="accept",
        decided_by="user_e",
    )
    decision = decisions.read_one(decision_id)
    assert decision is not None
    assert decision.resulting_acu_id is not None

    panel = MonoBaseDevPanel()
    try:
        panel._refresh_timer.stop()
        buttons = panel._acu_map.findChildren(QPushButton, "map_dot")
        acu_buttons = [
            button for button in buttons
            if f"ACU #{decision.resulting_acu_id}" in button.toolTip()
        ]

        assert acu_buttons
        acu_buttons[0].click()

        detail = panel._acu_detail.toPlainText()
        assert panel._view_stack.currentIndex() == 0
        assert f"ACU #{decision.resulting_acu_id}" in detail
        assert normalize_canonical("Monolith | uses | seven effort tiers") in detail
    finally:
        panel._refresh_timer.stop()
        panel.close()


def test_audit_button_cold_starts_worker_and_enqueues_manual_run(
    db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app()
    from core.acatalepsy import runtime
    from ui.addons import monobase_dev
    from ui.addons.monobase_dev import MonoBaseDevPanel

    runtime.deregister_worker()

    class _Thread:
        def __init__(self, alive: bool = False) -> None:
            self._alive = alive

        def is_alive(self) -> bool:
            return self._alive

    class _Queue:
        def __init__(self) -> None:
            self.enqueued: list[tuple[str, str | None]] = []

        def size(self) -> int:
            return len(self.enqueued)

        def enqueue(self, kind: str, enqueued_by: str | None = None, **_kwargs) -> bool:
            self.enqueued.append((kind, enqueued_by))
            return True

    created: list["_Worker"] = []

    class _Worker:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.queue_handle = _Queue()
            self._thread = _Thread(False)
            self._stop_event = SimpleNamespace(is_set=lambda: False)
            self._size_threshold = 50
            self._max_events = kwargs.get("max_events_per_run")
            created.append(self)

        def start(self) -> None:
            self._thread = _Thread(True)

        def stop(self) -> None:
            self._thread = _Thread(False)

        def set_max_events(self, value: int) -> None:
            self._max_events = int(value)

    monkeypatch.setattr(
        monobase_dev.MonoBaseDevPanel,
        "_probe_sidecar",
        lambda self: (True, "auditor-test-model", ""),
    )
    monkeypatch.setattr(
        "core.acatalepsy.llm_sidecar.make_auditor_llm",
        lambda: (lambda **_kwargs: '{"candidates": []}'),
    )
    monkeypatch.setattr("core.acatalepsy.triggers.AuditorWorker", _Worker)

    panel = MonoBaseDevPanel()
    try:
        panel._refresh_timer.stop()
        runtime.deregister_worker()
        panel._sync_auditor_controls()
        panel._apply_status_snapshot(monobase_dev._monobase_status.build_monobase_snapshot())

        assert panel._audit_btn.isEnabled()
        assert panel._audit_btn.text() == "Start + Audit"

        panel._audit_btn.click()

        assert created
        worker = runtime.get_active_worker()
        assert worker is created[-1]
        assert worker.queue_handle.enqueued == [("manual", "user_e")]
        assert panel._trigger_queue is worker.queue_handle
    finally:
        panel._refresh_timer.stop()
        runtime.deregister_worker()
        panel.close()


def test_audit_button_runs_worker_and_surfaces_candidate(
    db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    app = _app()
    from core.acatalepsy import auditor, canonical_log, candidates, runtime
    from ui.addons import monobase_dev
    from ui.addons.monobase_dev import MonoBaseDevPanel

    runtime.deregister_worker()
    evidence_id = canonical_log.append(
        "user_message",
        payload={"text": "MonoBase writes ACUs only on Accept."},
    )

    def stub_llm(*, system_prompt: str, user_content: str) -> str:
        assert "Acatalepsy auditor" in system_prompt
        assert f"event_id={evidence_id}" in user_content
        return (
            '{"candidates":[{'
            '"canonical_form":"MonoBase | writes | ACUs on Accept",'
            f'"evidence_log_id":{evidence_id},'
            '"evidence_char_start":0,'
            '"evidence_char_end":35,'
            '"evidence_span":"MonoBase writes ACUs only on Accept.",'
            '"reason":"clarifies candidate versus ACU lifecycle"'
            '}]}'
        )

    monkeypatch.setattr(
        monobase_dev.MonoBaseDevPanel,
        "_probe_sidecar",
        lambda self: (True, "stub-auditor-model", ""),
    )
    monkeypatch.setattr("core.acatalepsy.llm_sidecar.make_auditor_llm", lambda: stub_llm)

    panel = MonoBaseDevPanel()
    worker = None
    try:
        panel._refresh_timer.stop()
        panel._apply_status_snapshot(monobase_dev._monobase_status.build_monobase_snapshot())
        assert panel._audit_btn.text() == "Start + Audit"

        panel._audit_btn.click()

        deadline = time.time() + 3.0
        pending = []
        runs = []
        while time.time() < deadline:
            app.processEvents()
            pending = candidates.read_pending(limit=10)
            runs = auditor.read_recent_runs(limit=1)
            if pending and runs and runs[0].get("status") == "success":
                break
            time.sleep(0.02)

        assert len(pending) == 1
        assert pending[0].canonical_form == "MonoBase | writes | ACUs on Accept"
        assert pending[0].source == "auditor_monolith"
        assert runs and runs[0]["candidates_inserted"] == 1

        panel._refresh_all()
        assert panel._selected_candidate_id() == pending[0].id
        assert "canonical_form: MonoBase | writes | ACUs on Accept" in panel._evidence.toPlainText()
        assert panel._accept_btn.isEnabled()
    finally:
        worker = runtime.get_active_worker()
        if worker is not None:
            worker.stop(timeout=1.0)
        runtime.deregister_worker()
        panel._refresh_timer.stop()
        panel.close()


def test_status_scan_line_shows_zero_cursor(db: Path) -> None:
    _app()
    from core.acatalepsy.monobase_status import MonobaseSnapshot, WorkerStatus
    from ui.addons.monobase_dev import MonoBaseDevPanel

    panel = MonoBaseDevPanel()
    try:
        panel._refresh_timer.stop()
        snapshot = MonobaseSnapshot(
            now=100.0,
            phase="needs_audit",
            phase_title="Audit available",
            phase_detail="8 event(s) behind cursor.",
            phase_tone="warn",
            cursor=0,
            latest_event_id=8,
            pending_log_events=8,
            candidate_counts={"pending": 0, "accepted": 0, "rejected": 0},
            pending_candidate_count=0,
            worker=WorkerStatus(
                registered=False,
                thread_alive=False,
                stop_requested=False,
                queue_size=None,
                size_threshold=None,
                max_events_per_run=None,
            ),
            in_flight_run=None,
            run_elapsed_secs=None,
            llm_elapsed_secs=None,
            recent_runs=(),
            recent_acu_writes=(),
        )

        panel._apply_status_snapshot(snapshot)

        assert "cursor 0 / latest 8" in panel._scan_progress.text()
    finally:
        panel._refresh_timer.stop()
        panel.close()


def test_evidence_scroll_survives_auto_refresh(db: Path) -> None:
    """The 2s auto-refresh must NOT reset the evidence-pane scroll position when
    the selected candidate is unchanged. Previously each refresh re-ran
    setPlainText() on the QTextEdit, teleporting the scroll back to the top."""
    app = _app()
    from core.acatalepsy import canonical_log, candidates
    from ui.addons.monobase_dev import MonoBaseDevPanel

    log_id = canonical_log.append("user_message", payload={"text": "test"})
    long_reason = "\n".join(
        f"reasoning line {i:03d} — explaining the candidate at some length" for i in range(120)
    )
    cid = candidates.insert_candidate(
        canonical_form="Monolith | uses | seven effort tiers",
        evidence_log_id=log_id,
        evidence_char_start=0,
        evidence_char_end=4,
        evidence_span="test",
        source="auditor_test",
        reason=long_reason,
    )
    candidates.insert_candidate(
        canonical_form="core/acu_store.py | defines | ACU table",
        evidence_log_id=log_id,
        evidence_char_start=0,
        evidence_char_end=4,
        evidence_span="test",
        source="auditor_test",
        reason="short",
    )

    panel = MonoBaseDevPanel()
    try:
        panel._refresh_timer.stop()
        panel.resize(420, 600)
        panel.show()
        app.processEvents()

        # Select the long candidate.
        for row in range(panel._list.count()):
            if panel._list.item(row).data(Qt.ItemDataRole.UserRole) == cid:
                panel._list.setCurrentRow(row)
                break
        app.processEvents()
        assert panel._selected_candidate_id() == cid

        scrollbar = panel._evidence.verticalScrollBar()
        assert scrollbar.maximum() > 0, "evidence pane must be scrollable for this test to mean anything"
        scrollbar.setValue(scrollbar.maximum())
        app.processEvents()
        scrolled_to = scrollbar.value()
        assert scrolled_to > 0

        # Simulate a 2s auto-refresh tick with the same candidate still selected.
        panel._refresh_pending_list()
        app.processEvents()

        assert panel._selected_candidate_id() == cid
        assert panel._evidence.verticalScrollBar().value() == scrolled_to, (
            "auto-refresh reset the evidence scroll position to the top"
        )
    finally:
        panel._refresh_timer.stop()
        panel.close()


def test_audit_page_selection_shows_readable_run_and_event_details(db: Path) -> None:
    _app()
    from core.acatalepsy import canonical_log
    from ui.addons.monobase_dev import MonoBaseDevPanel

    run_id = canonical_log.append(
        "auditor_run_started",
        payload={
            "source": "auditor_test",
            "slice_start_event_id": 1,
            "slice_end_event_id": 4,
            "prompt_version": 1,
            "max_events_per_run": 50,
        },
    )
    canonical_log.append(
        "auditor_llm_call_started",
        payload={
            "run_id": run_id,
            "prompt_chars": 100,
            "user_chars": 200,
            "events_in_slice": 3,
        },
    )
    canonical_log.append(
        "auditor_run_complete",
        payload={
            "run_id": run_id,
            "status": "success",
            "events_processed": 3,
            "proposals_returned": 2,
            "candidates_inserted": 1,
            "candidates_rejected": 1,
        },
    )

    panel = MonoBaseDevPanel()
    try:
        panel._refresh_timer.stop()
        panel._show_view(1)

        assert panel._runs_list.count() >= 1
        panel._runs_list.setCurrentRow(0)
        run_detail = panel._run_detail.toPlainText()
        assert "Run summary" in run_detail
        assert "proposals_returned: 2" in run_detail
        assert "candidates_inserted: 1" in run_detail

        assert panel._log_list.count() >= 1
        panel._log_list.setCurrentRow(0)
        log_detail = panel._log_detail.toPlainText()
        assert f"Audit event #{panel._log_list.item(0).data(Qt.ItemDataRole.UserRole)}" in log_detail
        assert "payload:" in log_detail
        assert '"run_id":' in log_detail
    finally:
        panel._refresh_timer.stop()
        panel.close()


def _make_runs(canonical_log) -> int:
    run_id = canonical_log.append(
        "auditor_run_started",
        payload={
            "source": "auditor_test",
            "slice_start_event_id": 1,
            "slice_end_event_id": 4,
            "prompt_version": 1,
            "max_events_per_run": 50,
        },
    )
    canonical_log.append(
        "auditor_run_complete",
        payload={
            "run_id": run_id,
            "status": "success",
            "events_processed": 3,
            "proposals_returned": 2,
            "candidates_inserted": 1,
            "candidates_rejected": 1,
        },
    )
    return run_id


def test_run_selection_survives_auto_refresh(db: Path) -> None:
    """A selected run (and its detail pane) must survive the 2s refresh.
    Pre-fix _refresh_runs_list cleared the list with no selection preservation,
    so the selection — and the detail — were lost on every tick."""
    _app()
    from core.acatalepsy import canonical_log
    from ui.addons.monobase_dev import MonoBaseDevPanel

    _make_runs(canonical_log)
    panel = MonoBaseDevPanel()
    try:
        panel._refresh_timer.stop()
        panel._show_view(1)
        assert panel._runs_list.count() >= 1
        panel._runs_list.setCurrentRow(0)
        sel = panel._runs_list.currentItem().data(Qt.ItemDataRole.UserRole)
        assert "Run summary" in panel._run_detail.toPlainText()

        panel._refresh_runs_list()  # simulate an auto-refresh tick

        current = panel._runs_list.currentItem()
        assert current is not None and current.data(Qt.ItemDataRole.UserRole) == sel
        assert "Run summary" in panel._run_detail.toPlainText()
        assert "Select a run" not in panel._run_detail.toPlainText()
    finally:
        panel._refresh_timer.stop()
        panel.close()


def test_log_selection_survives_auto_refresh(db: Path) -> None:
    """A selected audit-log event (and its detail) must survive the refresh."""
    _app()
    from core.acatalepsy import canonical_log
    from ui.addons.monobase_dev import MonoBaseDevPanel

    _make_runs(canonical_log)
    panel = MonoBaseDevPanel()
    try:
        panel._refresh_timer.stop()
        panel._show_view(1)
        assert panel._log_list.count() >= 1
        panel._log_list.setCurrentRow(0)
        sel = panel._log_list.currentItem().data(Qt.ItemDataRole.UserRole)
        assert "payload:" in panel._log_detail.toPlainText()

        panel._refresh_log_tail()  # simulate an auto-refresh tick

        current = panel._log_list.currentItem()
        assert current is not None and current.data(Qt.ItemDataRole.UserRole) == sel
        assert "payload:" in panel._log_detail.toPlainText()
        assert "Select an audit event" not in panel._log_detail.toPlainText()
    finally:
        panel._refresh_timer.stop()
        panel.close()


def test_acu_detail_scroll_survives_auto_refresh(db: Path) -> None:
    """ALL ACUs tab: scrolling the ACU detail pane must survive the refresh."""
    app = _app()
    from core.acatalepsy import canonical_log, candidates, decisions
    from ui.addons.monobase_dev import MonoBaseDevPanel

    log_id = canonical_log.append("user_message", payload={"text": "test"})
    cid = candidates.insert_candidate(
        canonical_form="Monolith | uses | seven effort tiers",
        evidence_log_id=log_id,
        evidence_char_start=0,
        evidence_char_end=4,
        evidence_span="test",
        source="auditor_test",
        reason="load-bearing scaffold count",
    )
    decision_id = decisions.insert_decision(
        candidate_id=cid, decision="accept", decided_by="user_e"
    )
    acu_id = decisions.read_one(decision_id).resulting_acu_id
    assert acu_id is not None

    panel = MonoBaseDevPanel()
    try:
        panel._refresh_timer.stop()
        panel.resize(420, 600)
        panel._show_view(0)
        panel.show()
        app.processEvents()

        for row in range(panel._acus_list.count()):
            if panel._acus_list.item(row).data(Qt.ItemDataRole.UserRole) == acu_id:
                panel._acus_list.setCurrentRow(row)
                break
        app.processEvents()
        assert panel._selected_acu_id() == acu_id

        # Force a small viewport so the short ACU detail is scrollable. Let the
        # relayout settle, then scroll to a stable mid-point (scrolling to the
        # exact maximum is sensitive to the scrollbar range still settling).
        panel._acu_detail.setFixedHeight(40)
        app.processEvents()
        app.processEvents()
        scrollbar = panel._acu_detail.verticalScrollBar()
        assert scrollbar.maximum() > 0
        target = max(1, scrollbar.maximum() // 2)
        scrollbar.setValue(target)
        app.processEvents()
        scrolled_to = scrollbar.value()
        assert scrolled_to > 0

        panel._refresh_acus_list()  # simulate an auto-refresh tick
        app.processEvents()

        assert panel._selected_acu_id() == acu_id
        assert panel._acu_detail.verticalScrollBar().value() == scrolled_to
    finally:
        panel._refresh_timer.stop()
        panel.close()


def test_llm_output_set_preserves_scroll_on_identical_text(db: Path) -> None:
    """LLM OUTPUT tab: re-setting identical text (the refresh's common case)
    must not reset the scrollbar."""
    app = _app()
    from ui.addons.monobase_dev import MonoBaseDevPanel

    panel = MonoBaseDevPanel()
    try:
        panel._refresh_timer.stop()
        panel.resize(420, 600)
        panel._show_view(3)
        panel.show()
        app.processEvents()

        long_text = "\n".join(f"auditor output line {i:03d}" for i in range(200))
        panel._set_llm_text(long_text)
        panel._llm_output.setFixedHeight(60)
        app.processEvents()
        app.processEvents()
        scrollbar = panel._llm_output.verticalScrollBar()
        assert scrollbar.maximum() > 0
        target = max(1, scrollbar.maximum() // 2)
        scrollbar.setValue(target)
        app.processEvents()
        scrolled_to = scrollbar.value()
        assert scrolled_to > 0

        panel._set_llm_text(long_text)  # identical content -> must be a no-op
        app.processEvents()

        assert panel._llm_output.verticalScrollBar().value() == scrolled_to
    finally:
        panel._refresh_timer.stop()
        panel.close()


def test_auto_review_accepts_safe_and_rejects_invalid_without_user(db: Path) -> None:
    from core.acatalepsy import auto_review, canonical_log, candidates, decisions

    log_id = canonical_log.append(
        "user_message",
        payload={"text": "Monolith has a recall lane. Want me to trim any of those?"},
    )
    safe_id = candidates.insert_candidate(
        canonical_form="Monolith | has | recall lane",
        evidence_log_id=log_id,
        evidence_char_start=0,
        evidence_char_end=27,
        evidence_span="Monolith has a recall lane.",
        source="auditor_monolith",
        reason="runtime-state recall substrate claim",
    )
    invalid_id = candidates.insert_candidate(
        canonical_form="Want me to | trim | any of those?",
        evidence_log_id=log_id,
        evidence_char_start=29,
        evidence_char_end=61,
        evidence_span="Want me to trim any of those?",
        source="auditor_monolith",
        reason="conversational fragment",
    )

    summary = auto_review.review_pending(limit=10)

    assert summary.accepted == 1
    assert summary.rejected == 1
    assert summary.skipped == 0
    assert candidates.read_one(safe_id).state == "accepted"
    assert candidates.read_one(invalid_id).state == "rejected"

    safe_decision = decisions.read_by_candidate(safe_id)[0]
    assert safe_decision.decided_by == "agent_monolith"
    assert safe_decision.decision == "accept"
    assert safe_decision.resulting_acu_id is not None

    invalid_decision = decisions.read_by_candidate(invalid_id)[0]
    assert invalid_decision.decided_by == "agent_monolith"
    assert invalid_decision.decision == "reject"
    assert invalid_decision.reject_reason.startswith("auto_review:extraction_quality:")


def test_monobase_auto_review_button_runs_batch_decisions(
    db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _app()
    from PySide6.QtWidgets import QMessageBox
    from core.acatalepsy import canonical_log, candidates
    from ui.addons.monobase_dev import MonoBaseDevPanel

    log_id = canonical_log.append("user_message", payload={"text": "test"})
    candidate_id = candidates.insert_candidate(
        canonical_form="Monolith | has | recall lane",
        evidence_log_id=log_id,
        evidence_char_start=0,
        evidence_char_end=4,
        evidence_span="test",
        source="auditor_monolith",
        reason="runtime-state recall substrate claim",
    )
    seen: list[tuple[int, str]] = []
    monkeypatch.setattr(QMessageBox, "question", lambda *args, **kwargs: QMessageBox.Yes)

    panel = MonoBaseDevPanel(on_decision=lambda cid, action: seen.append((cid, action)))
    try:
        panel._refresh_timer.stop()

        assert panel._auto_review_btn.isEnabled()
        panel._on_auto_review()

        assert candidates.read_one(candidate_id).state == "accepted"
        assert seen == [(candidate_id, "accept")]
        assert panel._list.count() == 0
    finally:
        panel._refresh_timer.stop()
        panel.close()
