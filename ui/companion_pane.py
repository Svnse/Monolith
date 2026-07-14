from __future__ import annotations

from enum import Enum, auto

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, Signal
from PySide6.QtGui import QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QStackedLayout,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

import core.style as _s
from core.paths import ARCHIVE_DIR
from ui.pages.databank import PageFiles
from ui.panels import (
    ActionReviewPanel,
    AudioPanel,
    ArchiveBrowserPanel,
    AuditLogPanel,
    ExpeditionPanel,
    SelfMaintPanel,
    GenerationTracePanel,
    ModelConfigPanel,
    WorkshopPane,
    WorkshopLibraryPane,
)


class CompanionState(Enum):
    COLLAPSED = auto()
    CONFIG = auto()
    GENERATING = auto()
    ACTION_REVIEW = auto()
    ASK_USER = auto()
    VISION = auto()
    AUDIO = auto()
    ARCHIVE = auto()
    AUDIT = auto()
    DATABANK = auto()
    STATS = auto()
    EXPEDITION = auto()
    WORKSHOP = auto()
    REASONING_TREE = auto()
    SELF_MAINT = auto()
    ADDON = auto()


class _ResizeGrip(QWidget):
    """Thin vertical strip on the LEFT edge of CompanionPane. Dragging
    horizontally resizes the pane width — leftward drag widens it (the pane
    is anchored to the right side of the window). Provides the 'tabs longer
    from right to left' behavior: the user can pull the pane open as wide as
    they want instead of accepting the fixed 360px animation target."""

    _MIN_WIDTH = 360
    _MAX_WIDTH = 1200

    def __init__(self, pane: "CompanionPane"):
        super().__init__(pane)
        self._pane = pane
        self.setFixedWidth(4)
        self.setCursor(Qt.SizeHorCursor)
        self.setStyleSheet(f"background: {_s.BORDER_SUBTLE};")
        self._drag_origin_x: int | None = None
        self._drag_start_width: int = 0

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_origin_x = int(event.globalPosition().x())
            self._drag_start_width = int(self._pane.maximumWidth())
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent) -> None:
        if self._drag_origin_x is None:
            return
        # Leftward drag (globalX decreases) widens the pane.
        delta = self._drag_origin_x - int(event.globalPosition().x())
        new_w = max(self._MIN_WIDTH, min(self._MAX_WIDTH, self._drag_start_width + delta))
        self._pane.setMinimumWidth(new_w)
        self._pane.setMaximumWidth(new_w)
        self._pane._user_width = new_w  # so collapse/expand cycles restore this

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:
        self._drag_origin_x = None
        super().mouseReleaseEvent(event)


class CompanionPane(QWidget):
    sig_state_changed = Signal(str)
    _ACTIVE_ENGINE_STATUSES = {"RUNNING", "LOADING", "GENERATING"}
    _DONE_TASK_STATUSES = {"DONE", "FAILED", "CANCELLED"}
    _AUTO_ENGINE_STATES = (
        (CompanionState.GENERATING, lambda self: getattr(self._conversation, "_engine_key", "")),
        (CompanionState.VISION, lambda _self: "vision"),
        (CompanionState.AUDIO, lambda _self: "audio"),
    )

    def __init__(self, state, ui_bridge, parent=None):
        super().__init__(parent)
        self.state = state
        self.ui_bridge = ui_bridge
        self._current_state = CompanionState.COLLAPSED
        self._pinned_state: CompanionState | None = None
        self._active_addon_id: str | None = None
        self._conversation = None
        self._conversation_panels: dict[CompanionState, QWidget] = {}
        self._addon_titles: dict[str, str] = {}
        self._suppressed_auto_states: dict[CompanionState, tuple[str, ...]] = {}

        self.setObjectName("companion_shell")
        self.setMaximumWidth(0)
        self.setMinimumWidth(0)
        # User-resizable expanded width. Updated by _ResizeGrip drag; read by
        # transition_to so collapse/expand cycles restore the user's choice.
        self._user_width = 360

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        self._resize_grip = _ResizeGrip(self)
        outer.addWidget(self._resize_grip)

        self._frame = QFrame()
        self._frame.setObjectName("companion_pane")
        frame_layout = QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(12, 12, 12, 12)
        frame_layout.setSpacing(8)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(8)

        self._title = QLabel("COMPANION")
        header.addWidget(self._title)
        header.addStretch()
        frame_layout.addLayout(header)

        # _title has a per-widget stylesheet that f-strings in live theme
        # tokens — those values would freeze at construction unless we
        # re-apply on each theme change. Wire the subscription and run it
        # once now for the initial paint.
        self._apply_chrome_theme()
        if hasattr(self.ui_bridge, "sig_theme_changed"):
            self.ui_bridge.sig_theme_changed.connect(self._apply_chrome_theme)

        self._stack_host = QWidget()
        self._stack = QStackedLayout(self._stack_host)
        frame_layout.addWidget(self._stack_host, 1)

        # VISION is registered lazily by main_window when the sd addon
        # launches — the full SDModule UI mounts directly here (no more
        # read-only VisionPanel summary that needed bind_module). AUDIO
        # still uses the proxy pattern via AudioPanel + bind_module pending
        # the same conversion.
        # WORKSHOP: bind the card library's registry to world_state so "Set Active" persists the
        # active-flow id to the SAME store the chat guard reads -- otherwise set_active is a silent
        # no-op and the chat always runs Genesis. (The library's Test/Edit/Create/focus signals are
        # wired to PageChat separately, via bind_controller when the conversation is set.)
        from core.workflow_registry import WorkflowRegistry as _WorkflowRegistry
        _workshop_registry = _WorkflowRegistry()
        _workshop_ws = getattr(state, "world_state", None)
        if _workshop_ws is not None:
            _workshop_registry.bind_world_state(_workshop_ws)
        self._panels: dict[CompanionState, QWidget] = {
            CompanionState.CONFIG: ModelConfigPanel(state, ui_bridge),
            CompanionState.GENERATING: GenerationTracePanel(),
            CompanionState.ACTION_REVIEW: ActionReviewPanel(),
            CompanionState.AUDIO: AudioPanel(state),
            CompanionState.ARCHIVE: ArchiveBrowserPanel(ARCHIVE_DIR),
            CompanionState.AUDIT: AuditLogPanel(getattr(state, "world_state", None)),
            CompanionState.DATABANK: PageFiles(state, ui_bridge),
            CompanionState.EXPEDITION: ExpeditionPanel(),
            CompanionState.SELF_MAINT: SelfMaintPanel(getattr(state, "world_state", None)),
            CompanionState.WORKSHOP: WorkshopLibraryPane(registry=_workshop_registry),
        }
        self._placeholder = QLabel("Companion collapsed.")
        self._placeholder.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 11px; font-family: Consolas; padding: 8px 4px;"
        )
        self._stack.addWidget(self._placeholder)

        self._addon_panel = QWidget()
        addon_layout = QVBoxLayout(self._addon_panel)
        addon_layout.setContentsMargins(0, 0, 0, 0)
        addon_layout.setSpacing(6)
        # Inner addon title is hidden: the outer companion title (_title)
        # now shows the actual addon name (e.g. "MONOBASE" instead of the
        # generic "ADDON"). _addon_title is kept around as a no-op widget
        # in case external code still reads it.
        self._addon_title = QLabel("ADDON")
        self._addon_title.setVisible(False)
        addon_layout.addWidget(self._addon_title)
        self._addon_stack = QStackedWidget()
        addon_layout.addWidget(self._addon_stack, 1)
        self._stack.addWidget(self._addon_panel)
        self._addon_widgets: dict[str, QWidget] = {}

        for panel in self._panels.values():
            self._stack.addWidget(panel)

        outer.addWidget(self._frame, 1)

    def set_conversation(self, controller) -> None:
        self._conversation = controller
        self._conversation_panels = {}

        panel_map = {}
        provider = getattr(controller, "companion_panels", None)
        if callable(provider):
            try:
                panel_map = provider() or {}
            except Exception:
                panel_map = {}

        for state, panel in self._panels.items():
            bind = getattr(panel, "bind_controller", None)
            if callable(bind):
                bind(controller)

        for raw_key, panel in dict(panel_map).items():
            state = raw_key if isinstance(raw_key, CompanionState) else CompanionState[str(raw_key)]
            self._conversation_panels[state] = panel
            if self._stack.indexOf(panel) < 0:
                self._stack.addWidget(panel)
            bind = getattr(panel, "bind_controller", None)
            if callable(bind):
                bind(controller)
            if state == CompanionState.AUDIT:
                refresh = getattr(panel, "refresh", None)
                if callable(refresh):
                    refresh()
        self._sync_media_panels()
        self.refresh_from_world()

    def register_addon(self, mod_id: str, title: str, widget: QWidget) -> None:
        self._addon_titles[mod_id] = title
        if mod_id not in self._addon_widgets:
            self._addon_widgets[mod_id] = widget
            self._addon_stack.addWidget(widget)
        self._active_addon_id = mod_id
        self._addon_title.setText(title.upper())
        self._sync_media_panels()

    def unregister_addon(self, mod_id: str) -> None:
        widget = self._addon_widgets.pop(mod_id, None)
        self._addon_titles.pop(mod_id, None)
        if widget is not None:
            self._addon_stack.removeWidget(widget)
        self._sync_media_panels()
        if self._active_addon_id == mod_id:
            self._active_addon_id = None
            if self._current_state == CompanionState.ADDON:
                self.transition_to(CompanionState.COLLAPSED)

    def show_addon(self, mod_id: str, title: str, widget: QWidget, pin: bool = True) -> None:
        self.register_addon(mod_id, title, widget)
        if pin:
            self._pinned_state = CompanionState.ADDON
        self._active_addon_id = mod_id
        self._addon_title.setText(title.upper())
        self._addon_stack.setCurrentWidget(widget)
        self.transition_to(CompanionState.ADDON)

    def get_panel(self, state: CompanionState) -> QWidget | None:
        return self._conversation_panels.get(state) or self._panels.get(state)

    def pin_state(self, state: CompanionState) -> None:
        self._active_addon_id = None
        self._suppressed_auto_states.pop(state, None)
        self._pinned_state = None if self._pinned_state == state else state
        self.refresh_from_world()

    def show_state(self, state: CompanionState) -> None:
        self._active_addon_id = None
        self._suppressed_auto_states.pop(state, None)
        self._pinned_state = state
        self.refresh_from_world()

    def collapse(self) -> None:
        world_state = getattr(self.state, "world_state", None)
        snapshot = world_state.snapshot() if world_state is not None else {}
        self._suppress_auto_state(snapshot)
        self._pinned_state = None
        self.transition_to(CompanionState.COLLAPSED)

    def _apply_chrome_theme(self, *_args) -> None:
        """Re-apply the title stylesheet from live theme tokens on theme change."""
        self._title.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 11px; font-family: Consolas; font-weight: bold;"
        )

    def refresh_from_world(self) -> None:
        world_state = getattr(self.state, "world_state", None)
        snapshot = world_state.snapshot() if world_state is not None else {}
        self.transition_to(self.evaluate_state(snapshot))

    def evaluate_state(self, world_snapshot: dict) -> CompanionState:
        self._release_stale_auto_suppressions(world_snapshot)
        session = world_snapshot.get("session", {})
        if session.get("pending_action"):
            return CompanionState.ACTION_REVIEW
        if session.get("pending_question"):
            return CompanionState.ASK_USER

        if self._pinned_state == CompanionState.ADDON and self._active_addon_id:
            return CompanionState.ADDON
        if self._pinned_state is not None:
            return self._pinned_state

        if self._workshop_running(world_snapshot):
            return CompanionState.WORKSHOP   # more specific than GENERATING while spawning

        auto_state, _auto_key = self._current_auto_state(world_snapshot)
        if auto_state is not None:
            return auto_state
        return CompanionState.COLLAPSED

    def _workshop_running(self, world_snapshot: dict) -> bool:
        """True while the chat host has set the world_state 'workshop' engine flag
        (set on spawn, cleared when the turn's tree folds). The WorkshopPane itself
        polls turn_trace, so this only governs the companion's auto-show."""
        engines = (world_snapshot or {}).get("engines", {}) or {}
        ws = engines.get("workshop") or {}
        return str(ws.get("status", "")).strip().lower() == "running"

    def _latest_addon_widget(self, addon_id: str) -> QWidget | None:
        for widget in reversed(list(self._addon_widgets.values())):
            if getattr(widget, "_addon_id", None) == addon_id:
                return widget
        return None

    def _sync_media_panels(self) -> None:
        # Vision no longer needs a bind_module proxy — SDModule is mounted
        # directly via attach_module() when the sd addon launches. Audio
        # still uses the AudioPanel summary + bind_module path.
        audio_panel = self._panels.get(CompanionState.AUDIO)
        if audio_panel is not None and hasattr(audio_panel, "bind_module"):
            audio_panel.bind_module(self._latest_addon_widget("audiogen"))

    def attach_module(self, state: CompanionState, widget: QWidget) -> None:
        """Mount a full addon widget as the panel for `state` (e.g. VISION
        gets the actual SDModule). Replaces any prior panel for that state.

        Used by main_window when sd / audiogen / etc. addons launch — the
        user opens the full interactive UI in the companion pane instead of
        a read-only summary. Auto-detaches on widget destruction so a
        closed addon doesn't leave a dangling pointer in `_panels`.
        """
        if widget is None:
            return
        old = self._panels.get(state)
        if old is widget:
            return
        if old is not None and self._stack.indexOf(old) >= 0:
            self._stack.removeWidget(old)
        self._panels[state] = widget
        if self._stack.indexOf(widget) < 0:
            self._stack.addWidget(widget)
        widget.destroyed.connect(lambda _obj=None, st=state: self._detach_destroyed(st))
        # If the user has VISION/etc. currently showing, swap to the new
        # widget immediately so the placeholder vanishes.
        if self._current_state == state:
            self._stack.setCurrentWidget(widget)

    def _detach_destroyed(self, state: CompanionState) -> None:
        """Drop a destroyed widget reference from _panels. Triggered by Qt
        when main_window calls widget.deleteLater() on close_module."""
        self._panels.pop(state, None)
        if self._current_state == state:
            self.transition_to(CompanionState.COLLAPSED)

    def _current_auto_state(
        self,
        world_snapshot: dict,
        *,
        include_suppressed: bool = False,
    ) -> tuple[CompanionState | None, tuple[str, ...] | None]:
        for state, engine_key_factory in self._AUTO_ENGINE_STATES:
            engine_key = str(engine_key_factory(self) or "")
            auto_key = self._engine_task_key(engine_key, world_snapshot)
            if auto_key is None:
                continue
            if not include_suppressed and self._suppressed_auto_states.get(state) == auto_key:
                continue
            return state, auto_key
        return None, None

    def _auto_state_key(
        self,
        state: CompanionState,
        world_snapshot: dict,
    ) -> tuple[str, ...] | None:
        for candidate_state, engine_key_factory in self._AUTO_ENGINE_STATES:
            if candidate_state != state:
                continue
            return self._engine_task_key(str(engine_key_factory(self) or ""), world_snapshot)
        return None

    def _release_stale_auto_suppressions(self, world_snapshot: dict) -> None:
        for state, suppressed_key in list(self._suppressed_auto_states.items()):
            current_key = self._auto_state_key(state, world_snapshot)
            if current_key is None or current_key != suppressed_key:
                self._suppressed_auto_states.pop(state, None)

    def _suppress_auto_state(self, world_snapshot: dict) -> None:
        state, auto_key = self._current_auto_state(world_snapshot, include_suppressed=True)
        if state is not None and auto_key is not None:
            self._suppressed_auto_states[state] = auto_key

    def _engine_is_busy(self, engine_key: str, world_snapshot: dict) -> bool:
        return self._engine_task_key(engine_key, world_snapshot) is not None

    def _engine_task_key(self, engine_key: str, world_snapshot: dict) -> tuple[str, ...] | None:
        if not engine_key:
            return None
        engines = world_snapshot.get("engines", {})
        tasks = world_snapshot.get("tasks", {})
        engine_state = engines.get(engine_key, {}) if isinstance(engines, dict) else {}
        task_state = tasks.get(engine_key, {}) if isinstance(tasks, dict) else {}
        status = str(engine_state.get("status", "") or "").upper()
        active = task_state.get("active")
        active_status = ""
        active_command = ""
        active_id = ""
        if isinstance(active, dict):
            active_status = str(active.get("status", "") or "").upper()
            active_command = str(active.get("command", "") or "").lower()
            active_id = str(active.get("id", "") or "")
        if active_status and active_status in self._DONE_TASK_STATUSES:
            return None
        if active_status == "RUNNING" and active_command in {"generate", "load", "unload"}:
            return (engine_key, active_id or active_command or status, active_status)
        if status in self._ACTIVE_ENGINE_STATUSES and isinstance(active, dict):
            return (engine_key, active_id or active_command or status, active_status or status)
        return None

    def transition_to(self, new_state: CompanionState) -> None:
        if new_state == self._current_state and new_state != CompanionState.ADDON:
            return

        if new_state == CompanionState.COLLAPSED:
            self._title.setText("COMPANION")
            self._stack.setCurrentWidget(self._placeholder)
            self._animate_width(0)
        elif new_state == CompanionState.ADDON:
            widget = self._addon_widgets.get(self._active_addon_id or "")
            if widget is None:
                self._stack.setCurrentWidget(self._placeholder)
                self._animate_width(0)
                new_state = CompanionState.COLLAPSED
            else:
                self._stack.setCurrentWidget(self._addon_panel)
                # Show the actual addon name (e.g. "MONOBASE", "CONNECT")
                # at the top of the companion pane instead of the generic
                # "ADDON" placeholder.
                actual_title = self._addon_titles.get(self._active_addon_id or "", "ADDON")
                self._title.setText(actual_title.upper())
                self._addon_stack.setCurrentWidget(widget)
                self._animate_width(self._user_width)
        else:
            panel = self.get_panel(new_state)
            if panel is not None:
                self._stack.setCurrentWidget(panel)
                self._title.setText(new_state.name.replace("_", " "))
                refresh = getattr(panel, "refresh", None)
                if callable(refresh):
                    refresh()
                self._animate_width(self._user_width)

        self._current_state = new_state
        self.sig_state_changed.emit(new_state.name)

    def _animate_width(self, target: int, duration_ms: int = 200) -> None:
        # If the user drag-resized to a width N, the minimumWidth was pinned
        # to N. A subsequent _animate_width(0) for collapse would otherwise
        # leave minimumWidth=N and Qt would refuse to shrink below it. Drop
        # minimumWidth back to 0 on collapse, and lift it to the target on
        # expand so the layout actually allocates the new width.
        if target == 0:
            self.setMinimumWidth(0)
        else:
            self.setMinimumWidth(target)
        anim = QPropertyAnimation(self, b"maximumWidth", self)
        anim.setDuration(duration_ms)
        anim.setStartValue(self.maximumWidth())
        anim.setEndValue(target)
        anim.setEasingCurve(QEasingCurve.OutCubic)
        anim.start()
        self._anim = anim
