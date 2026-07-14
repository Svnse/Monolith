from __future__ import annotations

from dataclasses import dataclass
import json
import re

from PySide6.QtCore import QEasingCurve, QEvent, QObject, QPoint, Qt, QPropertyAnimation, QTimer, Signal
from PySide6.QtGui import QColor, QKeyEvent
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsDropShadowEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from core.operators import OperatorManager
from core.paths import ARCHIVE_DIR, CONFIG_DIR
from core.themes import list_theme_entries
from core.history_search import search_archives

import core.style as _s


_TELEMETRY_FILE = CONFIG_DIR / "omni_telemetry.json"
_TELEMETRY_MAX_RECENTS = 20


@dataclass
class _Result:
    icon: str
    title: str
    subtitle: str
    action: str
    payload: object = None
    keywords: tuple[str, ...] = ()
    category: str = "General"
    aliases: tuple[str, ...] = ()


class _OmniResult(QFrame):
    clicked = Signal(str)
    hovered = Signal(str)

    def __init__(self, result: _Result, parent=None):
        super().__init__(parent)
        self._action = result.action
        self.setProperty("class", "OmniResultRow")
        self.setObjectName("omni_result_row")
        self.setCursor(Qt.PointingHandCursor)

        row = QHBoxLayout(self)
        row.setContentsMargins(0, 8, 10, 8)
        row.setSpacing(8)

        # 2px left accent marker (real widget instead of a CSS border so
        # there are no QSS edge-path rendering artifacts).
        self._accent = QFrame()
        self._accent.setObjectName("omni_result_accent")
        self._accent.setFixedWidth(2)
        self._accent.setStyleSheet("background: transparent; border: none;")
        row.addWidget(self._accent)
        row.addSpacing(8)

        title_lbl = QLabel(result.title)
        title_lbl.setStyleSheet(f"color: {_s.FG_TEXT}; font-size: 12px;")
        subtitle_text = f"{result.category} • {result.subtitle}" if result.subtitle else result.category
        subtitle_lbl = QLabel(subtitle_text)
        subtitle_lbl.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas;")

        row.addWidget(title_lbl)

        # Leader/connector line that fills the gap between title and
        # subtitle on hover. Hidden when the row is not selected; on
        # selection it becomes a 1px horizontal line in the accent color,
        # visually linking the title to the category/subtitle on the right.
        self._connector = QFrame()
        self._connector.setObjectName("omni_result_connector")
        self._connector.setFixedHeight(1)
        self._connector.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self._connector.setStyleSheet("background: transparent; border: none;")
        row.addWidget(self._connector, 1)

        row.addWidget(subtitle_lbl)

    def set_selected_visual(self, selected: bool) -> None:
        accent = _s.ACCENT_PRIMARY if selected else "transparent"
        self._accent.setStyleSheet(f"background: {accent}; border: none;")
        # Connector picks up FG_DIM (subdued) when active rather than full
        # accent, so it reads as a quiet visual link, not a second loud
        # marker competing with the left accent strip.
        connector_color = _s.FG_DIM if selected else "transparent"
        self._connector.setStyleSheet(
            f"background: {connector_color}; border: none;"
        )

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._action)
            event.accept()
            return
        super().mousePressEvent(event)

    def enterEvent(self, event) -> None:
        self.hovered.emit(self._action)
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        # Clear the row's own selection visual when the mouse moves off.
        # Without this the accent strip + connector line stay lit forever.
        # _selected_index in the parent is intentionally NOT reset here —
        # pressing arrow keys after a mouse leave will re-render the
        # current keyboard selection via _set_selected_index.
        self.set_selected_visual(False)
        super().leaveEvent(event)


class OmniBar(QWidget):
    sig_action_requested = Signal(str, object)
    sig_minimize = Signal()
    sig_maximize = Signal()
    sig_close = Signal()
    sig_focus_glow_changed = Signal(bool)

    def __init__(self, state, ui_bridge, parent=None):
        super().__init__(parent)
        self.state = state
        self.ui_bridge = ui_bridge
        self._registry = None
        self._result_payloads: dict[str, object] = {}
        self._result_widgets: list[_OmniResult] = []
        self._selected_index = -1
        self._results_popup: QFrame | None = None
        self._results_layout: QVBoxLayout | None = None
        self._action_usage: dict[str, int] = {}
        self._recent_actions: list[str] = []
        self._load_telemetry()
        self._operator_manager = OperatorManager()
        self._suppress_rebuild = False
        self._default_results_armed = False
        self._focus_glow_active = False
        self._history_timer = QTimer(self)
        self._history_timer.setSingleShot(True)
        self._history_timer.setInterval(300)
        self._history_timer.timeout.connect(self._rebuild_results)

        root = QHBoxLayout(self)
        root.setContentsMargins(12, 8, 12, 8)
        root.setSpacing(10)

        self._title = QLabel("MONOLITH")
        root.addWidget(self._title)
        root.addStretch()

        self._frame = QFrame()
        self._frame.setObjectName("omni_bar_frame")
        frame_layout = QVBoxLayout(self._frame)
        frame_layout.setContentsMargins(0, 0, 0, 0)
        frame_layout.setSpacing(0)

        top_row = QHBoxLayout()
        top_row.setContentsMargins(12, 0, 12, 0)
        top_row.setSpacing(8)

        self._prefix = QLabel("/")
        top_row.addWidget(self._prefix)

        self._input = QLineEdit()
        self._input.setObjectName("omni_bar_input")
        self._input.setPlaceholderText("Search commands, modules, history")
        # ClickFocus instead of StrongFocus so the omni bar input does not
        # auto-grab focus at app startup (Qt picks the first StrongFocus
        # widget on show). Ctrl+K still focuses it via explicit setFocus.
        self._input.setFocusPolicy(Qt.ClickFocus)
        self._input.textChanged.connect(self._on_text_changed)
        self._input.installEventFilter(self)
        top_row.addWidget(self._input, 1)

        self._badge = QLabel("READY")
        top_row.addWidget(self._badge)

        self._hint = QLabel("Ctrl+K")
        top_row.addWidget(self._hint)

        frame_layout.addLayout(top_row)

        self._frame.setFixedWidth(520)
        self._frame.setProperty("activeGlow", False)
        self._focus_glow = QGraphicsDropShadowEffect(self._frame)
        self._focus_glow.setOffset(0, 0)
        self._focus_glow.setBlurRadius(16)
        self._focus_glow.setColor(self._focus_glow_color())
        self._focus_glow.setEnabled(False)
        self._frame.setGraphicsEffect(self._focus_glow)
        self._glow_anim = QPropertyAnimation(self._focus_glow, b"blurRadius", self)
        self._glow_anim.setStartValue(16.0)
        self._glow_anim.setEndValue(30.0)
        self._glow_anim.setDuration(900)
        self._glow_anim.setEasingCurve(QEasingCurve.InOutSine)
        self._glow_anim.setLoopCount(-1)
        root.addWidget(self._frame)
        root.addStretch()

        controls = QHBoxLayout()
        controls.setSpacing(4)
        for label, signal in [("_", self.sig_minimize), ("[]", self.sig_maximize), ("X", self.sig_close)]:
            btn = QPushButton(label)
            btn.setFixedSize(24, 24)
            btn.clicked.connect(signal.emit)
            controls.addWidget(btn)
        root.addLayout(controls)

        # The chrome labels (title, prefix, badge, hint) and the results-popup
        # frame each carry per-widget stylesheets that f-string theme tokens.
        # Without a refresh hook those colors freeze to whatever theme was
        # active at construction — so on Monolithic they kept Midnight's
        # cool blue-grays. Apply once, then re-apply on every theme change.
        self._apply_chrome_theme()
        if hasattr(self.ui_bridge, "sig_theme_changed"):
            self.ui_bridge.sig_theme_changed.connect(self._apply_chrome_theme)

        # Application-wide click watcher: dismiss the results popup when
        # the user clicks anywhere that isn't the omni frame, the results
        # popup, or one of their descendants. The input's FocusOut handler
        # alone doesn't catch this because much of the chat surface has
        # FocusPolicy.NoFocus, so clicking there never steals focus from
        # the input.
        from PySide6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    def _apply_chrome_theme(self, *_args) -> None:
        self._focus_glow.setColor(self._focus_glow_color())
        self._title.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 14px; font-family: Consolas; font-weight: bold;"
        )
        self._prefix.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 12px; font-family: Consolas;"
        )
        self._badge.setStyleSheet(
            f"color: {_s.FG_SECONDARY}; font-size: 10px; font-family: Consolas; padding: 0 4px;"
        )
        self._hint.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 10px; font-family: Consolas;"
        )
        if self._results_popup is not None:
            self._results_popup.setStyleSheet(
                f"""
                QFrame#omni_bar_results_popup {{
                    background: {_s.BG_SURFACE_1};
                    border: 1px solid {_s.BORDER_LIGHT};
                    border-radius: 10px;
                }}
                """
            )

    def _focus_glow_color(self) -> QColor:
        color = QColor(_s.ACCENT_PRIMARY)
        color.setAlpha(170)
        return color

    def _set_focus_glow_active(self, active: bool) -> None:
        active = bool(active)
        if self._focus_glow_active == active:
            return
        self._focus_glow_active = active
        self._frame.setProperty("activeGlow", active)
        self._frame.style().unpolish(self._frame)
        self._frame.style().polish(self._frame)
        self._frame.update()

        if active:
            self._focus_glow.setColor(self._focus_glow_color())
            self._focus_glow.setEnabled(True)
            self._glow_anim.start()
        else:
            self._glow_anim.stop()
            self._focus_glow.setBlurRadius(16)
            self._focus_glow.setEnabled(False)

        self.sig_focus_glow_changed.emit(active)

    def bind_registry(self, registry) -> None:
        self._registry = registry

    def focus_input(self) -> None:
        self._default_results_armed = True
        self._set_focus_glow_active(True)
        self._input.setFocus(Qt.ShortcutFocusReason)
        self._input.selectAll()
        self._rebuild_results()

    def clear(self) -> None:
        self._default_results_armed = False
        self._set_focus_glow_active(False)
        self._suppress_rebuild = True
        self._input.clear()
        self._suppress_rebuild = False
        self._clear_results()

    def set_title(self, title: str) -> None:
        self._title.setText(title or "MONOLITH")

    def set_status(self, status_text: str) -> None:
        self._badge.setText(status_text or "READY")

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._input:
            if event.type() == QEvent.MouseButtonPress:
                self._default_results_armed = True
                self._set_focus_glow_active(True)
                QTimer.singleShot(0, self._rebuild_results)
            if event.type() == QEvent.KeyPress:
                key_event = event
                if isinstance(key_event, QKeyEvent):
                    return self._handle_key_press(key_event)
            if event.type() == QEvent.FocusIn:
                self._set_focus_glow_active(True)
                if self._default_results_armed or self._input.text().strip():
                    QTimer.singleShot(0, self._rebuild_results)
            if event.type() == QEvent.FocusOut:
                self._set_focus_glow_active(False)
                QTimer.singleShot(0, self._clear_results)
        elif event.type() == QEvent.MouseButtonPress:
            # Global click watcher: dismiss the results popup if the click
            # lands outside the omni frame + popup (and their descendants).
            if self._results_popup is not None and self._results_popup.isVisible():
                from PySide6.QtWidgets import QWidget as _QW
                target = watched if isinstance(watched, _QW) else None
                if target is not None and not self._click_inside_omni(target):
                    self.clear()
        return super().eventFilter(watched, event)

    def _click_inside_omni(self, widget) -> bool:
        # Walk up the parent chain; if we hit the omni frame, the input,
        # or the results popup, the click is "inside" and we leave the
        # popup open.
        node = widget
        while node is not None:
            if node is self._frame or node is self._results_popup or node is self._input:
                return True
            node = node.parent()
        return False

    def _handle_key_press(self, event: QKeyEvent) -> bool:
        if event.key() == Qt.Key_Escape:
            self.clear()
            self.clearFocus()
            return True
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            self._activate_selected()
            return True
        if event.key() == Qt.Key_Down:
            self._set_selected_index(self._selected_index + 1)
            return True
        if event.key() == Qt.Key_Up:
            self._set_selected_index(self._selected_index - 1)
            return True
        return False

    def _on_text_changed(self, text: str) -> None:
        if self._suppress_rebuild:
            return
        if text.strip().startswith("/history"):
            self._history_timer.start()
            return
        self._rebuild_results()

    def _ensure_results_popup(self) -> None:
        host = self.window()
        if self._results_popup is not None and self._results_popup.parentWidget() is host:
            return

        if self._results_popup is not None:
            self._results_popup.deleteLater()

        self._results_popup = QFrame(host)
        self._results_popup.setObjectName("omni_bar_results_popup")
        self._results_popup.setStyleSheet(
            f"""
            QFrame#omni_bar_results_popup {{
                background: {_s.BG_SURFACE_1};
                border: 1px solid {_s.BORDER_LIGHT};
                border-radius: 10px;
            }}
            """
        )
        layout = QVBoxLayout(self._results_popup)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(4)
        self._results_layout = layout
        self._results_popup.hide()

    def _position_results_popup(self) -> None:
        if self._results_popup is None:
            return
        host = self.window()
        anchor = self._frame.mapTo(host, QPoint(0, self._frame.height() + 6))
        width = min(max(self._frame.width(), 520), max(320, host.width() - anchor.x() - 12))
        height = min(self._results_popup.sizeHint().height(), max(120, host.height() - anchor.y() - 12))
        self._results_popup.setGeometry(anchor.x(), anchor.y(), width, height)

    def _input_has_focus(self) -> bool:
        return self._input.hasFocus()

    def _rebuild_results(self) -> None:
        self._clear_results()
        query = self._input.text().strip()
        if not query and (not self._input_has_focus() or not self._default_results_armed):
            return
        results = self._query(query)
        if not results:
            return
        self._ensure_results_popup()
        assert self._results_layout is not None
        for result in results:
            row = _OmniResult(result)
            row.clicked.connect(self._emit_action)
            row.hovered.connect(self._select_action)
            self._result_widgets.append(row)
            self._result_payloads[result.action] = result.payload
            self._results_layout.addWidget(row)
        self._position_results_popup()
        self._results_popup.show()
        self._results_popup.raise_()
        self._set_selected_index(0)

    def _clear_results(self) -> None:
        if self._results_layout is None:
            return
        while self._results_layout.count():
            item = self._results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._result_widgets.clear()
        self._result_payloads.clear()
        self._selected_index = -1
        if self._results_popup is not None:
            self._results_popup.hide()

    def _set_selected_index(self, index: int) -> None:
        if not self._result_widgets:
            self._selected_index = -1
            return
        self._selected_index = max(0, min(index, len(self._result_widgets) - 1))
        for i, widget in enumerate(self._result_widgets):
            selected = i == self._selected_index
            bg = _s.BG_BUTTON_HOVER if selected else "transparent"
            # objectName-scoped so the rule does NOT cascade into the
            # row's QLabel descendants. No border properties — the accent
            # indicator is a real 2px QFrame inside the row (see
            # _OmniResult.__init__), which avoids the QSS edge-path
            # hairline artifact that border-left + border-radius produced.
            widget.setStyleSheet(
                f"""
                QFrame#omni_result_row {{
                    background: {bg};
                    border-radius: 6px;
                }}
                """
            )
            widget.set_selected_visual(selected)

    def _activate_selected(self) -> None:
        if not self._result_widgets or self._selected_index < 0:
            return
        self._emit_action(self._result_widgets[self._selected_index]._action)

    def _select_action(self, action: str) -> None:
        for i, widget in enumerate(self._result_widgets):
            if widget._action == action:
                self._set_selected_index(i)
                break

    def _emit_action(self, action: str) -> None:
        if action == "noop":
            self.clear()
            return
        self._action_usage[action] = int(self._action_usage.get(action, 0)) + 1
        self._recent_actions = [a for a in self._recent_actions if a != action]
        self._recent_actions.insert(0, action)
        if len(self._recent_actions) > _TELEMETRY_MAX_RECENTS:
            self._recent_actions = self._recent_actions[:_TELEMETRY_MAX_RECENTS]
        self._save_telemetry()
        payload = self._result_payloads.get(action)
        self.sig_action_requested.emit(action, payload)
        self.clear()

    def _load_telemetry(self) -> None:
        # Persist across launches so the default-result ordering reflects
        # actual usage rather than the hardcoded list in _default_results().
        # Best-effort: a malformed file shouldn't break the omni bar.
        try:
            with open(_TELEMETRY_FILE, "r", encoding="utf-8") as fh:
                payload = json.load(fh)
        except (FileNotFoundError, OSError, json.JSONDecodeError):
            return
        usage = payload.get("usage") if isinstance(payload, dict) else None
        if isinstance(usage, dict):
            self._action_usage = {
                str(k): int(v) for k, v in usage.items()
                if isinstance(v, (int, float))
            }
        recents = payload.get("recents") if isinstance(payload, dict) else None
        if isinstance(recents, list):
            self._recent_actions = [str(a) for a in recents if isinstance(a, str)][:_TELEMETRY_MAX_RECENTS]

    def _save_telemetry(self) -> None:
        try:
            _TELEMETRY_FILE.parent.mkdir(parents=True, exist_ok=True)
            tmp = _TELEMETRY_FILE.with_suffix(_TELEMETRY_FILE.suffix + ".tmp")
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump({
                    "usage": self._action_usage,
                    "recents": self._recent_actions,
                }, fh, ensure_ascii=False, indent=2)
            tmp.replace(_TELEMETRY_FILE)
        except OSError:
            # Telemetry is best-effort — never fail an action because we
            # couldn't write the counter file.
            pass

    @staticmethod
    def _normalize(text: str) -> str:
        return re.sub(r"[^a-z0-9]+", " ", str(text or "").lower()).strip()

    @staticmethod
    def _is_subsequence(needle: str, haystack: str) -> bool:
        if not needle:
            return False
        idx = 0
        for ch in haystack:
            if idx < len(needle) and ch == needle[idx]:
                idx += 1
        return idx == len(needle)

    def _match_score(self, query: str, *texts: str) -> int:
        q_norm = self._normalize(query)
        if not q_norm:
            return 1

        q_tokens = q_norm.split()
        q_compact = q_norm.replace(" ", "")
        best = 0
        for text in texts:
            t_norm = self._normalize(text)
            if not t_norm:
                continue
            words = t_norm.split()
            compact = "".join(words)
            if q_norm == t_norm:
                best = max(best, 360)
            if t_norm.startswith(q_norm):
                best = max(best, 320)
            if q_norm in t_norm:
                best = max(best, 280)
            if q_tokens:
                starts_or_exact = sum(
                    1 for tok in q_tokens if any(word == tok or word.startswith(tok) for word in words)
                )
                if starts_or_exact:
                    coverage = starts_or_exact / max(1, len(q_tokens))
                    best = max(best, int(220 + coverage * 90))
                if all(any(tok in word for word in words) for tok in q_tokens):
                    best = max(best, 240 + len(q_tokens) * 5)
            if self._is_subsequence(q_compact, compact):
                best = max(best, 160)
        return best

    @staticmethod
    def _contains_tokens(haystack: str, *tokens: str) -> bool:
        words = set(haystack.split())
        return all(token in words for token in tokens)

    def _intent_boost_actions(self, query: str) -> set[str]:
        q = self._normalize(query)
        actions: set[str] = set()
        if not q:
            return actions

        # Natural language intents so users don't need command syntax.
        if (
            self._contains_tokens(q, "new", "chat")
            or self._contains_tokens(q, "new", "session")
            or "start over" in q
            or "fresh chat" in q
        ):
            actions.add("cmd:new")
        if "history" in q or "archive" in q or "past chat" in q:
            actions.add("panel:ARCHIVE")
        if "databank" in q or "file browser" in q or "browse files" in q:
            actions.add("panel:DATABANK")
        if "config" in q or "settings" in q or "model settings" in q:
            actions.add("panel:CONFIG")
        if "audit" in q or "approval" in q:
            actions.add("panel:AUDIT")
        if "generating" in q or "running tasks" in q:
            actions.add("panel:GENERATING")
        if "expedition" in q or "explore" in q or "curiosity" in q:
            actions.add("panel:EXPEDITION")
        if "reasoning" in q or "branch" in q or "reasoning tree" in q:
            actions.add("panel:REASONING_TREE")
        if "workshop" in q or "workflow" in q or "card library" in q:
            actions.add("panel:WORKSHOP")
        if "maintenance" in q or "self maint" in q or "triage" in q or "mad cow" in q:
            actions.add("panel:SELF_MAINT")
        if "load model" in q or self._contains_tokens(q, "load", "model"):
            actions.add("cmd:load")
        if "unload model" in q or self._contains_tokens(q, "unload", "model"):
            actions.add("cmd:unload")
        if "vitals" in q or "resource usage" in q:
            actions.add("cmd:vitals")
        if "monitor" in q or "logs" in q or "trace" in q:
            actions.add("cmd:monitor")
        return actions

    def _default_results(self) -> list[_Result]:
        return [
            _Result(
                ">",
                "Refresh theme",
                "force re-apply current theme to every widget",
                "cmd:refresh-theme",
                keywords=("refresh", "theme", "redraw", "restyle", "reapply"),
                category="Command",
                aliases=("reapply theme", "force theme refresh", "redraw theme"),
            ),
            _Result(
                ">",
                "Monitor",
                "logs, traces, overseer",
                "cmd:monitor",
                keywords=("logs", "trace", "debug", "overseer"),
                category="Command",
                aliases=("open monitor", "show logs"),
            ),
            _Result(
                ">",
                "Vitals",
                "resource footer",
                "cmd:vitals",
                keywords=("resources", "status", "usage"),
                category="Command",
                aliases=("resource usage", "system usage"),
            ),
            _Result(
                ">",
                "New chat",
                "fresh session",
                "cmd:new",
                keywords=("new", "chat", "session"),
                category="Command",
                aliases=("start over", "new session"),
            ),
            _Result(
                "G",
                "Generating",
                "companion panel",
                "panel:GENERATING",
                "GENERATING",
                keywords=("running", "generation", "active"),
                category="Panel",
                aliases=("running tasks",),
            ),
            _Result(
                "C",
                "Config",
                "companion panel",
                "panel:CONFIG",
                "CONFIG",
                keywords=("settings", "model", "config"),
                category="Panel",
                aliases=("open settings", "model settings"),
            ),
            _Result(
                "H",
                "History",
                "companion panel",
                "panel:ARCHIVE",
                "ARCHIVE",
                keywords=("archive", "search", "past chats"),
                category="Panel",
                aliases=("past chats", "chat history"),
            ),
            _Result(
                "A",
                "Audit",
                "companion panel",
                "panel:AUDIT",
                "AUDIT",
                keywords=("actions", "approval", "review"),
                category="Panel",
                aliases=("approvals", "action review"),
            ),
            _Result(
                "D",
                "Databank",
                "companion panel",
                "panel:DATABANK",
                "DATABANK",
                keywords=("files", "knowledge", "browser"),
                category="Panel",
                aliases=("file browser", "open files"),
            ),
            _Result(
                "E",
                "Expedition",
                "companion panel · self-directed exploration",
                "panel:EXPEDITION",
                "EXPEDITION",
                keywords=("expedition", "explore", "curiosity", "autonomous", "monoexplore"),
                category="Panel",
                aliases=("explore", "start expedition", "autonomous explore"),
            ),
            _Result(
                "M",
                "Self-Maintenance",
                "companion panel · autonomous review-queue triage daemon",
                "panel:SELF_MAINT",
                "SELF_MAINT",
                keywords=("self", "maintenance", "daemon", "triage", "review", "queue", "mad cow", "autonomous"),
                category="Panel",
                aliases=("self maintenance", "maintenance daemon", "review daemon"),
            ),
            _Result(
                "R",
                "Reasoning Tree",
                "companion panel · reasoning branch navigator",
                "panel:REASONING_TREE",
                "REASONING_TREE",
                keywords=("reasoning", "branch", "tree", "fork", "take", "navigator"),
                category="Panel",
                aliases=("branch tree", "reasoning branches", "open reasoning tree"),
            ),
            _Result(
                "W",
                "Workshop",
                "companion panel · workflow card library + runs",
                "panel:WORKSHOP",
                "WORKSHOP",
                keywords=("workshop", "workflow", "cards", "library", "runs", "monoline"),
                category="Panel",
                aliases=("card library", "workflows", "open workshop"),
            ),
        ]

    def _sort_by_usage(self, results: list[_Result]) -> list[_Result]:
        # Stable sort by descending usage count, then by recency. Items with
        # no telemetry keep their hardcoded order (the original list is the
        # tiebreaker for unused entries).
        recents_index = {a: i for i, a in enumerate(self._recent_actions)}

        def key(item_index: tuple[int, _Result]) -> tuple[int, int, int]:
            i, r = item_index
            usage = -int(self._action_usage.get(r.action, 0))
            recency = recents_index.get(r.action, len(self._recent_actions) + 1)
            return (usage, recency, i)

        return [r for _i, r in sorted(enumerate(results), key=key)]

    def _query(self, query: str) -> list[_Result]:
        if not query:
            return self._sort_by_usage(self._default_results())
        if query.startswith(">model "):
            q = query.split(" ", 1)[1].strip().lower()
            current_models = list(dict.fromkeys(str(item) for item in self.state.__dict__.get("_omni_models", []) if str(item).strip()))
            return [
                _Result("M", model, "model", f"model:{model}", model, keywords=("model",), category="Model")
                for model in current_models
                if q in model.lower()
            ]
        if query.startswith(">theme "):
            q = query.split(" ", 1)[1].strip().lower()
            return [
                _Result("*", name, key, f"theme:{key}", key, keywords=("theme", "appearance"), category="Theme")
                for key, name, _builtin in list_theme_entries()
                if q in name.lower() or q in key.lower()
            ]
        if query.startswith(">profile "):
            q = query.split(" ", 1)[1].strip().lower()
            if not q or "new".startswith(q):
                return [
                    _Result(
                        "P",
                        "Create profile",
                        "snapshot current workspace",
                        "profile:new",
                        "new",
                        keywords=("profile", "workspace", "save"),
                        category="Profile",
                    )
                ]
            return [
                _Result(
                    "P",
                    item["name"],
                    "profile",
                    f"profile:{item['name']}",
                    item["name"],
                    keywords=("profile",),
                    category="Profile",
                )
                for item in self._operator_manager.list_operators()
                if q in item["name"].lower()
            ]
        if query.startswith(">lineage "):
            q = query.split(" ", 1)[1].strip().lower()
            return [
                _Result(
                    "L",
                    item["name"],
                    "profile lineage",
                    f"lineage:{item['name']}",
                    item["name"],
                    keywords=("lineage", "profile", "history"),
                    category="Profile",
                )
                for item in self._operator_manager.list_operators()
                if not q or q in item["name"].lower()
            ]
        if query.startswith("/agent "):
            return [
                _Result(
                    "A",
                    query,
                    "apply agent command",
                    "agent:apply",
                    query,
                    keywords=("agent", "command"),
                    category="Agent",
                )
            ]
        if query.startswith("/history "):
            q = query.split(" ", 1)[1].strip()
            return [
                _Result(
                    "H",
                    result.path.stem,
                    result.snippet[:60],
                    f"history:{result.path}",
                    str(result.path),
                    keywords=("history", "archive"),
                    category="History",
                )
                for result in search_archives(q, ARCHIVE_DIR, max_results=8)
            ]
        direct = {
            ">load": _Result(
                ">",
                "Load model",
                "current conversation",
                "cmd:load",
                keywords=("load", "model", "llm"),
                category="Command",
                aliases=("start model",),
            ),
            ">unload": _Result(
                ">",
                "Unload model",
                "current conversation",
                "cmd:unload",
                keywords=("unload", "model", "free memory"),
                category="Command",
                aliases=("stop model",),
            ),
            ">new": _Result(
                ">",
                "New chat",
                "fresh session",
                "cmd:new",
                keywords=("new", "chat", "session"),
                category="Command",
                aliases=("reset chat",),
            ),
            ">vitals": _Result(
                ">",
                "Toggle vitals",
                "footer",
                "cmd:vitals",
                keywords=("vitals", "resources", "status"),
                category="Command",
            ),
            ">monitor": _Result(
                ">",
                "Open monitor",
                "overseer",
                "cmd:monitor",
                keywords=("monitor", "logs", "trace", "debug"),
                category="Command",
            ),
            ">expedition": _Result(
                "E",
                "Open Expedition",
                "self-directed exploration (MonoExplore)",
                "panel:EXPEDITION",
                "EXPEDITION",
                keywords=("expedition", "explore", "curiosity", "monoexplore"),
                category="Panel",
            ),
            ">selfmaint": _Result(
                "M",
                "Open Self-Maintenance",
                "autonomous review-queue triage daemon",
                "panel:SELF_MAINT",
                "SELF_MAINT",
                keywords=("self", "maintenance", "daemon", "triage"),
                category="Panel",
            ),
            ">workshop": _Result(
                "W",
                "Open Workshop",
                "workflow card library + runs",
                "panel:WORKSHOP",
                "WORKSHOP",
                keywords=("workshop", "workflow", "cards", "runs"),
                category="Panel",
            ),
            ">reasoning": _Result(
                "R",
                "Open Reasoning Tree",
                "reasoning branch navigator",
                "panel:REASONING_TREE",
                "REASONING_TREE",
                keywords=("reasoning", "branch", "tree", "navigator"),
                category="Panel",
            ),
        }
        if query in direct:
            return [direct[query]]

        intent_actions = self._intent_boost_actions(query)
        ranked: dict[str, tuple[int, _Result]] = {}

        def add_result(result: _Result, *search_terms: str) -> None:
            score = self._match_score(
                query,
                result.title,
                result.subtitle,
                result.action,
                result.category,
                *result.keywords,
                *result.aliases,
                *search_terms,
            )
            if score <= 0:
                return
            if result.action in intent_actions:
                score += 140
            score += min(int(self._action_usage.get(result.action, 0)) * 4, 24)
            if result.action in self._recent_actions[:5]:
                score += max(4, 16 - self._recent_actions.index(result.action) * 3)
            current = ranked.get(result.action)
            if current is None or score > current[0]:
                ranked[result.action] = (score, result)

        for result in list(direct.values()) + self._default_results():
            add_result(result)

        if query.startswith(">"):
            command_query = query[1:]
            for result in direct.values():
                add_result(result, command_query)

        for item in self._operator_manager.list_operators():
            add_result(
                _Result(
                    "P",
                    item["name"],
                    "profile",
                    f"profile:{item['name']}",
                    item["name"],
                    keywords=("profile", "workspace"),
                    category="Profile",
                ),
                "profile",
                "lineage",
            )

        for key, name, _builtin in list_theme_entries():
            add_result(
                _Result(
                    "*",
                    name,
                    key,
                    f"theme:{key}",
                    key,
                    keywords=("theme", "appearance"),
                    category="Theme",
                )
            )

        for model in dict.fromkeys(str(item) for item in self.state.__dict__.get("_omni_models", []) if str(item).strip()):
            add_result(
                _Result(
                    "M",
                    model,
                    "model",
                    f"model:{model}",
                    model,
                    keywords=("model", "llm"),
                    category="Model",
                )
            )

        if self._registry is not None:
            for spec in self._registry.all():
                if spec.kind == "page":
                    continue
                if spec.id == "databank":
                    continue
                icon = spec.icon or ("P" if spec.kind == "page" else "*")
                add_result(
                    _Result(
                        icon,
                        spec.title,
                        spec.kind,
                        f"{spec.kind}:{spec.id}",
                        spec.id,
                        keywords=(spec.id, spec.kind),
                        category="Module" if spec.kind == "addon" else spec.kind.title(),
                    ),
                )

        category_order = {
            "Command": 0,
            "Panel": 1,
            "Module": 2,
            "Profile": 3,
            "Model": 4,
            "Theme": 5,
            "History": 6,
            "Agent": 7,
            "General": 9,
        }
        ordered = sorted(
            ranked.values(),
            key=lambda item: (
                -item[0],
                category_order.get(item[1].category, 99),
                item[1].title.lower(),
            ),
        )
        results = [result for _score, result in ordered[:12]]
        if results:
            return results
        return self._default_results()[:6]

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_results_popup()
