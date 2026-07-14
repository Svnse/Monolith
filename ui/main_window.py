from __future__ import annotations

from typing import Optional
from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QDateTime, QTimer
from PySide6.QtGui import QCursor, QDragEnterEvent, QDragLeaveEvent, QDropEvent, QKeyEvent, QMouseEvent, QShortcut, QKeySequence
from PySide6.QtWidgets import QFrame, QHBoxLayout, QMainWindow, QStackedWidget, QVBoxLayout, QWidget

from core.operators import OperatorManager
from core.state import AppState, SystemStatus
from ui.bridge import UIBridge
from ui.addons.host import AddonHost
from ui.companion_pane import CompanionPane, CompanionState
from ui.components.complex import GradientLine
from ui.components.drop_zone import DropZoneOverlay
from ui.dialogs.profile_dialogs import LineageDialog, NameDialog
from ui.icon_rail import IconRail
from ui.omni_bar import OmniBar
from ui.vitals_footer import VitalsFooter


class _WidgetRegistry:
    def __init__(self) -> None:
        self._widgets: list[QWidget] = []
        self._current: QWidget | None = None

    def addWidget(self, widget: QWidget) -> None:
        if widget not in self._widgets:
            self._widgets.append(widget)
            if self._current is None:
                self._current = widget

    def removeWidget(self, widget: QWidget) -> None:
        if widget not in self._widgets:
            return
        self._widgets.remove(widget)
        if self._current is widget:
            self._current = self._widgets[-1] if self._widgets else None

    def count(self) -> int:
        return len(self._widgets)

    def widget(self, index: int) -> QWidget:
        return self._widgets[index]

    def currentWidget(self) -> QWidget | None:
        return self._current

    def setCurrentWidget(self, widget: QWidget) -> None:
        if widget in self._widgets:
            self._current = widget


class MonolithUI(QMainWindow):
    def __init__(self, state: AppState, ui_bridge: UIBridge):
        super().__init__()
        self.state = state
        self.ui_bridge = ui_bridge
        self.host: Optional[AddonHost] = None
        self.stack = _WidgetRegistry()
        self._drag_pos = None
        self._normal_geometry = None
        self._active_terminal_mod_id: str | None = None
        self._terminal_titles: dict[str, tuple[str, str]] = {}
        self._module_order: list[str] = []
        self._selected_module_id: str | None = None
        self._ctx_refresh_timer = QTimer(self)
        self._ctx_refresh_timer.setSingleShot(True)
        self._ctx_refresh_timer.setInterval(150)
        self._ctx_refresh_timer.timeout.connect(self._flush_ctx_update)

        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setMouseTracking(True)
        self.resize(1100, 700)
        self.setMinimumSize(720, 460)
        self.setAcceptDrops(True)

        self._resize_edge = None
        self._resize_origin = None
        self._resize_geo = None
        self._GRIP = 4

        main_widget = QWidget()
        main_widget.setObjectName("MainFrame")
        main_widget.setMouseTracking(True)
        self.setCentralWidget(main_widget)

        root = QVBoxLayout(main_widget)
        root.setContentsMargins(1, 1, 1, 1)
        root.setSpacing(0)

        self.gradient_line = GradientLine()
        root.addWidget(self.gradient_line)

        self.omni_bar = OmniBar(state, ui_bridge)
        self.omni_bar.sig_action_requested.connect(self._handle_omni_action)
        self.omni_bar.sig_minimize.connect(self.showMinimized)
        self.omni_bar.sig_maximize.connect(self.toggle_maximize)
        self.omni_bar.sig_close.connect(self.close)
        self.omni_bar.sig_focus_glow_changed.connect(self._on_omni_focus_glow_changed)
        root.addWidget(self.omni_bar)

        content = QHBoxLayout()
        content.setContentsMargins(0, 0, 0, 0)
        content.setSpacing(0)

        self._conversation_stack = QStackedWidget()
        self._conversation_stack.setObjectName("conversation_stack")
        content.addWidget(self._conversation_stack, 1)

        self.icon_rail = IconRail()
        self.icon_rail.sig_panel_requested.connect(self._handle_rail_request)
        content.addWidget(self.icon_rail)

        self.companion = CompanionPane(state, ui_bridge)
        self.companion.sig_state_changed.connect(self._on_companion_state_changed)
        content.addWidget(self.companion)

        content_host = QWidget()
        content_host.setLayout(content)
        root.addWidget(content_host, 1)

        self.vitals = VitalsFooter(state, ui_bridge=self.ui_bridge)
        self.vitals.sig_unload_requested.connect(self._handle_footer_unload)
        root.addWidget(self.vitals)

        self.drop_zone: DropZoneOverlay | None = None
        self.ui_bridge.sig_terminal_header.connect(self.update_terminal_header)
        if hasattr(self.ui_bridge, "sig_world_action_pending"):
            self.ui_bridge.sig_world_action_pending.connect(lambda _action: self.companion.refresh_from_world())
        if hasattr(self.ui_bridge, "sig_world_action_approved"):
            self.ui_bridge.sig_world_action_approved.connect(lambda _action: self.companion.refresh_from_world())
        if hasattr(self.ui_bridge, "sig_world_action_rejected"):
            self.ui_bridge.sig_world_action_rejected.connect(lambda _action: self.companion.refresh_from_world())
        self._install_drag_filters()
        self._install_shortcuts()

    def attach_host(self, host: AddonHost) -> None:
        self.host = host
        self.omni_bar.bind_registry(host.registry)

        self.drop_zone = DropZoneOverlay(self.centralWidget(), host.registry, host)
        # Autonomous Connect: when MONOLITH_AGENT_AUTOSTART is set, instantiate the
        # CONNECT module at boot (it is normally on-demand, so its agent-server
        # autostart never fired until the tab was opened). Launched before terminal
        # so the chat stays the focused surface.
        import os as _os
        if _os.getenv("MONOLITH_AGENT_AUTOSTART", "").strip().lower() in ("1", "true", "yes", "on"):
            if self._addon_widget("connections") is None:
                # Headless: the page owns the agent-server (it must exist for the
                # server to autostart), but its screen must not open on boot.
                host.launch_module("connections", focus=False)
        if not self._terminal_mod_ids():
            host.launch_module("terminal")

    def _active_conversation(self) -> QWidget | None:
        current = self._conversation_stack.currentWidget()
        if current is not None and getattr(current, "_addon_id", None) == "terminal":
            return current
        if self._active_terminal_mod_id:
            widget = self._widget_for_mod(self._active_terminal_mod_id)
            if widget is not None and getattr(widget, "_addon_id", None) == "terminal":
                return widget
        terminal_ids = self._terminal_mod_ids()
        if terminal_ids:
            return self._widget_for_mod(terminal_ids[-1])
        return None

    def _refresh_omni_models(self) -> None:
        active = self._active_conversation()
        models: list[str] = []
        if active is not None:
            combo = getattr(active, "api_model_combo", None)
            if combo is not None:
                for index in range(combo.count()):
                    text = combo.itemText(index).strip()
                    if text and text not in models:
                        models.append(text)
            config = getattr(active, "config", {}) or {}
            for value in (
                config.get("api_model"),
                Path(str(config.get("gguf_path", ""))).name if config.get("gguf_path") else "",
            ):
                text = str(value or "").strip()
                if text and text not in models:
                    models.insert(0, text)
        self.state._omni_models = models

    def _on_omni_focus_glow_changed(self, active: bool) -> None:
        self.gradient_line.set_glow_active(not active)

    # ---------------- window behavior ----------------

    def _install_drag_filters(self) -> None:
        self._drag_filter_widgets = [
            self.omni_bar,
            getattr(self.omni_bar, "_title", None),
            getattr(self.omni_bar, "_frame", None),
            getattr(self.omni_bar, "_prefix", None),
            getattr(self.omni_bar, "_badge", None),
            getattr(self.omni_bar, "_hint", None),
        ]
        for widget in self._drag_filter_widgets:
            if widget is not None:
                widget.installEventFilter(self)

    def _install_shortcuts(self) -> None:
        # Keep omni launcher reliable regardless of focused child widget.
        self._shortcut_omni_ctrl = QShortcut(QKeySequence("Ctrl+K"), self)
        self._shortcut_omni_ctrl.setContext(Qt.WidgetWithChildrenShortcut)
        self._shortcut_omni_ctrl.activated.connect(self.omni_bar.focus_input)
        self._shortcut_omni_meta = QShortcut(QKeySequence("Meta+K"), self)
        self._shortcut_omni_meta.setContext(Qt.WidgetWithChildrenShortcut)
        self._shortcut_omni_meta.activated.connect(self.omni_bar.focus_input)

    def _is_top_drag_zone(self, pos) -> bool:
        return pos.y() < 48

    def _begin_window_drag(self, global_pos, local_pos) -> None:
        if self.isMaximized():
            ratio = local_pos.x() / max(1, self.width())
            normal_geo = self._normal_geometry or self.geometry()
            self.showNormal()
            if self._normal_geometry:
                self.setGeometry(self._normal_geometry)
            new_w = normal_geo.width()
            new_h = normal_geo.height()
            new_x = global_pos.x() - int(new_w * ratio)
            new_y = global_pos.y() - min(local_pos.y(), 40)
            self.setGeometry(new_x, new_y, new_w, new_h)
        self._drag_pos = global_pos - self.frameGeometry().topLeft()

    def eventFilter(self, watched, event):
        if watched in getattr(self, "_drag_filter_widgets", []):
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                local_pos = self.mapFromGlobal(event.globalPosition().toPoint())
                if self._is_top_drag_zone(local_pos):
                    self._begin_window_drag(event.globalPosition().toPoint(), local_pos)
                    event.accept()
                    return True
            elif event.type() == QEvent.MouseMove and self._drag_pos and event.buttons() & Qt.LeftButton:
                self.move(event.globalPosition().toPoint() - self._drag_pos)
                event.accept()
                return True
            elif event.type() == QEvent.MouseButtonRelease and self._drag_pos:
                self._drag_pos = None
                event.accept()
                return True
        return super().eventFilter(watched, event)

    def _edge_at(self, pos):
        g = self._GRIP * 4
        r = self.rect()
        if pos.x() >= r.width() - g and pos.y() >= r.height() - g:
            return "right+bottom"
        return None

    _CURSOR_MAP = {
        "right+bottom": Qt.SizeFDiagCursor,
    }

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() != Qt.LeftButton:
            return
        pos = event.position().toPoint()
        edge = self._edge_at(pos)
        if edge:
            self._resize_edge = edge
            self._resize_origin = event.globalPosition().toPoint()
            self._resize_geo = self.geometry()
            event.accept()
            return
        if self._is_top_drag_zone(pos):
            self._begin_window_drag(event.globalPosition().toPoint(), pos)
            event.accept()

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._resize_edge and event.buttons() == Qt.LeftButton:
            delta = event.globalPosition().toPoint() - self._resize_origin
            geo = self._resize_geo
            new_geo = geo.__class__(geo)
            if "right" in self._resize_edge:
                new_geo.setRight(geo.right() + delta.x())
            if "bottom" in self._resize_edge:
                new_geo.setBottom(geo.bottom() + delta.y())
            if "left" in self._resize_edge:
                new_geo.setLeft(geo.left() + delta.x())
            if "top" in self._resize_edge:
                new_geo.setTop(geo.top() + delta.y())
            if new_geo.width() >= self.minimumWidth() and new_geo.height() >= self.minimumHeight():
                self.setGeometry(new_geo)
            event.accept()
            return
        if self._drag_pos and event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        edge = self._edge_at(event.position().toPoint())
        if edge and edge in self._CURSOR_MAP:
            self.setCursor(self._CURSOR_MAP[edge])
        else:
            self.unsetCursor()

    def mouseReleaseEvent(self, event: QMouseEvent):
        self._drag_pos = None
        self._resize_edge = None
        self._resize_origin = None
        self._resize_geo = None
        super().mouseReleaseEvent(event)

    def keyPressEvent(self, event: QKeyEvent):
        if event.modifiers() == Qt.ControlModifier and event.key() == Qt.Key_K:
            self.omni_bar.focus_input()
            event.accept()
            return
        super().keyPressEvent(event)

    def dragEnterEvent(self, event: QDragEnterEvent):
        if event.mimeData().hasUrls() and self.drop_zone is not None:
            event.acceptProposedAction()
            self.drop_zone.activate(event)

    def dragLeaveEvent(self, event: QDragLeaveEvent):
        if self.drop_zone is not None:
            self.drop_zone.deactivate()
        super().dragLeaveEvent(event)

    def dropEvent(self, event: QDropEvent):
        if self.drop_zone is not None:
            self.drop_zone.handle_drop(event)
        event.acceptProposedAction()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not self.isMaximized():
            self._normal_geometry = self.geometry()
        if self.drop_zone is not None and self.drop_zone.isVisible():
            self.drop_zone.setGeometry(self.centralWidget().rect())

    def toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
            if self._normal_geometry:
                self.setGeometry(self._normal_geometry)
        else:
            self._normal_geometry = self.geometry()
            self.showMaximized()

    def toggle_vitals(self):
        self.vitals.toggle_expanded()

    # ---------------- compatibility helpers ----------------

    def _widget_for_mod(self, mod_id: str) -> QWidget | None:
        for i in range(self.stack.count()):
            widget = self.stack.widget(i)
            if getattr(widget, "_mod_id", None) == mod_id:
                return widget
        return None

    def _addon_widget(self, addon_id: str) -> QWidget | None:
        for i in range(self.stack.count()):
            widget = self.stack.widget(i)
            if getattr(widget, "_addon_id", None) == addon_id:
                return widget
        return None

    def reveal_attachment(self, att) -> None:
        """Open Databank and show the attachment the user clicked."""
        if self.host is None:
            return
        page = self._addon_widget("databank")
        if page is None:
            self.host.launch_module("databank")
            page = self._addon_widget("databank")
        if page is None:
            return
        mod_id = getattr(page, "_mod_id", None)
        if mod_id:
            try:
                self.switch_to_module(mod_id)
            except Exception:
                pass
        opener = getattr(page, "open_attachment", None)
        if callable(opener):
            try:
                opener(att)
            except Exception:
                pass

    def accept_external_text_to_chat(
        self,
        text: str,
        *,
        label: str = "external text",
        force_attachment: bool = False,
    ) -> str:
        chat = self._active_conversation()
        if chat is None and self.host is not None:
            self.host.launch_module("terminal")
            chat = self._active_conversation()
        if chat is None or not callable(getattr(chat, "accept_external_text", None)):
            return "unavailable"
        result = chat.accept_external_text(
            text,
            label=label,
            force_attachment=force_attachment,
        )
        self._show_chat_workspace()
        return str(result)

    def _terminal_mod_ids(self) -> list[str]:
        ids: list[str] = []
        for i in range(self.stack.count()):
            widget = self.stack.widget(i)
            if getattr(widget, "_addon_id", None) == "terminal":
                ids.append(str(getattr(widget, "_mod_id", "")))
        return [item for item in ids if item]

    def _show_chat_workspace(self) -> bool:
        mod_id = self._active_terminal_mod_id
        if not mod_id:
            terminal_ids = self._terminal_mod_ids()
            mod_id = terminal_ids[-1] if terminal_ids else None
        if mod_id:
            self.switch_to_module(mod_id)
            return True
        if self.host is not None:
            return bool(self.host.launch_module("terminal"))
        return False

    def register_module(self, mod_id: str, addon_id: str, icon: str, title: str) -> None:
        if mod_id not in self._module_order:
            self._module_order.append(mod_id)
        if addon_id and addon_id != "terminal":
            self.icon_rail.add_addon_icon(mod_id, icon or "*", title)

    def unregister_module(self, mod_id: str) -> None:
        self._module_order = [item for item in self._module_order if item != mod_id]
        self.icon_rail.remove_addon_icon(mod_id)
        if self._selected_module_id == mod_id:
            self._selected_module_id = None
            self.icon_rail.set_active(None)

    def get_module_order(self) -> list[str]:
        return list(self._module_order)

    def flash_module(self, _mod_id: str) -> None:
        return

    # ---------------- module/page system ----------------

    def close_module(self, mod_id):
        widget = self._widget_for_mod(mod_id)
        if widget is None:
            return

        addon_id = getattr(widget, "_addon_id", "")
        if addon_id == "terminal":
            index = self._conversation_stack.indexOf(widget)
            if index >= 0:
                self._conversation_stack.removeWidget(widget)
            if self._active_terminal_mod_id == mod_id:
                self._active_terminal_mod_id = None
        else:
            self.companion.unregister_addon(mod_id)

        self.unregister_module(mod_id)
        self.stack.removeWidget(widget)
        widget.deleteLater()

        terminal_ids = self._terminal_mod_ids()
        if addon_id == "terminal":
            if terminal_ids:
                self.switch_to_module(terminal_ids[-1])
            elif self.host is not None:
                self.host.launch_module("terminal")

    def switch_to_module(self, mod_id):
        widget = self._widget_for_mod(mod_id)
        if widget is None:
            return
        self.stack.setCurrentWidget(widget)
        self._selected_module_id = mod_id
        addon_id = getattr(widget, "_addon_id", "")
        title = getattr(widget, "windowTitle", lambda: addon_id.upper() or "MODULE")() or addon_id.upper()

        if addon_id == "terminal":
            if self._conversation_stack.indexOf(widget) < 0:
                self._conversation_stack.addWidget(widget)
            self._conversation_stack.setCurrentWidget(widget)
            self._active_terminal_mod_id = mod_id
            self.companion.set_conversation(widget)
            self._refresh_omni_models()
            self.icon_rail.set_active(None)
            self.companion.refresh_from_world()
            self.update_terminal_header(mod_id, *self._terminal_titles.get(mod_id, ("Untitled Chat", QDateTime.currentDateTime().toString("ddd • HH:mm"))))
        elif addon_id == "sd":
            # Mount the full SDModule directly in the companion pane so the
            # user gets the interactive UI (prompt, params, LOAD, GENERATE)
            # instead of the previous read-only VisionPanel summary that
            # silently dropped clicks when its proxy module wasn't bound.
            self.companion.attach_module(CompanionState.VISION, widget)
            self.companion.show_state(CompanionState.VISION)
            self.icon_rail.set_active(f"addon:{mod_id}")
        elif addon_id == "audiogen":
            self.companion.show_state(CompanionState.AUDIO)
            self.icon_rail.set_active(f"addon:{mod_id}")
        else:
            self.companion.show_addon(mod_id, title, widget, pin=True)
            self.icon_rail.set_active(f"addon:{mod_id}")

    # ---------------- shell/UI state ----------------

    def update_status(self, engine_key: str, status: SystemStatus):
        status_text = status.value if hasattr(status, "value") else str(status)
        display = status_text if engine_key.startswith("llm") else f"{engine_key.upper()}: {status_text}"
        self.omni_bar.set_status(display)
        self.vitals.update_engine(engine_key, status)
        self.companion.refresh_from_world()

    def update_ctx(self, used):
        self.state.ctx_used = used
        if not self._ctx_refresh_timer.isActive():
            self._ctx_refresh_timer.start()

    def _flush_ctx_update(self) -> None:
        self.vitals.refresh()

    def update_terminal_header(self, mod_id, title, timestamp):
        if mod_id:
            self._terminal_titles[mod_id] = (
                title or "Untitled Chat",
                timestamp or QDateTime.currentDateTime().toString("ddd • HH:mm"),
            )

        active_widget = self._active_conversation()
        active_mod_id = getattr(active_widget, "_mod_id", None) if active_widget is not None else None
        if mod_id and active_mod_id != mod_id:
            return

        current_title, current_time = self._terminal_titles.get(
            active_mod_id or "",
            ("Untitled Chat", QDateTime.currentDateTime().toString("ddd • HH:mm")),
        )
        self.omni_bar.set_title("MONOLITH")
        self.vitals.set_session_meta(current_title)

    def _load_operator_profile(self, name: str) -> None:
        if not self.host:
            return
        manager = OperatorManager()
        try:
            operator_data = manager.load_operator(name)
        except Exception as exc:
            self.ui_bridge.sig_monitor_log.emit("ERROR", f"[operator] failed to load '{name}': {exc}")
            return

        if "modules" in operator_data:
            for mod_id in list(self.get_module_order()):
                self.close_module(mod_id)

            first_terminal_mod_id = None
            for entry in operator_data.get("modules", []):
                addon_id = entry.get("addon_id")
                if not addon_id:
                    continue
                new_mod_id = self.host.launch_module(addon_id)
                if not new_mod_id:
                    continue
                widget = self._widget_for_mod(new_mod_id)
                apply_operator = getattr(widget, "apply_operator", None) if widget is not None else None
                if callable(apply_operator):
                    apply_operator(entry)
                if addon_id == "terminal":
                    if not first_terminal_mod_id:
                        first_terminal_mod_id = new_mod_id
            if first_terminal_mod_id:
                self.switch_to_module(first_terminal_mod_id)
            return

        target_widget = None
        for i in range(self.stack.count()):
            widget = self.stack.widget(i)
            if getattr(widget, "_addon_id", None) == "terminal":
                target_widget = widget
                break
        if target_widget is None:
            new_mod_id = self.host.launch_module("terminal")
            target_widget = self._widget_for_mod(new_mod_id)
        self.ui_bridge.sig_apply_operator.emit(operator_data)
        mod_id = getattr(target_widget, "_mod_id", None) if target_widget is not None else None
        if mod_id:
            self.switch_to_module(mod_id)

    def _snapshot_workspace(self) -> dict:
        modules = []
        module_order = []
        for mod_id in self.get_module_order():
            widget = self._widget_for_mod(mod_id)
            addon_id = getattr(widget, "_addon_id", None) if widget is not None else None
            if not addon_id:
                continue
            module_order.append(addon_id)
            entry = {"addon_id": addon_id}
            operator_snapshot = getattr(widget, "operator_snapshot", None)
            if callable(operator_snapshot):
                try:
                    snapshot = operator_snapshot()
                except Exception:
                    snapshot = None
                if isinstance(snapshot, dict):
                    entry.update(snapshot)
            elif hasattr(widget, "config"):
                entry["config"] = dict(getattr(widget, "config", {}) or {})
            session = getattr(widget, "_current_session", None)
            if isinstance(session, dict):
                entry["messages"] = list(session.get("messages", []))
                entry["session_title"] = session.get("title")
                entry["assistant_tokens"] = session.get("assistant_tokens", 0)
            modules.append(entry)
        return {"modules": modules, "module_order": module_order}

    def _create_operator_profile(self) -> None:
        dialog = NameDialog(self)
        if dialog.exec() != dialog.Accepted:
            return
        clean_name = dialog.value()
        if not clean_name:
            return

        snapshot = self._snapshot_workspace()
        manager = OperatorManager()
        previous_data = None
        try:
            previous_data = manager.load_operator(clean_name)
        except Exception:
            previous_data = None
        payload = {"name": clean_name, "layout": {}, "geometry": {}}
        payload.update(snapshot)
        manager.save_operator(clean_name, payload, previous_data=previous_data, trigger="saved")
        self.ui_bridge.sig_monitor_log.emit("INFO", f"[profile] saved '{clean_name}'")

    def _show_operator_lineage(self, name: str) -> None:
        clean_name = str(name or "").strip()
        if not clean_name:
            return
        manager = OperatorManager()
        lineage = manager.get_lineage(clean_name)
        dialog = LineageDialog(clean_name, lineage, self)
        dialog.exec()

    def _set_active_model(self, model_name: str) -> None:
        active = self._active_conversation()
        if active is None or not hasattr(active, "config"):
            return
        model_name = str(model_name or "").strip()
        if not model_name:
            return
        combo = getattr(active, "api_model_combo", None)
        if combo is not None:
            idx = combo.findText(model_name)
            if idx >= 0:
                combo.setCurrentIndex(idx)
                self._refresh_omni_models()
                return
        field = getattr(active, "api_model_input", None)
        if field is not None:
            field.setText(model_name)
        active.config["api_model"] = model_name
        if hasattr(active, "_emit_model_payload"):
            active._emit_model_payload()
        self._refresh_omni_models()

    # ---------------- shell interactions ----------------

    def _handle_rail_request(self, name: str) -> None:
        if name == "OVERSEER":
            self.ui_bridge.sig_open_overseer.emit()
            return
        if name.startswith("addon:"):
            mod_id = name.split(":", 1)[1]
            widget = self._widget_for_mod(mod_id)
            addon_id = getattr(widget, "_addon_id", None) if widget is not None else None
            if self._addon_is_visible(mod_id, addon_id):
                self.companion.collapse()
            else:
                self.switch_to_module(mod_id)
            return
        if name == "STATS":
            if getattr(self, "_stats_page", None) is None:
                from ui.pages.stats import PageStats
                self._stats_page = PageStats(self.state, self.ui_bridge, bridge=None, guard=None)
                self.companion.attach_module(CompanionState.STATS, self._stats_page)
            if self.companion._current_state == CompanionState.STATS:
                self.companion.collapse()
            else:
                self.companion.show_state(CompanionState.STATS)
            return
        requested_state = None
        if name == "DATABANK":
            requested_state = CompanionState.DATABANK
        elif name in CompanionState.__members__:
            requested_state = CompanionState[name]
        if requested_state is None:
            return
        if self.companion._current_state == requested_state:
            self.companion.collapse()
        else:
            self.companion.show_state(requested_state)

    def _addon_is_visible(self, mod_id: str, addon_id: str | None) -> bool:
        if not mod_id or not addon_id:
            return False
        if addon_id == "sd":
            return (
                self.companion._current_state == CompanionState.VISION
                and self._selected_module_id == mod_id
            )
        if addon_id == "audiogen":
            return (
                self.companion._current_state == CompanionState.AUDIO
                and self._selected_module_id == mod_id
            )
        return (
            self.companion._current_state == CompanionState.ADDON
            and getattr(self.companion, "_active_addon_id", None) == mod_id
        )

    def _on_companion_state_changed(self, state_name: str) -> None:
        if state_name in {"VISION", "AUDIO"} and self._selected_module_id:
            widget = self._widget_for_mod(self._selected_module_id)
            addon_id = getattr(widget, "_addon_id", None) if widget is not None else None
            if state_name == "VISION" and addon_id == "sd":
                self.icon_rail.set_active(f"addon:{self._selected_module_id}")
                return
            if state_name == "AUDIO" and addon_id == "audiogen":
                self.icon_rail.set_active(f"addon:{self._selected_module_id}")
                return
        if state_name not in {"ADDON", "COLLAPSED"}:
            self.icon_rail.set_active(state_name)
        elif state_name == "COLLAPSED":
            self.icon_rail.set_active(None)

    def _handle_omni_action(self, action: str, payload: object) -> None:
        if action.startswith("module:"):
            addon_id = str(payload or action.split(":", 1)[1])
            if self.host is not None:
                mod_id = self.host.launch_module(addon_id)
                if mod_id:
                    self.switch_to_module(mod_id)
            return
        if action.startswith("panel:"):
            panel_name = str(payload or action.split(":", 1)[1])
            if panel_name in CompanionState.__members__:
                if hasattr(self.companion, "show_state"):
                    self.companion.show_state(CompanionState[panel_name])
                else:
                    self.companion.pin_state(CompanionState[panel_name])
            return
        if action.startswith("theme:"):
            self.ui_bridge.sig_theme_changed.emit(str(payload or action.split(":", 1)[1]))
            return
        if action.startswith("profile:"):
            profile_name = str(payload or action.split(":", 1)[1])
            if profile_name == "new":
                self._create_operator_profile()
                return
            self._load_operator_profile(profile_name)
            return
        if action.startswith("lineage:"):
            operator_name = str(payload or action.split(":", 1)[1])
            self._show_operator_lineage(operator_name)
            return
        if action.startswith("model:"):
            self._set_active_model(str(payload or action.split(":", 1)[1]))
            return
        if action == "agent:apply":
            active = self._active_conversation()
            if active is not None and hasattr(active, "_apply_agent_command"):
                active._apply_agent_command(str(payload or ""))
            return
        if action.startswith("history:"):
            active = self._active_conversation()
            if active is not None and hasattr(active, "_archive"):
                try:
                    session = active._archive.load_session(Path(str(payload)))
                except Exception:
                    return
                active._set_current_session(session, show_reset=False, sync_history=True)
            return
        active = self._active_conversation()
        if active is None:
            return
        if action == "cmd:load" and hasattr(active, "toggle_load"):
            if not getattr(active, "_is_model_loaded", False):
                active.toggle_load()
        elif action == "cmd:unload" and hasattr(active, "toggle_load"):
            if getattr(active, "_is_model_loaded", False):
                active.toggle_load()
        elif action == "cmd:new" and hasattr(active, "_start_new_session"):
            active._start_new_session()
        elif action == "cmd:vitals":
            self.toggle_vitals()
        elif action == "cmd:monitor":
            self.ui_bridge.sig_open_overseer.emit()
        elif action == "cmd:refresh-theme":
            # Re-emit the current theme so bootstrap's _apply_theme handler
            # runs again, which now invokes deep_refresh_theme across every
            # visible top-level widget tree. Sweeps any per-widget
            # stylesheet that got stuck on a prior theme's colors.
            try:
                from core.themes import current_theme_key
                current = current_theme_key() or ""
            except Exception:
                current = ""
            self.ui_bridge.sig_theme_changed.emit(current)

    def _handle_footer_unload(self, engine_key: str) -> None:
        action = {
            "type": "engine_command",
            "engine": str(engine_key or ""),
            "command": "unload",
            "payload": {},
            "priority": 2,
        }
        if action["engine"] and hasattr(self.ui_bridge, "sig_world_action"):
            self.ui_bridge.sig_world_action.emit(action)
