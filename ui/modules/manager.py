from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFrame,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
from PySide6.QtCore import Signal, Qt, QTimer

from ui.components.atoms import MonoGroupBox, MonoButton
from ui.addons.registry import AddonRegistry
from ui.addons.loader import load_manifest_addons
from core.addon_manifest import delete_addon, load_addon_manifest, upsert_addon
from core.paths import ADDON_MANIFEST
import core.style as s

class PageAddons(QWidget):
    sig_launch_addon = Signal(str)
    sig_open_vitals = Signal()
    sig_open_overseer = Signal()

    def __init__(self, state, registry: AddonRegistry, ui_bridge=None):
        super().__init__()
        self._registry = registry
        self._ui_bridge = ui_bridge
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        scroll_content = QWidget()
        scroll_content.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        grp_modules = MonoGroupBox("MODULES")
        grp_modules.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        
        mod_layout = QVBoxLayout()
        mod_layout.setSpacing(15)
        
        self._status_label = QLabel("Open a module to get started.")
        self._status_label.setObjectName("status_label")

        mod_layout.addWidget(self._status_label)

        preferred_order = ["terminal", "monoline", "databank", "injector", "sd", "audiogen"]
        specs = [s for s in self._registry.all() if s.kind == "module" and s.id != "theme"]
        by_id = {s.id: s for s in specs}
        ordered = [by_id[pid] for pid in preferred_order if pid in by_id]
        remaining = sorted([s for s in specs if s.id not in preferred_order], key=lambda s: s.title)
        for spec in ordered + remaining:
            btn = MonoButton(spec.title)
            btn.clicked.connect(lambda _checked=False, addon_id=spec.id: self.sig_launch_addon.emit(addon_id))
            mod_layout.addWidget(btn)
        self._module_button_layout = mod_layout
        mod_layout.addStretch()
        
        grp_modules.add_layout(mod_layout)
        scroll_layout.addWidget(grp_modules)

        grp_addons = MonoGroupBox("ADDON BLUEPRINTS")
        grp_addons.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        addon_layout = QVBoxLayout()
        addon_layout.setSpacing(10)
        self._btn_add_addon = MonoButton("+ ADD ADDON")
        self._btn_add_addon.clicked.connect(self._open_addon_dialog)
        addon_layout.addWidget(self._btn_add_addon)
        self._btn_reload_addons = MonoButton("RELOAD ADDONS")
        self._btn_reload_addons.clicked.connect(lambda: self._reload_addons("Addons reloaded."))
        addon_layout.addWidget(self._btn_reload_addons)
        self._btn_hot_reload = MonoButton("HOT RELOAD: ON")
        self._btn_hot_reload.setCheckable(True)
        self._btn_hot_reload.setChecked(True)
        self._btn_hot_reload.clicked.connect(self._toggle_hot_reload)
        addon_layout.addWidget(self._btn_hot_reload)
        self._addon_list_layout = addon_layout
        grp_addons.add_layout(addon_layout)
        scroll_layout.addWidget(grp_addons)
        
        grp_system = MonoGroupBox("TOOLS")
        grp_system.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)
        system_layout = QVBoxLayout()
        system_layout.setSpacing(10)

        self._btn_theme = MonoButton("THEME")
        self._btn_theme.clicked.connect(lambda: self.sig_launch_addon.emit("theme"))
        system_layout.addWidget(self._btn_theme)

        btn_vitals = MonoButton("VITALS")
        btn_vitals.clicked.connect(self.sig_open_vitals.emit)

        btn_overseer = MonoButton("MONITOR")
        btn_overseer.clicked.connect(self.sig_open_overseer.emit)

        system_layout.addWidget(btn_vitals)
        system_layout.addWidget(btn_overseer)
        grp_system.add_layout(system_layout)
        scroll_layout.addWidget(grp_system)
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)

        self._refresh_modules()
        self._refresh_tools()
        self._refresh_addon_list()
        self._start_hot_reload_watch()

    def _start_hot_reload_watch(self) -> None:
        self._addon_watch_hash = None
        self._addon_watch_timer = QTimer(self)
        self._addon_watch_timer.setInterval(1500)
        self._addon_watch_timer.timeout.connect(self._poll_addon_changes)
        self._addon_watch_timer.start()

    def _toggle_hot_reload(self) -> None:
        enabled = self._btn_hot_reload.isChecked()
        if enabled:
            self._btn_hot_reload.setText("HOT RELOAD: ON")
            self._addon_watch_hash = self._addon_watch_stamp()
            self._addon_watch_timer.start()
            self._set_status("Hot reload enabled.")
        else:
            self._btn_hot_reload.setText("HOT RELOAD: OFF")
            self._addon_watch_timer.stop()
            self._set_status("Hot reload disabled.")

    def _addon_watch_stamp(self) -> int:
        paths = [Path(ADDON_MANIFEST)]
        for entry in load_addon_manifest():
            entry_path = str(entry.get("entry", "")).strip()
            if entry_path:
                paths.append(Path(entry_path).expanduser())
        parts = []
        for path in paths:
            try:
                mtime = path.stat().st_mtime
            except FileNotFoundError:
                mtime = 0
            parts.append((str(path), mtime))
        parts.sort()
        return hash(tuple(parts))

    def _poll_addon_changes(self) -> None:
        stamp = self._addon_watch_stamp()
        if self._addon_watch_hash is None:
            self._addon_watch_hash = stamp
            return
        if stamp != self._addon_watch_hash:
            self._addon_watch_hash = stamp
            self._reload_addons("Addons reloaded (hot).")

    def _open_addon_dialog(self) -> None:
        dlg = _AddAddonDialog(self)
        if dlg.exec() != QDialog.Accepted:
            return
        payload = dlg.payload()
        if not payload:
            return
        entry = self._persist_payload(payload)
        if not entry:
            return
        self._reload_addons(f"Addon '{entry.get('name', entry.get('id'))}' added.")

    def _edit_addon(self, entry: dict[str, str]) -> None:
        dlg = _AddAddonDialog(self, existing=entry)
        if dlg.exec() != QDialog.Accepted:
            return
        payload = dlg.payload()
        if not payload:
            return
        payload["id"] = entry.get("id", "")
        updated = self._persist_payload(payload)
        if not updated:
            return
        self._reload_addons(f"Addon '{updated.get('name', updated.get('id'))}' updated.")

    def _delete_addon(self, addon_id: str) -> None:
        answer = QMessageBox.question(
            self,
            "Delete Addon",
            "Delete this addon blueprint?",
            QMessageBox.Yes | QMessageBox.No,
            QMessageBox.No,
        )
        if answer != QMessageBox.Yes:
            return
        if delete_addon(addon_id):
            self._reload_addons("Addon deleted.")
        else:
            self._set_status("Addon not found", warn=True)

    def _persist_payload(self, payload: dict[str, str]) -> dict[str, str] | None:
        mode = payload.get("mode", "external_py")
        entry = {
            "id": payload.get("id", ""),
            "name": payload["name"],
            "entry": payload.get("entry", ""),
            "workdir": payload.get("workdir", ""),
            "command": payload.get("command", ""),
            "icon": payload.get("icon", "") or "*",
            "mode": "spec_py" if mode == "spec_py" else "external_py",
        }
        return upsert_addon(entry)

    def _reload_addons(self, status: str) -> None:
        def _on_error(msg: str) -> None:
            self._set_status(msg, warn=True)

        load_manifest_addons(self._registry, on_error=_on_error, replace_dynamic=True)
        self._refresh_modules()
        self._refresh_tools()
        self._refresh_addon_list()
        self._addon_watch_hash = self._addon_watch_stamp()
        self._set_status(status)
        if self._ui_bridge is not None:
            self._ui_bridge.sig_reload_modules.emit()

    def _set_status(self, text: str, warn: bool = False) -> None:
        self._status_label.setText(text)
        if warn:
            self._status_label.setProperty("state", "warn")
        if self._ui_bridge is not None:
            sev = "WARNING" if warn else "INFO"
            self._ui_bridge.sig_monitor_log.emit(sev, f"[addons] {text}")

    def _refresh_modules(self) -> None:
        self._clear_layout(self._module_button_layout, preserve=1)
        preferred_order = ["terminal", "monoline", "databank", "injector", "sd", "audiogen"]
        specs = [s for s in self._registry.all() if s.kind == "module" and s.id != "theme"]
        by_id = {s.id: s for s in specs}
        ordered = [by_id[pid] for pid in preferred_order if pid in by_id]
        remaining = sorted([s for s in specs if s.id not in preferred_order], key=lambda s: s.title)
        for spec in ordered + remaining:
            btn = MonoButton(spec.title)
            btn.clicked.connect(lambda _checked=False, addon_id=spec.id: self.sig_launch_addon.emit(addon_id))
            self._module_button_layout.addWidget(btn)
        self._module_button_layout.addStretch()

    def _refresh_tools(self) -> None:
        theme_spec = next((spec for spec in self._registry.all() if spec.id == "theme"), None)
        if theme_spec is None:
            self._btn_theme.setVisible(False)
            return
        self._btn_theme.setText(theme_spec.title)
        self._btn_theme.setVisible(True)

    def _refresh_addon_list(self) -> None:
        self._clear_layout(self._addon_list_layout, preserve=1)
        entries = load_addon_manifest()
        if not entries:
            empty = QLabel("No addon blueprints installed.")
            empty.setObjectName("status_label")
            self._addon_list_layout.addWidget(empty)
            return
        for entry in entries:
            row = QHBoxLayout()
            name = QLabel(str(entry.get("name", entry.get("id", "addon"))))
            mode = QLabel(str(entry.get("mode", "external_py")))
            mode.setStyleSheet("font-size: 10px;")
            row.addWidget(name)
            row.addWidget(mode)
            row.addStretch()
            btn_edit = MonoButton("EDIT")
            btn_delete = MonoButton("DELETE")
            btn_edit.clicked.connect(lambda _checked=False, e=entry: self._edit_addon(e))
            btn_delete.clicked.connect(lambda _checked=False, aid=entry.get("id", ""): self._delete_addon(aid))
            row.addWidget(btn_edit)
            row.addWidget(btn_delete)
            container = QWidget()
            container.setLayout(row)
            self._addon_list_layout.addWidget(container)

    @staticmethod
    def _clear_layout(layout: QVBoxLayout, preserve: int = 0) -> None:
        while layout.count() > preserve:
            item = layout.takeAt(preserve)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()


class _AddAddonDialog(QDialog):
    def __init__(self, parent: QWidget | None = None, existing: dict[str, str] | None = None):
        super().__init__(parent)
        self.setWindowFlags(Qt.Dialog | Qt.FramelessWindowHint)
        self.setModal(True)
        self.setMinimumWidth(520)
        self._existing = existing or {}
        self._drag_pos = None

        root = QVBoxLayout(self)
        root.setContentsMargins(14, 14, 14, 14)

        container = QFrame()
        container.setObjectName("addon_dialog")
        container.setStyleSheet(
            f"QFrame#addon_dialog {{ background: {s.BG_MAIN}; border: 1px solid {s.BORDER_LIGHT}; "
            f"border-radius: 6px; }}"
        )
        container_layout = QVBoxLayout(container)
        container_layout.setContentsMargins(14, 14, 14, 14)
        container_layout.setSpacing(10)

        header = QFrame()
        header.setObjectName("addon_header")
        header.setStyleSheet(
            f"QFrame#addon_header {{ background: transparent; border: none; }}"
            f"QLabel {{ color: {s.FG_DIM}; font-size: 10px; font-weight: bold; letter-spacing: 1px; }}"
        )
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 0)
        header_layout.setSpacing(6)
        header_title = QLabel("ADD ADDON")
        header_layout.addWidget(header_title)
        header_layout.addStretch()
        btn_close = MonoButton("X")
        btn_close.setFixedSize(24, 22)
        btn_close.clicked.connect(self.reject)
        header_layout.addWidget(btn_close)
        container_layout.addWidget(header)

        box = MonoGroupBox("ADD ADDON")
        box_layout = QVBoxLayout()
        box_layout.setSpacing(10)

        form = QFormLayout()

        self.mode_combo = QComboBox()
        self.mode_combo.addItem("External Process", "external_py")
        self.mode_combo.addItem("QWidget Spec", "spec_py")
        form.addRow("Mode", self.mode_combo)

        self.name_input = QLineEdit()
        self.name_input.setPlaceholderText("Addon name")
        form.addRow("Name", self.name_input)

        self.entry_input = QLineEdit()
        self.entry_input.setPlaceholderText("Path to entry .py (optional)")
        btn_entry = MonoButton("BROWSE")
        btn_entry.clicked.connect(self._pick_entry)
        entry_row = QHBoxLayout()
        entry_row.addWidget(self.entry_input, 1)
        entry_row.addWidget(btn_entry)
        form.addRow("Entry", entry_row)

        self.workdir_input = QLineEdit()
        self.workdir_input.setPlaceholderText("Working directory (optional)")
        btn_workdir = MonoButton("BROWSE")
        btn_workdir.clicked.connect(self._pick_workdir)
        workdir_row = QHBoxLayout()
        workdir_row.addWidget(self.workdir_input, 1)
        workdir_row.addWidget(btn_workdir)
        form.addRow("Workdir", workdir_row)

        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Command (e.g. python main.py)")
        form.addRow("Command", self.command_input)

        self.icon_input = QLineEdit()
        self.icon_input.setPlaceholderText("Icon (optional)")
        form.addRow("Icon", self.icon_input)

        box_layout.addLayout(form)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        btn_cancel = MonoButton("CANCEL")
        btn_ok = MonoButton("ADD", accent=True)
        btn_ok.setDefault(True)
        btn_cancel.clicked.connect(self.reject)
        btn_ok.clicked.connect(self._accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(btn_ok)
        box_layout.addLayout(btn_row)
        box.add_layout(box_layout)
        container_layout.addWidget(box)
        root.addWidget(container)

        if self._existing:
            self._apply_existing()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self._drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._drag_pos and event.buttons() == Qt.LeftButton:
            self.move(event.globalPosition().toPoint() - self._drag_pos)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        self._drag_pos = None
        super().mouseReleaseEvent(event)

    def _pick_entry(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Entry", "", "Python Files (*.py)")
        if path:
            self.entry_input.setText(path)
            if not self.workdir_input.text().strip():
                self.workdir_input.setText(str(Path(path).parent))
            if not self.command_input.text().strip() and self._current_mode() == "external_py":
                self.command_input.setText("python " + Path(path).name)
            if not self.name_input.text().strip():
                self.name_input.setText(Path(path).stem.replace("_", " ").title())

    def _pick_workdir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "Select Working Directory", "")
        if path:
            self.workdir_input.setText(path)

    def _accept(self) -> None:
        if not self.name_input.text().strip():
            QMessageBox.warning(self, "Missing Name", "Please enter a name.")
            return
        if self._current_mode() == "spec_py":
            if not self.entry_input.text().strip():
                QMessageBox.warning(self, "Missing Entry", "Provide a spec entry file.")
                return
        else:
            if not self.command_input.text().strip() and not self.entry_input.text().strip():
                QMessageBox.warning(self, "Missing Entry", "Provide a command or entry file.")
                return
        self.accept()

    def payload(self) -> dict[str, str]:
        return {
            "name": self.name_input.text().strip(),
            "entry": self.entry_input.text().strip(),
            "workdir": self.workdir_input.text().strip(),
            "command": self.command_input.text().strip(),
            "icon": self.icon_input.text().strip(),
            "mode": self._current_mode(),
        }

    def _current_mode(self) -> str:
        return str(self.mode_combo.currentData())

    def _apply_existing(self) -> None:
        mode = self._existing.get("mode", "external_py")
        idx = self.mode_combo.findData(mode)
        if idx >= 0:
            self.mode_combo.setCurrentIndex(idx)
        self.name_input.setText(str(self._existing.get("name", "")))
        self.entry_input.setText(str(self._existing.get("entry", "")))
        self.workdir_input.setText(str(self._existing.get("workdir", "")))
        self.command_input.setText(str(self._existing.get("command", "")))
        self.icon_input.setText(str(self._existing.get("icon", "")))
