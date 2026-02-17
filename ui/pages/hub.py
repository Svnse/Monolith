from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QGridLayout,
    QFrame,
    QPushButton,
    QHBoxLayout,
    QDialog,
    QLineEdit,
    QMessageBox,
    QComboBox,
    QScrollArea,
)

from core.operators import OperatorManager
from core.themes import list_themes, current_theme, apply_theme
from core.theme_config import save_theme_config
from ui.components.atoms import MonoButton


class _NameDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        import core.style as s

        self.setWindowTitle("New Profile")
        self.setModal(True)
        self.setStyleSheet(
            f"""
            QDialog {{ background: {s.BG_INPUT}; color: {s.FG_TEXT}; }}
            QLineEdit {{ background: {s.BG_INPUT}; color: {s.FG_TEXT}; border: 1px solid {s.BORDER_LIGHT}; padding: 6px; }}
            QPushButton {{ color: {s.FG_TEXT}; background: transparent; border: 1px solid {s.BORDER_LIGHT}; padding: 6px 12px; }}
            QPushButton:hover {{ border: 1px solid {s.ACCENT_PRIMARY}; color: {s.ACCENT_PRIMARY}; }}
        """
        )
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Profile name:"))
        self.input = QLineEdit()
        layout.addWidget(self.input)
        row = QHBoxLayout()
        row.addStretch()
        ok_btn = MonoButton("OK")
        cancel_btn = MonoButton("CANCEL")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(ok_btn)
        row.addWidget(cancel_btn)
        layout.addLayout(row)

    def value(self) -> str:
        return self.input.text().strip()


class _LineageDialog(QDialog):
    def __init__(self, operator_name: str, lineage: list[dict], parent=None):
        super().__init__(parent)
        import core.style as s

        self.setWindowTitle(f"History: {operator_name}")
        self.setModal(True)
        self.setMinimumWidth(560)
        self.setMaximumHeight(400)
        self.setStyleSheet(
            f"""
            QDialog {{ background: {s.BG_INPUT}; color: {s.FG_TEXT}; }}
            QFrame#lineage_item {{ border: 1px solid {s.BORDER_LIGHT}; background: transparent; border-radius: 2px; }}
            QLabel {{ background: transparent; color: {s.FG_TEXT}; }}
        """
        )

        layout = QVBoxLayout(self)
        layout.setSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        container = QWidget()
        items_layout = QVBoxLayout(container)
        items_layout.setContentsMargins(0, 0, 0, 0)
        items_layout.setSpacing(8)

        if not lineage:
            empty = QLabel("No lineage available.")
            empty.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px;")
            items_layout.addWidget(empty)
        else:
            for snapshot in lineage:
                if not isinstance(snapshot, dict):
                    continue
                item = QFrame()
                item.setObjectName("lineage_item")
                item_layout = QVBoxLayout(item)
                item_layout.setContentsMargins(10, 8, 10, 8)
                item_layout.setSpacing(4)

                version = snapshot.get("version", "?")
                trigger = snapshot.get("trigger", "unknown")
                timestamp = snapshot.get("timestamp", "")
                headline = QLabel(f"v{version} — {trigger} — {timestamp}")
                headline.setStyleSheet(f"color: {s.FG_TEXT}; font-size: 10px;")
                item_layout.addWidget(headline)

                diff = snapshot.get("diff", {})
                keys = list(diff.keys()) if isinstance(diff, dict) else []
                detail = f"Changed: {', '.join(keys)}" if keys else "Changed: none"
                detail_lbl = QLabel(detail)
                detail_lbl.setWordWrap(True)
                detail_lbl.setStyleSheet(f"color: {s.FG_DIM}; font-size: 9px;")
                item_layout.addWidget(detail_lbl)

                items_layout.addWidget(item)

        items_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll)

        close_row = QHBoxLayout()
        close_row.addStretch()
        btn_close = MonoButton("CLOSE")
        btn_close.clicked.connect(self.accept)
        close_row.addWidget(btn_close)
        layout.addLayout(close_row)


class _OperatorCard(QPushButton):
    """Glassmorphic operator card with structured info."""

    sig_double_clicked = Signal(str)

    def __init__(
        self,
        name: str,
        gguf_path: str,
        tag_count: int,
        module_count: int = 0,
        presence_info: dict | None = None,
    ):
        super().__init__()
        import core.style as s

        self.op_name = name
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(80)
        self.setMinimumWidth(180)
        self._selected = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        lbl_name = QLabel(name.upper())
        lbl_name.setStyleSheet(
            f"color: {s.FG_TEXT}; font-size: 11px; font-weight: bold; background: transparent; letter-spacing: 1px;"
        )
        layout.addWidget(lbl_name)

        lbl_model = QLabel(gguf_path)
        lbl_model.setStyleSheet(f"color: {s.FG_DIM}; font-size: 9px; background: transparent;")
        lbl_model.setWordWrap(True)
        layout.addWidget(lbl_model)

        info_parts = []
        if module_count > 0:
            info_parts.append(f"{module_count} module{'s' if module_count != 1 else ''}")
        if tag_count > 0:
            info_parts.append(f"{tag_count} tag{'s' if tag_count != 1 else ''}")
        lbl_info = QLabel(" · ".join(info_parts) if info_parts else "empty")
        lbl_info.setStyleSheet(f"color: {s.FG_INFO}; font-size: 9px; background: transparent;")
        layout.addWidget(lbl_info)

        if presence_info:
            version = int(presence_info.get("current_version", 0) or 0)
            drift = float(presence_info.get("drift_score", 0.0) or 0.0)
            threshold = float(presence_info.get("drift_threshold", 0.5) or 0.5)
            color = s.FG_WARN if drift >= threshold else s.FG_DIM
            lbl_presence = QLabel(f"v{version} · drift {drift:.0%}")
            lbl_presence.setStyleSheet(f"color: {color}; font-size: 8px; background: transparent;")
            layout.addWidget(lbl_presence)

        layout.addStretch()
        self._apply_style(False)

    def _apply_style(self, selected: bool):
        import core.style as s

        self._selected = selected
        border = s.ACCENT_PRIMARY if selected else s.BORDER_DARK
        bg = s.BORDER_SUBTLE if selected else s.BG_INPUT
        self.setStyleSheet(
            f"""
            _OperatorCard {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 3px;
            }}
            _OperatorCard:hover {{
                border: 1px solid {s.ACCENT_PRIMARY};
                background: {s.BG_BUTTON_HOVER};
            }}
        """
        )

    def set_selected(self, selected: bool):
        self._apply_style(selected)

    def mouseDoubleClickEvent(self, event):
        self.sig_double_clicked.emit(self.op_name)


class PageHub(QWidget):
    sig_load_operator = Signal(str)
    sig_save_operator = Signal(str, dict)
    sig_presence_drift = Signal(str, float, float)

    def __init__(self, config_provider=None, operator_manager: OperatorManager | None = None, ui_bridge=None):
        super().__init__()
        import core.style as s

        self._operator_manager = operator_manager or OperatorManager()
        self._config_provider = config_provider
        self._ui_bridge = ui_bridge
        self._selected_name: str | None = None
        self._cards: dict[str, _OperatorCard] = {}

        self.setStyleSheet(f"background: {s.BG_MAIN};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(0)

        header = QWidget()
        header.setStyleSheet("background: transparent;")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 20)
        header_layout.setSpacing(4)

        self._lbl_welcome = QLabel("MONOLITH")
        self._lbl_welcome.setStyleSheet(
            f"color: {s.ACCENT_PRIMARY}; font-size: 20px; font-weight: bold; letter-spacing: 4px; background: transparent;"
        )
        header_layout.addWidget(self._lbl_welcome)

        self._lbl_sub = QLabel("Select a profile to restore your workspace, or create a new one.")
        self._lbl_sub.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px; background: transparent;")
        header_layout.addWidget(self._lbl_sub)

        layout.addWidget(header)

        self._separators = []
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {s.BORDER_DARK};")
        self._separators.append(sep)
        layout.addWidget(sep)
        layout.addSpacing(16)

        ops_header = QHBoxLayout()
        self._lbl_ops = QLabel("PROFILES")
        self._lbl_ops.setStyleSheet(
            f"color: {s.FG_DIM}; font-size: 9px; font-weight: bold; letter-spacing: 2px; background: transparent;"
        )
        ops_header.addWidget(self._lbl_ops)
        ops_header.addStretch()
        layout.addLayout(ops_header)
        layout.addSpacing(10)

        self.grid_wrap = QWidget()
        self.grid_wrap.setStyleSheet("background: transparent;")
        self.grid = QGridLayout(self.grid_wrap)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(10)
        layout.addWidget(self.grid_wrap, 1)

        self.empty_label = QLabel("No profiles saved yet.\nCreate one to snapshot your current workspace.")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet(f"color: {s.FG_INFO}; font-size: 11px; padding: 40px; background: transparent;")
        self.grid.addWidget(self.empty_label, 0, 0, 1, 3, Qt.AlignCenter)

        layout.addStretch()

        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet(f"background: {s.BORDER_DARK};")
        self._separators.append(sep2)
        layout.addWidget(sep2)
        layout.addSpacing(10)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_new = MonoButton("＋ NEW")
        self.btn_new.setFixedHeight(28)
        self.btn_new.clicked.connect(self._create_operator_from_current)
        self.btn_load = MonoButton("▶ LOAD")
        self.btn_load.setFixedHeight(28)
        self.btn_load.clicked.connect(self._load_selected)
        self.btn_load.setEnabled(False)
        self.btn_delete = MonoButton("— DELETE")
        self.btn_delete.setFixedHeight(28)
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_delete.setEnabled(False)
        self.btn_lineage = MonoButton("⟳ LINEAGE")
        self.btn_lineage.setFixedHeight(28)
        self.btn_lineage.clicked.connect(self._show_lineage)
        self.btn_lineage.setEnabled(False)
        btn_row.addWidget(self.btn_new)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_delete)
        btn_row.addWidget(self.btn_lineage)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        layout.addSpacing(16)
        sep3 = QFrame()
        sep3.setFixedHeight(1)
        sep3.setStyleSheet(f"background: {s.BORDER_DARK};")
        self._separators.append(sep3)
        layout.addWidget(sep3)
        layout.addSpacing(10)

        self._lbl_appearance = QLabel("APPEARANCE")
        self._lbl_appearance.setStyleSheet(
            f"color: {s.FG_DIM}; font-size: 9px; font-weight: bold; letter-spacing: 2px; background: transparent;"
        )
        layout.addWidget(self._lbl_appearance)
        layout.addSpacing(6)

        theme_row = QHBoxLayout()
        theme_row.setSpacing(8)
        self._lbl_theme = QLabel("Theme")
        self._lbl_theme.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px; background: transparent;")
        self.theme_combo = QComboBox()
        self.theme_combo.setFixedWidth(140)
        self.theme_combo.setFixedHeight(28)
        self.theme_combo.setStyleSheet(
            f"""
            QComboBox {{
                background: {s.BG_INPUT}; color: {s.FG_TEXT};
                border: 1px solid {s.BORDER_LIGHT}; padding: 4px 8px;
                font-size: 10px; font-weight: bold; border-radius: 2px;
            }}
            QComboBox:hover {{ border: 1px solid {s.ACCENT_PRIMARY}; }}
            QComboBox::drop-down {{
                border: none; width: 20px;
            }}
            QComboBox::down-arrow {{
                image: none; border: none;
            }}
            QComboBox QAbstractItemView {{
                background: {s.BG_INPUT}; color: {s.FG_TEXT};
                border: 1px solid {s.BORDER_LIGHT};
                selection-background-color: {s.BG_BUTTON_HOVER};
                selection-color: {s.ACCENT_PRIMARY};
            }}
        """
        )
        for name in list_themes():
            self.theme_combo.addItem(name)
        active = current_theme().name
        idx = self.theme_combo.findText(active)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)
        self.theme_combo.currentTextChanged.connect(self._on_theme_changed)

        theme_row.addWidget(self._lbl_theme)
        theme_row.addWidget(self.theme_combo)
        theme_row.addStretch()
        layout.addLayout(theme_row)

        if self._ui_bridge:
            self._ui_bridge.sig_theme_changed.connect(
                lambda _: QTimer.singleShot(0, self._refresh_theme)
            )

        self.refresh_cards()

    def _refresh_theme(self):
        """Re-apply all inline styles after a theme change."""
        import core.style as s

        self.setStyleSheet(f"background: {s.BG_MAIN};")

        # Walk named widgets and refresh
        self._lbl_welcome.setStyleSheet(
            f"color: {s.ACCENT_PRIMARY}; font-size: 20px; font-weight: bold; letter-spacing: 4px; background: transparent;"
        )
        self._lbl_sub.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px; background: transparent;")
        self._lbl_ops.setStyleSheet(
            f"color: {s.FG_DIM}; font-size: 9px; font-weight: bold; letter-spacing: 2px; background: transparent;"
        )
        self._lbl_appearance.setStyleSheet(
            f"color: {s.FG_DIM}; font-size: 9px; font-weight: bold; letter-spacing: 2px; background: transparent;"
        )
        self._lbl_theme.setStyleSheet(f"color: {s.FG_DIM}; font-size: 10px; background: transparent;")
        self.empty_label.setStyleSheet(f"color: {s.FG_INFO}; font-size: 11px; padding: 40px; background: transparent;")

        for sep in self._separators:
            sep.setStyleSheet(f"background: {s.BORDER_DARK};")

        self.theme_combo.setStyleSheet(
            f"""
            QComboBox {{
                background: {s.BG_INPUT}; color: {s.FG_TEXT};
                border: 1px solid {s.BORDER_LIGHT}; padding: 4px 8px;
                font-size: 10px; font-weight: bold; border-radius: 2px;
            }}
            QComboBox:hover {{ border: 1px solid {s.ACCENT_PRIMARY}; }}
            QComboBox::drop-down {{ border: none; width: 20px; }}
            QComboBox::down-arrow {{ image: none; border: none; }}
            QComboBox QAbstractItemView {{
                background: {s.BG_INPUT}; color: {s.FG_TEXT};
                border: 1px solid {s.BORDER_LIGHT};
                selection-background-color: {s.BG_BUTTON_HOVER};
                selection-color: {s.ACCENT_PRIMARY};
            }}
        """
        )

        self.refresh_cards()

    def refresh_cards(self):
        for card in self._cards.values():
            self.grid.removeWidget(card)
            card.deleteLater()
        self._cards.clear()

        operators = self._operator_manager.list_operators()
        self.empty_label.setVisible(len(operators) == 0)

        for idx, item in enumerate(operators):
            name = item["name"]
            try:
                data = self._operator_manager.load_operator(name)
            except Exception:
                continue
            modules = data.get("modules", [])
            module_count = len(modules)
            if modules:
                term = next((m for m in modules if m.get("addon_id") == "terminal"), None)
                cfg = term.get("config", {}) if term else {}
            else:
                cfg = data.get("config", {})
            gguf_path = self._truncate_path(cfg.get("gguf_path"))
            tag_count = len(cfg.get("behavior_tags") or [])
            presence_info = self._operator_manager.get_presence_info(name)

            card = _OperatorCard(name, gguf_path, tag_count, module_count, presence_info=presence_info)
            card.clicked.connect(lambda _checked=False, op_name=name: self._on_card_clicked(op_name))
            card.sig_double_clicked.connect(self._load_operator)
            row, col = divmod(idx, 3)
            self.grid.addWidget(card, row + 1, col)
            self._cards[name] = card

        if self._selected_name not in self._cards:
            self._selected_name = None
            self.btn_load.setEnabled(False)
            self.btn_delete.setEnabled(False)
            self.btn_lineage.setEnabled(False)

    def _truncate_path(self, path: str | None, max_len: int = 40) -> str:
        if not path:
            return "no model"
        if len(path) <= max_len:
            return path
        return "…" + path[-(max_len - 1):]

    def _on_card_clicked(self, name: str):
        self._selected_name = name
        for op_name, card in self._cards.items():
            card.set_selected(op_name == name)
        self.btn_load.setEnabled(True)
        self.btn_delete.setEnabled(True)
        self.btn_lineage.setEnabled(True)

    def _load_selected(self):
        if self._selected_name:
            self.sig_load_operator.emit(self._selected_name)

    def _load_operator(self, name: str):
        self._on_card_clicked(name)
        self.sig_load_operator.emit(name)

    def _create_operator_from_current(self):
        if self._config_provider is None:
            QMessageBox.warning(self, "Profile", "Terminal page is not mounted.")
            return
        dialog = _NameDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        clean_name = dialog.value()
        if not clean_name:
            return

        snapshot = dict(self._config_provider() or {})
        data = {"name": clean_name, "layout": {}, "geometry": {}}
        data.update(snapshot)
        if "modules" not in data:
            data["config"] = snapshot

        previous_data = None
        try:
            previous_data = self._operator_manager.load_operator(clean_name)
        except Exception:
            previous_data = None

        _, drift_exceeded = self._operator_manager.save_operator(
            clean_name,
            data,
            previous_data=previous_data,
            trigger="saved",
        )

        if drift_exceeded:
            self._show_drift_warning(clean_name, data)

        self.refresh_cards()

    def _show_drift_warning(self, name: str, data: dict):
        import core.style as s

        info = self._operator_manager.get_presence_info(name) or {}
        drift_score = float(info.get("drift_score", 0.0) or 0.0)
        threshold = float(info.get("drift_threshold", 0.5) or 0.5)

        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Warning)
        msg.setWindowTitle("Presence Drift")
        msg.setText(
            f"Your session has drifted significantly from the saved presence '{name}'. "
            "Save as new presence or update existing?"
        )
        msg.setStyleSheet(
            f"""
            QMessageBox {{ background: {s.BG_INPUT}; color: {s.FG_TEXT}; }}
            QLabel {{ color: {s.FG_TEXT}; }}
            QPushButton {{ color: {s.FG_TEXT}; background: transparent; border: 1px solid {s.BORDER_LIGHT}; padding: 6px 12px; }}
            QPushButton:hover {{ border: 1px solid {s.ACCENT_PRIMARY}; color: {s.ACCENT_PRIMARY}; }}
        """
        )
        btn_update = msg.addButton("Update", QMessageBox.AcceptRole)
        btn_save_new = msg.addButton("Save As New", QMessageBox.ActionRole)
        msg.addButton("Cancel", QMessageBox.RejectRole)
        msg.setDefaultButton(btn_update)
        msg.exec()

        self.sig_presence_drift.emit(name, drift_score, threshold)

        if msg.clickedButton() is btn_save_new:
            fork_dialog = _NameDialog(self)
            fork_dialog.setWindowTitle("Save As New Profile")
            fork_dialog.input.setText(f"{name} Copy")
            if fork_dialog.exec() != QDialog.Accepted:
                return
            fork_name = fork_dialog.value()
            if not fork_name:
                return
            fork_payload = dict(data)
            self._operator_manager.save_operator(fork_name, fork_payload, previous_data=None, trigger="user_forked")

    def _show_lineage(self):
        if not self._selected_name:
            return
        lineage = self._operator_manager.get_lineage(self._selected_name)
        dialog = _LineageDialog(self._selected_name, lineage, self)
        dialog.exec()

    def _delete_selected(self):
        if not self._selected_name:
            return
        if not self._operator_manager.delete_operator(self._selected_name):
            QMessageBox.warning(self, "Profile", "Delete failed.")
            return
        self._selected_name = None
        self.btn_load.setEnabled(False)
        self.btn_delete.setEnabled(False)
        self.btn_lineage.setEnabled(False)
        self.refresh_cards()

    def _on_theme_changed(self, theme_name: str):
        key = theme_name.lower()
        save_theme_config({"theme": key})
        if self._ui_bridge:
            self._ui_bridge.sig_theme_changed.emit(key)
        else:
            apply_theme(key)

