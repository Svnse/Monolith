from PySide6.QtCore import Signal, Qt
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
)

from core.operators import OperatorManager
from core.style import (
    BG_GROUP, BG_INPUT, BG_MAIN, FG_TEXT, FG_DIM, ACCENT_GOLD,
    BORDER_DARK, GLASS_BG, GLASS_BORDER, GLASS_HOVER,
)
from ui.components.atoms import SkeetButton


class _NameDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Operator")
        self.setModal(True)
        self.setStyleSheet(f"""
            QDialog {{ background: {BG_INPUT}; color: {FG_TEXT}; }}
            QLineEdit {{ background: #101010; color: {FG_TEXT}; border: 1px solid #333; padding: 6px; }}
            QPushButton {{ color: {FG_TEXT}; background: transparent; border: 1px solid #333; padding: 6px 12px; }}
            QPushButton:hover {{ border: 1px solid {ACCENT_GOLD}; color: {ACCENT_GOLD}; }}
        """)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Operator name:"))
        self.input = QLineEdit()
        layout.addWidget(self.input)
        row = QHBoxLayout()
        row.addStretch()
        ok_btn = SkeetButton("OK")
        cancel_btn = SkeetButton("CANCEL")
        ok_btn.clicked.connect(self.accept)
        cancel_btn.clicked.connect(self.reject)
        row.addWidget(ok_btn)
        row.addWidget(cancel_btn)
        layout.addLayout(row)

    def value(self) -> str:
        return self.input.text().strip()

class _OperatorCard(QPushButton):
    """Glassmorphic operator card with structured info."""
    sig_double_clicked = Signal(str)

    def __init__(self, name: str, gguf_path: str, tag_count: int, module_count: int = 0):
        super().__init__()
        self.op_name = name
        self.setCursor(Qt.PointingHandCursor)
        self.setMinimumHeight(80)
        self.setMinimumWidth(180)
        self._selected = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        lbl_name = QLabel(name.upper())
        lbl_name.setStyleSheet(f"color: {FG_TEXT}; font-size: 11px; font-weight: bold; background: transparent; letter-spacing: 1px;")
        layout.addWidget(lbl_name)

        lbl_model = QLabel(gguf_path)
        lbl_model.setStyleSheet(f"color: {FG_DIM}; font-size: 9px; background: transparent;")
        lbl_model.setWordWrap(True)
        layout.addWidget(lbl_model)

        info_parts = []
        if module_count > 0:
            info_parts.append(f"{module_count} module{'s' if module_count != 1 else ''}")
        if tag_count > 0:
            info_parts.append(f"{tag_count} tag{'s' if tag_count != 1 else ''}")
        lbl_info = QLabel(" · ".join(info_parts) if info_parts else "empty")
        lbl_info.setStyleSheet(f"color: #444; font-size: 9px; background: transparent;")
        layout.addWidget(lbl_info)

        layout.addStretch()
        self._apply_style(False)

    def _apply_style(self, selected: bool):
        self._selected = selected
        border = ACCENT_GOLD if selected else BORDER_DARK
        bg = "#1a1a1a" if selected else BG_INPUT
        self.setStyleSheet(f"""
            _OperatorCard {{
                background: {bg};
                border: 1px solid {border};
                border-radius: 3px;
            }}
            _OperatorCard:hover {{
                border: 1px solid {ACCENT_GOLD};
                background: #141414;
            }}
        """)

    def set_selected(self, selected: bool):
        self._apply_style(selected)

    def mouseDoubleClickEvent(self, event):
        self.sig_double_clicked.emit(self.op_name)


class PageHub(QWidget):
    sig_load_operator = Signal(str)
    sig_save_operator = Signal(str, dict)

    def __init__(self, config_provider=None, operator_manager: OperatorManager | None = None):
        super().__init__()
        self._operator_manager = operator_manager or OperatorManager()
        self._config_provider = config_provider
        self._selected_name: str | None = None
        self._cards: dict[str, _OperatorCard] = {}

        self.setStyleSheet(f"background: {BG_MAIN};")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(30, 30, 30, 30)
        layout.setSpacing(0)

        # --- Welcome header ---
        header = QWidget()
        header.setStyleSheet("background: transparent;")
        header_layout = QVBoxLayout(header)
        header_layout.setContentsMargins(0, 0, 0, 20)
        header_layout.setSpacing(4)

        lbl_welcome = QLabel("MONOLITH")
        lbl_welcome.setStyleSheet(
            f"color: {ACCENT_GOLD}; font-size: 20px; font-weight: bold; "
            f"letter-spacing: 4px; background: transparent;"
        )
        header_layout.addWidget(lbl_welcome)

        lbl_sub = QLabel("Select an operator to restore your workspace, or create a new one.")
        lbl_sub.setStyleSheet(f"color: {FG_DIM}; font-size: 10px; background: transparent;")
        header_layout.addWidget(lbl_sub)

        layout.addWidget(header)

        # --- Separator ---
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet(f"background: {BORDER_DARK};")
        layout.addWidget(sep)
        layout.addSpacing(16)

        # --- Operator label ---
        ops_header = QHBoxLayout()
        lbl_ops = QLabel("OPERATORS")
        lbl_ops.setStyleSheet(
            f"color: {FG_DIM}; font-size: 9px; font-weight: bold; "
            f"letter-spacing: 2px; background: transparent;"
        )
        ops_header.addWidget(lbl_ops)
        ops_header.addStretch()
        layout.addLayout(ops_header)
        layout.addSpacing(10)

        # --- Card grid ---
        self.grid_wrap = QWidget()
        self.grid_wrap.setStyleSheet("background: transparent;")
        self.grid = QGridLayout(self.grid_wrap)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(10)
        layout.addWidget(self.grid_wrap, 1)

        # --- Empty state ---
        self.empty_label = QLabel("No operators saved yet.\nCreate one to snapshot your current workspace.")
        self.empty_label.setAlignment(Qt.AlignCenter)
        self.empty_label.setStyleSheet(f"color: #333; font-size: 11px; padding: 40px; background: transparent;")
        self.grid.addWidget(self.empty_label, 0, 0, 1, 3, Qt.AlignCenter)

        layout.addStretch()

        # --- Bottom action bar ---
        sep2 = QFrame()
        sep2.setFixedHeight(1)
        sep2.setStyleSheet(f"background: {BORDER_DARK};")
        layout.addWidget(sep2)
        layout.addSpacing(10)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(8)
        self.btn_new = SkeetButton("＋ NEW")
        self.btn_new.setFixedHeight(28)
        self.btn_new.clicked.connect(self._create_operator_from_current)
        self.btn_load = SkeetButton("▶ LOAD")
        self.btn_load.setFixedHeight(28)
        self.btn_load.clicked.connect(self._load_selected)
        self.btn_load.setEnabled(False)
        self.btn_delete = SkeetButton("— DELETE")
        self.btn_delete.setFixedHeight(28)
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_delete.setEnabled(False)
        btn_row.addWidget(self.btn_new)
        btn_row.addWidget(self.btn_load)
        btn_row.addWidget(self.btn_delete)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.refresh_cards()

    def refresh_cards(self):
        # Clear existing cards (but not empty_label — we control it separately)
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
            # New format: modules list; legacy: top-level config
            modules = data.get("modules", [])
            module_count = len(modules)
            if modules:
                # Find first terminal's config for display
                term = next((m for m in modules if m.get("addon_id") == "terminal"), None)
                cfg = term.get("config", {}) if term else {}
            else:
                cfg = data.get("config", {})
            gguf_path = self._truncate_path(cfg.get("gguf_path"))
            tag_count = len(cfg.get("behavior_tags") or [])

            card = _OperatorCard(name, gguf_path, tag_count, module_count)
            card.clicked.connect(lambda _checked=False, op_name=name: self._on_card_clicked(op_name))
            card.sig_double_clicked.connect(self._load_operator)
            row, col = divmod(idx, 3)
            self.grid.addWidget(card, row + 1, col)  # +1 to skip row 0 (empty_label)
            self._cards[name] = card

        if self._selected_name not in self._cards:
            self._selected_name = None
            self.btn_load.setEnabled(False)
            self.btn_delete.setEnabled(False)

    def _on_card_clicked(self, name: str):
        self._selected_name = name
        for op_name, card in self._cards.items():
            card.set_selected(op_name == name)
        self.btn_load.setEnabled(True)
        self.btn_delete.setEnabled(True)

    def _load_selected(self):
        if self._selected_name:
            self.sig_load_operator.emit(self._selected_name)

    def _load_operator(self, name: str):
        self._on_card_clicked(name)
        self.sig_load_operator.emit(name)

    def _create_operator_from_current(self):
        if self._config_provider is None:
            QMessageBox.warning(self, "Operator", "Terminal page is not mounted.")
            return
        dialog = _NameDialog(self)
        if dialog.exec() != QDialog.Accepted:
            return
        clean_name = dialog.value()
        if not clean_name:
            return
        snapshot = dict(self._config_provider() or {})
        data = {"name": clean_name, "layout": {}, "geometry": {}}
        data.update(snapshot)  # merges "modules" and "module_order" into top level
        # Keep a "config" key for backward compat if snapshot has no modules
        if "modules" not in data:
            data["config"] = snapshot
        self.sig_save_operator.emit(clean_name, data)
        self.refresh_cards()

    def _delete_selected(self):
        if not self._selected_name:
            return
        if not self._operator_manager.delete_operator(self._selected_name):
            QMessageBox.warning(self, "Operator", "Delete failed.")
            return
        self._selected_name = None
        self.btn_load.setEnabled(False)
        self.btn_delete.setEnabled(False)
        self.refresh_cards()

    def _truncate_path(self, value) -> str:
        if not value:
            return "No model path"
        path = str(value)
        if len(path) <= 42:
            return path
        return f"...{path[-39:]}"
