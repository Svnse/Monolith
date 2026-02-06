from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QLabel,
    QGridLayout,
    QFrame,
    QPushButton,
    QHBoxLayout,
    QInputDialog,
    QMessageBox,
)

from core.operators import OperatorManager
from core.style import BG_GROUP, BG_INPUT, FG_TEXT, FG_DIM, ACCENT_GOLD
from ui.components.atoms import SkeetButton


class PageHub(QWidget):
    sig_load_operator = Signal(str)
    sig_save_operator = Signal(str, dict)

    def __init__(self, config_provider=None, operator_manager: OperatorManager | None = None):
        super().__init__()
        self._operator_manager = operator_manager or OperatorManager()
        self._config_provider = config_provider
        self._selected_name: str | None = None
        self._cards: dict[str, QPushButton] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(12)

        title = QLabel("HUB")
        title.setStyleSheet(f"color: {FG_TEXT}; font-size: 16px; font-weight: bold;")
        layout.addWidget(title)

        self.grid_wrap = QFrame()
        self.grid_wrap.setStyleSheet(f"background: {BG_GROUP}; border: 1px solid #222;")
        self.grid = QGridLayout(self.grid_wrap)
        self.grid.setContentsMargins(12, 12, 12, 12)
        self.grid.setSpacing(10)
        layout.addWidget(self.grid_wrap, 1)

        btn_row = QHBoxLayout()
        self.btn_new = SkeetButton("NEW")
        self.btn_new.clicked.connect(self._create_operator_from_current)
        self.btn_delete = SkeetButton("DELETE")
        self.btn_delete.clicked.connect(self._delete_selected)
        self.btn_delete.setEnabled(False)
        btn_row.addWidget(self.btn_new)
        btn_row.addWidget(self.btn_delete)
        btn_row.addStretch()
        layout.addLayout(btn_row)

        self.refresh_cards()

    def refresh_cards(self):
        while self.grid.count():
            item = self.grid.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._cards.clear()

        operators = self._operator_manager.list_operators()
        for idx, item in enumerate(operators):
            name = item["name"]
            try:
                data = self._operator_manager.load_operator(name)
            except Exception:
                continue
            cfg = data.get("config", {})
            gguf_path = self._truncate_path(cfg.get("gguf_path"))
            tag_count = len(cfg.get("behavior_tags") or [])
            btn = QPushButton(f"{name}\n{gguf_path}\nTags: {tag_count}")
            btn.setCursor(Qt.PointingHandCursor)
            btn.setMinimumHeight(90)
            btn.setStyleSheet(self._card_style(selected=False))
            btn.clicked.connect(lambda _checked=False, op_name=name: self._on_card_clicked(op_name))
            row, col = divmod(idx, 3)
            self.grid.addWidget(btn, row, col)
            self._cards[name] = btn

        if self._selected_name not in self._cards:
            self._selected_name = None
            self.btn_delete.setEnabled(False)

    def _card_style(self, selected: bool) -> str:
        border = ACCENT_GOLD if selected else "#333"
        fg = FG_TEXT if selected else FG_DIM
        return (
            f"QPushButton {{background: {BG_INPUT}; border: 1px solid {border}; color: {fg};"
            "padding: 8px; text-align: left; font-size: 10px; font-weight: bold;}}"
            f"QPushButton:hover {{border: 1px solid {ACCENT_GOLD}; color: {FG_TEXT};}}"
        )

    def _on_card_clicked(self, name: str):
        self._selected_name = name
        for op_name, card in self._cards.items():
            card.setStyleSheet(self._card_style(selected=op_name == name))
        self.btn_delete.setEnabled(True)
        self.sig_load_operator.emit(name)

    def _create_operator_from_current(self):
        if self._config_provider is None:
            QMessageBox.warning(self, "Operator", "Terminal page is not mounted.")
            return
        name, ok = QInputDialog.getText(self, "New Operator", "Operator name:")
        if not ok:
            return
        clean_name = name.strip()
        if not clean_name:
            return
        config = dict(self._config_provider() or {})
        data = {"name": clean_name, "config": config, "layout": {}, "geometry": {}}
        self.sig_save_operator.emit(clean_name, data)
        self.refresh_cards()

    def _delete_selected(self):
        if not self._selected_name:
            return
        if not self._operator_manager.delete_operator(self._selected_name):
            QMessageBox.warning(self, "Operator", "Delete failed.")
            return
        self._selected_name = None
        self.btn_delete.setEnabled(False)
        self.refresh_cards()

    def _truncate_path(self, value) -> str:
        if not value:
            return "No model path"
        path = str(value)
        if len(path) <= 42:
            return path
        return f"...{path[-39:]}"
