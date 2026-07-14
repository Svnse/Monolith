from __future__ import annotations

from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from ui.components.atoms import MonoButton


class NameDialog(QDialog):
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


class LineageDialog(QDialog):
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
                headline = QLabel(f"v{version} - {trigger} - {timestamp}")
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
