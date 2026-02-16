from __future__ import annotations

from pathlib import Path

import core.style as _s
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QVBoxLayout, QWidget

from ui.addons.host import AddonHost
from ui.addons.registry import AddonRegistry
from ui.addons.spec import AddonSpec

_EXT_APPETITE_MAP = {
    ".gguf": "gguf_file",
    ".txt": "text_content",
    ".md": "text_content",
    ".py": "text_content",
    ".json": "text_content",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
    ".wav": "audio",
    ".mp3": "audio",
    ".flac": "audio",
}


def _to_rgba(color_value: str, alpha: int) -> str:
    color = QColor(color_value)
    if not color.isValid():
        return f"rgba(0, 0, 0, {alpha})"
    color.setAlpha(alpha)
    return color.name(QColor.HexArgb)


class _ChoiceRow(QFrame):
    clicked = Signal(str)

    def __init__(self, spec: AddonSpec, appetite: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._addon_id = spec.id
        self.setCursor(Qt.PointingHandCursor)
        self.setObjectName("DropZoneChoice")

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 6, 10, 6)
        row.setSpacing(8)

        icon = QLabel(spec.icon or "◻")
        icon.setStyleSheet(f"color: {_s.FG_TEXT}; font-size: 12px;")
        icon.setFixedWidth(18)
        row.addWidget(icon)

        title = QLabel(spec.title)
        title.setStyleSheet(f"color: {_s.FG_TEXT}; font-size: 11px; font-weight: 600;")
        row.addWidget(title)

        appetite_lbl = QLabel(f"({appetite})")
        appetite_lbl.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 10px;")
        row.addWidget(appetite_lbl)

        self.setStyleSheet(
            f"""
            QFrame#DropZoneChoice {{
                border: 1px solid {_s.BORDER_SUBTLE};
                border-radius: 5px;
                background: transparent;
            }}
            QFrame#DropZoneChoice:hover {{
                background: {_s.BG_GROUP};
                border-color: {_s.ACCENT_PRIMARY};
            }}
            """
        )

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._addon_id)
            event.accept()
            return
        super().mousePressEvent(event)


class DropZoneOverlay(QWidget):
    sig_file_dropped = Signal(str, str)

    def __init__(self, parent: QWidget, registry: AddonRegistry, host: AddonHost):
        super().__init__(parent)
        self._registry = registry
        self._host = host
        self._matched_specs: list[tuple[AddonSpec, str]] = []

        self.setObjectName("DropZoneOverlay")
        self.setAttribute(Qt.WA_StyledBackground, True)

        root = QVBoxLayout(self)
        root.setContentsMargins(28, 28, 28, 28)

        self._panel = QFrame(self)
        self._panel.setObjectName("DropZonePanel")
        panel_layout = QVBoxLayout(self._panel)
        panel_layout.setContentsMargins(20, 20, 20, 20)
        panel_layout.setSpacing(10)
        panel_layout.setAlignment(Qt.AlignCenter)

        self._title = QLabel("DROP TO OPEN WITH")
        self._title.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(self._title)

        self._subtitle = QLabel("")
        self._subtitle.setAlignment(Qt.AlignCenter)
        self._subtitle.setWordWrap(True)
        panel_layout.addWidget(self._subtitle)

        self._choices = QWidget(self._panel)
        self._choices_layout = QVBoxLayout(self._choices)
        self._choices_layout.setContentsMargins(0, 6, 0, 0)
        self._choices_layout.setSpacing(6)
        panel_layout.addWidget(self._choices)

        root.addWidget(self._panel)

        self._apply_style()
        self.hide()

    def _apply_style(self) -> None:
        overlay_bg = _to_rgba(_s.BG_MAIN, 180)
        panel_bg = _to_rgba(_s.BG_MAIN, 110)
        self.setStyleSheet(
            f"""
            QWidget#DropZoneOverlay {{
                background: {overlay_bg};
            }}
            QFrame#DropZonePanel {{
                background: {panel_bg};
                border: 2px dashed {_s.ACCENT_PRIMARY};
                border-radius: 14px;
            }}
            """
        )
        self._title.setStyleSheet(f"color: {_s.ACCENT_PRIMARY}; font-size: 18px; font-weight: bold;")
        self._subtitle.setStyleSheet(f"color: {_s.FG_TEXT}; font-size: 11px;")

    def activate(self, event: QDragEnterEvent) -> None:
        if not event.mimeData().hasUrls():
            return

        self.setGeometry(self.parentWidget().rect())
        files = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        self._matched_specs = self._resolve_matches(files)

        if self._matched_specs:
            names = ", ".join(sorted({spec.title for spec, _ in self._matched_specs}))
            self._subtitle.setText(f"Suggested modules: {names}")
        else:
            self._subtitle.setText("No module can handle this file")

        self._clear_choices()
        self.show()
        self.raise_()

    def handle_drop(self, event: QDropEvent) -> None:
        files = [url.toLocalFile() for url in event.mimeData().urls() if url.isLocalFile()]
        if not self._matched_specs and files:
            self._matched_specs = self._resolve_matches(files)

        unique_specs: dict[str, tuple[AddonSpec, str]] = {}
        for spec, appetite in self._matched_specs:
            unique_specs[spec.id] = (spec, appetite)

        if len(unique_specs) == 1:
            spec = next(iter(unique_specs.values()))[0]
            self._host.launch_module(spec.id)
            self.sig_file_dropped.emit(files[0] if files else "", spec.id)
            self.deactivate()
            return

        if len(unique_specs) > 1:
            self._show_picker(list(unique_specs.values()), files[0] if files else "")
            return

        self.deactivate()

    def deactivate(self) -> None:
        self._matched_specs = []
        self._clear_choices()
        self.hide()

    def _show_picker(self, choices: list[tuple[AddonSpec, str]], file_path: str) -> None:
        self._subtitle.setText("Choose a module to open")
        self._clear_choices()
        for spec, appetite in choices:
            row = _ChoiceRow(spec, appetite, self._choices)
            row.clicked.connect(lambda addon_id, fp=file_path: self._pick(addon_id, fp))
            self._choices_layout.addWidget(row)

    def _pick(self, addon_id: str, file_path: str) -> None:
        self._host.launch_module(addon_id)
        self.sig_file_dropped.emit(file_path, addon_id)
        self.deactivate()

    def _clear_choices(self) -> None:
        while self._choices_layout.count():
            item = self._choices_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()

    def _resolve_matches(self, files: list[str]) -> list[tuple[AddonSpec, str]]:
        appetites = set()
        for file_path in files:
            ext = Path(file_path).suffix.lower()
            appetites.add(_EXT_APPETITE_MAP.get(ext, "file_path"))

        matches: list[tuple[AddonSpec, str]] = []
        for appetite in appetites:
            for spec in self._registry.query_by_appetite(appetite):
                if spec.kind == "module":
                    matches.append((spec, appetite))
        return matches
