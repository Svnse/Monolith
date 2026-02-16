from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QKeyEvent, QMouseEvent
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ui.addons.host import AddonHost
from ui.addons.registry import AddonRegistry
from ui.addons.spec import AddonSpec



class _PaletteResultRow(QFrame):
    clicked = Signal(str)

    def __init__(self, addon_id: str, icon: str, title: str, match_label: str, parent: QWidget | None = None):
        super().__init__(parent)
        self._addon_id = addon_id
        self._selected = False
        self.setProperty("class", "PaletteResultRow")
        self.setCursor(Qt.PointingHandCursor)

        row = QHBoxLayout(self)
        row.setContentsMargins(10, 8, 10, 8)
        row.setSpacing(8)

        icon_lbl = QLabel(icon or "◻")
        icon_lbl.setFixedWidth(18)
        row.addWidget(icon_lbl)

        title_lbl = QLabel(title)
        title_lbl.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        row.addWidget(title_lbl)

        match_lbl = QLabel(match_label)
        row.addWidget(match_lbl)


    def set_selected(self, selected: bool) -> None:
        self._selected = selected
        self.setProperty("selected", "true" if selected else "false")
        self.style().unpolish(self)
        self.style().polish(self)


    def mousePressEvent(self, event: QMouseEvent) -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self._addon_id)
            event.accept()
            return
        super().mousePressEvent(event)


class CommandPalette(QWidget):
    sig_launch_module = Signal(str)

    def __init__(self, parent: QWidget, registry: AddonRegistry, host: AddonHost):
        super().__init__(parent)
        self._registry = registry
        self._host = host
        self._results: list[_PaletteResultRow] = []
        self._result_ids: list[str] = []
        self._selected_index = -1

        self.setObjectName("palette_overlay")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.setFocusPolicy(Qt.StrongFocus)

        root = QVBoxLayout(self)
        root.setContentsMargins(30, 30, 30, 30)
        root.setAlignment(Qt.AlignCenter)

        self._container = QFrame(self)
        self._container.setObjectName("palette_container")
        self._container.setMaximumWidth(500)
        self._container.setMaximumHeight(400)
        container_layout = QVBoxLayout(self._container)
        container_layout.setContentsMargins(14, 14, 14, 14)
        container_layout.setSpacing(10)

        self._input = QLineEdit(self._container)
        self._input.setObjectName("palette_input")
        self._input.setPlaceholderText("Type a command...")
        self._input.textChanged.connect(self._on_text_changed)
        self._input.installEventFilter(self)
        container_layout.addWidget(self._input)

        self._results_scroll = QScrollArea(self._container)
        self._results_scroll.setWidgetResizable(True)
        self._results_scroll.setFrameShape(QFrame.NoFrame)

        self._results_widget = QWidget(self._results_scroll)
        self._results_layout = QVBoxLayout(self._results_widget)
        self._results_layout.setContentsMargins(0, 0, 0, 0)
        self._results_layout.setSpacing(6)
        self._results_layout.addStretch()

        self._results_scroll.setWidget(self._results_widget)
        container_layout.addWidget(self._results_scroll)
        root.addWidget(self._container)

        super().hide()


    def toggle(self) -> None:
        self.hide() if self.isVisible() else self.show()

    def show(self) -> None:
        self.setGeometry(self.parentWidget().rect())
        self._input.clear()
        self._refresh_results("")
        super().show()
        self.raise_()
        self._input.setFocus()

    def hide(self) -> None:
        self._input.clear()
        self._clear_results()
        super().hide()

    def _iter_module_specs(self) -> list[AddonSpec]:
        return [spec for spec in self._registry.all() if spec.kind == "module"]

    def _on_text_changed(self, text: str) -> None:
        self._refresh_results(text)

    def _match_specs(self, query: str) -> list[tuple[AddonSpec, str]]:
        q = query.strip().lower()
        matched: list[tuple[AddonSpec, str]] = []
        for spec in self._iter_module_specs():
            if not q:
                matched.append((spec, "module"))
                continue

            if q in spec.title.lower():
                matched.append((spec, f"title:{spec.title}"))
                continue

            descriptor = spec.descriptor
            if descriptor is None:
                continue

            verb_match = next((verb for verb in descriptor.verbs if q in verb.lower()), None)
            if verb_match:
                matched.append((spec, f"verb:{verb_match}"))
                continue

            appetite_match = next((app for app in descriptor.appetites if q in app.lower()), None)
            if appetite_match:
                matched.append((spec, f"appetite:{appetite_match}"))

        return matched

    def _refresh_results(self, query: str | None = None) -> None:
        self._clear_results()
        query_text = self._input.text() if query is None else query
        matches = self._match_specs(query_text)

        for spec, match_label in matches:
            row = _PaletteResultRow(spec.id, spec.icon or "◻", spec.title, match_label)
            row.clicked.connect(self._on_result_clicked)
            self._results.append(row)
            self._result_ids.append(spec.id)
            self._results_layout.insertWidget(self._results_layout.count() - 1, row)

        if self._results:
            self._set_selected_index(0)
        else:
            self._selected_index = -1
            empty = QLabel("No matching modules")
            empty.setObjectName("palette_empty")
            self._results_layout.insertWidget(0, empty)

    def _clear_results(self) -> None:
        while self._results_layout.count() > 1:
            item = self._results_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._results.clear()
        self._result_ids.clear()
        self._selected_index = -1

    def _on_result_clicked(self, addon_id: str) -> None:
        self.sig_launch_module.emit(addon_id)
        self._host.launch_module(addon_id)
        self.hide()

    def _set_selected_index(self, index: int) -> None:
        if not self._results:
            self._selected_index = -1
            return
        self._selected_index = max(0, min(index, len(self._results) - 1))
        for i, row in enumerate(self._results):
            row.set_selected(i == self._selected_index)

    def _activate_selected(self) -> None:
        if 0 <= self._selected_index < len(self._result_ids):
            self._on_result_clicked(self._result_ids[self._selected_index])

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:
        if watched is self._input and event.type() == QEvent.KeyPress:
            key_event = event
            if isinstance(key_event, QKeyEvent):
                return self._handle_key_press(key_event)
        return super().eventFilter(watched, event)

    def _handle_key_press(self, event: QKeyEvent) -> bool:
        if event.key() == Qt.Key_Escape:
            self.hide()
            return True
        if event.modifiers() & Qt.ControlModifier and event.key() == Qt.Key_K:
            self.hide()
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

    def keyPressEvent(self, event: QKeyEvent) -> None:
        if self._handle_key_press(event):
            return
        super().keyPressEvent(event)

    def mousePressEvent(self, event: QMouseEvent) -> None:
        if not self._container.geometry().contains(event.position().toPoint()):
            self.hide()
            event.accept()
            return
        super().mousePressEvent(event)
