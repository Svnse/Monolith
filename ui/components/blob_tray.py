"""BlobTray — file drop and large paste chip tray above chat input."""
from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

import core.style as _s

_TEXT_EXTENSIONS = frozenset({
    ".txt", ".md", ".py", ".json", ".yaml", ".yml", ".csv", ".log",
    ".xml", ".html", ".css", ".js", ".ts", ".toml", ".ini", ".cfg",
    ".sh", ".bat", ".ps1", ".sql", ".r", ".rs", ".go", ".java",
    ".c", ".cpp", ".h", ".hpp",
})
_TEXT_SIZE_CAP = 100 * 1024
_PASTE_BLOB_THRESHOLD = 1000


def _human_size(n: int) -> str:
    if n < 1024:
        return f"{n}B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f}KB"
    return f"{n / (1024 * 1024):.1f}MB"


def is_text_file(path: str) -> bool:
    return Path(path).suffix.lower() in _TEXT_EXTENSIONS


def is_zip_file(path: str) -> bool:
    return Path(path).suffix.lower() in (".zip", ".tar", ".gz", ".tgz", ".7z", ".rar")


class BlobChip(QFrame):
    sig_removed = Signal(int)
    sig_to_input = Signal(int)  # "un-attach": move this blob's content into the composer

    _COMPACT_H = 22   # file / zip pills
    _CARD_H = 66      # paste preview box (header + preview line + button)

    def __init__(
        self,
        index: int,
        kind: str,
        label: str,
        size: int,
        content: str | None = None,
        path: str | None = None,
        parent: QWidget | None = None,
    ):
        super().__init__(parent)
        self.index = index
        self.kind = kind
        self.label = label
        self.size = size
        self.content = content
        self.path = path
        self._to_input_btn: QLabel | None = None

        self.setObjectName("blob_chip")
        self.setCursor(Qt.PointingHandCursor)
        self.setStyleSheet(
            f"QFrame#blob_chip {{"
            f"  background: {_s.BG_PANEL};"
            f"  border: 1px solid {_s.ACCENT_PRIMARY};"
            f"  border-radius: 3px;"
            f"}}"
        )

        icon = {"file": "\U0001f4c4", "paste": "\U0001f4cb", "zip": "\U0001f4e6"}.get(kind, "\U0001f4c4")
        icon_lbl = QLabel(icon)
        icon_lbl.setStyleSheet("background: transparent; font-size: 10px;")
        name_lbl = QLabel(label)
        name_lbl.setStyleSheet(
            f"color: {_s.FG_TEXT}; font-size: 9px; font-family: Consolas, monospace; background: transparent;"
        )
        size_lbl = QLabel(_human_size(size))
        size_lbl.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 8px; font-family: Consolas, monospace; background: transparent;"
        )
        remove_btn = QLabel("✕")
        remove_btn.setStyleSheet(
            f"color: {_s.FG_DIM}; font-size: 9px; background: transparent; padding: 0 2px;"
        )
        remove_btn.setCursor(Qt.PointingHandCursor)
        remove_btn.mousePressEvent = lambda e: self._on_remove_clicked()

        if kind == "paste":
            # A taller preview "box": header row, a 1-line content preview, and a button
            # underneath to push the text into the chat input instead of attaching it.
            self.setFixedHeight(self._CARD_H)
            self.setMinimumWidth(220)
            outer = QVBoxLayout(self)
            outer.setContentsMargins(6, 3, 4, 3)
            outer.setSpacing(2)

            header = QHBoxLayout()
            header.setContentsMargins(0, 0, 0, 0)
            header.setSpacing(4)
            header.addWidget(icon_lbl)
            header.addWidget(name_lbl)
            header.addWidget(size_lbl)
            header.addStretch()
            header.addWidget(remove_btn)
            outer.addLayout(header)

            preview_lbl = QLabel(self._preview_text(content))
            preview_lbl.setStyleSheet(
                f"color: {_s.FG_DIM}; font-size: 8px; font-family: Consolas, monospace; background: transparent;"
            )
            outer.addWidget(preview_lbl)

            to_input = QLabel("↧ Move to chat input")
            to_input.setStyleSheet(
                f"color: {_s.ACCENT_PRIMARY}; font-size: 8px; font-family: Consolas, monospace;"
                f"  background: transparent; border: 1px solid {_s.ACCENT_PRIMARY};"
                f"  border-radius: 3px; padding: 1px 6px;"
            )
            to_input.setCursor(Qt.PointingHandCursor)
            to_input.mousePressEvent = lambda e: self._on_to_input_clicked()
            self._to_input_btn = to_input
            outer.addWidget(to_input, alignment=Qt.AlignLeft)
        else:
            self.setFixedHeight(self._COMPACT_H)
            row = QHBoxLayout(self)
            row.setContentsMargins(6, 0, 4, 0)
            row.setSpacing(4)
            row.addWidget(icon_lbl)
            row.addWidget(name_lbl)
            row.addWidget(size_lbl)
            row.addWidget(remove_btn)

    @staticmethod
    def _preview_text(content: str | None) -> str:
        s = (content or "").strip()
        if not s:
            return "(empty)"
        first = s.splitlines()[0][:80]
        return f'"{first}…"' if len(s) > len(first) else f'"{first}"'

    def _on_remove_clicked(self) -> None:
        self.sig_removed.emit(self.index)

    def _on_to_input_clicked(self) -> None:
        self.sig_to_input.emit(self.index)


class BlobTray(QFrame):
    sig_blob_to_input = Signal(str)  # a chip's content was pushed toward the composer

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("blob_tray")
        self.setFixedHeight(28)
        self.setStyleSheet(
            f"QFrame#blob_tray {{"
            f"  background: transparent;"
            f"  border-top: 1px solid {_s.FG_DIM};"
            f"}}"
        )

        self._chips: list[BlobChip] = []

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setFixedHeight(26)
        self._scroll = scroll
        scroll.setStyleSheet(
            "QScrollArea { background: transparent; border: none; }"
            "QScrollBar:horizontal { height: 3px; background: transparent; }"
            f"QScrollBar::handle:horizontal {{ background: {_s.FG_DIM}; border-radius: 1px; }}"
            "QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }"
        )

        self._inner = QWidget()
        self._inner.setStyleSheet("background: transparent;")
        self._layout = QHBoxLayout(self._inner)
        self._layout.setContentsMargins(4, 2, 4, 2)
        self._layout.setSpacing(4)
        self._layout.addStretch()

        scroll.setWidget(self._inner)

        outer = QHBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

        self.hide()

    def _rebuild_indices(self) -> None:
        for i, chip in enumerate(self._chips):
            chip.index = i

    def _remove_chip(self, index: int) -> None:
        if 0 <= index < len(self._chips):
            chip = self._chips.pop(index)
            self._layout.removeWidget(chip)
            chip.deleteLater()
            self._rebuild_indices()
            self._recompute_height()
            self.setVisible(bool(self._chips))

    def _add_chip(self, chip: BlobChip) -> None:
        chip.sig_removed.connect(self._remove_chip)
        chip.sig_to_input.connect(self._on_chip_to_input)
        insert_pos = len(self._chips)
        self._layout.insertWidget(insert_pos, chip)
        self._chips.append(chip)
        self._recompute_height()
        self.setVisible(True)

    def _on_chip_to_input(self, index: int) -> None:
        """Hand the chip's content to the composer (sig_blob_to_input) and drop the chip."""
        if 0 <= index < len(self._chips):
            content = self._chips[index].content or ""
            self.sig_blob_to_input.emit(content)
            self._remove_chip(index)

    def _recompute_height(self) -> None:
        """Grow the tray to fit a paste 'card' chip; shrink back to a pill row otherwise."""
        chip_h = BlobChip._CARD_H if any(c.kind == "paste" for c in self._chips) else BlobChip._COMPACT_H
        self._scroll.setFixedHeight(chip_h + 4)
        self.setFixedHeight(chip_h + 6)

    def add_file(self, path: str) -> None:
        p = Path(path)
        name = p.name
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        content = None
        if is_text_file(path) and size <= _TEXT_SIZE_CAP:
            try:
                content = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                pass
        chip = BlobChip(
            index=len(self._chips),
            kind="file",
            label=name,
            size=size,
            content=content,
            path=str(p),
        )
        self._add_chip(chip)

    def add_paste(self, text: str, label: str = "pasted text") -> None:
        chip = BlobChip(
            index=len(self._chips),
            kind="paste",
            label=str(label or "pasted text"),
            size=len(text),
            content=text,
        )
        self._add_chip(chip)

    def add_zip(self, path: str) -> None:
        p = Path(path)
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        chip = BlobChip(
            index=len(self._chips),
            kind="zip",
            label=p.name,
            size=size,
            path=str(p),
        )
        self._add_chip(chip)

    def has_blobs(self) -> bool:
        return bool(self._chips)

    def blobs(self) -> list[dict]:
        return [
            {
                "kind": c.kind,
                "label": c.label,
                "size": c.size,
                "content": c.content,
                "path": c.path,
            }
            for c in self._chips
        ]

    def clear(self) -> None:
        for chip in self._chips:
            self._layout.removeWidget(chip)
            chip.deleteLater()
        self._chips.clear()
        self.setVisible(False)


def format_attached_blocks(blobs: list[dict], user_text: str) -> str:
    if not blobs:
        return user_text
    blocks: list[str] = []
    for blob in blobs:
        kind = blob.get("kind", "file")
        label = blob.get("label", "unknown")
        size = blob.get("size", 0)
        content = blob.get("content")
        path = blob.get("path")
        size_str = _human_size(size)

        if kind == "zip":
            type_tag = "archive"
        elif kind == "paste":
            type_tag = "paste"
        else:
            type_tag = "text" if content else "binary"

        header = f"[ATTACHED: {label} ({size_str}, {type_tag})]"

        if content:
            blocks.append(f"{header}\n{content}\n[/ATTACHED]")
        elif path:
            if kind == "zip":
                blocks.append(f"{header}\nPath: {path}\nUse open_file tool to list or preview archive contents.\n[/ATTACHED]")
            else:
                blocks.append(f"{header}\nPath: {path}\nUse open_file tool to extract readable contents.\n[/ATTACHED]")
        else:
            blocks.append(f"{header}\n(no content available)\n[/ATTACHED]")

    return "\n\n".join(blocks) + "\n\n---\n" + user_text
