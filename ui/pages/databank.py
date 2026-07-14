import os
import shutil
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit,
    QTreeView, QHeaderView, QFileSystemModel, QInputDialog,
    QLabel, QMessageBox, QMenu, QTextEdit
)
from PySide6.QtCore import QDir, Qt
from PySide6.QtGui import QAction

from ui.components.atoms import MonoGroupBox, MonoButton
import core.style as _s  # dynamic theme bridge


def _load_file_view(path: str, *, size_cap: int = 1024 * 1024) -> tuple[str, str]:
    """Load a file's content for the databank viewer.

    Returns ``(mode, text)`` where mode is ``'text'`` (readable content),
    ``'binary'`` (a placeholder note), or ``'missing'`` (no such file /
    unreadable). Text over *size_cap* is truncated. Pure — no Qt, unit-testable.
    """
    from ui.components.blob_tray import is_text_file
    raw = str(path or "")
    if not raw or not os.path.isfile(raw):
        return ("missing", "")
    if not is_text_file(raw):
        try:
            sz = os.path.getsize(raw)
        except OSError:
            sz = 0
        return ("binary", f"[binary file — {sz} bytes]\n{raw}")
    try:
        with open(raw, "r", encoding="utf-8", errors="replace") as f:
            data = f.read(size_cap + 1)
    except OSError as exc:
        return ("missing", f"[could not read: {exc}]")
    if len(data) > size_cap:
        data = data[:size_cap] + "\n…(truncated)"
    return ("text", data)

class TerminalFileTree(QTreeView):
    def __init__(self, start_path):
        super().__init__()
        self.model = QFileSystemModel()
        
        self.model.setReadOnly(False)
        self.model.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot)
        self.model.setNameFilterDisables(False)
        self.setModel(self.model)
        self.change_root(start_path)

        self.setDragEnabled(True) 
        self.setDragDropMode(QTreeView.DragOnly)

        self.setStyleSheet(f"""
            QTreeView {{
                background: {_s.BG_INPUT};
                color: {_s.FG_SECONDARY};
                border: 1px solid {_s.BORDER_DARK};
                font-family: 'Consolas', monospace;
                font-size: 12px;
                outline: 0;
            }}
            QTreeView::item {{ padding: 4px; }}
            QTreeView::item:hover {{ background: {_s.BG_BUTTON_HOVER}; }}
            QTreeView::item:selected {{ background: {_s.ACCENT_PRIMARY}; color: black; }}
            
            QHeaderView::section {{
                background: {_s.BG_SIDEBAR};
                color: {_s.FG_DIM};
                border: none;
                padding: 4px;
                font-weight: bold;
            }}
        """)
        
        self.header().setSectionResizeMode(0, QHeaderView.Stretch)
        self.setColumnWidth(1, 80)
        self.setColumnHidden(2, True) 
        self.setColumnHidden(3, True) 
        self.setAnimated(False)
        self.setIndentation(20)
        self.setSortingEnabled(False)
        self.setContextMenuPolicy(Qt.CustomContextMenu)

    def change_root(self, path):
        abs_path = os.path.abspath(path)
        if not os.path.exists(abs_path):
            try:
                os.makedirs(abs_path)
            except OSError:
                pass 
                
        self.model.setRootPath(abs_path)
        self.setRootIndex(self.model.index(abs_path))

class PageFiles(QWidget):
    def __init__(self, state, ui_bridge=None):
        super().__init__()
        self.state = state
        self._ui_bridge = ui_bridge
        
        base_dir = os.path.expanduser("~")
            
        self.current_path = base_dir
        
        # No outer MonoGroupBox wrapper — the companion pane already titles
        # this as FILES. Path + search on top, tree fills the middle, action
        # bar at the bottom. Tight margins for narrow companion widths.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        _input_ss = (
            f"QLineEdit {{ background: {_s.BG_INPUT}; color: {_s.FG_TEXT};"
            f" border: 1px solid {_s.BORDER_LIGHT}; padding: 5px;"
            f" font-family: Consolas; }}"
        )

        # --- TOP EXPLORER BAR ---
        # Path on row 1 (the address bar), search on row 2. Stacking them
        # avoids horizontal clipping at narrow widths.
        self.inp_path = QLineEdit()
        self.inp_path.setText(self.current_path)
        self.inp_path.setPlaceholderText("Path")
        self.inp_path.setStyleSheet(
            f"QLineEdit {{ background: {_s.BG_INPUT}; color: {_s.ACCENT_PRIMARY};"
            f" border: 1px solid {_s.BORDER_LIGHT}; padding: 5px;"
            f" font-family: Consolas; }}"
        )
        self.inp_path.returnPressed.connect(self.navigate_to_path)
        layout.addWidget(self.inp_path)

        self.inp_search = QLineEdit()
        self.inp_search.setPlaceholderText("Filter…")
        self.inp_search.setStyleSheet(_input_ss)
        self.inp_search.textChanged.connect(self.on_search)
        layout.addWidget(self.inp_search)

        # --- FILE TREE ---
        self.tree = TerminalFileTree(self.current_path)
        self.tree.customContextMenuRequested.connect(self.open_menu)
        self.tree.clicked.connect(self.on_click_item)
        layout.addWidget(self.tree, 1)

        # --- ATTACHMENT VIEWER (hidden until a file/attachment is opened) ---
        self.viewer_box = QWidget()
        _vlay = QVBoxLayout(self.viewer_box)
        _vlay.setContentsMargins(0, 0, 0, 0)
        _vlay.setSpacing(2)
        _vhead = QHBoxLayout()
        self.viewer_title = QLabel("")
        self.viewer_title.setStyleSheet(
            f"color: {_s.ACCENT_PRIMARY}; font-size: 11px; font-family: Consolas;"
        )
        _vhead.addWidget(self.viewer_title, 1)
        btn_close_view = MonoButton("CLOSE")
        btn_close_view.clicked.connect(self.close_viewer)
        _vhead.addWidget(btn_close_view)
        _vlay.addLayout(_vhead)
        self.viewer = QTextEdit()
        self.viewer.setReadOnly(True)
        self.viewer.setStyleSheet(
            f"QTextEdit {{ background: {_s.BG_INPUT}; color: {_s.FG_TEXT};"
            f" border: 1px solid {_s.BORDER_LIGHT}; font-family: Consolas; font-size: 12px; }}"
        )
        _vlay.addWidget(self.viewer, 1)
        self.viewer_box.setVisible(False)
        layout.addWidget(self.viewer_box, 1)

        # --- BOTTOM ACTION BAR ---
        actions = QHBoxLayout()
        actions.setSpacing(6)
        btn_add = MonoButton("MKDIR")
        btn_add.clicked.connect(self.new_folder)

        btn_del = MonoButton("DELETE")
        btn_del.clicked.connect(self.delete_item)

        btn_ref = MonoButton("REFRESH")
        btn_ref.clicked.connect(self.refresh)

        actions.addWidget(btn_add)
        actions.addWidget(btn_del)
        actions.addWidget(btn_ref)
        actions.addStretch()
        layout.addLayout(actions)

        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 10px;")
        self.lbl_status.setWordWrap(True)
        layout.addWidget(self.lbl_status)

    def reveal_path(self, target: str) -> None:
        raw = str(target or "").strip()
        if not raw:
            return
        path = os.path.abspath(raw)
        if os.path.isfile(path):
            self.current_path = os.path.dirname(path)
        else:
            self.current_path = path
        self.inp_path.setText(self.current_path)
        self.tree.change_root(self.current_path)
        self.lbl_status.setText(f"Focused: {self.current_path}")
        self._log("INFO", f"[files] focused {path}")

    def navigate_to_path(self):
        new_path = self.inp_path.text()
        if os.path.exists(new_path):
            self.current_path = new_path
            self.tree.change_root(new_path)
            self.lbl_status.setText(f"Navigated to: {new_path}")
            self._log("INFO", f"[files] navigated to {new_path}")
        else:
            self.lbl_status.setText("Error: Path does not exist")
            self._log("WARNING", f"[files] path not found: {new_path}")

    def on_click_item(self, index):
        path = self.tree.model.filePath(index)
        if os.path.isdir(path):
            self.inp_path.setText(path)
        elif os.path.isfile(path):
            self.open_file(path)

    def open_file(self, path: str) -> None:
        """Open a file in the viewer pane. Text files show their content;
        binary files show a placeholder. Also focuses the tree on the file."""
        mode, text = _load_file_view(str(path or ""))
        name = os.path.basename(str(path or "")) or "file"
        if mode == "missing":
            self.lbl_status.setText(f"Cannot open: {name}")
            self._log("WARNING", f"[files] cannot open {path}")
            return
        self.reveal_path(str(path))
        self.viewer_title.setText(f"📎 {name}")
        self.viewer.setPlainText(text)
        self.viewer_box.setVisible(True)
        self.lbl_status.setText(f"Viewing: {name}")
        self._log("INFO", f"[files] viewing {path}")

    def open_inline(self, label: str, content: str) -> None:
        """Show inline (pasted) attachment content in the viewer — no file."""
        self.viewer_title.setText(f"📎 {label or 'attachment'} (pasted)")
        self.viewer.setPlainText(str(content or ""))
        self.viewer_box.setVisible(True)
        self.lbl_status.setText(f"Viewing pasted: {label or 'attachment'}")

    def open_attachment(self, att) -> None:
        """Open an Attachment coming from a chat chip. File-backed -> open_file;
        inline paste/text -> open_inline."""
        path = getattr(att, "path", None)
        content = getattr(att, "content", None)
        label = getattr(att, "label", "") or "attachment"
        if path:
            self.open_file(path)
        elif content is not None:
            self.open_inline(label, content)
        else:
            self.lbl_status.setText(f"No content for {label}")

    def close_viewer(self) -> None:
        self.viewer_box.setVisible(False)
        self.viewer.clear()

    def refresh(self):
        self.tree.change_root(self.current_path)
        self.lbl_status.setText("Refreshed")
        self._log("INFO", "[files] refreshed")

    def on_search(self, text):
        if text:
            self.tree.model.setNameFilters([f"*{text}*"])
        else:
            self.tree.model.setNameFilters([])

    def get_selected_path(self):
        indexes = self.tree.selectedIndexes()
        if indexes:
            return self.tree.model.filePath(indexes[0])
        return self.current_path

    def new_folder(self):
        target_path = self.get_selected_path()
        if os.path.isfile(target_path):
            target_path = os.path.dirname(target_path)
            
        name, ok = self.ask_input("New Folder", "Folder Name:")
        if ok and name:
            new_dir = os.path.join(target_path, name)
            try:
                os.makedirs(new_dir, exist_ok=True)
                self.lbl_status.setText(f"Created: {name}")
                self._log("INFO", f"[files] created folder: {new_dir}")
            except Exception as e:
                self.lbl_status.setText(f"Error: {e}")
                self._log("WARNING", f"[files] create failed: {e}")

    def delete_item(self):
        target = self.get_selected_path()
        if target == self.current_path:
            self.lbl_status.setText("Cannot delete root folder")
            return
            
        if os.path.exists(target):
            msg = QMessageBox(self)
            msg.setWindowTitle("Confirm Delete")
            msg.setText(f"Are you sure you want to delete:\\n{os.path.basename(target)}?")
            msg.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
            msg.setStyleSheet(f"""
                QMessageBox {{ background: {_s.BG_MAIN}; }}
                QLabel {{ color: {_s.FG_TEXT}; }}
                QPushButton {{ background: {_s.BG_BUTTON_HOVER}; color: {_s.FG_SECONDARY}; border: 1px solid {_s.FG_INFO}; padding: 5px; }}
            """)
            if msg.exec() == QMessageBox.Yes:
                try:
                    if os.path.isdir(target):
                        shutil.rmtree(target)
                    else:
                        os.remove(target)
                    self.lbl_status.setText("Item deleted")
                    self._log("INFO", f"[files] deleted: {target}")
                except Exception as e:
                    self.lbl_status.setText(f"Delete Error: {e}")
                    self._log("WARNING", f"[files] delete failed: {e}")

    def open_menu(self, position):
        menu = QMenu()
        menu.setStyleSheet(f"""
            QMenu {{ background: {_s.BG_SIDEBAR}; color: {_s.FG_TEXT}; border: 1px solid {_s.ACCENT_PRIMARY}; }}
            QMenu::item:selected {{ background: {_s.ACCENT_PRIMARY}; color: black; }}
        """)
        
        act_del = QAction("Delete", self)
        act_del.triggered.connect(self.delete_item)
        
        act_new = QAction("New Folder", self)
        act_new.triggered.connect(self.new_folder)
        
        menu.addAction(act_new)
        menu.addAction(act_del)
        menu.exec(self.tree.viewport().mapToGlobal(position))

    def ask_input(self, title, label):
        dlg = QInputDialog(self)
        dlg.setWindowTitle(title)
        dlg.setLabelText(label)
        dlg.setStyleSheet(f"""
            QDialog {{ background: {_s.BG_MAIN}; border: 1px solid {_s.ACCENT_PRIMARY}; }}
            QLabel {{ color: {_s.FG_TEXT}; }}
            QLineEdit {{ background: {_s.BG_INPUT}; color: white; border: 1px solid {_s.BORDER_LIGHT}; }}
            QPushButton {{ background: {_s.BG_BUTTON_HOVER}; color: {_s.FG_SECONDARY}; border: 1px solid {_s.FG_INFO}; padding: 5px; }}
        """)
        ok = dlg.exec()
        return dlg.textValue(), (ok == 1)

    def _log(self, severity: str, message: str) -> None:
        if self._ui_bridge is None:
            return
        self._ui_bridge.sig_monitor_log.emit(severity, message)
