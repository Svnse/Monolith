import os
import shutil
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLineEdit, 
    QTreeView, QHeaderView, QFileSystemModel, QInputDialog, 
    QLabel, QMessageBox, QMenu
)
from PySide6.QtCore import QDir, Qt
from PySide6.QtGui import QAction

from ui.components.atoms import MonoGroupBox, MonoButton
import core.style as _s  # dynamic theme bridge
from core.config import DEFAULT_WORKSPACE_ROOT

class TerminalFileTree(QTreeView):
    def __init__(self, start_path):
        super().__init__()
        self.model = QFileSystemModel()
        
        self.model.setReadOnly(False)
        self.model.setFilter(QDir.AllEntries | QDir.NoDotAndDotDot)
        self.model.setNameFilterDisables(False)
        
        self.change_root(start_path)
        self.setModel(self.model)

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
    def __init__(self, state):
        super().__init__()
        self.state = state
        
        base_dir = "C:\\Models\\knowledge_base"
        if not os.path.exists("C:\\Models"):
            base_dir = str(DEFAULT_WORKSPACE_ROOT / "knowledge_base")
            
        self.current_path = base_dir
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        grp = MonoGroupBox("FILES")
        gl = QVBoxLayout()
        gl.setSpacing(10)
        
        # --- TOP EXPLORER BAR ---
        nav_bar = QHBoxLayout()
        
        self.inp_path = QLineEdit()
        self.inp_path.setText(self.current_path)
        self.inp_path.setPlaceholderText("Path...")
        self.inp_path.setStyleSheet(f"""
            QLineEdit {{
                background: {_s.BG_INPUT}; color: {_s.ACCENT_PRIMARY}; 
                border: 1px solid {_s.BORDER_LIGHT}; padding: 6px; font-family: 'Consolas';
            }}
        """)
        self.inp_path.returnPressed.connect(self.navigate_to_path)
        
        self.inp_search = QLineEdit()
        self.inp_search.setPlaceholderText("Search files...")
        self.inp_search.setFixedWidth(200)
        self.inp_search.setStyleSheet(f"""
            QLineEdit {{
                background: {_s.BG_INPUT}; color: white; 
                border: 1px solid {_s.BORDER_LIGHT}; padding: 6px;
            }}
        """)
        self.inp_search.textChanged.connect(self.on_search)
        
        nav_bar.addWidget(QLabel("📂"))
        nav_bar.addWidget(self.inp_path)
        nav_bar.addSpacing(10)
        nav_bar.addWidget(QLabel("🔍"))
        nav_bar.addWidget(self.inp_search)
        gl.addLayout(nav_bar)
        
        # --- FILE TREE ---
        self.tree = TerminalFileTree(self.current_path)
        self.tree.customContextMenuRequested.connect(self.open_menu)
        self.tree.clicked.connect(self.on_click_item) 
        gl.addWidget(self.tree)
        
        # --- BOTTOM ACTION BAR ---
        actions = QHBoxLayout()
        
        btn_add = MonoButton("+ MKDIR")
        btn_add.clicked.connect(self.new_folder)
        
        btn_del = MonoButton("× DELETE")
        btn_del.clicked.connect(self.delete_item)
        
        btn_ref = MonoButton("⟳ REFRESH")
        btn_ref.clicked.connect(self.refresh)
        
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet(f"color: {_s.FG_DIM}; font-size: 10px;")
        
        actions.addWidget(btn_add)
        actions.addWidget(btn_del)
        actions.addWidget(btn_ref)
        actions.addStretch()
        actions.addWidget(self.lbl_status)
        
        gl.addLayout(actions)
        grp.add_layout(gl)
        layout.addWidget(grp)

    def navigate_to_path(self):
        new_path = self.inp_path.text()
        if os.path.exists(new_path):
            self.current_path = new_path
            self.tree.change_root(new_path)
            self.lbl_status.setText(f"Navigated to: {new_path}")
        else:
            self.lbl_status.setText("Error: Path does not exist")

    def on_click_item(self, index):
        path = self.tree.model.filePath(index)
        if os.path.isdir(path):
            self.inp_path.setText(path)

    def refresh(self):
        self.tree.change_root(self.current_path)
        self.lbl_status.setText("Refreshed")

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
            except Exception as e:
                self.lbl_status.setText(f"Error: {e}")

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
                except Exception as e:
                    self.lbl_status.setText(f"Delete Error: {e}")

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
