from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import Signal

from ui.components.atoms import SkeetGroupBox, SkeetButton
from core.style import FG_DIM

class PageAddons(QWidget):
    sig_launch_addon = Signal(str)

    def __init__(self, state):
        super().__init__()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        grp_modules = SkeetGroupBox("AVAILABLE MODULES")
        
        mod_layout = QVBoxLayout()
        mod_layout.setSpacing(15)
        
        lbl_info = QLabel("Select a runtime module to attach to the workspace.")
        lbl_info.setStyleSheet(f"color: {FG_DIM}; font-size: 11px;")

        btn_terminal = SkeetButton("LAUNCH TERMINAL")
        btn_terminal.clicked.connect(lambda: self.sig_launch_addon.emit("terminal"))

        btn_databank = SkeetButton("LAUNCH DATABANK")
        btn_databank.clicked.connect(lambda: self.sig_launch_addon.emit("databank"))

        btn_injector = SkeetButton("LAUNCH PY INJECTOR")
        btn_injector.clicked.connect(lambda: self.sig_launch_addon.emit("injector"))

        mod_layout.addWidget(lbl_info)
        mod_layout.addWidget(btn_terminal)
        mod_layout.addWidget(btn_databank)
        mod_layout.addWidget(btn_injector)
        mod_layout.addStretch()
        
        grp_modules.add_layout(mod_layout)
        layout.addWidget(grp_modules)
