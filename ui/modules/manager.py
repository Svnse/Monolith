from PySide6.QtWidgets import QWidget, QVBoxLayout, QLabel, QScrollArea
from PySide6.QtCore import Signal, Qt

from ui.components.atoms import SkeetGroupBox, SkeetButton
from core.style import FG_DIM

class PageAddons(QWidget):
    sig_launch_addon = Signal(str)

    def __init__(self, state):
        super().__init__()
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)
        
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)

        grp_modules = SkeetGroupBox("AVAILABLE MODULES")
        
        mod_layout = QVBoxLayout()
        mod_layout.setSpacing(15)
        
        lbl_info = QLabel("Select a runtime module to attach to the workspace.")
        lbl_info.setStyleSheet(f"color: {FG_DIM}; font-size: 11px;")

        btn_terminal = SkeetButton("TERMINAL")
        btn_terminal.clicked.connect(lambda: self.sig_launch_addon.emit("terminal"))

        btn_databank = SkeetButton("DATABANK")
        btn_databank.clicked.connect(lambda: self.sig_launch_addon.emit("databank"))

        btn_injector = SkeetButton("INJECTOR")
        btn_injector.clicked.connect(lambda: self.sig_launch_addon.emit("injector"))

        btn_sd = SkeetButton("VISION")
        btn_sd.clicked.connect(lambda: self.sig_launch_addon.emit("sd"))

        btn_audiogen = SkeetButton("AUDIO")
        btn_audiogen.clicked.connect(lambda: self.sig_launch_addon.emit("audiogen"))

        mod_layout.addWidget(lbl_info)
        mod_layout.addWidget(btn_terminal)
        mod_layout.addWidget(btn_databank)
        mod_layout.addWidget(btn_injector)
        mod_layout.addWidget(btn_sd)
        mod_layout.addWidget(btn_audiogen)
        mod_layout.addStretch()
        
        grp_modules.add_layout(mod_layout)
        scroll_layout.addWidget(grp_modules)
        scroll_layout.addStretch()
        scroll_area.setWidget(scroll_content)
        layout.addWidget(scroll_area)
