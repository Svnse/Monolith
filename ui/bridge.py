from PySide6.QtCore import QObject, Signal


class UIBridge(QObject):
    sig_terminal_header = Signal(str, str, str)
    sig_apply_operator = Signal(dict)
    sig_open_overseer = Signal()
    sig_overseer_viz_toggle = Signal(bool)
    sig_theme_changed = Signal(str)
