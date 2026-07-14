from PySide6.QtCore import QObject, Signal


class UIBridge(QObject):
    sig_terminal_header = Signal(str, str, str)
    sig_apply_operator = Signal(dict)
    sig_open_overseer = Signal()
    sig_overseer_viz_toggle = Signal(bool)
    sig_theme_changed = Signal(str)
    sig_config_changed = Signal(dict)
    sig_launch_addon = Signal(str)  # addon id -> host.launch_module
    sig_reveal_attachment = Signal(object)  # Attachment -> open databank + show it
    sig_world_action = Signal(dict)  # validated world action
    sig_world_action_pending = Signal(dict)  # pending approval action
    sig_world_action_approved = Signal(dict)  # approved action to execute
    sig_world_action_rejected = Signal(object)  # rejected action
    sig_monitor_log = Signal(str, str)  # severity, message
    sig_reload_modules = Signal()  # request running modules to reload/refresh
