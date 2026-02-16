from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

from core.state import AppState
from monokernel.bridge import MonoBridge
from monokernel.guard import MonoGuard
from ui.bridge import UIBridge

if TYPE_CHECKING:
    from ui.addons.host import AddonHost
    from ui.main_window import MonolithUI


@dataclass
class AddonContext:
    state: AppState
    guard: MonoGuard
    bridge: MonoBridge
    ui: Optional["MonolithUI"]
    host: Optional["AddonHost"]
    ui_bridge: UIBridge
