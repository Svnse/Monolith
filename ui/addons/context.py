from dataclasses import dataclass, field
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
    services: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.services.setdefault("state", self.state)
        self.services.setdefault("guard", self.guard)
        self.services.setdefault("bridge", self.bridge)
        self.services.setdefault("ui_bridge", self.ui_bridge)
        self.services.setdefault("ui", self.ui)
        self.services.setdefault("host", self.host)
        self.services.setdefault("world_state", getattr(self.state, "world_state", None))

    def resolve(self, name: str) -> object | None:
        return self.services.get(name)
