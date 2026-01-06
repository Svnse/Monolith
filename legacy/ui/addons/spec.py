from dataclasses import dataclass
from typing import Callable, Literal

from PySide6.QtWidgets import QWidget

from ui.addons.context import AddonContext

AddonKind = Literal["page", "module"]


@dataclass(frozen=True)
class AddonSpec:
    id: str
    kind: AddonKind
    title: str
    icon: str | None
    factory: Callable[[AddonContext], QWidget]
