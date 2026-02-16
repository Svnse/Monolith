from dataclasses import dataclass, field
from typing import Callable, Literal

from PySide6.QtWidgets import QWidget

from ui.addons.context import AddonContext
from ui.addons.descriptors import CapabilityDescriptor

AddonKind = Literal["page", "module"]


@dataclass(frozen=True)
class AddonSpec:
    id: str
    kind: AddonKind
    title: str
    icon: str | None
    factory: Callable[[AddonContext], QWidget]
    descriptor: CapabilityDescriptor | None = None
