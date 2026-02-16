from typing import Dict, Iterable

from ui.addons.spec import AddonSpec


class AddonRegistry:
    def __init__(self):
        self._specs: Dict[str, AddonSpec] = {}

    def register(self, spec: AddonSpec) -> None:
        self._specs[spec.id] = spec

    def get(self, addon_id: str) -> AddonSpec:
        if addon_id not in self._specs:
            raise KeyError(f"Addon '{addon_id}' not found. Known addons: {list(self._specs.keys())}")
        return self._specs[addon_id]

    def all(self) -> Iterable[AddonSpec]:
        return self._specs.values()
