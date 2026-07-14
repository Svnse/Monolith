from typing import Dict, Iterable, List

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

    def unregister(self, addon_id: str) -> None:
        if addon_id in self._specs:
            del self._specs[addon_id]

    def all(self) -> Iterable[AddonSpec]:
        return self._specs.values()

    def query_by_verb(self, verb: str) -> List[AddonSpec]:
        """Return all addons whose descriptor declares the given verb."""
        return [
            spec for spec in self._specs.values()
            if spec.descriptor is not None and verb in spec.descriptor.verbs
        ]

    def query_by_appetite(self, context: str) -> List[AddonSpec]:
        """Return all addons whose descriptor declares appetite for the given context type."""
        return [
            spec for spec in self._specs.values()
            if spec.descriptor is not None and context in spec.descriptor.appetites
        ]

    def query_by_emission(self, emission: str) -> List[AddonSpec]:
        """Return all addons whose descriptor declares the given emission type."""
        return [
            spec for spec in self._specs.values()
            if spec.descriptor is not None and emission in spec.descriptor.emissions
        ]

    def verify_capabilities(self, spec: AddonSpec, widget: object | None = None) -> list[str]:
        issues: list[str] = []
        if spec.descriptor is None or widget is None:
            return issues
        for verb in spec.descriptor.verbs or ():
            if not callable(getattr(widget, verb, None)):
                issues.append(f"missing verb handler: {verb}()")
        return issues
