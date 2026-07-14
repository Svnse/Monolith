"""Bearing — first-party system addon for situational coherence.

Public entry: `build_addon()`. Called once at bootstrap-time. Returns
`None` if MONOLITH_BEARING_V1=0; otherwise returns a `BearingAddon` with
`.provider` (for kernel DI) and `.interceptor` (for register_interceptor).

The whole addon constructs once per process. No reload, no hot-swap.
"""
from __future__ import annotations

from dataclasses import dataclass

from . import compiler
from . import kill_switch
from . import store  # noqa: F401 — re-exported for tests
from . import updater  # noqa: F401 — re-exported for tests
from .provider import BearingProvider


@dataclass(frozen=True)
class BearingAddon:
    provider: BearingProvider
    interceptor: object  # signature: (list[dict], dict) -> list[dict] | None


def build_addon() -> BearingAddon | None:
    """Construct the Bearing addon. Returns None when kill switch is off.

    Idempotent — calling twice yields equivalent addons (no shared mutable
    state besides the bearing.json file itself).
    """
    if not kill_switch.is_enabled():
        return None
    provider = BearingProvider()
    return BearingAddon(
        provider=provider,
        interceptor=compiler.bearing_interceptor,
    )
